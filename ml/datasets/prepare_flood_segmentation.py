"""
Phase D Step 2: Rasterize verified flood-extent polygons from WorldFloods v2
GeoJSON floodmaps onto the S2 raster grid.

Outputs 3-class segmentation masks:
    0 = ignore / no supervision (cloud, invalid, outside AOI)
    1 = non-flood supervised pixel (clear sky, inside AOI, no flood polygon)
    2 = flood extent (verified 'Flooded area' polygon)

Also outputs a companion validity mask:
    0 = unsupervised (cloud/invalid/outside AOI)
    1 = supervised (clear sky, inside AOI, has GT data)

Does NOT overwrite any source files.
Writes to: data/processed/flood/segmentation_v1/
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.crs import CRS

PROJECT_ROOT = r"d:\superiorparab\final year DL project"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "flood", "worldfloods_v2")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "flood", "segmentation_v1")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mask values
IGNORE = 0       # No supervision available
NON_FLOOD = 1    # Supervised non-flood pixel
FLOOD = 2        # Supervised flood-extent pixel


def extract_crs_from_geojson(data: dict) -> Optional[CRS]:
    """Extract CRS from GeoJSON 'crs' field if present."""
    crs_info = data.get("crs")
    if crs_info is None:
        return None

    # Standard GeoJSON CRS format
    if isinstance(crs_info, dict):
        props = crs_info.get("properties", {})
        name = props.get("name", "")
        if "EPSG" in name:
            # Parse "urn:ogc:def:crs:EPSG::32629" or "EPSG:32629"
            parts = name.split(":")
            for i, part in enumerate(parts):
                if part == "EPSG" and i + 1 < len(parts):
                    code = parts[-1]  # Take last part
                    if code.isdigit():
                        return CRS.from_epsg(int(code))
    return None


def load_flood_polygons(geojson_path: str) -> Tuple[List, Dict[str, int]]:
    """Load verified flood-extent polygons from a GeoJSON file.

    Returns:
        geometries: list of (shapely_geometry, burn_value) tuples for rasterize
        counts: dict of feature type counts
    """
    with open(geojson_path) as f:
        data = json.load(f)

    geojson_crs = extract_crs_from_geojson(data)
    features = data.get("features", [])

    flood_geoms = []
    counts = {
        "flood": 0,
        "flood_trace": 0,
        "water": 0,
        "hydro": 0,
        "aoi": 0,
        "other": 0,
    }

    for feat in features:
        props = feat.get("properties") or {}
        geom_dict = feat.get("geometry")
        if geom_dict is None:
            continue

        w_class = props.get("w_class", "")

        if w_class == "Flooded area":
            counts["flood"] += 1
            if geom_dict.get("type") and geom_dict.get("coordinates"):
                flood_geoms.append((geom_dict, FLOOD))
        elif w_class == "Flood trace":
            counts["flood_trace"] += 1
        elif w_class == "water":
            counts["water"] += 1
        elif w_class == "area_of_interest":
            counts["aoi"] += 1
        elif w_class in ("BH140-River", "BH080-Lake", "BH141-Stream",
                         "BH130-Reservoir", "BA010-Coastline", "BA040-Open Water",
                         "BH141-River Bank", "River", "Not Applicable",
                         "Standing water", "Erosion", "Riverine flood",
                         "Dike breach"):
            counts["hydro"] += 1
        else:
            counts["other"] += 1

    return flood_geoms, counts, geojson_crs


def create_validity_mask(gt_path: str, transform, height: int, width: int) -> np.ndarray:
    """Create a validity mask from the GT raster Band 1 (cloud/validity).

    Returns:
        validity: 2D array where 1 = supervised (clear sky + valid),
                  0 = unsupervised (cloud/invalid)
    """
    with rasterio.open(gt_path) as src:
        # Verify alignment
        if src.height != height or src.width != width:
            raise ValueError(
                f"GT dimensions ({src.height}x{src.width}) differ from "
                f"S2 ({height}x{width})"
            )
        gt_tf = list(src.transform)[:6]
        s2_tf = list(transform)[:6]
        if gt_tf != s2_tf:
            raise ValueError("GT transform differs from S2 transform")

        cloud_mask = src.read(1)
        water_mask = src.read(2)

    # Valid = clear sky (Band 1 == 1) AND not-invalid water layer (Band 2 != 0)
    valid = ((cloud_mask == 1) & (water_mask != 0)).astype(np.uint8)
    return valid


def rasterize_sample(
    name: str,
    split: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Rasterize flood polygons for a single sample.

    Returns a result dict with validation metrics.
    """
    s2_path = os.path.join(DATA_ROOT, split, "S2", name + ".tif")
    gt_path = os.path.join(DATA_ROOT, split, "gt", name + ".tif")
    fm_path = os.path.join(DATA_ROOT, split, "floodmaps", name + ".geojson")
    meta_path = os.path.join(DATA_ROOT, split, "meta", name + ".json")

    result = {
        "name": name,
        "split": split,
        "success": False,
    }

    # Load S2 metadata for grid definition
    with rasterio.open(s2_path) as s2_src:
        s2_h = s2_src.height
        s2_w = s2_src.width
        s2_crs = s2_src.crs
        s2_transform = s2_src.transform

    result["height"] = s2_h
    result["width"] = s2_w
    result["crs"] = str(s2_crs)

    # Load flood polygons
    if not os.path.exists(fm_path):
        result["error"] = "No floodmap GeoJSON found"
        return result

    flood_geoms, counts, geojson_crs = load_flood_polygons(fm_path)
    result["polygon_counts"] = counts

    # CRS verification
    if geojson_crs is not None and geojson_crs != s2_crs:
        result["error"] = (
            f"GeoJSON CRS ({geojson_crs}) differs from S2 CRS ({s2_crs}). "
            "Rasterization aborted — CRS mismatch."
        )
        result["crs_match"] = False
        return result
    result["crs_match"] = True if geojson_crs else "geojson_crs_missing_assumed_match"

    # Create validity mask from GT
    try:
        validity = create_validity_mask(gt_path, s2_transform, s2_h, s2_w)
    except Exception as e:
        result["error"] = f"Validity mask error: {e}"
        return result

    result["valid_px"] = int(validity.sum())
    result["total_px"] = int(s2_h * s2_w)

    # Initialize segmentation mask
    # Start with all IGNORE, then mark valid pixels as NON_FLOOD
    seg_mask = np.zeros((s2_h, s2_w), dtype=np.uint8)
    seg_mask[validity == 1] = NON_FLOOD

    # Rasterize flood polygons
    if len(flood_geoms) > 0:
        flood_raster = rasterize(
            flood_geoms,
            out_shape=(s2_h, s2_w),
            transform=s2_transform,
            fill=0,
            dtype=np.uint8,
        )
        # Only mark flood pixels where they are also valid
        flood_and_valid = (flood_raster == FLOOD) & (validity == 1)
        seg_mask[flood_and_valid] = FLOOD

        result["rasterized_flood_px"] = int(flood_and_valid.sum())
        result["rasterized_flood_px_total"] = int((flood_raster == FLOOD).sum())
    else:
        result["rasterized_flood_px"] = 0
        result["rasterized_flood_px_total"] = 0

    result["non_flood_px"] = int((seg_mask == NON_FLOOD).sum())
    result["ignore_px"] = int((seg_mask == IGNORE).sum())

    # Flood percentage among supervised pixels
    supervised = result["rasterized_flood_px"] + result["non_flood_px"]
    result["supervised_px"] = supervised
    result["flood_pct"] = (
        round(result["rasterized_flood_px"] / supervised * 100, 4)
        if supervised > 0 else 0.0
    )

    # Load metadata for comparison
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        result["meta_flood_px"] = meta.get("pixels flood water S2")
        result["meta_hydro_px"] = meta.get("pixels hydro water S2")

    # Save segmentation mask
    os.makedirs(output_dir, exist_ok=True)
    seg_out_path = os.path.join(output_dir, name + "_seg.tif")
    val_out_path = os.path.join(output_dir, name + "_validity.tif")

    profile = {
        "driver": "GTiff",
        "height": s2_h,
        "width": s2_w,
        "count": 1,
        "dtype": "uint8",
        "crs": s2_crs,
        "transform": s2_transform,
        "compress": "lzw",
        "nodata": IGNORE,
    }

    with rasterio.open(seg_out_path, "w", **profile) as dst:
        dst.write(seg_mask, 1)
        dst.update_tags(
            mask_values="0=ignore,1=non_flood,2=flood",
            source="rasterized from floodmap GeoJSON w_class=Flooded area",
            sample=name,
            created=datetime.now().isoformat(),
        )

    with rasterio.open(val_out_path, "w", **profile) as dst:
        dst.write(validity, 1)
        dst.update_tags(
            mask_values="0=unsupervised,1=supervised",
            source="GT Band1 cloud/validity mask",
            sample=name,
        )

    result["seg_mask_path"] = seg_out_path
    result["validity_mask_path"] = val_out_path
    result["success"] = True

    return result


def main():
    print(f"Phase D: Rasterize Flood Segmentation Masks")
    print(f"Data root: {DATA_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load label status from Step 1
    status_path = os.path.join(
        PROJECT_ROOT, "data", "reports", "phase_d_label_status.json"
    )
    with open(status_path) as f:
        label_data = json.load(f)

    # Process only positive samples (verified flood polygons)
    positive_samples = [
        s for s in label_data["samples"] if s["label_status"] == "positive"
    ]
    negative_samples = [
        s for s in label_data["samples"]
        if s["label_status"] == "negative_candidate"
    ]

    print(f"\nPositive samples (flood polygons): {len(positive_samples)}")
    print(f"Negative candidate samples: {len(negative_samples)}")

    # Process positive samples
    results = []
    masks_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  RASTERIZING FLOOD-POSITIVE SAMPLES")
    print(f"{'='*70}")

    for sample in positive_samples:
        name = sample["name"]
        split = sample["split"]
        r = rasterize_sample(name, split, masks_dir)
        r["label_status"] = "positive"
        results.append(r)

        status = "OK" if r["success"] else f"FAIL: {r.get('error', '?')}"
        flood_px = r.get("rasterized_flood_px", 0)
        meta_px = r.get("meta_flood_px", "N/A")
        print(f"  {split:5s} | {name:55s} | flood_px={flood_px:8d} | meta={meta_px!s:>10s} | {status}")

    # Process negative candidates (rasterize with no flood polygons = all NON_FLOOD)
    print(f"\n{'='*70}")
    print(f"  RASTERIZING NEGATIVE-CANDIDATE SAMPLES")
    print(f"{'='*70}")

    for sample in negative_samples:
        name = sample["name"]
        split = sample["split"]
        r = rasterize_sample(name, split, masks_dir)
        r["label_status"] = "negative_candidate"
        results.append(r)

        status = "OK" if r["success"] else f"FAIL: {r.get('error', '?')}"
        print(f"  {split:5s} | {name:55s} | non_flood={r.get('non_flood_px',0):8d} | {status}")

    # Summary
    ok = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    print(f"\n{'='*70}")
    print(f"  RASTERIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Processed: {len(results)}")
    print(f"  Success: {len(ok)}")
    print(f"  Failed: {len(failed)}")
    for f in failed:
        print(f"    FAILED: {f['name']} — {f.get('error', '?')}")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "rasterization_results.json")
    with open(results_path, "w") as fp:
        json.dump({
            "created": datetime.now().isoformat(),
            "mask_values": {"0": "ignore", "1": "non_flood", "2": "flood"},
            "total_processed": len(results),
            "success": len(ok),
            "failed": len(failed),
            "results": results,
        }, fp, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
