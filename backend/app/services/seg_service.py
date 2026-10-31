"""
Flood segmentation service for the FastAPI backend.

Connects the U-Net inference module (ml/inference/seg_inference.py) to the API:

    * a deterministic demo-sample catalogue built from the segmentation split
      manifest (validation + held-out test samples only, never training),
      with every listed sample verified to have imagery, reference mask,
      validity mask and matching dimensions;
    * demo assessments (prediction + EMS reference comparison);
    * live assessments on Copernicus imagery (experimental, no reference);
    * an in-memory + on-disk result cache so slow large scenes are computed once.

All images returned to the client are JPEG data URLs, downsampled together
to the same display grid so they stay aligned.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import seg_inference as si  # noqa: E402
from ml.inference import assessment as asm  # noqa: E402
from ml.datasets.flood_seg_dataset import STORED_FLOOD, STORED_IGNORE  # noqa: E402

logger = logging.getLogger(__name__)

from app.services.water_reference import REFERENCE_VERSION as WATER_REFERENCE_VERSION  # noqa: E402

SET_ASIDE_COLOR = (150, 160, 175)  # flood-like pixels on known permanent water / sea

SEG_DIR = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1"
DEFAULTS = {
    "checkpoint": None,  # resolved from app.config (DI_SEG_MODEL) at construction time
    "manifest": SEG_DIR / "seg_split_manifest.json",
    "data_root": PROJECT_ROOT / "data" / "raw" / "flood" / "worldfloods_v2",
    "masks_dir": SEG_DIR / "masks",
    "cache_dir": PROJECT_ROOT / "data" / "cache" / "seg_api",
    "test_report": PROJECT_ROOT / "data" / "reports" / "phase_h_test_evaluation.json",
}

CACHE_VERSION = "2"          # bump when the response format or inference changes
DISPLAY_MAX_SIDE = 1024
JPEG_QUALITY = 90
DEFAULT_METHOD = "tiled"
DEMO_SPLITS = ("val", "test")
MEM_CACHE_SIZE = 16

DISCLAIMER = (
    "Satellite-image flood extent analysis from a research model. Not an official warning; it does not "
    "forecast floods."
)

LIMITATIONS = [
    "Accuracy on unseen flood events is limited; many flood pixels can be missed.",
    "Bright or dark surfaces (snow, cloud, haze, shadows, dark soil, urban areas) can be mistaken for flood water.",
    "The model can mark permanent water (sea, lakes, rivers) as flood: it sees one image and has no record of "
    "normal water levels.",
    "Flood coverage is a ratio of image pixels, not a measured area in km².",
    "The model's flood probability is a raw output, not a calibrated confidence.",
    "Weather and air-quality information is not used by the model.",
    "The prototype maps visible water in one image; it does not forecast future floods.",
]

SPLIT_NOTES = {
    "val": "Validation sample. It was not used to train the weights, but it was used to choose the checkpoint "
           "and the outcome thresholds, so scores on it are optimistic.",
    "test": "Held-out test sample. Never used for training, checkpoint selection or tuning; shown for final "
            "illustration only.",
}

METHOD_NOTES = {
    "tiled": "256 px tiles at native 10 m resolution (the scale the model was trained on), blended and stitched.",
    "resize": "Whole scene shrunk to 256 x 256 px (the Phase H evaluation procedure), then upsampled back.",
}


class SampleNotFoundError(KeyError):
    """Requested demo sample is not in the verified catalogue."""


# ============================================================
# Demo catalogue
# ============================================================
@dataclass(frozen=True)
class DemoSample:
    id: str
    event_id: str
    aoi_code: Optional[str]
    split: str
    original_split: str
    event_type: Optional[str]
    s2_date: Optional[str]
    center_lat: Optional[float]
    center_lng: Optional[float]
    bbox: Optional[Dict[str, float]]
    width: int
    height: int
    reference_flood_pct: float
    label_status: str
    s2_path: Path = field(repr=False)
    seg_path: Path = field(repr=False)
    validity_path: Path = field(repr=False)
    crs: Optional[str] = field(default=None, repr=False)
    transform: Optional[Tuple[float, ...]] = field(default=None, repr=False)
    pw_path: Optional[Path] = field(default=None, repr=False)

    @property
    def label(self) -> str:
        aoi = re.sub(r"^\d+", "", self.aoi_code or "") or self.id
        return f"{self.event_id} · {aoi}"

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "event_id": self.event_id,
            "aoi_code": self.aoi_code,
            "split": self.split,
            "split_note": SPLIT_NOTES[self.split],
            "event_type": self.event_type,
            "s2_date": self.s2_date,
            "center": None if self.center_lat is None else {"lat": self.center_lat, "lng": self.center_lng},
            "bbox": self.bbox,
            "width": self.width,
            "height": self.height,
            "reference_flood_pct": self.reference_flood_pct,
            "label_status": self.label_status,
        }


def _clean(v):
    return None if v in (None, "", "NaN", "None") else v


def _read_meta(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def build_catalogue(
    manifest_path: Path, data_root: Path, masks_dir: Path, splits=DEMO_SPLITS,
) -> Tuple[List[DemoSample], List[Dict[str, str]]]:
    """Verified demo samples (+ list of excluded samples with reasons)."""
    import rasterio
    from rasterio.warp import transform_bounds

    manifest = json.loads(Path(manifest_path).read_text())
    samples, excluded = [], []
    for split in splits:
        for s in manifest["splits"][split]["samples"]:
            name, osplit = s["name"], s["original_split"]
            s2 = Path(data_root) / osplit / "S2" / f"{name}.tif"
            seg = Path(masks_dir) / f"{name}_seg.tif"
            val = Path(masks_dir) / f"{name}_validity.tif"
            missing = [k for k, p in (("imagery", s2), ("reference mask", seg), ("validity mask", val)) if not p.is_file()]
            if missing:
                excluded.append({"id": name, "split": split, "reason": "missing " + ", ".join(missing)})
                continue
            try:
                with rasterio.open(s2) as r:
                    h, w, count, crs, bounds = r.height, r.width, r.count, r.crs, r.bounds
                    transform = tuple(r.transform)[:6]
                shapes = set()
                for p in (seg, val):
                    with rasterio.open(p) as m:
                        shapes.add((m.height, m.width))
            except Exception as e:
                excluded.append({"id": name, "split": split, "reason": f"unreadable raster: {e}"})
                continue
            if shapes != {(h, w)} or count < 13:
                excluded.append({"id": name, "split": split, "reason": "dimension or band-count mismatch"})
                continue

            meta = _read_meta(Path(data_root) / osplit / "meta" / f"{name}.json")
            bb = meta.get("bounding box")
            if not (isinstance(bb, dict) and all(k in bb for k in ("west", "east", "south", "north"))):
                try:
                    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds)
                    bb = {"west": west, "east": east, "south": south, "north": north}
                except Exception:
                    bb = None
            if bb:
                bb = {k: round(float(bb[k]), 5) for k in ("west", "east", "south", "north")}
            samples.append(DemoSample(
                id=name,
                event_id=s.get("event_id") or name.split("_")[0],
                aoi_code=_clean(meta.get("aoi_code")) or (name.split("_")[1] if "_" in name else None),
                split=split,
                original_split=osplit,
                event_type=_clean(meta.get("event type")),
                s2_date=(_clean(meta.get("s2_date")) or "")[:10] or None,
                center_lat=None if not bb else round((bb["south"] + bb["north"]) / 2, 5),
                center_lng=None if not bb else round((bb["west"] + bb["east"]) / 2, 5),
                bbox=bb,
                width=w,
                height=h,
                reference_flood_pct=round(float(s.get("flood_pct", 0.0)), 4),
                label_status=s.get("label_status", "unknown"),
                s2_path=s2, seg_path=seg, validity_path=val,
                crs=crs.to_string() if crs else None,
                transform=transform,
                pw_path=(lambda p: p if p.is_file() else None)(
                    Path(data_root) / osplit / "PERMANENTWATERJRC" / f"{name}.tif"),
            ))
    order = {sp: i for i, sp in enumerate(splits)}
    samples.sort(key=lambda d: (order[d.split], d.event_id, d.id))
    return samples, excluded


# ============================================================
# Image encoding helpers
# ============================================================
def to_display(arr: np.ndarray, max_side: int = DISPLAY_MAX_SIDE) -> np.ndarray:
    return si.downsample_for_display(arr, max_side)


def jpeg_data_url(arr: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def webp_data_url(rgba: np.ndarray, lossless: bool = False, quality: int = 82) -> str:
    """RGBA uint8 -> WebP data URL (keeps transparency, much smaller than PNG)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="WEBP", lossless=lossless, quality=quality, method=4)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def map_layers(
    rgb_d: np.ndarray,
    flood_d: np.ndarray,
    set_aside_d: Optional[np.ndarray],
    src_crs: str,
    src_transform,
) -> Dict[str, Any]:
    """Georeferenced map layers in EPSG:4326 (for Leaflet ImageOverlay).

    rgb_d / flood_d / set_aside_d share one display grid described by src_crs + src_transform.
    Projected (e.g. UTM) grids are reprojected so overlays line up with the web map.
    """
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import array_bounds
    from rasterio.warp import calculate_default_transform, reproject

    h, w = flood_d.shape
    src = CRS.from_user_input(src_crs)
    dst = CRS.from_epsg(4326)
    left_bottom_right_top = array_bounds(h, w, src_transform)  # (west, south, east, north)

    if src == dst:
        dst_t, dw, dh = src_transform, w, h
        rgb4, valid4 = rgb_d, np.ones((h, w), np.uint8)
        flood4 = flood_d.astype(np.uint8)
        aside4 = None if set_aside_d is None else set_aside_d.astype(np.uint8)
    else:
        dst_t, dw, dh = calculate_default_transform(src, dst, w, h, *left_bottom_right_top)

        def warp(arr, resampling):
            # 255 marks "outside the source footprint"; real values are 0..254 (masks are 0/1,
            # RGB 255 is clipped to 254, invisible) so GDAL never re-maps genuine data values.
            out = np.full((dh, dw), 255, np.uint8)
            reproject(np.minimum(arr.astype(np.uint8), 254), out, src_transform=src_transform, src_crs=src,
                      dst_transform=dst_t, dst_crs=dst, resampling=resampling, src_nodata=255, dst_nodata=255)
            return out

        bands = [warp(rgb_d[..., c], Resampling.bilinear) for c in range(3)]
        valid4 = (bands[0] != 255).astype(np.uint8)
        rgb4 = np.dstack([np.where(b == 255, 0, b) for b in bands]).astype(np.uint8)
        flood4 = (warp(flood_d, Resampling.nearest) == 1).astype(np.uint8)
        aside4 = None if set_aside_d is None else (warp(set_aside_d, Resampling.nearest) == 1).astype(np.uint8)

    west, south, east, north = array_bounds(dh, dw, dst_t)
    sat = np.dstack([rgb4, (valid4 > 0).astype(np.uint8) * 255])
    fl = np.zeros((dh, dw, 4), np.uint8)
    fl[flood4 > 0] = (*si.FLOOD_COLOR, 205)
    layers = {
        "crs": "EPSG:4326",
        "bounds": [[round(south, 6), round(west, 6)], [round(north, 6), round(east, 6)]],
        "satellite": webp_data_url(sat),
        "flood": webp_data_url(fl, lossless=True),
        "set_aside": None,
        "width": int(dw),
        "height": int(dh),
    }
    if aside4 is not None and aside4.any():
        sa = np.zeros((dh, dw, 4), np.uint8)
        sa[aside4 > 0] = (*SET_ASIDE_COLOR, 150)
        layers["set_aside"] = webp_data_url(sa, lossless=True)
    return layers


def freshness(acquisition_date: Optional[str], archived: bool) -> Dict[str, Any]:
    """Age of the imagery at request time (never cached)."""
    from datetime import date, datetime, timezone

    age = None
    if acquisition_date:
        try:
            d = date.fromisoformat(acquisition_date[:10])
            age = (datetime.now(timezone.utc).date() - d).days
        except ValueError:
            age = None
    if archived:
        label = "Archived demonstration image"
    elif age is None:
        label = "Acquisition date unknown"
    elif age <= 3:
        label = "Recent image"
    elif age <= 14:
        label = f"Image is {age} days old"
    elif age <= 31:
        label = f"Image is {age} days old — conditions may have changed"
    else:
        label = "Old image — may not reflect current conditions"
    return {"acquisition_date": acquisition_date, "age_days": age, "archived": archived, "label": label}


def _round(d: Dict[str, Any], n: int = 4) -> Dict[str, Any]:
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


# ============================================================
# Service
# ============================================================
class SegmentationService:
    """Thread-safe facade used by the API routes."""

    def __init__(
        self,
        checkpoint: Path = DEFAULTS["checkpoint"],
        manifest: Path = DEFAULTS["manifest"],
        data_root: Path = DEFAULTS["data_root"],
        masks_dir: Path = DEFAULTS["masks_dir"],
        cache_dir: Optional[Path] = DEFAULTS["cache_dir"],
        test_report: Optional[Path] = DEFAULTS["test_report"],
        segmenter: Optional[si.FloodSegmenter] = None,
        device: Optional[str] = None,
        model_version: Optional[str] = None,
        evaluation_report: Optional[Path] = None,
    ):
        from app import config

        if checkpoint is None:
            self.model_version = model_version or config.seg_model_version()
            checkpoint = config.seg_checkpoint(self.model_version)
        else:
            self.model_version = model_version or Path(checkpoint).parent.name
        if Path(checkpoint).name in config.LEGACY_CLASSIFIER_FILES:
            raise ValueError("The legacy EfficientNet-B0 classifier cannot be used for flood segmentation.")
        self.evaluation_report = evaluation_report
        self.checkpoint = Path(checkpoint)
        self.manifest = Path(manifest)
        self.data_root = Path(data_root)
        self.masks_dir = Path(masks_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.test_report = Path(test_report) if test_report else None
        self.device = device
        self._segmenter = segmenter
        if segmenter is not None:
            self.model_version = segmenter.spec.version
        self._load_error: Optional[str] = None
        self._loading = False
        self._catalogue: Optional[List[DemoSample]] = None
        self._excluded: List[Dict[str, str]] = []
        self._model_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._cat_lock = threading.Lock()
        self._mem_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._policy_loaded = False
        self._policy: Optional[Dict[str, Any]] = None

    @property
    def policy(self) -> Optional[Dict[str, Any]]:
        """Validation-derived outcome thresholds for the ACTIVE model (None if not derived yet)."""
        if not self._policy_loaded:
            self._policy = asm.load_policy(self.model_version)
            self._policy_loaded = True
        return self._policy

    @policy.setter
    def policy(self, value):
        self._policy, self._policy_loaded = value, True

    # ---------- model ----------
    def segmenter(self) -> si.FloodSegmenter:
        if self._segmenter is not None:
            return self._segmenter
        with self._model_lock:
            if self._segmenter is None:
                self._loading = True
                try:
                    self._segmenter = si.FloodSegmenter(self.checkpoint, device=self.device)
                    self._load_error = None
                    logger.info("Segmentation model loaded on %s", self._segmenter.device)
                except Exception as e:
                    self._load_error = str(e)
                    raise
                finally:
                    self._loading = False
        return self._segmenter

    def _reported_test_metrics(self) -> Optional[Dict[str, Any]]:
        if not self.test_report or not self.test_report.is_file():
            return None
        try:
            d = json.loads(self.test_report.read_text())
            a = d["aggregate"]
            return {
                "split": "test",
                "n_samples": d.get("n_samples"),
                "protocol": "Phase H: whole scene resized to 256 x 256 px",
                **{k: round(a[k], 4) for k in ("flood_iou", "flood_dice", "flood_precision", "flood_recall",
                                                "pixel_accuracy")},
                "note": "Pixel accuracy is misleading here because most pixels are non-flood.",
            }
        except Exception:
            return None

    def public_error(self) -> Optional[str]:
        """Model load error without local paths (safe to return to clients)."""
        if not self._load_error:
            return None
        if "not found" in self._load_error:
            return f"The {self.model_version} checkpoint is not installed on this server."
        return f"The {self.model_version} checkpoint could not be loaded."

    def _heldout_metrics(self) -> Optional[Dict[str, Any]]:
        """Held-out test metrics for the ACTIVE model under the product protocol, if evaluated."""
        rep = self.evaluation_report or (asm.REPORTS / f"evaluation_{self.model_version}_test.json")
        if not Path(rep).is_file():
            return None
        try:
            d = json.loads(Path(rep).read_text())
            m = d["summary"]["tiled"]["pixel_metrics_excluding_known_water"]
            return {"split": "test", "n_samples": len(d.get("per_sample", [])),
                    "protocol": "full resolution, tiled; flood = positive & ~permanent water vs EMS flood",
                    **{k: round(m[k], 4) for k in ("flood_iou", "flood_dice", "flood_precision", "flood_recall")}}
        except Exception:
            return None

    def status(self) -> Dict[str, Any]:
        try:
            seg = self.segmenter()
        except Exception:
            return {"status": "not_ready", "error": self.public_error() or "Model unavailable.", "model": None,
                    "model_version": self.model_version}
        meta = seg.meta
        return {
            "status": "ready",
            "model_version": seg.spec.version,
            "model": {
                "version": seg.spec.version,
                "task": "pixel-level segmentation",
                "target": seg.spec.target,
                "bands": list(seg.spec.bands),
                "preprocessing": seg.spec.preprocessing_id,
                "postprocessing": seg.spec.postprocessing_id,
                "architecture": meta.get("architecture"),
                "encoder": meta.get("encoder"),
                "epoch": meta.get("epoch"),
                "checkpoint": Path(meta["checkpoint"]).name if meta.get("checkpoint") else None,
                "input_size": seg.input_size,
                "device": str(seg.device),
                "val_metrics_at_save": meta.get("val_metrics_at_save"),
                "reported_test_metrics": self._heldout_metrics() or (
                    self._reported_test_metrics() if seg.spec.version == "segmentation_v1" else None),
                "legacy_classifier": "not used",
            },
            "policy_version": (self.policy or {}).get("version"),
            "methods": {m: METHOD_NOTES[m] for m in si.METHODS},
            "default_method": DEFAULT_METHOD,
            "demo_samples": len(self.catalogue()),
        }

    # ---------- catalogue ----------
    def catalogue(self) -> List[DemoSample]:
        if self._catalogue is None:
            with self._cat_lock:
                if self._catalogue is None:
                    self._catalogue, self._excluded = build_catalogue(self.manifest, self.data_root, self.masks_dir)
                    if self._excluded:
                        logger.warning("Demo catalogue excluded %d samples: %s", len(self._excluded), self._excluded)
        return self._catalogue

    def excluded(self) -> List[Dict[str, str]]:
        self.catalogue()
        return self._excluded

    def get_sample(self, sample_id: str) -> DemoSample:
        for s in self.catalogue():
            if s.id == sample_id:
                return s
        raise SampleNotFoundError(sample_id)

    # ---------- cache ----------
    def _cache_key(self, sample: DemoSample, method: str) -> str:
        seg = self.segmenter()
        ck = seg.meta.get("checkpoint")
        ck_sig = ""
        if ck and Path(ck).is_file():
            st = Path(ck).stat()
            ck_sig = f"{st.st_size}_{st.st_mtime_ns}"
        raw = (f"{CACHE_VERSION}|{sample.id}|{method}|{ck_sig}|{seg.meta.get('architecture')}"
               f"|{(self.policy or {}).get('version')}|{seg.spec.preprocessing_id}|{seg.spec.postprocessing_id}"
               f"|{WATER_REFERENCE_VERSION}")
        return hashlib.sha1(raw.encode()).hexdigest()[:20]

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._mem_cache:
            self._mem_cache.move_to_end(key)
            return self._mem_cache[key]
        if self.cache_dir:
            p = self.cache_dir / f"{key}.json"
            if p.is_file():
                try:
                    payload = json.loads(p.read_text())
                    self._mem_put(key, payload)
                    return payload
                except Exception:
                    logger.warning("Ignoring unreadable cache file %s", p)
        return None

    def _mem_put(self, key: str, payload: Dict[str, Any]):
        self._mem_cache[key] = payload
        self._mem_cache.move_to_end(key)
        while len(self._mem_cache) > MEM_CACHE_SIZE:
            self._mem_cache.popitem(last=False)

    def _cache_put(self, key: str, payload: Dict[str, Any]):
        self._mem_put(key, payload)
        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_dir / f"{key}.json.tmp"
                tmp.write_text(json.dumps(payload))
                tmp.replace(self.cache_dir / f"{key}.json")
            except Exception as e:
                logger.warning("Could not write cache: %s", e)

    # ---------- assessments ----------
    def _check_method(self, method: str):
        if method not in si.METHODS:
            raise ValueError(f"Unknown method '{method}'. Use one of {list(si.METHODS)}.")

    def _model_block(self) -> Dict[str, Any]:
        seg = self.segmenter()
        return {
            "version": seg.spec.version,
            "architecture": seg.meta.get("architecture"),
            "encoder": seg.meta.get("encoder"),
            "epoch": seg.meta.get("epoch"),
            "checkpoint": Path(seg.meta["checkpoint"]).name if seg.meta.get("checkpoint") else None,
            "target": seg.spec.target,
            "bands": list(seg.spec.bands),
            "preprocessing": seg.spec.preprocessing_id,
            "postprocessing": seg.spec.postprocessing_id,
            "policy_version": (self.policy or {}).get("version"),
            "device": str(seg.device),
        }

    def limitations(self) -> List[str]:
        """Model limitations, with this model's measured held-out score when available."""
        out = list(LIMITATIONS)
        m = self._heldout_metrics() or (self._reported_test_metrics()
                                        if self.model_version == "segmentation_v1" else None)
        if m:
            out[0] = (f"On held-out test events this model's flood IoU was {m['flood_iou']:.2f} "
                      f"(precision {m['flood_precision']:.2f}, recall {m['flood_recall']:.2f}), so many flood "
                      "pixels are missed and some non-flood pixels are flagged.")
        return out

    @staticmethod
    def _prediction_block(res: si.SegmentationResult) -> Dict[str, Any]:
        st = res.stats
        target = st.get("target", "ems_flood")
        label = "surface water (including permanent water)" if target == "surface_water" else "EMS-style flood"
        return {
            "target": target,
            "raw_label": label,
            "raw_coverage_pct": st["flood_coverage_pct"],
            # Backward-compatible name for the RAW positive-class share (= raw_coverage_pct).
            "flood_coverage_pct": st["flood_coverage_pct"],
            "coverage_definition": f"Raw model positives ({label}) divided by eligible pixels, x 100. Eligible "
                                   "pixels have image data and, where a cloud/validity mask exists, are cloud-free "
                                   "and valid. This is a pixel ratio, not an area. The reported flood-like share is "
                                   "assessment.flood_like_outside_known_water_pct.",
            "n_flood_pixels": st["n_flood_pred"],
            "n_eligible_pixels": st["n_eligible"],
            "n_pixels": st["n_pixels"],
            "eligible_pct": st["eligible_pct"],
            "nodata_pct": st["nodata_pct"],
            "eligibility_source": st["eligibility_source"],
            "threshold": st["flood_threshold"],
            "threshold_note": "A pixel is positive when the model's softmax score exceeds 0.5. The score is not a "
                              "calibrated probability or confidence.",
            "n_tiles": st["n_tiles"],
            "inference_s": st["elapsed_s"],
            "postprocessing": st.get("postprocessing", "none"),
            "n_removed_by_water_gate": st.get("n_removed_by_water_gate", 0),
            "water_gate_applied": st.get("water_gate_applied", False),
            "haze_index": st.get("haze_index"),
            "hazy": st.get("hazy", False),
        }

    def assess_demo(self, sample_id: str, method: str = DEFAULT_METHOD) -> Dict[str, Any]:
        self._check_method(method)
        sample = self.get_sample(sample_id)
        key = self._cache_key(sample, method)
        cached = self._cache_get(key)
        if cached is not None:
            # Wording and imagery age are applied fresh so cached results never show outdated text.
            return {**cached, "limitations": self.limitations(), "disclaimer": DISCLAIMER,
                    "freshness": freshness(sample.s2_date, archived=True), "cached": True}

        t0 = time.time()
        rgb = si.read_scene(sample.s2_path, self.segmenter().spec)
        stored = si.read_mask(sample.seg_path)
        validity = si.read_mask(sample.validity_path)
        stored[validity == 0] = STORED_IGNORE  # same rule as FloodSegDataset
        known_water = None
        if sample.pw_path is not None:
            from app.services.water_reference import from_worldfloods_jrc
            known_water = from_worldfloods_jrc(si.read_mask(sample.pw_path))
            if known_water.shape != stored.shape:
                known_water = None

        with self._infer_lock:
            res = self.segmenter().predict_array(rgb, validity=validity, method=method, source=sample.id)
        del rgb
        wc = asm.water_context(res.flood_mask, res.eligible_mask, known_water)
        assessment = asm.classify(wc, method, res.stats["quality_status"], self.policy,
                                  res.stats["eligible_pct"], live=False, hazy=res.stats.get("hazy", False))
        outside = asm.outside_water_mask(res.flood_mask, known_water)
        # Reference comparison uses what the product reports (positive outside known water);
        # the raw comparison is kept for diagnostics.
        codes, metrics = si.compare_with_reference(outside, stored)
        _, metrics_raw = si.compare_with_reference(res.flood_mask, stored)
        aside = None if known_water is None else (res.flood_mask & (known_water > 0))

        rgb_d = to_display(res.rgb_display)
        mask_d = to_display(res.flood_mask)
        stored_d = to_display(stored)
        codes_d = to_display(codes)
        outside_d = to_display(outside)
        aside_d = None if aside is None else to_display(aside)
        n_sup = int((stored != STORED_IGNORE).sum())
        n_ref_flood = int((stored == STORED_FLOOD).sum())
        map_block = None
        if sample.crs and sample.transform:
            from affine import Affine
            sx, sy = sample.width / outside_d.shape[1], sample.height / outside_d.shape[0]
            disp_t = Affine(*sample.transform) * Affine.scale(sx, sy)
            map_block = map_layers(rgb_d, outside_d, aside_d, sample.crs, disp_t)

        payload = {
            "status": "success",
            "mode": "demo",
            "method": method,
            "method_note": METHOD_NOTES[method],
            "experimental": False,
            "sample": sample.public(),
            "imagery": {
                "source": "WorldFloods v2 archive — Copernicus Sentinel-2 Level-1C",
                "acquisition_date": sample.s2_date,
                "resolution_m": 10,
                "width": sample.width,
                "height": sample.height,
                "center": sample.public()["center"],
                "bbox": sample.bbox,
                "location_note": "This archived scene shows the event's own location, not the location "
                                 "selected on the map.",
            },
            "images": {
                "rgb": jpeg_data_url(rgb_d),
                "overlay": jpeg_data_url(si.render_overlay(rgb_d, mask_d)),
                "reference": jpeg_data_url(si.render_reference(rgb_d, stored_d)),
                "error": jpeg_data_url(si.render_error_map(rgb_d, codes_d)),
                "display_width": int(rgb_d.shape[1]),
                "display_height": int(rgb_d.shape[0]),
                "legend": {
                    "flood": "#%02x%02x%02x" % si.FLOOD_COLOR,
                    "ignored": "#%02x%02x%02x" % si.IGNORE_COLOR,
                    "true_positive": "#%02x%02x%02x" % si.ERROR_COLORS[si.ERR_TP],
                    "false_positive": "#%02x%02x%02x" % si.ERROR_COLORS[si.ERR_FP],
                    "false_negative": "#%02x%02x%02x" % si.ERROR_COLORS[si.ERR_FN],
                },
            },
            "prediction": self._prediction_block(res),
            "reference": {
                "available": True,
                "source": "Copernicus EMS mapped flood polygons, rasterised to the image grid (reference data, "
                          "not a model output)",
                "flood_pct_of_supervised": round(100 * n_ref_flood / n_sup, 4) if n_sup else None,
                "supervised_pct": round(100 * n_sup / stored.size, 4),
                "metrics": _round(metrics),
                "metrics_raw": _round(metrics_raw),
                "metrics_note": "Pixel comparison of the reported flood-like water (model positives outside known "
                                "permanent water) with the EMS reference for this one sample at full resolution. "
                                "Ignored (cloud / invalid / outside the mapped area) pixels are excluded. "
                                "A single sample says little about general accuracy.",
                "split_note": SPLIT_NOTES[sample.split],
            },
            "quality": {"status": res.stats["quality_status"], "warnings": res.warnings},
            "model": self._model_block(),
            "assessment": assessment,
            "water_context": {**wc, "source": "JRC Global Surface Water (WorldFloods PERMANENTWATERJRC: "
                                              "3 = permanent, 2 = seasonal)" if known_water is not None else None,
                              "reference_version": WATER_REFERENCE_VERSION,
                              "layers": {"jrc": "ok" if known_water is not None else "unavailable",
                                         "osm": "not used for archived scenes"}},
            "map": map_block,
            "limitations": self.limitations(),
            "disclaimer": DISCLAIMER,
            "processing_s": round(time.time() - t0, 2),
        }
        self._cache_put(key, payload)
        return {**payload, "freshness": freshness(sample.s2_date, archived=True), "cached": False}

    def _water_change(self, res, baseline, known_water, method, current_date):
        """Year-earlier comparison on the same grid; never raises (failures are reported, not hidden)."""
        from ml.inference import water_change as chg

        if baseline is None:
            return chg.unavailable("Historical comparison was not requested.")
        if baseline.get("error"):
            return chg.unavailable(baseline["error"])
        try:
            with self._infer_lock:
                b = self.segmenter().predict_array(baseline["image"], validity=baseline.get("validity"),
                                                   method=method, source="baseline", make_display=False)
            return chg.compare(res.flood_mask, res.eligible_mask, b.flood_mask, b.eligible_mask, known_water,
                               current_date, baseline.get("acquisition_date"), b.stats.get("hazy", False))
        except Exception as e:
            logger.warning("Historical comparison failed: %s", type(e).__name__)
            return chg.unavailable("The earlier image could not be analysed.")

    def assess_live(
        self,
        rgb_raw: np.ndarray,
        validity: Optional[np.ndarray],
        imagery: Dict[str, Any],
        method: str = DEFAULT_METHOD,
        extra_warnings: Optional[List[str]] = None,
        known_water: Optional[np.ndarray] = None,
        water_info: Optional[Dict[str, Any]] = None,
        bbox: Optional[List[float]] = None,
        baseline: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Segment live imagery (H x W x C in the model's band order). No reference exists, so no accuracy.

        bbox: [west, south, east, north] in EPSG:4326 of the image grid (for map overlays).
        """
        self._check_method(method)
        t0 = time.time()
        with self._infer_lock:
            res = self.segmenter().predict_array(rgb_raw, validity=validity, method=method, source="live")
        if known_water is not None and known_water.shape != res.flood_mask.shape:
            known_water = None
        wc = asm.water_context(res.flood_mask, res.eligible_mask, known_water)
        change = self._water_change(res, baseline, known_water, method, imagery.get("acquisition_date"))
        assessment = asm.classify(wc, method, res.stats["quality_status"], self.policy,
                                  res.stats["eligible_pct"], live=True, hazy=res.stats.get("hazy", False),
                                  change=change)
        outside = asm.outside_water_mask(res.flood_mask, known_water)
        aside = None if known_water is None else (res.flood_mask & (known_water > 0))

        rgb_d = to_display(res.rgb_display)
        mask_d = to_display(res.flood_mask)
        outside_d = to_display(outside)
        aside_d = None if aside is None else to_display(aside)
        map_block = None
        if bbox and len(bbox) == 4:
            from rasterio.transform import from_bounds
            t = from_bounds(*bbox, outside_d.shape[1], outside_d.shape[0])
            map_block = map_layers(rgb_d, outside_d, aside_d, "EPSG:4326", t)
        warnings = [
            "Experimental: live imagery has no ground-truth reference, so this prediction is unvalidated.",
            *(extra_warnings or []),
            *res.warnings,
        ]
        if known_water is None:
            warnings.append("Permanent-water reference unavailable: lakes, rivers and sea were NOT set aside.")
        elif (water_info or {}).get("layers", {}).get("osm") == "unavailable":
            warnings.append("Mapped water bodies (OpenStreetMap) could not be loaded: small ponds, canals and drains "
                            "below the 30 m JRC resolution may be counted as flood-like.")
        return {
            "status": "success",
            "mode": "live",
            "method": method,
            "method_note": METHOD_NOTES[method],
            "experimental": True,
            "sample": None,
            "imagery": imagery,
            "images": {
                "rgb": jpeg_data_url(rgb_d),
                "overlay": jpeg_data_url(si.render_overlay(rgb_d, mask_d)),
                "reference": None,
                "error": None,
                "display_width": int(rgb_d.shape[1]),
                "display_height": int(rgb_d.shape[0]),
                "legend": {"flood": "#%02x%02x%02x" % si.FLOOD_COLOR},
            },
            "prediction": self._prediction_block(res),
            "reference": {"available": False,
                          "note": "No EMS reference mask exists for live imagery; accuracy cannot be measured."},
            "quality": {"status": res.stats["quality_status"], "warnings": warnings},
            "model": self._model_block(),
            "assessment": assessment,
            "water_context": {**wc, "source": (water_info or {}).get("source") if known_water is not None else None,
                              "reference_status": (water_info or {}).get("status", "unavailable"),
                              "reference_version": (water_info or {}).get("version"),
                              "layers": (water_info or {}).get("layers", {}),
                              "osm_categories_px": (water_info or {}).get("osm_categories_px")},
            "map": map_block,
            "change": change,
            "freshness": freshness(imagery.get("acquisition_date"), archived=False),
            "limitations": self.limitations(),
            "disclaimer": DISCLAIMER,
            "processing_s": round(time.time() - t0, 2),
            "cached": False,
        }


_service: Optional[SegmentationService] = None
_service_lock = threading.Lock()


def get_seg_service() -> SegmentationService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = SegmentationService()
    return _service
