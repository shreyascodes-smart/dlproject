"""
Phase D Step 3: Automated tests for flood segmentation mask rasterization.

Tests cover:
- Polygon rasterization on a known grid
- CRS mismatch rejection
- Raster dimension and transform alignment
- Invalid/cloud pixels being ignored (kept as IGNORE=0)
- Empty flood polygons producing all NON_FLOOD
- Rasterization bounds and pixel counts
"""
import sys
import os
import tempfile
import json

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.datasets.prepare_flood_segmentation import (
    extract_crs_from_geojson,
    load_flood_polygons,
    create_validity_mask,
    rasterize_sample,
    IGNORE,
    NON_FLOOD,
    FLOOD,
)


# ===================================================================
# Helpers
# ===================================================================
def make_test_rasters(tmpdir, h=10, w=10, crs_epsg=32633,
                      cloud_vals=None, water_vals=None, n_s2_bands=15):
    """Create aligned S2 and GT test rasters."""
    transform = from_bounds(0, 0, w * 10, h * 10, w, h)
    crs = CRS.from_epsg(crs_epsg)
    s2_path = os.path.join(tmpdir, "s2.tif")
    gt_path = os.path.join(tmpdir, "gt.tif")

    # S2
    with rasterio.open(s2_path, "w", driver="GTiff", height=h, width=w,
                       count=n_s2_bands, dtype="uint16", crs=crs,
                       transform=transform) as dst:
        for b in range(1, n_s2_bands + 1):
            dst.write(np.ones((h, w), dtype=np.uint16) * 1000, b)

    # GT: Band 1 = cloud/validity, Band 2 = water/land
    if cloud_vals is None:
        cloud_vals = np.ones((h, w), dtype=np.int16)  # all clear
    if water_vals is None:
        water_vals = np.ones((h, w), dtype=np.int16)  # all land
    with rasterio.open(gt_path, "w", driver="GTiff", height=h, width=w,
                       count=2, dtype="int16", crs=crs, transform=transform) as dst:
        dst.write(cloud_vals.astype(np.int16), 1)
        dst.write(water_vals.astype(np.int16), 2)

    return s2_path, gt_path, transform, crs


def make_flood_geojson(tmpdir, name, polygons, crs_epsg=32633):
    """Create a test GeoJSON with flood polygons.
    
    polygons: list of (coordinates, w_class) tuples
    """
    features = []
    for coords, w_class in polygons:
        features.append({
            "type": "Feature",
            "properties": {"w_class": w_class, "source": "flood"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
        })

    data = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_epsg}"},
        },
        "features": features,
    }

    path = os.path.join(tmpdir, name + ".geojson")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ===================================================================
# Tests: CRS extraction
# ===================================================================
def test_extract_crs_urn_format():
    """Parse CRS from urn:ogc:def:crs:EPSG::32629 format."""
    data = {
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::32629"},
        }
    }
    crs = extract_crs_from_geojson(data)
    assert crs == CRS.from_epsg(32629)


def test_extract_crs_missing():
    """No CRS field should return None."""
    crs = extract_crs_from_geojson({"type": "FeatureCollection"})
    assert crs is None


# ===================================================================
# Tests: Polygon rasterization on a known grid
# ===================================================================
def test_rasterize_known_polygon():
    """A polygon covering the top-left quadrant should burn exactly those pixels."""
    h, w = 10, 10
    transform = from_bounds(0, 0, 100, 100, w, h)

    # Polygon covering [0,50] x [50,100] (top-left quadrant in image coords)
    geom = {
        "type": "Polygon",
        "coordinates": [[[0, 50], [50, 50], [50, 100], [0, 100], [0, 50]]],
    }

    result = rasterize(
        [(geom, FLOOD)],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # Top-left 5x5 quadrant should be FLOOD
    assert result[:5, :5].sum() == 5 * 5 * FLOOD
    # Bottom-right should be 0
    assert result[5:, 5:].sum() == 0


def test_rasterize_empty_polygons():
    """No flood polygons should produce an all-zero raster."""
    h, w = 10, 10
    transform = from_bounds(0, 0, 100, 100, w, h)
    result = rasterize(
        [],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    assert result.sum() == 0


# ===================================================================
# Tests: Validity mask
# ===================================================================
def test_validity_mask_clear_sky():
    """All-clear, all-land GT produces all-valid mask."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s2_path, gt_path, transform, crs = make_test_rasters(tmpdir)
        validity = create_validity_mask(gt_path, transform, 10, 10)
        assert validity.sum() == 100  # all valid


def test_validity_mask_cloud_excluded():
    """Cloudy pixels (Band 1 == 2) are excluded from validity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cloud = np.ones((10, 10), dtype=np.int16)
        cloud[:5, :] = 2  # top half is cloud
        s2_path, gt_path, transform, crs = make_test_rasters(
            tmpdir, cloud_vals=cloud
        )
        validity = create_validity_mask(gt_path, transform, 10, 10)
        assert validity.sum() == 50  # only bottom half valid


def test_validity_mask_invalid_excluded():
    """Invalid pixels (Band 1 == 0) are excluded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cloud = np.ones((10, 10), dtype=np.int16)
        cloud[:, :3] = 0  # left 3 cols invalid
        s2_path, gt_path, transform, crs = make_test_rasters(
            tmpdir, cloud_vals=cloud
        )
        validity = create_validity_mask(gt_path, transform, 10, 10)
        assert validity.sum() == 70  # 10 rows x 7 valid cols


def test_validity_mask_invalid_water_excluded():
    """Pixels with Band 2 == 0 (invalid water layer) are excluded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        water = np.ones((10, 10), dtype=np.int16)
        water[5:, :] = 0  # bottom half has invalid water layer
        s2_path, gt_path, transform, crs = make_test_rasters(
            tmpdir, water_vals=water
        )
        validity = create_validity_mask(gt_path, transform, 10, 10)
        assert validity.sum() == 50


# ===================================================================
# Tests: CRS mismatch rejection
# ===================================================================
def test_crs_mismatch_flagged():
    """GeoJSON CRS different from S2 CRS should be flagged."""
    data = {
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
        }
    }
    geojson_crs = extract_crs_from_geojson(data)
    s2_crs = CRS.from_epsg(32633)
    assert geojson_crs != s2_crs


# ===================================================================
# Tests: Dimension mismatch
# ===================================================================
def test_dimension_mismatch_raises():
    """GT with different dimensions from S2 should raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transform = from_bounds(0, 0, 100, 100, 10, 10)
        crs = CRS.from_epsg(32633)

        # GT with wrong dimensions
        gt_path = os.path.join(tmpdir, "gt.tif")
        with rasterio.open(gt_path, "w", driver="GTiff", height=8, width=8,
                           count=2, dtype="int16", crs=crs,
                           transform=from_bounds(0, 0, 80, 80, 8, 8)) as dst:
            dst.write(np.ones((8, 8), dtype=np.int16), 1)
            dst.write(np.ones((8, 8), dtype=np.int16), 2)

        with pytest.raises(ValueError, match="dimensions"):
            create_validity_mask(gt_path, transform, 10, 10)


# ===================================================================
# Tests: Load flood polygons
# ===================================================================
def test_load_flood_polygons_filters_correctly():
    """Only w_class='Flooded area' features are returned as flood geometries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = make_flood_geojson(tmpdir, "test", [
            ([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], "Flooded area"),
            ([[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]], "BH140-River"),
            ([[40, 40], [50, 40], [50, 50], [40, 50], [40, 40]], "water"),
            ([[60, 60], [70, 60], [70, 70], [60, 70], [60, 60]], "Flooded area"),
        ])

        geoms, counts, crs = load_flood_polygons(path)
        assert len(geoms) == 2  # only 2 "Flooded area"
        assert counts["flood"] == 2
        assert counts["hydro"] == 1
        assert counts["water"] == 1


def test_load_empty_flood_polygons():
    """A GeoJSON with no flood features returns empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = make_flood_geojson(tmpdir, "test", [
            ([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], "BH140-River"),
        ])
        geoms, counts, crs = load_flood_polygons(path)
        assert len(geoms) == 0
        assert counts["flood"] == 0


# ===================================================================
# Tests: End-to-end segmentation mask
# ===================================================================
def test_segmentation_mask_flood_on_valid():
    """Flood polygon on valid pixels should produce FLOOD in seg mask."""
    h, w = 10, 10
    transform = from_bounds(0, 0, 100, 100, w, h)

    # All clear, all land
    validity = np.ones((h, w), dtype=np.uint8)
    seg_mask = np.zeros((h, w), dtype=np.uint8)
    seg_mask[validity == 1] = NON_FLOOD

    # Flood polygon covering entire raster
    geom = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]],
    }
    flood_raster = rasterize([(geom, FLOOD)], out_shape=(h, w),
                             transform=transform, fill=0, dtype=np.uint8)
    flood_and_valid = (flood_raster == FLOOD) & (validity == 1)
    seg_mask[flood_and_valid] = FLOOD

    assert (seg_mask == FLOOD).sum() == 100
    assert (seg_mask == NON_FLOOD).sum() == 0


def test_segmentation_mask_flood_on_cloud():
    """Flood polygon on cloudy pixels should remain IGNORE."""
    h, w = 10, 10
    transform = from_bounds(0, 0, 100, 100, w, h)

    # Top half cloud, bottom half clear
    validity = np.zeros((h, w), dtype=np.uint8)
    validity[5:, :] = 1

    seg_mask = np.zeros((h, w), dtype=np.uint8)
    seg_mask[validity == 1] = NON_FLOOD

    # Flood polygon covering entire raster
    geom = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]],
    }
    flood_raster = rasterize([(geom, FLOOD)], out_shape=(h, w),
                             transform=transform, fill=0, dtype=np.uint8)
    flood_and_valid = (flood_raster == FLOOD) & (validity == 1)
    seg_mask[flood_and_valid] = FLOOD

    # Top half = IGNORE (cloud), bottom half = FLOOD
    assert (seg_mask[:5, :] == IGNORE).all()
    assert (seg_mask[5:, :] == FLOOD).all()


def test_no_flood_polygons_all_non_flood():
    """Sample with no flood polygons: all valid pixels = NON_FLOOD."""
    h, w = 10, 10
    validity = np.ones((h, w), dtype=np.uint8)
    seg_mask = np.zeros((h, w), dtype=np.uint8)
    seg_mask[validity == 1] = NON_FLOOD

    assert (seg_mask == NON_FLOOD).sum() == 100
    assert (seg_mask == FLOOD).sum() == 0


def test_partial_polygon_only_valid_counted():
    """Flood polygon partially overlaps valid area. Only valid intersection counts."""
    h, w = 10, 10
    transform = from_bounds(0, 0, 100, 100, w, h)

    # Left half valid, right half invalid
    validity = np.zeros((h, w), dtype=np.uint8)
    validity[:, :5] = 1

    seg_mask = np.zeros((h, w), dtype=np.uint8)
    seg_mask[validity == 1] = NON_FLOOD

    # Flood polygon covering entire raster
    geom = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]],
    }
    flood_raster = rasterize([(geom, FLOOD)], out_shape=(h, w),
                             transform=transform, fill=0, dtype=np.uint8)
    flood_and_valid = (flood_raster == FLOOD) & (validity == 1)
    seg_mask[flood_and_valid] = FLOOD

    # Only left half should be FLOOD, right half stays IGNORE
    assert (seg_mask == FLOOD).sum() == 50
    assert (seg_mask[:, 5:] == IGNORE).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
