"""
Tests for the corrected WorldFloods mask logic in flood_dataset.py.

Uses small synthetic arrays — no file I/O, no API calls, no dataset downloads.
Tests cover:
  1. Clear land pixels
  2. Clear water pixels
  3. Cloudy water pixels (should be excluded)
  4. Invalid pixels (should be excluded from numerator AND denominator)
  5. A mixture of valid land and water
  6. No valid pixels
  7. Permanent-water exclusion
  8. Flood-ratio threshold boundary
  9. Failed sample must not be replaced by another sample

Run with:
    python -m pytest ml/tests/test_mask_logic.py -v
"""
import sys
import os
import numpy as np
import pytest
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.datasets.flood_dataset import (
    WorldFloodsClassificationDataset,
    SampleLoadError,
    LABEL_UNKNOWN,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal dataset instance with _derive_label accessible.
# We don't need actual files for pure mask-logic tests — we call the method
# directly via a class instance with mocked attributes.
# ---------------------------------------------------------------------------

class FakeDataset:
    """Minimal stand-in that exposes _derive_label with configurable threshold."""

    def __init__(self, flood_threshold=0.05, use_permanent_water=True):
        self.flood_threshold = flood_threshold
        self.use_permanent_water = use_permanent_water

    # Bind the real method
    _derive_label = WorldFloodsClassificationDataset._derive_label


def make_masks(cloud_vals, water_vals, shape=(4, 4)):
    """Create cloud_mask and water_mask arrays from flat lists."""
    cloud = np.array(cloud_vals, dtype=np.int16).reshape(shape)
    water = np.array(water_vals, dtype=np.int16).reshape(shape)
    return cloud, water


# ===================================================================
# Test 1: Clear land pixels only → no flood
# ===================================================================
def test_clear_land_only():
    """All pixels are clear sky (Band 1=1) and land (Band 2=1).
    Expected: label=0, flood_ratio=0.0"""
    ds = FakeDataset()
    cloud = np.ones((4, 4), dtype=np.int16)    # all clear
    water = np.ones((4, 4), dtype=np.int16)    # all land
    label, ratio = ds._derive_label(cloud, water, pw_mask=None)
    assert label == 0
    assert ratio == 0.0


# ===================================================================
# Test 2: Clear water pixels only → flood
# ===================================================================
def test_clear_water_only():
    """All pixels are clear sky (Band 1=1) and water (Band 2=2).
    Expected: label=1, flood_ratio=1.0"""
    ds = FakeDataset()
    cloud = np.ones((4, 4), dtype=np.int16)        # all clear
    water = np.full((4, 4), 2, dtype=np.int16)      # all water
    label, ratio = ds._derive_label(cloud, water, pw_mask=None)
    assert label == 1
    assert ratio == 1.0


# ===================================================================
# Test 3: Cloudy water pixels → excluded from valid count
# ===================================================================
def test_cloudy_water_excluded():
    """Water pixels under cloud cover must NOT count as flood.
    4x4 grid: top 2 rows = cloud (Band 1=2) + water (Band 2=2)
              bottom 2 rows = clear (Band 1=1) + land (Band 2=1)
    Only the 8 clear-land pixels are valid. No water is visible.
    Expected: label=0, flood_ratio=0.0"""
    ds = FakeDataset()
    cloud = np.array([
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=np.int16)
    water = np.array([
        [2, 2, 2, 2],  # water under cloud — invalid observation
        [2, 2, 2, 2],
        [1, 1, 1, 1],  # land under clear sky
        [1, 1, 1, 1],
    ], dtype=np.int16)
    label, ratio = ds._derive_label(cloud, water, pw_mask=None)
    assert label == 0
    assert ratio == 0.0


# ===================================================================
# Test 4: Invalid pixels → excluded from BOTH numerator and denominator
# ===================================================================
def test_invalid_pixels_excluded():
    """Pixels where cloud_mask=0 (invalid) or water_mask=0 (invalid)
    must not appear in the denominator or numerator.
    Grid: 8 invalid (cloud=0) + 4 clear-land + 4 clear-water
    Valid = 8, water = 4 → ratio = 0.5
    Expected: label=1, flood_ratio=0.5"""
    ds = FakeDataset()
    cloud = np.array([
        [0, 0, 0, 0],  # invalid (cloud_mask=0)
        [0, 0, 0, 0],  # invalid
        [1, 1, 1, 1],  # clear
        [1, 1, 1, 1],  # clear
    ], dtype=np.int16)
    water = np.array([
        [1, 1, 2, 2],  # doesn't matter — cloud=0 makes them invalid
        [1, 2, 1, 2],
        [1, 1, 1, 1],  # 4 land
        [2, 2, 2, 2],  # 4 water
    ], dtype=np.int16)
    label, ratio = ds._derive_label(cloud, water, pw_mask=None)
    assert label == 1
    assert ratio == pytest.approx(0.5)


def test_invalid_water_mask_excluded():
    """Pixels where water_mask=0 (invalid) are excluded even if cloud is clear.
    Grid: all clear, but 4 pixels have water_mask=0
    Valid = 12, of which 4 are water (Band 2=2)
    Expected: ratio = 4/12 = 0.333"""
    ds = FakeDataset()
    cloud = np.ones((4, 4), dtype=np.int16)  # all clear
    water = np.array([
        [0, 0, 0, 0],  # invalid water layer
        [1, 1, 1, 1],  # land
        [1, 1, 1, 1],  # land
        [2, 2, 2, 2],  # water
    ], dtype=np.int16)
    label, ratio = ds._derive_label(cloud, water, pw_mask=None)
    assert label == 1
    assert ratio == pytest.approx(4 / 12)


# ===================================================================
# Test 5: Mixture of valid land and water
# ===================================================================
def test_mixture_land_water():
    """12 clear-land + 4 clear-water = 25% water.
    With threshold=0.05, label should be FLOOD (1).
    Expected: ratio = 4/16 = 0.25"""
    ds = FakeDataset()
    cloud = np.ones((4, 4), dtype=np.int16)    # all clear
    water = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ], dtype=np.int16)
    label, ratio = ds._derive_label(cloud, water, pw_mask=None)
    assert label == 1
    assert ratio == pytest.approx(0.25)


# ===================================================================
# Test 6: No valid pixels → LABEL_UNKNOWN with ratio 0.0
# ===================================================================
def test_no_valid_pixels_returns_unknown():
    """All pixels are invalid or cloudy — no valid observations.
    Expected: label=LABEL_UNKNOWN (-1), ratio=0.0.
    These samples must NOT be silently labeled NON_FLOOD."""
    ds = FakeDataset()
    # All cloud
    cloud_all_cloud = np.full((4, 4), 2, dtype=np.int16)
    water_all_water = np.full((4, 4), 2, dtype=np.int16)
    label, ratio = ds._derive_label(cloud_all_cloud, water_all_water, pw_mask=None)
    assert label == LABEL_UNKNOWN
    assert ratio == 0.0

    # All invalid
    cloud_all_inv = np.zeros((4, 4), dtype=np.int16)
    water_all_inv = np.zeros((4, 4), dtype=np.int16)
    label2, ratio2 = ds._derive_label(cloud_all_inv, water_all_inv, pw_mask=None)
    assert label2 == LABEL_UNKNOWN
    assert ratio2 == 0.0


# ===================================================================
# Test 7: Permanent-water exclusion
# ===================================================================
def test_permanent_water_exclusion():
    """4 water pixels, but 2 are permanent water (JRC=3).
    Only 2 remain as flood candidates out of 16 valid.
    ratio = 2/16 = 0.125 → label=1 (above 0.05 threshold)"""
    ds = FakeDataset(use_permanent_water=True)
    cloud = np.ones((4, 4), dtype=np.int16)
    water = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ], dtype=np.int16)
    pw_mask = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [3, 3, 1, 1],  # first 2 water pixels are permanent
    ], dtype=np.int16)
    label, ratio = ds._derive_label(cloud, water, pw_mask)
    assert label == 1
    assert ratio == pytest.approx(2 / 16)


def test_permanent_water_all_excluded():
    """All water is permanent → ratio=0 → label=0."""
    ds = FakeDataset(use_permanent_water=True)
    cloud = np.ones((4, 4), dtype=np.int16)
    water = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ], dtype=np.int16)
    pw_mask = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [3, 3, 3, 3],  # all water pixels are permanent
    ], dtype=np.int16)
    label, ratio = ds._derive_label(cloud, water, pw_mask)
    assert label == 0
    assert ratio == 0.0


def test_permanent_water_disabled():
    """If use_permanent_water=False, permanent water is NOT excluded."""
    ds = FakeDataset(use_permanent_water=False)
    cloud = np.ones((4, 4), dtype=np.int16)
    water = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ], dtype=np.int16)
    pw_mask = np.full((4, 4), 3, dtype=np.int16)
    label, ratio = ds._derive_label(cloud, water, pw_mask)
    assert label == 1
    assert ratio == pytest.approx(4 / 16)


# ===================================================================
# Test 8: Flood-ratio threshold boundary
# ===================================================================
def test_threshold_boundary_above():
    """Exactly at threshold + epsilon → FLOOD."""
    ds = FakeDataset(flood_threshold=0.25)
    cloud = np.ones((4, 4), dtype=np.int16)
    # 5 water + 11 land = 5/16 = 0.3125 > 0.25
    water = np.ones((4, 4), dtype=np.int16)
    water[0, 0] = 2; water[0, 1] = 2; water[0, 2] = 2; water[0, 3] = 2; water[1, 0] = 2
    label, ratio = ds._derive_label(cloud, water, pw_mask=None)
    assert label == 1
    assert ratio == pytest.approx(5 / 16)


def test_threshold_boundary_at():
    """Exactly at threshold → NOT flood (strictly greater-than)."""
    # Need ratio == threshold exactly. 1 water pixel out of 20 valid
    ds = FakeDataset(flood_threshold=0.05)
    cloud = np.ones((4, 5), dtype=np.int16)   # 20 pixels
    water = np.ones((4, 5), dtype=np.int16)    # all land
    water[0, 0] = 2  # 1 water pixel → ratio = 1/20 = 0.05
    label, ratio = ds._derive_label(cloud, water, pw_mask=None)
    assert label == 0  # strictly greater than required
    assert ratio == pytest.approx(0.05)


def test_threshold_boundary_below():
    """Just below threshold → NOT flood."""
    # 0 water pixels out of 16 → ratio = 0
    ds = FakeDataset(flood_threshold=0.05)
    cloud = np.ones((4, 4), dtype=np.int16)
    water = np.ones((4, 4), dtype=np.int16)
    label, ratio = ds._derive_label(cloud, water, pw_mask=None)
    assert label == 0
    assert ratio == 0.0


# ===================================================================
# Test 9: Failed sample must NOT be replaced by another sample
# ===================================================================
def test_failed_sample_raises_error():
    """A sample that fails to load must raise SampleLoadError, not
    silently return a different sample."""
    from pathlib import Path
    from unittest.mock import patch

    # We'll create a dataset with fake samples that will fail
    # because the files don't exist.
    ds = WorldFloodsClassificationDataset.__new__(WorldFloodsClassificationDataset)
    ds.target_size = 224
    ds.flood_threshold = 0.05
    ds.rgb_bands = (3, 2, 1)
    ds.transform = None
    ds.use_permanent_water = True
    ds.samples = [
        {"s2": Path("nonexistent_a.tif"), "gt": Path("nonexistent_a_gt.tif"),
         "pw": None, "name": "sample_A"},
        {"s2": Path("nonexistent_b.tif"), "gt": Path("nonexistent_b_gt.tif"),
         "pw": None, "name": "sample_B"},
    ]

    # Requesting sample 0 should raise SampleLoadError, not return sample 1
    with pytest.raises(SampleLoadError) as exc_info:
        ds[0]

    assert exc_info.value.sample_name == "sample_A"
    assert "sample_A" in str(exc_info.value)


def test_sample_load_error_includes_reason():
    """SampleLoadError carries a human-readable reason."""
    err = SampleLoadError("my_sample", "file not found: /a/b.tif")
    assert err.sample_name == "my_sample"
    assert "file not found" in err.reason
    assert "my_sample" in str(err)


# ===================================================================
# Regression guard: old bug treated Band-1 value 1 (clear) as flood
# ===================================================================
def test_regression_band1_clear_is_not_flood():
    """The old code read only Band 1 and treated value 1 (= clear sky)
    as flood water. With the fix, a scene that is 100% clear sky and
    100% land should be NON-FLOOD with ratio 0.0."""
    ds = FakeDataset()
    cloud = np.ones((10, 10), dtype=np.int16)  # all clear
    water = np.ones((10, 10), dtype=np.int16)  # all land
    label, ratio = ds._derive_label(cloud, water, pw_mask=None)
    assert label == 0, "clear-sky + land must not be classified as flood"
    assert ratio == 0.0


# ===================================================================
# Spatial alignment tests
# ===================================================================
def test_center_window_small_raster():
    """A raster smaller than target_px returns a window covering the full raster."""
    w = WorldFloodsClassificationDataset._center_window(100, 200, target_px=512)
    assert w.col_off == 0
    assert w.row_off == 0
    assert w.width == 200
    assert w.height == 100


def test_center_window_large_raster():
    """A raster larger than target_px returns a centered 512x512 window."""
    w = WorldFloodsClassificationDataset._center_window(1000, 2000, target_px=512)
    assert w.width == 512
    assert w.height == 512
    assert w.row_off == (1000 - 512) // 2
    assert w.col_off == (2000 - 512) // 2


def test_center_window_exact_size():
    """A raster exactly target_px returns a zero-offset full window."""
    w = WorldFloodsClassificationDataset._center_window(512, 512, target_px=512)
    assert w.col_off == 0
    assert w.row_off == 0
    assert w.width == 512
    assert w.height == 512


def test_unknown_label_in_getitem_raises():
    """A sample with all-invalid GT in the window must raise SampleLoadError
    from __getitem__ (not silently return label=0)."""
    import tempfile
    import rasterio
    from rasterio.transform import from_bounds

    # Create tiny aligned rasters: S2 (3 bands), GT (2 bands), no PW
    h, w = 4, 4
    transform = from_bounds(0, 0, 40, 40, w, h)
    crs = rasterio.crs.CRS.from_epsg(32633)

    with tempfile.TemporaryDirectory() as tmpdir:
        s2_path = os.path.join(tmpdir, "s2.tif")
        gt_path = os.path.join(tmpdir, "gt.tif")

        # S2: 15 bands, some data
        with rasterio.open(s2_path, "w", driver="GTiff", height=h, width=w,
                           count=15, dtype="uint16", crs=crs, transform=transform) as dst:
            for b in range(1, 16):
                dst.write(np.ones((h, w), dtype=np.uint16) * 1000, b)

        # GT: Band 1 all invalid (0), Band 2 all invalid (0)
        with rasterio.open(gt_path, "w", driver="GTiff", height=h, width=w,
                           count=2, dtype="int16", crs=crs, transform=transform) as dst:
            dst.write(np.zeros((h, w), dtype=np.int16), 1)
            dst.write(np.zeros((h, w), dtype=np.int16), 2)

        ds = WorldFloodsClassificationDataset.__new__(WorldFloodsClassificationDataset)
        ds.target_size = 224
        ds.flood_threshold = 0.05
        ds.rgb_bands = (3, 2, 1)
        ds.transform = None
        ds.use_permanent_water = True
        ds.samples = [
            {"s2": Path(s2_path), "gt": Path(gt_path), "pw": None, "name": "test_unknown"},
        ]

        with pytest.raises(SampleLoadError) as exc_info:
            ds[0]
        assert "unknown" in str(exc_info.value).lower()


def test_dimension_mismatch_raises():
    """Mismatched S2 and GT dimensions must raise SampleLoadError."""
    import tempfile
    import rasterio
    from rasterio.transform import from_bounds

    crs = rasterio.crs.CRS.from_epsg(32633)

    with tempfile.TemporaryDirectory() as tmpdir:
        s2_path = os.path.join(tmpdir, "s2.tif")
        gt_path = os.path.join(tmpdir, "gt.tif")

        # S2: 4x4
        tf4 = from_bounds(0, 0, 40, 40, 4, 4)
        with rasterio.open(s2_path, "w", driver="GTiff", height=4, width=4,
                           count=15, dtype="uint16", crs=crs, transform=tf4) as dst:
            for b in range(1, 16):
                dst.write(np.ones((4, 4), dtype=np.uint16), b)

        # GT: 8x8 (different dimensions)
        tf8 = from_bounds(0, 0, 80, 80, 8, 8)
        with rasterio.open(gt_path, "w", driver="GTiff", height=8, width=8,
                           count=2, dtype="int16", crs=crs, transform=tf8) as dst:
            dst.write(np.ones((8, 8), dtype=np.int16), 1)
            dst.write(np.ones((8, 8), dtype=np.int16), 2)

        ds = WorldFloodsClassificationDataset.__new__(WorldFloodsClassificationDataset)
        ds.target_size = 224
        ds.flood_threshold = 0.05
        ds.rgb_bands = (3, 2, 1)
        ds.transform = None
        ds.use_permanent_water = True
        ds.samples = [
            {"s2": Path(s2_path), "gt": Path(gt_path), "pw": None, "name": "mismatch"},
        ]

        with pytest.raises(SampleLoadError) as exc_info:
            ds[0]
        assert "dimensions" in str(exc_info.value).lower() or "differ" in str(exc_info.value).lower()


def test_missing_pw_mask_works():
    """A sample with pw=None should still load successfully."""
    import tempfile
    import rasterio
    from rasterio.transform import from_bounds

    h, w = 4, 4
    transform = from_bounds(0, 0, 40, 40, w, h)
    crs = rasterio.crs.CRS.from_epsg(32633)

    with tempfile.TemporaryDirectory() as tmpdir:
        s2_path = os.path.join(tmpdir, "s2.tif")
        gt_path = os.path.join(tmpdir, "gt.tif")

        with rasterio.open(s2_path, "w", driver="GTiff", height=h, width=w,
                           count=15, dtype="uint16", crs=crs, transform=transform) as dst:
            for b in range(1, 16):
                dst.write(np.ones((h, w), dtype=np.uint16) * 1000, b)

        # GT: all clear + all water
        with rasterio.open(gt_path, "w", driver="GTiff", height=h, width=w,
                           count=2, dtype="int16", crs=crs, transform=transform) as dst:
            dst.write(np.ones((h, w), dtype=np.int16), 1)       # Band 1: all clear
            dst.write(np.full((h, w), 2, dtype=np.int16), 2)    # Band 2: all water

        ds = WorldFloodsClassificationDataset.__new__(WorldFloodsClassificationDataset)
        ds.target_size = 224
        ds.flood_threshold = 0.05
        ds.rgb_bands = (3, 2, 1)
        ds.transform = None
        ds.use_permanent_water = True
        ds.samples = [
            {"s2": Path(s2_path), "gt": Path(gt_path), "pw": None, "name": "no_pw"},
        ]

        img, lbl = ds[0]
        assert img.shape == (3, 224, 224)
        assert lbl.item() == 1  # all water → flood


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
