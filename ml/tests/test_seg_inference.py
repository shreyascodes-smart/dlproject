"""
Tests for ml/inference/seg_inference.py (standalone flood segmentation inference).

Most tests use synthetic arrays and a tiny deterministic "mock" model so they
run on CPU in seconds without the dataset. Tests that need the real checkpoint
or WorldFloods rasters are skipped automatically if those files are missing.

The mock model is per-pixel: P(flood) is high for dark pixels and low for
bright pixels. Because it looks at each pixel independently, the stitched
prediction must match the dark region exactly, which lets us test tiling,
padding, cropping and overlay alignment precisely.
"""
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import seg_inference as si  # noqa: E402
from ml.datasets.flood_seg_dataset import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    FloodSegDataset,
    get_val_transform,
    normalize_s2_rgb,
)

CKPT = PROJECT_ROOT / "models" / "flood" / "segmentation_v1" / "best_seg_model.pt"
MANIFEST = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "seg_split_manifest.json"
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "flood" / "worldfloods_v2"
MASKS = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "masks"

DARK_DN = 100.0    # -> normalised about -2 per channel -> mock says flood
BRIGHT_DN = 5000.0  # -> normalised about +0.2..+0.4 -> mock says non-flood


class PerPixelMock(nn.Module):
    """Logits: class0 = 0, class1 = -k * sum(channels). Dark pixels => flood."""

    def __init__(self, k: float = 5.0):
        super().__init__()
        self.k = k
        self.dummy = nn.Parameter(torch.zeros(1))  # so .to(device) works

    def forward(self, x):
        flood = -self.k * x.sum(dim=1, keepdim=True)
        return torch.cat([torch.zeros_like(flood), flood], dim=1)


class BadShapeMock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])


def make_scene(h, w, box):
    """Bright scene with a dark rectangle box=(y0, y1, x0, x1). Returns (rgb, expected_flood)."""
    rgb = np.full((h, w, 3), BRIGHT_DN, dtype=np.float32)
    y0, y1, x0, x1 = box
    rgb[y0:y1, x0:x1] = DARK_DN
    expected = np.zeros((h, w), bool)
    expected[y0:y1, x0:x1] = True
    return rgb, expected


@pytest.fixture
def seg():
    return si.FloodSegmenter(model=PerPixelMock(), device="cpu")


# ------------------------------------------------------------
# Device selection
# ------------------------------------------------------------
class TestDevice:
    def test_cpu_explicit(self):
        dev, warn = si.select_device("cpu")
        assert dev.type == "cpu" and warn is None

    def test_cuda_requested_but_missing_falls_back(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        dev, warn = si.select_device("cuda")
        assert dev.type == "cpu" and "not available" in warn

    def test_auto_without_cuda_is_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert si.select_device(None)[0].type == "cpu"

    def test_unknown_device_rejected(self):
        with pytest.raises(ValueError):
            si.select_device("tpu")

    def test_fallback_warning_reaches_result(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        s = si.FloodSegmenter(model=PerPixelMock(), device="cuda")
        rgb, _ = make_scene(64, 64, (0, 10, 0, 10))
        r = s.predict_array(rgb, method="tiled")
        assert any("CUDA" in w for w in r.warnings)


# ------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------
class TestCheckpointLoading:
    def test_missing_checkpoint(self, tmp_path):
        with pytest.raises(si.ModelLoadError, match="not found"):
            si.load_segmentation_model(tmp_path / "nope.pt")

    def test_checkpoint_without_state_dict(self, tmp_path):
        p = tmp_path / "bad.pt"
        torch.save({"epoch": 1}, p)
        with pytest.raises(si.ModelLoadError, match="model_state_dict"):
            si.load_segmentation_model(p)

    def test_checkpoint_with_wrong_weights(self, tmp_path):
        p = tmp_path / "wrong.pt"
        torch.save({"model_state_dict": {"foo": torch.zeros(1)}, "config": {"num_classes": 2}}, p)
        with pytest.raises(si.ModelLoadError, match="do not match"):
            si.load_segmentation_model(p)

    def test_corrupt_file(self, tmp_path):
        p = tmp_path / "corrupt.pt"
        p.write_bytes(b"not a checkpoint")
        with pytest.raises(si.ModelLoadError):
            si.load_segmentation_model(p)

    def test_wrong_class_count(self, tmp_path):
        p = tmp_path / "three.pt"
        torch.save({"model_state_dict": {}, "config": {"num_classes": 3}}, p)
        with pytest.raises(si.ModelLoadError, match="2-class"):
            si.load_segmentation_model(p)

    @pytest.mark.skipif(not CKPT.exists(), reason="trained checkpoint not present")
    def test_real_checkpoint_loads_and_runs(self):
        model, meta = si.load_segmentation_model(CKPT, torch.device("cpu"))
        assert meta["encoder"] == "efficientnet-b2"
        assert meta["epoch"] == 14
        assert meta["input_size"] == 256
        assert not model.training
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 256, 256))
        assert tuple(out.shape) == (1, 2, 256, 256)
        assert torch.isfinite(out).all()

    @pytest.mark.skipif(not CKPT.exists(), reason="trained checkpoint not present")
    def test_checkpoint_file_unchanged_by_loading(self):
        before = (CKPT.stat().st_size, CKPT.stat().st_mtime_ns)
        si.load_segmentation_model(CKPT, torch.device("cpu"))
        assert (CKPT.stat().st_size, CKPT.stat().st_mtime_ns) == before


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
class TestPreprocessing:
    def test_resize_path_matches_dataset_pipeline(self):
        rng = np.random.default_rng(0)
        rgb = rng.uniform(0, 12000, (300, 410, 3)).astype(np.float32)
        expected = normalize_s2_rgb(get_val_transform(256)(image=rgb)["image"])
        np.testing.assert_array_equal(si.preprocess_resize(rgb, 256), expected)

    def test_small_image_padded_like_training(self):
        rgb = np.full((100, 300, 3), 2000, np.float32)
        padded = si.pad_to_min_size(rgb, 256)
        assert padded.shape == (256, 300, 3)
        assert np.all(padded[100:] == 0) and np.all(padded[:100] == 2000)

    def test_normalisation_constants(self):
        x = normalize_s2_rgb(np.full((1, 1, 3), 10000.0, np.float32))[0, 0]
        exp = (1.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        np.testing.assert_allclose(x, exp, rtol=1e-6)

    def test_dn_clipping_above_10000(self):
        a = normalize_s2_rgb(np.full((1, 1, 3), 25000.0, np.float32))
        b = normalize_s2_rgb(np.full((1, 1, 3), 10000.0, np.float32))
        np.testing.assert_array_equal(a, b)

    def test_negative_dn_clipped_on_input(self, seg):
        rgb, _ = make_scene(64, 64, (0, 0, 0, 0))
        rgb[0, 0] = -500
        r = seg.predict_array(rgb, method="tiled")
        assert np.isfinite(r.flood_probability).all()

    def test_tile_starts_cover_everything(self):
        for length in (256, 257, 300, 511, 1000):
            starts = si.tile_starts(length, 256, 192)
            covered = np.zeros(length, bool)
            for s in starts:
                assert 0 <= s <= length - 256
                covered[s:s + 256] = True
            assert covered.all() and starts[-1] == length - 256

    def test_blend_window_positive(self):
        w = si.blend_window(256, 64)
        assert w.shape == (256, 256) and w.min() > 0 and w.max() == pytest.approx(1.0)

    @pytest.mark.skipif(not (MANIFEST.exists() and DATA_ROOT.exists() and MASKS.exists()),
                        reason="dataset not present")
    def test_matches_dataset_on_real_val_sample(self):
        ds = FloodSegDataset(str(MANIFEST), "val", str(DATA_ROOT), str(MASKS), 256, augment=False)
        idx = min(range(len(ds)), key=lambda i: ds.samples[i].get("supervised_px", 0) or 1e18)
        s = ds.samples[idx]
        img_ds, _ = ds[idx]
        rgb = si.read_s2_rgb(DATA_ROOT / s["original_split"] / "S2" / f"{s['name']}.tif")
        mine = si.preprocess_resize(rgb, 256)
        np.testing.assert_allclose(mine, img_ds.permute(1, 2, 0).numpy(), atol=1e-6)


# ------------------------------------------------------------
# Mask generation, tiling and alignment
# ------------------------------------------------------------
class TestMaskAndAlignment:
    @pytest.mark.parametrize("h,w,box", [
        (256, 256, (10, 50, 200, 250)),
        (300, 517, (0, 40, 480, 517)),       # touches right edge
        (700, 333, (650, 700, 0, 30)),        # touches bottom-left corner
        (100, 80, (20, 60, 10, 70)),          # smaller than one tile (padded)
    ])
    @pytest.mark.parametrize("overlap", [0, 64])
    def test_tiled_mask_exactly_aligned(self, seg, h, w, box, overlap):
        rgb, expected = make_scene(h, w, box)
        r = seg.predict_array(rgb, method="tiled", overlap=overlap)
        assert r.flood_mask.shape == (h, w)
        assert r.flood_probability.shape == (h, w)
        assert r.rgb_display.shape == (h, w, 3)
        np.testing.assert_array_equal(r.flood_mask, expected)

    def test_resize_output_on_original_grid(self, seg):
        rgb, expected = make_scene(600, 900, (100, 400, 200, 700))
        r = seg.predict_array(rgb, method="resize")
        assert r.flood_mask.shape == (600, 900)
        # resize blurs edges; interior must be flood, far background must not
        assert r.flood_mask[150:350, 250:650].all()
        assert not r.flood_mask[:50].any() and not r.flood_mask[:, 850:].any()

    def test_threshold_is_argmax(self, seg):
        rgb, _ = make_scene(64, 64, (0, 32, 0, 64))
        r = seg.predict_array(rgb, method="tiled")
        np.testing.assert_array_equal(r.flood_mask, (r.flood_probability > 0.5) & r.eligible_mask)

    def test_probabilities_in_unit_range(self, seg):
        rgb, _ = make_scene(300, 300, (0, 100, 0, 100))
        p = seg.predict_array(rgb, method="tiled").flood_probability
        assert p.min() >= 0 and p.max() <= 1

    def test_overlay_changes_only_flood_pixels(self, seg):
        rgb, expected = make_scene(200, 300, (50, 120, 30, 90))
        r = seg.predict_array(rgb, method="tiled")
        ov = si.render_overlay(r.rgb_display, r.flood_mask)
        changed = np.any(ov != r.rgb_display, axis=-1)
        np.testing.assert_array_equal(changed, expected)

    def test_overlay_shape_mismatch_rejected(self):
        with pytest.raises(si.InvalidInputError):
            si.render_overlay(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 11), bool))

    def test_bad_model_output_shape_raises(self):
        s = si.FloodSegmenter(model=BadShapeMock(), device="cpu")
        rgb, _ = make_scene(64, 64, (0, 1, 0, 1))
        with pytest.raises(RuntimeError, match="Unexpected model output"):
            s.predict_array(rgb, method="tiled")


# ------------------------------------------------------------
# Coverage, validity and quality
# ------------------------------------------------------------
class TestCoverage:
    def test_coverage_without_validity(self, seg):
        rgb, _ = make_scene(100, 100, (0, 25, 0, 100))  # 25% dark
        r = seg.predict_array(rgb, method="tiled")
        assert r.stats["flood_coverage_pct"] == pytest.approx(25.0)
        assert r.stats["eligibility_source"] == "nodata_only"
        assert any("No cloud/validity mask" in w for w in r.warnings)

    def test_nodata_pixels_excluded_from_denominator(self, seg):
        rgb, _ = make_scene(100, 100, (0, 20, 0, 100))  # 20 rows flood
        rgb[80:] = 0                                      # 20 rows no data
        r = seg.predict_array(rgb, method="tiled")
        assert r.stats["n_eligible"] == 8000
        assert r.stats["nodata_pct"] == pytest.approx(20.0)
        assert r.stats["flood_coverage_pct"] == pytest.approx(100 * 2000 / 8000)
        assert not r.flood_mask[80:].any()

    def test_validity_mask_restricts_flood_and_denominator(self, seg):
        rgb, _ = make_scene(100, 100, (0, 50, 0, 100))  # top half dark
        validity = np.zeros((100, 100), np.uint8)
        validity[:, :50] = 1                              # left half valid
        r = seg.predict_array(rgb, validity=validity, method="tiled")
        assert r.stats["n_eligible"] == 5000
        assert r.stats["n_flood_pred"] == 2500
        assert r.stats["flood_coverage_pct"] == pytest.approx(50.0)
        assert not r.flood_mask[:, 50:].any()
        assert r.stats["eligibility_source"] == "validity_mask+nodata"

    def test_no_eligible_pixels(self, seg):
        rgb = np.zeros((64, 64, 3), np.float32)
        r = seg.predict_array(rgb, method="tiled")
        assert r.stats["flood_coverage_pct"] is None
        assert r.stats["quality_status"] == "no_valid_pixels"
        assert not r.flood_mask.any()

    def test_limited_quality_flag(self, seg):
        rgb, _ = make_scene(100, 100, (0, 0, 0, 0))
        validity = np.zeros((100, 100), np.uint8)
        validity[:30] = 1
        r = seg.predict_array(rgb, validity=validity, method="tiled")
        assert r.stats["quality_status"] == "limited"

    def test_resize_warning_for_large_scene(self, seg):
        rgb, _ = make_scene(600, 400, (0, 0, 0, 0))
        r = seg.predict_array(rgb, method="resize")
        assert any("Whole-scene resize" in w for w in r.warnings)

    def test_stats_document_definition(self, seg):
        rgb, _ = make_scene(64, 64, (0, 0, 0, 0))
        st = seg.predict_array(rgb, method="tiled").stats
        assert "eligible" in st["coverage_definition"] and "not area" in st["coverage_definition"]


# ------------------------------------------------------------
# Invalid inputs
# ------------------------------------------------------------
class TestInvalidInputs:
    @pytest.mark.parametrize("arr", [
        np.zeros((64, 64), np.float32),
        np.zeros((64, 64, 4), np.float32),
        np.zeros((0, 64, 3), np.float32),
    ])
    def test_bad_shapes(self, seg, arr):
        with pytest.raises(si.InvalidInputError):
            seg.predict_array(arr)

    def test_nan_rejected(self, seg):
        rgb = np.full((64, 64, 3), 1000, np.float32)
        rgb[3, 3, 1] = np.nan
        with pytest.raises(si.InvalidInputError, match="NaN"):
            seg.predict_array(rgb)

    def test_validity_shape_mismatch(self, seg):
        with pytest.raises(si.InvalidInputError, match="Validity"):
            seg.predict_array(np.ones((64, 64, 3), np.float32), validity=np.ones((64, 63)))

    def test_unknown_method(self, seg):
        with pytest.raises(ValueError, match="method"):
            seg.predict_array(np.ones((64, 64, 3), np.float32), method="magic")

    def test_bad_overlap(self, seg):
        with pytest.raises(ValueError, match="overlap"):
            seg.predict_array(np.ones((64, 64, 3), np.float32), method="tiled", overlap=300)

    def test_missing_image_file(self, seg, tmp_path):
        with pytest.raises(si.InvalidInputError, match="not found"):
            seg.predict_file(tmp_path / "missing.tif")

    def test_too_few_bands(self, seg, tmp_path):
        import rasterio
        p = tmp_path / "two.tif"
        with rasterio.open(p, "w", driver="GTiff", height=8, width=8, count=2, dtype="uint16") as dst:
            dst.write(np.ones((2, 8, 8), np.uint16))
        with pytest.raises(si.InvalidInputError, match="bands"):
            seg.predict_file(p)


# ------------------------------------------------------------
# File reading and band order
# ------------------------------------------------------------
class TestFileReading:
    def _write(self, path, bands):
        import rasterio
        c, h, w = bands.shape
        with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=c, dtype=bands.dtype) as dst:
            dst.write(bands)

    def test_worldfloods_band_order(self, tmp_path):
        bands = np.zeros((15, 4, 4), np.int16)
        bands[3], bands[2], bands[1] = 400, 300, 200   # B4, B3, B2
        bands[0] = 999                                   # B1 must not be used
        p = tmp_path / "wf.tif"
        self._write(p, bands)
        rgb = si.read_s2_rgb(p)
        assert rgb[0, 0].tolist() == [400, 300, 200]

    def test_live_band_order(self, tmp_path):
        bands = np.zeros((6, 4, 4), np.uint16)
        bands[0], bands[1], bands[2] = 200, 300, 400    # B02, B03, B04
        p = tmp_path / "live.tif"
        self._write(p, bands)
        rgb = si.read_s2_rgb(p, si.LIVE_RGB_BANDS_0IDX)
        assert rgb[0, 0].tolist() == [400, 300, 200]

    def test_predict_file_with_validity(self, seg, tmp_path):
        import rasterio
        bands = np.full((15, 64, 64), int(BRIGHT_DN), np.int16)
        bands[1:4, :32] = int(DARK_DN)
        img, val = tmp_path / "s2.tif", tmp_path / "val.tif"
        self._write(img, bands)
        with rasterio.open(val, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint8") as dst:
            v = np.ones((64, 64), np.uint8); v[:, 48:] = 0
            dst.write(v, 1)
        r = seg.predict_file(img, validity_path=val, method="tiled")
        assert r.stats["source"] == "s2.tif"
        assert r.stats["n_eligible"] == 64 * 48
        assert r.stats["flood_coverage_pct"] == pytest.approx(50.0)


# ------------------------------------------------------------
# Reference comparison and rendering
# ------------------------------------------------------------
class TestReferenceAndRendering:
    def test_confusion_counts(self):
        ref = np.array([[0, 1, 1, 2, 2]], np.uint8)
        pred = np.array([[True, True, False, True, False]])
        codes, m = si.compare_with_reference(pred, ref)
        assert codes.tolist() == [[si.ERR_IGNORE, si.ERR_FP, si.ERR_TN, si.ERR_TP, si.ERR_FN]]
        assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
        assert m["flood_iou"] == pytest.approx(1 / 3)

    def test_metrics_undefined_when_no_flood(self):
        m = si.confusion_metrics(0, 0, 0, 10)
        assert m["flood_iou"] is None and m["pixel_accuracy"] == 1.0

    def test_reference_render_distinguishes_ignore(self):
        rgb = np.full((1, 3, 3), 200, np.uint8)
        out = si.render_reference(rgb, np.array([[0, 1, 2]], np.uint8))
        assert tuple(out[0, 0]) == si.IGNORE_COLOR
        assert tuple(out[0, 2]) == si.FLOOD_COLOR
        assert tuple(out[0, 1]) not in (si.IGNORE_COLOR, si.FLOOD_COLOR)

    def test_error_map_colours(self):
        codes = np.array([[si.ERR_IGNORE, si.ERR_TP, si.ERR_FP, si.ERR_FN]], np.uint8)
        out = si.render_error_map(np.full((1, 4, 3), 100, np.uint8), codes)
        assert tuple(out[0, 0]) == si.IGNORE_COLOR
        assert tuple(out[0, 1]) == si.ERROR_COLORS[si.ERR_TP]
        assert tuple(out[0, 2]) == si.ERROR_COLORS[si.ERR_FP]
        assert tuple(out[0, 3]) == si.ERROR_COLORS[si.ERR_FN]

    def test_compare_shape_mismatch(self):
        with pytest.raises(si.InvalidInputError):
            si.compare_with_reference(np.zeros((2, 2), bool), np.zeros((2, 3), np.uint8))

    def test_png_roundtrip(self):
        from PIL import Image
        arr = np.random.default_rng(1).integers(0, 255, (20, 30, 3), dtype=np.uint8)
        back = np.array(Image.open(io.BytesIO(base64.b64decode(si.encode_png_base64(arr)))))
        np.testing.assert_array_equal(back, arr)

    def test_downsample_keeps_mask_binary_and_aspect(self):
        m = np.zeros((2000, 1000), bool); m[:1000] = True
        small = si.downsample_for_display(m, 512)
        assert small.dtype == bool and small.shape == (512, 256)
        assert small[:250].all() and not small[260:].any()


# ------------------------------------------------------------
# Milestone 5: validated SWIR water gate (v2 surface-water models)
# ------------------------------------------------------------
class AllWater(torch.nn.Module):
    """Every pixel positive (P = 1): isolates the effect of the post-processing gate."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = torch.zeros_like(x[:, :1])
        return torch.cat([z - 10, z + 10], dim=1)


V2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
V2_NORM = {"kind": "standardize", "scale": 10000.0, "clip": [0, 1], "mean": [0.1] * 6, "std": [0.05] * 6}


class TestWaterGate:
    def _scene(self):
        img = np.full((64, 64, 6), 1500.0, np.float32)
        img[:, :32, 4] = 400.0     # left half: dark SWIR (water-like)
        img[:, 32:, 4] = 1600.0    # right half: bright SWIR (urban / land under haze)
        return img

    def test_gate_removes_bright_swir_positives(self):
        spec = si.ModelSpec("segmentation_v2", V2_BANDS, "surface_water", V2_NORM,
                            water_gate={"band": "B11", "max_toa_reflectance": 0.10})
        seg = si.FloodSegmenter(model=AllWater(), device="cpu", spec=spec)
        res = seg.predict_array(self._scene(), method="tiled", make_display=False)
        assert res.flood_mask[:, :32].all() and not res.flood_mask[:, 32:].any()
        assert res.stats["n_removed_by_water_gate"] == 64 * 32
        assert res.stats["postprocessing"] == "B11<0.100"
        assert res.stats["flood_coverage_pct"] == pytest.approx(50.0)

    def test_no_gate_keeps_raw_output(self):
        spec = si.ModelSpec("segmentation_v2", V2_BANDS, "surface_water", V2_NORM)
        seg = si.FloodSegmenter(model=AllWater(), device="cpu", spec=spec)
        res = seg.predict_array(self._scene(), method="resize", make_display=False)
        assert res.flood_mask.all() and res.stats["postprocessing"] == "none"

    def test_gate_config_loaded_and_validated(self, tmp_path):
        spec = si.ModelSpec("segmentation_v2", V2_BANDS, "surface_water", V2_NORM)
        assert si.load_water_gate(tmp_path, spec) is None                       # no file -> no gate
        (tmp_path / "postprocessing.json").write_text(json.dumps({"water_gate": {"band": "B11", "max_toa_reflectance": 0.1}}))
        assert si.load_water_gate(tmp_path, spec) == {"band": "B11", "max_toa_reflectance": 0.1}
        assert si.load_water_gate(tmp_path, si.V1_SPEC) is None                 # never applied to the RGB flood model
        (tmp_path / "postprocessing.json").write_text(json.dumps({"water_gate": {"band": "B99", "max_toa_reflectance": 0.1}}))
        with pytest.raises(si.ModelLoadError):
            si.load_water_gate(tmp_path, spec)

    def test_gate_only_applies_to_scenes_hazier_than_training(self):
        spec = si.ModelSpec("segmentation_v2", V2_BANDS, "surface_water", V2_NORM,
                            water_gate={"band": "B11", "max_toa_reflectance": 0.10},
                            haze_check={"band": "B02", "percentile": 5, "max_toa_reflectance": 0.14})
        seg = si.FloodSegmenter(model=AllWater(), device="cpu", spec=spec)
        clear = self._scene()
        clear[..., 0] = 800.0                                   # blue 0.08: inside the training range
        r = seg.predict_array(clear, method="tiled", make_display=False)
        assert r.flood_mask.all() and r.stats["hazy"] is False and r.stats["water_gate_applied"] is False
        assert r.stats["haze_index"] == pytest.approx(0.08)
        hazy = self._scene()
        hazy[..., 0] = 1900.0                                   # blue 0.19: hazier than any training scene
        r = seg.predict_array(hazy, method="tiled", make_display=False)
        assert r.stats["hazy"] is True and r.stats["water_gate_applied"] is True
        assert not r.flood_mask[:, 32:].any() and r.flood_mask[:, :32].all()
        assert any("Hazy image" in w for w in r.warnings)
        assert spec.postprocessing_id == "B11<0.100 if B02p5>0.1400"

    def test_served_v2_checkpoint_loads_safely_with_gate(self):
        ck = PROJECT_ROOT / "models" / "flood" / "segmentation_v2" / "best_seg_model.pt"
        if not ck.is_file():
            pytest.skip("v2 checkpoint not installed")
        torch.load(ck, map_location="cpu", weights_only=True)                    # no pickled objects
        seg = si.FloodSegmenter(ck, device="cpu")
        assert seg.spec.version == "segmentation_v2" and seg.spec.bands == V2_BANDS
        if (ck.parent / "postprocessing.json").is_file():
            assert seg.spec.water_gate == {"band": "B11", "max_toa_reflectance": 0.1}
