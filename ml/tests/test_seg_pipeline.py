"""
Unit tests for the flood segmentation pipeline (Phase F).

Covers:
- Stored-to-target label mapping
- Ignore/invalid pixel exclusion from CE, Dice, and metrics
- All-ignore masks and no-flood-pixel batches
- Band order, scaling, normalization
- Consistent geometric transforms (image + mask)
- Deterministic val/test loading
- Manifest integrity and event leakage
- No automatic pretrained weight download
- Safe checkpoint path selection
- Combined loss behavior
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.flood_seg_dataset import (
    FloodSegDataset,
    stored_to_target,
    normalize_s2_rgb,
    compute_class_weights,
    STORED_IGNORE, STORED_NON_FLOOD, STORED_FLOOD,
    TARGET_NON_FLOOD, TARGET_FLOOD, IGNORE_INDEX,
    IMAGENET_MEAN, IMAGENET_STD,
    S2_RGB_BANDS_0IDX,
)
from ml.training.train_flood_seg import (
    MaskedWeightedCELoss,
    MaskedDiceLoss,
    CombinedLoss,
    FloodMetrics,
    DEFAULT_CONFIG,
)


# ============================================================
# Label Mapping Tests
# ============================================================
class TestLabelMapping:
    """Test stored-to-target label conversion."""

    def test_ignore_maps_to_255(self):
        mask = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        target = stored_to_target(mask)
        assert np.all(target == IGNORE_INDEX)

    def test_non_flood_maps_to_0(self):
        mask = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        target = stored_to_target(mask)
        assert np.all(target == TARGET_NON_FLOOD)

    def test_flood_maps_to_1(self):
        mask = np.array([[2, 2], [2, 2]], dtype=np.uint8)
        target = stored_to_target(mask)
        assert np.all(target == TARGET_FLOOD)

    def test_mixed_labels(self):
        mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        target = stored_to_target(mask)
        assert target[0, 0] == IGNORE_INDEX
        assert target[0, 1] == TARGET_NON_FLOOD
        assert target[1, 0] == TARGET_FLOOD
        assert target[1, 1] == IGNORE_INDEX

    def test_output_dtype_uint8(self):
        mask = np.array([[0, 1, 2]], dtype=np.uint8)
        target = stored_to_target(mask)
        assert target.dtype == np.uint8

    def test_target_values_valid_for_2class_ce(self):
        """Target values must be 0, 1, or 255 — never 2."""
        mask = np.array([[0, 1, 2]], dtype=np.uint8)
        target = stored_to_target(mask)
        unique = set(np.unique(target).tolist())
        assert unique.issubset({0, 1, 255}), f"Got unexpected values: {unique}"


# ============================================================
# Loss Function Tests
# ============================================================
class TestMaskedCELoss:
    """Test weighted cross-entropy with ignore masking."""

    def setup_method(self):
        self.weights = torch.tensor([0.5, 26.0])
        self.loss_fn = MaskedWeightedCELoss(self.weights, ignore_index=255)

    def test_all_ignore_returns_zero(self):
        logits = torch.randn(2, 2, 4, 4)
        targets = torch.full((2, 4, 4), 255, dtype=torch.long)
        loss = self.loss_fn(logits, targets)
        assert loss.item() == 0.0

    def test_valid_pixels_produce_nonzero_loss(self):
        logits = torch.randn(1, 2, 4, 4)
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        loss = self.loss_fn(logits, targets)
        assert loss.item() > 0.0

    def test_ignore_pixels_not_counted(self):
        """Loss should be identical whether ignore pixels exist or not."""
        logits = torch.randn(1, 2, 4, 4)
        # Targets: top row valid, bottom row ignored
        targets_partial = torch.zeros(1, 4, 4, dtype=torch.long)
        targets_partial[0, 2:, :] = 255

        targets_full = torch.zeros(1, 2, 4, dtype=torch.long)
        logits_cropped = logits[:, :, :2, :]

        loss_partial = self.loss_fn(logits, targets_partial)
        loss_full = self.loss_fn(logits_cropped, targets_full)
        assert abs(loss_partial.item() - loss_full.item()) < 1e-5

    def test_gradient_flows(self):
        logits = torch.randn(1, 2, 4, 4, requires_grad=True)
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


class TestMaskedDiceLoss:
    """Test Dice loss with ignore masking."""

    def setup_method(self):
        self.loss_fn = MaskedDiceLoss(ignore_index=255)

    def test_all_ignore_returns_zero(self):
        logits = torch.randn(2, 2, 4, 4)
        targets = torch.full((2, 4, 4), 255, dtype=torch.long)
        loss = self.loss_fn(logits, targets)
        assert loss.item() == 0.0

    def test_no_flood_pixels_returns_zero(self):
        """When all valid pixels are non-flood, Dice should return 0 (no NaN)."""
        logits = torch.randn(1, 2, 4, 4)
        # Make model predict non-flood everywhere
        logits[:, TARGET_NON_FLOOD] = 100.0
        logits[:, TARGET_FLOOD] = -100.0
        targets = torch.zeros(1, 4, 4, dtype=torch.long)  # all non-flood
        loss = self.loss_fn(logits, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_perfect_prediction_near_zero_loss(self):
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, TARGET_FLOOD] = 100.0  # Predict flood everywhere
        targets = torch.ones(1, 4, 4, dtype=torch.long)  # All flood
        loss = self.loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_gradient_flows(self):
        logits = torch.randn(1, 2, 4, 4, requires_grad=True)
        targets = torch.ones(1, 4, 4, dtype=torch.long)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


class TestCombinedLoss:
    """Test the combined CE + Dice loss."""

    def test_all_ignore_returns_zero(self):
        weights = torch.tensor([0.5, 26.0])
        loss_fn = CombinedLoss(weights)
        logits = torch.randn(1, 2, 4, 4)
        targets = torch.full((1, 4, 4), 255, dtype=torch.long)
        loss = loss_fn(logits, targets)
        assert loss.item() == 0.0

    def test_returns_finite_for_mixed_targets(self):
        weights = torch.tensor([0.5, 26.0])
        loss_fn = CombinedLoss(weights)
        logits = torch.randn(1, 2, 8, 8)
        targets = torch.zeros(1, 8, 8, dtype=torch.long)
        targets[0, :4, :4] = 1  # Some flood
        targets[0, 6:, 6:] = 255  # Some ignore
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        assert loss.item() > 0


# ============================================================
# Metrics Tests
# ============================================================
class TestFloodMetrics:
    """Test pixel-level flood metrics."""

    def test_perfect_prediction(self):
        metrics = FloodMetrics()
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 1, :2, :] = 100  # Predict flood on top half
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        targets[0, :2, :] = 1  # True flood on top half
        metrics.update(logits, targets)
        m = metrics.compute()
        assert m["flood_iou"] == 1.0
        assert m["flood_dice"] == 1.0

    def test_no_prediction_no_target(self):
        metrics = FloodMetrics()
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0] = 100  # All non-flood
        targets = torch.zeros(1, 4, 4, dtype=torch.long)  # All non-flood
        metrics.update(logits, targets)
        m = metrics.compute()
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_ignore_pixels_excluded(self):
        metrics = FloodMetrics()
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 1] = 100  # Predict flood everywhere
        targets = torch.full((1, 4, 4), 255, dtype=torch.long)
        metrics.update(logits, targets)
        m = metrics.compute()
        assert m["tp"] == 0
        assert m["fp"] == 0

    def test_accumulation(self):
        """Metrics should accumulate across multiple update calls."""
        metrics = FloodMetrics()
        # Batch 1: 1 TP
        logits1 = torch.zeros(1, 2, 1, 1)
        logits1[:, 1] = 100
        targets1 = torch.ones(1, 1, 1, dtype=torch.long)
        metrics.update(logits1, targets1)
        # Batch 2: 1 FP
        logits2 = torch.zeros(1, 2, 1, 1)
        logits2[:, 1] = 100
        targets2 = torch.zeros(1, 1, 1, dtype=torch.long)
        metrics.update(logits2, targets2)

        m = metrics.compute()
        assert m["tp"] == 1
        assert m["fp"] == 1

    def test_reset(self):
        metrics = FloodMetrics()
        logits = torch.zeros(1, 2, 1, 1)
        logits[:, 1] = 100
        targets = torch.ones(1, 1, 1, dtype=torch.long)
        metrics.update(logits, targets)
        metrics.reset()
        m = metrics.compute()
        assert m["tp"] == 0


# ============================================================
# Normalization Tests
# ============================================================
class TestNormalization:
    """Test S2 normalization pipeline."""

    def test_scaling_range(self):
        """DN of 10000 should map to 1.0 before ImageNet norm."""
        image = np.full((2, 2, 3), 10000, dtype=np.float32)
        result = normalize_s2_rgb(image)
        # After DN->1.0 and ImageNet norm: (1.0 - mean) / std
        for c in range(3):
            expected = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            assert abs(result[0, 0, c] - expected) < 1e-5

    def test_zero_dn_normalization(self):
        """DN of 0 should map to (0 - mean) / std."""
        image = np.zeros((2, 2, 3), dtype=np.float32)
        result = normalize_s2_rgb(image)
        for c in range(3):
            expected = -IMAGENET_MEAN[c] / IMAGENET_STD[c]
            assert abs(result[0, 0, c] - expected) < 1e-5

    def test_negative_dn_clipped(self):
        """Negative DN values should be clipped to 0."""
        image = np.full((2, 2, 3), -100, dtype=np.float32)
        result = normalize_s2_rgb(image)
        zero_result = normalize_s2_rgb(np.zeros((2, 2, 3), dtype=np.float32))
        np.testing.assert_array_almost_equal(result, zero_result)

    def test_output_dtype_float32(self):
        image = np.ones((2, 2, 3), dtype=np.float32) * 5000
        result = normalize_s2_rgb(image)
        assert result.dtype == np.float32


# ============================================================
# Band Order Tests
# ============================================================
class TestBandOrder:
    """Verify correct S2 band selection."""

    def test_rgb_bands_are_4_3_2(self):
        """S2 bands 4,3,2 = 0-indexed 3,2,1."""
        assert S2_RGB_BANDS_0IDX == (3, 2, 1)

    def test_rasterio_reads_1indexed(self):
        """Rasterio uses 1-indexed bands, so 0-idx 3 = rasterio band 4."""
        for band_0idx in S2_RGB_BANDS_0IDX:
            rasterio_band = band_0idx + 1
            assert rasterio_band in (2, 3, 4)


# ============================================================
# Manifest Integrity Tests
# ============================================================
class TestManifestIntegrity:
    """Verify split manifest integrity and event leakage."""

    @pytest.fixture
    def manifest(self):
        path = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "seg_split_manifest.json"
        with open(path) as f:
            return json.load(f)

    def test_three_splits_exist(self, manifest):
        assert set(manifest["splits"].keys()) == {"train", "val", "test"}

    def test_sample_counts(self, manifest):
        train = manifest["splits"]["train"]
        val = manifest["splits"]["val"]
        test = manifest["splits"]["test"]
        assert train["n_samples"] == 107
        assert val["n_samples"] == 24
        assert test["n_samples"] == 22

    def test_zero_event_leakage(self, manifest):
        train_events = set(manifest["splits"]["train"]["event_ids"])
        val_events = set(manifest["splits"]["val"]["event_ids"])
        test_events = set(manifest["splits"]["test"]["event_ids"])
        assert train_events & val_events == set(), "Train-val event leakage!"
        assert train_events & test_events == set(), "Train-test event leakage!"
        assert val_events & test_events == set(), "Val-test event leakage!"

    def test_total_events_is_39(self, manifest):
        all_events = set()
        for split in manifest["splits"].values():
            all_events.update(split["event_ids"])
        assert len(all_events) == 39

    def test_mask_values_documented(self, manifest):
        mv = manifest["meta"]["mask_values"]
        assert mv["0"] == "ignore"
        assert mv["1"] == "non_flood"
        assert mv["2"] == "flood"


# ============================================================
# Checkpoint Safety Tests
# ============================================================
class TestCheckpointSafety:
    """Verify checkpoint paths don't overwrite existing models."""

    def test_seg_checkpoint_dir_differs_from_classifier(self):
        seg_dir = Path(DEFAULT_CONFIG["checkpoint_dir"]).resolve()
        cls_dir = (PROJECT_ROOT / "models" / "flood").resolve()
        assert seg_dir != cls_dir, "Segmentation checkpoints would overwrite classifier!"

    def test_seg_checkpoint_is_subdirectory(self):
        seg_dir = Path(DEFAULT_CONFIG["checkpoint_dir"]).resolve()
        assert "segmentation_v1" in str(seg_dir)

    def test_existing_classifier_files_not_in_seg_dir(self):
        seg_dir = Path(DEFAULT_CONFIG["checkpoint_dir"])
        for name in ["best_flood_model.pt", "final_flood_model.pt"]:
            assert not (seg_dir / name).exists(), f"Would overwrite {name}!"


# ============================================================
# Pretrained Weight Safety Test
# ============================================================
class TestPretrainedWeightSafety:
    """Verify offline mode prevents automatic downloads."""

    def test_hf_hub_offline_set(self):
        """The training script sets HF_HUB_OFFLINE=1."""
        from ml.training.train_flood_seg import create_segmentation_model
        import inspect
        source = inspect.getsource(create_segmentation_model)
        assert 'HF_HUB_OFFLINE' in source
        assert 'TRANSFORMERS_OFFLINE' in source

    def test_no_silent_fallback_to_random_init(self):
        """create_segmentation_model must NOT silently fall back to random weights."""
        from ml.training.train_flood_seg import create_segmentation_model
        import inspect
        source = inspect.getsource(create_segmentation_model)
        # Must raise RuntimeError, not silently create with encoder_weights=None
        assert 'raise RuntimeError' in source
        assert 'encoder_weights=None' not in source


# ============================================================
# Geometric Transform Consistency Tests
# ============================================================
class TestGeometricConsistency:
    """Test that augmentations apply identically to image and mask."""

    def test_deterministic_val_transform(self):
        """Validation transform is deterministic (Resize)."""
        from ml.datasets.flood_seg_dataset import get_val_transform
        tf = get_val_transform(64)
        image = np.random.rand(128, 256, 3).astype(np.float32)
        mask = np.zeros((128, 256), dtype=np.uint8)
        mask[30:60, 100:200] = 2  # Flood region

        r1 = tf(image=image, mask=mask)
        r2 = tf(image=image, mask=mask)
        np.testing.assert_array_equal(r1["image"], r2["image"])
        np.testing.assert_array_equal(r1["mask"], r2["mask"])

    def test_augmented_shapes_match(self):
        """After augmentation, image and mask have the same spatial dims."""
        from ml.datasets.flood_seg_dataset import get_train_augmentation
        tf = get_train_augmentation(64)
        image = np.random.rand(128, 128, 3).astype(np.float32)
        mask = np.zeros((128, 128), dtype=np.uint8)

        result = tf(image=image, mask=mask)
        assert result["image"].shape[:2] == result["mask"].shape
        assert result["image"].shape[:2] == (64, 64)

    def test_val_resize_preserves_flood_pixels(self):
        """Resize must preserve the presence of flood pixels (unlike CenterCrop)."""
        from ml.datasets.flood_seg_dataset import get_val_transform
        tf = get_val_transform(64)

        # Place flood pixels in a corner (CenterCrop would miss them)
        image = np.random.rand(256, 256, 3).astype(np.float32)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[0:30, 0:30] = 2  # Flood in top-left corner

        result = tf(image=image, mask=mask)
        assert result["image"].shape[:2] == (64, 64)
        assert result["mask"].shape == (64, 64)
        # Flood pixels must still exist after resize
        assert 2 in result["mask"], "Resize lost flood pixels!"

    def test_val_resize_uses_nearest_for_mask(self):
        """Mask values must remain categorical {0,1,2} after resize (no interpolation artifacts)."""
        from ml.datasets.flood_seg_dataset import get_val_transform
        tf = get_val_transform(64)

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 1  # Non-flood
        mask[80:120, 80:120] = 2  # Flood in center
        image = np.random.rand(256, 256, 3).astype(np.float32)

        result = tf(image=image, mask=mask)
        unique_vals = set(np.unique(result["mask"]).tolist())
        assert unique_vals.issubset({0, 1, 2}), \
            f"Resize created non-categorical mask values: {unique_vals}"


# ============================================================
# Class Weight Tests
# ============================================================
class TestClassWeights:
    """Test class weight computation."""

    def test_weights_from_manifest(self):
        manifest_path = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "seg_split_manifest.json"
        w_nf, w_flood = compute_class_weights(str(manifest_path))
        # From Phase E audit: w_nf=0.5098, w_flood=26.1314
        assert abs(w_nf - 0.5098) < 0.001
        assert abs(w_flood - 26.1314) < 0.01
        assert w_flood > w_nf  # Flood class should be upweighted

    def test_weight_ratio(self):
        manifest_path = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "seg_split_manifest.json"
        w_nf, w_flood = compute_class_weights(str(manifest_path))
        ratio = w_flood / w_nf
        assert abs(ratio - 51.3) < 0.5  # ~51:1 ratio


# ============================================================
# Output Shape Tests
# ============================================================
class TestOutputShapes:
    """Test dataset output shapes using real data (first sample)."""

    @pytest.fixture
    def val_dataset(self):
        return FloodSegDataset(
            manifest_path=str(PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "seg_split_manifest.json"),
            split="val",
            data_root=str(PROJECT_ROOT / "data" / "raw" / "flood" / "worldfloods_v2"),
            masks_dir=str(PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "masks"),
            input_size=256,
            augment=False,
        )

    def test_image_shape(self, val_dataset):
        image, target = val_dataset[0]
        assert image.shape == (3, 256, 256)

    def test_target_shape(self, val_dataset):
        image, target = val_dataset[0]
        assert target.shape == (256, 256)

    def test_image_dtype(self, val_dataset):
        image, target = val_dataset[0]
        assert image.dtype == torch.float32

    def test_target_dtype(self, val_dataset):
        image, target = val_dataset[0]
        assert target.dtype == torch.int64

    def test_target_values_valid(self, val_dataset):
        image, target = val_dataset[0]
        unique = set(target.unique().tolist())
        assert unique.issubset({0, 1, 255}), f"Unexpected target values: {unique}"

    def test_no_target_value_2(self, val_dataset):
        """Stored value 2 must be remapped to 1, never passed through."""
        image, target = val_dataset[0]
        assert 2 not in target.unique().tolist()

    def test_dataset_length(self, val_dataset):
        assert len(val_dataset) == 24
