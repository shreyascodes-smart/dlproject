# Phase F — Implementation Report

> **Status**: Pipeline implemented and tested. Ready for training authorization.  
> **Date**: 2026-10-24  
> **Tests**: 86/86 passed (full suite)  
> **Smoke test**: PASSED (one-batch forward pass on GPU)

---

## 1. Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `ml/datasets/flood_seg_dataset.py` | Segmentation dataset loader | ~230 |
| `ml/training/train_flood_seg.py` | Training script (U-Net + CE+Dice) | ~330 |
| `ml/tests/test_seg_pipeline.py` | 48 unit tests for the pipeline | ~380 |

No existing files were modified. No raw data, checkpoints, frontend, or backend were touched.

---

## 2. Segmentation Dataset Loader

**File**: `ml/datasets/flood_seg_dataset.py`

### Label Mapping

The stored 3-class mask is remapped to 2-class targets for `CrossEntropyLoss`:

| Stored Value | Meaning | Target Value | CE Handling |
|---|---|---|---|
| 0 | ignore (cloud/invalid/outside AOI) | 255 | `ignore_index=255` |
| 1 | supervised non-flood | 0 | Class 0 |
| 2 | verified flood extent | 1 | Class 1 |

This prevents the out-of-range error that would occur if stored value 2 were passed directly to a 2-class loss.

### Data Loading Pipeline

1. Read S2 RGB bands (rasterio bands 4, 3, 2 = Sentinel-2 B4, B3, B2)
2. Read segmentation mask and validity mask
3. Verify dimension alignment across all three files
4. Combine validity into seg mask (invalid pixels become IGNORE)
5. Apply augmentation (train: random crop + flip + rotate; val/test: center crop)
6. Normalize: DN / 10000, clip to [0,1], then ImageNet normalization
7. Convert stored labels to target labels
8. Return `(image: 3xHxW float32, target: HxW int64)`

### Augmentation

- **Train**: `RandomCrop(256) + HorizontalFlip + VerticalFlip + RandomRotate90`
- **Val/Test**: `CenterCrop(256)` — fully deterministic
- All geometric transforms applied identically to image, seg mask, and validity mask via `albumentations`

---

## 3. Training Script

**File**: `ml/training/train_flood_seg.py`

### Architecture

- U-Net with EfficientNet-B2 encoder (10,046,436 parameters)
- Pretrained weights loaded via `timm` (locally cached after first download)
- Sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prevent silent downloads
- Falls back to random initialization if weights are not cached

### Loss Function

**Combined loss** = 0.5 * Weighted CE + 0.5 * Dice

| Component | Details |
|---|---|
| Weighted CE | `CrossEntropyLoss(weight=[0.5098, 26.1314], ignore_index=255)` |
| Dice | Flood-class only, masked, smooth=1.0 |
| All-ignore guard | Both CE and Dice return 0.0 (not NaN) when all pixels are ignored |
| No-flood guard | Dice returns 0.0 when batch has no flood pixels |

**Class weight derivation** (documented in code):
- `w_c = N_supervised / (2 * N_c)` computed from training split manifest
- `w_non_flood = 963,999,246 / (2 * 945,554,015) = 0.5098`
- `w_flood = 963,999,246 / (2 * 18,445,231) = 26.1314`
- Ratio: 51.3:1

### Training Configuration

| Parameter | Value |
|---|---|
| Input size | 256x256 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | CosineAnnealing (T_max=100) |
| Mixed precision | Yes (torch.amp) |
| Early stopping | patience=15 on val flood IoU |
| Seed | 42 (deterministic) |

### Checkpoint Safety

- Saves to `models/flood/segmentation_v1/` (new directory)
- Asserts checkpoint dir != classifier dir
- Cannot overwrite `best_flood_model.pt` or `final_flood_model.pt`
- Saves `training_config.json` alongside checkpoints

### Metrics Logged Per Epoch

Train loss, val loss, flood IoU, flood Dice/F1, flood precision, flood recall, learning rate, epoch time. Confusion counts (TP, FP, FN, TN) available from metrics object.

---

## 4. Dependency Actions

| Package | Version | Action |
|---|---|---|
| segmentation-models-pytorch | 0.5.0 | Installed |
| albumentations | 2.0.8 | Installed |
| timm | 1.0.30 | Installed (dependency of smp) |
| safetensors | 0.8.0 | Installed (dependency of smp) |
| albucore | 0.0.24 | Installed (dependency of albumentations) |
| opencv-python-headless | 5.0.0.93 | Installed (dependency of albumentations) |
| PyTorch | 2.14.0+cu126 | **NOT MODIFIED** |
| torchvision | 0.29.0+cu126 | **NOT MODIFIED** |
| rasterio | existing | **NOT MODIFIED** |

---

## 5. Test Results

### Full Suite: 86/86 passed in 42.97s

| Test File | Tests | Status |
|---|---|---|
| `test_flood_segmentation.py` | 16 | All passed |
| `test_mask_logic.py` | 22 | All passed |
| `test_seg_pipeline.py` | 48 | All passed |

### Phase F Test Coverage (48 tests)

| Category | Tests | What's Verified |
|---|---|---|
| Label mapping | 6 | stored-to-target conversion, dtype, valid values |
| Masked CE loss | 4 | all-ignore=0, valid>0, ignore exclusion, gradients |
| Masked Dice loss | 4 | all-ignore=0, no-flood=0, perfect~0, gradients |
| Combined loss | 2 | all-ignore=0, mixed targets finite |
| Flood metrics | 5 | perfect prediction, no-target, ignore exclusion, accumulation, reset |
| Normalization | 4 | DN scaling, zero, negatives clipped, dtype |
| Band order | 2 | correct S2 band indices |
| Manifest integrity | 5 | splits, counts, event leakage, mask values |
| Checkpoint safety | 3 | dir differs from classifier, subdirectory, no overwrite |
| Pretrained weight safety | 1 | HF_HUB_OFFLINE in source |
| Geometric consistency | 3 | deterministic val, shape match, spatial alignment |
| Class weights | 2 | values match Phase E, ratio correct |
| Output shapes (real data) | 7 | image/target shape, dtype, valid values, no value 2, length |

---

## 6. Smoke Test Results

```
Device: cuda (NVIDIA GeForce RTX 4050 Laptop GPU)
Model: U-Net + EfficientNet-B2 (10,046,436 params)
Input:  torch.Size([2, 3, 256, 256]), float32
Target: torch.Size([2, 256, 256]), int64
Output: torch.Size([2, 2, 256, 256]), float32
Loss:   0.7578 (finite, non-NaN)
VRAM:   44 MB allocated, 87 MB peak

SMOKE TEST PASSED
```

The 87 MB peak VRAM with batch=2 confirms the 6.4 GB GPU has ample headroom for batch=4 training.

---

## 7. Remaining Blockers

**None.** The pipeline is fully implemented, tested, and smoke-tested. All blockers from Phase E have been resolved:

| Phase E Blocker | Status |
|---|---|
| Segmentation dataset loader | CREATED |
| Segmentation training script | CREATED |
| segmentation_models_pytorch | INSTALLED |
| albumentations | INSTALLED |

---

## 8. Command to Launch Training

```bash
cd "d:\superiorparab\final year DL project"
.\venv\Scripts\python.exe ml\training\train_flood_seg.py
```

**Do NOT execute this command until authorized.** The script will:
1. Load 107 train + 24 val samples
2. Create U-Net with EfficientNet-B2 encoder
3. Train with weighted CE + Dice loss
4. Early-stop on val flood IoU (patience=15)
5. Save best checkpoint to `models/flood/segmentation_v1/best_seg_model.pt`
6. Save final checkpoint to `models/flood/segmentation_v1/final_seg_model.pt`

Estimated training time: ~2-4 hours for 100 epochs on RTX 4050.
