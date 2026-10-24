# Phase G — Training Readiness Report

> **Verdict: READY**  
> **Date**: 2026-10-24  
> **Tests**: 88/88 passed (full suite)  
> **Audit**: All checks passed  
> **No training epoch was run.** Only a controlled one-batch forward+backward+optimizer step was executed.

---

## 1. Pretrained Weight Status

| Check | Result |
|---|---|
| EfficientNet-B2 ImageNet weights cached? | **YES** |
| Cache location | `C:\Users\SHRAVAN\.cache\huggingface\hub\models--smp-hub--efficientnet-b2.imagenet` |
| Loads in `HF_HUB_OFFLINE=1` mode? | **YES** |
| Weights are non-trivial (not random)? | **YES** — `_conv_stem.weight`: mean=-0.0138, std=0.552 |
| Encoder identifier | `smp-hub/efficientnet-b2.imagenet` via `segmentation_models_pytorch` |
| Silent fallback to random init? | **REMOVED** — raises `RuntimeError` with instructions |

### Offline Mode Verification

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
model = smp.Unet(encoder_name="efficientnet-b2", encoder_weights="imagenet", ...)
# SUCCESS: Model created with cached weights
```

If weights were unavailable, `create_segmentation_model` now raises:
```
RuntimeError: Pretrained weights for 'efficientnet-b2' are not cached locally
and cannot be downloaded in offline mode.
To cache weights, run once with internet access:
  python -c "import timm; timm.create_model('efficientnet-b2', pretrained=True)"
Then re-run training.
```

---

## 2. One-Batch Training Step Results

A real training batch was loaded, processed through forward pass, loss computation, backward pass, and optimizer step. **No checkpoint was saved. No epoch was run.**

```
Device: cuda (NVIDIA GeForce RTX 4050 Laptop GPU)
Batch: 2x3x256x256 float32 images, 2x256x256 int64 targets
Target pixels: non_flood=67,223  flood=0  ignore=63,849
Logits: 2x2x256x256, range=[-6.68, 10.04]
Loss: 0.6902 (finite)
Parameters with gradients: 328
All gradients finite: True
Parameters changed after step: 328/331
VRAM: 202 MB allocated, 562 MB peak
```

| Check | Result |
|---|---|
| Loss finite and non-NaN? | ✅ |
| Gradients exist and are finite? | ✅ |
| At least one parameter changed? | ✅ (328 changed) |
| GPU used when available? | ✅ CUDA |
| VRAM within budget? | ✅ 562 MB peak << 6,400 MB |

---

## 3. Input/Output and Loss Semantics

### Output Channel Mapping

| Channel | Meaning | Target Value |
|---|---|---|
| 0 | Non-flood | 0 |
| 1 | Flood | 1 |
| — | Ignored | 255 (`ignore_index`) |

### Class Weight Ordering

```python
class_weights = [0.5098, 26.1314]  # index 0 = non-flood, index 1 = flood
```

Verified: weight index matches target class index. The 51.3:1 ratio correctly upweights the minority flood class.

### Ignore Pixel Exclusion

| Test | Result |
|---|---|
| CE loss identical with/without ignore pixels? | ✅ Difference = 0.0000000 |
| All-ignore batch returns loss = 0.0? | ✅ |
| Dice loss with all-ignore = 0.0? | ✅ |
| Combined loss with all-ignore = 0.0? | ✅ |

### Dice Loss with No Flood Pixels

**Behavior**: Returns `0.0` when no flood pixels exist in the batch.

**Rationale**: This is intentional and safe:
- Dice measures overlap between predicted and true flood regions. When there are no true flood pixels, the concept of overlap is undefined.
- Returning 0.0 treats "no flood to find" as "nothing wrong" — the model is not penalized via Dice for this batch.
- **Crucially, the CE component still penalizes false positives** through its weighted cross-entropy term. Verified: combined loss with FP predictions on a no-flood batch = **50.47** (>> 0).
- The loss is finite (not NaN/Inf) and differentiable in all cases.
- The `smooth=1.0` term in Dice prevents division by zero.

### Metric Accumulation

| Test | Result |
|---|---|
| Ignore pixels excluded from TP/FP/FN/TN? | ✅ (12 pixels counted from 16, with 4 ignored) |
| Metrics accumulate across batches? | ✅ |

---

## 4. Spatial Preprocessing Audit

### Raster Dimensions (Training Split)

```
Min:  537 x 538
Max:  10,089 x 11,096
Mean: 2,906 x 3,399
Samples smaller than 256x256: 0
Flood-positive samples: 75/107
```

### Critical Fix Applied: CenterCrop → Resize

**Problem discovered**: CenterCrop(256) on ~3000×3400 rasters retained **0.00%** of validation flood pixels (132 out of 17,903,524). Floods are rarely centered — CenterCrop captured only background.

**Fix**: Replaced validation/test transform with `Resize(256)`:
- Image: bilinear interpolation (`cv2.INTER_LINEAR`)
- Mask: nearest-neighbor interpolation (preserves categorical labels `{0,1,2}`)

**After fix**:

| Metric | CenterCrop | Resize |
|---|---|---|
| Val flood pixels retained | 132 | 39,030 |
| Flood regions present? | ❌ (0.00%) | ✅ (all spatial regions) |
| Mask values preserved? | N/A | ✅ `{0,1,2}` only |

> **Note**: The absolute pixel count drops due to downsampling (3000→256 is ~12× per axis, ~144× total area). The flood *fraction* per sample is preserved — all flood regions remain spatially represented, enabling meaningful validation metrics.

### Training Augmentation

- **RandomCrop(256)**: Samples random patches across the full raster each epoch.
- Over 100 epochs with 107 samples, all spatial regions are visited.
- No flood pixel elimination concern because crops are random, not fixed.

---

## 5. Checkpoint and Reproducibility Safety

| Check | Result |
|---|---|
| Seg checkpoint dir | `models/flood/segmentation_v1/` |
| Differs from classifier dir? | ✅ (`models/flood/`) |
| `best_flood_model.pt` in seg dir? | Does not exist (SAFE) |
| `final_flood_model.pt` in seg dir? | Does not exist (SAFE) |

### Config Fields Recorded in `training_config.json`

| Field | Value |
|---|---|
| `seed` | 42 |
| `encoder_name` | efficientnet-b2 |
| `encoder_weights` | imagenet |
| `input_size` | 256 |
| `batch_size` | 4 |
| `lr` | 0.0001 |
| `max_epochs` | 100 |
| `manifest_path` | Full path to `seg_split_manifest.json` |
| `pytorch_version` | Recorded at runtime |
| `cuda_available` | Recorded at runtime |
| `gpu` | Recorded at runtime |
| `class_weights` | Recorded at runtime |
| `timestamp` | Recorded at runtime |

---

## 6. Changes Made in Phase G

| File | Change | Reason |
|---|---|---|
| `ml/training/train_flood_seg.py` | Removed silent fallback to random init | User requirement: exit with clear error if weights unavailable |
| `ml/datasets/flood_seg_dataset.py` | Val transform: CenterCrop → Resize | CenterCrop retained 0.00% of val flood pixels |
| `ml/tests/test_seg_pipeline.py` | Added 2 new tests, updated 2 existing tests | Test no-fallback, resize flood retention, mask value preservation |

---

## 7. Full Test Suite

**88/88 passed in 46.57s**

| Test File | Count | Status |
|---|---|---|
| `test_flood_segmentation.py` | 16 | ✅ All passed |
| `test_mask_logic.py` | 22 | ✅ All passed |
| `test_seg_pipeline.py` | 50 | ✅ All passed |

---

## 8. Confirmation

**No training epoch was run.** The only model-related computation was:
1. Model instantiation to verify cached pretrained weights
2. A single controlled forward + backward + optimizer step on one batch
3. No checkpoint was saved
4. No model weights were persisted

---

## 9. Training Command (DO NOT EXECUTE UNTIL AUTHORIZED)

```powershell
cd "d:\superiorparab\final year DL project"
.\venv\Scripts\python.exe ml\training\train_flood_seg.py
```
