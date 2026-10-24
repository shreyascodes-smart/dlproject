# Phase E — Pre-Training Audit Report

> **Verdict: READY WITH CONDITIONS**  
> **Date**: 2026-10-24  
> **Tests**: 38/38 passed  
> **Files modified**: None (read-only audit)

---

## 1. Executive Summary

The segmentation dataset is **correctly prepared** — all masks are verified, aligned, and properly encoded. Three components must be created before training begins:

| Blocker | What's Needed | Difficulty |
|---------|---------------|------------|
| Segmentation dataset loader | `ml/datasets/flood_seg_dataset.py` | Medium |
| Segmentation training script | `ml/training/train_flood_seg.py` | Medium |
| `segmentation_models_pytorch` | `pip install segmentation-models-pytorch` | Trivial |

No data or label problems were found. No corrections are needed.

---

## 2. Verified Sample, Event, and Pixel Counts

### Sample Counts (verified against `phase_d_label_status.json`)

| Status | Count | Verified |
|--------|-------|----------|
| Positive (verified flood polygons) | 100 | PASS |
| Negative candidate (surveyed, no flood) | 53 | PASS |
| Ambiguous (excluded from training) | 30 | PASS |
| Invalid | 0 | PASS |
| **Total** | **183** | PASS |

### Event Count Change: Phase C (45) to Phase D (39)

Phase C had **45 events** from all 183 samples. Phase D used only the 153 usable samples (positive + negative), yielding **39 events**. The actual total from all 183 samples is **46 events** (the label-status file uses slightly different event-ID extraction than Phase C).

**7 events are exclusive to ambiguous samples** — every sample from these events was classified as ambiguous due to using ML-detected water or Flood trace only:

| Excluded Event | Samples | Reason |
|---------------|---------|--------|
| EMSR322 | 1 | ML water (WF2 model), no Flooded area |
| EMSR323 | 2 | Flood trace only, no Flooded area |
| GLOFIMR01042016 | 3 | w_class=water from source=flood, not Flooded area |
| GLOFIMR08162016 | 1 | ML water (WF2 model) |
| GLOFIMR10132016 | 2 | w_class=water from source=flood |
| UNOSATRS220161008 | 1 | ML water (WF2 model) |
| UNOSATST120161014 | 1 | ML water (WF2 model) |

No events disappeared unexpectedly. All 39 usable events are present in the manifest.

### Pixel Counts (recalculated from actual mask TIFFs)

| Split | Flood px | Non-flood px | Supervised px | Flood % | Matches Report |
|-------|----------|-------------|---------------|---------|----------------|
| **train** | 18,445,231 | 945,554,015 | 963,999,246 | 1.9134% | exact match |
| **val** | 17,903,524 | 124,576,349 | 142,479,873 | 12.5657% | exact match |
| **test** | 722,271 | 63,023,082 | 63,745,353 | 1.1331% | exact match |

All counts match the manifest exactly — zero discrepancies.

---

## 3. Label-Provenance and CRS Findings

### CRS Verification

| Case | Count | Handling |
|------|-------|---------|
| GeoJSON CRS matches S2 CRS | **153** | Rasterized correctly |
| GeoJSON CRS missing (assumed match) | **0** | N/A |
| GeoJSON CRS mismatch (rejected) | **0** | N/A |

**All 153 GeoJSONs** had explicit CRS fields that matched their corresponding S2 raster CRS. No CRS assumption was needed.

### Label Provenance

The rasterization script (prepare_flood_segmentation.py line 95) burns **only** w_class=="Flooded area" features. Verified by code inspection:

- w_class == "Flooded area": burned as FLOOD (value 2)
- w_class == "Flood trace": NOT burned
- w_class == "water": NOT burned (ML-detected, generic)
- w_class == "area_of_interest": NOT burned
- Hydrographic features: NOT burned

**No non-flood feature classes are silently burned as flood labels.** PASS

### CRS Transform Note

The script does not transform GeoJSON coordinates into the S2 CRS — it passes GeoJSON geometries directly to rasterio.features.rasterize() using the S2 transform. This is correct because all 153 GeoJSONs already share the same CRS as their S2 rasters (verified above). The script correctly rejects CRS mismatches rather than silently rasterizing in the wrong coordinate space.

### Negative Candidate Audit

Negative candidates are samples where Copernicus EMS surveyed the area during a flood event but found only pre-existing hydrographic features. This is a defensible but conservative negative label: EMS experts mapped the area and did not identify active flood extent. The risk is that EMS mapping may have incomplete coverage of the AOI, so some non-flood labels might be unmapped flood areas. However, this is the best available evidence.

---

## 4. Train/Validation/Test Leakage Checks

### Event Overlap: Zero

| Check | Result |
|-------|--------|
| train intersection val | empty |
| train intersection test | empty |
| val intersection test | empty |

### Split Composition

| Split | Events | Samples | Positive | Negative |
|-------|--------|---------|----------|----------|
| train | 27 | 107 | 75 | 32 |
| val | 6 | 24 | 10 | 14 |
| test | 6 | 22 | 15 | 7 |

### Duplicate Check

- No duplicate sample names across splits: PASS
- 30 AOI groups have multiple monitoring versions (MONIT01, MONIT02, etc.) — all kept within the same event and same split: PASS

---

## 5. Data-Pipeline Findings and Missing Components

### What Exists

| Component | Status | File |
|-----------|--------|------|
| Classification dataset loader | EXISTS | ml/datasets/flood_dataset.py |
| Classification model | EXISTS | ml/models/flood_classifier.py |
| Rasterization script | EXISTS | ml/datasets/prepare_flood_segmentation.py |
| Segmentation masks | 306 TIFFs | data/processed/flood/segmentation_v1/masks/ |
| Split manifests | EXISTS | seg_split_manifest.json |
| Tests | 38 passing | ml/tests/ |

### What Must Be Created

| Component | Notes |
|-----------|-------|
| **Segmentation dataset loader** (flood_seg_dataset.py) | Read S2 RGB bands, seg mask (nearest-neighbor), validity mask |
| **Segmentation training script** (train_flood_seg.py) | Training loop, optimizer, loss, logging |
| **segmentation_models_pytorch** package | pip install segmentation-models-pytorch |
| **albumentations** package | pip install albumentations |

### Critical Pipeline Requirements

1. **S2 RGB bands**: Existing classifier uses rgb_bands=(3,2,1) = rasterio bands 4,3,2 = Sentinel-2 B4(Red), B3(Green), B2(Blue). Segmentation must use same convention.

2. **Normalization**: Divide by 10,000, clip to [0,1], then apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for pretrained encoder.

3. **Mask resizing**: Segmentation masks MUST use nearest-neighbor interpolation, never bilinear. Bilinear creates fractional class values.

4. **Joint augmentation**: Geometric transforms must apply identical parameters to image, seg mask, and validity mask. albumentations handles this correctly.

5. **Loss masking**: Pixels where seg_mask == 0 (IGNORE) must be excluded from loss computation via ignore_index=0 in CrossEntropyLoss or manual masking.

6. **Model output**: 2-channel logits per pixel (non-flood, flood) with ignore_index=0 in loss, OR single-channel binary logits with masked BCE.

7. **No augmentation on val/test**: Only deterministic transforms (resize, normalize).

---

## 6. Class-Imbalance and Metric Recommendations

### Verified Class Weights (from actual training masks)

| Metric | Value |
|--------|-------|
| Train flood pixels | 18,445,231 |
| Train non-flood pixels | 945,554,015 |
| Train supervised total | 963,999,246 |
| **Train flood ratio** | **1.9134%** |

### Inverse-Frequency Weights

Using convention w_c = N_total / (2 * N_c):

| Class | Weight |
|-------|--------|
| Non-flood (1) | 0.5098 |
| **Flood (2)** | **26.1314** |
| **Ratio** | **51.3:1** |

### Recommended Loss Function

**Weighted CrossEntropy + Dice Loss (combined):**
- loss = 0.5 * weighted_ce_loss + 0.5 * dice_loss
- CrossEntropyLoss(weight=[0.51, 26.13], ignore_index=0)
- Dice computed only on flood class, ignoring class 0

Dice loss directly optimizes overlap metric, complementing pixel-wise cross-entropy. The 51:1 imbalance is too severe for unweighted CE alone.

### Evaluation Metrics

| Metric | Why |
|--------|-----|
| **Flood-class IoU** | Primary metric — overlap between predicted and true flood |
| **Flood-class Dice/F1** | Standard segmentation metric |
| **Flood-class Precision** | Fraction of predicted flood that is actual flood |
| **Flood-class Recall** | Fraction of actual flood that is detected |
| **Confusion counts** | TP, FP, FN, TN pixel totals |
| **Per-event aggregation** | Report metrics per test event (6 events) |

**Why pixel accuracy alone is insufficient**: With ~1% flood pixels, a model predicting "non-flood everywhere" achieves ~99% pixel accuracy. This is meaningless for flood detection. IoU and Dice penalize the model for missing the rare flood class.

---

## 7. GPU-Compatible Training Configuration

### Hardware

| Resource | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 4050 Laptop |
| VRAM | 6.4 GB |
| System RAM | 16 GB |
| PyTorch | 2.14.0+cu126 |
| torchvision | 0.29.0+cu126 |

### Recommended Architecture

**U-Net with EfficientNet-B2 encoder (via segmentation_models_pytorch):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architecture | U-Net | Skip connections for fine-grained segmentation |
| Encoder | EfficientNet-B2 | ~9M params, good accuracy/size tradeoff |
| Pretrained | ImageNet | Transfer learning essential for small dataset |
| Input size | 256x256 | Fits in 6 GB VRAM |
| Batch size | 4 | Conservative for 6 GB VRAM |
| Gradient accumulation | 4 steps | Effective batch = 16 |
| Mixed precision | Yes (torch.amp) | Halves VRAM for activations |
| Output classes | 2 (non-flood, flood) | Binary segmentation |
| Optimizer | AdamW, lr=1e-4, wd=1e-4 | Standard for fine-tuning |
| Scheduler | CosineAnnealing, T_max=50 | Smooth decay |
| Epochs | 50-100 | Early stopping on val flood IoU |

### VRAM Estimate

| Component | Estimated |
|-----------|-----------|
| Model weights (fp16) | ~20 MB |
| Activations (256x256, batch=4, fp16) | ~1.5 GB |
| Gradients | ~40 MB |
| Optimizer state | ~80 MB |
| **Total** | **~1.7 GB** |

Comfortable margin within 6.4 GB. Can increase to 384x384 or batch=8 if needed.

### Packages to Install

```
pip install segmentation-models-pytorch albumentations
```

Do NOT reinstall PyTorch, torchvision, or CUDA.

---

## 8. Blockers Summary

| # | Type | Issue | Fix |
|---|------|-------|-----|
| 1 | MISSING | Segmentation dataset loader | Create ml/datasets/flood_seg_dataset.py |
| 2 | MISSING | Segmentation training script | Create ml/training/train_flood_seg.py |
| 3 | MISSING | segmentation_models_pytorch | pip install segmentation-models-pytorch |
| 4 | MISSING | albumentations | pip install albumentations |

**No data or label blockers.** All masks, manifests, and splits are verified and ready.

---

## 9. Files Inspected and Test Results

### Files Inspected (not modified)

- data/reports/phase_d_label_status.json
- data/processed/flood/segmentation_v1/rasterization_results.json
- data/processed/flood/segmentation_v1/seg_split_manifest.json
- seg_train_samples.txt, seg_val_samples.txt, seg_test_samples.txt
- ml/datasets/prepare_flood_segmentation.py (code audit)
- ml/datasets/flood_dataset.py (reference)
- ml/models/flood_classifier.py (reference)
- ml/tests/test_flood_segmentation.py
- ml/tests/test_mask_logic.py
- All 153 seg + validity mask TIFFs (pixel-level verification)
- data/processed/flood/splits_v2/split_manifest.json (Phase C comparison)
- Representative S2 and GeoJSON files (band structure, CRS verification)

### Files Created

- data/reports/phase_e_pretraining_audit.md (this report)
- data/reports/phase_e_audit_findings.json (machine-readable findings)

### Test Results

```
38 passed in 3.22s (0 failed)
```

### Checkpoints Verified

| File | Size | Status |
|------|------|--------|
| models/flood/best_flood_model.pt | 46,049,598 B | Untouched |
| models/flood/final_flood_model.pt | 46,050,227 B | Untouched |
