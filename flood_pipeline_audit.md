# Flood Prediction Pipeline — Phase A Audit Report

> **Status**: DISCOVERY COMPLETE  
> **Scope**: Read-only audit — no files modified  
> **Date**: 2026-10-12

---

## Executive Summary

The audit uncovered **one critical bug** that invalidates the model's training labels, **two moderate bugs**, and **several minor issues**. The critical bug means the model was trained on the wrong labels — it learned to predict "is the image mostly cloud-free?" instead of "is there flooding?" — and **requires retraining** to fix.

---

## 🔴 CRITICAL: GT Mask Band Misinterpretation (Labels Are Wrong)

> [!CAUTION]
> This bug corrupts **every label** in the dataset. The model was trained on wrong labels and must be retrained.

### What the code does

[`flood_dataset.py:150-157`](file:///d:/superiorparab/final%20year%20DL%20project/ml/datasets/flood_dataset.py#L150-L157) — `_read_mask()` reads **Band 1** of the GT mask:

```python
def _read_mask(self, mask_path, s2_path=None, band=1):
    with rasterio.open(mask_path) as src:
        mask = src.read(band, window=window)  # Always reads band 1
```

[`flood_dataset.py:161-198`](file:///d:/superiorparab/final%20year%20DL%20project/ml/datasets/flood_dataset.py#L161-L198) — `_derive_label()` then interprets Band 1 values as:

```python
# Code comment says:
#   0 = land/dry
#   1 = flood water    ← WRONG
#   2 = cloud
```

### What the data actually is

Every metadata JSON in the dataset documents the encoding:

```json
"encoding_values": [
  {"0": "invalid", "1": "clear", "2": "cloud"},   // ← Band 1: cloud/validity mask
  {"0": "invalid", "1": "land",  "2": "water"}    // ← Band 2: water/land mask
]
```

This is confirmed by the [ml4floods source code](https://github.com/spaceml-org/ml4floods) and the [WorldFloods tutorial](https://spaceml-org.github.io/ml4floods/content/prep/exploring_worldfloods.html), which reads Band 2 for the water mask and Band 1 only as a cloud filter:

```python
v1gt = rst_gt.isel({"band": 1})                  # 0-indexed band 1 = rasterio band 2 = water mask
v1gt.values[rst_gt.values[0] == 2] = 3            # band 0 (rasterio band 1) == cloud → mark as cloud
```

### Evidence from data inspection

| Band | Actual Meaning | Sample Values (EMSR273) |
|------|---------------|------------------------|
| **Band 1** (cloud mask) | 0=invalid, 1=clear, 2=cloud | `[1:262144]` — all clear |
| **Band 2** (water mask) | 0=invalid, 1=land, 2=water | `[1:172885, 2:89259]` |

### Impact: labels are catastrophically wrong

The code reads Band 1 and treats value `1` (= "clear sky") as "flood water". Since most pixels are clear (value 1), **almost every image gets labeled as FLOOD**.

Verified across all splits:

| Split | Samples Checked | Labels Wrong | % Wrong |
|-------|----------------|--------------|---------|
| **test** | 18 | **10** | 56% |
| **val** | 16 | **10** | 63% |
| **train** | 20 (of 150) | **18** | 90% |

Example — `EMSR264_08VATOMANDRY_DEL_v2`:
- **Current** (Band 1): `flood_ratio = 0.9997` → label = FLOOD ❌
- **Correct** (Band 2): `water_ratio = 0.0260` → label = NON_FLOOD ✓

### Confirmation from checkpoint

The trained checkpoint confirms this bias:

```
flood_count: 52,  non_flood_count: 2   (96% flood!)
class_weights: [9.21, 0.53]            (tried to compensate with heavy weighting)
```

The training script tried to fix the imbalance with class weighting, but the fundamental problem is that the labels themselves are wrong.

### Fix required

In [`_read_mask()`](file:///d:/superiorparab/final%20year%20DL%20project/ml/datasets/flood_dataset.py#L150-L159):
- Read **Band 2** for the water/land mask (currently reads Band 1)

In [`_derive_label()`](file:///d:/superiorparab/final%20year%20DL%20project/ml/datasets/flood_dataset.py#L161-L198):
- Also read **Band 1** (cloud mask) to exclude cloudy/invalid pixels
- Water pixels = `(band2 == 2)` AND `(band1 == 1)` (clear AND water)
- Valid pixels = `(band1 == 1)` AND `(band2 != 0)` (clear AND not-invalid)

**After fixing labels**: **retrain the model** — the current checkpoint learned the wrong task.

---

## 🟡 MODERATE: Silent Sample Substitution in Dataset `__getitem__`

[`flood_dataset.py:239-246`](file:///d:/superiorparab/final%20year%20DL%20project/ml/datasets/flood_dataset.py#L239-L246)

```python
except Exception as e:
    print(f"WARNING: Error reading {sample['name']}: {e}")
    alt_idx = (idx + 1) % len(self.samples)
    return self.__getitem__(alt_idx)  # Silent substitution!
```

### Problem
When a sample fails to load, the code silently returns a **different sample** instead of raising an error. This:
1. **Duplicates samples** in training — the substitute sample appears twice per epoch
2. **Hides data errors** — corrupted files are never surfaced
3. **Skews evaluation** — `evaluate_existing_checkpoint.py` was already fixed to bypass this (it calls `_read_mask` and `_derive_label` directly), but `check_labels.py` and `smoke_test.py` still use `ds[i]`

### Fix
Replace the silent fallback with an explicit error raise, or at minimum log to a file and return a sentinel value that the DataLoader can filter.

---

## 🟡 MODERATE: `np.count_nonzero` Used as Data-Quality Check in Sentinel Service

[`sentinel_service.py:248-250`](file:///d:/superiorparab/final%20year%20DL%20project/backend/app/services/sentinel_service.py#L248-L250)

```python
non_zero_ratio = np.count_nonzero(image) / image.size
if non_zero_ratio < 0.1:
    logger.warning(f"Image is {(1 - non_zero_ratio) * 100:.0f}% empty/cloudy")
```

### Problem
This check conflates "zero-valued pixels" with "empty/cloudy". Sentinel-2 surface reflectance can legitimately be zero (e.g., deep water in SWIR bands). The warning message is misleading.

The same check is used as a hard gate in [`flood_routes.py:269-274`](file:///d:/superiorparab/final%20year%20DL%20project/backend/app/api/flood_routes.py#L269-L274):
```python
if metadata.get("non_zero_pct", 0) < 10:
    raise HTTPException(status_code=422, detail="...cloudy or empty...")
```

### Note
Per user instructions: `np.count_nonzero` should NOT be used as a cloud detector. The mosaicking with `leastCC` already handles cloud avoidance. This check should either be removed or changed to test for the no-data sentinel value (0 across ALL 6 bands simultaneously), not individual zero values.

---

## ✅ VERIFIED CORRECT: Band Selection for RGB

### WorldFloods S2 TIFFs (15 bands)

The 15-band S2 TIFFs follow the standard ml4floods `BANDS_S2` ordering:

```python
BANDS_S2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
#            1     2     3     4     5     6     7     8     9      10    11     12     13
# Bands 14 (all zeros = padding) and 15 (cloud probability, range 0-61) are extras.
```

Training uses `rgb_bands=(3, 2, 1)` (0-indexed) → reads rasterio bands 4, 3, 2 = **B4 (Red), B3 (Green), B2 (Blue)** ✅

### Copernicus Live Data (6 bands)

Evalscript returns `[B02, B03, B04, B08, B11, B12]` (6 bands).

Inference uses `rgb_bands=(2, 1, 0)` (0-indexed) → reads rasterio bands 3, 2, 1 = **B04 (Red), B03 (Green), B02 (Blue)** ✅

Both paths produce the same spectral content (B4, B3, B2 = Red, Green, Blue) ✅

### Normalization

Both training and inference divide by 10000.0 and then apply ImageNet normalization:
```python
mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```
This is consistent between training and inference ✅

---

## ✅ VERIFIED CORRECT: Copernicus API / Evalscript

| Aspect | Finding | Status |
|--------|---------|--------|
| Evalscript band order | `[B02, B03, B04, B08, B11, B12]` | ✅ Matches metadata comment |
| Sample type | `UINT16` / `"DN"` units | ✅ Correct for L2A reflectance × 10000 |
| Mosaic strategy | `leastCC` (least cloud cover) | ✅ Better than in-evalscript masking |
| No cloud zeroing | Correct — zeroed pixels look like water to the model | ✅ |
| CRS | `EPSG:4326` for bbox | ✅ |
| RGB preview evalscript | Returns `[B04, B03, B02]` | ✅ True color |

---

## 🔵 MINOR Issues

### 1. Windowed Read Misalignment Between S2 and GT

[`flood_dataset.py:106-123`](file:///d:/superiorparab/final%20year%20DL%20project/ml/datasets/flood_dataset.py#L106-L123) — `_get_window()` computes the window from `src` (each file independently). The S2 image and GT mask may have different dimensions (the S2 has been confirmed to match GT in tested samples, but this is not guaranteed). The window should be computed once from the S2 file and reused for the GT/PW reads of the same spatial extent, after verifying CRS/transform alignment.

### 2. Permanent Water Mask Interpretation

The PW mask values observed: `[0:65068, 1:43, 2:407, 3:18]`. The code uses `pw_mask == 3` to identify permanent water. This appears reasonable for the JRC Global Surface Water seasonality layer but is undocumented in the codebase. After fixing the GT band issue, the PW exclusion logic should be re-verified against the correct water pixels (Band 2).

### 3. `_read_mask` Ignores `s2_path` Parameter

[`flood_dataset.py:150`](file:///d:/superiorparab/final%20year%20DL%20project/ml/datasets/flood_dataset.py#L150) — The `s2_path` parameter is accepted but never used. This was likely intended for spatial alignment (computing a shared window) but was never implemented.

### 4. Checkpoint Validation Metrics Are Unreliable

The checkpoint reports `val_accuracy: 0.981` and `f1_score: 0.990`, but these were computed against the **wrong labels** (96% flood). With correct labels, the true performance is unknown and likely much worse, since the model learned "is the image cloud-free?" rather than "is there flooding?"

---

## Summary of Required Actions

| Priority | Issue | Action |
|----------|-------|--------|
| 🔴 **Critical** | GT mask reads Band 1 (cloud mask) instead of Band 2 (water mask) | Fix `_read_mask` + `_derive_label` → **retrain model** |
| 🟡 **Moderate** | Silent sample substitution | Replace with explicit error handling |
| 🟡 **Moderate** | `np.count_nonzero` misused as cloud check | Remove or replace with proper no-data check |
| 🔵 **Minor** | Window alignment | Compute window once from S2, reuse for GT/PW |
| 🔵 **Minor** | Unused `s2_path` param | Remove or implement alignment |
| 🔵 **Minor** | PW mask docs | Document JRC seasonality values |

> [!IMPORTANT]
> **The single most impactful fix is correcting the GT band read.** Everything else is secondary. After fixing the labels, the model must be retrained from scratch — the current checkpoint is fundamentally trained on the wrong task.
