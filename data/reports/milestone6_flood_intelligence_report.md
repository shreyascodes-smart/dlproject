 # Milestone 6 — Flood intelligence: ordinary water vs flood evidence (2026-10-31)

Served model unchanged: `segmentation_v2` (`models/flood/segmentation_v2/best_seg_model.pt`, epoch 4, SHA-256 in
`PROVENANCE.json`). No retraining was done. No commits were made, and no data, checkpoints or earlier reports were modified or deleted.
Credentials are not included.

## 1. Root cause of ordinary-water false positives

The Milestone 5 rule was: flood-like = model water outside JRC **permanent** water. The model itself detects water well
(validation water IoU about 0.77); the false positives came mainly from the **reference and the rule**, not the network:

| Cause | Evidence |
|---|---|
| JRC *seasonal* water (water 1–11 months a year: monsoon floodplains, sandbanks, wetlands, tidal channels) counted as flood-like | Validation: setting seasonal water aside lowers the largest flood-like share on a no-flood scene from 39.8 % to 28.6 % (North Normanton: 22 % of the scene is seasonal water) |
| JRC is 30 m and ~2021: small ponds, lakes, reservoirs, canals and drains are missing or misclassified | Mumbai T Ward (Vihar/Tulsi lakes): OSM maps 11,381 px of lakes/reservoirs that JRC did not mark permanent |
| One image cannot show what is normal for a place | No temporal baseline existed |
| Missing reference could still yield "flood-like detected" | `classify` only skipped the water-dominated check when the reference was missing |
| Outcome ceiling set by a single validation scene | 39.82 % (M5), so most real floods read "inconclusive" |

Haze (M5) and the policy derivation itself were working as designed.

## 2. Changes

**Known-water reference** (`backend/app/services/water_reference.py`, `ml/inference/assessment.py`)
- New codes: `KW_SEASONAL` (JRC seasonal water, OSM wetlands and tidal flats) and `KW_MAPPED` (OSM lakes, ponds, reservoirs,
  rivers, canals, drains). Existing codes are unchanged (0 dry, 1 permanent, 2 sea).
- Archived scenes: WorldFloods `PERMANENTWATERJRC` class 2 is now seasonal (class 3 permanent, as before).
- Live scenes: JRC GSW seasonality 1–11 months → seasonal; OpenStreetMap water polygons/lines rasterised on the exact
  image grid (EPSG:4326, `rasterio.features`, holes cut out, lines 1 px wide). JRC permanent water and sea keep priority.
- Rivers, reservoirs and wetlands are **not** suppressed as places: only water inside their mapped outline or normal
  seasonal extent is set aside. Water spreading onto normally-dry land still counts.
- OSM failures are explicit: reference status `partial`, a visible warning, and degraded references are never cached.
  A missing JRC reference can no longer produce "flood-like detected" (the outcome becomes inconclusive with a reason).
- The Overpass query (19–35 s measured) starts in a background thread before the imagery download, with one mirror.
- `water_context.by_class` reports, for each class, the eligible pixels and the predicted water on them. The raw prediction is unchanged.

**Historical comparison** (`ml/inference/water_change.py`, `sentinel_service.fetch_scene(end=…)`, `flood_routes._fetch_baseline`)
- The same 5 × 5 km request grid, bands and model, using a Sentinel-2 image from the same season about one year earlier (window ending 335 days
  before today's image, 60 days long). Only pixels usable in both images and on normally-dry land are compared.
- Reported: water then/now, new water, share of today's flood-like water already present then, compared area.
- Not used if under 50 % of today's usable pixels are usable in both images, or if the earlier image is hazier than the training range.
- Effect on the outcome is one-directional: a "flood-like" result whose water was mostly (≥ 50 %) present a year earlier
  becomes "inconclusive", and an "inconclusive" result with new water below the 1 % floor becomes "no flood-like patterns". It never creates a
  flood outcome. These two thresholds are design choices; the project has no labelled time series to validate them.
- Toggle: `DI_HISTORICAL_COMPARISON` (default on). Archived demo scenes have no baseline image, so "unavailable" is reported.

**Outcome policy**: re-derived on the frozen validation split with seasonal water set aside
(`data/reports/assessment_policy_segmentation_v2_seasonal_val.json`, now active via `POLICY_FILES`). The Milestone 5
policy file is kept. The demo cache key now includes the reference version.

**Frontend** (minimal): the coverage breakdown shows permanent water/sea, mapped water bodies and seasonally wet land
separately, and a "Compared with a year earlier" section shows the comparison when it is present. No other UI changes were made.

**Evaluation**: `assessment_calibration.py --known-water permanent|permanent+seasonal` (default keeps M5 behaviour).

## 3. Measured results (frozen splits, full resolution, product protocol vs EMS flood maps)

| | Validation tiled IoU (P / R) | Test tiled IoU (P / R) | Validation resize IoU | Test resize IoU | Ceiling tiled / resize |
|---|---|---|---|---|---|
| M5: permanent water set aside | 0.4565 (0.518 / 0.794) | 0.1572 (0.173 / 0.628) | 0.5471 | 0.0513 | 39.82 % / 20.97 % |
| **M6: permanent + seasonal** | **0.4937 (0.598 / 0.739)** | **0.1711 (0.205 / 0.512)** | 0.5597 | 0.0522 | **28.57 % / 14.28 %** |

The choice was made on validation (higher IoU and precision, lower false-alarm ceiling). The test split was evaluated once for
this configuration. It had already been used three times in Milestone 5, so it is not untouched.

Outcome counts, test tiled: 15 scenes with mapped flood went from (12 inconclusive, 1 no-flood, 2 insufficient) to
(10, 3, 2); the 7 scenes without mapped flood stayed at (5, 1, 1). **No test scene reaches "flood-like detected" in either version**,
and two flooded scenes moved to "no flood-like patterns". Recall drops (0.63 → 0.51) because some EMS-mapped flood lies on land that JRC marks
seasonal. This is a real trade-off: fewer ordinary-water false alarms in exchange for some missed floods inside the normal seasonal extent.

Real live checks (Copernicus + JRC + OSM + year-earlier image, 31 Oct 2026):

| Place | M5 | M6 outcome | Flood-like share (raw water) | Notes |
|---|---|---|---|---|
| Mumbai T Ward (Vihar/Tulsi lakes) | Inconclusive (user screenshot) | No flood-like patterns | 0.26 % (4.3 %) | OSM lakes/reservoirs 11,381 px set aside; 70 % of remaining water also present 27 Nov 2025 |
| Patna (Ganges) | No flood-like | No flood-like | 0.00 % | baseline hazy → comparison not used |
| Brahmaputra, Guwahati | Inconclusive (5.9 %) | No flood-like | 0.45 % (29 %) | river, wetland, seasonal sandbanks set aside |
| Sundarbans wetland | Inconclusive (4.5 %) | No flood-like | 2.7 % | OSM timed out (reported as partial); 90 % of water present a year earlier |
| Arabian Sea | Mostly permanent water | Mostly permanent water | — | unchanged |

Live latency rose to 25–65 s per analysis (Overpass plus a second Copernicus download).

## 4. Tests run (actual results)

- `pytest backend/tests ml/tests`: **242 passed**. Includes the new `backend/tests/test_normal_water.py` with 24 synthetic
  regression tests covering: permanent lake, OSM pond/lake/reservoir/wastewater, mapped river and drain lines, river expansion onto dry land →
  flood-like, flooded urban area → flood-like, seasonal wetland and tidal flat, sea-only, missing reference → inconclusive, insufficient
  land, haze preserved, no historical image → no change, OSM failure → partial and not cached, unmapped pond present a year earlier →
  no flood, persistent water withdraws a flood claim, new water keeps it, comparison never creates a flood, cloudy baseline
  ignored, grid mismatch rejected, known water excluded from comparison. These tests check the rules only. They do not measure accuracy.
- Frontend: `oxlint` 0 issues, vitest 46 passed, `vite build` OK. Browser smoke tests were not re-run (UI change was small).

## 5. Not done / limitations

- **Model**: not retrained. The training data has no explicit "ordinary water, no flood" labels, and the evidence points to the
  reference/rule rather than the network. The checkpoint is unchanged.
- **Sentinel-1 SAR**: not integrated. This needs a separate SAR model (VV/VH, speckle filtering, terrain correction) trained on
  SAR flood labels (e.g. Sen1Floods11), a separate `ModelSpec` and a fusion rule. Adding SAR bands to the 6-band optical model would not work.
- The historical rules and the OSM layer cannot be validated on the archived splits, because there are no baseline images and OSM was not
  fetched for those scenes. Their effect is measured only on the five live checks above.
- OSM quality varies by region. Large multipolygons made of unclosed member ways are skipped, and Overpass can time out (Sundarbans).
- GSW occurrence (1984–2021 history) is not used yet.
- The outcome ceiling still comes from one validation scene (24 scenes in total). Test recall for flooded scenes fell.

## 6. Recommended next milestone

1. Add validation scenes without flooding (to lower and stabilise the ceiling), and pre-fetch OSM for the frozen splits so the
   OSM rule can be measured.
2. Build a small labelled live set (dated EMS activations plus a year-earlier image) to validate the historical rules.
3. Build a Sentinel-1 SAR branch for cloudy scenes, as a separate model.
