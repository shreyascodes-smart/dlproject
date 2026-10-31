# Milestone 4 — Product Simplification, Risk Integrity & Release Readiness

> Date: 2026-10-26
> Status: implemented and verified against the running backend. Nothing committed or pushed.
> No retraining. Checkpoints, raw data, masks, manifest and Phase H reports unchanged (sizes/dates checked).

## 1. New normal-user workflow

1. Open the app → **Assess flood risk** panel (left) and a full-height map (right, ~66 % of a 1440 × 900 screen).
2. Search a place (OpenStreetMap Nominatim, submitted search only), click the map, or **Use my location**. The 5 × 5 km area is drawn on the map.
3. **Assess flood risk** → the backend finds the **most recent Sentinel-2 acquisition** (≤ 30 % tile cloud, last 45 days; else the least cloudy) and analyses that single image.
4. The result card shows, in plain language:
   - **Source label:** "Live satellite imagery" or "Archived demonstration sample", plus "Preliminary model estimate".
   - **One status** (see §3) with an explanation.
   - **Flood-like water found** as % of analysed land area, marked "Not a probability".
   - **Image date and age** (e.g. "18 Oct 2026 · 8 days old"), with a warning when older than 14 days.
   - **Uncertainty notes** and **What you can do**.
   - **Secondary actions:** Refresh, Change location, Developer / research details.
5. The map shows the georeferenced satellite image and flood-like overlay, with toggles; water set aside as known lakes, rivers and sea is shown in grey.
6. **Sample assessments** (secondary tab) lists archived past flood events; each is clearly labelled archived and "at its own location".
7. **Local weather & air quality** is a collapsed, lazily loaded disclosure labelled "context only — not used by the flood model".
8. **Light / dark theme** switch in the header. It follows the OS until chosen, is remembered, and is applied before first paint.

## 2. Moved to Developer / research details (dialog, opened on purpose)

Raw prediction, EMS reference, error view and compare slider; zoom and full screen; IoU, Dice, precision and recall with the confusion bar; Tiled/Resize method switch; validation/test split and event IDs; model, epoch and device; tiles, cache source and timings; assessment-rule internals (thresholds, policy version, raw vs outside-water %, water reference source); imagery details (acquisition time, tile cloud cover, cloud-mask source); quality warnings and model limitations. Nothing was deleted.

## 3. Risk labels — what they mean

Segmentation coverage is **not** a probability or a validated risk rating. The levels describe how much flood-like water the model marks outside known water bodies, compared with how much it marks **by mistake** on validation scenes that have no mapped flooding:

| Level | Rule (tiled / resize) | Wording |
|---|---|---|
| `unable` | < 25 % of the image usable, or no valid pixels | Unable to assess |
| `uncertain_water` | > 50 % of usable pixels are known permanent water / sea | Uncertain — permanent water may be misclassified |
| `none_detected` | < 1 % flood-like on land | No significant flooding detected (+ "does not mean the area is safe") |
| `inconclusive` | 1 % – **28.39 %** / **17.9 %** | Some flood-like water — inconclusive |
| `flood_like` | above the ceiling | Extensive flood-like water detected (unconfirmed) |

- **Ceilings** = maximum flood-like share predicted on the 14 validation scenes without EMS-mapped flood (`ml/evaluation/assessment_calibration.py` → `data/reports/kiro_assessment_policy_val_2026-10-26.json`). The 1 %, 25 % and 50 % values are documented design choices, not fitted.
- **Held-out test:** one check with the frozen policy → `data/reports/kiro_assessment_policy_test_check_2026-10-26.json`. No threshold was changed afterwards.
  - Tiled: the 7 test scenes without mapped flood → 6 inconclusive, 1 unable (no false "flood_like").
  - Tiled: the 15 test scenes with mapped flood → 13 inconclusive, 2 unable.
  - In other words, the model **cannot reliably confirm flooding on unseen events**. "Extensive" only triggered on validation scene EMSR342 (and one validation scene without mapped flood before the ceiling was rounded up to include it).
- No "Low/Moderate/High risk", no "% chance", and no "Safe" anywhere in the normal UI; frontend tests check this.

## 4. Permanent-water false positive — **partially mitigated**

**Data used:** JRC Global Surface Water v1.4 (Pekel et al., 2016; free under the Copernicus Programme; attribution "Source: EC JRC/Google").
- Archived scenes: the JRC layer already in WorldFloods (`PERMANENTWATERJRC`, class 3 = permanent), on the exact image grid.
- Live scenes: the GSW `seasonality` tiles (30 m) read over HTTPS. 12 months = permanent water; value 255 = outside dataset coverage, i.e. open sea.

**What changed:**
- **Raw prediction preserved:** the raw model prediction is unchanged and still reported (`prediction.flood_coverage_pct`, raw layer in developer details).
- **Assessment and overlay:** flood-like pixels on known water are **set aside** from the assessment and the map overlay, and shown separately in grey.
- **Water-dominated areas:** these get `uncertain_water` instead of a flood result.
- **Live imagery source:** now Sentinel-2 **Level-1C** (the training product level) from **one dated acquisition**, instead of an L2A 60-day mosaic.
- **Live cloud mask:** clouds and shadows are masked with the Sen2Cor scene classification (L2A SCL).

**Verified:**
- **Live point that gave 75.7 % in Milestone 3** (19.2535 N, 72.4662 E), one real Copernicus request:
  - Acquisition 18 Oct 2026, 0 % tile cloud; GSW marks 100 % open sea.
  - Raw coverage is now 0.35 %, because L1C with a cloud mask is used instead of the L2A mosaic.
  - Result: **"Uncertain — permanent water may be misclassified"**.
- **Regression tests:**
  - synthetic all-water sea scene → `uncertain_water`, raw 100 % preserved;
  - reference unavailable → explicit warning;
  - real GSW check that the Mumbai sea point is > 95 % sea and Mumbai city < 20 %.
- **Effect on accuracy:** pixel IoU when excluding known water — validation 0.2424 → 0.2520, test 0.0299 → 0.0392 (tiled, full resolution).

**Remaining limitations (not solved):**
- The model still marks the sea as flood if the reference is unavailable; this is disclosed in the result.
- JRC is 30 m and ~2021, so new reservoirs, river migration and coastline change are not captured.
- **Seasonal water** (1–11 months) is deliberately not excluded, because it can be genuine flooding.
- Flooding **over** permanent water areas (for example, a lake expanding) is set aside and can be missed.
- The model's false alarms on land (snow, haze, shadows, dark soil) are unaffected.
- A real fix needs retraining with permanent-water context or multispectral input, so it is a separate research task.

## 5. Credentials and live requests

- Credentials stay in `backend/.env`. A scan found no `.env` values in `frontend/src`, `frontend/dist`, `e2e`, `docs`, `data/reports` or `data/cache`, and no CDSE references in frontend code.
- **Error messages hardened:** exception text is no longer returned to clients in live routes or `/satellite-status`. The token error log no longer prints the response body, and Process/Catalog API logs print only the API's own error message. A test asserts that a fake secret in an exception never reaches the response.
- **Live requests this milestone:** 3 Copernicus calls (2 failed with HTTP 406 before the catalog `Accept` header fix, 1 success). Each made a token request plus catalog/process requests to `*.dataspace.copernicus.eu` containing only the bounding box, time range and evalscript.
- **Other outbound traffic:**
  - GSW tiles were read from `storage.googleapis.com` (bounding box only).
  - Browser tests used fixtures for live mode, so **no** Copernicus calls came from tests.
  - Browser tests did make real requests to Nominatim, Open-Meteo and OSM tiles.

## 6. Git hygiene

`frontend/dist/` was tracked (5 files). I ran `git rm -r --cached frontend/dist`: the deletions are **staged**, the local files are kept, and `frontend/.gitignore` (`dist`) now matches them. Nothing else was staged.

## 7. Verification (commands actually run)

| Check | Result |
|---|---|
| `pytest backend/tests ml/tests` | **200 passed** (52 backend incl. 1 live-network GSW test, 148 ML) |
| `npm test` (Vitest) | **29 passed** |
| `npm run lint` (oxlint, now with `no-undef`/`jsx-no-undef`) | clean |
| `npm run build` | ok (443 kB JS, 136 kB gzip) |
| `npm audit` | 0 vulnerabilities |
| `node e2e/smoke.mjs all` vs running backend | **Chromium 30/30, Firefox 29/29, WebKit 29/29** |
| axe-core 4.14 (WCAG 2.0/2.1/2.2 A+AA), 5 screens × 3 engines, dark + light | **0 violations** (one light-mode AQI contrast issue found and fixed) |

**Smoke test covers:**
- Default screen has no metrics; primary action disabled until a location is chosen.
- Map share of the screen; skip link and the `/` shortcut.
- Real place search and selection with the keyboard.
- Area drawn on the map; loading state.
- Live result labels, date and age (live response is a fixture).
- Location change clears the old result; sea → uncertain state.
- Real sample inference with georeferenced overlays and layer toggle; no split jargon in the normal view.
- Developer details: IoU 0.432 → resize 0.266 (real inference); Escape closes the dialog.
- Light theme, persisting across reload; real weather and AQI (Chromium).
- 1024 px and 390 px with no overflow; offline state; no console errors or unexpected failed requests.

Screenshots: `docs/figures/ui_m4/` (~15 MB).

**Not verified:**
- **Safari:** WebKit was tested via Playwright's Windows WebKit build, which is not Safari.
- **Devices and assistive tech:** no physical phone or tablet, no real touch input, and no screen-reader session (NVDA, VoiceOver).
- **Manual keyboard pass:** beyond the automated keyboard steps, there was no manual keyboard-only walkthrough.
- **WebKit skip link:** WebKit skips links in the Tab order by default (a Safari setting), so the skip link is not the first stop there.

## 8. Files changed

**Backend:**
- `backend/app/services/water_reference.py` (new)
- `backend/app/services/seg_service.py`: assessment, water context, georeferenced map layers, freshness, cache v2
- `backend/app/services/sentinel_service.py`: `fetch_scene`, L1C + SCL, acquisition picking, safe logging
- `backend/app/api/flood_routes.py`: new live flow, safe errors

**Backend tests:**
- `backend/tests/test_assessment_rules.py` (new)
- `backend/tests/test_flood_api.py`: updated live tests, water and map tests

**ML:**
- `ml/inference/assessment.py` (new)
- `ml/evaluation/assessment_calibration.py` (new)

**Frontend:**
- `frontend/src/App.jsx`, `index.css`, `components/ui.jsx`
- New: `AssessMap`, `LocationSearch`, `ResultCard`, `SampleList`, `LocalConditions`, `DeveloperDetails`; `hooks/useTheme.js`; `lib/assessment.js`
- Updated: `lib/conditions.js` (place search), `lib/map.js` (`areaBounds`)
- Removed (superseded): `FloodWorkspace`, `ScenePicker`, `Conditions`, `MapDialog`, `MapView`
- Tests: `App.test.jsx`, `lib/lib.test.js`, `test/fixtures.js`
- Tooling: `e2e/smoke.mjs`, `index.html` (theme pre-paint), `.oxlintrc.json`
- Dependencies: `package.json` / `package-lock.json` (dev: `axe-core` 4.14.0)

**Reports:**
- `data/reports/kiro_assessment_policy_val_2026-10-26.json`
- `data/reports/kiro_assessment_policy_test_check_2026-10-26.json`
- this report
- The server result cache was rebuilt (`data/cache/seg_api`, git-ignored).

## 9. Blockers before a real-world pilot

1. **Model accuracy on unseen events is too low** (test IoU ~0.03). It cannot confirm floods; this needs retraining (more bands, permanent-water context, more events) and independent validation.
2. **Level thresholds** come from 24 validation scenes (14 without mapped flood). They need calibration and validation on many more events, including truly flood-free scenes.
3. **Optical imagery only:** clouds during floods are common. A pilot needs SAR (Sentinel-1) for cloud-penetrating observations.
4. **Imagery timeliness:** revisit time plus processing delay means results can be days old; the age is shown, but there is no near-real-time path.
5. **Operations:** no authentication or rate limiting on the API; one GPU process; Nominatim/OSM tile usage limits; no monitoring; credentials in a local `.env`.
6. **Testing gaps:** real Safari, real devices, a screen-reader audit and user testing with non-technical users.
7. **Governance:** it needs alignment with official warning services and clear liability wording before any public use.
