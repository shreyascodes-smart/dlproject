# Milestone 2 — Backend Integration (U-Net segmentation)

> Date: 2026-10-26
> Status: **code written, syntax-checked, NOT yet executed** (see "Blocker").
> No retraining. Checkpoints, masks, manifest, raw data and Phase H reports untouched.

## What changed

| File | Change |
|---|---|
| `backend/app/services/seg_service.py` | **New.** Demo catalogue, demo/live assessment, result caching, response building. |
| `backend/app/api/flood_routes.py` | **Rewritten.** Uses the U-Net service; legacy classifier no longer imported. |
| `backend/app/main.py` | Title/description/version updated to the segmentation API. |
| `backend/tests/test_flood_api.py` | **New.** 30 API tests (synthetic data + mock model; 1 optional real-model test). |
| `.gitignore` | Added `data/cache/` (result cache). |

`ml/inference/flood_inference.py` and both legacy checkpoints remain on disk, unused.

## API contract

| Endpoint | Purpose |
|---|---|
| `GET /api/flood/model-status` | `status` (`ready`/`not_ready`), model info (U-Net, EfficientNet-B2, epoch 14, device), Phase H test metrics, available methods, `satellite` (`configured`/`not_configured`), `live_mode: "experimental"` |
| `GET /api/flood/demo-samples` | Verified samples: validation first, then held-out test. Each has id, label, event id, split + split note, event type, Sentinel-2 date, centre lat/lng, bbox, size, reference flood %. Also lists excluded samples with reasons. |
| `GET /api/flood/satellite-status` | Copernicus credential check (unchanged behaviour). |
| `POST /api/flood/assess` | Demo: `{"mode":"demo","sample_id":"...","method":"tiled"\|"resize"}`. Live (experimental): `{"mode":"live","latitude":..,"longitude":..}`. |

Assessment response (both modes): `images` (rgb, overlay, and for demo also EMS reference and error view, all JPEG data URLs on the same ≤1024 px grid, plus colour legend), `prediction` (coverage %, definition, pixel counts, eligibility source, threshold note), `reference` (demo: per-sample metrics vs EMS mask with notes; live: `available: false`), `quality` (status + warnings), `imagery` (source, date, location note), `model`, `limitations`, `disclaimer`, `cached`.

Not returned on purpose: any "confidence", risk category, or km² area.

## Key decisions

- **Longitude bug fixed:** latitude −90…90, longitude −180…180.
- **No coordinate arithmetic:** demo mode requires an explicit `sample_id` from the catalogue. Training samples cannot be requested (404). Every response says the archived scene shows the event's own location, not the map selection.
- **Catalogue verification:** a sample is listed only if imagery, reference mask and validity mask exist and share the same dimensions (≥ 4 bands). Location comes from the sample's meta JSON bounding box, or from the raster bounds reprojected to lat/lng.
- **Reference handling:** EMS mask with validity == 0 set to ignore, same as training. Labelled "reference data, not a model output".
- **Live mode (experimental):** reads RGB from the 6-band CDSE order (B04, B03, B02). Warnings state: no ground truth (unvalidated), Level-2A live imagery vs Level-1C training imagery ([WorldFloods dataset card](https://huggingface.co/datasets/tacofoundation/worldfloods)), 60-day least-cloudy mosaic, unknown cloud status.
- **Performance:** endpoints are sync `def`, so FastAPI runs inference in a worker thread; a lock serialises GPU use. Results are cached per (sample, method, checkpoint signature) in memory (16 entries) and on disk under `data/cache/seg_api/` (git-ignored), so a ~20 s scene is computed once.
- **Default method:** `tiled` (from the Milestone 1 validation comparison); `resize` selectable.

## Blocker: command sandbox lost access to the project

After the IDE restarted, Kiro's Windows command sandbox began running commands under a new AppContainer identity (`kiro-sandbox-pool-1`). That identity cannot read many folders, including `venv\Lib\site-packages\torch`, `pygments`, `frontend\node_modules` and `data\raw\...` (95 of 170 package folders denied; no progress after waiting). The folders were created by Python 3.14 with owner-only permissions, which only the previous identity had been granted.

Result: no Python package or Node command can run in agent commands at the moment. Only a syntax check with the base Python (`C:\Python314`) succeeded for all four new/changed backend files.

This does not affect a normal terminal.

## Verification still required

```powershell
cd "D:\superiorparab\final year DL project"
.\venv\Scripts\python.exe -m pytest backend/tests ml/tests -q
```

Expected: all `ml/tests` (148) plus `backend/tests` (30) pass.

Manual check:

```powershell
cd backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
# then open http://localhost:8000/docs
```

## Known gap

The current React UI still expects the old response (`prediction.flood_probability`, `risk`, `gradcam_base64`) and does not send `sample_id`, so the Flood tab will not work until Milestone 3 (frontend). Weather, AQI, map and geocoding are unaffected because they call external APIs directly.
