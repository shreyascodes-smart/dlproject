# Milestone 2 — Backend Integration: Verification & Completion

> Date: 2026-10-26
> Follows: `milestone2_backend_integration_report_2026-10-26.md` (written while the command sandbox blocked execution).
> Status: **complete and verified.** No retraining; checkpoints, masks, manifest, raw data and Phase H reports unchanged (sizes/dates checked).

## Result

- `pytest backend/tests ml/tests` → **176 passed, 0 failed, 0 skipped** (28 backend + 148 ML), run in a normal (unsandboxed) shell.
- A real Uvicorn server, started the documented way (`app.main:app` from inside `backend/`), served the U-Net model on CUDA with 46 verified demo samples (24 validation, 22 test, 0 excluded).
- All 46 samples × 2 methods are pre-computed in the result cache, so every demo assessment answers in about 0.1 s.

## Changes since the first Milestone 2 report

| File | Change |
|---|---|
| `backend/app/main.py` | Model + catalogue load in a background thread at startup (FastAPI `lifespan`), so the first UI request is fast. Added `http://127.0.0.1:5173` to CORS. |
| `backend/scripts/warm_cache.py` | **New.** Pre-computes demo results into `data/cache/seg_api/` (git-ignored). |
| `backend/app/services/seg_service.py` | AOI label falls back to the sample name when meta lacks `aoi_code`. |
| `backend/tests/test_flood_api.py` | Test for the AOI fallback. Final count: 28 tests (the earlier report said 30; 28 is correct). |

## End-to-end checks on the running server

| Check | Result |
|---|---|
| `GET /api/flood/model-status` | `ready`, U-Net / efficientnet-b2, epoch 14, cuda; reported test IoU 0.0223, Dice 0.0437; 0.2 s after warm-up |
| `GET /api/flood/demo-samples` | 46 samples (val 24, test 22), none excluded; e.g. `EMSR264 · VATOMANDRY`, val, 2018-01-16, centre −19.39, 48.88 |
| Demo, EMSR342 South Normanton, tiled, first run | 200, 35.2 s, coverage 43.48 %, IoU vs EMS reference 0.432, 1004×1024 display images, ~2.4 MB response |
| Same request again | 200, 0.1 s, `cached: true` |
| Same sample, resize | 200, 16.8 s, coverage 23.45 %, IoU 0.266 |
| Demo, EMSR9284 (test split, Finland), tiled | 200, 5.1 s, labelled `test`, IoU 0.012 |
| After server restart | cached result served from disk in 0.1 s |
| CORS from `http://localhost:5173` | `Access-Control-Allow-Origin` returned |
| Demo without `sample_id` | 422 |
| Training sample (`EMSR249_03GALWAY_DEL_v2`) | 404 |
| Live, lat 23.8 / lng 90.4 (Dhaka; longitude > 90) | 200, `experimental: true`, coverage 7.37 %, no reference |
| Live, longitude 200 | 422 |

The IoU values match the Milestone 1 validation comparison for the same samples, confirming the API runs the same inference.

The live check made one real request to Copernicus Data Space using the credentials in `backend/.env` (only the coordinates and date range were sent). Credential values were not read or logged.

## Cache warm-up

`backend/scripts/warm_cache.py --methods tiled resize`: 92 results in 267 s on the RTX 4050, 117.9 MB in `data/cache/seg_api/` (git-ignored). The cache key includes the checkpoint size and modification time, so results are recomputed automatically if the model file ever changes.

## Running the backend

```powershell
cd "D:\superiorparab\final year DL project\backend"
..\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
# API docs: http://localhost:8000/docs
```

Optional, once before the demo (results already cached on this machine):

```powershell
cd "D:\superiorparab\final year DL project"
.\venv\Scripts\python.exe backend\scripts\warm_cache.py
```

## Environment notes

- The Kiro command sandbox is now disabled (`~/.kiro/settings/sandbox.json`), by the user. Agent commands now run with normal user permissions; re-enable it after the deadline if desired. The `scratch/kiro_run.py` launcher and `scratch/kiro_dll/` copies are no longer needed and are git-ignored.
- Nothing was committed.

## Remaining

- Milestone 3: React frontend — still uses the old response format (`flood_probability`, `risk`, Grad-CAM), so the Flood tab does not work yet. Weather, AQI, map and geocoding are unaffected.
- Milestone 5: `backend/requirements.txt` and README still need updating (e.g. `segmentation_models_pytorch`, `albumentations`, `opencv-python-headless`, `requests` are used but not listed).
