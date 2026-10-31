# Disaster Insights

Flood extent analysis from Sentinel-2 satellite imagery, with local weather and air quality.
B.Sc. AI & ML final-year project — **research prototype**. It is not an official warning service and it does not forecast floods.

| Page | What it does |
|---|---|
| Flood analysis | Pick a place; the latest clear Sentinel-2 image of a 5 × 5 km area (last 45 days) is segmented by a U-Net. Known permanent water and sea are set aside; the rest is reported as a flood-like share of the valid land, with its denominator, image date and quality, and a map overlay. |
| Weather & air | Current conditions, 72 h hourly and 7-day forecast, sunrise/sunset, wind and gusts, visibility; US AQI and European AQI (separately), pollutants, pollen, trends and general health guidance (Open-Meteo / CAMS). |
| Demo scenes | Archived WorldFloods scenes of past events (validation and held-out test only), with agreement against Copernicus EMS flood maps. |
| Research | Readiness checks, model registry and evaluation summaries, outcome policy, raw diagnostics of the last result (scene viewer, error map, method re-run, response JSON). |
| Methodology | How results are produced, outcome definitions, limitations, sources and licences. |

## Architecture

```
frontend/  React 19 + Vite 8, Leaflet. Hash routes (#/, #/weather, #/demo, #/research, #/methodology).
           Calls the API with relative URLs (/api, /ready, /health); Open-Meteo and Nominatim directly.
backend/   FastAPI. /api/flood/{model-status,demo-samples,assess,research,satellite-status}, /health, /ready.
ml/        Inference (seg_inference.py), outcome rules (assessment.py), training, evaluation, dataset tools.
models/    flood/segmentation_v2 (served), flood/segmentation_v1 (previous RGB model). Legacy classifier files are never served.
data/      WorldFloods v2 subset (raw), patch cache (interim), masks/splits (processed), reports.
```

## Model

`segmentation_v2`: U-Net with an EfficientNet-B2 encoder, Sentinel-2 L1C bands B02, B03, B04, B08, B11, B12, trained to detect
**surface water**; flood-like = water outside JRC permanent water. Trained on 3,646 patches from 203 scenes / 101 events
(frozen validation and test events excluded). Checkpoint and thresholds were chosen on the validation split; the test split
was evaluated once per variant (see the Milestone 5 report for the full history, including one test-informed decision).

Full-resolution product protocol (positives outside JRC permanent water vs Copernicus EMS flood, tiled inference):

| Model | Validation IoU (P / R) | Held-out test IoU (P / R) |
|---|---|---|
| segmentation_v1 (RGB, EMS-flood target) | 0.252 (0.33 / 0.52) | 0.039 (0.05 / 0.18) |
| **segmentation_v2** (6 bands, surface water) | **0.457** (0.52 / 0.79) | **0.157** (0.17 / 0.63) |

Pixel-level agreement on held-out events is still low: treat results as indicative. Outcomes are descriptive
("flood-like patterns detected", "no flood-like patterns detected", "inconclusive", "inconclusive — mostly permanent water",
"insufficient valid coverage") and are never probabilities, confidences or risk ratings. Scenes hazier than every training
image are reported as inconclusive.

## Quick start (Windows, PowerShell)

```powershell
# Python 3.14 virtual environment
python -m venv venv
.\venv\Scripts\python.exe -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126   # or the CPU build
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt      # training, evaluation, tests

# Live imagery (optional): copy .env.example to backend\.env and set CDSE_CLIENT_ID / CDSE_CLIENT_SECRET
# Backend (terminal 1)
cd backend; ..\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Frontend (terminal 2)
cd frontend; npm ci; npm run dev          # http://localhost:5173 (proxies /api to :8000)
```

Without Copernicus credentials the app still runs: live analysis is disabled with an explanation, and demo scenes, weather and
air quality work. Pre-compute demo results with `.\venv\Scripts\python.exe backend\scripts\warm_cache.py --methods tiled resize`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET` | — | Copernicus Data Space credentials (server only, `backend/.env`) |
| `DI_SEG_MODEL` | `segmentation_v2` | Served model version; there is no fallback if its checkpoint is missing |
| `DI_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `DI_ALLOWED_ORIGINS` | local Vite ports | Comma-separated CORS origins |
| `DI_RATE_LIMIT_ASSESS` | `12` | Live assessments per client per minute (demo: 120) |
| `DI_MAX_BODY_BYTES` | `16384` | Request size limit (POST without Content-Length → 411) |
| `VITE_API_URL` | empty (same origin) | Build-time API origin for the frontend |
| `VITE_PROXY_TARGET` | `http://127.0.0.1:8000` | Dev/preview proxy target |

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest backend/tests ml/tests -q          # backend + ML (218 tests)
cd frontend; npm test; npm run lint; npm run build; npm audit           # unit/integration (45 tests), lint, build
node e2e/smoke.mjs all        # real app in Chromium (Edge/Chrome), Firefox and WebKit + axe-core; needs both servers running
```

## Reproducing the model

```powershell
.\venv\Scripts\python.exe ml\datasets\expand_worldfloods_v2.py --budget-gb 10 --per-event 2 --threads 4   # more training events (90 GB hard cap)
.\venv\Scripts\python.exe ml\datasets\build_patch_cache_v2.py --workers 6
.\venv\Scripts\python.exe ml\training\train_flood_seg_v2.py --epochs 20 --patience 8 --out scratch\v2_runB
.\venv\Scripts\python.exe ml\training\promote_checkpoint_v2.py --source scratch\v2_runB --rule "..."
.\venv\Scripts\python.exe ml\evaluation\assessment_calibration.py --model segmentation_v2 --split val  --output data\reports\<new>.json
.\venv\Scripts\python.exe ml\evaluation\assessment_calibration.py --model segmentation_v2 --split test --output data\reports\<new>.json
```

Training on an RTX 4050 (6 GB) takes about 70–150 s per epoch; the full-resolution validation evaluation about 4 minutes.

## Deployment notes

- Build the UI with `npm run build` and serve `frontend/dist` and the API from **one origin** behind a reverse proxy
  (`/api`, `/ready`, `/health` → uvicorn). Otherwise build with `VITE_API_URL` and set `DI_ALLOWED_ORIGINS`.
- Run uvicorn with a single worker per GPU (the model is loaded once per process; rate limits are per process).
- The API has **no authentication**. Rate limits and request-size limits reduce abuse, but put it behind HTTPS and,
  for public use, an authenticating proxy. Copernicus credentials stay on the server; error responses never include them.
- External services: Copernicus Data Space (imagery), JRC Global Surface Water on Google Cloud Storage (known water),
  Open-Meteo (weather/AQ), OpenStreetMap tiles and Nominatim (map, search — follow their usage policies for heavy traffic).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Service offline" in the header | Backend not running or proxy target wrong; the UI re-checks every 20 s. |
| "Model unavailable" | Checkpoint for `DI_SEG_MODEL` missing (`/ready` shows `segmentation_checkpoint: missing`). |
| Live analysis disabled | `CDSE_CLIENT_ID/SECRET` not set in `backend/.env`. |
| "Imagery unavailable" | No usable acquisition in 45 days, or Copernicus unavailable (HTTP 404 / 502). |
| "Too many requests" | Rate limit reached; wait for the time shown. |
| First demo scene slow | Large scenes take up to ~40 s once, then load from `data/cache/seg_api`. |

## Data and licences

Contains modified Copernicus Sentinel data. WorldFloods v2 (CC BY-NC 4.0, isp-uv-es/WorldFloodsv2). Copernicus EMS flood maps.
JRC Global Surface Water (Source: EC JRC/Google). Open-Meteo (CC BY 4.0) with CAMS air-quality forecasts.
Map data © OpenStreetMap contributors (ODbL). Academic, non-commercial use.
