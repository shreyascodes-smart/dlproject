# Disaster Insights — Kiro Context Audit & Repository Verification

> Date: 2026-10-26
> Type: Read-only inspection (no project files edited, no packages installed, no training run)
> Demonstration deadline: 31 October 2026

## Summary

The handoff is mostly accurate: 88/88 tests pass, both U-Net checkpoints load cleanly, and the backend and frontend still use the legacy classifier. Three problems the handoff does not mention change the plan:

- **Longitude bug:** the backend rejects any longitude beyond ±90°, so places such as eastern India, Bangladesh, Japan or the US West Coast fail.
- **Unrelated demo image:** demo mode picks an image with arithmetic on the clicked coordinates, so it has nothing to do with the chosen location. 11 of the 17 demo images were used to train the U-Net.
- **Scale mismatch:** the model learned from full-detail 256 px crops, but validation and test shrank whole scenes about 12× before scoring. The reported metrics measure the model at a scale it was never trained on.

## A. Project understanding

Today, Disaster Insights lets the user pick a location (GPS or map), shows weather and air quality from Open-Meteo, and runs a "flood assessment". That assessment uses the old EfficientNet-B0 classifier, which outputs one "flood probability" and a Grad-CAM heatmap (a picture of where the model "looked").

The goal is to replace that with the U-Net segmentation model. Instead of one yes/no score, it labels every pixel as flood or not-flood, and the app shows that as a coloured overlay with honest notes on its limits.

## B. Verified repository state

- **Branch / HEAD:** `main` at `566a67c` ("docs: add Kiro master migration and project handoff guide"). The handoff says `bccaa94`, which is now one commit behind.
- **Working tree:** clean. The audit left it unchanged.
- **Remotes:** `origin` → `dlproject.git`, `dl_legacy` → `dl.git`. Neither URL contains credentials. Not fetched, so GitHub state was not compared with local.
- **Tracked files:** no `.pt` weights, `.tif` files, raw data or `.env` files are tracked, and no `.env` was ever committed. Six `.docx` files (~4 MB each) are tracked. Old commits contain `frontend/node_modules` binaries, which make the history large but cause no harm.
- **Environment:**
  - Python 3.14.6, torch 2.14.0+cu126, CUDA available on the RTX 4050 Laptop GPU.
  - segmentation_models_pytorch 0.5.0, albumentations 2.0.8, rasterio 1.5.1, FastAPI 0.141.1.
  - Node 24.18.1, React 19.3.0, Vite 8.3.1, Leaflet 1.9.4, react-leaflet 5.0.0, axios 1.20.0.
- **Data:** raw WorldFloods v2 present (~26 GB). 306 mask files = 153 samples × 2 (seg + validity).
- **Credentials:** `backend/.env` has `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` set. Values were not read.

## C. Completed vs. verified work

**Verified by running it:**

- **Tests:** `pytest ml/tests` → 88 passed, 0 failed, 0 skipped, 0 errors (16 + 22 + 50 across the three test files).
- **Checkpoints:** both U-Net files load into `smp.Unet(efficientnet-b2, encoder_weights=None)` with all keys matching. They do not need the internet or the pretrained-weights cache.
- **Backend:** imports and starts when run from inside `backend/`.
- **Legacy demo flow:** returns a result. For a Mumbai location it reported "Very High Risk, 97%" using an image from Vatomandry, Madagascar.
- **Splits:** train, validation and test share no flood events. Train 107 samples / 27 events, validation 24 / 6, test 22 / 6.

**Exists but not re-verified in this audit:**

- Phase H metrics (saved result files were read, evaluation was not re-run).
- The mask-preparation run.
- Live satellite mode (would send credentials to Copernicus, so it was skipped).

## D. Model status

| | Legacy classifier | U-Net segmentation |
|---|---|---|
| Files | `best_flood_model.pt` (epoch 16), `final_flood_model.pt` (epoch 20) | `best_seg_model.pt` (epoch 14), `final_seg_model.pt` (epoch 29) |
| Output | One flood probability per image | A flood/not-flood label for every pixel |
| Scientifically valid? | No — trained on misread ground-truth bands | Yes — trained on Copernicus EMS mapped flood polygons |
| Used by the app? | Yes | No |

**Reported metrics (match the saved JSON files):**

| Split | Flood IoU | Flood Dice | Precision | Recall | Pixel accuracy |
|---|---|---|---|---|---|
| Validation (24 rasters, 6 events) | 0.1205 | 0.2150 | 0.1624 | 0.3180 | 90.74% |
| Test (22 rasters, 6 events) | 0.0223 | 0.0437 | 0.0324 | 0.0670 | 94.64% |

IoU (intersection over union) measures how much of the predicted flood area overlaps the real flood area, from 0 to 1. Pixel accuracy is misleading because most pixels are non-flood.

Caveats:

- The validation score is almost entirely one event: EMSR342 (Australia) produced 12,316 of the 12,410 correct flood pixels.
- Because of the scale mismatch, the shipped model has never been scored at its native 10 m detail.

## E. Current application flow (traced in code)

1. `frontend/src/App.jsx` sends `POST /api/flood/assess` with latitude, longitude and mode. It never sends a demo image choice.
2. `backend/app/api/flood_routes.py` validates longitude against ±90 (the bug).
   - **Demo mode:** picks the image with `abs(lat*100 + lng*10) % 17` from the raw WorldFloods `test/S2` folder.
   - **Live mode:** fetches a 6-band Sentinel-2 L2A image from Copernicus.
3. `ml/inference/flood_inference.py` (`FloodInference`):
   - **Model:** legacy EfficientNet-B0.
   - **Preprocessing:** central 512 px crop → resize to 224 px → divide by 10000 → ImageNet normalisation.
   - **Output:** a score plus a Grad-CAM image.
4. The frontend shows the percentage as "Confidence" and labels it "EfficientNet-B0 / Grad-CAM". Weather, air quality, the map and reverse geocoding are independent and stay as they are.

### Demo image → new split mapping

| Index | Image | Segmentation split |
|---|---|---|
| 0 | EMSR264_08VATOMANDRY_DEL_v2 | val |
| 1 | EMSR264_18MIANDRIVAZODETAIL_DEL_v2 | val |
| 2 | EMSR273_01GRILE_DEL_MONIT04_v2 | train |
| 3 | EMSR286_08ITUANGONORTH_DEL_MONIT02_v1 | train |
| 4 | EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1 | train |
| 5 | EMSR333_01RATTALORO_DEL_MONIT01_v1 | not in manifest (excluded) |
| 6 | EMSR333_02PORTOPALO_DEL_MONIT01_v1 | train |
| 7 | EMSR342_06NORTHNORMANTON_DEL_v1 | val |
| 8 | EMSR342_07SOUTHNORMANTON_DEL_MONIT03_v2 | val |
| 9 | EMSR347_07ZOMBA_DEL_MONIT01_v1 | train |
| 10 | EMSR347_07ZOMBA_DEL_v2 | train |
| 11 | EMSR419_AOI01_DEL_MONIT01 | train |
| 12 | EMSR422_AOI06_DEL_PRODUCT | val |
| 13 | EMSR438_AOI02_DEL_MONIT03 | train |
| 14 | EMSR438_AOI06_DEL_PRODUCT | train |
| 15 | EMSR466_AOI01_DEL_PRODUCT | train |
| 16 | EMSR9284_01YLITORNIONORTHERN_DEL_MONIT01_v1 | test |

## F. Remaining work (priority for 31 October)

1. Segmentation inference module (does not exist yet).
2. Point the backend at the U-Net, return the overlay and flood percentage, fix the longitude bug.
3. Replace the frontend results panel, add a demo-sample picker, remove the EfficientNet/Grad-CAM wording.
4. Choose demo samples honestly from validation/test and label each one's split.
5. Update `README.md` and `backend/requirements.txt` (missing segmentation_models_pytorch, albumentations, opencv, timm, requests, pytest).
6. Optional: side-by-side images for the presentation.

## G. Risks and discrepancies

- **Wrong start command in handoff:** `uvicorn backend.app.main:app` from the project root fails with `No module named 'app'`. The README's version is correct: run from inside `backend/` as `app.main:app`.
- **Evaluation overwrites results:** `ml/evaluate_seg.py` overwrites `data/reports/phase_h_*_evaluation.json` by default. Any future run must pass `--output` with a new file.
- **Live mode mismatch:** live mode fetches Level-2A (atmospherically corrected) imagery, while WorldFloods is usually Level-1C (top-of-atmosphere). Not confirmed from local files. Live results would be unvalidated, and there is no cloud mask.
- **Live by default:** the frontend switches to live mode automatically when credentials exist. Risky for a viva.
- **Small report errors:** the Phase H report mentions `eta_min=1e-6`, which the code does not set; its training log lives outside the repo (Antigravity folder).
- **Outdated text:** README and the welcome screen still describe EfficientNet-B0 and Grad-CAM.
- **Missing image:** `EMSR347_06MWANZA_DEL_v1` has ground truth but no Sentinel-2 file in raw `test/`.
- **Minor display bug:** coordinates always show "°N/°E", even for south or west.

## H. Recommended implementation plan

1. **Inference module** (`ml/inference/seg_inference.py`) — low risk.
   - Load the checkpoint with `encoder_weights=None` (no cache needed).
   - Preprocess exactly as in training: Sentinel-2 bands 4, 3, 2 → ÷10000 → clip 0–1 → ImageNet normalisation.
   - Two run modes: resize whole scene to 256 px (how it was scored) or slide 256 px tiles at full detail (how it was trained).
   - Return mask, flood %, invalid-pixel share, and overlay PNG. Unit tests on synthetic data.
2. **Validation-only comparison** of the two modes, saved to a new file, test set untouched. Medium cost (reads ~2.4 GB). Purpose: choose the demo default honestly, not to tune the model.
3. **Backend** — low–medium risk. Demo-sample list from the manifest, update `/assess` and `/model-status`, fix longitude. Keep legacy files but remove them from the main flow.
4. **Frontend** — medium risk.
   - Results: original image, flood overlay, ground truth (demo only), flood %, data quality, limitations.
   - Weather, AQI and map unchanged. Verify with `npm run build` and lint.
5. **Live mode** kept but labelled experimental; demo becomes the default.
6. **Docs and presentation images:** update README and requirements; export validation/test sample images for slides.

No retraining is included.

## I. Architecture and constraints

Two independent parts sit side by side:

- **Context:** weather, air quality and location (Open-Meteo, Nominatim, Leaflet map).
- **Flood mapping:** satellite image → U-Net → pixel flood map.

They are not connected, and the app must not suggest that weather predicts the flood map. The U-Net is the only trustworthy model but is weak on unseen events, so the demo should be honest rather than impressive.

Fixed rules: never touch raw data, masks or legacy checkpoints; no tuning on the test set; never commit weights or secrets.

Next task: step 1, the inference module.

## J. Open decisions (awaiting approval)

1. Approve the plan above?
2. Demo samples only from validation/test (recommended), or also training samples if clearly labelled?
3. Show the ground-truth mask next to the prediction in demo mode (recommended: yes)?
4. Keep live mode as an experimental option with demo as the default?
