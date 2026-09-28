# Development Log

## Project: Deep Learning-Based Early Detection and Classification of Climate-Induced Natural Disasters Using Satellite Imagery

### Current Phase: Flood Pipeline (10-Hour Sprint)

---

## 2026-09-28 — Sprint Start

### Phase 0: System Inspection ✅
- **What:** Inspected the development environment before making any changes
- **Result:**
  - Python 3.14.6, Node v24.18.1, npm 11.5.2, Git 2.55.0
  - GPU: NVIDIA RTX 4050 Laptop (6 GB VRAM)
  - Disk: ~172 GB free on D:
  - HF CLI: v1.26.0 with dry-run support
  - Project: Clean slate (only sprint plan file existed)
  - WorldFloods: Not yet downloaded
- **Decision:** Proceed with project foundation setup
- **Concern:** Python 3.14 is very new — PyTorch compatibility verified ✅

### Phase 1: Project Foundation ✅
- Created virtual environment with Python 3.14
- PyTorch 2.14.0+cu126 installed (CUDA 12.6) — verified working
- Git repo initialized with 4 commits so far
- Folder structure: frontend/, backend/, ml/, data/, models/, notebooks/, docs/, tests/
- Installed: fastapi, uvicorn, rasterio, tifffile, matplotlib, scikit-learn, tqdm

### Phase 2: Dataset Acquisition (In Progress)
- **Decision:** Subset of WorldFloods v2 (90 GB ceiling) → download 184 image pairs (~32 GB)
  - All test (18 images) ✅
  - All val (16 images) ✅
  - 150 train images (downloading...)
- **Data inspection findings:**
  - S2 TIFFs: 15 bands, int16, values 0-14000+
  - GT masks: 2 bands, values {0=land, 1=flood, 2=cloud}
  - Permanent water: 1 band, values {0,1,2,3} where 3=permanent water
  - RGB bands: B4,B3,B2 (1-indexed) = indices 3,2,1 (0-indexed) ✅

### Phase 3: ML Pipeline ✅
- **Dataset loader** (`ml/datasets/flood_dataset.py`):
  - Reads 15-band Sentinel-2 TIFFs → extracts RGB
  - Reads 2-band GT masks → derives binary flood/non-flood labels
  - Label rule: ≥5% flood pixels (excluding permanent water) = FLOOD
  - Resizes to 224×224 for EfficientNet-B0
- **Model** (`ml/models/flood_classifier.py`):
  - EfficientNet-B0 with modified classifier head (2 classes)
  - Transfer learning: freeze backbone → train head → unfreeze later
  - 4,010,110 total params, 2,562 trainable when frozen
  - CUDA forward pass: 35.7 MB GPU memory (of 6 GB)
- **Training** (`ml/training/train_flood.py`, `quick_train.py`):
  - Mandatory 6-step smoke test before real training
  - Transfer learning schedule: 5 epochs frozen → unfreeze from layer 5
  - Checkpoint saving (best + final)
- **Grad-CAM** (`ml/explainability/gradcam.py`):
  - Hooks into last convolutional layer
  - Produces 224×224 heatmaps
  - Overlay visualization with probability display
- **Inference** (`ml/inference/flood_inference.py`):
  - Loads checkpoint, runs prediction, interprets risk level
  - Generates Grad-CAM explanation
  - Risk levels: Low (<20%), Moderate (20-50%), High (50-80%), Very High (>80%)

### Phase 4: End-to-End Smoke Test ✅
- **ALL 6 TESTS PASSED** on test split with real data
- Batch loading: [4, 3, 224, 224] tensors
- Forward pass on CUDA: works
- Training step: loss=0.8250, backward pass OK
- Grad-CAM visualization: saved and verified
- GPU peak memory: 120.1 MB

### Phase 5: FastAPI Backend ✅
- `backend/app/main.py` — FastAPI application with CORS
- `backend/app/api/flood_routes.py` — Assessment endpoints
  - POST `/api/flood/assess` — runs inference + Grad-CAM
  - GET `/api/flood/demo-images` — lists available test images
  - GET `/api/flood/model-status` — checks model readiness

### Phase 6: React Frontend ✅
- Vite + React setup with Leaflet + OpenStreetMap
- Premium dark-mode UI with glassmorphism design
- Interactive map (click to select location)
- Risk display with animated probability circle
- Grad-CAM image panel (original + heatmap overlay)
- Build test: PASS (0 errors)

### Label Distribution (test/val)
- Test: 16 flood, 2 non-flood (89% flood) — expected for a flood dataset
- Val: 16 flood, 0 non-flood (100% flood)

### Quick Training: IN PROGRESS
- Using test as train data, val as validation
- Will produce a working checkpoint for demo
- Full training will run when train/gt downloads complete

