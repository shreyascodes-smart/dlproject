# Deep Learning-Based Early Detection and Classification of Climate-Induced Natural Disasters Using Satellite Imagery

## Project Overview

An AI-based flood assessment and awareness tool that uses deep learning (EfficientNet-B0) to analyse Sentinel-2 satellite imagery and predict flood risk for a given location.

**Current phase:** Flood pipeline (binary classification + Grad-CAM explainability)

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React |
| Map | Leaflet + OpenStreetMap |
| Backend | FastAPI |
| ML Framework | PyTorch + torchvision |
| Model | EfficientNet-B0 (transfer learning) |
| Satellite Data | Sentinel-2 (WorldFloods v2 dataset) |
| Weather | Open-Meteo |
| AQI | WAQI |

## Project Structure

```
project-root/
├── frontend/          # React application
├── backend/           # FastAPI server
├── ml/                # ML pipeline (training, evaluation, inference, explainability)
├── data/              # Dataset storage (gitignored)
├── models/            # Trained model checkpoints (gitignored)
├── notebooks/         # Jupyter notebooks for exploration
├── docs/              # Documentation and development log
└── tests/             # Test files
```

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Start backend
cd backend && uvicorn app.main:app --reload

# 4. Start frontend
cd frontend && npm install && npm run dev
```

## Dataset

**WorldFloods v2** — Sentinel-2 flood segmentation dataset  
Source: https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2

## License

Academic project — B.Sc. AI & ML Final Year
