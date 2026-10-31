"""
Runtime configuration (environment variables; safe defaults for local development).

    DI_SEG_MODEL          segmentation model version directory under models/flood/
                          (default below; the checkpoint must exist — there is NO fallback)
    DI_DEVICE             auto | cuda | cpu
    DI_ALLOWED_ORIGINS    comma-separated CORS origins (default: local Vite dev/preview)
    DI_RATE_LIMIT_ASSESS  live/demo assessments per client per minute (default 12)
    DI_MAX_BODY_BYTES     max request body size (default 16 KiB)
    CDSE_CLIENT_ID / CDSE_CLIENT_SECRET   Copernicus Data Space credentials (server-side only)
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "flood"

# Chosen on the frozen validation split (see docs/milestone5_* report). Change only with evidence.
DEFAULT_SEG_MODEL = "segmentation_v2"
KNOWN_MODELS = ("segmentation_v1", "segmentation_v2")
LEGACY_CLASSIFIER_FILES = ("best_flood_model.pt", "final_flood_model.pt")


def seg_model_version() -> str:
    v = os.getenv("DI_SEG_MODEL", DEFAULT_SEG_MODEL).strip()
    if v not in KNOWN_MODELS:
        raise ValueError(f"DI_SEG_MODEL must be one of {KNOWN_MODELS}, got {v!r}.")
    return v


def seg_checkpoint(version: str) -> Path:
    return MODELS_DIR / version / "best_seg_model.pt"


def device() -> str:
    return os.getenv("DI_DEVICE", "auto")


def allowed_origins():
    raw = os.getenv("DI_ALLOWED_ORIGINS", "")
    if raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"]


def rate_limit_assess() -> int:
    return int(os.getenv("DI_RATE_LIMIT_ASSESS", "12"))


def historical_comparison() -> bool:
    """DI_HISTORICAL_COMPARISON (default on): compare live results with a same-season image a year earlier."""
    return os.getenv("DI_HISTORICAL_COMPARISON", "1") != "0"


def max_body_bytes() -> int:
    return int(os.getenv("DI_MAX_BODY_BYTES", str(16 * 1024)))
