"""
FastAPI backend for Disaster Insights — satellite-image flood extent assessment.

Run from inside backend/:
    ..\\venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000

Configuration: see app/config.py (environment variables; no secrets in code).
"""
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend directory (Copernicus credentials for live mode; never sent to clients)
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from app import config  # noqa: E402
from app.api.flood_routes import router as flood_router  # noqa: E402
from app.api.flood_routes import get_sentinel_service  # noqa: E402
from app.services.seg_service import get_seg_service  # noqa: E402

log = logging.getLogger("uvicorn.error")
STARTED = time.time()


def _warm_up():
    """Load the model and demo catalogue so the first UI request is fast."""
    status = get_seg_service().status()
    log.info("Segmentation model warm-up: %s %s on %s (%s demo samples)", status.get("status"),
             status.get("model_version"), (status.get("model") or {}).get("device"), status.get("demo_samples"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warm_up, name="seg-warmup", daemon=True).start()
    yield


app = FastAPI(
    title="Disaster Insights API",
    description="Pixel-level flood extent analysis from Sentinel-2 imagery (U-Net segmentation).",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Reject oversized request bodies before they are read (all endpoints take tiny JSON)."""
    length = request.headers.get("content-length")
    if length is not None:
        try:
            too_big = int(length) > config.max_body_bytes()
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)
        if too_big:
            return JSONResponse({"detail": "Request body too large."}, status_code=413)
    elif request.method == "POST":
        return JSONResponse({"detail": "Content-Length required."}, status_code=411)
    return await call_next(request)


@app.get("/health")
async def health_check():
    """Liveness: the API process is running."""
    return {"status": "healthy", "service": "disaster-insights-api", "version": app.version,
            "uptime_s": round(time.time() - STARTED)}


def _probe(url: str, timeout: float = 6.0) -> str:
    import urllib.request
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as r:
            return "reachable" if r.status < 500 else f"http_{r.status}"
    except Exception as e:  # network errors are reported, never raised
        return f"unreachable ({type(e).__name__})"


@app.get("/ready")
def readiness(deep: bool = False):
    """Readiness. `ready` depends only on the segmentation model; external providers are reported
    separately (and only probed with ?deep=true) so an optional outage never marks the API unready."""
    svc = get_seg_service()
    version = svc.model_version
    ckpt_present = config.seg_checkpoint(version).is_file()
    st = svc.status()
    model_loaded = st.get("status") == "ready"
    try:
        configured = get_sentinel_service().is_configured
    except Exception:
        configured = False
    checks = {
        "api": "ok",
        "segmentation_checkpoint": "present" if ckpt_present else "missing",
        "model": "loaded" if model_loaded else "not_loaded",
        "model_version": version,
        "outcome_policy": "present" if svc.policy else "missing",
        "satellite_credentials": "configured" if configured else "not_configured",
    }
    if deep:
        checks["satellite_service"] = _probe("https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/") \
            if configured else "skipped"
        checks["permanent_water_reference"] = _probe(
            "https://storage.googleapis.com/global-surface-water/downloads2021/seasonality/seasonality_0E_10Nv1_4_2021.tif")
        checks["weather_service"] = _probe("https://api.open-meteo.com/v1/forecast?latitude=0&longitude=0&current=temperature_2m")
        checks["air_quality_service"] = _probe(
            "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=0&longitude=0&current=pm10")
    ready = ckpt_present and model_loaded
    return JSONResponse({"ready": ready, "checks": checks, "error": None if ready else st.get("error")},
                        status_code=200 if ready else 503)


@app.get("/")
async def root():
    return {"message": "Disaster Insights flood segmentation API", "docs": "/docs", "health": "/health",
            "ready": "/ready", "model_status": "/api/flood/model-status"}


app.include_router(flood_router)
