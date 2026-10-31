"""
Flood Assessment API Routes (U-Net segmentation).

Endpoints
    GET  /api/flood/model-status      active segmentation model + live-imagery configuration
    GET  /api/flood/demo-samples      verified archived demonstration scenes (validation + held-out test)
    GET  /api/flood/satellite-status  Copernicus credential check (live mode)
    GET  /api/flood/research          model registry + evaluation summaries (diagnostic; no secrets/paths)
    POST /api/flood/assess            run an assessment
         demo: {"mode": "demo", "sample_id": "...", "method": "tiled"|"resize"}
         live: {"mode": "live", "latitude": .., "longitude": .., "method": ..}

Live and demo are strictly separate: a failed live request never falls back to an archived scene.
The legacy EfficientNet-B0 classifier (ml/inference/flood_inference.py) is not reachable from these
routes: the model is chosen by app.config (DI_SEG_MODEL, segmentation versions only).

Endpoints are plain `def` so FastAPI runs blocking inference in a worker thread.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app import config  # noqa: E402
from app.services.rate_limit import RateLimiter  # noqa: E402
from app.services.seg_service import (  # noqa: E402
    DEFAULT_METHOD,
    SampleNotFoundError,
    get_seg_service,
)
from app.services.water_reference import get_water_reference  # noqa: E402
from ml.inference import seg_inference as si  # noqa: E402
from ml.inference.seg_inference import METHODS  # noqa: E402

LIVE_SEARCH_DAYS = 45

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/flood", tags=["flood"])

_sentinel_service = None
live_limiter = RateLimiter(per_minute=config.rate_limit_assess())
demo_limiter = RateLimiter(per_minute=120)


def get_sentinel_service():
    """Get or create the Copernicus data service (singleton)."""
    global _sentinel_service
    if _sentinel_service is None:
        from app.services.sentinel_service import SentinelDataService
        _sentinel_service = SentinelDataService()
    return _sentinel_service


def _satellite_configured() -> bool:
    try:
        return get_sentinel_service().is_configured
    except Exception:
        return False


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class AssessmentRequest(BaseModel):
    """Request body for a flood assessment."""
    mode: Literal["demo", "live"] = Field(default="demo", description="'demo' (archived sample) or 'live'")
    sample_id: Optional[str] = Field(default=None, max_length=200, pattern=r"^[A-Za-z0-9_\-]+$",
                                     description="Demo sample id from /demo-samples")
    method: Literal["tiled", "resize"] = Field(default=DEFAULT_METHOD, description="Inference method")
    latitude: Optional[float] = Field(default=None, ge=-90, le=90, description="Latitude (live mode)")
    longitude: Optional[float] = Field(default=None, ge=-180, le=180, description="Longitude (live mode)")

    @model_validator(mode="after")
    def _check_mode_fields(self):
        if self.mode == "demo" and not self.sample_id:
            raise ValueError("Demo mode requires 'sample_id' (see GET /api/flood/demo-samples).")
        if self.mode == "live" and (self.latitude is None or self.longitude is None):
            raise ValueError("Live mode requires 'latitude' and 'longitude'.")
        return self


@router.get("/model-status")
def model_status():
    """Report whether the segmentation model is loaded and ready."""
    status = get_seg_service().status()
    status["satellite"] = "configured" if _satellite_configured() else "not_configured"
    return status


@router.get("/demo-samples")
def demo_samples():
    """List verified archived demonstration scenes. Validation first, then held-out test."""
    svc = get_seg_service()
    try:
        samples = [s.public() for s in svc.catalogue()]
    except Exception as e:
        logger.error("Demo catalogue failed: %s", type(e).__name__, exc_info=True)
        raise HTTPException(status_code=503, detail="The demonstration catalogue is unavailable on this server.")
    return {
        "count": len(samples),
        "default_method": DEFAULT_METHOD,
        "methods": list(METHODS),
        "samples": samples,
        "excluded": svc.excluded(),
        "note": "Demo samples come from the validation and held-out test splits only. Each scene shows its own "
                "event location, not the location selected on the map.",
    }


@router.get("/satellite-status")
def satellite_status():
    """Check Copernicus satellite data service status (live mode)."""
    try:
        service = get_sentinel_service()
        if not service.is_configured:
            return {
                "status": "not_configured",
                "message": "CDSE credentials not set. Add CDSE_CLIENT_ID and CDSE_CLIENT_SECRET to backend/.env",
                "registration_url": "https://dataspace.copernicus.eu/",
            }
        try:
            service._get_token()
            return {"status": "ready", "message": "Copernicus Sentinel Hub connected successfully",
                    "source": "Copernicus Data Space Ecosystem (CDSE)"}
        except Exception as e:
            logger.warning("Copernicus authentication check failed: %s", type(e).__name__)
            return {"status": "auth_error", "message": "Copernicus authentication failed. Check the server credentials."}
    except Exception as e:
        logger.warning("Satellite status check failed: %s", type(e).__name__)
        return {"status": "error", "message": "Satellite service status could not be checked."}


_PATH_KEYS = {"checkpoint", "data_root", "masks_dir", "manifest_path", "checkpoint_dir", "image_path", "file",
              "saved_to"}


def _scrub(obj):
    """Remove local filesystem paths from report structures before returning them."""
    if isinstance(obj, dict):
        return {k: ("<redacted>" if k in _PATH_KEYS and isinstance(v, str) else _scrub(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, str) and (":\\" in obj or obj.startswith("/home/") or obj.startswith("/Users/")):
        return "<redacted>"
    return obj


def _load_report(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@router.get("/research")
def research_summary():
    """Model registry and evaluation summaries for the developer/research view.

    Diagnostic only: aggregate metrics, no per-pixel data, no secrets, no local paths.
    """
    reports = project_root / "data" / "reports"
    models = []
    for version in config.KNOWN_MODELS:
        ck = config.seg_checkpoint(version)
        cfg = _load_report(ck.parent / "training_config.json") or {}
        cfg = cfg.get("config", cfg)
        entry = {"version": version, "installed": ck.is_file(), "active": version == get_seg_service().model_version,
                 "training": _scrub({k: cfg.get(k) for k in (
                     "task", "target", "bands", "encoder_name", "input_size", "train_scenes", "train_events",
                     "train_patches", "epochs", "batch", "lr", "seed", "selection_metric",
                     "best_val_flood_iou_tiles", "train_seconds", "device", "pytorch", "pytorch_version",
                     "max_epochs", "batch_size") if cfg.get(k) is not None})}
        for split in ("val", "test"):
            name = (f"assessment_policy_{version}_val.json" if split == "val" else f"evaluation_{version}_test.json")
            if version == "segmentation_v1":
                name = ("evaluation_segmentation_v1_val.json" if split == "val" else "evaluation_segmentation_v1_test.json")
            rep = _load_report(reports / name)
            if rep:
                entry[split] = _scrub({m: {k: rep["summary"][m][k] for k in (
                    "pixel_metrics_raw", "pixel_metrics_excluding_known_water", "per_event_excluding_known_water",
                    "levels", "negative_outside_water_pct", "seconds_total") if k in rep["summary"][m]}
                    for m in rep["summary"]})
                entry[f"{split}_n_samples"] = len(rep.get("per_sample", []))
                entry[f"{split}_timestamp"] = rep.get("timestamp")
        models.append(entry)
    svc = get_seg_service()
    return {
        "active_model": svc.model_version,
        "policy": _scrub(svc.policy),
        "protocol": "Full-resolution inference on the frozen event-disjoint splits; flood-like = model positives "
                    "outside JRC permanent water, scored against Copernicus EMS flood masks (validity applied). "
                    "Validation was used for checkpoint and threshold selection; test was evaluated once.",
        "models": models,
    }


@router.post("/assess")
def assess_flood(body: AssessmentRequest, request: Request):
    """Run a flood segmentation assessment."""
    limiter = live_limiter if body.mode == "live" else demo_limiter
    retry = limiter.check(_client_key(request))
    if retry:
        raise HTTPException(status_code=429, detail="Too many assessment requests. Please wait a moment.",
                            headers={"Retry-After": str(retry)})
    svc = get_seg_service()
    try:
        svc.segmenter()
    except Exception:
        raise HTTPException(status_code=503, detail=svc.public_error() or "The segmentation model is not available.")

    if body.mode == "demo":
        try:
            return svc.assess_demo(body.sample_id, body.method)
        except SampleNotFoundError:
            raise HTTPException(status_code=404, detail=f"Unknown demo sample '{body.sample_id}'. "
                                                        "Choose one from GET /api/flood/demo-samples.")
        except Exception as e:
            logger.error("Demo assessment failed: %s", type(e).__name__, exc_info=True)
            raise HTTPException(status_code=500, detail="The demonstration assessment could not be completed.")

    return _assess_live(svc, body)


BASELINE_OFFSET_DAYS = 335   # search window ends ~11 months before today's image ...
BASELINE_WINDOW_DAYS = 60    # ... and covers 60 days: same season, about one year earlier


def _fetch_baseline(service, lat, lng, metadata, spec, shape):
    """Same-season image about one year before today's acquisition, on the same request grid.
    Returns {"image", "validity", "acquisition_date"} or {"error": reason}. Never raises."""
    from datetime import datetime as _dt, timedelta as _td
    day = metadata.get("acquisition_date")
    if not day:
        return {"error": "Today's acquisition date is unknown, so no earlier image can be matched."}
    end = _dt.fromisoformat(day[:10]) - _td(days=BASELINE_OFFSET_DAYS)
    try:
        b_img, b_val, b_meta = service.fetch_scene(lat=lat, lng=lng, size_km=5.0, resolution=10,
                                                   days_back=BASELINE_WINDOW_DAYS, end=end)
    except LookupError:
        return {"error": "No Sentinel-2 image of this area was found for the same season a year earlier."}
    except Exception as e:
        logger.warning("Baseline fetch failed: %s", type(e).__name__)
        return {"error": "The earlier satellite image could not be downloaded."}
    try:
        b = si.select_live_bands(b_img, spec)
    except si.InvalidInputError:
        return {"error": "The earlier satellite image was incomplete."}
    if b.shape[:2] != shape:
        return {"error": "The earlier image is on a different grid and cannot be compared."}
    if b_val is not None and b_val.shape != shape:
        b_val = None
    return {"image": b, "validity": b_val, "acquisition_date": b_meta.get("acquisition_date")}


def _assess_live(svc, request: AssessmentRequest):
    """Live assessment on one dated Copernicus Sentinel-2 L1C acquisition."""
    service = get_sentinel_service()
    if not service.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Live satellite imagery is not configured on this server (Copernicus credentials missing).",
        )
    lat, lng = request.latitude, request.longitude
    # The request bbox is known before the imagery: start the slow OpenStreetMap water query now
    osm_future = None
    make_bbox = getattr(service, "_make_bbox", None)
    water_ref = get_water_reference()
    if make_bbox is not None and getattr(water_ref, "use_osm", False):
        from app.services.water_reference import prefetch_osm
        osm_future = prefetch_osm(make_bbox(lat, lng, 5.0))
    try:
        logger.info("Fetching live Sentinel-2 data near (%.3f, %.3f)", lat, lng)
        image, validity, metadata = service.fetch_scene(lat=lat, lng=lng, size_km=5.0, resolution=10,
                                                        days_back=LIVE_SEARCH_DAYS)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=f"{e} Try again later.")
    except Exception as e:
        # Log the type only: exception text from HTTP clients can include request details.
        logger.error("Live satellite fetch failed: %s", type(e).__name__)
        raise HTTPException(
            status_code=502,
            detail="The Copernicus satellite service could not provide imagery right now. Try again later.",
        )

    if metadata.get("quality_status") == "empty":
        raise HTTPException(
            status_code=422,
            detail=f"The latest image covers only {metadata.get('data_coverage_pct', 0)}% of this area. "
                   "Try a different location or try again later.",
        )

    spec = svc.segmenter().spec
    try:
        img = si.select_live_bands(image, spec)   # [B02,B03,B04,B08,B11,B12] -> model band order
    except si.InvalidInputError:
        logger.error("Live raster had an unexpected band layout: %s", getattr(image, "shape", None))
        raise HTTPException(status_code=502, detail="The satellite service returned an incomplete image.")
    h, w = img.shape[:2]
    if validity is not None and validity.shape != (h, w):
        validity = None
    bbox = metadata.get("bbox") or []

    known_water, water_info = None, {"status": "unavailable"}
    if len(bbox) == 4:
        if osm_future is not None:
            known_water, water_info = water_ref.for_bbox(bbox, w, h, osm_future=osm_future)
        else:
            known_water, water_info = water_ref.for_bbox(bbox, w, h)

    imagery = {
        "source": metadata.get("source", "Copernicus Sentinel-2 Level-1C"),
        "processing_level": metadata.get("processing_level", "L1C"),
        "acquisition_date": metadata.get("acquisition_date"),
        "acquisition_datetime": metadata.get("acquisition_datetime"),
        "processed_at": metadata.get("fetched_at"),
        "date_range": None,
        "resolution_m": metadata.get("resolution_m", 10),
        "width": int(w),
        "height": int(h),
        "crs": "EPSG:4326",
        "center": {"lat": lat, "lng": lng},
        "bbox": dict(zip(("west", "south", "east", "north"), bbox)) or None,
        "data_coverage_pct": metadata.get("data_coverage_pct"),
        "tile_cloud_cover_pct": metadata.get("tile_cloud_cover_pct"),
        "cloud_mask_source": metadata.get("cloud_mask_source"),
        "cloudy_pct_of_data": metadata.get("cloudy_pct_of_data"),
        "cloud_status": "masked" if validity is not None else "unknown",
        "acquisitions_considered": metadata.get("acquisitions_considered"),
        "search_days": metadata.get("search_days"),
        "bands": list(spec.bands),
        "location_note": "A 5 × 5 km area centred on the selected point, from a single Sentinel-2 acquisition.",
    }
    extra = []
    if validity is None:
        extra.append("No cloud mask was available for this image; clouds and shadows may be counted as water.")
    else:
        extra.append("Clouds and cloud shadows were masked with the Sentinel-2 scene classification (Sen2Cor); "
                     "thin cloud and haze can still be missed.")
    baseline = _fetch_baseline(service, lat, lng, metadata, spec, (h, w)) if config.historical_comparison() else None
    try:
        return svc.assess_live(img, validity, imagery, request.method, extra,
                               known_water=known_water, water_info=water_info, bbox=bbox, baseline=baseline)
    except Exception as e:
        logger.error("Live assessment failed: %s", type(e).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail="The assessment could not be completed. Please try again.")
