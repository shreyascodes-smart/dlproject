"""
Flood Assessment API Routes
Provides endpoints for the React frontend to trigger flood assessments.
Supports demo mode (WorldFloods dataset) and live mode (Copernicus Sentinel-2).
"""
import os
import sys
import io
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile


# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/flood", tags=["flood"])

# Global singletons
_inference_engine = None
_sentinel_service = None


class AssessmentRequest(BaseModel):
    """Request body for flood assessment."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-90, le=90, description="Longitude")
    mode: str = Field(default="demo", description="'demo' or 'live'")
    demo_image_index: Optional[int] = Field(default=None, description="Index of demo image to use")


class AssessmentResponse(BaseModel):
    """Response body for flood assessment."""
    status: str
    mode: str
    location: dict
    prediction: dict
    risk: dict
    model_info: dict
    image_base64: Optional[str] = None
    gradcam_base64: Optional[str] = None
    disclaimer: str


def get_inference_engine():
    """Get or create the inference engine (singleton)."""
    global _inference_engine

    if _inference_engine is None:
        from ml.inference.flood_inference import FloodInference

        checkpoint_path = project_root / "models" / "flood" / "best_flood_model.pt"

        if not checkpoint_path.exists():
            checkpoint_path = project_root / "models" / "flood" / "final_flood_model.pt"

        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=503,
                detail="No trained flood model found. Please train the model first."
            )

        _inference_engine = FloodInference(str(checkpoint_path))

    return _inference_engine


def get_sentinel_service():
    """Get or create the sentinel data service (singleton)."""
    global _sentinel_service

    if _sentinel_service is None:
        from app.services.sentinel_service import SentinelDataService
        _sentinel_service = SentinelDataService()

    return _sentinel_service


def _numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert a numpy image array to base64-encoded PNG."""
    img_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def _save_array_as_tiff(image_array: np.ndarray, output_path: str):
    """Save a numpy array [C, H, W] as a GeoTIFF for the inference engine."""
    bands, height, width = image_array.shape
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=height, width=width, count=bands,
        dtype=image_array.dtype,
    ) as dst:
        for i in range(bands):
            dst.write(image_array[i], i + 1)


def _get_demo_images():
    """Get list of available demo test images."""
    test_dir = project_root / "data" / "raw" / "flood" / "worldfloods_v2" / "test" / "S2"

    if not test_dir.exists():
        return []

    return sorted(test_dir.glob("*.tif"))


@router.get("/demo-images")
async def list_demo_images():
    """List available demo test images."""
    images = _get_demo_images()
    return {
        "count": len(images),
        "images": [
            {"index": i, "name": img.stem, "filename": img.name}
            for i, img in enumerate(images)
        ]
    }


@router.get("/satellite-status")
async def satellite_status():
    """Check Copernicus satellite data service status."""
    try:
        service = get_sentinel_service()
        if not service.is_configured:
            return {
                "status": "not_configured",
                "message": "CDSE credentials not set. Add CDSE_CLIENT_ID and CDSE_CLIENT_SECRET to backend/.env",
                "registration_url": "https://dataspace.copernicus.eu/",
                "dashboard_url": "https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings",
            }

        # Try to get a token to verify credentials work
        try:
            service._get_token()
            return {
                "status": "ready",
                "message": "Copernicus Sentinel Hub connected successfully",
                "source": "Copernicus Data Space Ecosystem (CDSE)",
            }
        except Exception as e:
            return {
                "status": "auth_error",
                "message": f"Authentication failed: {str(e)}",
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@router.post("/assess", response_model=AssessmentResponse)
async def assess_flood(request: AssessmentRequest):
    """
    Run flood assessment.

    - demo mode: Uses WorldFloods test images
    - live mode: Fetches real-time Sentinel-2 imagery from Copernicus
    """
    try:
        engine = get_inference_engine()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading error: {str(e)}")

    if request.mode == "demo":
        return await _assess_demo(engine, request)
    elif request.mode == "live":
        return await _assess_live(engine, request)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}. Use 'demo' or 'live'."
        )


async def _assess_demo(engine, request: AssessmentRequest) -> AssessmentResponse:
    """Demo assessment using archived WorldFloods images."""
    demo_images = _get_demo_images()

    if not demo_images:
        raise HTTPException(
            status_code=404,
            detail="No demo images available. Please download the WorldFloods dataset first."
        )

    # Select image
    if request.demo_image_index is not None:
        idx = request.demo_image_index % len(demo_images)
    else:
        idx = int(abs(request.latitude * 100 + request.longitude * 10)) % len(demo_images)

    image_path = str(demo_images[idx])
    result = engine.predict(image_path, generate_gradcam=True)

    image_b64 = None
    gradcam_b64 = None
    if "gradcam" in result:
        image_b64 = _numpy_to_base64(result["gradcam"]["original_resized"])
        gradcam_b64 = _numpy_to_base64(result["gradcam"]["overlay"])

    return AssessmentResponse(
        status="success",
        mode="demo",
        location={
            "latitude": request.latitude,
            "longitude": request.longitude,
            "note": "Demo mode — using archived WorldFloods satellite image",
            "demo_image": demo_images[idx].stem,
        },
        prediction=result["prediction"],
        risk=result["risk"],
        model_info=result["model_info"],
        image_base64=image_b64,
        gradcam_base64=gradcam_b64,
        disclaimer=result["disclaimer"],
    )


async def _assess_live(engine, request: AssessmentRequest) -> AssessmentResponse:
    """Live assessment using real-time Sentinel-2 imagery from Copernicus."""
    service = get_sentinel_service()

    if not service.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Live satellite mode requires Copernicus credentials. "
                   "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in backend/.env. "
                   "Register at https://dataspace.copernicus.eu/"
        )

    lat, lng = request.latitude, request.longitude

    try:
        # Fetch live 6-band Sentinel-2 image
        logger.info(f"Fetching live Sentinel-2 data for ({lat}, {lng})")
        image_array, metadata = service.fetch_image(
            lat=lat, lng=lng,
            size_km=5.0,
            resolution=10,
            days_back=60,
        )

        # Check if we got usable data
        if metadata.get("non_zero_pct", 0) < 10:
            raise HTTPException(
                status_code=422,
                detail=f"Satellite image is {100 - metadata.get('non_zero_pct', 0):.0f}% cloudy or empty. "
                       "Try again later or use demo mode."
            )

        # Save as temporary GeoTIFF for the inference engine
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
            _save_array_as_tiff(image_array, tmp_path)

        # Run inference on the live image
        try:
            result = engine.predict(tmp_path, generate_gradcam=True)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

        # Convert images to base64
        image_b64 = None
        gradcam_b64 = None
        if "gradcam" in result:
            image_b64 = _numpy_to_base64(result["gradcam"]["original_resized"])
            gradcam_b64 = _numpy_to_base64(result["gradcam"]["overlay"])

        return AssessmentResponse(
            status="success",
            mode="live",
            location={
                "latitude": lat,
                "longitude": lng,
                "note": "Live Sentinel-2 satellite imagery from Copernicus Data Space",
                "satellite_data": {
                    "source": metadata.get("source", "Sentinel-2 L2A"),
                    "date_range": metadata.get("date_range", ""),
                    "resolution": f"{metadata.get('resolution_m', 10)}m",
                    "coverage": f"{metadata.get('non_zero_pct', 0)}% clear",
                    "bands": metadata.get("bands", []),
                },
            },
            prediction=result["prediction"],
            risk=result["risk"],
            model_info=result["model_info"],
            image_base64=image_b64,
            gradcam_base64=gradcam_b64,
            disclaimer="Live satellite assessment using Copernicus Sentinel-2 L2A imagery. "
                       "Results depend on image quality, cloud cover, and date of acquisition. "
                       "This is an AI prediction and should not be used as the sole basis for emergency decisions.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live satellite fetch failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch satellite data: {str(e)}. "
                   "The Copernicus service may be temporarily unavailable. Try demo mode."
        )


@router.get("/model-status")
async def model_status():
    """Check if the flood model is loaded and ready."""
    try:
        engine = get_inference_engine()

        # Also check satellite service
        sat_status = "not_configured"
        try:
            service = get_sentinel_service()
            if service.is_configured:
                sat_status = "configured"
        except:
            pass

        return {
            "status": "ready",
            "model_info": engine.checkpoint_info,
            "device": str(engine.device),
            "satellite": sat_status,
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "error": str(e),
            "satellite": "unknown",
        }
