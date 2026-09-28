"""
Flood Assessment API Routes
Provides endpoints for the React frontend to trigger flood assessments.
"""
import os
import sys
import io
import base64
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt


# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


router = APIRouter(prefix="/api/flood", tags=["flood"])

# Global inference engine (loaded once)
_inference_engine = None


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
            # Try final model
            checkpoint_path = project_root / "models" / "flood" / "final_flood_model.pt"
        
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=503,
                detail="No trained flood model found. Please train the model first."
            )
        
        _inference_engine = FloodInference(str(checkpoint_path))
    
    return _inference_engine


def _numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert a numpy image array to base64-encoded PNG."""
    img_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode("utf-8")


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


@router.post("/assess", response_model=AssessmentResponse)
async def assess_flood(request: AssessmentRequest):
    """
    Run flood assessment.
    
    In demo mode: Uses a real test image from WorldFloods dataset.
    In live mode: Would fetch satellite imagery (future implementation).
    """
    try:
        engine = get_inference_engine()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading error: {str(e)}")
    
    if request.mode == "demo":
        # Demo mode: use real WorldFloods test images
        demo_images = _get_demo_images()
        
        if not demo_images:
            raise HTTPException(
                status_code=404,
                detail="No demo images available. Please download the WorldFloods dataset first."
            )
        
        # Select image (by index or semi-random based on coordinates)
        if request.demo_image_index is not None:
            idx = request.demo_image_index % len(demo_images)
        else:
            # Use coordinates to deterministically select an image
            idx = int(abs(request.latitude * 100 + request.longitude * 10)) % len(demo_images)
        
        image_path = str(demo_images[idx])
        
        # Run inference
        result = engine.predict(image_path, generate_gradcam=True)
        
        # Convert images to base64 for frontend
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
                "note": "Demo mode — using WorldFloods test image, not live satellite data",
                "demo_image": demo_images[idx].stem,
            },
            prediction=result["prediction"],
            risk=result["risk"],
            model_info=result["model_info"],
            image_base64=image_b64,
            gradcam_base64=gradcam_b64,
            disclaimer=result["disclaimer"],
        )
    
    elif request.mode == "live":
        # Live mode: would integrate with Sentinel Hub
        raise HTTPException(
            status_code=501,
            detail="Live satellite mode not yet implemented. Use demo mode."
        )
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}. Use 'demo' or 'live'."
        )


@router.get("/model-status")
async def model_status():
    """Check if the flood model is loaded and ready."""
    try:
        engine = get_inference_engine()
        return {
            "status": "ready",
            "model_info": engine.checkpoint_info,
            "device": str(engine.device),
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "error": str(e),
        }
