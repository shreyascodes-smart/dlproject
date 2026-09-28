"""
Flood Model Inference Module
Loads a trained checkpoint and runs inference on new images.

This module provides:
    1. Model loading from checkpoint
    2. Image preprocessing (same as training)
    3. Flood probability prediction
    4. Risk interpretation
    5. Grad-CAM generation
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import rasterio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.models.flood_classifier import create_flood_classifier
from ml.explainability.gradcam import GradCAM


# Risk interpretation thresholds
# These are application-level interpretations, NOT official warnings
RISK_LEVELS = [
    {"min": 0.0, "max": 0.2, "level": "low",      "label": "Low Risk",      "color": "#4CAF50"},
    {"min": 0.2, "max": 0.5, "level": "moderate",  "label": "Moderate Risk", "color": "#FF9800"},
    {"min": 0.5, "max": 0.8, "level": "high",      "label": "High Risk",     "color": "#F44336"},
    {"min": 0.8, "max": 1.0, "level": "very_high", "label": "Very High Risk", "color": "#B71C1C"},
]


class FloodInference:
    """
    Runs flood assessment on images using a trained EfficientNet-B0 model.
    
    Usage:
        engine = FloodInference("models/flood/best_flood_model.pt")
        result = engine.predict("path/to/image.tif")
    """
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint (.pt)
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Load model
        self.model = create_flood_classifier(num_classes=2, pretrained=False)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create Grad-CAM
        self.gradcam = GradCAM(self.model)
        
        # Preprocessing (same as training — ImageNet normalization)
        self.preprocess = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        self.checkpoint_info = {
            "epoch": checkpoint.get("epoch", "unknown"),
            "val_accuracy": checkpoint.get("val_accuracy", "unknown"),
            "timestamp": checkpoint.get("timestamp", "unknown"),
        }
        
        print(f"Flood model loaded from: {checkpoint_path}")
        print(f"  Epoch: {self.checkpoint_info['epoch']}")
        print(f"  Val accuracy: {self.checkpoint_info['val_accuracy']}")
        print(f"  Device: {self.device}")
    
    def _load_image(self, image_path: str, rgb_bands=(3, 2, 1)) -> np.ndarray:
        """
        Load and preprocess an image (TIFF or regular image).
        
        Uses windowed reads for large GeoTIFFs to avoid loading 1GB+ files.
        
        Returns:
            numpy array (H, W, 3) in range [0, 1]
        """
        path = Path(image_path)
        
        if path.suffix.lower() in ['.tif', '.tiff']:
            # GeoTIFF (Sentinel-2 data) — use windowed read
            from rasterio.windows import Window
            with rasterio.open(image_path) as src:
                h, w = src.height, src.width
                crop_size = min(h, w, 512)
                row_off = max(0, (h - crop_size) // 2)
                col_off = max(0, (w - crop_size) // 2)
                window = Window(col_off, row_off, crop_size, crop_size)
                
                bands = []
                for band_idx in rgb_bands:
                    band = src.read(band_idx + 1, window=window).astype(np.float32)
                    bands.append(band)
                rgb = np.stack(bands, axis=-1)
                rgb = np.clip(rgb, 0, None)
                rgb = np.clip(rgb / 10000.0, 0.0, 1.0)
        else:
            # Regular image (JPEG, PNG)
            img = Image.open(image_path).convert('RGB')
            rgb = np.array(img).astype(np.float32) / 255.0
        
        return rgb.astype(np.float32)
    
    def _prepare_tensor(self, image: np.ndarray, target_size: int = 224) -> torch.Tensor:
        """Resize and convert to tensor."""
        # Resize
        img_uint8 = (image * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        resized = np.array(pil_img).astype(np.float32) / 255.0
        
        # To tensor (HWC -> CHW)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        
        # Normalize
        tensor = self.preprocess(tensor)
        
        return tensor
    
    def _interpret_risk(self, flood_probability: float) -> Dict[str, Any]:
        """Convert flood probability to risk interpretation."""
        for level in RISK_LEVELS:
            if level["min"] <= flood_probability < level["max"] or \
               (level["max"] == 1.0 and flood_probability == 1.0):
                return {
                    "level": level["level"],
                    "label": level["label"],
                    "color": level["color"],
                    "probability": flood_probability,
                }
        
        return {"level": "unknown", "label": "Unknown", "color": "#999", "probability": flood_probability}
    
    def predict(
        self,
        image_path: str,
        generate_gradcam: bool = True,
        rgb_bands: tuple = (3, 2, 1),
    ) -> Dict[str, Any]:
        """
        Run flood assessment on an image.
        
        Args:
            image_path: Path to the image file
            generate_gradcam: Whether to generate Grad-CAM heatmap
            rgb_bands: Band indices for TIFF files
        
        Returns:
            dict with prediction results, risk interpretation, and optional Grad-CAM
        """
        # Load image
        original_image = self._load_image(image_path, rgb_bands)
        
        # Prepare tensor
        input_tensor = self._prepare_tensor(original_image)
        input_tensor = input_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
        
        flood_prob = probs[0, 1].item()
        predicted_class = output.argmax(dim=1).item()
        
        result = {
            "image_path": str(image_path),
            "prediction": {
                "class": predicted_class,
                "class_name": "flood" if predicted_class == 1 else "non_flood",
                "flood_probability": flood_prob,
                "non_flood_probability": probs[0, 0].item(),
                "raw_output": output[0].detach().cpu().numpy().tolist(),
            },
            "risk": self._interpret_risk(flood_prob),
            "model_info": self.checkpoint_info,
            "disclaimer": "This is an AI-based assessment tool, not an official warning system.",
        }
        
        # Generate Grad-CAM
        if generate_gradcam:
            # Need to re-run with gradients for Grad-CAM
            input_for_cam = self._prepare_tensor(original_image).to(self.device)
            heatmap, _ = self.gradcam.generate(input_for_cam, target_class=predicted_class)
            
            # Resize original for visualization
            pil_img = Image.fromarray((original_image * 255).astype(np.uint8))
            pil_img = pil_img.resize((224, 224), Image.BILINEAR)
            resized_original = np.array(pil_img).astype(np.float32) / 255.0
            
            overlay = GradCAM.create_overlay(resized_original, heatmap)
            
            result["gradcam"] = {
                "heatmap": heatmap,
                "overlay": overlay,
                "original_resized": resized_original,
            }
        
        return result


if __name__ == "__main__":
    print("FloodInference module loaded successfully.")
    print("Usage: engine = FloodInference('models/flood/best_flood_model.pt')")
    print("       result = engine.predict('path/to/image.tif')")
