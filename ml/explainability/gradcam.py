"""
Grad-CAM for EfficientNet-B0 Flood Classifier
Generates heatmap overlays showing which image regions influenced the prediction.

What Grad-CAM does:
    - Takes a trained model and an input image
    - Hooks into the last convolutional layer
    - Captures gradients flowing back from the predicted class
    - Produces a heatmap highlighting important regions
    
What it does NOT do:
    - It does NOT prove the prediction is correct
    - It does NOT generate the prediction itself
    - It is an explanation aid, not a ground truth
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B0.
    
    Usage:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate(image_tensor)
        overlay = gradcam.overlay(original_image, heatmap)
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: Trained EfficientNet-B0 model
            target_layer: Layer to hook for Grad-CAM (default: last feature block)
        """
        self.model = model
        self.model.eval()
        
        # Default target: last block of EfficientNet features
        if target_layer is None:
            self.target_layer = model.features[-1]
        else:
            self.target_layer = target_layer
        
        # Storage for hooks
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, image_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for an input image.
        
        Args:
            image_tensor: Input image tensor (1, 3, H, W) or (3, H, W)
            target_class: Class to generate heatmap for (default: predicted class)
        
        Returns:
            heatmap: numpy array (H, W) in range [0, 1]
            prediction: dict with class, probability, and raw output
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(image_tensor)
        probs = F.softmax(output, dim=1)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Generate heatmap
        # Global average pool the gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU — we only want positive influences
        cam = F.relu(cam)
        
        # Resize to input image size
        cam = F.interpolate(cam, size=(image_tensor.shape[2], image_tensor.shape[3]),
                           mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        prediction = {
            "class": target_class,
            "class_name": "flood" if target_class == 1 else "non_flood",
            "probability": probs[0, target_class].item(),
            "flood_probability": probs[0, 1].item(),
            "raw_output": output[0].detach().cpu().numpy().tolist(),
        }
        
        return cam, prediction
    
    @staticmethod
    def create_overlay(original_image, heatmap, alpha=0.4):
        """
        Create a colored heatmap overlay on the original image.
        
        Args:
            original_image: numpy array (H, W, 3) in range [0, 1]
            heatmap: numpy array (H, W) in range [0, 1]
            alpha: Overlay transparency (0=image only, 1=heatmap only)
        
        Returns:
            overlay: numpy array (H, W, 3) in range [0, 1]
        """
        # Apply colormap to heatmap
        colormap = cm.jet(heatmap)[:, :, :3]  # Drop alpha channel
        
        # Resize heatmap to match image if needed
        if colormap.shape[:2] != original_image.shape[:2]:
            heatmap_pil = Image.fromarray((colormap * 255).astype(np.uint8))
            heatmap_pil = heatmap_pil.resize(
                (original_image.shape[1], original_image.shape[0]),
                Image.BILINEAR
            )
            colormap = np.array(heatmap_pil).astype(np.float32) / 255.0
        
        # Blend
        overlay = (1 - alpha) * original_image + alpha * colormap
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    @staticmethod
    def save_visualization(original_image, heatmap, prediction, save_path):
        """
        Save a complete Grad-CAM visualization with original, heatmap, and overlay.
        
        Args:
            original_image: numpy array (H, W, 3) in range [0, 1]
            heatmap: numpy array (H, W) in range [0, 1]
            prediction: dict from generate()
            save_path: Path to save the visualization
        """
        overlay = GradCAM.create_overlay(original_image, heatmap)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Heatmap
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Overlay — {prediction['class_name'].upper()}\n"
            f"Flood prob: {prediction['flood_probability']:.1%}"
        )
        axes[2].axis("off")
        
        plt.suptitle("Grad-CAM: Regions that influenced the model's prediction",
                     fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        print(f"Grad-CAM visualization saved: {save_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
    from ml.models.flood_classifier import create_flood_classifier, freeze_backbone
    
    print("Testing Grad-CAM with dummy data...")
    
    # Create model
    model = create_flood_classifier(num_classes=2, pretrained=True)
    model.eval()
    
    # Create Grad-CAM
    gradcam = GradCAM(model)
    
    # Test with random image
    dummy = torch.randn(1, 3, 224, 224)
    heatmap, prediction = gradcam.generate(dummy)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"Prediction: {prediction}")
    
    print("\nGrad-CAM test: PASS")
