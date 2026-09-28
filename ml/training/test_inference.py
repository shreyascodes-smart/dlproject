"""Quick test of inference engine with trained model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.inference.flood_inference import FloodInference

engine = FloodInference(r"d:\superiorparab\final year DL project\models\flood\best_flood_model.pt")

# Test on real images
test_dir = Path(r"d:\superiorparab\final year DL project\data\raw\flood\worldfloods_v2\test\S2")
images = sorted(test_dir.glob("*.tif"))
print(f"Found {len(images)} test images\n")

for img in images[:5]:
    result = engine.predict(str(img), generate_gradcam=True)
    p = result["prediction"]
    r = result["risk"]
    has_cam = "gradcam" in result
    print(f"  {img.stem}")
    print(f"    Class: {p['class_name']}, Flood prob: {p['flood_probability']:.3f}")
    print(f"    Risk: {r['label']}, Grad-CAM: {'yes' if has_cam else 'no'}")
    print()

# Save a Grad-CAM for the first image
result = engine.predict(str(images[0]), generate_gradcam=True)
if "gradcam" in result:
    from ml.explainability.gradcam import GradCAM
    save_path = r"d:\superiorparab\final year DL project\data\reports\trained_gradcam.png"
    GradCAM.save_visualization(
        result["gradcam"]["original_resized"],
        result["gradcam"]["heatmap"],
        result["prediction"],
        save_path,
    )
    print(f"Grad-CAM saved: {save_path}")

print("\nInference test PASSED!")
