"""
End-to-end smoke test of the flood pipeline.
Tests: dataset loading -> model forward pass -> Grad-CAM -> inference
Uses the test split (fully downloaded) to validate everything works.
"""
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.datasets.flood_dataset import WorldFloodsClassificationDataset
from ml.models.flood_classifier import create_flood_classifier, freeze_backbone, count_parameters
from ml.explainability.gradcam import GradCAM
from torchvision import transforms


def main():
    data_dir = r"d:\superiorparab\final year DL project\data\raw\flood\worldfloods_v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("END-TO-END SMOKE TEST")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # 1. Dataset Loading
    print("\n[1/6] Dataset Loading...")
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_ds = WorldFloodsClassificationDataset(
        data_dir, split="test", transform=test_transform
    )
    print(f"  Test dataset: {len(test_ds)} samples")
    
    # 2. DataLoader
    print("\n[2/6] DataLoader Test...")
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(test_loader))
    images, labels = batch
    print(f"  Batch images: {images.shape}")
    print(f"  Batch labels: {labels.tolist()}")
    
    # 3. Model Creation
    print("\n[3/6] Model Creation...")
    model = create_flood_classifier(num_classes=2, pretrained=True)
    freeze_backbone(model)
    model = model.to(device)
    params = count_parameters(model)
    print(f"  Total params: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    
    # 4. Forward Pass
    print("\n[4/6] Forward Pass...")
    images_gpu = images.to(device)
    with torch.no_grad():
        output = model(images_gpu)
        probs = torch.softmax(output, dim=1)
    print(f"  Output shape: {output.shape}")
    print(f"  Probabilities:")
    for i in range(len(labels)):
        name = test_ds.samples[i]["name"]
        print(f"    {name}: flood={probs[i,1]:.3f} non_flood={probs[i,0]:.3f} | GT={labels[i].item()}")
    
    # 5. Training Step
    print("\n[5/6] Training Step...")
    import torch.nn as nn
    import torch.optim as optim
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    labels_gpu = labels.to(device)
    output = model(images_gpu)
    loss = criterion(output, labels_gpu)
    loss.backward()
    optimizer.step()
    print(f"  Loss: {loss.item():.4f}")
    
    # 6. Grad-CAM
    print("\n[6/6] Grad-CAM...")
    model.eval()
    gradcam = GradCAM(model)
    
    # Use first image
    single_image = images[0:1].to(device)
    heatmap, prediction = gradcam.generate(single_image)
    print(f"  Heatmap shape: {heatmap.shape}")
    print(f"  Prediction: {prediction['class_name']} ({prediction['flood_probability']:.3f})")
    
    # Save visualization
    save_dir = Path(r"d:\superiorparab\final year DL project\data\reports")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get original image (without normalization) for visualization
    raw_ds = WorldFloodsClassificationDataset(data_dir, split="test")
    raw_image, _ = raw_ds[0]
    original = raw_image.permute(1, 2, 0).numpy()  # CHW -> HWC
    
    GradCAM.save_visualization(
        original, heatmap, prediction,
        str(save_dir / "smoke_test_gradcam.png")
    )
    
    # GPU memory usage
    if device.type == "cuda":
        print(f"\n  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  GPU memory peak: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    
    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    main()
