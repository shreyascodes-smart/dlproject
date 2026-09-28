"""
EfficientNet-B0 Flood Classifier
Transfer learning model for binary flood/non-flood classification.
"""
import torch
import torch.nn as nn
from torchvision import models


def create_flood_classifier(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create an EfficientNet-B0 model for flood classification.
    
    What this does:
        1. Loads EfficientNet-B0 pretrained on ImageNet
        2. Replaces the classifier head for our binary classification task
        3. Keeps the backbone frozen initially (transfer learning)
    
    Why EfficientNet-B0:
        - Efficient architecture (good accuracy/speed tradeoff)
        - Works well with transfer learning on small datasets
        - Designed for 224x224 input (matches our target size)
        - ~5.3M parameters (fits in 6GB VRAM easily)
    
    Args:
        num_classes: Number of output classes (2 for flood/non-flood)
        pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        EfficientNet-B0 model with modified classifier head
    """
    # Load pretrained EfficientNet-B0
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Get the number of features in the original classifier
    num_features = model.classifier[1].in_features
    
    # Replace classifier head for our task
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, num_classes),
    )
    
    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze the backbone (feature extraction layers) for transfer learning.
    Only the classifier head will be trained initially.
    
    Why:
        - The pretrained features are already good at recognizing patterns
        - Training only the head is faster and requires less data
        - We can unfreeze later for fine-tuning if needed
    """
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module, from_layer: int = 5) -> None:
    """
    Gradually unfreeze backbone layers for fine-tuning.
    
    Args:
        from_layer: Unfreeze layers from this index onwards (0-7 for EfficientNet-B0)
                    Higher = unfreeze fewer layers (more conservative)
    """
    for i, block in enumerate(model.features):
        if i >= from_layer:
            for param in block.parameters():
                param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0,
    }


if __name__ == "__main__":
    # Quick model verification
    print("Creating EfficientNet-B0 Flood Classifier...")
    model = create_flood_classifier(num_classes=2, pretrained=True)
    
    params = count_parameters(model)
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
    
    # Freeze backbone
    freeze_backbone(model)
    params = count_parameters(model)
    print(f"\nAfter freezing backbone:")
    print(f"  Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output: {output}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\nCUDA forward pass: OK")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory used: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    
    print("\nModel verification: PASS")
