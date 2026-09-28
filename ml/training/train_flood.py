"""
Flood Model Training Script
Trains EfficientNet-B0 for binary flood/non-flood classification.

Usage:
    python ml/training/train_flood.py

Training follows the mandatory test order:
    1. Dataset loads correctly
    2. One batch loads
    3. One forward pass works
    4. One training step works
    5. Short training smoke test
    6. Real training
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.datasets.flood_dataset import WorldFloodsClassificationDataset
from ml.models.flood_classifier import (
    create_flood_classifier,
    freeze_backbone,
    unfreeze_backbone,
    count_parameters,
)


# ─── Configuration ───────────────────────────────────────────────────────────

class TrainConfig:
    """Training configuration — all settings in one place."""
    
    # Paths
    data_dir = str(project_root / "data" / "raw" / "flood" / "worldfloods_v2")
    checkpoint_dir = str(project_root / "models" / "flood")
    log_dir = str(project_root / "data" / "reports")
    
    # Model
    num_classes = 2
    pretrained = True
    
    # Training
    batch_size = 8          # Conservative for 6GB VRAM
    num_workers = 2         # Conservative for Windows
    learning_rate = 0.001   # Starting point for Adam
    num_epochs = 15         # Sufficient for transfer learning
    
    # Transfer learning
    freeze_backbone_epochs = 5  # Train only head for first N epochs
    
    # Image
    target_size = 224       # EfficientNet-B0 standard input
    rgb_bands = (3, 2, 1)   # Default Sentinel-2 RGB (B4,B3,B2 are 0-indexed 3,2,1)
    
    # Label derivation
    flood_threshold = 0.05  # 5% flood pixels = "flood" label
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Training Functions ─────────────────────────────────────────────────────

def get_transforms(split: str):
    """Get data transforms for a given split."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Normalize using ImageNet statistics (since we use pretrained weights)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def create_dataloaders(config: TrainConfig):
    """Create train, val, and test dataloaders."""
    loaders = {}
    
    for split in ["train", "val", "test"]:
        try:
            dataset = WorldFloodsClassificationDataset(
                data_dir=config.data_dir,
                split=split,
                target_size=config.target_size,
                flood_threshold=config.flood_threshold,
                rgb_bands=config.rgb_bands,
                transform=get_transforms(split),
            )
            
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=(split == "train"),
                num_workers=config.num_workers,
                pin_memory=(config.device == "cuda"),
            )
            
            loaders[split] = loader
            print(f"  {split}: {len(dataset)} samples, {len(loader)} batches")
        except FileNotFoundError as e:
            print(f"  {split}: SKIPPED ({e})")
    
    return loaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (loss, accuracy, predictions, labels)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Flood probability
    
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_acc,
        "timestamp": datetime.now().isoformat(),
    }, path)
    print(f"  Checkpoint saved: {path}")


# ─── Smoke Tests ─────────────────────────────────────────────────────────────

def run_smoke_tests(config: TrainConfig):
    """
    Run all mandatory smoke tests before real training.
    Returns True if all tests pass.
    """
    print("\n" + "=" * 60)
    print("SMOKE TESTS — Mandatory Before Training")
    print("=" * 60)
    
    device = torch.device(config.device)
    
    # Test 1: Dataset loads
    print("\n[Test 1] Dataset loads...")
    try:
        loaders = create_dataloaders(config)
        if "train" not in loaders:
            print("  FAIL: No train split")
            return False, None, None
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False, None, None
    
    # Test 2: One batch loads
    print("\n[Test 2] One batch loads...")
    try:
        batch = next(iter(loaders["train"]))
        images, labels = batch
        print(f"  Images: {images.shape}, Labels: {labels.shape}")
        print(f"  Label values: {labels.tolist()}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False, None, None
    
    # Test 3: Model creation
    print("\n[Test 3] Model creation...")
    try:
        model = create_flood_classifier(num_classes=config.num_classes, pretrained=config.pretrained)
        freeze_backbone(model)
        model = model.to(device)
        params = count_parameters(model)
        print(f"  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False, None, None
    
    # Test 4: Forward pass
    print("\n[Test 4] Forward pass...")
    try:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        print(f"  Output shape: {outputs.shape}")
        print(f"  Output sample: {outputs[0].tolist()}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False, None, None
    
    # Test 5: Loss + backward pass
    print("\n[Test 5] Loss + backward pass...")
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        model.train()
        outputs = model(images)
        labels = labels.to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"  Loss: {loss.item():.4f}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False, None, None
    
    # Test 6: Checkpoint save/load
    print("\n[Test 6] Checkpoint save/load...")
    try:
        test_ckpt = Path(config.checkpoint_dir) / "smoke_test.pt"
        save_checkpoint(model, optimizer, 0, 0.0, str(test_ckpt))
        
        # Load it back
        ckpt = torch.load(str(test_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        
        # Clean up test checkpoint
        test_ckpt.unlink()
        
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False, None, None
    
    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED [OK]")
    print("=" * 60)
    
    return True, model, loaders


# ─── Main Training Loop ─────────────────────────────────────────────────────

def train(config: TrainConfig = None):
    """Main training function."""
    if config is None:
        config = TrainConfig()
    
    device = torch.device(config.device)
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Run smoke tests
    passed, model, loaders = run_smoke_tests(config)
    if not passed:
        print("\nSMOKE TESTS FAILED — cannot proceed with training.")
        return
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    best_val_acc = 0.0
    training_log = []
    
    print(f"\n{'='*60}")
    print(f"TRAINING — {config.num_epochs} epochs")
    print(f"{'='*60}")
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        
        # Unfreeze backbone after initial frozen epochs
        if epoch == config.freeze_backbone_epochs + 1:
            print(f"\n--- Unfreezing backbone from layer 5 ---")
            unfreeze_backbone(model, from_layer=5)
            # Lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate * 0.1
            params = count_parameters(model)
            print(f"  Trainable parameters: {params['trainable']:,}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = 0.0, 0.0
        if "val" in loaders:
            val_loss, val_acc, _, _, _ = evaluate(
                model, loaders["val"], criterion, device
            )
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_s": epoch_time,
        }
        training_log.append(log_entry)
        
        print(f"Epoch {epoch:2d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                str(Path(config.checkpoint_dir) / "best_flood_model.pt")
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, config.num_epochs, val_acc,
        str(Path(config.checkpoint_dir) / "final_flood_model.pt")
    )
    
    # Test evaluation
    if "test" in loaders:
        print(f"\n{'='*60}")
        print("TEST EVALUATION")
        print(f"{'='*60}")
        
        # Load best model for test
        best_ckpt = torch.load(
            str(Path(config.checkpoint_dir) / "best_flood_model.pt"),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_ckpt["model_state_dict"])
        
        test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
            model, loaders["test"], criterion, device
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test samples: {len(test_labels)}")
        print(f"  Flood predictions: {(test_preds == 1).sum()}")
        print(f"  Non-flood predictions: {(test_preds == 0).sum()}")
    
    # Save training log
    log_path = Path(config.log_dir) / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\nTraining log saved: {log_path}")
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print("Training complete!")


if __name__ == "__main__":
    train()
