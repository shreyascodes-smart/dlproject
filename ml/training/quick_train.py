"""
Quick Training Script — Uses available data to produce a working checkpoint ASAP.
If train/gt is not available yet, trains on test+val splits.
Once full train data arrives, run the full training script instead.
"""
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.datasets.flood_dataset import WorldFloodsClassificationDataset
from ml.models.flood_classifier import (
    create_flood_classifier, freeze_backbone, unfreeze_backbone, count_parameters
)


def main():
    data_dir = r"d:\superiorparab\final year DL project\data\raw\flood\worldfloods_v2"
    checkpoint_dir = Path(r"d:\superiorparab\final year DL project\models\flood")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Check which splits are available
    train_gt_dir = Path(data_dir) / "train" / "gt"
    has_train = train_gt_dir.exists() and len(list(train_gt_dir.glob("*.tif"))) > 0
    
    if has_train:
        print("\n--- Full training mode (train split available) ---")
        train_ds = WorldFloodsClassificationDataset(data_dir, split="train", transform=train_transform)
        val_ds = WorldFloodsClassificationDataset(data_dir, split="val", transform=val_transform)
        test_ds = WorldFloodsClassificationDataset(data_dir, split="test", transform=val_transform)
        epochs = 15
    else:
        print("\n--- Quick training mode (train/gt not yet downloaded) ---")
        print("  Using test+val as training data to produce a working checkpoint")
        # Use test as training data, val as validation
        train_ds = WorldFloodsClassificationDataset(data_dir, split="test", transform=train_transform)
        val_ds = WorldFloodsClassificationDataset(data_dir, split="val", transform=val_transform)
        test_ds = None
        epochs = 30  # More epochs since less data
    
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model
    model = create_flood_classifier(num_classes=2, pretrained=True)
    freeze_backbone(model)
    model = model.to(device)
    
    params = count_parameters(model)
    print(f"\n  Model params: {params['total']:,} total, {params['trainable']:,} trainable")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    training_log = []
    
    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        
        # Unfreeze backbone after 5 epochs
        if epoch == 6:
            print(f"\n--- Unfreezing backbone ---")
            unfreeze_backbone(model, from_layer=5)
            for pg in optimizer.param_groups:
                pg['lr'] = 0.0001
            params = count_parameters(model)
            print(f"  Trainable: {params['trainable']:,}")
        
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        train_loss = train_loss / train_total
        
        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss = val_loss / val_total if val_total > 0 else 0
        
        elapsed = time.time() - start
        
        log_entry = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                     "val_loss": val_loss, "val_acc": val_acc, "time": elapsed}
        training_log.append(log_entry)
        
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")
        
        # Save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "mode": "full" if has_train else "quick",
            }, str(checkpoint_dir / "best_flood_model.pt"))
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
    
    # Save final
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_acc,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "mode": "full" if has_train else "quick",
    }, str(checkpoint_dir / "final_flood_model.pt"))
    
    # Test evaluation
    if test_ds:
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
        # Load best
        ckpt = torch.load(str(checkpoint_dir / "best_flood_model.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        print(f"\nTest Accuracy: {correct/total:.4f} ({correct}/{total})")
    
    # Save log
    log_path = Path(r"d:\superiorparab\final year DL project\data\reports")
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path / "training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
