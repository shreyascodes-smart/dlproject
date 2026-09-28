"""
Full Training Script with Proper Split & Evaluation Metrics
- Combines all available data into 70/30 train/test split (stratified)
- Class weighting for imbalanced dataset (96% flood)
- Comprehensive evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix
"""
import os, sys, time, json
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np


class TransformSubset(torch.utils.data.Dataset):
    """Wrapper to apply transforms to a Subset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.datasets.flood_dataset import WorldFloodsClassificationDataset
from ml.models.flood_classifier import (
    create_flood_classifier, freeze_backbone, unfreeze_backbone, count_parameters,
)


def stratified_split(dataset, test_ratio=0.30, seed=42):
    """70/30 stratified split ensuring both classes in both sets."""
    np.random.seed(seed)
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label.item())
    labels = np.array(labels)

    flood_idx = np.where(labels == 1)[0]
    non_flood_idx = np.where(labels == 0)[0]
    np.random.shuffle(flood_idx)
    np.random.shuffle(non_flood_idx)

    n_flood_test = max(1, int(len(flood_idx) * test_ratio))
    n_nf_test = max(1, int(len(non_flood_idx) * test_ratio))

    test_idx = np.concatenate([flood_idx[:n_flood_test], non_flood_idx[:n_nf_test]])
    train_idx = np.concatenate([flood_idx[n_flood_test:], non_flood_idx[n_nf_test:]])
    np.random.shuffle(test_idx)
    np.random.shuffle(train_idx)

    return train_idx.tolist(), test_idx.tolist(), labels


def compute_class_weights(labels, train_idx):
    """Compute inverse frequency weights for CrossEntropyLoss."""
    train_labels = labels[train_idx]
    counts = Counter(train_labels.tolist())
    total = sum(counts.values())
    weights = []
    for c in sorted(counts.keys()):
        weights.append(total / (len(counts) * counts[c]))
    return torch.FloatTensor(weights)


def compute_metrics(preds, labels, probs):
    """Compute full classification metrics."""
    preds = np.array(preds)
    labels = np.array(labels)
    probs = np.array(probs)

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "total_samples": len(labels),
        "flood_count": int((labels == 1).sum()),
        "non_flood_count": int((labels == 0).sum()),
    }


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels, all_probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_dir = str(project_root / "data" / "raw" / "flood" / "worldfloods_v2")
    ckpt_dir = project_root / "models" / "flood"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ALL data from all splits into one combined dataset ---
    print("\n--- Loading all splits ---")
    all_datasets = []
    for split in ["train", "val", "test"]:
        try:
            ds = WorldFloodsClassificationDataset(
                data_dir, split=split, target_size=224, flood_threshold=0.05,
            )
            all_datasets.append(ds)
            print(f"  {split}: {len(ds)} samples")
        except Exception as e:
            print(f"  {split}: skipped ({e})")

    # Combine into single dataset
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(all_datasets)
    print(f"\nCombined: {len(combined)} total samples")

    # --- 70/30 stratified split ---
    print("\n--- 70/30 Stratified Split ---")
    train_idx, test_idx, labels = stratified_split(combined, test_ratio=0.30)
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")

    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    print(f"  Train distribution: flood={int((train_labels==1).sum())}, non-flood={int((train_labels==0).sum())}")
    print(f"  Test distribution:  flood={int((test_labels==1).sum())}, non-flood={int((test_labels==0).sum())}")

    # --- Class weights for imbalance ---
    class_weights = compute_class_weights(labels, train_idx).to(device)
    print(f"  Class weights: non-flood={class_weights[0]:.2f}, flood={class_weights[1]:.2f}")

    # --- Data transforms ---
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply transforms to subsets
    train_subset = Subset(combined, train_idx)
    test_subset = Subset(combined, test_idx)

    train_ds = TransformSubset(train_subset, train_transform)
    test_ds = TransformSubset(test_subset, test_transform)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    # --- Model ---
    model = create_flood_classifier(num_classes=2, pretrained=True)
    freeze_backbone(model)
    model = model.to(device)
    params = count_parameters(model)
    print(f"\nModel: EfficientNet-B0, {params['total']:,} total, {params['trainable']:,} trainable")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    freeze_epochs = 5
    best_test_f1 = 0.0

    print(f"\n{'='*60}")
    print(f"TRAINING - {num_epochs} epochs, 70/30 split, class-weighted loss")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if epoch == freeze_epochs + 1:
            print("\n--- Unfreezing backbone ---")
            unfreeze_backbone(model, from_layer=5)
            for pg in optimizer.param_groups:
                pg['lr'] = 0.0001
            print(f"  Trainable: {count_parameters(model)['trainable']:,}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, preds, lbls, probs = evaluate(model, test_loader, criterion, device)
        metrics = compute_metrics(preds, lbls, probs)
        dt = time.time() - t0

        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {metrics['f1_score']:.4f} | {dt:.1f}s")

        if metrics['f1_score'] > best_test_f1:
            best_test_f1 = metrics['f1_score']
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": test_acc,
                "f1_score": metrics['f1_score'],
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "split": "70/30 stratified",
                "class_weights": class_weights.cpu().tolist(),
            }, str(ckpt_dir / "best_flood_model.pt"))
            print(f"  -> Saved best model (F1={best_test_f1:.4f})")

    # --- Final model ---
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": test_acc,
        "f1_score": metrics['f1_score'],
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }, str(ckpt_dir / "final_flood_model.pt"))

    # --- Final Evaluation with best model ---
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (Best Model)")
    print(f"{'='*60}")
    best = torch.load(str(ckpt_dir / "best_flood_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    _, _, preds, lbls, probs = evaluate(model, test_loader, criterion, device)
    final_metrics = compute_metrics(preds, lbls, probs)

    print(f"  Accuracy:    {final_metrics['accuracy']:.4f}")
    print(f"  Precision:   {final_metrics['precision']:.4f}")
    print(f"  Recall:      {final_metrics['recall']:.4f}")
    print(f"  F1 Score:    {final_metrics['f1_score']:.4f}")
    print(f"  Specificity: {final_metrics['specificity']:.4f}")
    cm = final_metrics['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Non-Flood  Flood")
    print(f"  Actual Non-Flood   {cm['TN']:4d}    {cm['FP']:4d}")
    print(f"  Actual Flood       {cm['FN']:4d}    {cm['TP']:4d}")
    print(f"\n  Samples: {final_metrics['total_samples']} "
          f"(flood={final_metrics['flood_count']}, non-flood={final_metrics['non_flood_count']})")

    # Save report
    report_path = project_root / "data" / "reports" / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump({
            "split": "70/30 stratified",
            "best_epoch": best["epoch"],
            "metrics": final_metrics,
            "training_config": {
                "epochs": num_epochs,
                "freeze_epochs": freeze_epochs,
                "class_weighted": True,
                "class_weights": best.get("class_weights", []),
            },
        }, f, indent=2)
    print(f"\nReport saved: {report_path}")
    print("Done!")


if __name__ == "__main__":
    main()
