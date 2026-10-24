"""
Flood Segmentation Training Script.

Trains a U-Net with a pretrained encoder on the WorldFloods v2 segmentation
dataset prepared by Phase D.

Usage (DO NOT RUN UNTIL AUTHORIZED):
    python ml/training/train_flood_seg.py

Architecture: U-Net (segmentation_models_pytorch)
Encoder: EfficientNet-B2 (ImageNet pretrained, must be cached locally)
Loss: Weighted CE + Dice (excludes ignored pixels)
Metrics: Flood IoU, Dice/F1, Precision, Recall
"""
import json
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.flood_seg_dataset import (
    FloodSegDataset,
    compute_class_weights,
    IGNORE_INDEX,
    TARGET_FLOOD,
    TARGET_NON_FLOOD,
)

# ============================================================
# Configuration
# ============================================================
DEFAULT_CONFIG = {
    "seed": 42,
    "encoder_name": "efficientnet-b2",
    "encoder_weights": "imagenet",
    "input_size": 256,
    "batch_size": 4,
    "grad_accum_steps": 4,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "patience": 15,
    "num_workers": 2,
    "use_amp": True,
    "dice_weight": 0.5,
    "ce_weight": 0.5,
    "num_classes": 2,
    "data_root": str(PROJECT_ROOT / "data" / "raw" / "flood" / "worldfloods_v2"),
    "masks_dir": str(PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "masks"),
    "manifest_path": str(PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "seg_split_manifest.json"),
    "checkpoint_dir": str(PROJECT_ROOT / "models" / "flood" / "segmentation_v1"),
}


# ============================================================
# Losses
# ============================================================
class MaskedWeightedCELoss(nn.Module):
    """Weighted cross-entropy that ignores pixels with target == IGNORE_INDEX."""

    def __init__(self, class_weights: torch.Tensor, ignore_index: int = 255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) float tensor
            targets: (B, H, W) long tensor with 0, 1, or IGNORE_INDEX
        """
        # Guard: if every pixel is ignored, CE returns NaN. Return 0 instead.
        if (targets != self.ce.ignore_index).sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return self.ce(logits, targets)


class MaskedDiceLoss(nn.Module):
    """Dice loss for the flood class only, ignoring IGNORE_INDEX pixels.

    Handles batches with no flood pixels safely (returns 0 loss).
    """

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 2, H, W) float tensor
            targets: (B, H, W) long tensor
        """
        # Create valid mask
        valid = (targets != self.ignore_index)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Get flood probabilities
        probs = F.softmax(logits, dim=1)
        flood_probs = probs[:, TARGET_FLOOD]  # (B, H, W)

        # Flood targets
        flood_targets = (targets == TARGET_FLOOD).float()

        # Apply valid mask
        flood_probs = flood_probs * valid.float()
        flood_targets = flood_targets * valid.float()

        intersection = (flood_probs * flood_targets).sum()
        union = flood_probs.sum() + flood_targets.sum()

        if union < self.smooth:
            # No flood pixels in batch — return 0 (not NaN)
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Weighted CE + Dice loss."""

    def __init__(self, class_weights: torch.Tensor, ce_weight: float = 0.5,
                 dice_weight: float = 0.5, ignore_index: int = 255):
        super().__init__()
        self.ce_loss = MaskedWeightedCELoss(class_weights, ignore_index)
        self.dice_loss = MaskedDiceLoss(ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


# ============================================================
# Metrics
# ============================================================
class FloodMetrics:
    """Accumulates pixel-level confusion counts for flood segmentation."""

    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update counts from a batch.

        Args:
            logits: (B, 2, H, W) raw logits
            targets: (B, H, W) target labels (0, 1, or ignore)
        """
        preds = logits.argmax(dim=1)  # (B, H, W)
        valid = (targets != self.ignore_index)

        pred_flood = (preds == TARGET_FLOOD) & valid
        pred_nf = (preds == TARGET_NON_FLOOD) & valid
        true_flood = (targets == TARGET_FLOOD) & valid
        true_nf = (targets == TARGET_NON_FLOOD) & valid

        self.tp += (pred_flood & true_flood).sum().item()
        self.fp += (pred_flood & true_nf).sum().item()
        self.fn += (pred_nf & true_flood).sum().item()
        self.tn += (pred_nf & true_nf).sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated counts."""
        eps = 1e-8
        precision = self.tp / max(self.tp + self.fp, eps)
        recall = self.tp / max(self.tp + self.fn, eps)
        iou = self.tp / max(self.tp + self.fp + self.fn, eps)
        dice = 2 * self.tp / max(2 * self.tp + self.fp + self.fn, eps)
        total = self.tp + self.fp + self.fn + self.tn
        accuracy = (self.tp + self.tn) / max(total, eps) if total > 0 else 0.0

        return {
            "flood_iou": iou,
            "flood_dice": dice,
            "flood_precision": precision,
            "flood_recall": recall,
            "pixel_accuracy": accuracy,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


# ============================================================
# Model creation
# ============================================================
def create_segmentation_model(config: dict, device: torch.device):
    """Create U-Net model with pretrained encoder.

    Requires pretrained weights to be cached locally. Will NOT silently
    fall back to random initialization. If weights are unavailable,
    raises RuntimeError with instructions to cache them.
    """
    import segmentation_models_pytorch as smp

    encoder_name = config["encoder_name"]
    encoder_weights = config["encoder_weights"]

    # Prevent automatic downloads — weights must be pre-cached
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=config["num_classes"],
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "offline" in error_msg or "download" in error_msg or "connect" in error_msg:
            raise RuntimeError(
                f"Pretrained weights for '{encoder_name}' are not cached "
                f"locally and cannot be downloaded in offline mode.\n"
                f"To cache weights, run once with internet access:\n"
                f"  python -c \"import timm; timm.create_model('{encoder_name}', pretrained=True)\"\n"
                f"Then re-run training."
            ) from e
        raise

    return model.to(device)


# ============================================================
# Training loop
# ============================================================
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    config, metrics):
    """Train for one epoch with gradient accumulation and AMP."""
    model.train()
    metrics.reset()
    total_loss = 0.0
    n_batches = 0
    accum = config["grad_accum_steps"]

    optimizer.zero_grad()
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        with autocast("cuda", enabled=config["use_amp"]):
            logits = model(images)
            loss = criterion(logits, targets) / accum

        scaler.scale(loss).backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum
        n_batches += 1
        metrics.update(logits.detach(), targets)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device, config, metrics):
    """Validate without augmentation."""
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with autocast("cuda", enabled=config["use_amp"]):
            logits = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item()
        n_batches += 1
        metrics.update(logits, targets)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def main(config: dict = None):
    if config is None:
        config = DEFAULT_CONFIG.copy()

    print("=" * 70)
    print("  FLOOD SEGMENTATION TRAINING")
    print("=" * 70)
    print(f"  Timestamp: {datetime.now().isoformat()}")

    set_seed(config["seed"])

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  Device: CPU (WARNING: training will be slow)")

    print(f"  Config: {json.dumps(config, indent=2)}")

    # Datasets
    print("\nLoading datasets...")
    train_ds = FloodSegDataset(
        manifest_path=config["manifest_path"],
        split="train", data_root=config["data_root"],
        masks_dir=config["masks_dir"],
        input_size=config["input_size"], augment=True,
    )
    val_ds = FloodSegDataset(
        manifest_path=config["manifest_path"],
        split="val", data_root=config["data_root"],
        masks_dir=config["masks_dir"],
        input_size=config["input_size"], augment=False,
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
    )

    # Class weights
    w_nf, w_flood = compute_class_weights(config["manifest_path"])
    class_weights = torch.tensor([w_nf, w_flood], dtype=torch.float32).to(device)
    print(f"\n  Class weights: non-flood={w_nf:.4f}, flood={w_flood:.4f} (ratio {w_flood/w_nf:.1f}:1)")

    # Model
    print(f"\nCreating model: U-Net + {config['encoder_name']}...")
    model = create_segmentation_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_train:,} trainable")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(
        class_weights, ce_weight=config["ce_weight"],
        dice_weight=config["dice_weight"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"],
    )
    scaler = GradScaler("cuda", enabled=config["use_amp"])

    # Checkpoint directory
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Verify we're not overwriting existing classifier checkpoints
    existing_cls = PROJECT_ROOT / "models" / "flood"
    for forbidden in ["best_flood_model.pt", "final_flood_model.pt"]:
        forbidden_path = existing_cls / forbidden
        assert str(ckpt_dir.resolve()) != str(existing_cls.resolve()), \
            f"Checkpoint dir would overwrite classifier at {existing_cls}!"

    # Save config
    config_path = ckpt_dir / "training_config.json"
    env_info = {
        "config": config,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "n_params": n_params,
        "n_trainable": n_train,
        "class_weights": {"non_flood": w_nf, "flood": w_flood},
        "timestamp": datetime.now().isoformat(),
    }
    if torch.cuda.is_available():
        env_info["gpu"] = torch.cuda.get_device_name(0)
        env_info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

    with open(config_path, "w") as f:
        json.dump(env_info, f, indent=2)
    print(f"\n  Config saved: {config_path}")

    # Metrics
    train_metrics = FloodMetrics()
    val_metrics = FloodMetrics()

    # Training
    best_val_iou = 0.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"  STARTING TRAINING ({config['max_epochs']} epochs)")
    print(f"{'='*70}")

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, config, train_metrics,
        )
        train_m = train_metrics.compute()

        # Validate
        val_loss = validate(model, val_loader, criterion, device, config, val_metrics)
        val_m = val_metrics.compute()

        scheduler.step()

        elapsed = time.time() - t0

        # Log
        print(
            f"  Epoch {epoch:3d}/{config['max_epochs']} | "
            f"t_loss={train_loss:.4f} v_loss={val_loss:.4f} | "
            f"v_IoU={val_m['flood_iou']:.4f} v_Dice={val_m['flood_dice']:.4f} | "
            f"v_P={val_m['flood_precision']:.4f} v_R={val_m['flood_recall']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | {elapsed:.1f}s"
        )

        # Checkpoint
        if val_m["flood_iou"] > best_val_iou:
            best_val_iou = val_m["flood_iou"]
            patience_counter = 0
            ckpt_path = ckpt_dir / "best_seg_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_m,
                "val_loss": val_loss,
                "config": config,
            }, ckpt_path)
            print(f"    >> Saved best model (IoU={best_val_iou:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"\n  Early stopping at epoch {epoch} (patience={config['patience']})")
            break

    # Save final model
    final_path = ckpt_dir / "final_seg_model.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_path)
    print(f"\n  Final model saved: {final_path}")
    print(f"  Best val flood IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
