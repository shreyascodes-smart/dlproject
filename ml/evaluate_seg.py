"""
Flood Segmentation Split Evaluation.

Loads the best checkpoint from Phase H training and evaluates it on either
the held-out test split (22 samples, 6 events) or the validation split
(24 samples, 6 events).  Produces per-event, per-sample, and aggregate metrics.
Does NOT train or modify any weights.

Usage:
    python ml/evaluate_seg.py --split test
    python ml/evaluate_seg.py --split val
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.flood_seg_dataset import (
    FloodSegDataset,
    IGNORE_INDEX,
    TARGET_FLOOD,
    TARGET_NON_FLOOD,
)
from ml.training.train_flood_seg import (
    FloodMetrics,
    create_segmentation_model,
    DEFAULT_CONFIG,
)


def evaluate_split(
    checkpoint_path: Path,
    output_path: Path,
    split: str = "test",
) -> Dict:
    """Evaluate the best model on a designated dataset split.

    Args:
        checkpoint_path: Path to best_seg_model.pt
        output_path: Path for JSON results file
        split: 'test' or 'val'

    Returns:
        Dictionary with aggregate, per-event, and per-sample metrics.
    """
    print("=" * 70)
    print(f"  FLOOD SEGMENTATION — {split.upper()} SPLIT EVALUATION")
    print("=" * 70)
    print(f"  Timestamp:  {datetime.now().isoformat()}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Split:      {split}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    train_epoch = ckpt["epoch"]
    train_val_metrics = ckpt.get("val_metrics", {})

    print(f"  Trained epoch: {train_epoch}")
    print(f"  Val flood IoU at save: {train_val_metrics.get('flood_iou', 'N/A'):.4f}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Model
    print("\n  Loading model...")
    model = create_segmentation_model(config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("  Model loaded and set to eval mode.")

    # Dataset
    print(f"\n  Loading {split} dataset...")
    ds = FloodSegDataset(
        manifest_path=config["manifest_path"],
        split=split,
        data_root=config["data_root"],
        masks_dir=config["masks_dir"],
        input_size=config["input_size"],
        augment=False,
    )
    print(f"  Total {split} samples: {len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=1,  # Per-sample metrics require batch_size=1
        shuffle=False,
        num_workers=0,  # Windows safe
    )

    # Evaluate
    print("\n  Running inference...")
    aggregate_metrics = FloodMetrics()
    event_metrics: Dict[str, FloodMetrics] = {}
    per_sample_results: List[Dict] = []

    t0 = time.time()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device)

            with autocast("cuda", enabled=config.get("use_amp", True)):
                logits = model(images)

            # Per-sample metrics
            sample_metrics = FloodMetrics()
            sample_metrics.update(logits, targets)
            sm = sample_metrics.compute()

            sample_name = ds.get_sample_name(idx)
            event_id = sample_name.split("_")[0]
            sm["name"] = sample_name
            sm["event_id"] = event_id
            per_sample_results.append(sm)

            # Per-event accumulation
            if event_id not in event_metrics:
                event_metrics[event_id] = FloodMetrics()
            event_metrics[event_id].update(logits, targets)

            # Aggregate
            aggregate_metrics.update(logits, targets)

            print(
                f"    [{idx+1:3d}/{len(ds)}] {sample_name:48s} | "
                f"IoU={sm['flood_iou']:.4f}  Dice={sm['flood_dice']:.4f}  "
                f"P={sm['flood_precision']:.4f}  R={sm['flood_recall']:.4f}"
            )

    elapsed = time.time() - t0
    agg = aggregate_metrics.compute()
    event_results = {eid: m.compute() for eid, m in event_metrics.items()}

    print(f"\n{'='*70}")
    print(f"  AGGREGATE {split.upper()} RESULTS")
    print(f"{'='*70}")
    print(f"  Flood IoU:       {agg['flood_iou']:.4f}")
    print(f"  Flood Dice (F1): {agg['flood_dice']:.4f}")
    print(f"  Flood Precision: {agg['flood_precision']:.4f}")
    print(f"  Flood Recall:    {agg['flood_recall']:.4f}")
    print(f"  Pixel Accuracy:  {agg['pixel_accuracy']:.4f}")
    print(f"  Inference time:  {elapsed:.1f}s ({elapsed/len(ds):.2f}s/sample)")

    print(f"\n{'='*70}")
    print(f"  PER-EVENT {split.upper()} BREAKDOWN")
    print(f"{'='*70}")
    for eid, em in sorted(event_results.items()):
        print(
            f"  {eid:10s} | IoU={em['flood_iou']:.4f}  Dice={em['flood_dice']:.4f}  "
            f"P={em['flood_precision']:.4f}  R={em['flood_recall']:.4f}  "
            f"TP={em['tp']:5d}  FP={em['fp']:6d}  FN={em['fn']:5d}"
        )

    # Assemble results
    results = {
        "phase": "H",
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "train_epoch": train_epoch,
        "train_val_metrics": {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in train_val_metrics.items()
        },
        "aggregate": {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in agg.items()
        },
        "per_event": {
            eid: {k: round(v, 6) if isinstance(v, float) else v for k, v in em.items()}
            for eid, em in sorted(event_results.items())
        },
        "per_sample": [
            {k: round(v, 6) if isinstance(v, float) else v for k, v in s.items()}
            for s in per_sample_results
        ],
        "n_samples": len(ds),
        "inference_time_s": round(elapsed, 2),
        "device": str(device),
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate flood segmentation model on a split.")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "models" / "flood" / "segmentation_v1" / "best_seg_model.pt"),
        help="Path to checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output JSON results (optional)",
    )

    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = PROJECT_ROOT / "data" / "reports" / f"phase_h_{args.split}_evaluation.json"

    evaluate_split(ckpt_path, out_path, split=args.split)
