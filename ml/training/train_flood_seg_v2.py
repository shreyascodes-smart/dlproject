"""
v2 training: multispectral (B02 B03 B04 B08 B11 B12) U-Net / EfficientNet-B2 for SURFACE WATER.

Why the change from v1 (verified in the Milestone 5 audit):
  * v1 saw only RGB. Water is separated from land mainly in NIR/SWIR; a fixed MNDWI index
    (green vs SWIR, no training) already reaches validation flood IoU 0.230 vs v1's 0.252.
  * v1 had to learn "flood water vs permanent water" from one image, which a single image
    cannot show. v2 learns surface water (WorldFloods gt band 2) and the product removes JRC
    permanent water afterwards (the published WorldFloods approach). On validation that
    decomposition matches the EMS flood maps at IoU 0.62.
  * v1 read whole rasters each step (~550 s/epoch); v2 trains from a patch cache.

Checkpoint selection uses the PRODUCT metric on the frozen validation split:
  flood = (P(water) > 0.5) & ~JRC permanent water, scored against the EMS flood reference.
The held-out test split is never touched here.

Usage:  python ml/training/train_flood_seg_v2.py [--epochs 30]
Writes: models/flood/segmentation_v2/{best_seg_model.pt, final_seg_model.pt, training_config.json, train_log.jsonl}
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CACHE = PROJECT_ROOT / "data" / "interim" / "flood" / "patches_v2"
OUT_DIR = PROJECT_ROOT / "models" / "flood" / "segmentation_v2"
IGNORE = 255


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class PatchSet(Dataset):
    def __init__(self, names, split, mean, std, augment):
        self.split, self.augment = split, augment
        self.mean = np.array(mean, np.float32)[:, None, None]
        self.std = np.array(std, np.float32)[:, None, None]
        self.names = names
        self.items = []
        for n in names:
            k = np.load(CACHE / split / f"{n}.y.npy", mmap_mode="r").shape[0]
            self.items += [(n, i) for i in range(k)]
        self._mm = {}

    def _arr(self, n, kind):
        key = (n, kind)
        if key not in self._mm:
            self._mm[key] = np.load(CACHE / self.split / f"{n}.{kind}.npy", mmap_mode="r")
        return self._mm[key]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        n, i = self.items[idx]
        x = np.asarray(self._arr(n, "x")[i], np.float32)
        y = np.asarray(self._arr(n, "y")[i], np.int64)
        if self.augment:
            if random.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if random.random() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
            k = random.randint(0, 3)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            # mild per-band gain jitter: robustness to calibration / processing-level differences
            x = x * np.random.uniform(0.92, 1.08, size=(x.shape[0], 1, 1)).astype(np.float32)
        x = (np.clip(x / 10000.0, 0, 1) - self.mean) / self.std
        y = y.copy()
        y[y == IGNORE] = -100
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), torch.from_numpy(np.ascontiguousarray(y))


def dice_loss(logits, y):
    valid = y != -100
    p = F.softmax(logits.float(), 1)[:, 1]
    t = (y == 1).float()
    p, t = p[valid], t[valid]
    if t.numel() == 0:
        return logits.sum() * 0
    inter = (p * t).sum()
    return 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)


@torch.no_grad()
def evaluate(model, val_names, mean, std, device, batch=32):
    """Product metric on frozen validation tiles (+ water IoU vs WorldFloods water target)."""
    model.eval()
    m = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    s = torch.tensor(std, device=device).view(1, -1, 1, 1)
    c = np.zeros(4, np.int64)   # flood tp fp fn tn vs EMS
    w = np.zeros(3, np.int64)   # water tp fp fn vs WF gt
    for n in val_names:
        X = np.load(CACHE / "val" / f"{n}.x.npy", mmap_mode="r")
        Y = np.load(CACHE / "val" / f"{n}.y.npy", mmap_mode="r")
        E = np.load(CACHE / "val" / f"{n}.ems.npy", mmap_mode="r")
        PW = np.load(CACHE / "val" / f"{n}.pw.npy", mmap_mode="r")
        for i in range(0, X.shape[0], batch):
            x = torch.from_numpy(np.asarray(X[i:i + batch], np.float32)).to(device)
            nod = (x == 0).all(1)
            x = (torch.clamp(x / 10000.0, 0, 1) - m) / s
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logit = model(x)
            water = (F.softmax(logit.float(), 1)[:, 1] > 0.5) & ~nod
            water = water.cpu().numpy()
            e = np.asarray(E[i:i + batch])
            pw = np.asarray(PW[i:i + batch]).astype(bool)
            y = np.asarray(Y[i:i + batch])
            flood = water & ~pw
            sup = (e == 1) | (e == 2)
            tf = e == 2
            c += [(sup & tf & flood).sum(), (sup & ~tf & flood).sum(), (sup & tf & ~flood).sum(),
                  (sup & ~tf & ~flood).sum()]
            vy = y != IGNORE
            w += [(vy & (y == 1) & water).sum(), (vy & (y == 0) & water).sum(), (vy & (y == 1) & ~water).sum()]
    tp, fp, fn, tn = (int(v) for v in c)
    return {
        "flood_iou": tp / max(tp + fp + fn, 1), "flood_precision": tp / max(tp + fp, 1),
        "flood_recall": tp / max(tp + fn, 1), "flood_dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "water_iou": int(w[0]) / max(int(w.sum()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 = full pass (debug: small number)")
    ap.add_argument("--out", default=str(OUT_DIR), help="output directory (use scratch for smoke runs)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay for the evaluated/saved weights (0 = off). Reduces epoch-to-epoch noise.")
    ap.add_argument("--pin-memory", action="store_true",
                    help="pin host memory in the DataLoader (off by default: a Windows pinned-allocator crash "
                         "ended one run)")
    a = ap.parse_args()
    set_seed(a.seed)
    out_dir = Path(a.out)
    legacy = PROJECT_ROOT / "models" / "flood"
    if out_dir.resolve() in (legacy.resolve(), (legacy / "segmentation_v1").resolve()):
        sys.exit("Refusing to write into the legacy classifier or v1 checkpoint directory.")

    index = json.loads((CACHE / "index.json").read_text())
    mean, std = index["mean"], index["std"]
    train_names = sorted(index["train"])
    val_names = sorted(k for k, v in index["val"].items() if v["n"] > 0)
    manifest_val = json.loads((PROJECT_ROOT / "data/processed/flood/segmentation_v1/seg_split_manifest.json")
                              .read_text())["splits"]["val"]["samples"]
    assert len(val_names) == len(manifest_val), "validation cache incomplete"
    events = {v["event"] for v in index["train"].values()}
    assert not events & set(index["frozen_events_excluded"]), "leakage: frozen event in training set"

    p_water = index["train_water_fraction"]
    w_water = float(np.clip(math.sqrt((1 - p_water) / max(p_water, 1e-3)), 1.0, 6.0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["HF_HUB_OFFLINE"] = "1"
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="efficientnet-b2", encoder_weights="imagenet", in_channels=6, classes=2).to(device)

    ds = PatchSet(train_names, "train", mean, std, augment=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                    pin_memory=a.pin_memory, persistent_workers=a.workers > 0)
    ema = None
    if a.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
    steps = len(dl) if not a.max_steps_per_epoch else min(len(dl), a.max_steps_per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = steps * a.epochs
    warm = steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, w_water], device=device), ignore_index=-100)

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "version": "segmentation_v2", "task": "surface_water", "architecture": "U-Net",
        "encoder_name": "efficientnet-b2", "encoder_weights": "imagenet", "in_channels": 6, "num_classes": 2,
        "input_size": 256, "bands": index["bands"], "band_source": "Sentinel-2 L1C TOA DN (x10000)",
        "normalization": {"kind": "standardize", "scale": 10000.0, "clip": [0, 1], "mean": mean, "std": std},
        "target": "surface_water", "flood_rule": "water & ~JRC permanent water",
        "train_scenes": len(train_names), "train_events": len(events), "train_patches": len(ds),
        "train_water_fraction": p_water, "class_weights": [1.0, w_water],
        "batch": a.batch, "lr": a.lr, "epochs": a.epochs, "patience": a.patience, "seed": a.seed,
        "ema_decay": a.ema or None,
        "selection_metric": "val flood IoU (water & ~permanent vs EMS flood), 256 px tiles",
        "pytorch": torch.__version__, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "started": datetime.now().isoformat(),
    }
    (out_dir / "training_config.json").write_text(json.dumps(config, indent=2))
    log = open(out_dir / "train_log.jsonl", "a")
    print(f"train {len(ds)} patches / {len(train_names)} scenes / {len(events)} events; val {len(val_names)} scenes; "
          f"water frac {p_water:.3f} -> weight {w_water:.2f}; {steps} steps/epoch on {device}", flush=True)

    best, bad, t_all = -1.0, 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tl, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(dl):
            if step >= steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = 0.5 * ce(out.float(), y) + 0.5 * dice_loss(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tl += loss.item()
            n += 1
        net = ema.module if ema is not None else model
        vm = evaluate(net, val_names, mean, std, device)
        rec = {"epoch": ep, "train_loss": tl / max(n, 1), **vm, "lr": opt.param_groups[0]["lr"],
               "epoch_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        improved = vm["flood_iou"] > best
        print(f"ep {ep:2d} loss {rec['train_loss']:.4f} | val flood IoU {vm['flood_iou']:.4f} "
              f"P {vm['flood_precision']:.3f} R {vm['flood_recall']:.3f} | water IoU {vm['water_iou']:.4f} "
              f"| {rec['epoch_s']}s{' *' if improved else ''}", flush=True)
        if improved:
            best, bad = vm["flood_iou"], 0
            torch.save({"epoch": ep, "model_state_dict": net.state_dict(), "config": config, "val_metrics": vm},
                       out_dir / "best_seg_model.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop at epoch {ep}")
                break
    torch.save({"epoch": ep, "model_state_dict": net.state_dict(), "config": config}, out_dir / "final_seg_model.pt")
    config["finished"] = datetime.now().isoformat()
    config["best_val_flood_iou_tiles"] = best
    config["train_seconds"] = round(time.time() - t_all)
    (out_dir / "training_config.json").write_text(json.dumps(config, indent=2))
    print(f"done in {time.time()-t_all:.0f}s; best val flood IoU (tiles) {best:.4f}")


if __name__ == "__main__":
    main()
