"""
Real-time Visual Monitor for Flood Segmentation Training.

Can be run in any terminal at any time without disturbing the ongoing training.
Usage:
    .\\venv\\Scripts\\python.exe ml/monitor_training.py
    .\\venv\\Scripts\\python.exe ml/monitor_training.py --once
"""
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "models" / "flood" / "segmentation_v1"
BEST_CKPT = CKPT_DIR / "best_seg_model.pt"
FINAL_CKPT = CKPT_DIR / "final_seg_model.pt"
CONFIG_FILE = CKPT_DIR / "training_config.json"

def get_gpu_stats():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            parts = [x.strip() for x in res.stdout.strip().split(",")]
            if len(parts) >= 4:
                util, mem_used, mem_total, temp = parts[:4]
                return f"GPU: {util}% util | {mem_used} / {mem_total} MiB | {temp} C"
    except Exception:
        pass
    return "GPU: Active"

def is_training_alive():
    try:
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -like '*train_flood_seg*' } | Select-Object -ExpandProperty ProcessId"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        pids = res.stdout.strip().split()
        if pids and pids[0].isdigit():
            return True, int(pids[0])
    except Exception:
        pass
    return False, None

def read_checkpoint():
    if not BEST_CKPT.exists():
        return None
    try:
        ckpt = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
        mtime = datetime.fromtimestamp(BEST_CKPT.stat().st_mtime)
        return {
            "epoch": ckpt.get("epoch", "?"),
            "val_loss": ckpt.get("val_loss", 0.0),
            "val_metrics": ckpt.get("val_metrics", {}),
            "last_updated": mtime.strftime("%H:%M:%S"),
            "seconds_ago": int(time.time() - BEST_CKPT.stat().st_mtime)
        }
    except Exception as e:
        return {"error": str(e)}

def render_dashboard(data, alive, pid, gpu_info):
    os.system("cls" if os.name == "nt" else "clear")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("=" * 72)
    print("         FLOOD SEGMENTATION TRAINING LIVE DASHBOARD")
    print("=" * 72)
    print(f" Current Time: {now_str}")
    if FINAL_CKPT.exists():
        print(f" Status:       [COMPLETED] Final model saved successfully!")
    elif alive:
        print(f" Status:       [RUNNING] Active on GPU (PID: {pid})")
    else:
        print(f" Status:       [STOPPED / IDLE]")
    print(f" Hardware:     {gpu_info}")
    print("-" * 72)

    if data and "epoch" in data:
        vm = data.get("val_metrics", {})
        iou = vm.get("flood_iou", 0.0)
        dice = vm.get("flood_dice", 0.0)
        prec = vm.get("flood_precision", 0.0)
        rec = vm.get("flood_recall", 0.0)
        acc = vm.get("pixel_accuracy", 0.0)
        loss = data.get("val_loss", 0.0)
        
        # Simple progress bar for IoU (target ~ 0.50)
        bar_len = 25
        filled = min(bar_len, max(0, int(iou / 0.50 * bar_len)))
        iou_bar = "[" + "=" * filled + ">" + " " * (bar_len - filled) + "]"

        print(f" Best Epoch:         {data['epoch']} / 100")
        print(f" Best Val Flood IoU: {iou:.4f}  {iou_bar}")
        print(f" Best Val Dice:      {dice:.4f}")
        print(f" Best Val Precision: {prec:.4f}")
        print(f" Best Val Recall:    {rec:.4f}")
        print(f" Best Pixel Accuracy:{acc*100:.2f}%")
        print(f" Best Val Loss:      {loss:.4f}")
        print(f" Checkpoint Updated: {data['last_updated']} ({data['seconds_ago']}s ago)")
    elif data and "error" in data:
        print(" [Checkpoint is being written by GPU right now, retrying in 5s...]")
    else:
        print(" Waiting for initial checkpoint to be written...")

    print("-" * 72)
    if FINAL_CKPT.exists():
        print(" Training has finished! Best weights: models/flood/segmentation_v1/best_seg_model.pt")
    else:
        print(" [Press Ctrl+C to close monitor. Training continues safely in background.]")
    print("=" * 72)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Print status once and exit")
    args = parser.parse_args()

    if args.once:
        alive, pid = is_training_alive()
        gpu_info = get_gpu_stats()
        data = read_checkpoint()
        render_dashboard(data, alive, pid, gpu_info)
        return

    while True:
        alive, pid = is_training_alive()
        gpu_info = get_gpu_stats()
        data = read_checkpoint()
        render_dashboard(data, alive, pid, gpu_info)
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited monitor. Background training was NOT interrupted.")
