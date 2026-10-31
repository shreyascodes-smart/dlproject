"""
Validation-only comparison of the two inference methods in seg_inference.py.

Protocols computed for every validation sample:
    A256    "resize" method, scored on the 256x256 resized reference mask.
            This is the Phase H protocol; it should reproduce
            data/reports/phase_h_val_evaluation.json (sanity check that
            inference preprocessing matches training/evaluation).
    A_full  "resize" method, prediction upsampled and scored at full 10 m resolution.
    B_full  "tiled"  method (256 px tiles at native resolution), scored at full resolution.

Reference: stored EMS seg mask with validity==0 pixels set to ignore, exactly
as FloodSegDataset does. Ignored pixels are excluded from all counts.

This script REFUSES the test split. It never trains or modifies weights, masks
or manifests, and writes only to a NEW output file.

Usage:
    python ml/evaluation/compare_inference_methods.py --output data/reports/<new>.json
    python ml/evaluation/compare_inference_methods.py --limit 1      # timing probe
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import seg_inference as si  # noqa: E402
from ml.datasets.flood_seg_dataset import STORED_IGNORE, get_val_transform  # noqa: E402

MANIFEST = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "seg_split_manifest.json"
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "flood" / "worldfloods_v2"
MASKS = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "masks"
PROTOCOLS = ("A256", "A_full", "B_full")


def counts(pred_flood: np.ndarray, stored: np.ndarray) -> np.ndarray:
    _, m = si.compare_with_reference(pred_flood, stored)
    return np.array([m["tp"], m["fp"], m["fn"], m["tn"]], dtype=np.int64)


def metrics(c: np.ndarray) -> dict:
    return si.confusion_metrics(*[int(v) for v in c])


def load_reference(name: str) -> np.ndarray:
    seg = si.read_mask(MASKS / f"{name}_seg.tif")
    val = si.read_mask(MASKS / f"{name}_validity.tif")
    seg[val == 0] = STORED_IGNORE
    return seg, val


def run(output: Path, limit: int, overlap: int, device: str):
    manifest = json.loads(MANIFEST.read_text())
    samples = manifest["splits"]["val"]["samples"][: limit or None]
    seg = si.FloodSegmenter(device=device)
    size = seg.input_size
    print(f"Model: {seg.meta['architecture']} {seg.meta['encoder']} epoch {seg.meta['epoch']} on {seg.device}")
    print(f"Validation samples: {len(samples)}  (test split is never used)")

    per_sample, per_event = [], {}
    totals = {p: np.zeros(4, np.int64) for p in PROTOCOLS}
    t_all = time.time()
    for i, s in enumerate(samples, 1):
        name, event = s["name"], s["event_id"]
        t0 = time.time()
        rgb = si.read_s2_rgb(DATA_ROOT / s["original_split"] / "S2" / f"{name}.tif")
        ref, val = load_reference(name)
        h, w = ref.shape
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"{name}: image {rgb.shape[:2]} vs mask {(h, w)}")

        # --- A256: Phase H protocol (resize in, score on resized mask) ---
        x = si.preprocess_resize(rgb, size)
        p256 = seg._forward_probs(x[None])[0]
        ref_pad = np.pad(ref, ((0, max(0, size - h)), (0, max(0, size - w))), constant_values=STORED_IGNORE)
        ref256 = get_val_transform(size)(image=np.zeros(ref_pad.shape + (3,), np.float32), mask=ref_pad)["mask"]
        c = {"A256": counts(p256 > si.FLOOD_THRESHOLD, ref256)}
        t_a256 = time.time() - t0

        # --- A_full / B_full: full-resolution scoring ---
        t1 = time.time()
        ra = seg.predict_array(rgb, validity=val, method="resize", make_display=False)
        t_af = time.time() - t1
        t1 = time.time()
        rb = seg.predict_array(rgb, validity=val, method="tiled", overlap=overlap, make_display=False)
        t_bf = time.time() - t1
        c["A_full"] = counts(ra.flood_mask, ref)
        c["B_full"] = counts(rb.flood_mask, ref)

        row = {"name": name, "event_id": event, "original_split": s["original_split"],
               "width": w, "height": h, "n_tiles": rb.stats["n_tiles"],
               "ref_flood_px_full": int((ref == 2).sum()), "ref_supervised_px_full": int((ref > 0).sum()),
               "pred_coverage_pct": {"A_full": ra.stats["flood_coverage_pct"], "B_full": rb.stats["flood_coverage_pct"]},
               "seconds": {"A256": round(t_a256, 2), "A_full": round(t_af, 2), "B_full": round(t_bf, 2)}}
        for p in PROTOCOLS:
            totals[p] += c[p]
            per_event.setdefault(event, {q: np.zeros(4, np.int64) for q in PROTOCOLS})[p] += c[p]
            row[p] = metrics(c[p])
        per_sample.append(row)

        def f(v):
            return "  n/a " if v is None else f"{v:.4f}"
        print(f"[{i:2d}/{len(samples)}] {name:46s} {w}x{h} | IoU A256={f(row['A256']['flood_iou'])} "
              f"A_full={f(row['A_full']['flood_iou'])} B_full={f(row['B_full']['flood_iou'])} | {time.time()-t0:.1f}s")

    elapsed = time.time() - t_all
    result = {
        "kind": "inference_method_comparison",
        "split": "val",
        "note": "Validation split only. Not a generalisation estimate; the held-out test split was not used.",
        "timestamp": datetime.now().isoformat(),
        "model": {k: v for k, v in seg.meta.items() if k != "n_params"},
        "settings": {"threshold": si.FLOOD_THRESHOLD, "tile_size": size, "tile_overlap": overlap,
                     "amp": seg.use_amp, "reference": "stored EMS seg mask, validity==0 -> ignore"},
        "protocols": {
            "A256": "resize to 256, scored on nearest-resized 256x256 reference (Phase H protocol)",
            "A_full": "resize to 256, probability upsampled, scored at full resolution",
            "B_full": "256 px tiles at native resolution with overlap blending, scored at full resolution",
        },
        "aggregate": {p: metrics(totals[p]) for p in PROTOCOLS},
        "per_event": {e: {p: metrics(v[p]) for p in PROTOCOLS} for e, v in sorted(per_event.items())},
        "per_sample": per_sample,
        "n_samples": len(samples),
        "elapsed_s": round(elapsed, 1),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file {output}")
    output.write_text(json.dumps(result, indent=2))
    print(f"\nTotal {elapsed:.1f}s. Saved {output}")
    for p in PROTOCOLS:
        m = result["aggregate"][p]
        print(f"  {p:7s} IoU={m['flood_iou']:.4f} Dice={m['flood_dice']:.4f} "
              f"P={m['flood_precision']:.4f} R={m['flood_recall']:.4f} "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", help="only 'val' is allowed")
    ap.add_argument("--output", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N samples only (timing probe)")
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.split != "val":
        sys.exit("Refusing: this comparison may only use the validation split (no test-set tuning).")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(a.output) if a.output else PROJECT_ROOT / "data" / "reports" / f"kiro_val_method_comparison_{stamp}.json"
    run(out, a.limit, a.overlap, a.device)
