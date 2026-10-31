"""
Validation study: a physical short-wave-infrared (SWIR) consistency check on v2 water pixels.

Motivation (Milestone 5 live check, Mumbai, 18 Oct 2026): in a hazy image the v2 model marked
dense urban blocks as water. Haze brightens the visible bands but barely affects SWIR, and open
water is dark in SWIR (B11). Candidate rule: a model-positive pixel counts as water only if its
B11 top-of-atmosphere reflectance is below T.

This script evaluates T on the frozen VALIDATION split only (full resolution, tiled, product
protocol: positives outside JRC permanent water vs the EMS flood reference, validity applied).
The test split is not read. Output: JSON with pooled and per-event metrics per threshold.

  python ml/evaluation/swir_gate_validation.py --output data/reports/swir_gate_validation_segmentation_v2.json
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import seg_inference as si  # noqa: E402
from ml.inference import assessment as asm  # noqa: E402
from ml.datasets.flood_seg_dataset import STORED_FLOOD, STORED_IGNORE  # noqa: E402
from ml.evaluation.assessment_calibration import DATA_ROOT, MANIFEST, MASKS, MODELS, known_water_for  # noqa: E402

THRESHOLDS = [None, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06]


def metrics(c):
    tp, fp, fn = c
    return {"flood_iou": tp / max(tp + fp + fn, 1), "flood_precision": tp / max(tp + fp, 1),
            "flood_recall": tp / max(tp + fn, 1), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="segmentation_v2")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    seg = si.FloodSegmenter(MODELS / a.model / "best_seg_model.pt")
    # The study compares thresholds against the UNGATED model output
    seg.spec = replace(seg.spec, water_gate=None)
    b11 = list(seg.spec.bands).index("B11")
    samples = json.loads(MANIFEST.read_text())["splits"]["val"]["samples"]
    pooled = {str(t): np.zeros(3, np.int64) for t in THRESHOLDS}
    per_event = {}
    neg_share = {str(t): [] for t in THRESHOLDS}
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        img = si.read_scene(DATA_ROOT / s["original_split"] / "S2" / f"{s['name']}.tif", seg.spec)
        validity = si.read_mask(MASKS / f"{s['name']}_validity.tif")
        stored = si.read_mask(MASKS / f"{s['name']}_seg.tif")
        stored[validity == 0] = STORED_IGNORE
        kw = known_water_for(s)
        res = seg.predict_array(img, validity=validity, method="tiled", make_display=False)
        sup = stored != STORED_IGNORE
        ref = stored == STORED_FLOOD
        has_flood = bool(ref.any())
        land = res.eligible_mask & (kw == asm.KW_NONE)
        for t in THRESHOLDS:
            water = res.flood_mask if t is None else (res.flood_mask & (img[..., b11] < t * 10000))
            flood = water & (kw == asm.KW_NONE)
            c = np.array([(sup & ref & flood).sum(), (sup & ~ref & flood).sum(), (sup & ref & ~flood).sum()], np.int64)
            pooled[str(t)] += c
            per_event.setdefault(s["event_id"], {}).setdefault(str(t), np.zeros(3, np.int64))
            per_event[s["event_id"]][str(t)] += c
            if not has_flood and land.sum():
                neg_share[str(t)].append(round(100.0 * float((flood & land).sum()) / float(land.sum()), 4))
        print(f"[{i:2d}/{len(samples)}] {s['name']:46s} " + " ".join(
            f"{t}:{100 * pooled[str(t)][0] / max(pooled[str(t)].sum(), 1):.1f}" for t in THRESHOLDS[:1]), flush=True)

    summary = {}
    for t in THRESHOLDS:
        k = str(t)
        pe = [metrics(v[k])["flood_iou"] for v in per_event.values()]
        summary[k] = {**metrics(pooled[k]), "per_event_mean_iou": float(np.mean(pe)),
                      "negative_outside_water_pct_max": max(neg_share[k]) if neg_share[k] else None}
    report = {
        "kind": "swir_gate_validation", "model_version": a.model, "split": "val", "method": "tiled",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rule": "water pixel kept only if B11 TOA reflectance (DN/10000) < T; None = no gate",
        "protocol": "full resolution; flood = kept positives & ~JRC permanent water vs EMS flood (validity applied)",
        "summary": summary,
        "per_event": {e: {k: metrics(v) for k, v in d.items()} for e, d in per_event.items()},
        "seconds": round(time.time() - t0, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for t in THRESHOLDS:
        s_ = summary[str(t)]
        print(f"T={t}: IoU {s_['flood_iou']:.4f} P {s_['flood_precision']:.4f} R {s_['flood_recall']:.4f} "
              f"per-event {s_['per_event_mean_iou']:.4f} neg-max {s_['negative_outside_water_pct_max']}")


if __name__ == "__main__":
    main()
