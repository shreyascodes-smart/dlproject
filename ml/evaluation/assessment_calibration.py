"""
Evaluate a segmentation model with the PRODUCT protocol and derive (validation) or check (test)
its outcome thresholds.

For every sample this runs the model at full resolution, sets aside predicted positive pixels on
JRC permanent water (WorldFloods PERMANENTWATERJRC, class 3 = permanent), and scores
  flood = positive & ~permanent water   against the frozen EMS flood reference (validity applied).
Raw (no exclusion) counts are reported too. Per-event and per-sample metrics and runtimes are saved.

  --split val   DERIVES the policy:
                  inconclusive ceiling = max flood-like share (outside permanent water) on validation
                  scenes with NO EMS-mapped flood (the model's observed false-alarm range).
  --split test  CHECKS the frozen policy for that model on held-out events (no thresholds changed).
                Run once per model, after the policy is fixed.

Never modifies data, masks, manifests or weights; refuses to overwrite an existing report.
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
from ml.inference import assessment as asm  # noqa: E402
from ml.datasets.flood_seg_dataset import STORED_FLOOD, STORED_IGNORE  # noqa: E402

MANIFEST = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "seg_split_manifest.json"
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "flood" / "worldfloods_v2"
MASKS = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "masks"
MODELS = PROJECT_ROOT / "models" / "flood"
JRC_PERMANENT = 3

FIXED_CHOICES = {
    "none_below_pct": 1.0,
    "min_assessable_land_frac": 0.25,
    "water_dominated_frac": 0.5,
}


KNOWN_WATER_MODE = "permanent"  # set from --known-water


def known_water_for(sample, mode: str = None) -> np.ndarray:
    """JRC known water on the scene grid. mode 'permanent' (Milestone 5) or 'permanent+seasonal' (Milestone 6)."""
    mode = mode or KNOWN_WATER_MODE
    p = DATA_ROOT / sample["original_split"] / "PERMANENTWATERJRC" / f"{sample['name']}.tif"
    jrc = si.read_mask(p)
    out = np.where(jrc == JRC_PERMANENT, asm.KW_PERMANENT, asm.KW_NONE).astype(np.uint8)
    if mode == "permanent+seasonal":
        out[jrc == 2] = asm.KW_SEASONAL
    return out


def run(model, split, methods, policy_in, out, checkpoint=None):
    manifest = json.loads(MANIFEST.read_text())
    samples = manifest["splits"][split]["samples"]
    seg = si.FloodSegmenter(checkpoint or MODELS / model / "best_seg_model.pt")
    print(f"{model}: {seg.spec.preprocessing_id} target={seg.spec.target} epoch={seg.meta.get('epoch')} on {seg.device}")
    rows = []
    t_all = time.time()
    for i, s in enumerate(samples, 1):
        img = si.read_scene(DATA_ROOT / s["original_split"] / "S2" / f"{s['name']}.tif", seg.spec)
        validity = si.read_mask(MASKS / f"{s['name']}_validity.tif")
        stored = si.read_mask(MASKS / f"{s['name']}_seg.tif")
        stored[validity == 0] = STORED_IGNORE
        kw = known_water_for(s)
        ref_flood = int((stored == STORED_FLOOD).sum())
        row = {"name": s["name"], "event_id": s["event_id"], "reference_has_flood": ref_flood > 0,
               "reference_flood_px": ref_flood,
               "reference_flood_on_permanent_water_px": int(((stored == STORED_FLOOD) & (kw > 0)).sum())}
        for m in methods:
            t0 = time.time()
            res = seg.predict_array(img, validity=validity, method=m, make_display=False)
            wc = asm.water_context(res.flood_mask, res.eligible_mask, kw)
            _, raw_m = si.compare_with_reference(res.flood_mask, stored)
            _, ex_m = si.compare_with_reference(asm.outside_water_mask(res.flood_mask, kw), stored)
            row[m] = {
                "raw_positive_pct": wc["raw_flood_like_pct"],
                "outside_known_water_pct": wc["flood_like_outside_known_water_pct"],
                "known_water_pct_of_eligible": wc["known_water_pct_of_eligible"],
                "flood_like_on_known_water_share_pct": wc["flood_like_on_known_water_share_pct"],
                "eligible_pct": res.stats["eligible_pct"],
                "quality_status": res.stats["quality_status"],
                "haze_index": res.stats.get("haze_index"),
                "hazy": res.stats.get("hazy", False),
                "water_gate_applied": res.stats.get("water_gate_applied", False),
                "counts_raw": {k: raw_m[k] for k in ("tp", "fp", "fn", "tn")},
                "counts_excluding_known_water": {k: ex_m[k] for k in ("tp", "fp", "fn", "tn")},
                "seconds": round(time.time() - t0, 2),
                "_wc": wc,
            }
        rows.append(row)
        print(f"[{i:2d}/{len(samples)}] {s['name']:46s} ref={'Y' if ref_flood else '-'} "
              + " ".join(f"{m}: raw={row[m]['raw_positive_pct']}% out={row[m]['outside_known_water_pct']}%"
                         for m in methods), flush=True)

    def agg(m, key, subset):
        c = np.zeros(4, np.int64)
        for r in subset:
            c += [r[m][key][k] for k in ("tp", "fp", "fn", "tn")]
        return si.confusion_metrics(*[int(x) for x in c])

    summary = {}
    for m in methods:
        neg = [r[m]["outside_known_water_pct"] for r in rows
               if not r["reference_has_flood"] and r[m]["outside_known_water_pct"] is not None]
        pos = [r[m]["outside_known_water_pct"] for r in rows
               if r["reference_has_flood"] and r[m]["outside_known_water_pct"] is not None]
        events = sorted({r["event_id"] for r in rows})
        summary[m] = {
            "n_negative_scenes": len(neg), "n_positive_scenes": len(pos),
            "negative_outside_water_pct": {"min": min(neg) if neg else None, "max": max(neg) if neg else None,
                                           "median": float(np.median(neg)) if neg else None},
            "positive_outside_water_pct": sorted(pos),
            "pixel_metrics_raw": agg(m, "counts_raw", rows),
            "pixel_metrics_excluding_known_water": agg(m, "counts_excluding_known_water", rows),
            "per_event_excluding_known_water": {
                e: agg(m, "counts_excluding_known_water", [r for r in rows if r["event_id"] == e]) for e in events},
            "seconds_total": round(sum(r[m]["seconds"] for r in rows), 1),
        }

    if split == "val":
        ceilings = {m: float(np.ceil(summary[m]["negative_outside_water_pct"]["max"] * 100) / 100) for m in methods}
        tag = "" if KNOWN_WATER_MODE == "permanent" else "-seasonal"
        policy = {"version": f"{model}-val-{datetime.now():%Y%m%d}{tag}", "model_version": model,
                  "known_water_mode": KNOWN_WATER_MODE, **FIXED_CHOICES,
                  "inconclusive_ceiling_pct": ceilings,
                  "provenance": "inconclusive_ceiling_pct = maximum flood-like share (outside JRC permanent water) "
                                "predicted on validation scenes that have no EMS-mapped flood. Other values are "
                                "documented design choices, not fitted."}
    else:
        policy = policy_in

    for r in rows:
        for m in methods:
            wc = r[m].pop("_wc")
            r[m]["level"] = asm.classify(wc, m, r[m]["quality_status"], policy, r[m]["eligible_pct"], live=False,
                                         hazy=r[m]["hazy"])["level"]
    for m in methods:
        lv = {}
        for r in rows:
            key = "with_mapped_flood" if r["reference_has_flood"] else "without_mapped_flood"
            lv.setdefault(key, {}).setdefault(r[m]["level"], 0)
            lv[key][r[m]["level"]] += 1
        summary[m]["levels"] = lv

    report = {
        "kind": "assessment_policy" if split == "val" else "assessment_policy_heldout_check",
        "model_version": model,
        "model": {k: v for k, v in seg.meta.items() if k in ("version", "architecture", "encoder", "epoch", "bands",
                                                             "target", "preprocessing", "postprocessing")},
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "note": ("Thresholds DERIVED from the validation split." if split == "val"
                 else "Held-out TEST check of a FROZEN policy; no thresholds were changed."),
        "protocol": "full-resolution inference; flood = positive & ~JRC permanent water vs EMS flood (validity "
                    "applied); raw = positive vs EMS flood",
        "known_water_source": "WorldFloods v2 PERMANENTWATERJRC (JRC Global Surface Water, class 3 = permanent"
                              + (", class 2 = seasonal)" if KNOWN_WATER_MODE == "permanent+seasonal" else ")"),
        "known_water_mode": KNOWN_WATER_MODE,
        "policy": policy,
        "summary": summary,
        "per_sample": rows,
        "elapsed_s": round(time.time() - t_all, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out} in {report['elapsed_s']} s")
    for m in methods:
        sm = summary[m]
        print(f"{m}: negatives max={sm['negative_outside_water_pct']['max']} | levels {sm['levels']}")
        print(f"   IoU raw={sm['pixel_metrics_raw']['flood_iou']:.4f} | excluding known water="
              f"{sm['pixel_metrics_excluding_known_water']['flood_iou']:.4f} "
              f"P={sm['pixel_metrics_excluding_known_water']['flood_precision']:.4f} "
              f"R={sm['pixel_metrics_excluding_known_water']['flood_recall']:.4f}")
    print("policy:", json.dumps(policy))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="segmentation_v1", help="model version directory under models/flood/")
    ap.add_argument("--checkpoint", default=None, help="explicit checkpoint path (default: <model>/best_seg_model.pt)")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--methods", nargs="+", default=["tiled", "resize"])
    ap.add_argument("--policy", default=None, help="frozen policy JSON for --split test (default: the model's)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--known-water", choices=["permanent", "permanent+seasonal"], default="permanent",
                    help="which JRC classes are set aside before scoring and classification")
    a = ap.parse_args()
    KNOWN_WATER_MODE = a.known_water
    pol = None
    if a.split == "test":
        p = Path(a.policy) if a.policy else asm.policy_path(a.model)
        data = json.loads(p.read_text())
        if data.get("split") != "val":
            sys.exit("Refusing: the frozen policy must come from the validation split.")
        pol = data["policy"]
    run(a.model, a.split, a.methods, pol, Path(a.output), a.checkpoint)
