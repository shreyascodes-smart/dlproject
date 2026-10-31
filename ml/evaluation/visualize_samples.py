"""
Export side-by-side inspection panels for WorldFloods samples:

    Sentinel-2 RGB | EMS reference (ground truth) | Model prediction | Error view

All four panels are on the same pixel grid (the original raster), downsampled
together only for display. Ignored/unsupervised reference pixels are brown in
both the reference and error panels, so they are never confused with known
non-flood pixels (shown as the dimmed satellite image).

Default split is 'val'. The test split is allowed ONLY for final illustration
(--allow-test) and must never be used to choose methods or thresholds.

Usage:
    python ml/evaluation/visualize_samples.py --names EMSR342_07SOUTHNORMANTON_DEL_MONIT03_v2
    python ml/evaluation/visualize_samples.py --auto 4 --method tiled
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import seg_inference as si  # noqa: E402
from ml.datasets.flood_seg_dataset import STORED_IGNORE  # noqa: E402

MANIFEST = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "seg_split_manifest.json"
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "flood" / "worldfloods_v2"
MASKS = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "masks"


def find_samples(manifest, split):
    return {s["name"]: s for s in manifest["splits"][split]["samples"]}


def auto_pick(samples, n):
    """Deterministic, unbiased pick: largest-flood positives first, then one negative.

    Chosen from reference labels only (never from model output) so good or bad
    predictions are not cherry-picked.
    """
    pos = sorted((s for s in samples.values() if s["label_status"] == "positive"),
                 key=lambda s: -s["flood_px"])
    neg = sorted((s for s in samples.values() if s["label_status"] != "positive"), key=lambda s: s["name"])
    # one positive per event first, for diversity
    picked, seen = [], set()
    for s in pos:
        if s["event_id"] not in seen:
            picked.append(s); seen.add(s["event_id"])
    picked = picked[: max(1, n - 1)]
    if neg and len(picked) < n:
        picked.append(neg[0])
    return picked[:n]


def make_panel(seg, sample, method, max_side):
    name = sample["name"]
    rgb = si.read_s2_rgb(DATA_ROOT / sample["original_split"] / "S2" / f"{name}.tif")
    stored = si.read_mask(MASKS / f"{name}_seg.tif")
    validity = si.read_mask(MASKS / f"{name}_validity.tif")
    stored[validity == 0] = STORED_IGNORE
    res = seg.predict_array(rgb, validity=validity, method=method, source=name)
    codes, m = si.compare_with_reference(res.flood_mask, stored)

    views = [
        res.rgb_display,
        si.render_reference(res.rgb_display, stored),
        si.render_overlay(res.rgb_display, res.flood_mask),
        si.render_error_map(res.rgb_display, codes),
    ]
    views = [si.downsample_for_display(v, max_side) for v in views]
    return views, res, m


def save_figure(views, titles, header, legend, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    h, w = views[0].shape[:2]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6 * h / w + 1.2), dpi=110)
    for ax, v, t in zip(axes, views, titles):
        ax.imshow(v)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(header, fontsize=12)
    fig.legend(handles=[Patch(color=np.array(c) / 255, label=l) for l, c in legend],
               loc="lower center", ncol=len(legend), fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true", help="required for --split test (illustration only)")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--auto", type=int, default=4)
    ap.add_argument("--method", choices=si.METHODS, default="tiled")
    ap.add_argument("--max-side", type=int, default=900)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        sys.exit("Test split is held out. Use --allow-test only for final illustrative figures.")

    manifest = json.loads(MANIFEST.read_text())
    samples = find_samples(manifest, a.split)
    if a.names:
        missing = [n for n in a.names if n not in samples]
        if missing:
            sys.exit(f"Not in {a.split} split: {missing}")
        chosen = [samples[n] for n in a.names]
    else:
        chosen = auto_pick(samples, a.auto)

    out_dir = Path(a.out_dir) if a.out_dir else PROJECT_ROOT / "docs" / "figures" / f"seg_inspection_{a.split}_{a.method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = si.FloodSegmenter()
    summary = []
    for s in chosen:
        views, res, m = make_panel(seg, s, a.method, a.max_side)
        f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        header = (f"{s['name']}  |  event {s['event_id']}  |  split: {a.split.upper()}"
                  f"{' (held-out, illustration only)' if a.split == 'test' else ''}  |  method: {a.method}\n"
                  f"flood IoU {f(m['flood_iou'])}  precision {f(m['flood_precision'])}  recall {f(m['flood_recall'])}  "
                  f"|  predicted coverage {res.stats['flood_coverage_pct']}% of eligible pixels")
        titles = ["Sentinel-2 RGB (B4,B3,B2)", "EMS reference (ground truth)",
                  "Model prediction (U-Net)", "Error view (prediction vs reference)"]
        legend = [("flood (reference / prediction)", si.FLOOD_COLOR),
                  ("ignored: cloud / invalid / outside EMS area", si.IGNORE_COLOR),
                  ("correct flood (TP)", si.ERROR_COLORS[si.ERR_TP]), ("false alarm (FP)", si.ERROR_COLORS[si.ERR_FP]),
                  ("missed flood (FN)", si.ERROR_COLORS[si.ERR_FN])]
        path = out_dir / f"{s['name']}.png"
        save_figure(views, titles, header, legend, path)
        summary.append({"name": s["name"], "event_id": s["event_id"], "split": a.split, "method": a.method,
                        "file": str(path.relative_to(PROJECT_ROOT)), "metrics": m, "stats": res.stats,
                        "warnings": res.warnings})
        print(f"saved {path}  IoU={f(m['flood_iou'])}")
    (out_dir / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
