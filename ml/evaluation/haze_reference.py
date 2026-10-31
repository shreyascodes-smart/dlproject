"""
Haze range of the TRAINING images, used as an out-of-distribution check for live scenes.

Haze index of a scene = 5th percentile of Sentinel-2 B02 (blue) top-of-atmosphere reflectance over
its valid pixels. Atmospheric haze adds path radiance that raises even the darkest pixels in the blue
band, so the darkest 5 % of a clear scene stay low while a hazy scene's darkest pixels are bright.

Reads only the training patch cache (data/interim/flood/patches_v2/train); validation and test
scenes are not used. Writes the per-scene distribution and the 99th percentile.

  python ml/evaluation/haze_reference.py --output data/reports/haze_reference_training_v2.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = PROJECT_ROOT / "data" / "interim" / "flood" / "patches_v2"
IGNORE = 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    index = json.loads((CACHE / "index.json").read_text())
    b02 = list(index["bands"]).index("B02")
    per_scene = {}
    for name in sorted(index["train"]):
        X = np.load(CACHE / "train" / f"{name}.x.npy", mmap_mode="r")
        Y = np.load(CACHE / "train" / f"{name}.y.npy", mmap_mode="r")
        vals = []
        for i in range(X.shape[0]):
            v = (np.asarray(Y[i]) != IGNORE) & (np.asarray(X[i]).sum(0) > 0)
            if v.any():
                vals.append(np.asarray(X[i][b02])[v])
        if vals:
            per_scene[name] = round(float(np.percentile(np.concatenate(vals), 5)) / 10000.0, 4)
    arr = np.array(list(per_scene.values()))
    q = {p: round(float(np.percentile(arr, p)), 4) for p in (50, 90, 95, 99)}
    report = {
        "kind": "haze_reference", "timestamp": datetime.now().isoformat(timespec="seconds"),
        "definition": "5th percentile of B02 TOA reflectance (DN/10000) over valid pixels of each training scene "
                      "(sampled 256 px patches)",
        "n_scenes": len(per_scene), "quantiles": q, "max": round(float(arr.max()), 4),
        "per_scene": per_scene,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"{len(per_scene)} training scenes; haze index quantiles {q}; max {arr.max():.4f}")
    print("highest:", sorted(per_scene.items(), key=lambda kv: -kv[1])[:6])


if __name__ == "__main__":
    main()
