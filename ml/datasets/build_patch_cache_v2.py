"""
Build the patch cache for the v2 multispectral surface-water model.

Why a cache: WorldFloods S2 GeoTIFFs are 15-band pixel-interleaved deflate tiles, so every
random window read decompresses all 15 bands (~80 ms). v1 training read whole rasters every
step (~550 s/epoch). Extracting fixed 256 px patches once makes an epoch take seconds of I/O.

Inputs per patch: Sentinel-2 L1C B02, B03, B04, B08, B11, B12 (WorldFloods bands 2,3,4,8,12,13),
uint16 DN (negatives clipped to 0) — the same six bands the live Copernicus request returns.

Targets:
  train  : surface water from the WorldFloods ground truth (gt band 2: 1 land, 2 water;
           band 1: 1 clear, 2 cloud, 0 invalid) -> 0 land, 1 water, 255 ignore (cloud/invalid/no data).
  val    : every 256 px tile (grid) of each frozen validation scene, with the EMS flood reference
           (stored seg + validity), JRC permanent water and the WorldFloods water target, so the
           training loop can score the PRODUCT metric (flood = water & ~permanent vs EMS flood).

Split safety: training scenes are every local scene whose event is NOT in the frozen
validation/test event lists of seg_split_manifest.json. The test split is never read.

Output: data/interim/flood/patches_v2/{train,val}/<scene>.{x,y,...}.npy + index.json
"""
import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "raw" / "flood" / "worldfloods_v2"
SEG = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1"
OUT = PROJECT_ROOT / "data" / "interim" / "flood" / "patches_v2"
BANDS_1IDX = (2, 3, 4, 8, 12, 13)          # B02 B03 B04 B08 B11 B12
BAND_NAMES = ("B02", "B03", "B04", "B08", "B11", "B12")
P = 256
IGNORE = 255


def event_of(name: str, meta_rows: dict) -> str:
    r = meta_rows.get(name)
    if r and r.get("ems_code"):
        return r["ems_code"]
    if name.startswith("EMSR"):
        return name.split("_")[0]
    return name.split("_")[0] if name[:8].isdigit() else name


def water_target(gt1: np.ndarray, gt2: np.ndarray) -> np.ndarray:
    t = np.full(gt1.shape, IGNORE, np.uint8)
    ok = (gt1 == 1) & (gt2 > 0)
    t[ok & (gt2 == 1)] = 0
    t[ok & (gt2 == 2)] = 1
    return t


def read_windows(path: Path, wins, bands=BANDS_1IDX) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window

    out = np.zeros((len(wins), len(bands), P, P), np.uint16)
    with rasterio.open(path) as src:
        for i, (y, x) in enumerate(wins):
            a = src.read(list(bands), window=Window(x, y, P, P), boundless=True, fill_value=0)
            out[i] = np.clip(a, 0, None).astype(np.uint16)
    return out


def train_scene(task):
    name, osplit, seed, max_patches = task
    import rasterio

    s2 = DATA / osplit / "S2" / f"{name}.tif"
    gt = DATA / osplit / "gt" / f"{name}.tif"
    dst = OUT / "train"
    if (dst / f"{name}.y.npy").exists():
        return name, "exists", int(np.load(dst / f"{name}.y.npy", mmap_mode="r").shape[0])
    with rasterio.open(s2) as a, rasterio.open(gt) as b:
        if (a.height, a.width) != (b.height, b.width) or a.transform != b.transform or a.count < 13:
            return name, "skip: grid/band mismatch", 0
        gt1, gt2 = b.read(1), b.read(2)
    tgt = water_target(gt1, gt2)
    del gt1, gt2
    h, w = tgt.shape
    if h < P or w < P:
        return name, "skip: smaller than one patch", 0
    rng = np.random.default_rng(seed)
    oy, ox = rng.integers(0, min(P, h - P + 1)), rng.integers(0, min(P, w - P + 1))
    cands = []
    for y in range(int(oy), h - P + 1, P):
        for x in range(int(ox), w - P + 1, P):
            t = tgt[y:y + P, x:x + P]
            valid = (t != IGNORE).mean()
            if valid >= 0.5:
                wat = (t == 1).sum() / max((t != IGNORE).sum(), 1)
                cands.append((y, x, float(wat)))
    if not cands:
        return name, "skip: no valid patches", 0
    k = int(np.clip(len(cands) // 4, 8, max_patches))
    mixed = [c for c in cands if 0.02 <= c[2] <= 0.98]
    rest = [c for c in cands if not (0.02 <= c[2] <= 0.98)]
    rng.shuffle(mixed)
    rng.shuffle(rest)
    take = mixed[: int(math.ceil(k * 0.6))]
    take += rest[: k - len(take)]
    if len(take) < k:
        take += mixed[len(take): k]
    wins = [(y, x) for y, x, _ in take]
    X = read_windows(s2, wins)
    Y = np.stack([tgt[y:y + P, x:x + P] for y, x in wins]).astype(np.uint8)
    Y[np.all(X == 0, axis=1)] = IGNORE  # S2 no data
    dst.mkdir(parents=True, exist_ok=True)
    np.save(dst / f"{name}.x.npy", X)
    np.save(dst / f"{name}.y.npy", Y)
    return name, "ok", len(wins)


def val_scene(task):
    name, osplit = task
    import rasterio

    dst = OUT / "val"
    if (dst / f"{name}.ems.npy").exists():
        return name, "exists", int(np.load(dst / f"{name}.ems.npy", mmap_mode="r").shape[0])
    s2 = DATA / osplit / "S2" / f"{name}.tif"
    with rasterio.open(DATA / osplit / "gt" / f"{name}.tif") as b:
        tgt = water_target(b.read(1), b.read(2))
    with rasterio.open(SEG / "masks" / f"{name}_seg.tif") as r:
        seg = r.read(1).astype(np.uint8)
    with rasterio.open(SEG / "masks" / f"{name}_validity.tif") as r:
        seg[r.read(1) == 0] = 0
    with rasterio.open(DATA / osplit / "PERMANENTWATERJRC" / f"{name}.tif") as r:
        pw = (r.read(1) == 3).astype(np.uint8)
    h, w = seg.shape
    wins = [(y, x) for y in range(0, h, P) for x in range(0, w, P)]

    def cut(a, fill):
        out = np.full((len(wins), P, P), fill, a.dtype)
        for i, (y, x) in enumerate(wins):
            t = a[y:y + P, x:x + P]
            out[i, : t.shape[0], : t.shape[1]] = t
        return out

    X = read_windows(s2, wins)
    dst.mkdir(parents=True, exist_ok=True)
    np.save(dst / f"{name}.x.npy", X)
    np.save(dst / f"{name}.y.npy", cut(tgt, IGNORE))
    np.save(dst / f"{name}.ems.npy", cut(seg, 0))     # 0 ignore, 1 non-flood, 2 flood
    np.save(dst / f"{name}.pw.npy", cut(pw, 0))
    np.save(dst / f"{name}.win.npy", np.array(wins, np.int32))
    return name, "ok", len(wins)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-patches", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    manifest = json.loads((SEG / "seg_split_manifest.json").read_text())
    frozen = set(manifest["splits"]["val"]["event_ids"]) | set(manifest["splits"]["test"]["event_ids"])
    meta_rows = {r["event id"]: r for r in csv.DictReader(open(DATA / "dataset_metadata.csv"))}

    train, skipped = [], []
    for osplit in ("train", "val", "test"):
        for p in sorted((DATA / osplit / "S2").glob("*.tif")):
            ev = event_of(p.stem, meta_rows)
            if ev in frozen:
                skipped.append((p.stem, ev))
                continue
            if not (DATA / osplit / "gt" / f"{p.stem}.tif").exists():
                continue
            train.append((p.stem, osplit, ev))
    val = [(s["name"], s["original_split"]) for s in manifest["splits"]["val"]["samples"]]
    print(f"train scenes: {len(train)} from {len({e for *_, e in train})} events "
          f"| skipped (frozen val/test events): {len(skipped)} | val scenes: {len(val)}")

    t0 = time.time()
    results = {"train": {}, "val": {}}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(train_scene, (n, o, a.seed + i, a.max_patches)): ("train", n, ev)
                for i, (n, o, ev) in enumerate(train)}
        futs.update({ex.submit(val_scene, t): ("val", t[0], None) for t in val})
        for i, f in enumerate(as_completed(futs), 1):
            kind, name, ev = futs[f]
            try:
                n, status, k = f.result()
            except Exception as e:
                status, k = f"error: {type(e).__name__}: {e}"[:160], 0
            results[kind][name] = {"status": status, "n": k, "event": ev}
            if i % 20 == 0 or not status.startswith(("ok", "exists")):
                print(f"[{i}/{len(futs)}] {kind} {name}: {status} ({k}) {time.time()-t0:.0f}s", flush=True)

    # Band statistics on valid training pixels (reflectance = DN/10000 clipped to [0, 1])
    s = np.zeros(6)
    s2 = np.zeros(6)
    n = 0
    rng = np.random.default_rng(a.seed)
    for name, r in results["train"].items():
        if r["n"] == 0:
            continue
        X = np.load(OUT / "train" / f"{name}.x.npy", mmap_mode="r")
        Y = np.load(OUT / "train" / f"{name}.y.npy", mmap_mode="r")
        for i in rng.choice(X.shape[0], size=min(4, X.shape[0]), replace=False):
            v = Y[i] != IGNORE
            if v.sum() == 0:
                continue
            x = np.clip(X[i][:, v].astype(np.float64) / 10000.0, 0, 1)
            s += x.sum(1)
            s2 += (x ** 2).sum(1)
            n += x.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    water = []
    for name, r in results["train"].items():
        if r["n"]:
            Y = np.load(OUT / "train" / f"{name}.y.npy", mmap_mode="r")
            water.append(((Y == 1).sum(), (Y != IGNORE).sum()))
    wp = sum(w for w, _ in water) / max(sum(v for _, v in water), 1)
    index = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bands": BAND_NAMES, "worldfloods_bands_1idx": BANDS_1IDX, "patch": P,
        "scaling": "reflectance = clip(DN / 10000, 0, 1); then (x - mean) / std",
        "mean": mean.tolist(), "std": std.tolist(), "train_water_fraction": wp,
        "frozen_events_excluded": sorted(frozen),
        "train": {k: v for k, v in results["train"].items() if v["n"] > 0},
        "train_skipped": {k: v for k, v in results["train"].items() if v["n"] == 0},
        "val": results["val"],
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    n_tr = sum(v["n"] for v in index["train"].values())
    size = sum(f.stat().st_size for f in OUT.rglob("*.npy")) / 1e9
    print(f"\nDone in {time.time()-t0:.0f}s: {n_tr} train patches from {len(index['train'])} scenes "
          f"({len({v['event'] for v in index['train'].values()})} events), "
          f"{sum(v['n'] for v in results['val'].values())} val tiles; cache {size:.2f} GB; water {wp:.3f}")
    print("mean", np.round(mean, 4).tolist(), "std", np.round(std, 4).tolist())


if __name__ == "__main__":
    sys.exit(main())
