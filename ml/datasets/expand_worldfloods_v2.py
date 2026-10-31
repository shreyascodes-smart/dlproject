"""
Expand the local WorldFloods v2 subset with EVENT DIVERSITY (not more of the same events).

Why: the original download took the first 150 train scenes alphabetically, covering only
28 of 116 training events. This adds scenes from events with no local data, cheapest
scene first, under a hard disk budget.

Rules:
  * Only WorldFloods v2 train-split rows (never val/test rows).
  * Never an event that appears in the frozen validation or test split (no leakage).
  * Up to `--per-event` scenes per new event, smallest S2 file first, until `--budget-gb`.
  * Downloads S2, gt, floodmaps, PERMANENTWATERJRC and meta for each chosen scene from the
    pinned Hugging Face revision; files are verified by size and SHA-256 against the
    repository's LFS index where available. Existing files are never overwritten.
  * Writes a manifest (data/metadata/worldfloods_expansion_<date>.json) with provenance.

Licence: WorldFloods v2 is CC BY-NC 4.0 (Portalés-Julià et al., 2023, Sci. Rep. 13, 20316).
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "raw" / "flood" / "worldfloods_v2"
REPO = "isp-uv-es/WorldFloodsv2"
REVISION = "1f3faa2989e69930ac31d5c0a79fd224461123f8"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
MANIFEST = PROJECT_ROOT / "data" / "processed" / "flood" / "segmentation_v1" / "seg_split_manifest.json"
FOLDERS = (("S2", ".tif"), ("gt", ".tif"), ("PERMANENTWATERJRC", ".tif"), ("floodmaps", ".geojson"), ("meta", ".json"))
HARD_CAP_GB = 90.0


def file_index():
    tree = next((DATA / ".cache" / "huggingface" / "trees").iterdir())
    raw = tree.read_text()
    return json.loads(raw[raw.find("{"):])["files"]


def dir_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(rel: str, dst: Path, expected_size: int, expected_sha: str = None, retries: int = 4) -> str:
    if dst.exists() and dst.stat().st_size == expected_size:
        return "exists"
    tmp = dst.with_suffix(dst.suffix + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{rel}", timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if tmp.stat().st_size != expected_size:
                raise IOError(f"size {tmp.stat().st_size} != {expected_size}")
            if expected_sha and sha256(tmp) != expected_sha:
                raise IOError("sha256 mismatch")
            tmp.replace(dst)
            return "downloaded"
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"{rel}: {e}") from e
            time.sleep(3 * attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-gb", type=float, default=12.0, help="max NEW data to add")
    ap.add_argument("--per-event", type=int, default=2)
    ap.add_argument("--max-scene-mb", type=float, default=900.0, help="skip very large scenes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=4, help="parallel downloads")
    a = ap.parse_args()

    files = file_index()
    rows = list(csv.DictReader(open(DATA / "dataset_metadata.csv")))
    manifest = json.loads(MANIFEST.read_text())
    frozen = set(manifest["splits"]["val"]["event_ids"]) | set(manifest["splits"]["test"]["event_ids"])
    local = {p.stem for p in (DATA / "train" / "S2").glob("*.tif")}
    local_events = {r["ems_code"] for r in rows if r["event id"] in local}

    def size(folder, name, ext):
        return (files.get(f"train/{folder}/{name}{ext}") or {}).get("size") or 0

    cands = {}
    for r in rows:
        name, ev = r["event id"], r["ems_code"]
        if r["split"] != "train" or name in local or r["source"] != "CopernicusEMS":
            continue
        if ev in frozen or ev in local_events or not ev:
            continue
        if any(f"train/{fo}/{name}{ext}" not in files for fo, ext in FOLDERS):
            continue
        s2 = size("S2", name, ".tif")
        if s2 == 0 or s2 / 1e6 > a.max_scene_mb:
            continue
        cands.setdefault(ev, []).append((sum(size(fo, name, ext) for fo, ext in FOLDERS), name, r))

    # Round-robin over events (1st scene of every event, then 2nd, ...) so the budget favours diversity
    for ev in cands:
        cands[ev].sort()
    chosen, total = [], 0
    for k in range(a.per_event):
        for ev in sorted(cands, key=lambda e: cands[e][0][0]):
            if k < len(cands[ev]):
                sz, name, r = cands[ev][k]
                if (total + sz) / 1e9 > a.budget_gb:
                    continue
                chosen.append((name, ev, r, sz))
                total += sz

    current = dir_size_gb(PROJECT_ROOT / "data")
    print(f"Current data/ footprint: {current:.2f} GB; adding {total/1e9:.2f} GB "
          f"({len(chosen)} scenes, {len({c[1] for c in chosen})} new events)")
    if current + total / 1e9 > HARD_CAP_GB:
        sys.exit(f"Refusing: would exceed the {HARD_CAP_GB} GB data cap.")
    if a.dry_run:
        for name, ev, _, sz in chosen:
            print(f"  {ev:10s} {name:50s} {sz/1e6:8.1f} MB")
        return

    out = {"created": datetime.now().isoformat(), "source": f"huggingface.co/datasets/{REPO}@{REVISION}",
           "licence": "CC BY-NC 4.0", "budget_gb": a.budget_gb, "per_event": a.per_event,
           "excluded_events_frozen_splits": sorted(frozen), "scenes": []}
    t0 = time.time()

    def fetch(item):
        name, ev, r, sz = item
        rec = {"name": name, "event_id": ev, "satellite_date": r["satellite date"], "s2_date": r["s2_date"],
               "crs": r["crs"], "files": {}}
        try:
            for fo, ext in FOLDERS:
                rel = f"train/{fo}/{name}{ext}"
                meta = files[rel]
                sha = meta.get("lfs_sha256")  # LFS files carry a SHA-256 in the HF index; small files do not
                status = download(rel, DATA / "train" / fo / f"{name}{ext}", meta["size"], sha)
                rec["files"][fo] = {"size": meta["size"], "sha256": sha, "status": status}
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = str(e)[:200]
        return rec, sz

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=a.threads) as pool:
        futs = [pool.submit(fetch, c) for c in chosen]
        for i, f in enumerate(as_completed(futs), 1):
            rec, sz = f.result()
            out["scenes"].append(rec)
            print(f"[{i:3d}/{len(chosen)}] {rec['event_id']:10s} {rec['name']:48s} {sz/1e6:7.1f} MB "
                  f"{'ok' if rec['ok'] else 'FAILED ' + rec.get('error', '')} ({time.time()-t0:.0f}s)", flush=True)
    path = PROJECT_ROOT / "data" / "metadata" / f"worldfloods_expansion_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}; data/ now {dir_size_gb(PROJECT_ROOT / 'data'):.2f} GB")


if __name__ == "__main__":
    main()
