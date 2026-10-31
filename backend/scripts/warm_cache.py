"""
Pre-compute demo assessments so every sample opens instantly during the demo.

Results are stored in data/cache/seg_api/ (git-ignored) and reused by the API
as long as the checkpoint file is unchanged. Safe to re-run: cached samples
are skipped. Reads only; never changes data, masks or weights.

Usage (from the project root):
    .\\venv\\Scripts\\python.exe backend\\scripts\\warm_cache.py                 # tiled, all samples
    .\\venv\\Scripts\\python.exe backend\\scripts\\warm_cache.py --methods tiled resize
    .\\venv\\Scripts\\python.exe backend\\scripts\\warm_cache.py --split val
"""
import argparse
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parent))

from app.services.seg_service import get_seg_service  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["tiled"], choices=["tiled", "resize"])
    ap.add_argument("--split", choices=["val", "test"], default=None, help="only this split (default: both)")
    a = ap.parse_args()

    svc = get_seg_service()
    samples = [s for s in svc.catalogue() if a.split in (None, s.split)]
    print(f"Model on {svc.segmenter().device}; {len(samples)} samples x {len(a.methods)} method(s)")
    t_all = time.time()
    for i, s in enumerate(samples, 1):
        for m in a.methods:
            t = time.time()
            r = svc.assess_demo(s.id, m)
            print(f"[{i:2d}/{len(samples)}] {s.split:4s} {s.id:46s} {m:6s} "
                  f"{'cached' if r['cached'] else f'{time.time() - t:5.1f}s':>7s} "
                  f"coverage={r['prediction']['flood_coverage_pct']}%")
    print(f"Done in {time.time() - t_all:.0f}s. Cache: {svc.cache_dir}")


if __name__ == "__main__":
    main()
