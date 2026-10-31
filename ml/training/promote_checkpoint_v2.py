"""
Promote a validated v2 training run into the served model directory.

  python ml/training/promote_checkpoint_v2.py --source scratch/v2_runC_ema --rule "..."

What it does (never overwrites anything):
  * loads the run's best_seg_model.pt (a local file written by train_flood_seg_v2.py),
  * rewrites its metadata as plain JSON types so the served copy loads with
    torch.load(weights_only=True) — no arbitrary pickled objects,
  * writes models/flood/segmentation_v2/{best_seg_model.pt, training_config.json, train_log.jsonl,
    PROVENANCE.json} and verifies the safe load and the weights.
Refuses to write into the legacy classifier directory, segmentation_v1, or an existing destination.
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = PROJECT_ROOT / "models" / "flood"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plain(obj):
    """JSON round-trip: str subclasses (e.g. TorchVersion), numpy scalars, tuples -> plain types."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="training run directory (contains best_seg_model.pt)")
    ap.add_argument("--dest", default=str(MODELS / "segmentation_v2"))
    ap.add_argument("--rule", required=True, help="selection rule / evidence, recorded in PROVENANCE.json")
    a = ap.parse_args()

    src, dest = Path(a.source), Path(a.dest)
    if dest.resolve() in (MODELS.resolve(), (MODELS / "segmentation_v1").resolve()):
        sys.exit("Refusing to write into the legacy classifier or v1 directory.")
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"Refusing to overwrite existing {dest}")
    ck_src = src / "best_seg_model.pt"
    ck = torch.load(ck_src, map_location="cpu", weights_only=False)  # our own local training output
    state = ck["model_state_dict"]
    clean = {
        "epoch": int(ck["epoch"]),
        "model_state_dict": state,
        "config": plain(ck["config"]),
        "val_metrics": plain(ck.get("val_metrics", {})),
    }
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "best_seg_model.pt"
    torch.save(clean, out)

    # Verify: safe load + identical weights
    back = torch.load(out, map_location="cpu", weights_only=True)
    assert back["config"]["version"] == "segmentation_v2", "unexpected model version"
    assert set(back["model_state_dict"]) == set(state), "state-dict keys changed"
    assert all(torch.equal(back["model_state_dict"][k], state[k]) for k in state), "weights changed"

    cfg = plain(json.loads((src / "training_config.json").read_text()))
    cfg["selected_epoch"] = clean["epoch"]
    cfg["selected_val_metrics_tiles"] = clean["val_metrics"]
    (dest / "training_config.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy2(src / "train_log.jsonl", dest / "train_log.jsonl")
    prov = {
        "promoted": datetime.now().isoformat(timespec="seconds"),
        "source_run": (src.resolve().relative_to(PROJECT_ROOT).as_posix()
                       if src.resolve().is_relative_to(PROJECT_ROOT) else src.name),
        "source_sha256": sha256(ck_src),
        "served_sha256": sha256(out),
        "epoch": clean["epoch"],
        "selection_rule": a.rule,
        "loads_with_weights_only": True,
    }
    (dest / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
