"""
Phase C Step 3: Build leakage-safe event-level splits.

Groups all samples from the same flood event into a single split.
Uses metadata JSON (ems_code, event id) as the authoritative event identifier.
Produces split manifests under data/processed/flood/splits_v2/.

Does NOT modify or overwrite original train/val/test directories.
"""
import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import rasterio

PROJECT_ROOT = r"d:\superiorparab\final year DL project"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "flood", "worldfloods_v2")
sys.path.insert(0, PROJECT_ROOT)

from ml.datasets.flood_dataset import WorldFloodsClassificationDataset, LABEL_UNKNOWN

# Fixed seed for reproducibility
SEED = 42
FLOOD_THRESHOLD = 0.05

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "flood", "splits_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_event_id_from_metadata(meta_path):
    """Extract the authoritative event identifier from metadata JSON."""
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    # Prefer ems_code (e.g. "EMSR264"), then "event id"
    ems_code = meta.get("ems_code", "")
    event_id = meta.get("event id", "")
    if ems_code:
        return ems_code
    if event_id:
        return str(event_id)
    return None


def get_event_id_from_filename(name):
    """Fallback: extract event prefix from filename.
    EMSR264_08VATOMANDRY_DEL_v2 -> EMSR264
    01042016_Choctawhatchee_River_near_Bellwood_AL -> 01042016 (date-based)
    RS2_20161008_... -> RS2_20161008
    ST1_20161014_... -> ST1_20161014
    """
    parts = name.split("_")
    if parts[0].startswith("EMSR"):
        return parts[0]
    elif parts[0].isdigit() and len(parts[0]) == 8:
        # Date-based event prefix (e.g. 01042016)
        return parts[0]
    elif len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def compute_label(ds, sample):
    """Compute label for a sample using the corrected mask logic."""
    try:
        window = ds._compute_shared_window(
            sample["s2"], sample["gt"], sample["pw"], sample["name"]
        )
        cloud_mask, water_mask = ds._read_gt_masks(sample["gt"], window)
        pw_mask = (
            ds._read_pw_mask(sample["pw"], window)
            if sample["pw"] is not None
            else None
        )
        label, flood_ratio = ds._derive_label(cloud_mask, water_mask, pw_mask)
        return label, flood_ratio, None
    except Exception as e:
        return None, None, str(e)


print(f"Phase C Step 3: Build Leakage-Safe Event Splits")
print(f"Data root: {DATA_ROOT}")
print(f"Seed: {SEED}")
print(f"Timestamp: {datetime.now().isoformat()}")

# ---- Discover all samples across all original splits ----
all_samples = []

for orig_split in ["train", "val", "test"]:
    try:
        ds = WorldFloodsClassificationDataset(
            data_dir=DATA_ROOT,
            split=orig_split,
            flood_threshold=FLOOD_THRESHOLD,
            use_permanent_water=True,
        )
    except FileNotFoundError:
        continue

    for sample in ds.samples:
        name = sample["name"]

        # Get event ID from metadata first, filename as fallback
        meta_dir = os.path.join(DATA_ROOT, orig_split, "meta")
        meta_path = os.path.join(meta_dir, name + ".json")
        event_id = get_event_id_from_metadata(meta_path)
        event_source = "metadata"
        if event_id is None:
            event_id = get_event_id_from_filename(name)
            event_source = "filename"

        # Compute label
        label, flood_ratio, error = compute_label(ds, sample)

        entry = {
            "name": name,
            "original_split": orig_split,
            "event_id": event_id,
            "event_source": event_source,
            "s2_path": str(sample["s2"]),
            "gt_path": str(sample["gt"]),
            "pw_path": str(sample["pw"]) if sample["pw"] else None,
        }

        if error:
            entry["label"] = None
            entry["label_status"] = "error"
            entry["flood_ratio"] = None
            entry["exclusion_reason"] = error
        elif label == LABEL_UNKNOWN:
            entry["label"] = None
            entry["label_status"] = "unknown"
            entry["flood_ratio"] = 0.0
            entry["exclusion_reason"] = "No valid observation pixels in center window"
        else:
            entry["label"] = label
            entry["label_status"] = "ok"
            entry["flood_ratio"] = round(flood_ratio, 6)
            entry["exclusion_reason"] = None

        all_samples.append(entry)

print(f"\nTotal samples discovered: {len(all_samples)}")

# ---- Separate usable from excluded ----
usable = [s for s in all_samples if s["label_status"] == "ok"]
excluded = [s for s in all_samples if s["label_status"] != "ok"]

print(f"Usable samples: {len(usable)}")
print(f"Excluded samples: {len(excluded)}")
for e in excluded:
    print(f"  EXCLUDED: {e['name']} ({e['label_status']}): {e['exclusion_reason']}")

# ---- Check for duplicate filenames ----
name_counts = defaultdict(int)
for s in all_samples:
    name_counts[s["name"]] += 1
duplicates = {k: v for k, v in name_counts.items() if v > 1}
if duplicates:
    print(f"\nWARNING: {len(duplicates)} duplicate filenames across splits:")
    for k, v in duplicates.items():
        print(f"  {k}: appears {v} times")
else:
    print(f"\nNo duplicate filenames found.")

# ---- Group by event ----
event_groups = defaultdict(list)
for s in usable:
    event_groups[s["event_id"]].append(s)

events = sorted(event_groups.keys())
print(f"\nDistinct events (usable): {len(events)}")

# Event-level stats
event_stats = []
for eid in events:
    samples = event_groups[eid]
    n_flood = sum(1 for s in samples if s["label"] == 1)
    n_non = sum(1 for s in samples if s["label"] == 0)
    event_stats.append({
        "event_id": eid,
        "total": len(samples),
        "flood": n_flood,
        "non_flood": n_non,
        "has_flood": n_flood > 0,
    })
    print(f"  {eid}: {len(samples)} samples ({n_flood} flood, {n_non} non-flood)")

# ---- Split events: ~70% train, ~15% val, ~15% test ----
# Prioritize: every split should have at least some flood events
flood_events = [e for e in event_stats if e["has_flood"]]
nonflood_events = [e for e in event_stats if not e["has_flood"]]

print(f"\nFlood events: {len(flood_events)}")
print(f"Non-flood-only events: {len(nonflood_events)}")

rng = random.Random(SEED)

# Shuffle both lists
rng.shuffle(flood_events)
rng.shuffle(nonflood_events)

# Calculate target counts
total_events = len(events)
target_val = max(2, round(total_events * 0.15))
target_test = max(2, round(total_events * 0.15))

# Split flood events proportionally
n_flood_val = max(1, round(len(flood_events) * 0.15))
n_flood_test = max(1, round(len(flood_events) * 0.15))
n_flood_train = len(flood_events) - n_flood_val - n_flood_test

flood_train = flood_events[:n_flood_train]
flood_val = flood_events[n_flood_train:n_flood_train + n_flood_val]
flood_test = flood_events[n_flood_train + n_flood_val:]

# Split non-flood events similarly
n_nf_val = max(0, target_val - n_flood_val)
n_nf_test = max(0, target_test - n_flood_test)
n_nf_train = len(nonflood_events) - n_nf_val - n_nf_test

nf_train = nonflood_events[:max(0, n_nf_train)]
nf_val = nonflood_events[max(0, n_nf_train):max(0, n_nf_train) + n_nf_val]
nf_test = nonflood_events[max(0, n_nf_train) + n_nf_val:]

# Combine
train_events = [e["event_id"] for e in flood_train + nf_train]
val_events = [e["event_id"] for e in flood_val + nf_val]
test_events = [e["event_id"] for e in flood_test + nf_test]

# Verify no overlap
train_set = set(train_events)
val_set = set(val_events)
test_set = set(test_events)

assert train_set.isdisjoint(val_set), "train/val event overlap!"
assert train_set.isdisjoint(test_set), "train/test event overlap!"
assert val_set.isdisjoint(test_set), "val/test event overlap!"
assert len(train_set | val_set | test_set) == len(events), "Missing events!"

print(f"\n{'='*60}")
print(f"  LEAKAGE-SAFE SPLIT ASSIGNMENT")
print(f"{'='*60}")

# Build final manifests
manifests = {}
for split_name, event_ids in [("train", train_events), ("val", val_events), ("test", test_events)]:
    samples = []
    for eid in event_ids:
        samples.extend(event_groups[eid])
    
    n_flood = sum(1 for s in samples if s["label"] == 1)
    n_non = sum(1 for s in samples if s["label"] == 0)
    n_events = len(event_ids)

    print(f"\n  {split_name}:")
    print(f"    Events: {n_events}")
    print(f"    Samples: {len(samples)}")
    print(f"    Flood: {n_flood} ({n_flood/max(len(samples),1)*100:.1f}%)")
    print(f"    Non-Flood: {n_non} ({n_non/max(len(samples),1)*100:.1f}%)")

    manifests[split_name] = {
        "split": split_name,
        "event_ids": sorted(event_ids),
        "n_events": n_events,
        "n_samples": len(samples),
        "n_flood": n_flood,
        "n_non_flood": n_non,
        "flood_pct": round(n_flood / max(len(samples), 1) * 100, 2),
        "samples": sorted(samples, key=lambda x: x["name"]),
    }

# Save manifests
output = {
    "meta": {
        "created": datetime.now().isoformat(),
        "seed": SEED,
        "flood_threshold": FLOOD_THRESHOLD,
        "data_root": DATA_ROOT,
        "total_samples": len(all_samples),
        "usable_samples": len(usable),
        "excluded_samples": len(excluded),
        "total_events": len(events),
        "label_task_description": (
            "Proxy classification of non-permanent surface water presence. "
            "Band 2 of the WorldFloods GT mask records ALL detected water, "
            "not flood-specific water. Permanent water (JRC value 3) is excluded. "
            "The resulting 'flood' label means 'significant non-permanent water detected', "
            "which is a useful but imperfect proxy for actual flooding."
        ),
    },
    "splits": manifests,
    "excluded": excluded,
    "leakage_proof": {
        "train_events": sorted(train_events),
        "val_events": sorted(val_events),
        "test_events": sorted(test_events),
        "train_val_overlap": sorted(train_set & val_set),
        "train_test_overlap": sorted(train_set & test_set),
        "val_test_overlap": sorted(val_set & test_set),
    },
}

manifest_path = os.path.join(OUTPUT_DIR, "split_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nManifest saved to: {manifest_path}")

# Also save per-split sample lists for easy loading
for split_name, manifest in manifests.items():
    sample_list = [s["name"] for s in manifest["samples"]]
    list_path = os.path.join(OUTPUT_DIR, f"{split_name}_samples.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(sample_list))
    print(f"  {split_name} sample list: {list_path}")

# Save excluded samples
excluded_path = os.path.join(OUTPUT_DIR, "excluded_samples.json")
with open(excluded_path, "w") as f:
    json.dump(excluded, f, indent=2)
print(f"  Excluded samples: {excluded_path}")

print(f"\nDone.")
