"""Check label distribution across all splits to understand model bias."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.datasets.flood_dataset import WorldFloodsClassificationDataset

data_dir = r"d:\superiorparab\final year DL project\data\raw\flood\worldfloods_v2"

for split in ["train", "val", "test"]:
    print(f"\n=== {split.upper()} ===")
    try:
        ds = WorldFloodsClassificationDataset(data_dir, split=split, flood_threshold=0.05)
        flood = 0
        non_flood = 0
        for i in range(len(ds)):
            _, label = ds[i]
            if label.item() == 1:
                flood += 1
            else:
                non_flood += 1
        total = flood + non_flood
        print(f"  Total: {total}")
        print(f"  Flood: {flood} ({100*flood/total:.1f}%)")
        print(f"  Non-flood: {non_flood} ({100*non_flood/total:.1f}%)")
    except Exception as e:
        print(f"  Error: {e}")
