"""
WorldFloods v2 Dataset Inspector
Inspects the downloaded TIFF files to understand:
- Number of bands (channels)
- Image dimensions
- Data types
- Mask values
- Band statistics
This information is needed before we can design the training pipeline.
"""
import os
import sys
from pathlib import Path
import numpy as np

# We need rasterio for reading GeoTIFF files
try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)


def inspect_tiff(filepath):
    """Inspect a single TIFF file and return its properties."""
    with rasterio.open(filepath) as src:
        return {
            'path': str(filepath),
            'filename': filepath.name,
            'bands': src.count,
            'width': src.width,
            'height': src.height,
            'dtype': str(src.dtypes[0]),
            'crs': str(src.crs) if src.crs else None,
            'bounds': src.bounds,
            'nodata': src.nodata,
            'transform': src.transform,
        }


def inspect_mask(filepath):
    """Inspect a ground truth mask file."""
    with rasterio.open(filepath) as src:
        data = src.read()
        unique_values = np.unique(data)
        return {
            'path': str(filepath),
            'filename': filepath.name,
            'bands': src.count,
            'width': src.width,
            'height': src.height,
            'dtype': str(src.dtypes[0]),
            'unique_values': unique_values.tolist(),
            'value_counts': {int(v): int(np.sum(data == v)) for v in unique_values},
            'min': float(np.nanmin(data)),
            'max': float(np.nanmax(data)),
        }


def inspect_dataset(data_dir):
    """Inspect the downloaded WorldFloods v2 dataset."""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory does not exist: {data_dir}")
        return
    
    print("=" * 70)
    print("WorldFloods v2 Dataset Inspection")
    print("=" * 70)
    
    # Check directory structure
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"\n[MISSING] {split}/ directory not found")
            continue
        
        print(f"\n--- {split.upper()} ---")
        for subdir in ['S2', 'gt', 'PERMANENTWATERJRC', 'floodmaps', 'meta']:
            sub_path = split_dir / subdir
            if sub_path.exists():
                files = list(sub_path.iterdir())
                print(f"  {subdir}/: {len(files)} files")
            else:
                print(f"  {subdir}/: [MISSING]")
    
    # Inspect representative files
    print("\n" + "=" * 70)
    print("Representative File Inspection")
    print("=" * 70)
    
    # Find first available S2 image
    for split in ['test', 'val', 'train']:
        s2_dir = data_dir / split / 'S2'
        if s2_dir.exists():
            tif_files = sorted(s2_dir.glob('*.tif'))
            if tif_files:
                print(f"\n--- Sentinel-2 Image ({split}) ---")
                info = inspect_tiff(tif_files[0])
                for k, v in info.items():
                    print(f"  {k}: {v}")
                
                # Read band statistics
                with rasterio.open(tif_files[0]) as src:
                    print(f"\n  Band statistics:")
                    for i in range(1, min(src.count + 1, 14)):
                        band = src.read(i)
                        valid = band[~np.isnan(band)] if np.issubdtype(band.dtype, np.floating) else band
                        print(f"    Band {i}: min={np.nanmin(valid):.4f}, max={np.nanmax(valid):.4f}, "
                              f"mean={np.nanmean(valid):.4f}, dtype={band.dtype}")
                break
    
    # Find first available mask
    for split in ['test', 'val', 'train']:
        gt_dir = data_dir / split / 'gt'
        if gt_dir.exists():
            tif_files = sorted(gt_dir.glob('*.tif'))
            if tif_files:
                print(f"\n--- Ground Truth Mask ({split}) ---")
                info = inspect_mask(tif_files[0])
                for k, v in info.items():
                    print(f"  {k}: {v}")
                break
    
    # Find first available permanent water mask
    for split in ['test', 'val', 'train']:
        pw_dir = data_dir / split / 'PERMANENTWATERJRC'
        if pw_dir.exists():
            tif_files = sorted(pw_dir.glob('*.tif'))
            if tif_files:
                print(f"\n--- Permanent Water Mask ({split}) ---")
                info = inspect_mask(tif_files[0])
                for k, v in info.items():
                    print(f"  {k}: {v}")
                break
    
    # Check a metadata JSON
    for split in ['test', 'val', 'train']:
        meta_dir = data_dir / split / 'meta'
        if meta_dir.exists():
            json_files = sorted(meta_dir.glob('*.json'))
            if json_files:
                import json
                print(f"\n--- Metadata JSON ({split}) ---")
                with open(json_files[0]) as f:
                    meta = json.load(f)
                print(f"  File: {json_files[0].name}")
                print(f"  Keys: {list(meta.keys())}")
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        print(f"  {k}: {v}")
                    elif isinstance(v, dict):
                        print(f"  {k}: {{...}} ({len(v)} keys)")
                    elif isinstance(v, list):
                        print(f"  {k}: [...] ({len(v)} items)")
                break
    
    print("\n" + "=" * 70)
    print("Inspection complete.")
    print("=" * 70)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else r"d:\superiorparab\final year DL project\data\raw\flood\worldfloods_v2"
    inspect_dataset(data_dir)
