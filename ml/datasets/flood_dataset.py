"""
WorldFloods Dataset for EfficientNet-B0 Classification
Converts the segmentation dataset into image-level flood/non-flood labels.

Label Derivation Rule:
    Given a ground truth mask and permanent water mask:
    1. Count pixels labeled as flood water (value=1 in gt) that are NOT permanent water
    2. Calculate flood_ratio = flood_pixels / total_valid_pixels
    3. If flood_ratio > threshold (default 0.05 = 5%), label = FLOOD (1)
    4. Otherwise, label = NON_FLOOD (0)

This approach is transparent, defensible, and documented.
"""
import os
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from PIL import Image


# Sentinel-2 band indices (0-indexed) in WorldFloods v2
# The S2 TIFFs contain the following bands:
# We'll inspect the actual bands and use RGB (B4, B3, B2) or equivalent
S2_RGB_BANDS = None  # Will be set after inspection


class WorldFloodsClassificationDataset(Dataset):
    """
    Converts WorldFloods segmentation data to binary classification.
    
    Each sample:
        - Reads the Sentinel-2 image (selected bands -> RGB)
        - Reads the ground truth mask
        - Reads the permanent water mask (optional)
        - Derives a binary flood/non-flood label
        - Resizes to target_size for EfficientNet-B0
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        target_size: int = 224,
        flood_threshold: float = 0.05,
        rgb_bands: Tuple[int, ...] = (3, 2, 1),  # Default: B4,B3,B2 (0-indexed)
        transform=None,
        use_permanent_water: bool = True,
    ):
        """
        Args:
            data_dir: Path to worldfloods_v2 root directory
            split: 'train', 'val', or 'test'
            target_size: Image resize target (224 for EfficientNet-B0)
            flood_threshold: Minimum flood pixel ratio to label as FLOOD
            rgb_bands: Tuple of 0-indexed band indices to extract as RGB
            transform: Optional torchvision transforms
            use_permanent_water: Whether to exclude permanent water from flood count
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_size = target_size
        self.flood_threshold = flood_threshold
        self.rgb_bands = rgb_bands
        self.transform = transform
        self.use_permanent_water = use_permanent_water
        
        # Discover image files
        self.s2_dir = self.data_dir / split / "S2"
        self.gt_dir = self.data_dir / split / "gt"
        self.pw_dir = self.data_dir / split / "PERMANENTWATERJRC"
        
        if not self.s2_dir.exists():
            raise FileNotFoundError(f"S2 directory not found: {self.s2_dir}")
        
        # Get paired files (S2 image must have matching gt mask)
        self.samples = self._discover_pairs()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid image-mask pairs found in {self.s2_dir}")
    
    def _discover_pairs(self) -> List[Dict[str, Path]]:
        """Find all valid S2 image + gt mask pairs."""
        pairs = []
        s2_files = sorted(self.s2_dir.glob("*.tif"))
        
        for s2_path in s2_files:
            gt_path = self.gt_dir / s2_path.name
            pw_path = self.pw_dir / s2_path.name
            
            if gt_path.exists():
                pair = {
                    "s2": s2_path,
                    "gt": gt_path,
                    "pw": pw_path if pw_path.exists() else None,
                    "name": s2_path.stem,
                }
                pairs.append(pair)
        
        return pairs
    
    def _read_rgb(self, s2_path: Path) -> np.ndarray:
        """Read and extract RGB bands from a Sentinel-2 TIFF.
        
        Data format (from inspection):
            - 15 bands, dtype=int16
            - Values: 0-14000+ (reflectance × 10000)
            - Bands 4,3,2 (1-indexed) = Red,Green,Blue for true color
        """
        with rasterio.open(s2_path) as src:
            # Read selected bands (1-indexed for rasterio)
            bands = []
            for band_idx in self.rgb_bands:
                band = src.read(band_idx + 1).astype(np.float32)  # rasterio is 1-indexed
                bands.append(band)
            
            rgb = np.stack(bands, axis=-1)  # HxWx3
            
            # Replace negative values (nodata) with 0
            rgb = np.clip(rgb, 0, None)
            
            # Normalize to [0, 1] range
            # Sentinel-2 reflectance values are typically 0-10000
            rgb = np.clip(rgb / 10000.0, 0.0, 1.0)
            
            return rgb.astype(np.float32)
    
    def _read_mask(self, mask_path: Path, band: int = 1) -> np.ndarray:
        """Read a mask TIFF file.
        
        Args:
            mask_path: Path to the mask TIFF
            band: Band index to read (1-indexed). GT masks have 2 bands;
                  band 1 = classification (land/flood/cloud).
        """
        with rasterio.open(mask_path) as src:
            mask = src.read(band)
            mask = np.nan_to_num(mask, nan=0)
            return mask
    
    def _derive_label(self, gt_mask: np.ndarray, pw_mask: Optional[np.ndarray]) -> Tuple[int, float]:
        """
        Derive binary flood label from segmentation mask.
        
        WorldFloods gt mask values (from data inspection):
            0 = land/dry
            1 = flood water  
            2 = cloud
        
        Permanent water mask values (from data inspection):
            0 = unknown/invalid
            1 = not water
            2 = seasonal water
            3 = permanent water
        
        Returns:
            (label, flood_ratio) where label is 0 (non-flood) or 1 (flood)
        """
        # Valid pixels: not cloud (2) and not nodata
        valid_mask = (gt_mask == 0) | (gt_mask == 1)
        total_valid = valid_mask.sum()
        
        if total_valid == 0:
            return 0, 0.0
        
        # Flood pixels
        flood_pixels = (gt_mask == 1)
        
        # Optionally exclude permanent water (value 3 = permanent water in JRC)
        if pw_mask is not None and self.use_permanent_water:
            permanent_water = (pw_mask == 3)
            flood_pixels = flood_pixels & ~permanent_water
        
        flood_count = flood_pixels.sum()
        flood_ratio = flood_count / total_valid
        
        label = 1 if flood_ratio > self.flood_threshold else 0
        return label, float(flood_ratio)
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target_size using PIL."""
        # image is HxWx3 float32 [0,1]
        img_uint8 = (image * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_img = pil_img.resize((self.target_size, self.target_size), Image.BILINEAR)
        return np.array(pil_img).astype(np.float32) / 255.0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Read RGB image
            rgb = self._read_rgb(sample["s2"])
            
            # Read masks
            gt_mask = self._read_mask(sample["gt"])
            pw_mask = self._read_mask(sample["pw"]) if sample["pw"] is not None else None
            
            # Derive label
            label, flood_ratio = self._derive_label(gt_mask, pw_mask)
            
            # Resize to 224x224
            rgb = self._resize_image(rgb)
            
            # Convert to tensor: HxWx3 -> 3xHxW
            image_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
            
            # Apply transforms if any
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return image_tensor, label_tensor
        
        except Exception as e:
            # Handle corrupted files — return a different valid sample
            print(f"WARNING: Error reading {sample['name']}: {e}")
            # Try the next valid sample (wrap around)
            alt_idx = (idx + 1) % len(self.samples)
            if alt_idx == idx:
                # Only one sample, create a dummy
                dummy_image = torch.zeros(3, self.target_size, self.target_size)
                return dummy_image, torch.tensor(0, dtype=torch.long)
            return self.__getitem__(alt_idx)
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Calculate label distribution across the dataset."""
        flood_count = 0
        non_flood_count = 0
        
        for sample in self.samples:
            gt_mask = self._read_mask(sample["gt"])
            pw_mask = self._read_mask(sample["pw"]) if sample["pw"] is not None else None
            label, _ = self._derive_label(gt_mask, pw_mask)
            
            if label == 1:
                flood_count += 1
            else:
                non_flood_count += 1
        
        return {"flood": flood_count, "non_flood": non_flood_count, "total": len(self.samples)}


def verify_dataset(data_dir: str, split: str = "test"):
    """Quick verification that the dataset loads correctly."""
    print(f"\nVerifying {split} split...")
    
    try:
        dataset = WorldFloodsClassificationDataset(data_dir, split=split)
        print(f"  Found {len(dataset)} samples")
        
        # Load first sample
        image, label = dataset[0]
        print(f"  First sample: image shape={image.shape}, label={label.item()}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Name: {dataset.samples[0]['name']}")
        
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


if __name__ == "__main__":
    data_dir = r"d:\superiorparab\final year DL project\data\raw\flood\worldfloods_v2"
    
    for split in ["test", "val", "train"]:
        verify_dataset(data_dir, split)
