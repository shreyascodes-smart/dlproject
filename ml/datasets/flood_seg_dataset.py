"""
Flood Segmentation Dataset Loader for WorldFloods v2.

Reads Sentinel-2 RGB imagery and pre-rasterized segmentation masks from
data/processed/flood/segmentation_v1/.

Stored mask encoding:
    0 = ignore (cloud/invalid/outside AOI)
    1 = supervised non-flood
    2 = verified flood extent

Training target mapping (for 2-class CrossEntropyLoss):
    stored 1 -> target 0 (non-flood)
    stored 2 -> target 1 (flood)
    stored 0 -> target IGNORE_INDEX (255, excluded from loss)

S2 bands: rasterio bands 4, 3, 2 = Sentinel-2 B4(Red), B3(Green), B2(Blue)
Normalization: DN / 10000 -> [0,1] -> ImageNet normalize
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

# Label mapping constants
STORED_IGNORE = 0
STORED_NON_FLOOD = 1
STORED_FLOOD = 2

TARGET_NON_FLOOD = 0  # Class index 0 for CE loss
TARGET_FLOOD = 1      # Class index 1 for CE loss
IGNORE_INDEX = 255    # Excluded from loss via ignore_index=255

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# S2 band indices (0-indexed): band 3=Red(B4), band 2=Green(B3), band 1=Blue(B2)
S2_RGB_BANDS_0IDX = (3, 2, 1)


def stored_to_target(mask: np.ndarray) -> np.ndarray:
    """Convert stored mask values to training target indices.

    stored 0 (ignore)    -> 255 (IGNORE_INDEX)
    stored 1 (non-flood) -> 0   (TARGET_NON_FLOOD)
    stored 2 (flood)     -> 1   (TARGET_FLOOD)
    """
    target = np.full_like(mask, IGNORE_INDEX, dtype=np.uint8)
    target[mask == STORED_NON_FLOOD] = TARGET_NON_FLOOD
    target[mask == STORED_FLOOD] = TARGET_FLOOD
    return target


def get_train_augmentation(input_size: int = 256) -> "A.Compose":
    """Training augmentations with identical transforms for image + mask."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for augmentation")
    return A.Compose([
        A.RandomCrop(height=input_size, width=input_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


def get_val_transform(input_size: int = 256) -> "A.Compose":
    """Deterministic validation transform — resize to fixed size.

    Uses Resize instead of CenterCrop because rasters are large
    (mean ~3000x3400) and flood pixels are rarely centered.
    CenterCrop(256) was found to retain 0.00% of val flood pixels.
    Resize preserves all flood extent at the cost of spatial resolution,
    which is acceptable for computing validation metrics.

    Mask interpolation uses NEAREST to preserve categorical labels.
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required")
    import cv2
    return A.Compose([
        A.Resize(
            height=input_size, width=input_size,
            interpolation=cv2.INTER_LINEAR,   # bilinear for image
        ),
    ])


def normalize_s2_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize S2 DN values to ImageNet range.

    1. Divide by 10000 (S2 L2A reflectance scaling).
    2. Clip to [0, 1].
    3. Apply ImageNet channel normalization.

    Args:
        image: HxWx3 float32 array with raw DN values.

    Returns:
        HxWx3 float32 array normalized for pretrained encoder.
    """
    img = np.clip(image / 10000.0, 0.0, 1.0).astype(np.float32)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    return img


class FloodSegDataset(Dataset):
    """PyTorch Dataset for flood segmentation.

    Loads S2 imagery, segmentation masks, and validity masks from the
    prepared segmentation_v1 directory. Applies the stored-to-target
    label mapping and optional augmentations.

    Output shapes:
        image: (3, H, W) float32 tensor (ImageNet-normalized RGB)
        target: (H, W) long tensor (0=non-flood, 1=flood, 255=ignore)
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        data_root: str,
        masks_dir: str,
        input_size: int = 256,
        augment: bool = False,
    ):
        """
        Args:
            manifest_path: Path to seg_split_manifest.json
            split: 'train', 'val', or 'test'
            data_root: Path to worldfloods_v2 root
            masks_dir: Path to segmentation_v1/masks/
            input_size: Crop/resize target (pixels)
            augment: Apply random augmentations (train only)
        """
        self.data_root = Path(data_root)
        self.masks_dir = Path(masks_dir)
        self.input_size = input_size
        self.augment = augment

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        if split not in manifest["splits"]:
            raise ValueError(
                f"Split '{split}' not in manifest. "
                f"Available: {list(manifest['splits'].keys())}"
            )

        split_data = manifest["splits"][split]
        self.samples = split_data["samples"]
        self.split = split

        # Validate all files exist
        self._validate_files()

        # Set up augmentation
        if augment:
            self.transform = get_train_augmentation(input_size)
        else:
            self.transform = get_val_transform(input_size)

    def _validate_files(self):
        """Check that all required files exist for each sample."""
        missing = []
        for s in self.samples:
            name = s["name"]
            orig_split = s["original_split"]

            s2_path = self.data_root / orig_split / "S2" / (name + ".tif")
            seg_path = self.masks_dir / (name + "_seg.tif")
            val_path = self.masks_dir / (name + "_validity.tif")

            for p, kind in [(s2_path, "S2"), (seg_path, "seg"), (val_path, "validity")]:
                if not p.exists():
                    missing.append(f"{name}: missing {kind} at {p}")

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files missing:\n" + "\n".join(missing[:10])
            )

    def _read_s2_rgb(self, s2_path: Path) -> np.ndarray:
        """Read RGB bands from S2 TIFF. Returns HxWx3 float32."""
        with rasterio.open(s2_path) as src:
            bands = []
            for band_0idx in S2_RGB_BANDS_0IDX:
                band = src.read(band_0idx + 1).astype(np.float32)
                bands.append(band)
            rgb = np.stack(bands, axis=-1)  # HxWx3
            rgb = np.clip(rgb, 0, None)  # Replace negatives
            return rgb

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        """Read a single-band mask TIFF. Returns HxW uint8."""
        with rasterio.open(mask_path) as src:
            return src.read(1).astype(np.uint8)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        name = sample["name"]
        orig_split = sample["original_split"]

        s2_path = self.data_root / orig_split / "S2" / (name + ".tif")
        seg_path = self.masks_dir / (name + "_seg.tif")
        val_path = self.masks_dir / (name + "_validity.tif")

        # Read data
        image = self._read_s2_rgb(s2_path)         # HxWx3 float32
        seg_mask = self._read_mask(seg_path)        # HxW uint8 {0,1,2}
        val_mask = self._read_mask(val_path)        # HxW uint8 {0,1}

        # Verify dimensions match
        if image.shape[:2] != seg_mask.shape or image.shape[:2] != val_mask.shape:
            raise ValueError(
                f"Dimension mismatch for {name}: "
                f"S2={image.shape[:2]}, seg={seg_mask.shape}, val={val_mask.shape}"
            )

        # Combine validity into seg mask: invalid pixels -> stored 0
        seg_mask[val_mask == 0] = STORED_IGNORE

        # Apply augmentation (identical transform for image and mask)
        h, w = image.shape[:2]
        if min(h, w) < self.input_size:
            # Pad if smaller than input_size
            pad_h = max(0, self.input_size - h)
            pad_w = max(0, self.input_size - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)),
                           mode="constant", constant_values=0)
            seg_mask = np.pad(seg_mask, ((0, pad_h), (0, pad_w)),
                              mode="constant", constant_values=STORED_IGNORE)

        transformed = self.transform(image=image, mask=seg_mask)
        image = transformed["image"]
        seg_mask = transformed["mask"]

        # Normalize image
        image = normalize_s2_rgb(image)

        # Convert stored labels to target labels
        target = stored_to_target(seg_mask)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # 3xHxW
        target_tensor = torch.from_numpy(target).long()           # HxW

        return image_tensor, target_tensor

    def get_sample_name(self, idx: int) -> str:
        """Get sample name by index."""
        return self.samples[idx]["name"]


def compute_class_weights(manifest_path: str) -> Tuple[float, float]:
    """Compute inverse-frequency class weights from training split.

    Uses the actual pixel counts from the manifest (already verified
    against real mask files in Phase E).

    Convention: w_c = N_total / (2 * N_c)

    Returns:
        (w_non_flood, w_flood) tuple for CrossEntropyLoss weights.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    train = manifest["splits"]["train"]
    flood_px = train["total_flood_px"]
    supervised_px = train["total_supervised_px"]
    non_flood_px = supervised_px - flood_px

    w_nf = supervised_px / (2 * non_flood_px)
    w_flood = supervised_px / (2 * flood_px)

    return w_nf, w_flood
