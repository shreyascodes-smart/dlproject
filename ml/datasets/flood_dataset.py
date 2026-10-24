"""
WorldFloods Dataset for EfficientNet-B0 Classification
Converts the segmentation dataset into image-level flood/non-flood labels.

Ground Truth Mask Band Semantics (verified from dataset metadata JSON):
    Band 1 (cloud/validity mask):
        0 = invalid (no data)
        1 = clear (valid observation)
        2 = cloud (obscured)

    Band 2 (water/land mask):
        0 = invalid (no data)
        1 = land
        2 = water

    Source: encoding_values field in every metadata JSON, consistent with
    the ml4floods library (https://github.com/spaceml-org/ml4floods) and
    the WorldFloods tutorial which reads band index 1 (0-indexed = rasterio
    Band 2) for the water mask and band index 0 (0-indexed = rasterio Band 1)
    as the cloud filter.

Permanent Water Mask Semantics (JRC Global Surface Water):
    0 = no data / outside coverage
    1 = not water
    2 = seasonal water
    3 = permanent water

Label Derivation Rule:
    1. Read Band 1 (cloud mask) and Band 2 (water mask) from the GT TIFF.
    2. Valid observation pixels = (Band 1 == 1) AND (Band 2 != 0)
       i.e. clear sky AND not-invalid in the water layer.
    3. Water pixels = (Band 2 == 2) AND (Band 1 == 1)
       i.e. water that is also under clear sky — cloudy or invalid pixels
       cannot contribute as water OR as land.
    4. If use_permanent_water is True, subtract permanent water pixels
       (JRC mask value == 3) from the water count.
    5. flood_ratio = water_pixels / valid_pixels
    6. If flood_ratio > threshold (default 0.05 = 5%), label = FLOOD (1),
       otherwise label = NON_FLOOD (0).

Limitation:
    The WorldFloods v2 water mask (Band 2) records ALL detected water — it
    does not distinguish flood water from pre-existing water bodies (rivers,
    lakes, reservoirs). The permanent-water exclusion via the JRC mask
    partially addresses this by removing pixels classified as "permanent
    water" (JRC value 3), but seasonal water bodies and small pre-existing
    water features may still be counted as flood candidates. This means the
    derived image-level "flood" label really means "the image contains
    significant non-permanent surface water", which is a useful but imperfect
    proxy for actual flooding. True flood detection would require pre-event
    vs. post-event comparison, which is beyond the scope of this classifier.

This approach is transparent, defensible, and documented.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
from PIL import Image

logger = logging.getLogger(__name__)

# Sentinel label value for samples with no valid observation pixels.
# These must NOT be treated as NON_FLOOD; they are genuinely unknown.
LABEL_UNKNOWN = -1


class SampleLoadError(Exception):
    """Raised when a dataset sample cannot be loaded.

    Attributes:
        sample_name: Identifier of the failed sample.
        reason: Human-readable failure description.
    """

    def __init__(self, sample_name: str, reason: str):
        self.sample_name = sample_name
        self.reason = reason
        super().__init__(f"Failed to load sample '{sample_name}': {reason}")


class WorldFloodsClassificationDataset(Dataset):
    """
    Converts WorldFloods segmentation data to binary classification.

    Each sample:
        - Reads the Sentinel-2 image (selected bands -> RGB)
        - Reads the ground truth mask (Band 1 = cloud, Band 2 = water)
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

    @staticmethod
    def _center_window(height: int, width: int, target_px: int = 512) -> Window:
        """Calculate a center window given raster dimensions.

        Returns a Window of at most target_px x target_px pixels centred
        on the raster.  This is a pure function of the dimensions so the
        same window can be reused for any spatially-aligned raster of the
        same size.
        """
        crop_h = min(height, target_px)
        crop_w = min(width, target_px)
        row_off = max(0, (height - crop_h) // 2)
        col_off = max(0, (width - crop_w) // 2)
        return Window(col_off, row_off, crop_w, crop_h)

    def _compute_shared_window(
        self,
        s2_path: Path,
        gt_path: Path,
        pw_path: Optional[Path],
        sample_name: str,
    ) -> Window:
        """Compute a single center-crop window from S2 and verify alignment.

        Opens the S2 raster to get dimensions/CRS/transform, then verifies
        that the GT (and optionally PW) raster has identical dimensions,
        CRS, and affine transform.  Returns the shared window.

        Raises:
            SampleLoadError: If dimensions, CRS, or transforms do not match.
        """
        with rasterio.open(s2_path) as s2_src:
            s2_h, s2_w = s2_src.height, s2_src.width
            s2_crs = s2_src.crs
            s2_tf = s2_src.transform

        with rasterio.open(gt_path) as gt_src:
            if gt_src.height != s2_h or gt_src.width != s2_w:
                raise SampleLoadError(
                    sample_name,
                    f"GT dimensions ({gt_src.height}x{gt_src.width}) "
                    f"differ from S2 ({s2_h}x{s2_w}).",
                )
            if gt_src.crs != s2_crs:
                raise SampleLoadError(
                    sample_name,
                    f"GT CRS ({gt_src.crs}) differs from S2 ({s2_crs}).",
                )
            if list(gt_src.transform)[:6] != list(s2_tf)[:6]:
                raise SampleLoadError(
                    sample_name,
                    f"GT transform differs from S2.",
                )

        if pw_path is not None:
            with rasterio.open(pw_path) as pw_src:
                if pw_src.height != s2_h or pw_src.width != s2_w:
                    raise SampleLoadError(
                        sample_name,
                        f"PW dimensions ({pw_src.height}x{pw_src.width}) "
                        f"differ from S2 ({s2_h}x{s2_w}).",
                    )
                if pw_src.crs != s2_crs:
                    raise SampleLoadError(
                        sample_name,
                        f"PW CRS ({pw_src.crs}) differs from S2 ({s2_crs}).",
                    )
                if list(pw_src.transform)[:6] != list(s2_tf)[:6]:
                    raise SampleLoadError(
                        sample_name,
                        f"PW transform differs from S2.",
                    )

        return self._center_window(s2_h, s2_w)

    def _read_rgb(self, s2_path: Path, window: Window) -> np.ndarray:
        """Read and extract RGB bands from a Sentinel-2 TIFF.

        Uses the pre-computed *window* (derived from S2 dimensions) to
        read a consistent spatial subset.
        """
        with rasterio.open(s2_path) as src:
            bands = []
            for band_idx in self.rgb_bands:
                band = src.read(band_idx + 1, window=window).astype(np.float32)
                bands.append(band)

            rgb = np.stack(bands, axis=-1)  # HxWx3

            # Replace negative values (nodata) with 0
            rgb = np.clip(rgb, 0, None)

            # Normalize to [0, 1] range
            rgb = np.clip(rgb / 10000.0, 0.0, 1.0)

            return rgb.astype(np.float32)

    def _read_gt_masks(self, gt_path: Path, window: Window) -> Tuple[np.ndarray, np.ndarray]:
        """Read both bands of the GT mask TIFF using the shared *window*.

        Returns:
            cloud_mask: Band 1 — cloud/validity mask
                        (0=invalid, 1=clear, 2=cloud)
            water_mask: Band 2 — water/land mask
                        (0=invalid, 1=land, 2=water)

        Raises:
            ValueError: If the GT TIFF does not contain at least 2 bands.
        """
        with rasterio.open(gt_path) as src:
            if src.count < 2:
                raise ValueError(
                    f"GT mask '{gt_path.name}' has {src.count} band(s), "
                    f"expected 2 (Band 1=cloud, Band 2=water)."
                )
            cloud_mask = src.read(1, window=window)  # Band 1: cloud/validity
            water_mask = src.read(2, window=window)  # Band 2: water/land
            cloud_mask = np.nan_to_num(cloud_mask, nan=0).astype(np.int16)
            water_mask = np.nan_to_num(water_mask, nan=0).astype(np.int16)
            return cloud_mask, water_mask

    def _read_pw_mask(self, pw_path: Path, window: Window) -> np.ndarray:
        """Read the permanent water (JRC) mask TIFF using the shared *window*.

        JRC Global Surface Water mask values:
            0 = no data / outside coverage
            1 = not water
            2 = seasonal water
            3 = permanent water

        Returns:
            pw_mask: 2-D array of JRC mask values.
        """
        with rasterio.open(pw_path) as src:
            mask = src.read(1, window=window)
            mask = np.nan_to_num(mask, nan=0).astype(np.int16)
            return mask

    def _derive_label(
        self,
        cloud_mask: np.ndarray,
        water_mask: np.ndarray,
        pw_mask: Optional[np.ndarray],
    ) -> Tuple[int, float]:
        """
        Derive binary flood label from ground-truth masks.

        GT Band 1 (cloud_mask):
            0 = invalid, 1 = clear, 2 = cloud

        GT Band 2 (water_mask):
            0 = invalid, 1 = land, 2 = water

        Permanent water (JRC) mask:
            0 = no data, 1 = not water, 2 = seasonal, 3 = permanent

        Logic:
            valid_pixel  = clear sky (cloud_mask == 1) AND
                           not-invalid water layer (water_mask != 0)
            water_pixel  = valid_pixel AND (water_mask == 2)
            If use_permanent_water: subtract permanent water (JRC == 3)
            flood_ratio  = water_pixels / valid_pixels
            label        = FLOOD (1) if flood_ratio > threshold, else NON_FLOOD (0)

        Invalid and cloudy pixels are excluded from BOTH numerator and
        denominator so they cannot inflate or deflate the flood ratio.

        Returns:
            (label, flood_ratio)
        """
        # A pixel is a valid observation if the sky is clear (Band 1 == 1)
        # AND the water-mask layer has a definite classification (not 0 = invalid).
        clear_sky = (cloud_mask == 1)
        not_invalid_water = (water_mask != 0)
        valid_mask = clear_sky & not_invalid_water
        total_valid = int(valid_mask.sum())

        if total_valid == 0:
            # No valid observations — label is genuinely unknown.
            # Returning LABEL_UNKNOWN (-1) so callers can distinguish
            # "no data" from a real NON_FLOOD classification.
            return LABEL_UNKNOWN, 0.0

        # Water pixels: water_mask == 2, but only where observation is valid.
        water_pixels = (water_mask == 2) & clear_sky

        # Optionally exclude permanent water (JRC value 3).
        if pw_mask is not None and self.use_permanent_water:
            permanent_water = (pw_mask == 3)
            water_pixels = water_pixels & ~permanent_water

        water_count = int(water_pixels.sum())
        flood_ratio = water_count / total_valid

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
        """Load a single sample.

        Raises:
            SampleLoadError: If reading the sample files fails for any reason.
                The error includes the sample name and the underlying cause
                so the caller (DataLoader, evaluation script, etc.) can
                report the failure honestly instead of silently substituting
                a different sample.
        """
        sample = self.samples[idx]

        try:
            # Compute a shared spatial window from S2, with alignment checks.
            window = self._compute_shared_window(
                sample["s2"], sample["gt"], sample["pw"], sample["name"]
            )

            # Read RGB image using the shared window
            rgb = self._read_rgb(sample["s2"], window)

            # Read GT mask — both bands — using the same window
            cloud_mask, water_mask = self._read_gt_masks(sample["gt"], window)

            # Read permanent water mask if available, same window
            pw_mask = (
                self._read_pw_mask(sample["pw"], window)
                if sample["pw"] is not None
                else None
            )

            # Derive label
            label, flood_ratio = self._derive_label(cloud_mask, water_mask, pw_mask)

            if label == LABEL_UNKNOWN:
                raise SampleLoadError(
                    sample["name"],
                    "No valid observation pixels in the center window — "
                    "label is unknown. This sample must be excluded from "
                    "supervised training and evaluation.",
                )

            # Resize to 224x224
            rgb = self._resize_image(rgb)

            # Convert to tensor: HxWx3 -> 3xHxW
            image_tensor = torch.from_numpy(rgb).permute(2, 0, 1)

            # Apply transforms if any
            if self.transform:
                image_tensor = self.transform(image_tensor)

            label_tensor = torch.tensor(label, dtype=torch.long)

            return image_tensor, label_tensor

        except SampleLoadError:
            raise  # Already wrapped — re-raise as-is.
        except Exception as e:
            # Wrap with sample identity so callers can report which file failed.
            raise SampleLoadError(sample["name"], str(e)) from e

    def get_label_distribution(self) -> Dict[str, int]:
        """Calculate label distribution across the dataset.

        Returns a dict with counts and any failures encountered.
        """
        flood_count = 0
        non_flood_count = 0
        failures = []

        unknown_count = 0

        for sample in self.samples:
            try:
                window = self._compute_shared_window(
                    sample["s2"], sample["gt"], sample["pw"], sample["name"]
                )
                cloud_mask, water_mask = self._read_gt_masks(sample["gt"], window)
                pw_mask = (
                    self._read_pw_mask(sample["pw"], window)
                    if sample["pw"] is not None
                    else None
                )
                label, _ = self._derive_label(cloud_mask, water_mask, pw_mask)

                if label == LABEL_UNKNOWN:
                    unknown_count += 1
                elif label == 1:
                    flood_count += 1
                else:
                    non_flood_count += 1
            except Exception as e:
                failures.append({"name": sample["name"], "error": str(e)})

        return {
            "flood": flood_count,
            "non_flood": non_flood_count,
            "unknown": unknown_count,
            "total": len(self.samples),
            "failures": failures,
        }


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
