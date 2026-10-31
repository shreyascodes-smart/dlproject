"""
Flood Segmentation Inference (U-Net + EfficientNet-B2).

Loads the Phase H checkpoint and turns a Sentinel-2 scene into a per-pixel
flood mask. Preprocessing reuses the exact functions used in training and
evaluation (ml/datasets/flood_seg_dataset.py), so the model sees inputs the
same way it did during training.

Preprocessing (verified against FloodSegDataset):
    1. Read Sentinel-2 bands B4, B3, B2 (Red, Green, Blue).
       - WorldFloods 15-band rasters: rasterio bands 4, 3, 2 (0-idx 3, 2, 1).
       - Live CDSE 6-band rasters [B02, B03, B04, B08, B11, B12]: 0-idx 2, 1, 0.
    2. Replace negative DN values with 0.
    3. If the scene is smaller than 256 px on either side, zero-pad it
       (bottom/right), exactly like training.
    4. Divide by 10000, clip to [0, 1], apply ImageNet mean/std.

Two inference methods:
    "resize": shrink the whole scene to 256x256 (the procedure used for the
              Phase H validation/test numbers), then upsample the predicted
              probabilities back to the original grid.
    "tiled":  slide a 256x256 window over the scene at native 10 m
              resolution (the scale the model was trained on), blend
              overlapping tiles, and stitch them back together.

Neither method is assumed to be better; see the validation comparison report.

Flood mask: a pixel is "flood" when softmax P(flood) > 0.5, which is the same
as argmax over the two classes used by FloodMetrics during evaluation.
P(flood) is the model's raw softmax output. It is NOT a calibrated confidence.

Flood-pixel coverage:
    coverage % = predicted flood pixels / eligible pixels * 100
    eligible   = pixel has image data (not all RGB bands == 0)
                 AND, if a validity mask is supplied, validity == 1
    The validity mask (cloud-free, valid ground-truth area) only exists for
    WorldFloods dataset samples. For live imagery only the no-data check is
    available, so clouds and cloud shadows can be counted and misclassified.
    Coverage is a pixel ratio, not an area in km^2.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# albumentations otherwise makes a network call to check for updates on import
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from ml.datasets.flood_seg_dataset import (  # noqa: E402
    S2_RGB_BANDS_0IDX,
    STORED_FLOOD,
    STORED_IGNORE,
    STORED_NON_FLOOD,
    get_val_transform,
    normalize_s2_rgb,
)

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "flood" / "segmentation_v1" / "best_seg_model.pt"

# Band order of the 6-band GeoTIFF produced by backend/app/services/sentinel_service.py
LIVE_RGB_BANDS_0IDX = (2, 1, 0)

METHODS = ("resize", "tiled")
FLOOD_THRESHOLD = 0.5  # equivalent to argmax for 2 classes; not tuned

# Error-map codes used by compare_with_reference()
ERR_IGNORE = 0  # reference has no supervision here
ERR_TN = 1
ERR_TP = 2
ERR_FP = 3
ERR_FN = 4

# Display colours (RGB)
FLOOD_COLOR = (0, 170, 255)
# Distinct brown so ignored pixels are never confused with dimmed bright land (snow, cloud)
IGNORE_COLOR = (140, 90, 50)
ERROR_COLORS = {
    ERR_TP: (0, 200, 255),   # correct flood: cyan
    ERR_FP: (255, 60, 160),  # false alarm: magenta
    ERR_FN: (255, 210, 0),   # missed flood: yellow
}


# Sentinel-2 band name -> 1-indexed band in WorldFloods v2 S2 GeoTIFFs (13 L1C bands, then 2 extra layers)
WF_BAND_1IDX = {"B01": 1, "B02": 2, "B03": 3, "B04": 4, "B05": 5, "B06": 6, "B07": 7, "B08": 8,
                "B8A": 9, "B09": 10, "B10": 11, "B11": 12, "B12": 13}
# Band order of the live 6-band request (backend/app/services/sentinel_service.py)
LIVE_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


@dataclass(frozen=True)
class ModelSpec:
    """What a checkpoint expects as input and what its positive class means.

    target: "ems_flood"     (v1: Copernicus EMS mapped flood vs everything else)
            "surface_water" (v2: any surface water; flood = water outside known permanent water)
    """

    version: str
    bands: Tuple[str, ...]
    target: str
    normalization: Dict[str, Any]
    input_size: int = 256
    # Optional post-processing (models/<version>/postprocessing.json):
    #   water_gate {"band": "B11", "max_toa_reflectance": 0.10}: a positive pixel counts as water only if
    #       its SWIR reflectance is below the limit (open water is dark in SWIR; haze barely affects it).
    #   haze_check {"band": "B02", "percentile": 5, "max_toa_reflectance": t}: scene haze index = that
    #       percentile of blue TOA reflectance over eligible pixels; above t (the training maximum) the
    #       scene is outside the training range ("hazy"). When a haze_check exists, the water gate is
    #       applied ONLY to hazy scenes.
    water_gate: Optional[Dict[str, Any]] = None
    haze_check: Optional[Dict[str, Any]] = None

    @property
    def rgb_index(self) -> Tuple[int, int, int]:
        return tuple(self.bands.index(b) for b in ("B04", "B03", "B02"))

    @property
    def preprocessing_id(self) -> str:
        n = self.normalization.get("kind")
        return f"{self.version}:{'-'.join(self.bands)}:{n}"

    @property
    def postprocessing_id(self) -> str:
        g, h = self.water_gate, self.haze_check
        if not g:
            return "none"
        gate = f"{g['band']}<{float(g['max_toa_reflectance']):.3f}"
        if not h:
            return gate
        return f"{gate} if {h['band']}p{int(h['percentile'])}>{float(h['max_toa_reflectance']):.4f}"

    def _scale(self) -> float:
        return float(self.normalization.get("scale", 10000.0))

    def haze_index(self, raw: np.ndarray, eligible: np.ndarray) -> Optional[float]:
        """Scene haze index (TOA reflectance), or None without a haze check or eligible pixels."""
        h = self.haze_check
        if not h or not eligible.any():
            return None
        v = raw[..., self.bands.index(h["band"])][eligible]
        return float(np.percentile(v, float(h["percentile"]))) / self._scale()

    def is_hazy(self, index: Optional[float]) -> bool:
        return index is not None and index > float(self.haze_check["max_toa_reflectance"])

    def gate_mask(self, raw: np.ndarray) -> Optional[np.ndarray]:
        """H x W bool: pixels allowed to be water by the SWIR gate (None = no gate)."""
        g = self.water_gate
        if not g:
            return None
        i = self.bands.index(g["band"])
        return raw[..., i] < float(g["max_toa_reflectance"]) * self._scale()

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """H x W x C raw DN -> normalised float32 (exactly as in training)."""
        if self.normalization.get("kind") == "imagenet_rgb":
            return normalize_s2_rgb(raw)
        scale = float(self.normalization.get("scale", 10000.0))
        lo, hi = self.normalization.get("clip", [0, 1])
        mean = np.asarray(self.normalization["mean"], np.float32).reshape(1, 1, -1)
        std = np.asarray(self.normalization["std"], np.float32).reshape(1, 1, -1)
        x = np.clip(raw.astype(np.float32) / scale, lo, hi)
        return ((x - mean) / std).astype(np.float32)


V1_SPEC = ModelSpec("segmentation_v1", ("B04", "B03", "B02"), "ems_flood", {"kind": "imagenet_rgb"}, 256)


def spec_from_config(config: Dict[str, Any]) -> ModelSpec:
    """Model spec from a checkpoint config. Checkpoints without a band list are the v1 RGB model."""
    if not config.get("bands"):
        return ModelSpec(config.get("version", "segmentation_v1"), V1_SPEC.bands, "ems_flood",
                         {"kind": "imagenet_rgb"}, int(config.get("input_size", 256)))
    norm = config.get("normalization") or {}
    if norm.get("kind") != "standardize" or len(norm.get("mean", [])) != len(config["bands"]):
        raise ModelLoadError("Checkpoint normalisation does not match its band list.")
    unknown = [b for b in config["bands"] if b not in WF_BAND_1IDX]
    if unknown:
        raise ModelLoadError(f"Checkpoint uses unknown bands {unknown}.")
    return ModelSpec(config.get("version", "segmentation_vX"), tuple(config["bands"]),
                     config.get("target", "surface_water"), norm, int(config.get("input_size", 256)))


def read_scene(path: Path | str, spec: ModelSpec) -> np.ndarray:
    """Read the bands a model needs from a WorldFloods-layout GeoTIFF. Returns H x W x C float32 (>= 0)."""
    import rasterio

    path = Path(path)
    if not path.is_file():
        raise InvalidInputError(f"Image not found: {path}")
    idx = [WF_BAND_1IDX[b] for b in spec.bands]
    with rasterio.open(path) as src:
        if max(idx) > src.count:
            raise InvalidInputError(f"{path.name} has {src.count} bands; the model needs {spec.bands}.")
        arr = src.read(idx).astype(np.float32)
    return np.clip(np.moveaxis(arr, 0, -1), 0, None)


def select_live_bands(image_chw: np.ndarray, spec: ModelSpec) -> np.ndarray:
    """[6, H, W] live array in LIVE_BANDS order -> H x W x C in the model's band order."""
    if image_chw.ndim != 3 or image_chw.shape[0] != len(LIVE_BANDS):
        raise InvalidInputError(f"Live image must have {len(LIVE_BANDS)} bands, got shape {image_chw.shape}.")
    missing = [b for b in spec.bands if b not in LIVE_BANDS]
    if missing:
        raise InvalidInputError(f"Live imagery does not provide bands {missing}.")
    return np.stack([image_chw[LIVE_BANDS.index(b)] for b in spec.bands], axis=-1).astype(np.float32)


class ModelLoadError(RuntimeError):
    """Raised when the segmentation checkpoint cannot be loaded."""


class InvalidInputError(ValueError):
    """Raised when an input image or mask is unusable."""


# ============================================================
# Result container
# ============================================================
@dataclass
class SegmentationResult:
    """Everything the backend/UI needs, all on the same H x W pixel grid."""

    rgb_display: np.ndarray        # H x W x 3 uint8, contrast-stretched for viewing only
    flood_probability: np.ndarray  # H x W float32, softmax P(flood), uncalibrated
    flood_mask: np.ndarray         # H x W bool, predicted flood AND eligible
    eligible_mask: np.ndarray      # H x W bool, pixels counted in coverage
    stats: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.flood_mask.shape


# ============================================================
# Device and model loading
# ============================================================
def select_device(preferred: Optional[str] = None) -> Tuple[torch.device, Optional[str]]:
    """Pick a torch device. Returns (device, warning_or_None).

    preferred: None/"auto" -> CUDA if available else CPU; "cpu"; "cuda".
    Requesting CUDA when it is unavailable falls back to CPU with a warning.
    """
    pref = (preferred or "auto").lower()
    if pref not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unknown device '{preferred}'. Use 'auto', 'cpu' or 'cuda'.")
    if pref == "cpu":
        return torch.device("cpu"), None
    if torch.cuda.is_available():
        return torch.device("cuda"), None
    if pref == "cuda":
        return torch.device("cpu"), "CUDA was requested but is not available; using CPU."
    return torch.device("cpu"), None


def _torch_load(path: Path) -> Dict[str, Any]:
    """Load a checkpoint, preferring the safe weights_only loader."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as safe_err:  # older pickles may need the full loader
        logger.warning("weights_only load failed (%s); retrying full load of local checkpoint", safe_err)
        return torch.load(path, map_location="cpu", weights_only=False)


def load_postprocessing(model_dir: Path, spec: ModelSpec) -> Dict[str, Optional[Dict[str, Any]]]:
    """water_gate / haze_check from <model_dir>/postprocessing.json (surface-water models only)."""
    none = {"water_gate": None, "haze_check": None}
    p = Path(model_dir) / "postprocessing.json"
    if not p.is_file() or spec.target != "surface_water":
        return none
    try:
        cfg = json.loads(p.read_text())
    except Exception as e:
        raise ModelLoadError(f"Unreadable post-processing config {p.name}: {e}") from e
    g, h = cfg.get("water_gate"), cfg.get("haze_check")
    out = dict(none)
    if g:
        if g.get("band") not in spec.bands or not 0 < float(g.get("max_toa_reflectance", 0)) < 1:
            raise ModelLoadError(f"Invalid water gate in {p.name}: {g}")
        out["water_gate"] = {"band": g["band"], "max_toa_reflectance": float(g["max_toa_reflectance"])}
    if h:
        if (h.get("band") not in spec.bands or not 0 < float(h.get("max_toa_reflectance", 0)) < 1
                or not 0 <= float(h.get("percentile", -1)) <= 100):
            raise ModelLoadError(f"Invalid haze check in {p.name}: {h}")
        out["haze_check"] = {"band": h["band"], "percentile": float(h["percentile"]),
                             "max_toa_reflectance": float(h["max_toa_reflectance"])}
    return out


def load_water_gate(model_dir: Path, spec: ModelSpec) -> Optional[Dict[str, Any]]:
    """Backward-compatible helper: only the water gate."""
    return load_postprocessing(model_dir, spec)["water_gate"]


def load_segmentation_model(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild the U-Net and load the trained weights (read-only).

    encoder_weights=None is used because every weight, including the encoder,
    comes from the checkpoint; no ImageNet download or cache is needed.

    Returns:
        (model in eval mode on `device`, checkpoint metadata dict)
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ModelLoadError(f"Segmentation checkpoint not found: {path}")

    try:
        ckpt = _torch_load(path)
    except Exception as e:
        raise ModelLoadError(f"Could not read checkpoint {path}: {e}") from e

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ModelLoadError(f"Checkpoint {path} has no 'model_state_dict'.")

    config = ckpt.get("config") or {}
    encoder = config.get("encoder_name", "efficientnet-b2")
    num_classes = int(config.get("num_classes", 2))
    input_size = int(config.get("input_size", 256))
    if num_classes != 2:
        raise ModelLoadError(f"Expected a 2-class model, checkpoint says {num_classes}.")
    spec = spec_from_config(config)
    post = load_postprocessing(path.parent, spec)
    if post["water_gate"] or post["haze_check"]:
        spec = replace(spec, **post)
    in_channels = int(config.get("in_channels", len(spec.bands)))
    if in_channels != len(spec.bands):
        raise ModelLoadError(f"Checkpoint expects {in_channels} input channels but lists {len(spec.bands)} bands.")

    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise ModelLoadError("segmentation_models_pytorch is not installed.") from e

    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=in_channels, classes=num_classes)
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except RuntimeError as e:
        raise ModelLoadError(f"Checkpoint weights do not match U-Net/{encoder}: {e}") from e

    device = device or select_device()[0]
    model = model.to(device).eval()

    val_metrics = ckpt.get("val_metrics") or {}
    meta = {
        "checkpoint": str(path),
        "version": spec.version,
        "architecture": "U-Net",
        "encoder": encoder,
        "num_classes": num_classes,
        "input_size": input_size,
        "bands": list(spec.bands),
        "target": spec.target,
        "preprocessing": spec.preprocessing_id,
        "postprocessing": spec.postprocessing_id,
        "spec": spec,
        "epoch": ckpt.get("epoch"),
        "val_metrics_at_save": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in val_metrics.items()
            if k in ("flood_iou", "flood_dice", "flood_precision", "flood_recall", "water_iou")
        },
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    return model, meta


# ============================================================
# Raster reading and preprocessing helpers
# ============================================================
def read_s2_rgb(path: Path | str, rgb_bands_0idx: Sequence[int] = S2_RGB_BANDS_0IDX) -> np.ndarray:
    """Read Red, Green, Blue DN values from a GeoTIFF. Returns H x W x 3 float32 (>= 0).

    Same as FloodSegDataset._read_s2_rgb.
    """
    import rasterio

    path = Path(path)
    if not path.is_file():
        raise InvalidInputError(f"Image not found: {path}")
    with rasterio.open(path) as src:
        if max(rgb_bands_0idx) + 1 > src.count:
            raise InvalidInputError(
                f"{path.name} has {src.count} bands; cannot read 0-indexed bands {tuple(rgb_bands_0idx)}."
            )
        bands = [src.read(b + 1).astype(np.float32) for b in rgb_bands_0idx]
    rgb = np.stack(bands, axis=-1)
    return np.clip(rgb, 0, None)


def read_mask(path: Path | str) -> np.ndarray:
    """Read a single-band mask GeoTIFF as H x W uint8."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8)


def _validate_rgb(rgb: np.ndarray, channels: int = 3) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != channels:
        raise InvalidInputError(f"Expected an H x W x {channels} array, got shape {getattr(rgb, 'shape', None)}.")
    if rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise InvalidInputError("Image has zero height or width.")
    rgb = rgb.astype(np.float32, copy=False)
    if not np.all(np.isfinite(rgb)):
        raise InvalidInputError("Image contains NaN or infinite values.")
    return np.clip(rgb, 0, None)


def nodata_mask(rgb_raw: np.ndarray) -> np.ndarray:
    """True where all input bands are 0 (no image data)."""
    return np.all(rgb_raw == 0, axis=-1)


def pad_to_min_size(rgb_raw: np.ndarray, size: int) -> np.ndarray:
    """Zero-pad bottom/right so both sides are >= size (training behaviour)."""
    h, w = rgb_raw.shape[:2]
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return rgb_raw
    return np.pad(rgb_raw, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)


def preprocess_resize(rgb_raw: np.ndarray, size: int = 256) -> np.ndarray:
    """Whole-scene path: pad (if small) -> Resize(size) -> normalise. Returns size x size x 3.

    Identical to FloodSegDataset.__getitem__ with augment=False.
    """
    padded = pad_to_min_size(rgb_raw, size)
    resized = get_val_transform(size)(image=padded)["image"]
    return normalize_s2_rgb(resized)


def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    """Start offsets so tiles cover [0, length) and the last tile ends at length."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def blend_window(tile: int, overlap: int) -> np.ndarray:
    """2-D weight that tapers linearly over `overlap` px at tile edges (reduces seams)."""
    w = np.ones(tile, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(1.0 / (overlap + 1), 1.0, overlap, dtype=np.float32)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return np.outer(w, w)


def stretch_for_display(rgb_raw: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Percentile (2-98%) contrast stretch to uint8. Display only; never fed to the model."""
    reflect = np.clip(rgb_raw / 10000.0, 0.0, 1.0)
    sample = reflect[valid] if valid is not None and valid.any() else reflect.reshape(-1, 3)
    if sample.shape[0] > 2_000_000:  # subsample huge scenes for speed
        sample = sample[:: sample.shape[0] // 2_000_000 + 1]
    lo = np.percentile(sample, 2, axis=0)
    hi = np.percentile(sample, 98, axis=0)
    out = (reflect - lo) / np.maximum(hi - lo, 1e-6)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


# ============================================================
# Visual rendering helpers (pure numpy; all inputs share one grid)
# ============================================================
def render_overlay(
    rgb_uint8: np.ndarray, flood_mask: np.ndarray,
    color: Tuple[int, int, int] = FLOOD_COLOR, alpha: float = 0.55,
) -> np.ndarray:
    """Blend a translucent colour over predicted flood pixels."""
    if rgb_uint8.shape[:2] != flood_mask.shape:
        raise InvalidInputError(f"Overlay shape mismatch: image {rgb_uint8.shape[:2]} vs mask {flood_mask.shape}.")
    out = rgb_uint8.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    out[flood_mask] = (1 - alpha) * out[flood_mask] + alpha * c
    return out.astype(np.uint8)


def render_reference(rgb_uint8: np.ndarray, stored_seg: np.ndarray) -> np.ndarray:
    """EMS reference: flood in blue, ignored/unsupervised pixels brown, non-flood dimmed image."""
    if rgb_uint8.shape[:2] != stored_seg.shape:
        raise InvalidInputError("Reference shape mismatch.")
    out = (rgb_uint8.astype(np.float32) * 0.6)
    out[stored_seg == STORED_IGNORE] = IGNORE_COLOR
    out[stored_seg == STORED_FLOOD] = FLOOD_COLOR
    return out.astype(np.uint8)


def compare_with_reference(flood_mask: np.ndarray, stored_seg: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pixel-wise comparison of a prediction with the stored EMS mask (0 ignore, 1 non-flood, 2 flood).

    Returns (error code map, counts/metrics). Ignored reference pixels are excluded.
    """
    if flood_mask.shape != stored_seg.shape:
        raise InvalidInputError(f"Shape mismatch: prediction {flood_mask.shape} vs reference {stored_seg.shape}.")
    sup = (stored_seg == STORED_NON_FLOOD) | (stored_seg == STORED_FLOOD)
    true_f = stored_seg == STORED_FLOOD
    pred_f = flood_mask.astype(bool)

    codes = np.full(stored_seg.shape, ERR_IGNORE, dtype=np.uint8)
    codes[sup & ~true_f & ~pred_f] = ERR_TN
    codes[sup & true_f & pred_f] = ERR_TP
    codes[sup & ~true_f & pred_f] = ERR_FP
    codes[sup & true_f & ~pred_f] = ERR_FN

    tp = int((codes == ERR_TP).sum()); fp = int((codes == ERR_FP).sum())
    fn = int((codes == ERR_FN).sum()); tn = int((codes == ERR_TN).sum())
    return codes, confusion_metrics(tp, fp, fn, tn)


def confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    """Same formulas as FloodMetrics.compute(); None where undefined."""
    def ratio(a, b):
        return a / b if b > 0 else None
    total = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flood_iou": ratio(tp, tp + fp + fn),
        "flood_dice": ratio(2 * tp, 2 * tp + fp + fn),
        "flood_precision": ratio(tp, tp + fp),
        "flood_recall": ratio(tp, tp + fn),
        "pixel_accuracy": ratio(tp + tn, total),
    }


def render_error_map(rgb_uint8: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Error view: TP cyan, FP magenta, FN yellow, TN dimmed image, ignored brown."""
    if rgb_uint8.shape[:2] != codes.shape:
        raise InvalidInputError("Error-map shape mismatch.")
    out = rgb_uint8.astype(np.float32) * 0.45
    out[codes == ERR_IGNORE] = IGNORE_COLOR
    for code, col in ERROR_COLORS.items():
        out[codes == code] = col
    return out.astype(np.uint8)


def downsample_for_display(arr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Shrink for display/transport. Masks/codes use nearest so labels are not blended."""
    h, w = arr.shape[:2]
    scale = max(h, w) / max_side
    if scale <= 1:
        return arr
    from PIL import Image

    new_size = (max(1, round(w / scale)), max(1, round(h / scale)))
    if arr.dtype == bool:
        img = Image.fromarray(arr.astype(np.uint8) * 255)
        return np.array(img.resize(new_size, Image.NEAREST)) > 127
    resample = Image.BILINEAR if (arr.ndim == 3 and arr.dtype == np.uint8) else Image.NEAREST
    return np.array(Image.fromarray(arr).resize(new_size, resample))


def encode_png_base64(arr: np.ndarray) -> str:
    """Encode an H x W x 3 uint8 (or H x W bool/uint8) array as base64 PNG."""
    from PIL import Image

    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ============================================================
# Inference engine
# ============================================================
class FloodSegmenter:
    """Runs the trained U-Net on Sentinel-2 RGB scenes.

    Usage:
        seg = FloodSegmenter()                      # loads best_seg_model.pt
        res = seg.predict_file("scene.tif", method="tiled")
        res.stats["flood_coverage_pct"]

    For tests, pass `model=` (any nn.Module mapping B x 3 x H x W -> B x 2 x H x W).
    """

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
        model: Optional[nn.Module] = None,
        input_size: int = 256,
        use_amp: bool = True,
        spec: Optional[ModelSpec] = None,
    ):
        self.device, dev_warning = select_device(device)
        self.load_warnings = [dev_warning] if dev_warning else []
        if model is not None:
            self.model = model.to(self.device).eval()
            self.spec = spec or V1_SPEC
            self.meta = {"checkpoint": None, "architecture": type(model).__name__, "input_size": input_size,
                         "version": self.spec.version, "bands": list(self.spec.bands), "target": self.spec.target,
                         "preprocessing": self.spec.preprocessing_id}
        else:
            self.model, self.meta = load_segmentation_model(checkpoint_path, self.device)
            self.spec = self.meta.pop("spec")
            input_size = self.meta.get("input_size", input_size)
        self.input_size = int(input_size)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.meta["device"] = str(self.device)

    # ---------- model forward ----------
    @torch.no_grad()
    def _forward_probs(self, batch_hwc: np.ndarray) -> np.ndarray:
        """B x H x W x 3 normalised -> B x H x W float32 P(flood)."""
        x = torch.from_numpy(np.ascontiguousarray(batch_hwc)).permute(0, 3, 1, 2).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits = self.model(x)
        if logits.ndim != 4 or logits.shape[1] != 2 or logits.shape[2:] != x.shape[2:]:
            raise RuntimeError(f"Unexpected model output shape {tuple(logits.shape)} for input {tuple(x.shape)}.")
        probs = F.softmax(logits.float(), dim=1)[:, 1]
        return probs.cpu().numpy().astype(np.float32)

    def _predict_resize(self, rgb_raw: np.ndarray) -> np.ndarray:
        h, w = rgb_raw.shape[:2]
        if self.spec.normalization.get("kind") == "imagenet_rgb":
            x = preprocess_resize(rgb_raw, self.input_size)  # v1: exactly the Phase H evaluation transform
        else:
            norm = self.spec.normalize(pad_to_min_size(rgb_raw, self.input_size))
            t = torch.from_numpy(np.ascontiguousarray(norm)).permute(2, 0, 1)[None]
            x = F.interpolate(t, size=(self.input_size, self.input_size), mode="bilinear",
                              align_corners=False)[0].permute(1, 2, 0).numpy()
        p256 = self._forward_probs(x[None])[0]
        ph, pw = max(h, self.input_size), max(w, self.input_size)  # padded grid
        t = torch.from_numpy(p256)[None, None]
        up = F.interpolate(t, size=(ph, pw), mode="bilinear", align_corners=False)[0, 0].numpy()
        return up[:h, :w]

    def _predict_tiled(self, rgb_raw: np.ndarray, overlap: int, batch_size: int) -> Tuple[np.ndarray, int]:
        t = self.input_size
        if not 0 <= overlap < t // 2 + 1:
            raise ValueError(f"overlap must be in [0, {t // 2}], got {overlap}.")
        h, w = rgb_raw.shape[:2]
        padded = pad_to_min_size(rgb_raw, t)
        ph, pw = padded.shape[:2]
        norm = self.spec.normalize(padded)  # per-pixel op, identical to training normalisation
        stride = t - overlap
        weight = blend_window(t, overlap)
        acc = np.zeros((ph, pw), dtype=np.float32)
        wsum = np.zeros((ph, pw), dtype=np.float32)
        coords = [(y, x) for y in tile_starts(ph, t, stride) for x in tile_starts(pw, t, stride)]
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            batch = np.stack([norm[y:y + t, x:x + t] for y, x in chunk])
            probs = self._forward_probs(batch)
            for (y, x), p in zip(chunk, probs):
                acc[y:y + t, x:x + t] += p * weight
                wsum[y:y + t, x:x + t] += weight
        prob = acc / np.maximum(wsum, 1e-8)
        return prob[:h, :w], len(coords)

    # ---------- public API ----------
    def predict_array(
        self,
        rgb_raw: np.ndarray,
        validity: Optional[np.ndarray] = None,
        method: str = "tiled",
        overlap: int = 64,
        batch_size: int = 8,
        source: str = "array",
        make_display: bool = True,
    ) -> SegmentationResult:
        """Segment an H x W x C array of raw Sentinel-2 DN values in the model's band order
        (v1: R, G, B; v2: B02, B03, B04, B08, B11, B12 — see self.spec.bands).

        validity: optional H x W array, 1 = valid/cloud-free, 0 = exclude.
        make_display: build the contrast-stretched RGB (skip for batch evaluation).
        The positive class is self.spec.target (EMS flood for v1, surface water for v2).
        """
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Use one of {METHODS}.")
        rgb_raw = _validate_rgb(rgb_raw, len(self.spec.bands))
        h, w = rgb_raw.shape[:2]
        if validity is not None and validity.shape != (h, w):
            raise InvalidInputError(f"Validity mask shape {validity.shape} does not match image {(h, w)}.")

        t0 = time.time()
        has_data = ~nodata_mask(rgb_raw)
        eligible = has_data & (validity.astype(bool) if validity is not None else True)

        n_tiles = None
        if method == "resize":
            prob = self._predict_resize(rgb_raw)
        else:
            prob, n_tiles = self._predict_tiled(rgb_raw, overlap, batch_size)

        flood = (prob > FLOOD_THRESHOLD) & eligible
        haze = self.spec.haze_index(rgb_raw, eligible)
        hazy = self.spec.haze_check is not None and self.spec.is_hazy(haze)
        n_gated_out = 0
        gate = self.spec.gate_mask(rgb_raw) if (self.spec.haze_check is None or hazy) else None
        if gate is not None:
            n_before = int(flood.sum())
            flood &= gate
            n_gated_out = n_before - int(flood.sum())
        n_total = h * w
        n_elig = int(eligible.sum())
        n_flood = int(flood.sum())
        coverage = (100.0 * n_flood / n_elig) if n_elig > 0 else None

        warnings = list(self.load_warnings)
        if n_elig == 0:
            quality = "no_valid_pixels"
            warnings.append("No eligible pixels: the scene has no usable image data in the valid area.")
        elif n_elig / n_total < 0.5:
            quality = "limited"
            warnings.append(f"Only {100 * n_elig / n_total:.1f}% of pixels are eligible; interpretation is limited.")
        else:
            quality = "ok"
        if validity is None:
            warnings.append("No cloud/validity mask available; clouds and cloud shadows may be counted and misclassified.")
        if hazy:
            warnings.append(
                f"Hazy image: haze index {haze:.3f} is above every training image "
                f"({float(self.spec.haze_check['max_toa_reflectance']):.3f}). Only water that is also dark in "
                "short-wave infrared was counted, and the outcome is reported as inconclusive."
            )
        if method == "resize" and max(h, w) > self.input_size:
            warnings.append(
                f"Whole-scene resize: {w}x{h} px was shrunk to {self.input_size}x{self.input_size}; "
                "small or narrow flood features may be lost."
            )

        stats = {
            "source": source,
            "method": method,
            "height": h,
            "width": w,
            "n_pixels": n_total,
            "n_eligible": n_elig,
            "n_flood_pred": n_flood,
            "flood_coverage_pct": None if coverage is None else round(coverage, 4),
            "coverage_definition": "predicted flood pixels / eligible pixels x 100 (pixel ratio, not area)",
            "eligible_pct": round(100.0 * n_elig / n_total, 4),
            "nodata_pct": round(100.0 * float((~has_data).sum()) / n_total, 4),
            "eligibility_source": "validity_mask+nodata" if validity is not None else "nodata_only",
            "flood_threshold": FLOOD_THRESHOLD,
            "input_size": self.input_size,
            "tile_overlap": overlap if method == "tiled" else None,
            "n_tiles": n_tiles,
            "device": str(self.device),
            "elapsed_s": round(time.time() - t0, 3),
            "quality_status": quality,
            "target": self.spec.target,
            "model_version": self.spec.version,
            "postprocessing": self.spec.postprocessing_id,
            "n_removed_by_water_gate": n_gated_out,
            "water_gate_applied": gate is not None,
            "haze_index": None if haze is None else round(haze, 4),
            "hazy": bool(hazy),
        }
        rgb = rgb_raw[..., list(self.spec.rgb_index)]
        return SegmentationResult(
            rgb_display=stretch_for_display(rgb, has_data) if make_display else np.zeros((0, 0, 3), np.uint8),
            flood_probability=prob.astype(np.float32),
            flood_mask=flood,
            eligible_mask=eligible,
            stats=stats,
            warnings=warnings,
        )

    def predict_file(
        self,
        image_path: Path | str,
        rgb_bands_0idx: Sequence[int] = S2_RGB_BANDS_0IDX,
        validity_path: Optional[Path | str] = None,
        **kwargs,
    ) -> SegmentationResult:
        """Read a GeoTIFF (and optional validity mask) and segment it.

        v1 (RGB) honours `rgb_bands_0idx`; multispectral models read their own bands from the
        WorldFloods band layout.
        """
        if self.spec.bands == V1_SPEC.bands:
            rgb = read_s2_rgb(image_path, rgb_bands_0idx)
        else:
            rgb = read_scene(image_path, self.spec)
        validity = read_mask(validity_path) if validity_path else None
        kwargs.setdefault("source", Path(image_path).name)
        return self.predict_array(rgb, validity=validity, **kwargs)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run flood segmentation on one Sentinel-2 GeoTIFF.")
    ap.add_argument("image")
    ap.add_argument("--validity", default=None)
    ap.add_argument("--method", choices=METHODS, default="tiled")
    ap.add_argument("--live-band-order", action="store_true", help="Use 6-band CDSE order (B02,B03,B04,...)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    seg = FloodSegmenter(device=args.device)
    bands = LIVE_RGB_BANDS_0IDX if args.live_band_order else S2_RGB_BANDS_0IDX
    r = seg.predict_file(args.image, bands, args.validity, method=args.method)
    for k, v in r.stats.items():
        print(f"{k:22s} {v}")
    for wmsg in r.warnings:
        print("WARNING:", wmsg)
