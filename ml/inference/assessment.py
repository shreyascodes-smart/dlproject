"""
User-facing assessment rules for the U-Net flood segmentation output.

What this module does NOT do: it does not produce a flood probability, a
disaster-risk rating, or a statement that an area is safe. The segmentation
model only marks pixels that look like flood water in one image.

What it does:
  1. Separates predicted flood-like pixels on KNOWN water (permanent water
     bodies / open sea, from the JRC Global Surface Water reference) from
     those elsewhere. The raw prediction is kept unchanged.
  2. Turns the remaining "flood-like water outside known water bodies"
     share into one of a few plain-language assessment states, using
     thresholds derived from the VALIDATION split only
     (see ml/evaluation/assessment_calibration.py and its JSON report).

Known-water codes (same grid as the prediction):
    0 = not known water (or unknown)
    1 = permanent water (JRC: permanent / 12 months of water)
    2 = outside the JRC water dataset's coverage, i.e. open sea
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

KW_NONE = 0        # normally dry land (no known water)
KW_PERMANENT = 1   # permanent water (JRC: water all year)
KW_SEA = 2         # open sea (outside the JRC dataset coverage)
KW_SEASONAL = 3    # seasonal water (JRC: water part of the year) or mapped wetland / tidal flat
KW_MAPPED = 4      # mapped ordinary water body (OpenStreetMap lake, pond, reservoir, river, canal, drain)
KW_NAMES = {KW_PERMANENT: "permanent", KW_SEA: "sea", KW_SEASONAL: "seasonal", KW_MAPPED: "mapped"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = PROJECT_ROOT / "data" / "reports"
# Thresholds are model-specific: each is derived on the frozen VALIDATION split for that model.
POLICY_FILES = {
    "segmentation_v1": REPORTS / "kiro_assessment_policy_val_2026-10-26.json",
    # Milestone 6: derived with JRC permanent AND seasonal water set aside (matches the served reference).
    # Milestone 5 (permanent only): assessment_policy_segmentation_v2_val.json (kept for comparison).
    "segmentation_v2": REPORTS / "assessment_policy_segmentation_v2_seasonal_val.json",
}
POLICY_FILE = POLICY_FILES["segmentation_v1"]  # backward-compatible name

LEVELS = ("insufficient_coverage", "uncertain_water", "none_detected", "inconclusive", "flood_like")


def policy_path(model_version: str = "segmentation_v1") -> Path:
    return POLICY_FILES.get(model_version, REPORTS / f"assessment_policy_{model_version}_val.json")


def load_policy(path_or_version="segmentation_v1") -> Optional[Dict[str, Any]]:
    """Validation-derived policy for a model, or None when none exists (no invented fallback)."""
    p = Path(path_or_version) if str(path_or_version).endswith(".json") else policy_path(str(path_or_version))
    try:
        data = json.loads(Path(p).read_text())
        if data.get("split") != "val":
            return None
        return data["policy"]
    except Exception:
        return None


def water_context(flood_mask: np.ndarray, eligible: np.ndarray, known_water: Optional[np.ndarray]) -> Dict[str, Any]:
    """Pixel accounting of predicted flood-like water against known water bodies."""
    if known_water is not None and known_water.shape != flood_mask.shape:
        raise ValueError(f"Known-water mask {known_water.shape} does not match prediction {flood_mask.shape}.")
    kw = np.zeros(flood_mask.shape, bool) if known_water is None else known_water > 0
    sea = np.zeros(flood_mask.shape, bool) if known_water is None else known_water == KW_SEA
    flood = flood_mask.astype(bool) & eligible.astype(bool)

    n_elig = int(eligible.sum())
    n_land = int((eligible & ~kw).sum())
    n_flood_raw = int(flood.sum())
    n_flood_known = int((flood & kw).sum())
    n_flood_land = n_flood_raw - n_flood_known

    def pct(a, b):
        return None if b == 0 else round(100.0 * a / b, 4)

    by_class = {}
    if known_water is not None:
        for code, name in KW_NAMES.items():
            m = eligible & (known_water == code)
            by_class[name] = {"n_eligible": int(m.sum()), "n_water_predicted": int((flood & m).sum())}

    return {
        "available": known_water is not None,
        "by_class": by_class,
        "n_eligible": n_elig,
        "n_eligible_outside_known_water": n_land,
        "known_water_pct_of_eligible": pct(n_elig - n_land, n_elig),
        "sea_pct_of_eligible": pct(int((eligible & sea).sum()), n_elig),
        "n_flood_like_raw": n_flood_raw,
        "n_flood_like_on_known_water": n_flood_known,
        "n_flood_like_outside_known_water": n_flood_land,
        "raw_flood_like_pct": pct(n_flood_raw, n_elig),
        "flood_like_on_known_water_share_pct": pct(n_flood_known, n_flood_raw),
        "flood_like_outside_known_water_pct": pct(n_flood_land, n_land),
    }


def outside_water_mask(flood_mask: np.ndarray, known_water: Optional[np.ndarray]) -> np.ndarray:
    """Flood-like pixels that are NOT on known permanent water / sea (for the map overlay)."""
    if known_water is None:
        return flood_mask.astype(bool)
    return flood_mask.astype(bool) & (known_water == KW_NONE)


HEADLINES = {
    "insufficient_coverage": "Insufficient valid coverage",
    "uncertain_water": "Inconclusive — mostly permanent water",
    "none_detected": "No flood-like patterns detected",
    "inconclusive": "Inconclusive",
    "flood_like": "Flood-like patterns detected",
}


def classify(
    wc: Dict[str, Any],
    method: str,
    quality_status: str,
    policy: Optional[Dict[str, Any]],
    eligible_pct: float,
    live: bool,
    hazy: bool = False,
    change: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map pixel accounting to a descriptive outcome (not a probability and not a risk rating).

    Outcomes:
      insufficient_coverage  too little usable (cloud-free, valid) image for this area
      uncertain_water        the usable area is mostly known permanent water / sea
      none_detected          flood-like share below the floor, in the valid analysed pixels
      inconclusive           within the range this model also marks on validation scenes without flooding
      flood_like             above anything the model marked on validation scenes without flooding
    """
    n_elig = wc["n_eligible"]
    land_frac = (wc["n_eligible_outside_known_water"] / n_elig) if n_elig else 0.0
    value = wc["flood_like_outside_known_water_pct"]
    limitations: List[str] = []
    if not wc["available"]:
        limitations.append("Permanent-water reference unavailable: lakes, rivers or sea may be counted as flood-like.")
    on_water = wc["flood_like_on_known_water_share_pct"]
    if on_water is not None and on_water >= 25:
        limitations.append(f"{on_water:.0f}% of the water the model found lies on known permanent water or sea and "
                           "was set aside.")

    reasons: List[str] = []
    if policy is None:
        reasons.append("This model has no validated outcome thresholds yet.")
        min_land = 0.25
    else:
        min_land = policy["min_assessable_land_frac"]
    if n_elig == 0 or quality_status == "no_valid_pixels":
        reasons.insert(0, "The image has no usable cloud-free pixels for this area.")
    elif eligible_pct < 100 * min_land:
        reasons.insert(0, f"Only {eligible_pct:.0f}% of the image is usable; the rest is cloud, shadow or missing data.")

    if n_elig == 0 or quality_status == "no_valid_pixels" or eligible_pct < 100 * min_land:
        level = "insufficient_coverage"
    elif policy is None:
        level = "inconclusive"
    elif land_frac < 1 - policy["water_dominated_frac"] and wc["available"]:
        level = "uncertain_water"
    elif value is None:
        level = "insufficient_coverage"
        reasons.append("No usable land pixels remain after setting aside known water.")
    elif value < policy["none_below_pct"]:
        level = "none_detected"
    elif value <= policy["inconclusive_ceiling_pct"].get(method, policy["inconclusive_ceiling_pct"]["tiled"]):
        level = "inconclusive"
    else:
        level = "flood_like"

    # Without a known-water reference, ordinary lakes/rivers/sea cannot be separated: never claim flooding
    if not wc["available"] and level == "flood_like":
        level = "inconclusive"
        reasons.append("Known lakes, rivers and sea could not be set aside, so ordinary water cannot be told apart "
                       "from flood water.")

    # Outside the training range (haze): no claim either way
    if hazy and level in ("flood_like", "none_detected", "inconclusive"):
        if level != "inconclusive":
            level = "inconclusive"
        reasons.append("The image is hazier than any image the model was trained on, so built-up areas can look "
                       "like water. Only water that is also dark in short-wave infrared was counted.")

    # Year-earlier comparison (live only): can withdraw a claim or confirm "no new water", never create a flood
    by_history = False
    if not hazy and change is not None:
        from ml.inference.water_change import apply_to_outcome
        new_level, why = apply_to_outcome(level, change, None if policy is None else policy["none_below_pct"])
        if new_level:
            level, by_history = new_level, True
            reasons.append(why)

    hazy_level = hazy and level == "inconclusive"
    summary = {
        "insufficient_coverage": "There are not enough clear pixels in this image to assess the area.",
        "uncertain_water": "Most of the usable area is permanent water or sea, where flood water cannot be told "
                           "apart from normal water in a single image.",
        "none_detected": "No flood-like water was found outside known water bodies in the valid analysed pixels. "
                         "Flooding under cloud, vegetation or after the image date would not be visible.",
        "inconclusive": "Some flood-like water was found, but no more than this model also marks on reference "
                        "scenes without mapped flooding.",
        "flood_like": "More flood-like water was found outside known water bodies than this model marked on any "
                      "reference scene without mapped flooding. Not verified against ground observations.",
    }[level]
    if hazy_level:
        summary = "The image is too hazy for a reliable reading: haze can make built-up areas look like water."
    elif by_history and level == "none_detected":
        summary = ("The water found on normally dry land was already there in the same season a year earlier, so "
                   "there is no evidence of new flooding in the valid analysed pixels.")
    elif by_history and level == "inconclusive":
        summary = "Flood-like water was found, but most of it was also present a year earlier."
    suggestion = {
        "insufficient_coverage": "Try again when a clearer image is available.",
        "uncertain_water": "Choose a point further inland to assess land areas.",
    }.get(level)
    if hazy_level:
        suggestion = "Try again when a clearer image is available."

    ceiling = None if policy is None else policy["inconclusive_ceiling_pct"].get(
        method, policy["inconclusive_ceiling_pct"]["tiled"])
    return {
        "level": level,
        "headline": HEADLINES[level],
        "summary": summary,
        "reasons": reasons,
        "limitations": limitations,
        "uncertainty": limitations,   # backward-compatible alias
        "suggestion": suggestion,
        "next_steps": [suggestion] if suggestion else [],
        "flood_like_outside_known_water_pct": value,
        "hazy": bool(hazy),
        "historical_comparison_used": by_history,
        "basis": {
            "kind": "descriptive_model_outcome",
            "is_risk_rating": False,
            "is_probability": False,
            "method": method,
            "none_below_pct": None if policy is None else policy["none_below_pct"],
            "inconclusive_ceiling_pct": ceiling,
            "min_assessable_land_frac": min_land,
            "water_dominated_frac": None if policy is None else policy["water_dominated_frac"],
            "policy_version": None if policy is None else policy.get("version"),
            "explanation": (
                "Outcomes compare the flood-like share outside known water bodies with the largest share this model "
                "marked on validation scenes without mapped flooding. They are not probabilities or risk ratings."
            ),
        },
    }
