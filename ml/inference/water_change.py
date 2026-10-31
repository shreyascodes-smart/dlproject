"""
Water-change evidence: compare today's predicted water with the same area about one year earlier.

Both images come from the same Copernicus request grid (same bbox, size and 10 m resolution), the same
bands and the same model, and the baseline is searched in the same season (about 335-395 days
earlier). Only pixels valid (cloud-free, with data) in BOTH images are compared.

Definitions (on normally-dry land, i.e. outside known permanent / seasonal / mapped water):
  candidate   predicted water today
  persistent  candidate pixels that were also predicted water in the baseline image
  new         candidate pixels that were NOT water in the baseline image

How the assessment uses it (conservative by design; it can only remove a claim, never create one):
  * flood_like + most candidate water also present a year earlier  -> inconclusive (may be ordinary water)
  * inconclusive + new water on dry land below the policy's "none" floor -> none_detected
    (the flood-like water there was already present a year earlier, so it is not new water)
These rules are design choices; no labelled time-series data exists in this project to validate them.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ml.inference.assessment import KW_NONE

MIN_OVERLAP_FRAC = 0.5       # both images must share at least half of today's usable pixels
PERSISTENT_MAJORITY = 0.5    # "most candidate water was already there"


def unavailable(reason: str) -> Dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def compare(
    current_water: np.ndarray,
    current_eligible: np.ndarray,
    baseline_water: np.ndarray,
    baseline_eligible: np.ndarray,
    known_water: Optional[np.ndarray],
    current_date: Optional[str] = None,
    baseline_date: Optional[str] = None,
    baseline_hazy: bool = False,
) -> Dict[str, Any]:
    if baseline_water.shape != current_water.shape or baseline_eligible.shape != current_eligible.shape:
        return unavailable("The earlier image is on a different grid and cannot be compared pixel by pixel.")
    cur_e = current_eligible.astype(bool)
    both = cur_e & baseline_eligible.astype(bool)
    dry = np.ones(cur_e.shape, bool) if known_water is None else (known_water == KW_NONE)
    zone = both & dry
    cand = current_water.astype(bool) & zone
    base = baseline_water.astype(bool) & zone
    persistent = cand & base
    new = cand & ~base
    receded = base & ~current_water.astype(bool)

    n_cur = int(cur_e.sum())
    n_zone = int(zone.sum())
    overlap = (int(both.sum()) / n_cur) if n_cur else 0.0

    def pct(a, b):
        return None if b == 0 else round(100.0 * a / b, 4)

    days = None
    if current_date and baseline_date:
        from datetime import date
        days = (date.fromisoformat(current_date[:10]) - date.fromisoformat(baseline_date[:10])).days

    out = {
        "status": "available",
        "baseline_date": baseline_date,
        "days_between": days,
        "compared_pct_of_usable": round(100.0 * overlap, 2),
        "n_compared_dry_land": n_zone,
        "n_candidate": int(cand.sum()),
        "n_persistent": int(persistent.sum()),
        "n_new": int(new.sum()),
        "n_receded": int(receded.sum()),
        "water_now_pct_of_dry_land": pct(int(cand.sum()), n_zone),
        "water_then_pct_of_dry_land": pct(int(base.sum()), n_zone),
        "new_water_pct_of_dry_land": pct(int(new.sum()), n_zone),
        "persistent_share_of_candidate_pct": pct(int(persistent.sum()), int(cand.sum())),
        "baseline_hazy": bool(baseline_hazy),
        "method": "Same model, bands and 10 m grid; baseline from the same season about one year earlier; "
                  "only pixels usable in both images are compared.",
    }
    if overlap < MIN_OVERLAP_FRAC:
        out["status"] = "limited"
        out["reason"] = (f"Only {100 * overlap:.0f}% of today's usable pixels are also usable in the earlier image "
                         "(cloud or missing data), so the comparison is not used.")
    elif baseline_hazy:
        out["status"] = "limited"
        out["reason"] = "The earlier image is hazier than any training image, so its water map is not reliable."
    return out


def apply_to_outcome(level: str, change: Optional[Dict[str, Any]], none_below_pct: Optional[float]):
    """(new_level or None, reason or None). Only ever moves toward less certainty or toward 'none'
    when the water was demonstrably already present."""
    if not change or change.get("status") != "available":
        return None, None
    persistent = change.get("persistent_share_of_candidate_pct")
    new_pct = change.get("new_water_pct_of_dry_land")
    if level == "flood_like" and persistent is not None and persistent >= 100 * PERSISTENT_MAJORITY:
        return "inconclusive", (f"{persistent:.0f}% of the flood-like water was also water in the image from "
                                f"{change.get('baseline_date')}, so it may be ordinary water rather than flooding.")
    if level == "inconclusive" and none_below_pct is not None and new_pct is not None and new_pct < none_below_pct:
        return "none_detected", (f"Almost all of the flood-like water was already present on "
                                 f"{change.get('baseline_date')}; new water covers only {new_pct:.1f}% of the "
                                 "normally dry land.")
    return None, None
