"""
Unit tests for the assessment rules, permanent-water reference and live-scene helpers.

Network test (JRC Global Surface Water over the Mumbai sea point) runs only when
the data host is reachable; it never contacts Copernicus.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.inference import assessment as asm  # noqa: E402
from app.services import water_reference as wr  # noqa: E402
from app.services.sentinel_service import SCL_INVALID, pick_acquisition, validity_from_scl  # noqa: E402

POLICY = {"version": "t", "none_below_pct": 1.0, "inconclusive_ceiling_pct": {"tiled": 28.39, "resize": 17.9},
          "min_assessable_land_frac": 0.25, "water_dominated_frac": 0.5}


def wc_for(flood_frac, water_frac=0.0, sea=False, n=1000, available=True):
    """Build masks: first `water_frac` of pixels are known water; flood-like pixels spread over land."""
    flood = np.zeros(n, bool)
    eligible = np.ones(n, bool)
    kw = np.zeros(n, np.uint8)
    nw = int(n * water_frac)
    kw[:nw] = asm.KW_SEA if sea else asm.KW_PERMANENT
    land = np.arange(nw, n)
    flood[land[: int(round(len(land) * flood_frac))]] = True
    return asm.water_context(flood, eligible, kw if available else None)


class TestClassify:
    def c(self, wc, method="tiled", quality="ok", eligible_pct=100.0):
        return asm.classify(wc, method, quality, POLICY, eligible_pct, live=True)

    def test_levels_follow_documented_thresholds(self):
        assert self.c(wc_for(0.005))["level"] == "none_detected"
        assert self.c(wc_for(0.05))["level"] == "inconclusive"
        assert self.c(wc_for(0.28))["level"] == "inconclusive"
        wc = wc_for(0.3)
        wc["flood_like_outside_known_water_pct"] = 28.39
        assert self.c(wc)["level"] == "inconclusive"                 # ceiling is inclusive
        assert self.c(wc_for(0.35))["level"] == "flood_like"
        assert self.c(wc_for(0.20), method="resize")["level"] == "flood_like"   # method-specific ceiling

    def test_water_dominated_is_uncertain(self):
        a = self.c(wc_for(0.0, water_frac=0.8, sea=True))
        assert a["level"] == "uncertain_water"
        assert a["headline"] == "Inconclusive — mostly permanent water"

    def test_unusable_image_is_unable(self):
        assert self.c(wc_for(0.3), eligible_pct=10)["level"] == "insufficient_coverage"

    def test_hazy_scene_is_never_a_claim(self):
        for frac in (0.005, 0.05, 0.6):                       # would be none / inconclusive / flood_like
            a = asm.classify(wc_for(frac), "tiled", "ok", POLICY, 100.0, live=True, hazy=True)
            assert a["level"] == "inconclusive" and a["hazy"] is True
            assert any("hazier than any image" in r for r in a["reasons"])
            assert "too hazy" in a["summary"]
        # coverage and permanent-water states keep priority
        assert asm.classify(wc_for(0.3), "tiled", "ok", POLICY, 10.0, live=True, hazy=True)["level"] == "insufficient_coverage"
        assert asm.classify(wc_for(0.0, 0.8, sea=True), "tiled", "ok", POLICY, 100.0, live=True, hazy=True)["level"] == "uncertain_water"
        assert self.c(wc_for(0.3), quality="no_valid_pixels")["level"] == "insufficient_coverage"

    def test_wording_never_claims_safety_or_probability(self):
        for frac in (0.0, 0.05, 0.5):
            a = self.c(wc_for(frac))
            text = json.dumps({k: a[k] for k in ("headline", "summary", "next_steps")}).lower()
            text = text.replace("(not a probability)", "")
            assert "probability" not in text and "% chance" not in text
            assert "is safe" not in text and not a["headline"].lower().startswith("safe")
            assert a["basis"]["is_risk_rating"] is False
        assert "valid analysed pixels" in self.c(wc_for(0.0))["summary"]

    def test_missing_reference_is_disclosed(self):
        a = self.c(wc_for(0.05, available=False))
        assert any("Permanent-water reference unavailable" in u for u in a["limitations"])

    def test_water_context_accounting(self):
        wc = wc_for(0.5, water_frac=0.2)
        assert wc["n_eligible"] == 1000 and wc["n_eligible_outside_known_water"] == 800
        assert wc["flood_like_outside_known_water_pct"] == pytest.approx(50.0)
        assert wc["known_water_pct_of_eligible"] == pytest.approx(20.0)

    def test_outside_mask_keeps_raw_prediction(self):
        flood = np.array([True, True, False])
        kw = np.array([0, 1, 0], np.uint8)
        assert asm.outside_water_mask(flood, kw).tolist() == [True, False, False]
        assert flood.tolist() == [True, True, False]
        assert asm.outside_water_mask(flood, None).tolist() == [True, True, False]

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            asm.water_context(np.zeros((2, 2), bool), np.ones((2, 2), bool), np.zeros((3, 2), np.uint8))

    def test_policy_file_exists_and_is_validation_derived(self):
        assert asm.POLICY_FILE.is_file()
        data = json.loads(asm.POLICY_FILE.read_text())
        assert data["split"] == "val"
        p = asm.load_policy()
        assert set(p["inconclusive_ceiling_pct"]) == {"tiled", "resize"}
        for m, ceiling in p["inconclusive_ceiling_pct"].items():
            assert ceiling >= data["summary"][m]["negative_outside_water_pct"]["max"]


class TestWaterReference:
    def test_gsw_codes(self):
        s = np.array([0, 5, 12, 255], np.uint8)
        assert wr.from_gsw_seasonality(s).tolist() == [asm.KW_NONE, asm.KW_SEASONAL, asm.KW_PERMANENT, asm.KW_SEA]
        assert wr.from_gsw_seasonality(s, include_seasonal=False).tolist() == [
            asm.KW_NONE, asm.KW_NONE, asm.KW_PERMANENT, asm.KW_SEA]

    def test_worldfloods_codes(self):
        j = np.array([0, 1, 2, 3], np.int16)
        assert wr.from_worldfloods_jrc(j).tolist() == [0, 0, asm.KW_SEASONAL, asm.KW_PERMANENT]

    def test_tile_names(self):
        assert wr.gsw_tiles_for_bbox(72.40, 19.20, 72.45, 19.25) == ["70E_20N"]
        assert wr.gsw_tiles_for_bbox(140.6, -18.5, 141.2, -17.9) == ["140E_10S"]
        assert wr.gsw_tiles_for_bbox(-0.1, 51.4, 0.1, 51.6) == ["10W_60N", "0E_60N"]
        assert wr.gsw_tiles_for_bbox(23.6, 66.3, 23.7, 66.5) == ["20E_70N"]

    def test_failure_returns_none(self):
        ref = wr.WaterReference(url_template="https://invalid.invalid/{tile}.tif")
        mask, info = ref.for_bbox([72.4, 19.2, 72.45, 19.25], 10, 10)
        assert mask is None and info["status"] == "unavailable"


def _network_ok():
    import urllib.request
    try:
        urllib.request.urlopen("https://storage.googleapis.com/global-surface-water/downloads2021/seasonality/"
                               "seasonality_70E_20Nv1_4_2021.tif", timeout=5).close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _network_ok(), reason="JRC Global Surface Water host not reachable")
def test_mumbai_sea_point_is_known_water_network():
    """Real data: the live false-positive location (19.2535 N, 72.4662 E) is open sea in JRC GSW."""
    lat, lng, d = 19.2535, 72.4662, 0.0225
    mask, info = wr.WaterReference(use_osm=False).for_bbox([lng - d, lat - d, lng + d, lat + d], 100, 100)
    assert info["status"] == "ok"
    assert (mask == asm.KW_SEA).mean() > 0.95
    land_mask, _ = wr.WaterReference().for_bbox([72.85, 19.05, 72.90, 19.10], 100, 100)  # Mumbai city
    assert (land_mask == asm.KW_SEA).mean() < 0.2


class TestLiveHelpers:
    def test_pick_most_recent_clear(self):
        f = [{"properties": {"datetime": "2026-10-01T05:00:00Z", "eo:cloud_cover": 5}},
             {"properties": {"datetime": "2026-10-20T05:00:00Z", "eo:cloud_cover": 25}},
             {"properties": {"datetime": "2026-10-24T05:00:00Z", "eo:cloud_cover": 80}}]
        assert pick_acquisition(f, 30)["datetime"].startswith("2026-10-20")

    def test_pick_least_cloudy_when_none_clear(self):
        f = [{"properties": {"datetime": "2026-10-01T05:00:00Z", "eo:cloud_cover": 70}},
             {"properties": {"datetime": "2026-10-20T05:00:00Z", "eo:cloud_cover": 50}}]
        assert pick_acquisition(f, 30)["cloud_cover"] == 50
        assert pick_acquisition([], 30) is None

    def test_validity_from_scl(self):
        dm = np.array([1, 1, 1, 1, 0], np.uint16)
        scl = np.array([4, 9, 3, 6, 0], np.uint16)          # vegetation, cloud, shadow, water, no data
        v, how = validity_from_scl(dm, scl)
        assert how == "sen2cor_scl" and v.tolist() == [1, 0, 0, 1, 0]
        assert set(SCL_INVALID) >= {3, 8, 9, 10}
        v2, how2 = validity_from_scl(dm, np.zeros(5, np.uint16))
        assert v2 is None and how2 == "unavailable"
