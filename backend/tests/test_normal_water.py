"""
Milestone 6 regression tests: ordinary water vs flood evidence.

All tests use SYNTHETIC masks and fixtures. They check that the rules behave as designed; they say
nothing about real-world accuracy (see data/reports/milestone6_flood_intelligence_report.md).
"""
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
from ml.inference import water_change as chg  # noqa: E402
from app.services import water_reference as wr  # noqa: E402

POLICY = {"version": "t", "none_below_pct": 1.0, "inconclusive_ceiling_pct": {"tiled": 28.57, "resize": 14.28},
          "min_assessable_land_frac": 0.25, "water_dominated_frac": 0.5}
H = W = 100
BBOX = [10.0, 45.0, 10.01, 45.01]   # west, south, east, north (100 x 100 px grid)


def classify(water, kw, eligible=None, change=None, hazy=False):
    eligible = np.ones((H, W), bool) if eligible is None else eligible
    wc = asm.water_context(water, eligible, kw)
    return asm.classify(wc, "tiled", "ok", POLICY, 100.0 * eligible.mean(), live=True, hazy=hazy, change=change), wc


def lonlat(r0, c0, r1, c1):
    """Closed OSM ring for pixel rows/cols [r0, r1) x [c0, c1) on the test grid."""
    w, s, e, n = BBOX
    x = lambda c: w + (e - w) * c / W  # noqa: E731
    y = lambda r: n - (n - s) * r / H  # noqa: E731
    pts = [(r0, c0), (r0, c1), (r1, c1), (r1, c0), (r0, c0)]
    return [{"lon": x(c), "lat": y(r)} for r, c in pts]


# ---------------- ordinary water must not become a flood ----------------
class TestOrdinaryWater:
    def test_known_permanent_lake_is_not_flood(self):
        kw = np.zeros((H, W), np.uint8)
        kw[20:60, 20:60] = asm.KW_PERMANENT                   # 16 % of the area is a permanent lake
        water = kw == asm.KW_PERMANENT                        # the model sees exactly the lake
        a, wc = classify(water, kw)
        assert a["level"] == "none_detected" and wc["n_flood_like_outside_known_water"] == 0
        assert wc["by_class"]["permanent"]["n_water_predicted"] == 1600

    @pytest.mark.parametrize("tags", [
        {"natural": "water", "water": "pond"},
        {"natural": "water", "water": "lake"},
        {"landuse": "reservoir"},
        {"natural": "water", "water": "wastewater"},
    ])
    def test_osm_mapped_water_bodies_are_set_aside(self, tags):
        el = {"type": "way", "tags": tags, "geometry": lonlat(30, 30, 70, 70)}
        osm, cats = wr.rasterize_osm([el], BBOX, W, H)
        assert (osm == asm.KW_MAPPED).sum() >= 1600 and sum(cats.values()) >= 1600
        kw = wr.merge_known_water(np.zeros((H, W), np.uint8), osm)
        a, wc = classify(osm == asm.KW_MAPPED, kw)
        assert a["level"] == "none_detected" and wc["by_class"]["mapped"]["n_water_predicted"] >= 1600

    def test_normal_mapped_river_and_drain_lines(self):
        river = {"type": "way", "tags": {"natural": "water", "water": "river"}, "geometry": lonlat(0, 45, 100, 55)}
        w, s, e, n = BBOX
        drain = {"type": "way", "tags": {"waterway": "drain"},
                 "geometry": [{"lon": w + (e - w) * 0.2, "lat": n}, {"lon": w + (e - w) * 0.2, "lat": s}]}
        osm, cats = wr.rasterize_osm([river, drain], BBOX, W, H)
        assert set(cats) == {"river", "drain"}
        assert (osm[:, 18:23] == asm.KW_MAPPED).any(axis=1).all()     # drain burned along its whole length
        water = osm == asm.KW_MAPPED
        a, _ = classify(water, wr.merge_known_water(np.zeros((H, W), np.uint8), osm))
        assert a["level"] == "none_detected"

    def test_river_expanding_onto_dry_land_is_evaluated_as_flood(self):
        kw = np.zeros((H, W), np.uint8)
        kw[:, 45:55] = asm.KW_PERMANENT                       # normal river channel
        water = np.zeros((H, W), bool)
        water[:, 10:90] = True                                # water now spreads far beyond the channel
        a, wc = classify(water, kw)
        assert wc["flood_like_outside_known_water_pct"] == pytest.approx(100 * 7000 / 9000, abs=0.01)
        assert a["level"] == "flood_like"

    def test_flooded_urban_area_with_sufficient_evidence(self):
        water = np.zeros((H, W), bool)
        water[:40] = True                                     # 40 % of normally-dry land under water
        a, _ = classify(water, np.zeros((H, W), np.uint8))
        assert a["level"] == "flood_like" and a["basis"]["is_probability"] is False

    def test_seasonal_wetland_is_context_not_flood_evidence(self):
        season = np.zeros((H, W), np.uint8)
        season[:60] = 5                                       # water 5 months a year
        kw = wr.from_gsw_seasonality(season)
        assert (kw[:60] == asm.KW_SEASONAL).all() and (kw[60:] == asm.KW_NONE).all()
        a, wc = classify(kw == asm.KW_SEASONAL, kw)           # the wetland is wet today
        assert a["level"] == "uncertain_water"                # mostly normally-wet land: no flood claim
        assert wc["by_class"]["seasonal"]["n_water_predicted"] == 6000
        osm, _ = wr.rasterize_osm([{"type": "way", "tags": {"natural": "wetland", "wetland": "tidalflat"},
                                    "geometry": lonlat(0, 0, 20, 20)}], BBOX, W, H)
        assert (osm == asm.KW_SEASONAL).sum() >= 400          # tidal flats are seasonal context too

    def test_worldfloods_seasonal_class_and_milestone5_mode(self):
        jrc = np.array([[0, 1, 2, 3]], np.uint8)
        assert wr.from_worldfloods_jrc(jrc).tolist() == [[0, 0, asm.KW_SEASONAL, asm.KW_PERMANENT]]
        assert wr.from_worldfloods_jrc(jrc, include_seasonal=False).tolist() == [[0, 0, 0, asm.KW_PERMANENT]]

    def test_osm_priority_never_overrides_jrc_permanent_or_sea(self):
        jrc = np.array([[0, 1, 2, 3, 0]], np.uint8)
        osm = np.array([[4, 4, 4, 4, 3]], np.uint8)
        assert wr.merge_known_water(jrc, osm).tolist() == [[4, 1, 2, 4, 3]]

    def test_sea_only_location_is_never_flood(self):
        kw = np.full((H, W), asm.KW_SEA, np.uint8)
        a, _ = classify(np.ones((H, W), bool), kw)
        assert a["level"] == "uncertain_water"


# ---------------- missing / degraded data must be explicit ----------------
class TestMissingData:
    def test_missing_reference_never_claims_flooding(self):
        water = np.zeros((H, W), bool)
        water[:60] = True
        a, wc = classify(water, None)
        assert wc["available"] is False and a["level"] == "inconclusive"
        assert any("could not be set aside" in r for r in a["reasons"])
        assert any("reference unavailable" in lim for lim in a["limitations"])

    def test_insufficient_usable_land(self):
        eligible = np.zeros((H, W), bool)
        eligible[:10] = True
        a, _ = classify(np.zeros((H, W), bool), np.zeros((H, W), np.uint8), eligible=eligible)
        assert a["level"] == "insufficient_coverage"

    def test_haze_policy_preserved(self):
        water = np.zeros((H, W), bool)
        water[:50] = True
        a, _ = classify(water, np.zeros((H, W), np.uint8), hazy=True)
        assert a["level"] == "inconclusive" and a["hazy"] is True

    def test_historical_imagery_unavailable_changes_nothing(self):
        water = np.zeros((H, W), bool)
        water[:5] = True                                      # 5 % -> inconclusive
        a, _ = classify(water, np.zeros((H, W), np.uint8), change=chg.unavailable("No earlier image."))
        assert a["level"] == "inconclusive" and a["historical_comparison_used"] is False

    def test_osm_failure_is_reported_and_not_cached(self, monkeypatch):
        ref = wr.WaterReference(use_osm=True)
        monkeypatch.setattr(ref, "_fetch", lambda b, w, h: np.zeros((h, w), np.uint8))

        def boom(*a, **k):
            raise TimeoutError

        monkeypatch.setattr(wr, "fetch_osm_water", boom)
        mask, info = ref.for_bbox(BBOX, W, H)
        assert mask is not None and info["status"] == "partial" and info["layers"]["osm"] == "unavailable"
        assert ref._cache == {}


# ---------------- year-earlier comparison ----------------
class TestWaterChange:
    def _cmp(self, now, then, kw=None, cur_e=None, base_e=None):
        ones = np.ones((H, W), bool)
        return chg.compare(now, cur_e if cur_e is not None else ones, then, base_e if base_e is not None else ones,
                           kw, "2026-10-18", "2025-10-20")

    def test_unmapped_pond_present_a_year_earlier_is_not_new_water(self):
        pond = np.zeros((H, W), bool)
        pond[40:60, 40:60] = True                             # 4 % unmapped pond, also there last year
        c = self._cmp(pond, pond)
        assert c["status"] == "available" and c["n_new"] == 0 and c["persistent_share_of_candidate_pct"] == 100
        a, _ = classify(pond, np.zeros((H, W), np.uint8), change=c)
        assert a["level"] == "none_detected" and a["historical_comparison_used"] is True

    def test_persistent_water_withdraws_a_flood_claim(self):
        water = np.zeros((H, W), bool)
        water[:40] = True
        a, _ = classify(water, np.zeros((H, W), np.uint8), change=self._cmp(water, water))
        assert a["level"] == "inconclusive" and any("also water" in r for r in a["reasons"])

    def test_new_water_on_dry_land_keeps_the_flood_outcome(self):
        then = np.zeros((H, W), bool)
        then[:5] = True                                       # normal extent
        now = np.zeros((H, W), bool)
        now[:45] = True                                       # spreads over normally-dry land
        c = self._cmp(now, then)
        assert c["new_water_pct_of_dry_land"] == pytest.approx(40.0)
        a, _ = classify(now, np.zeros((H, W), np.uint8), change=c)
        assert a["level"] == "flood_like" and a["historical_comparison_used"] is False

    def test_comparison_never_creates_a_flood(self):
        now = np.zeros((H, W), bool)
        now[:10] = True                                       # 10 % new water -> stays inconclusive
        a, _ = classify(now, np.zeros((H, W), np.uint8), change=self._cmp(now, np.zeros((H, W), bool)))
        assert a["level"] == "inconclusive"

    def test_cloudy_baseline_is_not_used(self):
        base_e = np.zeros((H, W), bool)
        base_e[:30] = True                                    # only 30 % of pixels usable in both images
        c = self._cmp(np.zeros((H, W), bool), np.zeros((H, W), bool), base_e=base_e)
        assert c["status"] == "limited"
        assert chg.apply_to_outcome("inconclusive", c, 1.0) == (None, None)

    def test_different_grid_is_rejected(self):
        c = chg.compare(np.zeros((H, W), bool), np.ones((H, W), bool), np.zeros((50, 50), bool),
                        np.ones((50, 50), bool), None)
        assert c["status"] == "unavailable"

    def test_known_water_excluded_from_comparison(self):
        kw = np.zeros((H, W), np.uint8)
        kw[:50] = asm.KW_PERMANENT
        now = np.zeros((H, W), bool)
        now[:50] = True                                       # lake full today, empty last year: ignored
        c = self._cmp(now, np.zeros((H, W), bool), kw=kw)
        assert c["n_candidate"] == 0 and c["n_compared_dry_land"] == 5000
