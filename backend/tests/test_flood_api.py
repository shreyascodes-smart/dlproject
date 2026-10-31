"""
API tests for the flood segmentation backend.

Uses a tiny synthetic WorldFloods-style dataset written to a temp folder and a
per-pixel mock model, so the tests run on CPU in seconds and never touch the
real dataset, checkpoint, or Copernicus. One optional test runs the real
checkpoint on a real validation sample if those files exist.

Run from the project root:
    .\\venv\\Scripts\\python.exe -m pytest backend/tests
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.api import flood_routes  # noqa: E402
from app.services import seg_service as ss  # noqa: E402
from ml.inference import seg_inference as si  # noqa: E402

DARK, BRIGHT = 100, 5000


class PerPixelMock(nn.Module):
    """Dark pixels -> flood, bright pixels -> non-flood."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f = -5.0 * x.sum(dim=1, keepdim=True)
        return torch.cat([torch.zeros_like(f), f], dim=1)


def _write(path, arr, crs="EPSG:32632"):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = arr if arr.ndim == 3 else arr[None]
    c, h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=c, dtype=arr.dtype,
                       crs=crs, transform=from_origin(500000, 4600000, 10, 10)) as dst:
        dst.write(arr)


def _sample(root, masks, name, split_dir, h=96, w=128, flood_rows=(0, 32), with_meta=True):
    s2 = np.full((15, h, w), BRIGHT, np.int16)
    s2[1:4, flood_rows[0]:flood_rows[1]] = DARK            # B2,B3,B4 dark band -> mock flood
    _write(root / split_dir / "S2" / f"{name}.tif", s2)
    seg = np.ones((h, w), np.uint8)
    seg[flood_rows[0]:flood_rows[1], : w // 2] = 2          # reference flood: left half of dark rows
    validity = np.ones((h, w), np.uint8)
    validity[:, -16:] = 0                                   # right 16 columns invalid
    _write(masks / f"{name}_seg.tif", seg)
    _write(masks / f"{name}_validity.tif", validity)
    pw = np.ones((h, w), np.int16)                          # JRC: 1 = not water
    pw[0:8, 64:112] = 3                                     # permanent water inside the dark (flood-like) band
    _write(root / split_dir / "PERMANENTWATERJRC" / f"{name}.tif", pw)
    if with_meta:
        meta = {"aoi_code": "01TOWN", "event type": "Flood", "s2_date": "2019-02-19T00:52:03+00:00",
                "bounding box": {"west": 10.0, "east": 10.2, "south": 45.0, "north": 45.2}}
        (root / split_dir / "meta").mkdir(parents=True, exist_ok=True)
        (root / split_dir / "meta" / f"{name}.json").write_text(json.dumps(meta))


def _entry(name, osplit, flood_pct=10.0):
    return {"name": name, "original_split": osplit, "event_id": name.split("_")[0],
            "label_status": "positive" if flood_pct else "negative_candidate", "flood_pct": flood_pct}


@pytest.fixture
def env(tmp_path, monkeypatch):
    root, masks = tmp_path / "wf", tmp_path / "masks"
    _sample(root, masks, "EMSR900_01TOWN_DEL_v1", "test")              # val, has meta
    _sample(root, masks, "EMSR901_02CITY_DEL_v1", "train", with_meta=False)  # val, bounds fallback
    _sample(root, masks, "EMSR902_03PLACE_DEL_v1", "val")              # test split
    _sample(root, masks, "EMSR800_01TRAIN_DEL_v1", "train")            # train split, must be hidden
    _write(root / "val" / "S2" / "EMSR903_04BROKEN_DEL_v1.tif", np.ones((15, 32, 32), np.int16))  # no masks
    manifest = {"splits": {
        "train": {"samples": [_entry("EMSR800_01TRAIN_DEL_v1", "train")]},
        "val": {"samples": [_entry("EMSR900_01TOWN_DEL_v1", "test"), _entry("EMSR901_02CITY_DEL_v1", "train", 0.0)]},
        "test": {"samples": [_entry("EMSR902_03PLACE_DEL_v1", "val"), _entry("EMSR903_04BROKEN_DEL_v1", "val")]},
    }}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    seg = si.FloodSegmenter(model=PerPixelMock(), device="cpu")
    svc = ss.SegmentationService(manifest=mpath, data_root=root, masks_dir=masks,
                                 cache_dir=tmp_path / "cache", test_report=None, segmenter=seg)
    monkeypatch.setattr(ss, "_service", svc)

    class NoCreds:
        is_configured = False

    monkeypatch.setattr(flood_routes, "_sentinel_service", NoCreds())
    # Fresh, generous limiters per test: the rate limit has its own test (TestOperations)
    monkeypatch.setattr(flood_routes, "live_limiter", flood_routes.RateLimiter(per_minute=10_000))
    monkeypatch.setattr(flood_routes, "demo_limiter", flood_routes.RateLimiter(per_minute=10_000))
    # Year-earlier comparison has its own tests (test_normal_water.py); keep these single-image
    monkeypatch.setattr(flood_routes.config, "historical_comparison", lambda: False)
    return {"client": TestClient(app), "svc": svc, "tmp": tmp_path}


def _decode(data_url):
    from PIL import Image
    assert data_url.startswith("data:image/jpeg;base64,")
    return np.array(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))))


# ------------------------------------------------------------
class TestStatusAndCatalogue:
    def test_legacy_classifier_not_imported(self, env):
        env["client"].get("/api/flood/model-status")
        assert "ml.inference.flood_inference" not in sys.modules

    def test_model_status_ready(self, env):
        r = env["client"].get("/api/flood/model-status").json()
        assert r["status"] == "ready"
        assert r["satellite"] == "not_configured"
        assert r["model_version"] == "segmentation_v1" and r["model"]["legacy_classifier"] == "not used"
        assert r["model"]["target"] == "ems_flood" and r["model"]["bands"] == ["B04", "B03", "B02"]
        assert r["default_method"] == "tiled"
        assert r["demo_samples"] == 3

    def test_model_status_not_ready_when_checkpoint_missing(self, env, monkeypatch, tmp_path):
        broken = ss.SegmentationService(checkpoint=tmp_path / "missing.pt", cache_dir=None, test_report=None)
        monkeypatch.setattr(ss, "_service", broken)
        r = env["client"].get("/api/flood/model-status").json()
        assert r["status"] == "not_ready" and "not installed" in r["error"] and ":\\" not in r["error"]
        a = env["client"].post("/api/flood/assess", json={"mode": "demo", "sample_id": "x"})
        assert a.status_code == 503

    def test_catalogue_only_val_and_test(self, env):
        r = env["client"].get("/api/flood/demo-samples").json()
        ids = [s["id"] for s in r["samples"]]
        assert "EMSR800_01TRAIN_DEL_v1" not in ids
        assert ids == ["EMSR900_01TOWN_DEL_v1", "EMSR901_02CITY_DEL_v1", "EMSR902_03PLACE_DEL_v1"]
        assert [s["split"] for s in r["samples"]] == ["val", "val", "test"]
        assert "held-out" in r["samples"][2]["split_note"].lower()

    def test_catalogue_excludes_incomplete_samples(self, env):
        r = env["client"].get("/api/flood/demo-samples").json()
        assert [e["id"] for e in r["excluded"]] == ["EMSR903_04BROKEN_DEL_v1"]
        assert "missing" in r["excluded"][0]["reason"]

    def test_catalogue_metadata(self, env):
        s = {x["id"]: x for x in env["client"].get("/api/flood/demo-samples").json()["samples"]}
        a = s["EMSR900_01TOWN_DEL_v1"]
        assert a["center"] == {"lat": 45.1, "lng": 10.1}
        assert a["s2_date"] == "2019-02-19" and a["label"] == "EMSR900 · TOWN"
        b = s["EMSR901_02CITY_DEL_v1"]                      # no meta -> bounds transformed from UTM
        assert b["center"] is not None and -90 <= b["center"]["lat"] <= 90
        assert b["label"] == "EMSR901 · CITY"            # AOI code derived from the sample name
        assert b["width"] == 128 and b["height"] == 96


# ------------------------------------------------------------
class TestRequestValidation:
    @pytest.mark.parametrize("lng", [150.0, -179.9, 180.0, -180.0])
    def test_longitude_beyond_90_is_accepted(self, env, lng):
        r = env["client"].post("/api/flood/assess", json={"mode": "live", "latitude": 20, "longitude": lng})
        assert r.status_code == 503            # passed validation; fails only because no credentials

    @pytest.mark.parametrize("lat,lng", [(91, 0), (-91, 0), (0, 181), (0, -180.5)])
    def test_out_of_range_coordinates_rejected(self, env, lat, lng):
        r = env["client"].post("/api/flood/assess", json={"mode": "live", "latitude": lat, "longitude": lng})
        assert r.status_code == 422

    def test_demo_requires_sample_id(self, env):
        r = env["client"].post("/api/flood/assess", json={"mode": "demo", "latitude": 19, "longitude": 72})
        assert r.status_code == 422

    def test_live_requires_coordinates(self, env):
        assert env["client"].post("/api/flood/assess", json={"mode": "live"}).status_code == 422

    def test_unknown_mode_and_method_rejected(self, env):
        c = env["client"]
        assert c.post("/api/flood/assess", json={"mode": "magic", "sample_id": "a"}).status_code == 422
        assert c.post("/api/flood/assess", json={"mode": "demo", "sample_id": "EMSR900_01TOWN_DEL_v1",
                                                 "method": "fast"}).status_code == 422

    def test_unknown_sample_404(self, env):
        r = env["client"].post("/api/flood/assess", json={"mode": "demo", "sample_id": "nope"})
        assert r.status_code == 404

    def test_training_sample_not_assessable(self, env):
        r = env["client"].post("/api/flood/assess", json={"mode": "demo", "sample_id": "EMSR800_01TRAIN_DEL_v1"})
        assert r.status_code == 404


# ------------------------------------------------------------
class TestDemoAssessment:
    def _assess(self, env, method="tiled", sid="EMSR900_01TOWN_DEL_v1"):
        r = env["client"].post("/api/flood/assess", json={"mode": "demo", "sample_id": sid, "method": method})
        assert r.status_code == 200, r.text
        return r.json()

    def test_response_contract(self, env):
        r = self._assess(env)
        assert r["mode"] == "demo" and r["experimental"] is False and r["method"] == "tiled"
        assert r["sample"]["split"] == "val"
        for k in ("rgb", "overlay", "reference", "error"):
            assert r["images"][k].startswith("data:image/jpeg;base64,")
        assert r["reference"]["available"] is True
        assert "not a model output" in r["reference"]["source"]
        assert "not an official warning" in r["disclaimer"].lower()
        assert any("km" in l for l in r["limitations"])
        assert "not a calibrated probability or confidence" in r["prediction"]["threshold_note"]
        assert r["prediction"]["target"] == "ems_flood" and r["model"]["version"] == "segmentation_v1"
        assert "flood_probability" not in r["prediction"]       # no confidence-style field

    def test_coverage_and_metrics(self, env):
        r = self._assess(env)
        p, m = r["prediction"], r["reference"]["metrics"]
        # eligible = 96 x (128-16); predicted flood = 32 dark rows x 112 valid columns
        assert p["n_eligible_pixels"] == 96 * 112
        assert p["n_flood_pixels"] == 32 * 112
        assert p["flood_coverage_pct"] == pytest.approx(100 * 32 / 96, abs=1e-3)
        assert p["eligibility_source"] == "validity_mask+nodata"
        # reference flood = 32 rows x 64 cols; prediction also flags 32 x 48 extra valid cols
        assert (m["tp"], m["fn"], m["fp"]) == (32 * 64, 0, 24 * 48)   # permanent-water rows set aside
        assert m["flood_iou"] == pytest.approx(2048 / (2048 + 1152), abs=1e-4)

    def test_images_aligned_and_same_size(self, env):
        r = self._assess(env)
        imgs = [_decode(r["images"][k]) for k in ("rgb", "overlay", "reference", "error")]
        assert all(i.shape == (96, 128, 3) for i in imgs)
        assert r["images"]["display_width"] == 128 and r["images"]["display_height"] == 96
        rgb, ov = imgs[0].astype(int), imgs[1].astype(int)
        # overlay differs from the RGB inside the predicted flood band, not in the bright area
        assert np.abs(ov[4:28, 4:100] - rgb[4:28, 4:100]).mean() > 30
        assert np.abs(ov[50:90, 4:100] - rgb[50:90, 4:100]).mean() < 3

    def test_cache_memory_and_disk(self, env):
        first = self._assess(env)
        second = self._assess(env)
        assert first["cached"] is False and second["cached"] is True
        assert first["prediction"] == second["prediction"]
        assert second["limitations"] == env["svc"].limitations() and second["disclaimer"] == ss.DISCLAIMER
        assert len(list((env["tmp"] / "cache").glob("*.json"))) == 1
        env["svc"]._mem_cache.clear()
        assert self._assess(env)["cached"] is True              # served from disk
        assert self._assess(env, method="resize")["cached"] is False  # separate key per method

    def test_assessment_and_permanent_water(self, env):
        r = self._assess(env)
        wc = r["water_context"]
        # 8 rows x 48 valid cols of permanent water inside the predicted band are set aside
        assert wc["available"] is True
        assert wc["n_flood_like_on_known_water"] == 8 * 48
        assert wc["n_flood_like_outside_known_water"] == 32 * 112 - 8 * 48
        assert wc["n_eligible_outside_known_water"] == 96 * 112 - 8 * 48
        assert r["prediction"]["n_flood_pixels"] == 32 * 112      # raw prediction unchanged
        a = r["assessment"]
        assert a["level"] in ("inconclusive", "flood_like")
        assert a["basis"]["kind"] == "descriptive_model_outcome" and a["basis"]["is_risk_rating"] is False
        assert "safe" not in a["headline"].lower()
        assert r["freshness"]["archived"] is True and r["freshness"]["acquisition_date"] == "2019-02-19"

    def test_map_layers_reprojected_to_latlng(self, env):
        m = self._assess(env)["map"]
        (s, w), (n, e) = m["bounds"]
        assert m["crs"] == "EPSG:4326" and -90 < s < n < 90 and -180 < w < e < 180
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(m["flood"].split(",", 1)[1])))
        assert img.mode == "RGBA" and img.size == (m["width"], m["height"])
        alpha = np.array(img)[..., 3]
        assert 0 < (alpha > 0).mean() < 1                         # some flood, some transparent

    def test_test_split_sample_labelled(self, env):
        r = self._assess(env, sid="EMSR902_03PLACE_DEL_v1")
        assert r["sample"]["split"] == "test"
        assert "illustration only" in r["reference"]["split_note"]


# ------------------------------------------------------------
LIVE_BBOX = [72.4, 19.2, 72.45, 19.25]


def _live_service(img, validity=None, **meta):
    class Live:
        is_configured = True

        def __init__(self):
            self.calls = []

        def fetch_scene(self, **kw):
            self.calls.append(kw)
            m = {"source": "Copernicus Sentinel-2 Level-1C", "resolution_m": 10, "bbox": LIVE_BBOX,
                 "acquisition_date": "2026-10-20", "acquisition_datetime": "2026-10-20T05:32:11Z",
                 "data_coverage_pct": 100.0, "quality_status": "good", "cloud_mask_source": "sen2cor_scl",
                 "tile_cloud_cover_pct": 4.2, "search_days": 45}
            m.update(meta)
            return img, validity, m

    return Live()


class FakeWater:
    def __init__(self, mask=None, fail=False):
        self.mask, self.fail = mask, fail

    def for_bbox(self, bbox, width, height):
        if self.fail:
            return None, {"source": "JRC", "status": "unavailable"}
        m = self.mask if self.mask is not None else np.zeros((height, width), np.uint8)
        return m, {"source": "JRC GSW (test)", "status": "ok"}


class TestLiveAssessment:
    def _post(self, env):
        return env["client"].post("/api/flood/assess", json={"mode": "live", "latitude": 19.22, "longitude": 72.42})

    def test_live_contract(self, env, monkeypatch):
        h, w = 80, 80
        img = np.full((6, h, w), BRIGHT, np.uint16)
        img[:3, :20] = DARK                                    # B02,B03,B04 dark in top 20 rows
        img[3:, :20] = 9000                                    # NIR/SWIR bright: must be ignored by RGB read
        validity = np.ones((h, w), np.uint8)
        live = _live_service(img, validity)
        monkeypatch.setattr(flood_routes, "_sentinel_service", live)
        monkeypatch.setattr(flood_routes, "get_water_reference", lambda: FakeWater())
        r = self._post(env)
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["experimental"] is True and j["reference"]["available"] is False
        assert j["images"]["reference"] is None and j["images"]["error"] is None
        assert j["prediction"]["flood_coverage_pct"] == pytest.approx(25.0)
        assert j["prediction"]["eligibility_source"] == "validity_mask+nodata"   # cloud mask used
        assert j["imagery"]["acquisition_date"] == "2026-10-20"
        assert j["freshness"]["archived"] is False and isinstance(j["freshness"]["age_days"], int)
        assert j["imagery"]["bbox"] == dict(zip(("west", "south", "east", "north"), LIVE_BBOX))
        a = j["assessment"]
        assert a["basis"]["is_risk_rating"] is False and a["level"] in {"inconclusive", "flood_like"}
        text = json.dumps({k: a[k] for k in ("headline", "summary", "reasons", "limitations")}).lower()
        assert "probab" not in text and "risk" not in text and "safe" not in text
        assert a["basis"]["is_probability"] is False
        m = j["map"]
        assert m["bounds"] == [[19.2, 72.4], [19.25, 72.45]]
        assert m["flood"].startswith("data:image/webp;base64,") and m["satellite"].startswith("data:image/webp")
        assert live.calls[0]["days_back"] == flood_routes.LIVE_SEARCH_DAYS

    def test_sea_regression_uncertain_not_flood(self, env, monkeypatch):
        """The Mumbai false positive: open sea looks like water everywhere. Must not read as flooding."""
        h, w = 60, 60
        img = np.full((6, h, w), DARK, np.uint16)              # dark water everywhere -> model: 100% flood-like
        sea = np.full((h, w), 2, np.uint8)                     # known-water reference: all open sea
        sea[:, :6] = 0                                         # a thin strip of land
        monkeypatch.setattr(flood_routes, "_sentinel_service", _live_service(img, np.ones((h, w), np.uint8)))
        monkeypatch.setattr(flood_routes, "get_water_reference", lambda: FakeWater(sea))
        j = self._post(env).json()
        assert j["prediction"]["flood_coverage_pct"] == pytest.approx(100.0)  # raw output preserved
        assert j["water_context"]["sea_pct_of_eligible"] == pytest.approx(90.0)
        assert j["water_context"]["flood_like_on_known_water_share_pct"] == pytest.approx(90.0)
        a = j["assessment"]
        assert a["level"] == "uncertain_water"
        assert a["headline"] == "Inconclusive — mostly permanent water"
        assert "flood" not in a["headline"].lower().replace("misclassified", "")
        assert j["map"]["set_aside"] is not None               # set-aside pixels shown separately

    def test_reference_unavailable_is_disclosed(self, env, monkeypatch):
        h, w = 40, 40
        img = np.full((6, h, w), DARK, np.uint16)
        monkeypatch.setattr(flood_routes, "_sentinel_service", _live_service(img, np.ones((h, w), np.uint8)))
        monkeypatch.setattr(flood_routes, "get_water_reference", lambda: FakeWater(fail=True))
        j = self._post(env).json()
        assert j["water_context"]["available"] is False
        assert any("NOT set aside" in w for w in j["quality"]["warnings"])
        assert any("Permanent-water reference unavailable" in u for u in j["assessment"]["limitations"])

    def test_no_cloud_mask_is_disclosed(self, env, monkeypatch):
        img = np.full((6, 40, 40), BRIGHT, np.uint16)
        monkeypatch.setattr(flood_routes, "_sentinel_service", _live_service(img, None, cloud_mask_source="unavailable"))
        monkeypatch.setattr(flood_routes, "get_water_reference", lambda: FakeWater())
        j = self._post(env).json()
        assert j["prediction"]["eligibility_source"] == "nodata_only"
        assert any("No cloud mask" in w for w in j["quality"]["warnings"])

    def test_cloudy_image_unable_to_assess(self, env, monkeypatch):
        h, w = 40, 40
        validity = np.zeros((h, w), np.uint8)
        validity[:4] = 1                                       # 10% usable
        monkeypatch.setattr(flood_routes, "_sentinel_service",
                            _live_service(np.full((6, h, w), BRIGHT, np.uint16), validity))
        monkeypatch.setattr(flood_routes, "get_water_reference", lambda: FakeWater())
        a = self._post(env).json()["assessment"]
        assert a["level"] == "insufficient_coverage" and a["reasons"]

    def test_live_empty_image_rejected(self, env, monkeypatch):
        monkeypatch.setattr(flood_routes, "_sentinel_service",
                            _live_service(np.zeros((6, 8, 8), np.uint16), None, quality_status="empty",
                                          data_coverage_pct=0))
        assert self._post(env).status_code == 422

    def test_no_acquisition_is_404(self, env, monkeypatch):
        class NoImage:
            is_configured = True

            def fetch_scene(self, **kw):
                raise LookupError("No Sentinel-2 acquisition found for this area in the last 45 days.")

        monkeypatch.setattr(flood_routes, "_sentinel_service", NoImage())
        r = self._post(env)
        assert r.status_code == 404 and "45 days" in r.json()["detail"]

    def test_live_fetch_error_is_502_without_leaking_details(self, env, monkeypatch):
        class Down:
            is_configured = True

            def fetch_scene(self, **kw):
                raise RuntimeError("POST https://identity/token client_secret=SHOULD_NOT_LEAK")

        monkeypatch.setattr(flood_routes, "_sentinel_service", Down())
        r = self._post(env)
        assert r.status_code == 502
        assert "SHOULD_NOT_LEAK" not in r.text and "client_secret" not in r.text


# ------------------------------------------------------------
from app import config as app_config  # noqa: E402

REAL = (app_config.seg_checkpoint("segmentation_v1").is_file() and ss.DEFAULTS["manifest"].is_file()
        and ss.DEFAULTS["data_root"].is_dir())


@pytest.mark.skipif(not REAL, reason="real checkpoint/dataset not present")
def test_real_model_on_small_validation_sample(monkeypatch, tmp_path):
    """End-to-end with the trained checkpoint on the smallest validation scene (266 x 218 px)."""
    svc = ss.SegmentationService(cache_dir=tmp_path / "cache", model_version="segmentation_v1")
    monkeypatch.setattr(ss, "_service", svc)
    c = TestClient(app)
    st = c.get("/api/flood/model-status").json()
    assert st["status"] == "ready" and st["model"]["encoder"] == "efficientnet-b2" and st["model"]["epoch"] == 14
    heldout = ss.asm.REPORTS / "evaluation_segmentation_v1_test.json"
    if heldout.is_file():   # product-protocol held-out evaluation (Milestone 5)
        expected = json.loads(heldout.read_text())["summary"]["tiled"]["pixel_metrics_excluding_known_water"]
        assert st["model"]["reported_test_metrics"]["flood_iou"] == pytest.approx(expected["flood_iou"], abs=1e-4)
        assert "full resolution" in st["model"]["reported_test_metrics"]["protocol"]
    else:                   # Phase H report (whole scene resized)
        assert st["model"]["reported_test_metrics"]["flood_iou"] == pytest.approx(0.0223, abs=1e-4)
    samples = c.get("/api/flood/demo-samples").json()["samples"]
    assert {s["split"] for s in samples} <= {"val", "test"}
    r = c.post("/api/flood/assess", json={"mode": "demo", "sample_id": "EMSR312_02CALAYAN_GRA_v1"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"]["architecture"] == "U-Net"
    assert 0 <= j["prediction"]["flood_coverage_pct"] <= 100
    assert _decode(j["images"]["overlay"]).shape == (218, 266, 3)


# ------------------------------------------------------------
# Milestone 5: model selection, multispectral spec, readiness, limits, redaction
# ------------------------------------------------------------
class SixBandMock(nn.Module):
    """6-band 'water' mock: positive where SWIR (B11, channel 4) is dark."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        assert x.shape[1] == 6
        w = -5.0 * x[:, 4:5]
        return torch.cat([torch.zeros_like(w), w], dim=1)


V2_TEST_SPEC = si.ModelSpec("segmentation_v2", ("B02", "B03", "B04", "B08", "B11", "B12"), "surface_water",
                            {"kind": "standardize", "scale": 10000.0, "clip": [0, 1],
                             "mean": [0.1] * 6, "std": [0.05] * 6})


class TestModelSelection:
    def test_legacy_classifier_cannot_be_selected(self, monkeypatch):
        monkeypatch.setenv("DI_SEG_MODEL", "best_flood_model")
        with pytest.raises(ValueError):
            app_config.seg_model_version()
        with pytest.raises(ValueError, match="legacy"):
            ss.SegmentationService(checkpoint=ROOT / "models" / "flood" / "best_flood_model.pt", cache_dir=None)

    def test_configured_model_is_used_without_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DI_SEG_MODEL", "segmentation_v2")
        monkeypatch.setattr(app_config, "MODELS_DIR", tmp_path)          # v2 checkpoint absent here
        svc = ss.SegmentationService(cache_dir=None, test_report=None)
        assert svc.model_version == "segmentation_v2"
        assert svc.checkpoint == tmp_path / "segmentation_v2" / "best_seg_model.pt"
        st = svc.status()
        assert st["status"] == "not_ready" and "not installed" in st["error"]
        assert str(tmp_path) not in st["error"]                            # no local paths

    def test_v1_checkpoint_metadata_names_version(self):
        if not REAL:
            pytest.skip("real checkpoint not present")
        seg = si.FloodSegmenter(app_config.seg_checkpoint("segmentation_v1"), device="cpu")
        assert seg.spec.version == "segmentation_v1" and seg.spec.bands == ("B04", "B03", "B02")


class TestMultispectralSpec:
    def test_live_bands_mapped_to_model_order(self):
        img = np.arange(6)[:, None, None] * np.ones((6, 2, 2))
        out = si.select_live_bands(img, V2_TEST_SPEC)
        assert out[0, 0].tolist() == [0, 1, 2, 3, 4, 5]
        rgb = si.select_live_bands(img, si.V1_SPEC)
        assert rgb[0, 0].tolist() == [2, 1, 0]                             # B04, B03, B02

    def test_worldfloods_band_indices(self, tmp_path):
        bands = np.zeros((15, 4, 4), np.int16)
        for b in range(15):
            bands[b] = (b + 1) * 100
        _write(tmp_path / "s.tif", bands)
        x = si.read_scene(tmp_path / "s.tif", V2_TEST_SPEC)
        assert x[0, 0].tolist() == [200, 300, 400, 800, 1200, 1300]       # bands 2,3,4,8,12,13

    def test_standardize_normalisation(self):
        raw = np.full((1, 1, 6), 1500.0, np.float32)
        n = V2_TEST_SPEC.normalize(raw)
        assert n[0, 0].tolist() == pytest.approx([1.0] * 6)                # (0.15 - 0.1) / 0.05

    def test_v2_live_assessment_end_to_end(self, env, monkeypatch):
        seg = si.FloodSegmenter(model=SixBandMock(), device="cpu", spec=V2_TEST_SPEC)
        svc = ss.SegmentationService(manifest=env["svc"].manifest, data_root=env["svc"].data_root,
                                     masks_dir=env["svc"].masks_dir, cache_dir=None, test_report=None, segmenter=seg)
        monkeypatch.setattr(ss, "_service", svc)
        h, w = 40, 40
        img = np.full((6, h, w), 3000, np.uint16)
        img[4, :10] = 100                                                  # dark SWIR -> water in 10 rows
        monkeypatch.setattr(flood_routes, "_sentinel_service", _live_service(img, np.ones((h, w), np.uint8)))
        monkeypatch.setattr(flood_routes, "get_water_reference", lambda: FakeWater())
        j = env["client"].post("/api/flood/assess", json={"mode": "live", "latitude": 19.2, "longitude": 72.4}).json()
        assert j["model"]["version"] == "segmentation_v2" and j["prediction"]["target"] == "surface_water"
        assert j["prediction"]["raw_coverage_pct"] == pytest.approx(25.0)
        assert j["assessment"]["level"] == "inconclusive"                  # no v2 policy in this env


class TestOperations:
    def test_ready_endpoint(self, env):
        r = env["client"].get("/ready")
        j = r.json()
        assert set(j["checks"]) >= {"api", "segmentation_checkpoint", "model", "satellite_credentials"}
        assert j["checks"]["satellite_credentials"] == "not_configured"
        assert "weather_service" not in j["checks"]                        # external probes only with ?deep=true

    def test_rate_limit_live(self, env, monkeypatch):
        monkeypatch.setattr(flood_routes, "live_limiter", flood_routes.RateLimiter(per_minute=2))
        codes = [env["client"].post("/api/flood/assess", json={"mode": "live", "latitude": 1, "longitude": 1}).status_code
                 for _ in range(3)]
        assert codes[:2] == [503, 503] and codes[2] == 429                 # 503 = no credentials in test env

    def test_body_size_limit(self, env):
        r = env["client"].post("/api/flood/assess", content=b"{" + b" " * 20000 + b"}",
                               headers={"Content-Type": "application/json"})
        assert r.status_code == 413

    def test_invalid_sample_id_rejected(self, env):
        r = env["client"].post("/api/flood/assess", json={"mode": "demo", "sample_id": "../../etc/passwd"})
        assert r.status_code == 422

    def test_research_endpoint_has_no_local_paths(self, env):
        j = env["client"].get("/api/flood/research").json()
        assert j["active_model"] == "segmentation_v1"
        text = json.dumps(j)
        assert ":\\\\" not in text and str(ROOT) not in text
        assert {m["version"] for m in j["models"]} == {"segmentation_v1", "segmentation_v2"}
