"""
Sentinel-2 Satellite Data Service
Fetches live satellite imagery from Copernicus Data Space Ecosystem (CDSE)
using the Sentinel Hub Process API.

Registration: https://dataspace.copernicus.eu/
Dashboard (create OAuth client): https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings
"""

import os
import io
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

import requests
import rasterio
from rasterio.io import MemoryFile

logger = logging.getLogger(__name__)

# ─── CDSE Endpoints ──────────────────────────────────────
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

# ─── Evalscript: Returns 6-band GeoTIFF matching WorldFloods format ─────
# Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
# NOTE: We do NOT mask clouds in the evalscript because zeroed pixels look like
#       water to the model. Instead we rely on mosaickingOrder: "leastCC" to
#       automatically pick the least-cloudy acquisition.
EVALSCRIPT_6BAND = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "B11", "B12"],
      units: "DN"
    }],
    output: {
      bands: 6,
      sampleType: "UINT16"
    }
  };
}

function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11, sample.B12];
}
"""

# ─── Evalscript: True Color RGB for visualization ────────
EVALSCRIPT_RGB = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B04", "B03", "B02"],
      units: "DN"
    }],
    output: {
      bands: 3,
      sampleType: "UINT16"
    }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02];
}
"""


# ─── Evalscript: single-acquisition L1C bands + L2A scene classification (cloud mask) ───
# Model input comes from Level-1C (top-of-atmosphere), the same product level as the
# WorldFloods training images. SCL from Level-2A is used ONLY to mask clouds/shadows.
# Output bands: B02, B03, B04, B08, B11, B12 (DN, harmonized), dataMask, SCL (0 = not available)
EVALSCRIPT_L1C_SCL = """
//VERSION=3
function setup() {
  return {
    input: [
      { datasource: "l1c", bands: ["B02", "B03", "B04", "B08", "B11", "B12", "dataMask"], units: "DN" },
      { datasource: "l2a", bands: ["SCL"], units: "DN" }
    ],
    output: { bands: 8, sampleType: "UINT16" }
  };
}
function evaluatePixel(samples) {
  var a = samples.l1c[0];
  var b = (samples.l2a && samples.l2a.length) ? samples.l2a[0] : null;
  return [a.B02, a.B03, a.B04, a.B08, a.B11, a.B12, a.dataMask, b ? b.SCL : 0];
}
"""

EVALSCRIPT_L1C_ONLY = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08", "B11", "B12", "dataMask"], units: "DN" }],
    output: { bands: 8, sampleType: "UINT16" }
  };
}
function evaluatePixel(s) {
  return [s.B02, s.B03, s.B04, s.B08, s.B11, s.B12, s.dataMask, 0];
}
"""

# Sen2Cor scene classes treated as unusable for flood mapping
SCL_INVALID = (0, 1, 3, 8, 9, 10)  # no data, saturated, cloud shadow, cloud medium, cloud high, cirrus


def _api_error_text(resp) -> str:
    """Error message from a Sentinel Hub JSON error body (never includes request headers/credentials)."""
    try:
        err = resp.json().get("error", {})
        return str(err.get("message") or err.get("reason") or err)[:300]
    except Exception:
        return "(non-JSON error body)"


def pick_acquisition(features: list, max_cloud: float = 30.0) -> Optional[dict]:
    """Most recent acquisition with tile cloud cover <= max_cloud; otherwise the least cloudy one."""
    items = []
    for f in features or []:
        p = f.get("properties", {})
        dt = p.get("datetime")
        if not dt:
            continue
        items.append({"datetime": dt, "cloud_cover": p.get("eo:cloud_cover"), "id": f.get("id", "")})
    if not items:
        return None
    clear = [i for i in items if i["cloud_cover"] is not None and i["cloud_cover"] <= max_cloud]
    if clear:
        return max(clear, key=lambda i: i["datetime"])
    return min(items, key=lambda i: (i["cloud_cover"] if i["cloud_cover"] is not None else 101, i["datetime"]))


def validity_from_scl(data_mask: np.ndarray, scl: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """Validity (1 usable, 0 not) from dataMask + SCL. Returns (None, reason) when SCL is unavailable."""
    has = data_mask > 0
    if not np.any(scl[has] > 0):
        return None, "unavailable"
    valid = has & ~np.isin(scl, SCL_INVALID)
    return valid.astype(np.uint8), "sen2cor_scl"


class SentinelDataService:
    """Fetches Sentinel-2 data from Copernicus Data Space."""

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.client_id = client_id or os.getenv("CDSE_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("CDSE_CLIENT_SECRET", "")
        self._token = None
        self._token_expiry = 0
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _get_token(self) -> str:
        """Get OAuth2 access token (cached until expiry)."""
        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        if not self.is_configured:
            raise RuntimeError(
                "Copernicus credentials not configured. "
                "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET environment variables. "
                "Register at https://dataspace.copernicus.eu/ and create OAuth client at "
                "https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings"
            )

        resp = self._session.post(TOKEN_URL, data={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }, timeout=30)

        if resp.status_code != 200:
            # Do not log the response body: identity-server errors can echo request parameters.
            logger.error("Token request failed: HTTP %s", resp.status_code)
            raise RuntimeError(f"Failed to authenticate with Copernicus: {resp.status_code}")

        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 600)
        logger.info("Copernicus OAuth token acquired")
        return self._token

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}"}

    def check_availability(self, lat: float, lng: float, days_back: int = 30) -> dict:
        """Check if recent Sentinel-2 imagery is available for a location."""
        bbox = self._make_bbox(lat, lng, size_km=5)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        payload = {
            "bbox": bbox,
            "datetime": f"{start_date.strftime('%Y-%m-%dT00:00:00Z')}/{end_date.strftime('%Y-%m-%dT23:59:59Z')}",
            "collections": ["sentinel-2-l2a"],
            "limit": 5,
            "filter": "eo:cloud_cover < 80",
            "filter-lang": "cql2-text"
        }

        try:
            resp = self._session.post(
                CATALOG_URL,
                json=payload,
                headers=self._auth_headers(),
                timeout=30
            )
            if resp.status_code == 200:
                features = resp.json().get("features", [])
                if features:
                    latest = features[0]
                    return {
                        "available": True,
                        "date": latest["properties"]["datetime"][:10],
                        "cloud_cover": latest["properties"].get("eo:cloud_cover", "N/A"),
                        "tile_id": latest.get("id", ""),
                        "count": len(features),
                    }
            return {"available": False, "reason": "No clear imagery found in the last 30 days"}
        except Exception as e:
            logger.error(f"Catalog search failed: {e}")
            return {"available": False, "reason": str(e)}

    def fetch_image(
        self,
        lat: float,
        lng: float,
        size_km: float = 5.0,
        resolution: int = 10,
        days_back: int = 60,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Fetch Sentinel-2 image for given coordinates.

        Returns:
            tuple: (image_array [C, H, W], metadata dict)
        """
        bbox = self._make_bbox(lat, lng, size_km)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        # Calculate pixel dimensions
        bbox_width_deg = bbox[2] - bbox[0]
        bbox_height_deg = bbox[3] - bbox[1]
        # Approximate degrees to meters (at given latitude)
        meters_per_deg_lng = 111320 * np.cos(np.radians(lat))
        meters_per_deg_lat = 110540
        width_px = int((bbox_width_deg * meters_per_deg_lng) / resolution)
        height_px = int((bbox_height_deg * meters_per_deg_lat) / resolution)

        # Clamp to reasonable size
        width_px = min(max(width_px, 64), 2500)
        height_px = min(max(height_px, 64), 2500)

        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                            "to": end_date.strftime("%Y-%m-%dT23:59:59Z"),
                        },
                        "maxCloudCoverage": 80,
                        "mosaickingOrder": "leastCC",
                    },
                }],
            },
            "output": {
                "width": width_px,
                "height": height_px,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": EVALSCRIPT_6BAND,
        }

        logger.info(f"Fetching Sentinel-2 image: lat={lat}, lng={lng}, "
                     f"bbox={bbox}, size={width_px}x{height_px}px")

        resp = self._session.post(
            PROCESS_URL,
            json=payload,
            headers={
                **self._auth_headers(),
                "Accept": "image/tiff",
            },
            timeout=120,
        )

        if resp.status_code != 200:
            error_detail = resp.text[:500] if resp.text else "Unknown error"
            logger.error(f"Process API error {resp.status_code}: {error_detail}")
            raise RuntimeError(f"Sentinel Hub Process API error: {resp.status_code} - {error_detail}")

        # Parse GeoTIFF from response
        tiff_bytes = resp.content
        logger.info(f"Received {len(tiff_bytes)} bytes of GeoTIFF data")

        with MemoryFile(tiff_bytes) as memfile:
            with memfile.open() as dataset:
                image = dataset.read()  # Shape: (bands, height, width)

        # Check for nodata pixels — a pixel is nodata only if ALL bands
        # are zero simultaneously.  Individual zero values in single bands
        # (e.g. deep water in SWIR) are legitimate and must not be flagged.
        # NOTE: This is NOT a cloud detector.  The leastCC mosaicking
        # strategy already selects the least-cloudy tile.  We cannot
        # determine cloud percentage from the pixel values alone.
        num_pixels = image.shape[1] * image.shape[2]  # height * width
        all_bands_zero = np.all(image == 0, axis=0)  # True where every band is 0
        nodata_count = int(all_bands_zero.sum())
        data_coverage_pct = round((1 - nodata_count / num_pixels) * 100, 1) if num_pixels > 0 else 0.0

        if data_coverage_pct < 10:
            logger.warning(f"Image has only {data_coverage_pct}% data coverage (rest is nodata)")
            quality_status = "empty"
        elif data_coverage_pct < 50:
            logger.warning(f"Image has {data_coverage_pct}% data coverage — partial")
            quality_status = "partial"
        else:
            quality_status = "good"
        # Cloud status is unknown — we do not have a cloud mask for live data.

        metadata = {
            "source": "Copernicus Sentinel-2 L2A",
            "location": {"lat": lat, "lng": lng},
            "bbox": bbox,
            "resolution_m": resolution,
            "size_px": [width_px, height_px],
            "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "bands": ["B02", "B03", "B04", "B08", "B11", "B12"],
            "data_coverage_pct": data_coverage_pct,
            "quality_status": quality_status,
            "cloud_status": "unknown",
            "fetched_at": datetime.utcnow().isoformat(),
        }

        # Optionally save to disk
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tiff_bytes)
            metadata["saved_to"] = output_path
            logger.info(f"Saved GeoTIFF to {output_path}")

        return image, metadata

    def search_acquisitions(self, bbox: list, days_back: int, collection: str = "sentinel-2-l1c",
                            end: Optional[datetime] = None) -> list:
        """Catalog search: Sentinel-2 acquisitions intersecting bbox in the `days_back` days before `end` (default now)."""
        end = end or datetime.utcnow()
        start = end - timedelta(days=days_back)
        payload = {
            "bbox": bbox,
            "datetime": f"{start.strftime('%Y-%m-%dT00:00:00Z')}/{end.strftime('%Y-%m-%dT23:59:59Z')}",
            "collections": [collection],
            "limit": 100,
        }
        resp = self._session.post(
            CATALOG_URL, json=payload, timeout=30,
            # STAC search responses are GeoJSON; a plain application/json Accept header returns 406
            headers={**self._auth_headers(), "Accept": "application/geo+json, application/json"},
        )
        if resp.status_code != 200:
            logger.error("Catalog API error %s: %s", resp.status_code, _api_error_text(resp))
            raise RuntimeError(f"Sentinel Hub catalog error {resp.status_code}")
        return resp.json().get("features", [])

    def fetch_scene(
        self,
        lat: float,
        lng: float,
        size_km: float = 5.0,
        resolution: int = 10,
        days_back: int = 45,
        max_cloud: float = 30.0,
        end: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        """Fetch ONE Sentinel-2 acquisition (known date) around a point.

        Picks the most recent Level-1C acquisition with tile cloud cover <= max_cloud
        (else the least cloudy) from the `days_back` days before `end` (default: now), then
        downloads that day only. The grid depends only on (lat, lng, size_km, resolution), so two
        calls with different `end` dates return pixel-aligned images.

        Returns (image [6, H, W] uint16 as B02,B03,B04,B08,B11,B12; validity HxW or None; metadata).
        Raises LookupError when no acquisition exists in the period.
        """
        bbox = self._make_bbox(lat, lng, size_km)
        features = self.search_acquisitions(bbox, days_back, end=end)
        acq = pick_acquisition(features, max_cloud)
        if acq is None:
            raise LookupError(f"No Sentinel-2 acquisition found for this area in the last {days_back} days.")

        day = acq["datetime"][:10]
        meters_per_deg_lng = 111320 * np.cos(np.radians(lat))
        width_px = min(max(int((bbox[2] - bbox[0]) * meters_per_deg_lng / resolution), 64), 2500)
        height_px = min(max(int((bbox[3] - bbox[1]) * 110540 / resolution), 64), 2500)
        time_range = {"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"}
        bounds = {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}
        output = {"width": width_px, "height": height_px,
                  "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]}

        def process(data, evalscript):
            r = self._session.post(
                PROCESS_URL,
                json={"input": {"bounds": bounds, "data": data}, "output": output, "evalscript": evalscript},
                headers={**self._auth_headers(), "Accept": "image/tiff"},
                timeout=120,
            )
            if r.status_code != 200:
                logger.error("Process API error %s: %s", r.status_code, _api_error_text(r))
                raise RuntimeError(f"Sentinel Hub Process API error {r.status_code}")
            with MemoryFile(r.content) as mem, mem.open() as ds:
                return ds.read()

        l1c = {"type": "sentinel-2-l1c", "id": "l1c",
               "dataFilter": {"timeRange": time_range, "mosaickingOrder": "mostRecent"}}
        l2a = {"type": "sentinel-2-l2a", "id": "l2a",
               "dataFilter": {"timeRange": time_range, "mosaickingOrder": "mostRecent"}}
        cloud_source = "sen2cor_scl"
        try:
            arr = process([l1c, l2a], EVALSCRIPT_L1C_SCL)
        except RuntimeError as e:
            logger.warning("L1C+SCL request failed (%s); retrying without cloud mask", e)
            single = {k: v for k, v in l1c.items() if k != "id"}
            arr = process([single], EVALSCRIPT_L1C_ONLY)
            cloud_source = "unavailable"

        image = arr[:6]
        data_mask = arr[6]
        validity, how = validity_from_scl(data_mask, arr[7]) if cloud_source != "unavailable" else (None, "unavailable")
        cloud_source = how
        n = data_mask.size
        coverage = round(100.0 * float((data_mask > 0).sum()) / n, 1) if n else 0.0
        cloud_pct = None
        if validity is not None and (data_mask > 0).any():
            cloud_pct = round(100.0 * (1 - validity.sum() / float((data_mask > 0).sum())), 1)

        metadata = {
            "source": "Copernicus Sentinel-2 Level-1C",
            "processing_level": "L1C",
            "location": {"lat": lat, "lng": lng},
            "bbox": bbox,
            "resolution_m": resolution,
            "size_px": [width_px, height_px],
            "acquisition_datetime": acq["datetime"],
            "acquisition_date": day,
            "tile_cloud_cover_pct": acq["cloud_cover"],
            "acquisitions_considered": len(features),
            "search_days": days_back,
            "data_coverage_pct": coverage,
            "quality_status": "empty" if coverage < 10 else "partial" if coverage < 50 else "good",
            "cloud_mask_source": cloud_source,
            "cloudy_pct_of_data": cloud_pct,
            "fetched_at": datetime.utcnow().isoformat(),
        }
        return image, validity, metadata

    def fetch_rgb_preview(
        self,
        lat: float,
        lng: float,
        size_km: float = 5.0,
        width_px: int = 512,
        height_px: int = 512,
        days_back: int = 60,
    ) -> bytes:
        """Fetch a true-color RGB PNG preview."""
        bbox = self._make_bbox(lat, lng, size_km)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                            "to": end_date.strftime("%Y-%m-%dT23:59:59Z"),
                        },
                        "maxCloudCoverage": 30,
                        "mosaickingOrder": "leastCC",
                    },
                }],
            },
            "output": {
                "width": width_px,
                "height": height_px,
                "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
            },
            "evalscript": EVALSCRIPT_RGB,
        }

        resp = self._session.post(
            PROCESS_URL,
            json=payload,
            headers={**self._auth_headers(), "Accept": "image/png"},
            timeout=60,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"RGB preview fetch failed: {resp.status_code}")

        return resp.content

    @staticmethod
    def _make_bbox(lat: float, lng: float, size_km: float = 5.0) -> list:
        """Create a bounding box [west, south, east, north] around lat/lng."""
        # 1 degree latitude ~ 110.54 km
        # 1 degree longitude ~ 111.32 * cos(lat) km
        delta_lat = size_km / 110.54 / 2
        delta_lng = size_km / (111.32 * np.cos(np.radians(lat))) / 2
        return [
            round(lng - delta_lng, 6),
            round(lat - delta_lat, 6),
            round(lng + delta_lng, 6),
            round(lat + delta_lat, 6),
        ]


# Singleton instance
_service = None

def get_sentinel_service() -> SentinelDataService:
    """Get or create the singleton SentinelDataService instance."""
    global _service
    if _service is None:
        _service = SentinelDataService()
    return _service
