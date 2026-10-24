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


class SentinelDataService:
    """Fetches Sentinel-2 L2A data from Copernicus Data Space."""

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
            logger.error(f"Token request failed: {resp.status_code} {resp.text}")
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
