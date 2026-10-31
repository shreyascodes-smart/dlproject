"""
Known-water reference (permanent water bodies and open sea) for the model's grid.

Source: JRC Global Surface Water v1.4 (1984–2021), Pekel et al. (2016), Nature 540, 418–422.
Licence: Copernicus Programme, free of charge without restriction of use; attribution "Source: EC JRC/Google".

  * Archived WorldFloods scenes: the dataset ships the JRC layer on the exact
    Sentinel-2 grid (PERMANENTWATERJRC; class 3 = permanent water).
  * Live scenes: a window of the JRC "seasonality" tiles (30 m, 10° x 10°) is
    read over HTTP and resampled (nearest) onto the live image grid.
      seasonality 12  -> permanent water (water in all 12 months)
      value 255       -> "no data" = outside the dataset's coverage, which in
                         practice is open sea (JRC maps inland/coastal water)

Limitations: 30 m and ~2021 vintage (new reservoirs, river migration and
coastline change are not captured); "outside coverage" may rarely include
land without Landsat observations; small water bodies can be missed.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from typing import Dict, Optional, Tuple

import numpy as np

from ml.inference.assessment import KW_MAPPED, KW_NONE, KW_PERMANENT, KW_SEA, KW_SEASONAL

logger = logging.getLogger(__name__)

GSW_URL = (
    "https://storage.googleapis.com/global-surface-water/downloads2021/seasonality/"
    "seasonality_{tile}v1_4_2021.tif"
)
GSW_ATTRIBUTION = "Permanent water: JRC Global Surface Water v1.4 (Source: EC JRC/Google)"
JRC_YEARLY_PERMANENT = 3
GSW_PERMANENT_MONTHS = 12
GSW_NODATA = 255

_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_TIMEOUT": "20",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


JRC_YEARLY_SEASONAL = 2
# Bumped whenever the meaning of the known-water mask changes (part of the demo cache key)
REFERENCE_VERSION = "jrc-permanent+seasonal+osm-v1"


def from_worldfloods_jrc(jrc: np.ndarray, include_seasonal: bool = True) -> np.ndarray:
    """WorldFloods PERMANENTWATERJRC (0 no obs, 1 not water, 2 seasonal, 3 permanent) -> known-water codes."""
    out = np.where(jrc == JRC_YEARLY_PERMANENT, KW_PERMANENT, KW_NONE).astype(np.uint8)
    if include_seasonal:
        out[jrc == JRC_YEARLY_SEASONAL] = KW_SEASONAL
    return out


def from_gsw_seasonality(season: np.ndarray, include_seasonal: bool = True) -> np.ndarray:
    """GSW seasonality (0 not water, 1-12 months, 255 no data) -> known-water codes."""
    out = np.full(season.shape, KW_NONE, np.uint8)
    if include_seasonal:
        out[(season >= 1) & (season < GSW_PERMANENT_MONTHS)] = KW_SEASONAL
    out[season == GSW_PERMANENT_MONTHS] = KW_PERMANENT
    out[season == GSW_NODATA] = KW_SEA
    return out


# ---------------- OpenStreetMap mapped water bodies (live scenes) ----------------
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
OSM_ATTRIBUTION = "Mapped water bodies: © OpenStreetMap contributors (ODbL)"
# Ordinary water bodies (set aside inside their mapped outline; water beyond the outline still counts)
OSM_WATER_TAGS = {"natural": {"water"}, "landuse": {"reservoir", "basin"},
                  "waterway": {"riverbank", "river", "canal", "drain", "ditch", "stream", "dock"}}
# Normally wet land: treated as seasonal context, never as flood evidence on its own
OSM_WET_TAGS = {"natural": {"wetland"}}


def osm_category(tags: Dict[str, str]) -> Optional[str]:
    """'lake', 'pond', 'reservoir', 'river', 'canal', 'drain', 'wetland', 'tidal', 'water' or None."""
    t = tags or {}
    if t.get("natural") == "wetland":
        return "tidal" if t.get("wetland") in ("tidalflat", "saltmarsh", "mangrove") else "wetland"
    w = t.get("water")
    if w in ("lake", "pond", "reservoir", "river", "canal", "lagoon", "basin", "oxbow", "wastewater", "ditch"):
        return {"lagoon": "lake", "oxbow": "lake", "basin": "reservoir", "wastewater": "drain",
                "ditch": "drain"}.get(w, w)
    if t.get("landuse") in ("reservoir", "basin"):
        return "reservoir"
    ww = t.get("waterway")
    if ww in ("riverbank", "river", "stream"):
        return "river"
    if ww in ("canal", "dock"):
        return "canal"
    if ww in ("drain", "ditch"):
        return "drain"
    if t.get("natural") == "water":
        return "water"
    return None


def osm_query(bbox) -> str:
    w, s, e, n = bbox
    b = f"({s},{w},{n},{e})"
    return (f'[out:json][timeout:20];('
            f'way["natural"~"^(water|wetland)$"]{b};relation["natural"~"^(water|wetland)$"]{b};'
            f'way["landuse"~"^(reservoir|basin)$"]{b};relation["landuse"~"^(reservoir|basin)$"]{b};'
            f'way["waterway"~"^(riverbank|river|canal|drain|ditch|stream|dock)$"]{b};'
            f'relation["waterway"="riverbank"]{b};);out geom;')


def rasterize_osm(elements, bbox, width: int, height: int) -> Tuple[np.ndarray, Dict[str, int]]:
    """OSM Overpass `out geom` elements -> (codes H x W: 0 / KW_MAPPED / KW_SEASONAL, pixel counts per category).

    Closed ways and closed relation outer rings are filled; inner rings are cut out; open lines
    (drains, canals, small streams mapped as lines) are burned 1 px wide (all_touched).
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    t = from_bounds(*bbox, width, height)
    out = np.zeros((height, width), np.uint8)
    counts: Dict[str, int] = {}

    def ring(geom):
        return [(p["lon"], p["lat"]) for p in geom or [] if "lon" in p]

    for el in elements or []:
        cat = osm_category(el.get("tags", {}))
        if cat is None:
            continue
        code = KW_SEASONAL if cat in ("wetland", "tidal") else KW_MAPPED
        shapes, holes = [], []
        if el.get("type") == "way":
            pts = ring(el.get("geometry"))
            if len(pts) >= 4 and pts[0] == pts[-1]:
                shapes.append({"type": "Polygon", "coordinates": [pts]})
            elif len(pts) >= 2 and cat in ("river", "canal", "drain"):
                shapes.append({"type": "LineString", "coordinates": pts})
        elif el.get("type") == "relation":
            for m in el.get("members", []):
                pts = ring(m.get("geometry"))
                if len(pts) >= 4 and pts[0] == pts[-1]:
                    (holes if m.get("role") == "inner" else shapes).append({"type": "Polygon", "coordinates": [pts]})
        if not shapes:
            continue
        m = rasterize(((g, 1) for g in shapes), out_shape=(height, width), transform=t, fill=0,
                      all_touched=True, dtype="uint8").astype(bool)
        if holes:
            m &= ~rasterize(((g, 1) for g in holes), out_shape=(height, width), transform=t, fill=0,
                            dtype="uint8").astype(bool)
        # mapped water bodies take precedence over wetland context
        if code == KW_MAPPED:
            out[m] = KW_MAPPED
        else:
            out[m & (out == 0)] = KW_SEASONAL
        counts[cat] = counts.get(cat, 0) + int(m.sum())
    return out, counts


OVERPASS_MIRRORS = [OVERPASS_URL, "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
OSM_TIMEOUT_S = 45.0
_osm_pool = None


def fetch_osm_elements(bbox, timeout: float = OSM_TIMEOUT_S):
    """Overpass elements for a bbox, trying a mirror if the main server fails. Raises if all fail."""
    import requests

    last = None
    for url in OVERPASS_MIRRORS:
        try:
            r = requests.post(url, data={"data": osm_query(bbox)}, timeout=timeout,
                              headers={"User-Agent": "DisasterInsights/3.0 (academic research prototype)"})
            r.raise_for_status()
            return r.json().get("elements", [])
        except Exception as e:  # try the next mirror
            last = e
    raise last or RuntimeError("no Overpass endpoint")


def prefetch_osm(bbox):
    """Start the (slow, 20-35 s) Overpass request in the background; the bbox is known before the imagery."""
    global _osm_pool
    from concurrent.futures import ThreadPoolExecutor
    if _osm_pool is None:
        _osm_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="osm")
    return _osm_pool.submit(fetch_osm_elements, list(bbox))


def fetch_osm_water(bbox, width: int, height: int, future=None, timeout: float = OSM_TIMEOUT_S):
    """Mapped water bodies for a live bbox. Raises on network/HTTP errors (callers degrade explicitly)."""
    elements = future.result(timeout=timeout) if future is not None else fetch_osm_elements(bbox, timeout)
    return rasterize_osm(elements, bbox, width, height)


def merge_known_water(jrc_codes: np.ndarray, osm_codes: Optional[np.ndarray]) -> np.ndarray:
    """Combine JRC codes with OSM codes. JRC permanent water and sea keep priority; OSM mapped water
    bodies override 'none' and 'seasonal'; OSM wetlands only fill normally-dry land."""
    out = jrc_codes.copy()
    if osm_codes is None:
        return out
    open_ = (out == KW_NONE) | (out == KW_SEASONAL)
    out[open_ & (osm_codes == KW_MAPPED)] = KW_MAPPED
    out[(out == KW_NONE) & (osm_codes == KW_SEASONAL)] = KW_SEASONAL
    return out


def gsw_tiles_for_bbox(west: float, south: float, east: float, north: float):
    """GSW tile names (e.g. '70E_20N' = lon 70..80E, lat 10..20N) intersecting a bbox."""
    tiles = []
    lon0s = range(int(math.floor(west / 10) * 10), int(math.floor((east - 1e-9) / 10) * 10) + 1, 10)
    lat_tops = range(int(math.ceil((south + 1e-9) / 10) * 10), int(math.ceil(north / 10) * 10) + 1, 10)
    for lon0 in lon0s:
        for lat0 in lat_tops:
            ew = f"{abs(lon0)}{'E' if lon0 >= 0 else 'W'}"
            ns = f"{abs(lat0)}{'N' if lat0 >= 0 else 'S'}"
            tiles.append(f"{ew}_{ns}")
    return tiles


class WaterReference:
    """Fetches the known-water mask for live bounding boxes (cached in memory)."""

    def __init__(self, url_template: str = GSW_URL, cache_size: int = 32, use_osm: bool = True):
        self.url_template = url_template
        self.use_osm = use_osm
        self._cache: Dict[Tuple, np.ndarray] = {}
        self._order = []
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def for_bbox(self, bbox, width: int, height: int, osm_future=None) -> Tuple[Optional[np.ndarray], Dict[str, str]]:
        """Known-water codes on a (height, width) EPSG:4326 grid covering bbox [west, south, east, north].

        Returns (mask or None, info). None means the reference could not be read; callers must
        then say that permanent water was NOT excluded.
        """
        key = (tuple(round(v, 6) for v in bbox), width, height)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        try:
            jrc = self._fetch(bbox, width, height)
        except Exception as e:  # network/HTTP/GDAL errors
            logger.warning("Permanent-water reference unavailable: %s", e)
            return None, {"source": GSW_ATTRIBUTION, "status": "unavailable", "detail": type(e).__name__,
                          "version": REFERENCE_VERSION}
        info = {"source": GSW_ATTRIBUTION, "status": "ok", "version": REFERENCE_VERSION,
                "layers": {"jrc": "ok"}}
        mask = jrc
        if self.use_osm:
            try:
                osm, cats = fetch_osm_water(bbox, width, height, future=osm_future)
                mask = merge_known_water(jrc, osm)
                info["layers"]["osm"] = "ok"
                info["osm_categories_px"] = cats
                info["source"] = f"{GSW_ATTRIBUTION}; {OSM_ATTRIBUTION}"
            except Exception as e:
                logger.warning("OSM water bodies unavailable: %s", type(e).__name__)
                info["layers"]["osm"] = "unavailable"
                info["status"] = "partial"
        result = (mask, info)
        if info["status"] == "ok":   # never cache a degraded reference
            with self._lock:
                self._cache[key] = result
                self._order.append(key)
                while len(self._order) > self._cache_size:
                    self._cache.pop(self._order.pop(0), None)
        return result

    def _fetch(self, bbox, width, height) -> np.ndarray:
        """Nearest-neighbour sample of GSW seasonality at each output pixel centre.

        GSW tiles and the live grid are both EPSG:4326, so this is exact index arithmetic
        (no warping, so the 255 'no data' code is never altered by GDAL).
        """
        import rasterio
        from rasterio.windows import Window

        west, south, east, north = bbox
        lons = west + (np.arange(width) + 0.5) * (east - west) / width
        lats = north - (np.arange(height) + 0.5) * (north - south) / height
        lon_g, lat_g = np.meshgrid(lons, lats)
        season = np.full((height, width), GSW_NODATA, np.uint8)
        filled = np.zeros((height, width), bool)

        with rasterio.Env(**_GDAL_ENV):
            for tile in gsw_tiles_for_bbox(west, south, east, north):
                url = "/vsicurl/" + self.url_template.format(tile=tile)
                with rasterio.open(url) as src:
                    if src.crs and src.crs.to_epsg() != 4326:
                        raise ValueError(f"Unexpected CRS {src.crs} for {tile}")
                    inv = ~src.transform
                    cols_f, rows_f = inv * (lon_g, lat_g)
                    cols = np.floor(cols_f).astype(np.int64)
                    rows = np.floor(rows_f).astype(np.int64)
                    inside = (cols >= 0) & (cols < src.width) & (rows >= 0) & (rows < src.height) & ~filled
                    if not inside.any():
                        continue
                    r0, r1 = rows[inside].min(), rows[inside].max() + 1
                    c0, c1 = cols[inside].min(), cols[inside].max() + 1
                    data = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                    season[inside] = data[rows[inside] - r0, cols[inside] - c0]
                    filled |= inside
        if not filled.all():
            raise ValueError("Permanent-water reference does not cover the whole area")
        return from_gsw_seasonality(season)


_ref: Optional[WaterReference] = None


def get_water_reference() -> WaterReference:
    global _ref
    if _ref is None:
        _ref = WaterReference(os.getenv("GSW_URL_TEMPLATE", GSW_URL),
                              use_osm=os.getenv("DI_USE_OSM_WATER", "1") != "0")
    return _ref
