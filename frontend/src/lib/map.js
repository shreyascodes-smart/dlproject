import L from 'leaflet'

/* Local pin (no external icon URLs) */
export const pinIcon = L.divIcon({
  className: 'map-pin',
  html: '<span class="map-pin__dot"></span>',
  iconSize: [22, 22],
  iconAnchor: [11, 11],
})

export const DEFAULT_CENTER = [20.5, 78.9]
export const DEFAULT_ZOOM = 4

/** Leaflet can report longitudes outside ±180 after panning across the antimeridian. */
export function wrapLng(lng) {
  return ((((lng + 180) % 360) + 360) % 360) - 180
}

/** Same 5 × 5 km box the backend requests from Copernicus (sentinel_service._make_bbox). */
export function areaBounds(lat, lng, sizeKm = 5) {
  const dLat = sizeKm / 110.54 / 2
  const dLng = sizeKm / (111.32 * Math.cos((lat * Math.PI) / 180)) / 2
  return [
    [lat - dLat, lng - dLng],
    [lat + dLat, lng + dLng],
  ]
}

export function bboxToBounds(b) {
  if (!b) return null
  return [
    [b.south, b.west],
    [b.north, b.east],
  ]
}
