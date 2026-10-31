import { useCallback, useEffect, useState } from 'react'
import { reverseGeocode, shortPlace } from '../lib/geocode'

const KEY = 'di-location'

export function validLocation(v) {
  return (
    v &&
    typeof v.lat === 'number' &&
    typeof v.lng === 'number' &&
    Number.isFinite(v.lat) &&
    Number.isFinite(v.lng) &&
    Math.abs(v.lat) <= 90 &&
    Math.abs(v.lng) <= 180
  )
}

function read() {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || 'null')
    return validLocation(v) ? { lat: v.lat, lng: v.lng, name: typeof v.name === 'string' ? v.name : undefined } : null
  } catch {
    return null
  }
}

/** Selected location shared by every page, remembered across reloads (coordinates + optional name only). */
export function useStoredLocation() {
  const [location, setLocationState] = useState(read)
  const setLocation = useCallback((loc) => {
    const next = validLocation(loc) ? { lat: loc.lat, lng: loc.lng, name: loc.name || undefined } : null
    setLocationState(next)
    try {
      if (next) localStorage.setItem(KEY, JSON.stringify(next))
      else localStorage.removeItem(KEY)
    } catch {
      /* storage unavailable: keep in memory only */
    }
  }, [])
  return [location, setLocation]
}

/** Place name for a location: the search result name if known, else reverse geocoding (cached). */
export function usePlaceName(location) {
  const [res, setRes] = useState({ key: null, name: null, status: 'idle' })
  const key = location ? `${location.lat.toFixed(4)},${location.lng.toFixed(4)}` : null
  const named = location?.name
  useEffect(() => {
    if (!key || named) return undefined
    const ctrl = new AbortController()
    const [lat, lng] = key.split(',').map(Number)
    reverseGeocode({ lat, lng }, ctrl.signal)
      .then((name) => setRes({ key, name, status: 'done' }))
      .catch((e) => e?.name !== 'AbortError' && setRes({ key, name: null, status: 'error' }))
    return () => ctrl.abort()
  }, [key, named])
  if (!location) return { name: null, status: 'idle' }
  if (named) return { name: shortPlace(named), status: 'done' }
  if (res.key !== key) return { name: null, status: 'loading' }
  return { name: res.name ? shortPlace(res.name, 3) : null, status: res.status }
}
