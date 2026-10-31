/**
 * Place search, autocomplete and reverse geocoding (no API key).
 *
 * Autocomplete: Photon (photon.komoot.io), an OpenStreetMap geocoder built for search-as-you-type.
 * The public Nominatim endpoint is NOT used for keystroke autocomplete (its usage policy forbids it);
 * Nominatim is used only for reverse geocoding a clicked map point, with a per-coordinate cache.
 *
 * The provider is replaceable: anything that implements `autocomplete(query, { signal, near })`
 * returning Suggestion[] can be passed to setAutocompleteProvider().
 *
 * Suggestion: { id, name, secondary, label, category, kind, lat, lng, provider }
 */
const PHOTON = (import.meta.env.VITE_PHOTON_URL || 'https://photon.komoot.io').replace(/\/+$/, '')
const REVERSE = 'https://nominatim.openstreetmap.org/reverse'
export const AUTOCOMPLETE_ATTRIBUTION = 'Search © OpenStreetMap contributors, Photon by Komoot'
export const MAX_SUGGESTIONS = 6
const FETCH_LIMIT = 15 // over-fetch, then drop transit-stop duplicates (e.g. one "Bandra" per platform)
const CACHE_TTL_MS = 5 * 60 * 1000
const CACHE_MAX = 100

async function getJson(url, signal) {
  const res = await fetch(url, { signal, headers: { Accept: 'application/json' } })
  if (!res.ok) throw Object.assign(new Error(`Request failed (${res.status})`), { status: res.status })
  return res.json()
}

/* ───────────── Photon autocomplete ───────────── */

// Features that are noise for a 5 × 5 km area search (each platform / bus stop is its own feature)
const DROP = new Set(['railway:stop', 'railway:platform', 'public_transport:platform', 'public_transport:stop_position',
  'highway:bus_stop', 'railway:subway_entrance', 'railway:tram_stop'])

const CATEGORY = [
  [(k) => k === 'aeroway', 'airport', 'Airport'],
  [(k, v) => k === 'railway' || (k === 'public_transport' && v === 'station'), 'rail', 'Railway station'],
  [(k, v) => ['college', 'university', 'school', 'kindergarten'].includes(v) || v === 'educational_institution', 'education', 'Education'],
  [(k, v) => ['hospital', 'clinic', 'doctors'].includes(v), 'hospital', 'Hospital'],
  [(k) => k === 'highway', 'road', 'Road'],
  [(k) => k === 'natural' || k === 'water' || k === 'waterway', 'nature', 'Natural feature'],
  [(k) => ['tourism', 'historic', 'leisure'].includes(k), 'landmark', 'Landmark'],
  [(k, v) => k === 'place' && ['city', 'town', 'municipality'].includes(v), 'city', 'City'],
  [(k, v) => k === 'place' && ['village', 'hamlet', 'isolated_dwelling'].includes(v), 'village', 'Village'],
  [(k, v) => k === 'place' && ['suburb', 'neighbourhood', 'quarter', 'locality', 'borough'].includes(v), 'area', 'Area'],
  [(k, v) => k === 'place' && ['district', 'county', 'state', 'region', 'province', 'country'].includes(v), 'region', 'Region'],
  [(k) => k === 'boundary', 'region', 'Region'],
  [(k) => k === 'building' || k === 'amenity' || k === 'office' || k === 'shop', 'place', 'Place'],
]

function categorise(key, value) {
  const hit = CATEGORY.find(([test]) => test(key, value))
  return hit ? { kind: hit[1], category: hit[2] } : { kind: 'place', category: 'Place' }
}

const uniq = (xs) => xs.filter((x, i) => x && xs.indexOf(x) === i)

/** One Photon GeoJSON feature -> Suggestion (or null when unusable). */
export function fromPhoton(f) {
  const p = f?.properties || {}
  const [lng, lat] = f?.geometry?.coordinates || []
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null
  const street = p.street ? [p.housenumber, p.street].filter(Boolean).join(' ') : null
  const name = p.name || street || p.locality || p.district || p.city
  if (!name) return null
  // "Thakur Village, R/S Ward, Mumbai, Maharashtra, India" without repeating the name
  const parts = uniq([p.name ? street : null, p.locality, p.district, p.city || p.county, p.state, p.country])
    .filter((x) => x.toLowerCase() !== name.toLowerCase())
  const { kind, category } = categorise(p.osm_key, p.osm_value)
  return {
    id: `osm:${p.osm_type || ''}${p.osm_id ?? `${lat},${lng}`}`,
    name,
    secondary: parts.join(', '),
    label: [name, ...parts.slice(0, 3)].join(', '),
    category,
    kind,
    lat,
    lng,
    provider: 'photon',
    _key: `${p.osm_key}:${p.osm_value}`,
  }
}

const SAME_PLACE_KM = 1.5 // same name this close = the same place for a 5 × 5 km analysis area

function km(a, b) {
  const dLat = ((b.lat - a.lat) * Math.PI) / 180
  const dLng = ((b.lng - a.lng) * Math.PI) / 180
  const x = Math.sin(dLat / 2) ** 2 + Math.cos((a.lat * Math.PI) / 180) * Math.cos((b.lat * Math.PI) / 180) * Math.sin(dLng / 2) ** 2
  return 12742 * Math.asin(Math.sqrt(x))
}

/** Drop transit-stop noise and same-name duplicates of one place; keep provider order. */
export function rankSuggestions(list, limit = MAX_SUGGESTIONS) {
  const seen = new Set()
  const out = []
  for (const s of list) {
    if (!s || DROP.has(s._key)) continue
    const name = s.name.toLowerCase()
    const k = `${name}|${s.secondary.split(',')[0]?.trim().toLowerCase()}`
    if (seen.has(k) || out.some((o) => o.name.toLowerCase() === name && km(o, s) < SAME_PLACE_KM)) continue
    seen.add(k)
    const { _key, ...clean } = s
    out.push(clean)
    if (out.length >= limit) break
  }
  return out
}

const acCache = new Map() // key -> { at, value }

function cacheGet(key) {
  const hit = acCache.get(key)
  if (!hit) return null
  if (Date.now() - hit.at > CACHE_TTL_MS) {
    acCache.delete(key)
    return null
  }
  acCache.delete(key) // LRU refresh
  acCache.set(key, hit)
  return hit.value
}

function cachePut(key, value) {
  acCache.set(key, { at: Date.now(), value })
  while (acCache.size > CACHE_MAX) acCache.delete(acCache.keys().next().value)
}

export function clearGeocodeCache() {
  acCache.clear()
}

const photonProvider = {
  name: 'photon',
  attribution: AUTOCOMPLETE_ATTRIBUTION,
  async autocomplete(query, { signal, near } = {}) {
    const params = new URLSearchParams({ q: query, limit: FETCH_LIMIT, lang: 'en' })
    // Location bias only (results stay global): nearer matches rank first
    if (near && Number.isFinite(near.lat) && Number.isFinite(near.lng)) {
      params.set('lat', near.lat.toFixed(3))
      params.set('lon', near.lng.toFixed(3))
    }
    const data = await getJson(`${PHOTON}/api/?${params}`, signal)
    return rankSuggestions((Array.isArray(data?.features) ? data.features : []).map(fromPhoton))
  },
}

let provider = photonProvider

export function setAutocompleteProvider(p) {
  provider = p || photonProvider
  acCache.clear()
}

export function autocompleteProvider() {
  return provider
}

/**
 * Suggestions for a query (cached for 5 minutes per normalised query + bias cell).
 * Throws AbortError when cancelled; other errors mean the provider is unreachable.
 */
export async function autocomplete(query, { signal, near } = {}) {
  const q = query.trim().replace(/\s+/g, ' ')
  if (q.length < 2) return []
  const cell = near ? `${near.lat.toFixed(1)},${near.lng.toFixed(1)}` : ''
  const key = `${provider.name}|${q.toLowerCase()}|${cell}`
  const hit = cacheGet(key)
  if (hit) return hit
  const value = await provider.autocomplete(q, { signal, near })
  cachePut(key, value)
  return value
}

/** Back-compat: submit-style search now uses the autocomplete provider. */
export async function searchPlaces(query, signal) {
  return autocomplete(query, { signal })
}

/* ───────────── Reverse geocoding (map clicks, "Use my location") ───────────── */
const reverseCache = new Map()

export async function reverseGeocode({ lat, lng }, signal) {
  const key = `${lat.toFixed(3)},${lng.toFixed(3)}`
  if (reverseCache.has(key)) return reverseCache.get(key)
  const params = new URLSearchParams({ lat, lon: lng, format: 'json', zoom: 14, addressdetails: 1, 'accept-language': 'en' })
  const r = await getJson(`${REVERSE}?${params}`, signal)
  const a = r.address || {}
  const parts = [a.suburb || a.neighbourhood, a.city || a.town || a.village || a.county, a.state, a.country].filter(Boolean)
  const unique = parts.filter((p, i) => parts.indexOf(p) === i)
  const name = unique.join(', ') || r.display_name || null
  reverseCache.set(key, name)
  return name
}

/** "Andheri, Mumbai, Maharashtra, India" -> "Andheri, Mumbai" */
export function shortPlace(name, parts = 2) {
  return name ? name.split(',').slice(0, parts).map((s) => s.trim()).join(', ') : null
}
