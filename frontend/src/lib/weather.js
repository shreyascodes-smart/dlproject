/**
 * Open-Meteo forecast and air-quality clients (no API key; requests go directly from the browser).
 * Every variable requested here was verified against a live response on 28 Oct 2026.
 * Results are cached per location for 10 minutes and concurrent identical requests are shared,
 * so tab switches and re-renders never multiply network calls.
 */
const FORECAST = 'https://api.open-meteo.com/v1/forecast'
const AIR = 'https://air-quality-api.open-meteo.com/v1/air-quality'
export const TTL_MS = 10 * 60 * 1000

const CURRENT =
  'temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,precipitation_probability,weather_code,cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m,visibility,uv_index'
const HOURLY = 'temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m,relative_humidity_2m'
const DAILY =
  'weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,uv_index_max,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant'
const AIR_CURRENT =
  'us_aqi,european_aqi,pm2_5,pm10,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,dust,alder_pollen,birch_pollen,grass_pollen,ragweed_pollen'

const cache = new Map() // url -> { at, data } | { promise }

export function round4(v) {
  return Math.round(v * 1e4) / 1e4
}

const FETCH_TIMEOUT_MS = 20000

/** Let one caller stop waiting (AbortError) without cancelling a request other callers share. */
function withSignal(promise, signal) {
  if (!signal) return promise
  const abortError = () => Object.assign(new Error('Aborted'), { name: 'AbortError' })
  if (signal.aborted) return Promise.reject(abortError())
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(abortError())
    signal.addEventListener('abort', onAbort, { once: true })
    promise.then(
      (v) => {
        signal.removeEventListener('abort', onAbort)
        resolve(v)
      },
      (e) => {
        signal.removeEventListener('abort', onAbort)
        reject(e)
      },
    )
  })
}

/**
 * Cached JSON fetch. The network request is shared by every caller for the same URL and is not
 * tied to any one caller's AbortSignal (React StrictMode or a quick tab switch must not leave
 * the others waiting on a cancelled request); it has its own timeout instead.
 */
function cachedJson(url, signal, { force = false } = {}) {
  const hit = cache.get(url)
  if (!force && hit?.data && Date.now() - hit.at < TTL_MS) return Promise.resolve({ data: hit.data, fetchedAt: hit.at, cached: true })
  if (hit?.promise) return withSignal(hit.promise, signal)
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS)
  const promise = (async () => {
    let res
    try {
      res = await fetch(url, { signal: ctrl.signal })
    } catch (e) {
      if (ctrl.signal.aborted) throw Object.assign(new Error('The service took too long to respond. Try again.'), { kind: 'timeout' })
      throw Object.assign(new Error('The service could not be reached. Check your connection and try again.'), { kind: 'network', cause: e })
    }
    if (res.status === 429) throw Object.assign(new Error('The service is rate-limiting requests. Try again shortly.'), { kind: 'rate' })
    if (!res.ok) throw Object.assign(new Error(`The service returned an error (${res.status}).`), { kind: 'http' })
    const data = await res.json().catch(() => {
      throw Object.assign(new Error('The service returned an unreadable response.'), { kind: 'schema' })
    })
    return data
  })()
  const shared = promise.then(
    (data) => {
      clearTimeout(timer)
      const at = Date.now()
      cache.set(url, { at, data })
      return { data, fetchedAt: at, cached: false }
    },
    (e) => {
      clearTimeout(timer)
      if (hit?.data) cache.set(url, { at: hit.at, data: hit.data })
      else cache.delete(url)
      throw e
    },
  )
  cache.set(url, { ...(hit?.data ? { at: hit.at, data: hit.data } : {}), promise: shared })
  return withSignal(shared, signal)
}

export function forecastUrl({ lat, lng }) {
  const p = new URLSearchParams({
    // past_days/past_hours add recent rainfall (relevant context for flooding); verified live on 28 Oct 2026:
    // hourly = 24 past + 48 forecast hours, daily = 2 past days + 7 forecast days.
    latitude: round4(lat), longitude: round4(lng), timezone: 'auto', forecast_days: 7, past_days: 2,
    forecast_hours: 48, past_hours: 24,
    current: CURRENT, hourly: HOURLY, daily: DAILY,
  })
  return `${FORECAST}?${p}`
}

export function airUrl({ lat, lng }) {
  const p = new URLSearchParams({
    latitude: round4(lat), longitude: round4(lng), timezone: 'auto', past_days: 1, forecast_days: 3,
    current: AIR_CURRENT, hourly: 'us_aqi,european_aqi,pm2_5',
  })
  return `${AIR}?${p}`
}

function check(d, keys, what) {
  if (!d || typeof d !== 'object' || !keys.every((k) => d[k])) {
    throw Object.assign(new Error(`The ${what} service returned an unexpected response.`), { kind: 'schema' })
  }
  return d
}

export const fetchForecast = (loc, signal, opts) =>
  cachedJson(forecastUrl(loc), signal, opts).then((r) => ({ ...r, data: check(r.data, ['current', 'daily'], 'weather') }))

export const fetchAir = (loc, signal, opts) =>
  cachedJson(airUrl(loc), signal, opts).then((r) => ({ ...r, data: check(r.data, ['current'], 'air-quality') }))

export function clearWeatherCache() {
  cache.clear()
}

/* ───────────── Interpretation ───────────── */
const WMO = [
  [[0], 'Clear sky', 'sun'],
  [[1], 'Mainly clear', 'sun'],
  [[2], 'Partly cloudy', 'cloud-sun'],
  [[3], 'Overcast', 'cloud'],
  [[45, 48], 'Fog', 'fog'],
  [[51, 53, 55], 'Drizzle', 'drizzle'],
  [[56, 57], 'Freezing drizzle', 'drizzle'],
  [[61], 'Light rain', 'rain'],
  [[63], 'Rain', 'rain'],
  [[65], 'Heavy rain', 'rain'],
  [[66, 67], 'Freezing rain', 'rain'],
  [[71, 73, 75, 77], 'Snow', 'snow'],
  [[80], 'Light showers', 'rain'],
  [[81], 'Showers', 'rain'],
  [[82], 'Violent showers', 'rain'],
  [[85, 86], 'Snow showers', 'snow'],
  [[95], 'Thunderstorm', 'storm'],
  [[96, 99], 'Thunderstorm with hail', 'storm'],
]

export function describeWeather(code, isDay = 1) {
  const hit = WMO.find(([codes]) => codes.includes(code))
  if (!hit) return { text: 'Unknown', icon: 'cloud' }
  const icon = !isDay && hit[2] === 'sun' ? 'moon' : !isDay && hit[2] === 'cloud-sun' ? 'cloud-moon' : hit[2]
  return { text: hit[1], icon }
}

const DIRS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
export function compass(deg) {
  if (typeof deg !== 'number') return '—'
  return DIRS[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16]
}

export function uvCategory(v) {
  if (typeof v !== 'number') return null
  if (v < 3) return 'Low'
  if (v < 6) return 'Moderate'
  if (v < 8) return 'High'
  if (v < 11) return 'Very high'
  return 'Extreme'
}

/** US EPA AQI (0–500). Guidance follows the EPA's general category statements. */
export const US_AQI = [
  { max: 50, label: 'Good', tone: 'good', color: '#00e400', advice: 'Air quality is satisfactory.' },
  { max: 100, label: 'Moderate', tone: 'moderate', color: '#ffff00', advice: 'Unusually sensitive people should consider limiting prolonged outdoor exertion.' },
  { max: 150, label: 'Unhealthy for sensitive groups', tone: 'usg', color: '#ff7e00', advice: 'Sensitive groups (people with heart or lung disease, older adults, children) should reduce prolonged or heavy outdoor exertion.' },
  { max: 200, label: 'Unhealthy', tone: 'unhealthy', color: '#ff0000', advice: 'Everyone should reduce prolonged or heavy outdoor exertion; sensitive groups should avoid it.' },
  { max: 300, label: 'Very unhealthy', tone: 'very', color: '#8f3f97', advice: 'Everyone should avoid prolonged or heavy outdoor exertion.' },
  { max: Infinity, label: 'Hazardous', tone: 'hazard', color: '#7e0023', advice: 'Everyone should avoid all outdoor exertion.' },
]

/** European Air Quality Index (EEA bands and colours). */
export const EU_AQI = [
  { max: 20, label: 'Good', tone: 'good', color: '#50f0e6', advice: 'Usual outdoor activities are fine.' },
  { max: 40, label: 'Fair', tone: 'good', color: '#50ccaa', advice: 'Usual outdoor activities are fine.' },
  { max: 60, label: 'Moderate', tone: 'moderate', color: '#f0e641', advice: 'Sensitive people may consider reducing intense outdoor activity if they have symptoms.' },
  { max: 80, label: 'Poor', tone: 'usg', color: '#ff5050', advice: 'Consider reducing intense outdoor activity, especially with symptoms; sensitive people should reduce it.' },
  { max: 100, label: 'Very poor', tone: 'unhealthy', color: '#960032', advice: 'Reduce intense outdoor activity; sensitive people should avoid it.' },
  { max: Infinity, label: 'Extremely poor', tone: 'hazard', color: '#7d2181', advice: 'Avoid intense outdoor activity.' },
]

/** Gauge ranges (the upper scale ends used for drawing only). */
export const US_AQI_MAX = 300
export const EU_AQI_MAX = 120

export function aqiBand(scale, v) {
  if (typeof v !== 'number') return null
  return scale.find((b) => v <= b.max)
}

/** Back-compat helper (US scale) */
export function aqiCategory(v) {
  const b = aqiBand(US_AQI, v)
  return b ? { label: b.label, tone: b.tone } : { label: 'Unavailable', tone: 'muted' }
}

export const POLLUTANTS = [
  ['pm2_5', 'PM2.5', 'Fine particles'],
  ['pm10', 'PM10', 'Coarse particles'],
  ['ozone', 'O₃', 'Ozone'],
  ['nitrogen_dioxide', 'NO₂', 'Nitrogen dioxide'],
  ['sulphur_dioxide', 'SO₂', 'Sulphur dioxide'],
  ['carbon_monoxide', 'CO', 'Carbon monoxide'],
  ['dust', 'Dust', 'Mineral dust'],
]

export const POLLEN = [
  ['alder_pollen', 'Alder'],
  ['birch_pollen', 'Birch'],
  ['grass_pollen', 'Grass'],
  ['ragweed_pollen', 'Ragweed'],
]

/** Sum of the finite values in a slice (null when there are none). */
export function sumRange(values, from, to) {
  if (!Array.isArray(values)) return null
  const xs = values.slice(Math.max(0, from), Math.max(0, to)).filter((v) => typeof v === 'number' && Number.isFinite(v))
  return xs.length ? Math.round(xs.reduce((a, b) => a + b, 0) * 10) / 10 : null
}

export function maxRange(values, from, to) {
  if (!Array.isArray(values)) return null
  const xs = values.slice(Math.max(0, from), Math.max(0, to)).filter((v) => typeof v === 'number' && Number.isFinite(v))
  return xs.length ? Math.max(...xs) : null
}

/** Index of today's entry in the daily arrays (the response also includes two past days). */
export function todayIndex(days, currentTime) {
  if (!days?.length || !currentTime) return 0
  const i = days.indexOf(currentTime.slice(0, 10))
  return i < 0 ? 0 : i
}

/** Recent and upcoming rainfall around "now" from the hourly series. */
export function rainSummary(f) {
  const h = f?.hourly
  if (!h?.time) return { past24: null, next24: null, nextProbMax: null }
  const i = currentHourIndex(h.time, f.current?.time)
  return {
    past24: sumRange(h.precipitation, i - 24, i),
    next24: sumRange(h.precipitation, i, i + 24),
    nextProbMax: maxRange(h.precipitation_probability, i, i + 24),
  }
}

/** Day length "11 h 42 min" from local ISO sunrise/sunset strings. */
export function dayLength(sunrise, sunset) {
  if (!sunrise || !sunset) return null
  const toMin = (s) => Number(s.slice(11, 13)) * 60 + Number(s.slice(14, 16))
  const m = toMin(sunset) - toMin(sunrise)
  if (!Number.isFinite(m) || m <= 0) return null
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')} min`
}

/** Index of the hourly entry for the API's current time (local, same timezone strings). */
export function currentHourIndex(times, currentTime) {
  if (!times?.length || !currentTime) return 0
  const hour = currentTime.slice(0, 13)
  const i = times.findIndex((t) => t.slice(0, 13) === hour)
  return i < 0 ? 0 : i
}

/** "2026-10-28T02:45" (local to the location) -> "02:45" / "Tue 28 Oct" without converting timezones */
export function localClock(iso) {
  return iso ? iso.slice(11, 16) : '—'
}

export function localDay(isoDate, { long = false } = {}) {
  if (!isoDate) return '—'
  const d = new Date(`${isoDate.slice(0, 10)}T12:00:00Z`)
  return d.toLocaleDateString('en-GB', long ? { weekday: 'long', day: 'numeric', month: 'short', timeZone: 'UTC' } : { weekday: 'short', day: 'numeric', timeZone: 'UTC' })
}
