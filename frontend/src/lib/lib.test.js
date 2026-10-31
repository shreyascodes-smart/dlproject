import { afterEach, describe, expect, it, vi } from 'vitest'
import { fmtCoord, fmtDate, fmtPct, fmtRatio, fmtSeconds, productSuffix } from './format'
import { duplicateNames, filterSamples, groupByEvent, placeName } from './samples'
import { API_BASE, ApiError, request, validateAssessment, validateSamples, validateStatus } from './api'
import { coverage, describeError, fmtKm2, freshness, freshnessText, isStale, outcomeUi, showsShare } from './assessment'
import {
  aqiBand,
  clearWeatherCache,
  compass,
  currentHourIndex,
  dayLength,
  describeWeather,
  EU_AQI,
  fetchForecast,
  rainSummary,
  todayIndex,
  US_AQI,
  uvCategory,
} from './weather'
import { routeFromHash } from './router'
import { shortPlace } from './geocode'
import { validLocation } from '../hooks/useLocation'
import { areaBounds, wrapLng } from './map'
import { assessment, forecastResponse, liveAssessment, samples } from '../test/fixtures'

afterEach(() => {
  vi.restoreAllMocks()
  clearWeatherCache()
})

describe('format', () => {
  it('formats percentages, including tiny non-zero values', () => {
    expect(fmtPct(43.4775)).toBe('43.5%')
    expect(fmtPct(0)).toBe('0.0%')
    expect(fmtPct(0.01)).toBe('<0.1%')
    expect(fmtPct(null)).toBe('—')
  })
  it('formats ratios, coordinates, dates and durations', () => {
    expect(fmtRatio(0.43241)).toBe('0.432')
    expect(fmtRatio(null)).toBe('n/a')
    expect(fmtCoord(-18.2, 140.9, 1)).toBe('18.2° S, 140.9° E')
    expect(fmtCoord(40.7, -74.0, 1)).toBe('40.7° N, 74.0° W')
    expect(fmtDate('2019-02-19')).toBe('19 Feb 2019')
    expect(fmtSeconds(0.12)).toBe('120 ms')
    expect(productSuffix('EMSR265_04ESBLY_DEL_MONIT01_v1')).toBe('DEL MONIT01 v1')
  })
  it('formats approximate areas', () => {
    expect(fmtKm2(0)).toBe('0 km²')
    expect(fmtKm2(0.004)).toBe('<0.01 km²')
    expect(fmtKm2(1)).toBe('1.00 km²')
    expect(fmtKm2(21.375)).toBe('21.4 km²')
    expect(fmtKm2(null)).toBe('—')
  })
})

describe('samples', () => {
  const list = samples.samples
  it('derives readable place names, groups and filters', () => {
    expect(placeName(list[1])).toBe('Southnormanton')
    expect(placeName({ aoi_code: 'AOI06', event_id: 'EMSR422' })).toBe('Area AOI06')
    expect(groupByEvent(list).map((g) => g.event_id)).toEqual(['EMSR264', 'EMSR342', 'EMSR9284'])
    expect(filterSamples(list, { split: 'test' })).toHaveLength(1)
    expect(filterSamples(list, { query: 'normanton' })).toHaveLength(1)
  })
  it('flags duplicate place names within an event', () => {
    const d = duplicateNames([
      { event_id: 'E1', aoi_code: '04ESBLY', id: 'E1_04ESBLY_DEL_v1' },
      { event_id: 'E1', aoi_code: '04ESBLY', id: 'E1_04ESBLY_DEL_MONIT01_v1' },
    ])
    expect([...d]).toEqual(['E1|Esbly'])
  })
})

describe('api client', () => {
  it('uses same-origin relative URLs by default (no hard-coded localhost)', async () => {
    expect(API_BASE).toBe('')
    globalThis.fetch = vi.fn(() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ status: 'ready' }), headers: { get: () => null } }))
    await request('/api/flood/model-status')
    expect(globalThis.fetch.mock.calls[0][0]).toBe('/api/flood/model-status')
  })
  it('normalises network errors without exposing the server address', async () => {
    globalThis.fetch = vi.fn(() => Promise.reject(new TypeError('Failed to fetch')))
    const e = await request('/api/x').catch((x) => x)
    expect(e).toBeInstanceOf(ApiError)
    expect(e.kind).toBe('network')
    expect(e.message).not.toMatch(/localhost|8000/)
  })
  it('keeps Retry-After and the sanitised detail of HTTP errors', async () => {
    globalThis.fetch = vi.fn(() =>
      Promise.resolve({ ok: false, status: 429, json: () => Promise.resolve({ detail: 'Too many assessment requests.' }), headers: { get: (k) => (k === 'Retry-After' ? '12' : null) } }),
    )
    const e = await request('/api/x').catch((x) => x)
    expect([e.status, e.retryAfter, e.message]).toEqual([429, 12, 'Too many assessment requests.'])
  })
  it('validates response shapes', () => {
    expect(validateSamples(samples)).toHaveLength(3)
    expect(validateAssessment(assessment(samples.samples[0])).status).toBe('success')
    expect(validateStatus({ status: 'ready' }).status).toBe('ready')
    expect(validateSamples({ samples: [...samples.samples, { id: 'x', split: 'train' }] }).map((s) => s.split)).not.toContain('train')
    const bad = { ...assessment(samples.samples[0]), assessment: null }
    expect(() => validateAssessment(bad)).toThrow(ApiError)
  })
})

describe('assessment presentation', () => {
  it('maps every outcome (and the legacy "unable") to a state', () => {
    expect(outcomeUi('flood_like').short).toBe('Flood-like patterns detected')
    expect(outcomeUi('none_detected').short).toBe('No flood-like patterns detected')
    expect(outcomeUi('insufficient_coverage').tone).toBe('muted')
    expect(outcomeUi('unable').short).toBe('Insufficient valid coverage')
    expect(outcomeUi('imagery_unavailable').short).toBe('Imagery unavailable')
    expect(outcomeUi('service_unavailable').tone).toBe('error')
    expect(showsShare('uncertain_water')).toBe(false)
    expect(showsShare('inconclusive')).toBe(true)
  })
  it('maps request failures to outcome states', () => {
    expect(describeError({ status: 404, message: 'x' }).level).toBe('imagery_unavailable')
    expect(describeError({ status: 422, message: 'x' }).level).toBe('insufficient_coverage')
    expect(describeError({ status: 502, message: 'x' }).title).toBe('Satellite service unavailable')
    expect(describeError({ status: 503, message: 'x' }).level).toBe('service_unavailable')
    expect(describeError({ status: 429, retryAfter: 9 }).message).toMatch(/about 9 s/)
    expect(describeError({ kind: 'network' }).title).toBe('Analysis service unreachable')
    expect(describeError({ kind: 'timeout' }).level).toBe('service_unavailable')
  })
  it('describes imagery freshness honestly', () => {
    expect(freshness({ acquisition_date: '2026-10-18', age_days: 8, archived: false })).toEqual({ date: '18 Oct 2026', age: '8 days old', stale: false })
    expect(freshnessText({ acquisition_date: '2019-02-19', age_days: 2806, archived: true })).toBe('19 Feb 2019 · archived event')
    expect(freshness({ acquisition_date: null }).date).toBe('Date unknown')
    expect(isStale({ acquisition_date: '2026-10-01', age_days: 20, archived: false })).toBe(true)
    expect(isStale({ acquisition_date: '2016-10-01', age_days: 900, archived: true })).toBe(false)
  })
  it('accounts for every pixel with an explicit denominator', () => {
    const c = coverage(liveAssessment({ lat: 25, lng: 85 }))
    expect(c.total).toBe(250000)
    expect(c.land + c.knownWater + c.unusable).toBe(c.total)
    expect(c.floodLike).toBe(10000)
    expect(c.areaKm2.land).toBeCloseTo(21.375)
    expect(c.share).toBeCloseTo((100 * 10000) / 213750, 3)
  })
  it('builds the same 5 × 5 km area as the backend', () => {
    const [[s, w], [n, e]] = areaBounds(19.2535, 72.4662)
    expect(n - s).toBeCloseTo(5 / 110.54, 6)
    expect(e - w).toBeCloseTo(5 / (111.32 * Math.cos((19.2535 * Math.PI) / 180)), 6)
    expect(wrapLng(190)).toBe(-170)
  })
})

describe('weather', () => {
  it('interprets codes, wind, UV and AQI bands on their own scales', () => {
    expect(describeWeather(63).text).toBe('Rain')
    expect(describeWeather(0, 0).icon).toBe('moon')
    expect(describeWeather(999).text).toBe('Unknown')
    expect(compass(360)).toBe('N')
    expect(compass(225)).toBe('SW')
    expect(uvCategory(7)).toBe('High')
    expect(aqiBand(US_AQI, 42).label).toBe('Good')
    expect(aqiBand(US_AQI, 166).label).toBe('Unhealthy')
    expect(aqiBand(EU_AQI, 89).label).toBe('Very poor')
    expect(aqiBand(EU_AQI, 30).label).toBe('Fair')
    expect(aqiBand(US_AQI, null)).toBeNull()
  })
  it('finds "now" and today in responses that include past data', () => {
    const f = forecastResponse()
    expect(currentHourIndex(f.hourly.time, f.current.time)).toBe(24)
    expect(todayIndex(f.daily.time, f.current.time)).toBe(2)
    expect(rainSummary(f)).toEqual({ past24: 6, next24: 0.6, nextProbMax: 55 })
    expect(dayLength('2026-10-28T05:55', '2026-10-28T17:11')).toBe('11 h 16 min')
  })
  it('shares one request between callers and survives a cancelled caller', async () => {
    let resolve
    globalThis.fetch = vi.fn(() => new Promise((r) => (resolve = r)))
    const a = new AbortController()
    const first = fetchForecast({ lat: 1, lng: 2 }, a.signal)
    a.abort() // e.g. React StrictMode cleanup or leaving the page
    const second = fetchForecast({ lat: 1, lng: 2 }, new AbortController().signal)
    resolve({ ok: true, status: 200, json: () => Promise.resolve(forecastResponse()) })
    await expect(first).rejects.toMatchObject({ name: 'AbortError' })
    const r = await second
    expect(r.data.current.temperature_2m).toBe(26.4)
    expect(globalThis.fetch).toHaveBeenCalledTimes(1)
    const cached = await fetchForecast({ lat: 1, lng: 2 })
    expect(cached.cached).toBe(true)
  })
  it('reports schema and HTTP errors in plain language', async () => {
    globalThis.fetch = vi.fn(() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ nope: 1 }) }))
    await expect(fetchForecast({ lat: 3, lng: 4 })).rejects.toMatchObject({ kind: 'schema' })
    globalThis.fetch = vi.fn(() => Promise.resolve({ ok: false, status: 429, json: () => Promise.resolve({}) }))
    await expect(fetchForecast({ lat: 5, lng: 6 })).rejects.toMatchObject({ kind: 'rate' })
  })
})

describe('routing and location', () => {
  it('maps hashes to pages', () => {
    expect(routeFromHash('')).toBe('flood')
    expect(routeFromHash('#/weather')).toBe('weather')
    expect(routeFromHash('#/demo/')).toBe('demo')
    expect(routeFromHash('#/research')).toBe('research')
    expect(routeFromHash('#/unknown')).toBe('flood')
  })
  it('validates stored locations and shortens place names', () => {
    expect(validLocation({ lat: 10, lng: 20 })).toBe(true)
    expect(validLocation({ lat: 100, lng: 20 })).toBe(false)
    expect(validLocation({ lat: '10', lng: 20 })).toBe(false)
    expect(shortPlace('Patna, Bihar, India')).toBe('Patna, Bihar')
  })
})
