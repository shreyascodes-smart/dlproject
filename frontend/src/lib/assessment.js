/**
 * Presentation of assessment outcomes (see ml/inference/assessment.py).
 *
 * Outcomes are descriptive: they say what the model found in one image. They are never a
 * probability, a confidence, a risk rating or a statement that an area is safe.
 */

/** Backend outcome levels plus client-side states for requests that produced no result. */
export const OUTCOMES = {
  flood_like: { tone: 'flood', short: 'Flood-like patterns detected' },
  none_detected: { tone: 'clear', short: 'No flood-like patterns detected' },
  inconclusive: { tone: 'neutral', short: 'Inconclusive' },
  uncertain_water: { tone: 'neutral', short: 'Inconclusive — mostly permanent water' },
  insufficient_coverage: { tone: 'muted', short: 'Insufficient valid coverage' },
  imagery_unavailable: { tone: 'muted', short: 'Imagery unavailable' },
  service_unavailable: { tone: 'error', short: 'Service unavailable' },
}

// Older cached responses used `unable`
const ALIASES = { unable: 'insufficient_coverage' }

export function outcomeUi(level) {
  return OUTCOMES[ALIASES[level] || level] || OUTCOMES.inconclusive
}

/** Outcomes where a flood-like share is meaningful to display. */
export function showsShare(level) {
  return ['flood_like', 'none_detected', 'inconclusive'].includes(ALIASES[level] || level)
}

/**
 * Map a failed request to an outcome state with a plain-language explanation.
 * Messages from the backend are already sanitised; we only use them as supporting detail.
 */
export function describeError(err, { live = true } = {}) {
  const status = err?.status
  const detail = typeof err?.message === 'string' ? err.message : null
  if (err?.kind === 'network') {
    return {
      level: 'service_unavailable',
      title: 'Analysis service unreachable',
      message: 'The analysis service could not be reached. Check your connection; if it persists, the service may be offline.',
      retry: true,
    }
  }
  if (err?.kind === 'timeout') {
    return {
      level: 'service_unavailable',
      title: 'The analysis took too long',
      message: 'The request timed out. Satellite downloads can be slow; try again in a moment.',
      retry: true,
    }
  }
  if (status === 404) {
    return live
      ? {
          level: 'imagery_unavailable',
          title: 'Imagery unavailable',
          message: detail || 'No usable Sentinel-2 image of this area was found for the recent search window.',
          retry: true,
        }
      : { level: 'imagery_unavailable', title: 'Scene not found', message: detail || 'This demonstration scene is not available.', retry: false }
  }
  if (status === 422) {
    return {
      level: 'insufficient_coverage',
      title: 'Insufficient valid coverage',
      message: detail || 'The latest image does not cover enough of this area to analyse it.',
      retry: true,
    }
  }
  if (status === 429) {
    const wait = err?.retryAfter ? ` Try again in about ${err.retryAfter} s.` : ' Try again shortly.'
    return { level: 'service_unavailable', title: 'Too many requests', message: `The assessment limit was reached.${wait}`, retry: true }
  }
  if (status === 502) {
    return {
      level: 'imagery_unavailable',
      title: 'Satellite service unavailable',
      message: detail || 'The Copernicus satellite service could not provide imagery right now.',
      retry: true,
    }
  }
  if (status === 503) {
    return {
      level: 'service_unavailable',
      title: 'Service unavailable',
      message: detail || 'The analysis service is not ready.',
      retry: true,
    }
  }
  if (err?.kind === 'schema') {
    return {
      level: 'service_unavailable',
      title: 'Unexpected response',
      message: 'The analysis service returned a response this page cannot read. Reload the page; if it persists, the service may have been updated.',
      retry: true,
    }
  }
  return {
    level: 'service_unavailable',
    title: 'Assessment could not be completed',
    message: detail || 'Something went wrong while analysing this image.',
    retry: true,
  }
}

/** "18 Oct 2026" + "8 days old" */
export function freshness(f) {
  if (!f?.acquisition_date) return { date: 'Date unknown', age: null, stale: false }
  const d = new Date(`${f.acquisition_date.slice(0, 10)}T00:00:00Z`)
  const date = Number.isNaN(d.getTime())
    ? f.acquisition_date
    : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
  let age = null
  if (f.archived) age = 'archived event'
  else if (typeof f.age_days === 'number') age = f.age_days <= 0 ? 'today' : f.age_days === 1 ? '1 day old' : `${f.age_days} days old`
  return { date, age, stale: !f.archived && typeof f.age_days === 'number' && f.age_days > 14 }
}

/** Back-compat helper used by older tests: "18 Oct 2026 · 8 days old" */
export function freshnessText(f) {
  const x = freshness(f)
  return x.age ? `${x.date} · ${x.age}` : x.date
}

export function isStale(f) {
  return freshness(f).stale
}

/**
 * Area accounting for the analysed scene (pixel counts at native resolution).
 * Every share shown in the UI states its denominator; areas are approximate (pixel count × pixel area).
 */
export function coverage(data) {
  const p = data?.prediction || {}
  const wc = data?.water_context || {}
  const res = data?.imagery?.resolution_m || 10
  const km2 = (n) => (typeof n === 'number' ? (n * res * res) / 1e6 : null)
  const total = p.n_pixels ?? null
  const eligible = wc.n_eligible ?? p.n_eligible_pixels ?? null
  const land = wc.n_eligible_outside_known_water ?? (wc.available === false ? eligible : null)
  const knownWater = eligible != null && land != null ? eligible - land : null
  const unusable = total != null && eligible != null ? Math.max(0, total - eligible) : null
  const floodLike = wc.n_flood_like_outside_known_water ?? null
  const share = data?.assessment?.flood_like_outside_known_water_pct ?? wc.flood_like_outside_known_water_pct ?? null
  const byClass = wc.by_class && Object.keys(wc.by_class).length ? wc.by_class : null
  return {
    total, eligible, land, knownWater, unusable, floodLike, share, byClass, pxM2: res * res,
    areaKm2: { total: km2(total), land: km2(land), knownWater: km2(knownWater), unusable: km2(unusable), floodLike: km2(floodLike) },
    usablePct: p.eligible_pct ?? null,
    waterReference: wc.available !== false,
  }
}

export function fmtKm2(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—'
  if (v === 0) return '0 km²'
  if (v < 0.01) return '<0.01 km²'
  if (v < 10) return `${v.toFixed(2)} km²`
  if (v < 1000) return `${v.toFixed(1)} km²`
  return `${Math.round(v).toLocaleString('en-US')} km²`
}

export const MODEL_TARGET_LABEL = {
  surface_water: 'Surface water (flood-like = water outside known permanent water)',
  ems_flood: 'EMS-style flood extent',
}
