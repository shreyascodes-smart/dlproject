/** Helpers for the demo-sample catalogue returned by GET /api/flood/demo-samples. */

const titleCase = (s) =>
  s
    .toLowerCase()
    .replace(/(^|[\s-])\p{L}/gu, (m) => m.toUpperCase())

/** Human place name from the AOI code, e.g. "07SOUTHNORMANTON" -> "Southnormanton", "AOI06" -> "Area AOI06". */
export function placeName(sample) {
  const raw = (sample?.aoi_code || '').replace(/^\d+/, '')
  if (!raw) return sample?.event_id || 'Scene'
  if (/^AOI\d+$/i.test(raw)) return `Area ${raw.toUpperCase()}`
  return titleCase(raw)
}

/** Group samples by event (catalogue order is preserved). */
export function groupByEvent(samples) {
  const groups = []
  const index = new Map()
  for (const s of samples) {
    if (!index.has(s.event_id)) {
      index.set(s.event_id, groups.length)
      groups.push({ event_id: s.event_id, split: s.split, samples: [] })
    }
    groups[index.get(s.event_id)].samples.push(s)
  }
  return groups
}

/** Names that occur more than once (same AOI, different product) need a product suffix to tell them apart. */
export function duplicateNames(samples) {
  const counts = new Map()
  for (const s of samples) {
    const k = `${s.event_id}|${placeName(s)}`
    counts.set(k, (counts.get(k) || 0) + 1)
  }
  return new Set([...counts].filter(([, n]) => n > 1).map(([k]) => k))
}

export function filterSamples(samples, { query = '', split = 'all' } = {}) {
  const q = query.trim().toLowerCase()
  return samples.filter((s) => {
    if (split !== 'all' && s.split !== split) return false
    if (!q) return true
    return [s.id, s.event_id, s.aoi_code, placeName(s), s.event_type, s.s2_date]
      .filter(Boolean)
      .some((v) => String(v).toLowerCase().includes(q))
  })
}
