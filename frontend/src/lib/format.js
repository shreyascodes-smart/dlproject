const isNum = (v) => typeof v === 'number' && Number.isFinite(v)

export function fmtPct(v, digits = 1) {
  if (!isNum(v)) return '—'
  if (v > 0 && v < 10 ** -digits) return `<${(10 ** -digits).toFixed(digits)}%`
  return `${v.toFixed(digits)}%`
}

export function fmtRatio(v, digits = 3) {
  return isNum(v) ? v.toFixed(digits) : 'n/a'
}

export function fmtCount(n) {
  if (!isNum(n)) return '—'
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k`
  return String(n)
}

export function fmtInt(n) {
  return isNum(n) ? n.toLocaleString('en-US') : '—'
}

export function fmtCoord(lat, lng, digits = 3) {
  if (!isNum(lat) || !isNum(lng)) return '—'
  return `${Math.abs(lat).toFixed(digits)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lng).toFixed(digits)}° ${
    lng >= 0 ? 'E' : 'W'
  }`
}

export function fmtDate(iso) {
  if (!iso) return '—'
  const d = new Date(iso.length === 10 ? `${iso}T00:00:00Z` : iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
}

export function fmtSeconds(s) {
  if (!isNum(s)) return '—'
  if (s < 1) return `${Math.max(1, Math.round(s * 1000))} ms`
  return `${s.toFixed(s < 10 ? 1 : 0)} s`
}

/** "EMSR265_04ESBLY_DEL_MONIT01_v1" -> "DEL MONIT01 v1" (product suffix that tells duplicates apart) */
export function productSuffix(id) {
  const parts = String(id).split('_')
  return parts.length > 2 ? parts.slice(2).join(' ') : ''
}

export const SPLIT_LABEL = { val: 'Validation', test: 'Held-out test' }
