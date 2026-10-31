/**
 * Client for the Disaster Insights FastAPI backend.
 *
 * - The API base URL is environment-aware: by default requests are same-origin and relative
 *   (`/api/...`), which the Vite dev/preview proxy or a production reverse proxy forwards to
 *   the backend. Set VITE_API_URL at build time to call a backend on another origin.
 * - Every request has a timeout and can be cancelled with an AbortSignal.
 * - Errors are normalised into ApiError with a user-readable message (never raw server text
 *   beyond the backend's own sanitised `detail`).
 * - Responses are checked for the fields the UI depends on, so a backend change shows a clear
 *   message instead of a blank screen.
 */
export const API_BASE = String(import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')

export class ApiError extends Error {
  constructor(message, { status = null, kind = 'http', detail = null, retryAfter = null, body = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.kind = kind // 'network' | 'timeout' | 'http' | 'schema'
    this.detail = detail
    this.retryAfter = retryAfter
    this.body = body
  }
}

function linkSignals(signal, timeoutMs) {
  const ctrl = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    ctrl.abort()
  }, timeoutMs)
  const onAbort = () => ctrl.abort()
  if (signal) {
    if (signal.aborted) ctrl.abort()
    else signal.addEventListener('abort', onAbort, { once: true })
  }
  return {
    signal: ctrl.signal,
    timedOut: () => timedOut,
    done: () => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
    },
  }
}

function detailToMessage(detail) {
  if (!detail) return null
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    // FastAPI/Pydantic validation errors
    return detail.map((d) => (d?.msg ? d.msg.replace(/^Value error, /, '') : String(d))).join(' ')
  }
  return null
}

export async function request(path, { method = 'GET', body, signal, timeout = 30000 } = {}) {
  const link = linkSignals(signal, timeout)
  let res
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      signal: link.signal,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    })
  } catch (err) {
    link.done()
    if (link.timedOut()) {
      throw new ApiError('The analysis service took too long to respond.', { kind: 'timeout' })
    }
    if (err?.name === 'AbortError') throw err
    throw new ApiError("Can't reach the analysis service. Check your connection or try again shortly.", {
      kind: 'network',
    })
  }
  let data = null
  try {
    data = await res.json()
  } catch (err) {
    link.done()
    if (err?.name === 'AbortError' && !link.timedOut()) throw err
    if (!res.ok) throw new ApiError(`The analysis service returned an error (${res.status}).`, { status: res.status })
    throw new ApiError('The analysis service returned an unreadable response.', { kind: 'schema' })
  }
  link.done()
  if (!res.ok) {
    const ra = Number(res.headers?.get?.('Retry-After'))
    throw new ApiError(detailToMessage(data?.detail) || `The analysis service returned an error (${res.status}).`, {
      status: res.status,
      detail: data?.detail,
      retryAfter: Number.isFinite(ra) && ra > 0 ? ra : null,
      body: data,
    })
  }
  return data
}

function schemaError(what) {
  return new ApiError('The analysis service returned an unexpected response.', { kind: 'schema', detail: what })
}

export function validateStatus(d) {
  if (!d || typeof d.status !== 'string') throw schemaError('model-status')
  return d
}

export function validateSamples(d) {
  if (!d || !Array.isArray(d.samples)) throw schemaError('demo-samples')
  return d.samples.filter((s) => s && typeof s.id === 'string' && (s.split === 'val' || s.split === 'test'))
}

export function validateAssessment(d) {
  const missing = []
  if (!d || typeof d !== 'object') missing.push('body')
  else {
    if (d.status !== 'success') missing.push('status')
    if (!d.images || typeof d.images.rgb !== 'string') missing.push('images.rgb')
    if (!d.prediction || !('flood_coverage_pct' in d.prediction)) missing.push('prediction')
    if (!d.reference || typeof d.reference.available !== 'boolean') missing.push('reference')
    if (!d.assessment || typeof d.assessment.level !== 'string') missing.push('assessment')
    if (!d.mode) missing.push('mode')
  }
  if (missing.length) throw schemaError(missing.join(', '))
  return d
}

export const getModelStatus = (opts) => request('/api/flood/model-status', { timeout: 60000, ...opts }).then(validateStatus)

export const getDemoSamples = (opts) => request('/api/flood/demo-samples', { timeout: 60000, ...opts }).then(validateSamples)

/** /ready answers 503 with the same body when the model is not loaded: return it either way. */
export const getReadiness = ({ deep = false, ...opts } = {}) =>
  request(`/ready${deep ? '?deep=true' : ''}`, { timeout: deep ? 45000 : 30000, ...opts }).catch((e) => {
    if (e instanceof ApiError && e.status === 503 && e.body?.checks) return e.body
    throw e
  })

export const getResearch = (opts) => request('/api/flood/research', { timeout: 30000, ...opts })

/** Large first-time scenes take ~35 s; live mode adds a Copernicus download. */
export const assess = (body, opts) =>
  request('/api/flood/assess', { method: 'POST', body, timeout: 240000, ...opts }).then(validateAssessment)
