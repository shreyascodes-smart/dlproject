import { useCallback, useEffect, useRef, useState } from 'react'
import { assess } from '../lib/api'

const BROWSER_CACHE_LIMIT = 8

/**
 * Runs flood assessments.
 *
 * - Only the latest request may update state: older in-flight requests are
 *   aborted and their results ignored (no stale overwrites).
 * - Demo results are kept in a small in-browser LRU cache so switching back
 *   to a scene is instant. `data._source` records where a result came from:
 *   'computed' | 'server-cache' | 'browser'.
 * - While loading, the previous result stays in `data` so the viewer can
 *   show it dimmed behind the progress overlay.
 */
export function useAssessment() {
  const [state, setState] = useState({ status: 'idle', data: null, error: null, request: null, startedAt: null })
  const ctrlRef = useRef(null)
  const seqRef = useRef(0)
  const cacheRef = useRef(new Map())

  const run = useCallback(async (body, { cacheKey } = {}) => {
    ctrlRef.current?.abort()
    const seq = ++seqRef.current
    const cache = cacheRef.current

    if (cacheKey && cache.has(cacheKey)) {
      const hit = cache.get(cacheKey)
      cache.delete(cacheKey)
      cache.set(cacheKey, hit)
      setState({ status: 'success', data: { ...hit, _source: 'browser' }, error: null, request: body, startedAt: null })
      return
    }

    const ctrl = new AbortController()
    ctrlRef.current = ctrl
    setState((s) => ({ ...s, status: 'loading', error: null, request: body, startedAt: Date.now() }))
    try {
      const data = await assess(body, { signal: ctrl.signal })
      if (seq !== seqRef.current) return
      const tagged = { ...data, _source: data.cached ? 'server-cache' : 'computed', _receivedAt: Date.now() }
      if (cacheKey) {
        cache.set(cacheKey, tagged)
        while (cache.size > BROWSER_CACHE_LIMIT) cache.delete(cache.keys().next().value)
      }
      setState({ status: 'success', data: tagged, error: null, request: body, startedAt: null })
    } catch (err) {
      if (seq !== seqRef.current || err?.name === 'AbortError') return
      setState({ status: 'error', data: null, error: err, request: body, startedAt: null })
    }
  }, [])

  const reset = useCallback(() => {
    ctrlRef.current?.abort()
    seqRef.current += 1
    setState({ status: 'idle', data: null, error: null, request: null, startedAt: null })
  }, [])

  useEffect(() => () => ctrlRef.current?.abort(), [])

  return { ...state, run, reset }
}
