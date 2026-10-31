import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchAir, fetchForecast, TTL_MS } from '../lib/weather'

const STALE_MS = 30 * 60 * 1000

/**
 * One independent data source for the selected location (weather OR air quality).
 * - Results are tagged with the location they belong to: data for an old location is never shown.
 * - Late responses for a previous location are ignored.
 * - Re-fetches automatically when the tab becomes visible and the data is older than 30 min.
 */
function useSource(fetcher, location, enabled = true) {
  const key = location ? `${location.lat.toFixed(4)},${location.lng.toFixed(4)}` : null
  const [state, setState] = useState({ key: null, status: 'idle', data: null, error: null, fetchedAt: null })
  const [nonce, setNonce] = useState({ n: 0, force: false })
  const ctrl = useRef(null)

  useEffect(() => {
    if (!key || !enabled) return undefined
    ctrl.current?.abort()
    const c = new AbortController()
    ctrl.current = c
    setState((s) => (s.key === key ? { ...s, status: s.data ? 'refreshing' : 'loading', error: null } : { key, status: 'loading', data: null, error: null, fetchedAt: null }))
    fetcher(location, c.signal, { force: nonce.force })
      .then((r) => {
        if (c.signal.aborted) return
        setState({ key, status: 'success', data: r.data, error: null, fetchedAt: r.fetchedAt })
      })
      .catch((e) => {
        if (c.signal.aborted || e?.name === 'AbortError') return
        const msg = e?.kind ? e.message : 'The service could not be reached. Check your connection and try again.'
        setState((s) => ({ ...(s.key === key ? s : { data: null, fetchedAt: null }), key, status: s.key === key && s.data ? 'stale-error' : 'error', error: msg }))
      })
    return () => c.abort()
    // location object identity is irrelevant: key captures lat/lng
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled, nonce])

  useEffect(() => {
    if (!enabled) return undefined
    const onVis = () => {
      if (document.visibilityState === 'visible' && state.fetchedAt && Date.now() - state.fetchedAt > STALE_MS) {
        setNonce((n) => ({ n: n.n + 1, force: true }))
      }
    }
    document.addEventListener('visibilitychange', onVis)
    return () => document.removeEventListener('visibilitychange', onVis)
  }, [enabled, state.fetchedAt])

  const refresh = useCallback(() => setNonce((n) => ({ n: n.n + 1, force: true })), [])
  const current = state.key === key ? state : { key, status: key && enabled ? 'loading' : 'idle', data: null, error: null, fetchedAt: null }
  const stale = !!current.fetchedAt && Date.now() - current.fetchedAt > STALE_MS
  return { ...current, stale, refresh, ttlMs: TTL_MS }
}

export function useWeather(location, enabled) {
  return useSource(fetchForecast, location, enabled)
}

export function useAirQuality(location, enabled) {
  return useSource(fetchAir, location, enabled)
}
