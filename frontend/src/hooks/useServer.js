import { useCallback, useEffect, useRef, useState } from 'react'
import { getDemoSamples, getModelStatus } from '../lib/api'

const OFFLINE_RETRY_MS = 20000

/**
 * Backend model status. While the service is offline it is re-checked every 20 s and when the
 * window regains focus, so the UI recovers on its own once the backend is back.
 */
export function useModelStatus() {
  const [state, setState] = useState({ state: 'loading', data: null, error: null })
  const [nonce, setNonce] = useState(0)
  useEffect(() => {
    const ctrl = new AbortController()
    getModelStatus({ signal: ctrl.signal })
      .then((d) => setState({ state: d.status === 'ready' ? 'ready' : 'not_ready', data: d, error: null }))
      .catch((e) => e?.name !== 'AbortError' && setState({ state: 'offline', data: null, error: e }))
    return () => ctrl.abort()
  }, [nonce])

  const retry = useCallback(() => {
    setState((s) => ({ ...s, state: s.data ? s.state : 'loading', error: null }))
    setNonce((n) => n + 1)
  }, [])

  const down = state.state === 'offline' || state.state === 'not_ready'
  useEffect(() => {
    if (!down) return undefined
    const id = setInterval(() => setNonce((n) => n + 1), OFFLINE_RETRY_MS)
    const onFocus = () => setNonce((n) => n + 1)
    window.addEventListener('focus', onFocus)
    return () => {
      clearInterval(id)
      window.removeEventListener('focus', onFocus)
    }
  }, [down])

  return { ...state, retry }
}

/** Demo catalogue, fetched only when first needed (the demo page) and kept afterwards. */
export function useCatalogue(enabled) {
  const [state, setState] = useState({ state: 'idle', samples: [], error: null })
  const [nonce, setNonce] = useState(0)
  const loaded = useRef(-1)
  useEffect(() => {
    if (!enabled || loaded.current === nonce) return undefined
    loaded.current = nonce
    const ctrl = new AbortController()
    let done = false
    setState((s) => ({ ...s, state: 'loading', error: null }))
    getDemoSamples({ signal: ctrl.signal })
      .then((samples) => {
        done = true
        setState({ state: 'success', samples, error: null })
      })
      .catch((e) => {
        if (e?.name === 'AbortError') return
        done = true
        setState({ state: 'error', samples: [], error: e })
      })
    return () => {
      // Leaving the page before the list arrived: allow a fresh request next time
      if (!done) {
        loaded.current = -1
        ctrl.abort()
      }
    }
  }, [enabled, nonce])
  const retry = useCallback(() => setNonce((n) => n + 1), [])
  return { ...state, retry }
}
