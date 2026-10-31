import { useCallback, useEffect, useState } from 'react'

/** Hash routes (no server config needed for static hosting). */
export const ROUTES = {
  flood: '#/',
  weather: '#/weather',
  demo: '#/demo',
  research: '#/research',
  methodology: '#/methodology',
}

export function routeFromHash(hash) {
  const h = (hash || '').replace(/\/+$/, '')
  if (h === '#/weather') return 'weather'
  if (h === '#/demo') return 'demo'
  if (h === '#/research') return 'research'
  if (h === '#/methodology') return 'methodology'
  return 'flood'
}

export function useRoute() {
  const [route, setRoute] = useState(() => routeFromHash(window.location.hash))
  useEffect(() => {
    const on = () => setRoute(routeFromHash(window.location.hash))
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  const go = useCallback((name) => {
    const target = ROUTES[name] || ROUTES.flood
    if (window.location.hash !== target) window.location.hash = target
    else setRoute(name)
  }, [])
  return [route, go]
}
