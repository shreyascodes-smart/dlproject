import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { Archive, BookOpen, CloudSun, FlaskConical, Moon, RefreshCw, Satellite, Sun, WifiOff } from 'lucide-react'
import FloodPage from './pages/FloodPage'
import { Spinner, Tooltip } from './components/ui'
import ErrorBoundary from './components/ErrorBoundary'
import { ROUTES, useRoute } from './lib/router'
import { useAssessment } from './hooks/useAssessment'
import { usePlaceName, useStoredLocation } from './hooks/useLocation'
import { useCatalogue, useModelStatus } from './hooks/useServer'
import { useTheme } from './hooks/useTheme'

const loadWeather = () => import('./pages/WeatherPage')
const loadDemo = () => import('./pages/DemoPage')
const loadResearch = () => import('./pages/ResearchPage')
const loadMethodology = () => import('./pages/MethodologyPage')
const WeatherPage = lazy(loadWeather)
const DemoPage = lazy(loadDemo)
const ResearchPage = lazy(loadResearch)
const MethodologyPage = lazy(loadMethodology)

const NAV = [
  { route: 'flood', label: 'Flood analysis', short: 'Flood', icon: Satellite },
  { route: 'weather', label: 'Weather & air', short: 'Weather', icon: CloudSun },
  { route: 'demo', label: 'Demo scenes', short: 'Demo', icon: Archive },
  { route: 'research', label: 'Research', short: 'Research', icon: FlaskConical },
  { route: 'methodology', label: 'Methodology', short: 'Method', icon: BookOpen },
]

const TITLES = {
  flood: 'Flood analysis',
  weather: 'Weather & air quality',
  demo: 'Demo scenes',
  research: 'Research dashboard',
  methodology: 'Methodology',
}

function BrandMark() {
  return (
    <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden="true" className="brand__mark">
      <defs>
        <linearGradient id="bm-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--accent)" />
          <stop offset="1" stopColor="var(--accent-2)" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="9" fill="var(--surface-3)" stroke="var(--line-strong)" />
      <path d="M6 19.5c3 0 3-2.5 6-2.5s3 2.5 6 2.5 3-2.5 6-2.5 2 1 2 1" fill="none" stroke="url(#bm-g)" strokeWidth="2.2" strokeLinecap="round" />
      <path d="M6 24.5c3 0 3-2.5 6-2.5s3 2.5 6 2.5 3-2.5 6-2.5 2 1 2 1" fill="none" stroke="url(#bm-g)" strokeOpacity=".45" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="16" cy="9.5" r="3.6" fill="none" stroke="var(--accent)" strokeWidth="1.8" />
      <path d="M9 9.5h3.2M19.8 9.5H23" stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}

function StatusPill({ status }) {
  if (status.state === 'ready') {
    return (
      <span className="pill pill--ok" title="Analysis service connected">
        <span className="pill__dot" aria-hidden="true" /> Online
        <span className="pill__meta">{status.data?.model_version?.replace('segmentation_', 'model ')}</span>
      </span>
    )
  }
  if (status.state === 'loading') {
    return (
      <span className="pill" role="status">
        <Spinner size={12} /> Connecting
      </span>
    )
  }
  return (
    <span className="pill pill--down" role="status">
      <WifiOff size={13} aria-hidden="true" /> {status.state === 'not_ready' ? 'Model unavailable' : 'Service offline'}
      <button type="button" className="icon-btn icon-btn--xs" onClick={status.retry} aria-label="Reconnect to the analysis service">
        <RefreshCw size={12} aria-hidden="true" />
      </button>
    </span>
  )
}

function usePersistent(key, initial, allowed) {
  const [v, setV] = useState(() => {
    try {
      const s = localStorage.getItem(key)
      return s && (!allowed || allowed.includes(s)) ? s : initial
    } catch {
      return initial
    }
  })
  const set = useCallback(
    (next) => {
      setV(next)
      try {
        localStorage.setItem(key, next)
      } catch {
        /* ignore */
      }
    },
    [key],
  )
  return [v, set]
}

const sameLoc = (a, b) => !!a && !!b && Math.abs(a.lat - b.lat) < 1e-6 && Math.abs(a.lng - b.lng) < 1e-6

export default function App() {
  const { theme, toggle } = useTheme()
  const [route, go] = useRoute()
  const status = useModelStatus()
  const [location, setStoredLocation] = useStoredLocation()
  const place = usePlaceName(location)
  const live = useAssessment()
  const demo = useAssessment()
  const catalogue = useCatalogue(route === 'demo')
  const [basemap, setBasemap] = usePersistent('di-basemap', 'minimal', ['minimal', 'streets'])
  const [locating, setLocating] = useState(false)
  const [notice, setNotice] = useState(null)
  const searchRef = useRef(null)
  const mainRef = useRef(null)
  const shownRoute = useRef(route)
  const { reset: resetLive } = live

  /** A new location invalidates a live result for another place (never show a result for the wrong area). */
  const chooseLocation = useCallback(
    (loc) => {
      if (sameLoc(loc, location)) return
      setStoredLocation(loc)
      resetLive()
    },
    [location, setStoredLocation, resetLive],
  )

  const locateMe = useCallback(() => {
    if (!navigator.geolocation) {
      setNotice('Location is not available in this browser. Search for a place instead.')
      return
    }
    setLocating(true)
    navigator.geolocation.getCurrentPosition(
      (p) => {
        setLocating(false)
        chooseLocation({ lat: p.coords.latitude, lng: p.coords.longitude })
      },
      (err) => {
        setLocating(false)
        setNotice(err.code === 1 ? 'Location permission was denied. Search for a place instead.' : 'Could not get your location. Search for a place instead.')
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
    )
  }, [chooseLocation])

  useEffect(() => {
    if (!notice) return undefined
    const t = setTimeout(() => setNotice(null), 6000)
    return () => clearTimeout(t)
  }, [notice])

  // Route changes: update the title, scroll to top and move focus to the page (not on first load)
  useEffect(() => {
    document.title = `${TITLES[route]} · Disaster Insights`
    // Only on an actual route change (not on load, and not on StrictMode's effect re-run)
    if (shownRoute.current === route) return
    shownRoute.current = route
    window.scrollTo?.(0, 0)
    mainRef.current?.focus({ preventScroll: true })
  }, [route])

  // Warm the other pages' code once the app is idle, so first navigation is instant
  useEffect(() => {
    const warm = () => [loadWeather, loadDemo, loadResearch, loadMethodology].forEach((f) => f().catch(() => {}))
    const id = 'requestIdleCallback' in window ? window.requestIdleCallback(warm, { timeout: 4000 }) : setTimeout(warm, 2500)
    return () => ('cancelIdleCallback' in window ? window.cancelIdleCallback(id) : clearTimeout(id))
  }, [])

  // "/" focuses the place search on pages that have one
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target?.tagName) || e.target?.isContentEditable || document.querySelector('dialog[open]')) return
      if (route !== 'flood' && route !== 'weather') return
      e.preventDefault()
      searchRef.current?.focus()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [route])

  // The most recent result (live or demo) feeds the research diagnostics
  const last = [live.data, demo.data].filter(Boolean).sort((a, b) => (b._receivedAt || 0) - (a._receivedAt || 0))[0] || null
  const lastIsLive = last && last === live.data
  const rerun = (method) => {
    if (!last) return
    if (lastIsLive && live.request) live.run({ ...live.request, method })
    else if (demo.request) demo.run({ ...demo.request, method }, { cacheKey: `${demo.request.sample_id}|${method}` })
  }

  const common = { theme, basemap, onBasemap: setBasemap, navigate: go, status }
  const locProps = { location, onLocation: chooseLocation, place, onLocateMe: locateMe, locating, searchRef }

  return (
    <div className={`app app--${route}`}>
      <a
        className="skip-link"
        href="#main"
        onClick={(e) => {
          // Hash routing: move focus without changing the route
          e.preventDefault()
          mainRef.current?.focus()
        }}
      >
        Skip to content
      </a>
      <header className="appbar">
        <a className="brand" href={ROUTES.flood}>
          <BrandMark />
          <span className="brand__text">
            <span className="brand__name">Disaster Insights</span>
            <span className="brand__tag">Satellite flood analysis · research prototype</span>
          </span>
        </a>
        <nav className="topnav" aria-label="Primary">
          {NAV.map(({ route: r, label, icon: Icon }) => (
            <a key={r} href={ROUTES[r]} className={`topnav__link ${route === r ? 'is-active' : ''}`} aria-current={route === r ? 'page' : undefined}>
              <Icon size={15} aria-hidden="true" />
              <span>{label}</span>
            </a>
          ))}
        </nav>
        <div className="appbar__right">
          <StatusPill status={status} />
          <Tooltip content={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>
            <button
              type="button"
              className="icon-btn"
              onClick={toggle}
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              aria-pressed={theme === 'light'}
            >
              {theme === 'dark' ? <Sun size={17} aria-hidden="true" /> : <Moon size={17} aria-hidden="true" />}
            </button>
          </Tooltip>
        </div>
      </header>

      <main id="main" className="main" ref={mainRef} tabIndex={-1}>
        <ErrorBoundary key={route}>
        <Suspense
          fallback={
            <div className="page-loading" role="status">
              <Spinner size={18} /> Loading…
            </div>
          }
        >
          {route === 'flood' && <FloodPage {...common} {...locProps} live={live} />}
          {route === 'weather' && <WeatherPage {...locProps} />}
          {route === 'demo' && <DemoPage {...common} demo={demo} catalogue={catalogue} />}
          {route === 'research' && (
            <ResearchPage
              status={status}
              last={last}
              loading={lastIsLive ? live.status === 'loading' : demo.status === 'loading'}
              startedAt={lastIsLive ? live.startedAt : demo.startedAt}
              onRerun={rerun}
              onShowOnMap={() => go(lastIsLive ? 'flood' : 'demo')}
            />
          )}
          {route === 'methodology' && <MethodologyPage status={status} />}
        </Suspense>
        </ErrorBoundary>
      </main>

      <nav className="tabbar" aria-label="Primary">
        {NAV.map(({ route: r, short, icon: Icon }) => (
          <a key={r} href={ROUTES[r]} className={`tabbar__link ${route === r ? 'is-active' : ''}`} aria-current={route === r ? 'page' : undefined}>
            <Icon size={19} aria-hidden="true" />
            <span>{short}</span>
          </a>
        ))}
      </nav>

      {notice && (
        <div className="toast" role="alert">
          {notice}
        </div>
      )}
    </div>
  )
}
