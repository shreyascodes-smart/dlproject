import { useCallback, useEffect, useId, useRef, useState } from 'react'
import {
  Building,
  GraduationCap,
  Hospital,
  House,
  Landmark,
  LocateFixed,
  MapPin,
  MapPinned,
  Mountain,
  Plane,
  Route,
  Search,
  TrainFront,
  TriangleAlert,
  X,
} from 'lucide-react'
import { autocomplete, autocompleteProvider } from '../lib/geocode'
import { Spinner } from './ui'

export const DEBOUNCE_MS = 300
export const MIN_CHARS = 2

const KIND_ICON = {
  airport: Plane,
  rail: TrainFront,
  education: GraduationCap,
  hospital: Hospital,
  road: Route,
  landmark: Landmark,
  nature: Mountain,
  city: Building,
  village: House,
  area: MapPinned,
  region: MapPinned,
  place: MapPin,
}

/** Bold the part of the name that matches the typed text (display only). */
function Highlight({ text, query }) {
  const q = query.trim().toLowerCase()
  const i = q ? text.toLowerCase().indexOf(q) : -1
  if (i < 0) return text
  return (
    <>
      {text.slice(0, i)}
      <mark>{text.slice(i, i + q.length)}</mark>
      {text.slice(i + q.length)}
    </>
  )
}

/**
 * Place autocomplete (combobox + listbox, WAI-ARIA 1.2 pattern) + "Use my location".
 *
 * - Debounced (300 ms) provider requests from 2 characters; each new request aborts the previous one
 *   and a sequence number discards late responses, so older results never overwrite newer ones.
 * - ↑/↓ move the highlight, Enter selects (or searches immediately if results are still pending),
 *   Escape closes, Tab / clicking outside closes. Touch: tap a suggestion.
 * - Selecting passes the provider's exact coordinates to onSelect; nothing is analysed automatically.
 * - The list is absolutely positioned inside the search box, so it scrolls with the input inside the
 *   side panel and overlays (never pushes) the content below.
 */
export default function LocationSearch({ onSelect, onLocateMe, locating, inputRef, hint = 'or click the map', id, near }) {
  const [q, setQ] = useState('')
  const [state, setState] = useState({ status: 'idle', query: '', results: [], error: null })
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(-1)
  const rootRef = useRef(null)
  const listRef = useRef(null)

  const timer = useRef(0)
  const ctrl = useRef(null)
  const seq = useRef(0)
  const nearRef = useRef(near)
  const listId = useId()
  useEffect(() => {
    nearRef.current = near
  }, [near])

  const cancel = useCallback(() => {
    clearTimeout(timer.current)
    ctrl.current?.abort()
    seq.current += 1
  }, [])

  useEffect(() => cancel, [cancel])

  const run = useCallback(async (text) => {
    const query = text.trim()
    ctrl.current?.abort()
    const n = ++seq.current
    if (query.length < MIN_CHARS) {
      setState({ status: 'idle', query: '', results: [], error: null })
      return
    }
    const c = new AbortController()
    ctrl.current = c
    setState((s) => ({ ...s, status: 'loading', query }))
    try {
      const results = await autocomplete(query, { signal: c.signal, near: nearRef.current })
      if (n !== seq.current) return // a newer query was typed meanwhile
      setState({ status: 'done', query, results, error: null })
      setActive(results.length ? 0 : -1)
    } catch (err) {
      if (err?.name === 'AbortError' || n !== seq.current) return
      setState({ status: 'error', query, results: [], error: 'Place search is unavailable right now. Check your connection, use your location, or click the map.' })
      setActive(-1)
    }
  }, [])

  const onChange = (e) => {
    const v = e.target.value
    setQ(v)
    setOpen(true)
    clearTimeout(timer.current)
    if (v.trim().length < MIN_CHARS) {
      run('') // cancels any pending request and clears old suggestions
      return
    }
    timer.current = setTimeout(() => run(v), DEBOUNCE_MS)
  }

  const choose = (r) => {
    cancel()
    onSelect({ lat: r.lat, lng: r.lng, name: r.label || r.name })
    setQ(r.name)
    setOpen(false)
    setActive(-1)
  }

  const clear = () => {
    cancel()
    setQ('')
    setState({ status: 'idle', query: '', results: [], error: null })
    setOpen(false)
    setActive(-1)
    rootRef.current?.querySelector('input')?.focus()
  }

  // Close on a pointer press anywhere outside the search box
  useEffect(() => {
    if (!open) return undefined
    const onDown = (e) => {
      if (!rootRef.current?.contains(e.target)) setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [open])

  // Keep the highlighted option visible inside the list (the list scrolls, not the side panel)
  useEffect(() => {
    if (active < 0 || !listRef.current) return
    const el = listRef.current.querySelector(`[data-index="${active}"]`)
    if (!el) return
    const list = listRef.current
    if (el.offsetTop < list.scrollTop) list.scrollTop = el.offsetTop
    else if (el.offsetTop + el.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = el.offsetTop + el.offsetHeight - list.clientHeight
  }, [active])

  const tooShort = q.trim().length < MIN_CHARS
  const results = state.results
  const fresh = state.query === q.trim()
  const showList = open && !tooShort && (state.status !== 'idle' || results.length > 0)

  const onKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (!open) setOpen(true)
      if (results.length) setActive((a) => (a + 1) % results.length)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      if (results.length) setActive((a) => (a <= 0 ? results.length - 1 : a - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (showList && fresh && results[active]) choose(results[active])
      else if (!tooShort) {
        // Results pending or stale: search now instead of waiting for the debounce
        clearTimeout(timer.current)
        setOpen(true)
        run(q)
      }
    } else if (e.key === 'Escape') {
      if (open) {
        e.preventDefault()
        setOpen(false)
      }
    } else if (e.key === 'Tab') {
      setOpen(false)
    }
  }

  const loading = state.status === 'loading'
  const optId = (i) => `${listId}-opt-${i}`

  return (
    <div className="lsearch" ref={rootRef}>
      <div className={`lsearch__field ${showList ? 'is-open' : ''}`}>
        <Search size={16} aria-hidden="true" />
        <label className="sr-only" htmlFor={id || `${listId}-input`}>
          Search for a place
        </label>
        <input
          ref={inputRef}
          id={id || `${listId}-input`}
          type="text"
          value={q}
          onChange={onChange}
          onKeyDown={onKeyDown}
          onFocus={() => !tooShort && results.length > 0 && setOpen(true)}
          placeholder="Search a place, address or landmark"
          autoComplete="off"
          autoCorrect="off"
          spellCheck={false}
          enterKeyHint="search"
          role="combobox"
          aria-autocomplete="list"
          aria-expanded={showList}
          aria-controls={listId}
          aria-activedescendant={showList && fresh && results[active] ? optId(active) : undefined}
        />
        {loading ? (
          <Spinner size={15} label="Searching" />
        ) : (
          q && (
            <button type="button" className="lsearch__clear" onClick={clear} aria-label="Clear search">
              <X size={14} aria-hidden="true" />
            </button>
          )
        )}
      </div>

      <div id={listId} className="lsearch__menu" role="listbox" aria-label="Place suggestions" hidden={!showList} ref={listRef}>
        {showList && state.status === 'error' && (
          <p className="lsearch__state lsearch__state--error" role="presentation">
            <TriangleAlert size={14} aria-hidden="true" /> {state.error}
          </p>
        )}
        {showList && state.status === 'done' && results.length === 0 && (
          <p className="lsearch__state" role="presentation">
            No places found for “{state.query}”. Try a different spelling or a nearby town.
          </p>
        )}
        {showList && loading && results.length === 0 && (
          <p className="lsearch__state" role="presentation">
            <Spinner size={13} /> Searching…
          </p>
        )}
        {showList &&
          results.map((r, i) => {
            const Icon = KIND_ICON[r.kind] || MapPin
            return (
              <div
                key={r.id}
                id={optId(i)}
                data-index={i}
                role="option"
                aria-selected={i === active}
                className={`lsearch__opt ${i === active ? 'is-active' : ''} ${!fresh ? 'is-stale' : ''}`}
                onMouseEnter={() => setActive(i)}
                onMouseDown={(e) => e.preventDefault() /* keep focus in the input */}
                onClick={() => choose(r)}
              >
                <span className={`lsearch__icon lsearch__icon--${r.kind}`} aria-hidden="true">
                  <Icon size={15} />
                </span>
                <span className="lsearch__text">
                  <span className="lsearch__name">
                    <Highlight text={r.name} query={state.query} />
                  </span>
                  {r.secondary && <span className="lsearch__sub">{r.secondary}</span>}
                </span>
                <span className="lsearch__cat">{r.category}</span>
              </div>
            )
          })}
        {showList && (results.length > 0 || state.status === 'done') && (
          <p className="lsearch__attr" role="presentation">
            {autocompleteProvider().attribution}
          </p>
        )}
      </div>

      <p className="sr-only" role="status" aria-live="polite">
        {showList && state.status === 'done' ? `${results.length} ${results.length === 1 ? 'suggestion' : 'suggestions'}` : ''}
      </p>

      <div className="lsearch__alt">
        <button type="button" className="link-btn" onClick={onLocateMe} disabled={locating}>
          {locating ? <Spinner size={13} /> : <LocateFixed size={14} aria-hidden="true" />} Use my location
        </button>
        {hint && <span className="muted small">{hint}</span>}
      </div>
    </div>
  )
}
