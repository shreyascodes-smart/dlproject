/**
 * Autocomplete behaviour of LocationSearch with the provider stubbed at the network boundary (Photon shape).
 * Real provider results are checked separately in the browser smoke test.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import LocationSearch, { DEBOUNCE_MS } from './LocationSearch'
import { clearGeocodeCache, fromPhoton, rankSuggestions } from '../lib/geocode'

const feat = (id, name, extra = {}, [lng, lat] = [72.8735453, 19.2061004]) => ({
  type: 'Feature',
  geometry: { type: 'Point', coordinates: [lng, lat] },
  properties: { osm_type: 'W', osm_id: id, osm_key: 'building', osm_value: 'college', name, locality: 'Thakur Village',
    district: 'R/S Ward', city: 'Mumbai', state: 'Maharashtra', country: 'India', ...extra },
})
const THAKUR = { features: [feat(352780866, 'Thakur College of Science & Commerce'), feat(352780851, 'Thakur College of Engineering & Technology')] }
const ok = (body) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) })

function setup(props = {}) {
  const onSelect = vi.fn()
  render(<LocationSearch onSelect={onSelect} onLocateMe={() => {}} {...props} />)
  return { onSelect, box: screen.getByRole('combobox', { name: 'Search for a place' }) }
}
const type = (box, v) => fireEvent.change(box, { target: { value: v } })
const tick = (ms) => act(async () => { await vi.advanceTimersByTimeAsync(ms) })

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  clearGeocodeCache()
})
afterEach(() => vi.useRealTimers())

describe('LocationSearch autocomplete', () => {
  it('debounces typing into ONE request, shows suggestions and selects exact coordinates', async () => {
    globalThis.fetch = vi.fn(() => ok(THAKUR))
    const { box, onSelect } = setup({ near: { lat: 19.18, lng: 72.9 } })
    for (const v of ['T', 'Th', 'Tha', 'Thak', 'Thakur', 'Thakur Col']) {
      type(box, v)
      await tick(50)
    }
    expect(globalThis.fetch).not.toHaveBeenCalled() // still inside the debounce window
    await tick(DEBOUNCE_MS)
    expect(globalThis.fetch).toHaveBeenCalledTimes(1)
    const url = String(globalThis.fetch.mock.calls[0][0])
    expect(url).toMatch(/photon\.komoot\.io\/api\/\?q=Thakur\+Col/)
    expect(url).toMatch(/lat=19\.180&lon=72\.900/) // location bias, not a restriction
    const opts = await screen.findAllByRole('option')
    expect(opts).toHaveLength(2)
    expect(opts[0]).toHaveTextContent('Thakur College of Science & Commerce')
    expect(opts[0]).toHaveTextContent('Thakur Village, R/S Ward, Mumbai, Maharashtra, India')
    fireEvent.click(opts[0])
    expect(onSelect).toHaveBeenCalledWith({ lat: 19.2061004, lng: 72.8735453, name: 'Thakur College of Science & Commerce, Thakur Village, R/S Ward, Mumbai' })
    expect(box).toHaveValue('Thakur College of Science & Commerce')
    expect(screen.queryByRole('option')).not.toBeInTheDocument()
    await tick(DEBOUNCE_MS * 2)
    expect(globalThis.fetch).toHaveBeenCalledTimes(1) // selecting does not search again
  })

  it('does not search below 2 characters', async () => {
    globalThis.fetch = vi.fn(() => ok(THAKUR))
    const { box } = setup()
    type(box, 'T')
    await tick(DEBOUNCE_MS * 2)
    expect(globalThis.fetch).not.toHaveBeenCalled()
  })

  it('keyboard: ↓/↑ move the highlight, Enter selects, Escape closes', async () => {
    globalThis.fetch = vi.fn(() => ok(THAKUR))
    const { box, onSelect } = setup()
    type(box, 'Thakur')
    await tick(DEBOUNCE_MS)
    const opts = await screen.findAllByRole('option')
    expect(opts[0]).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(box, { key: 'ArrowDown' })
    expect(opts[1]).toHaveAttribute('aria-selected', 'true')
    expect(box).toHaveAttribute('aria-activedescendant', opts[1].id)
    fireEvent.keyDown(box, { key: 'ArrowDown' }) // wraps
    fireEvent.keyDown(box, { key: 'ArrowUp' })
    expect(opts[1]).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(box, { key: 'Escape' })
    expect(screen.queryByRole('option')).not.toBeInTheDocument()
    expect(box).toHaveAttribute('aria-expanded', 'false')
    fireEvent.keyDown(box, { key: 'ArrowDown' }) // reopens
    fireEvent.keyDown(box, { key: 'Enter' })
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ lng: 72.8735453 }))
  })

  it('closes on outside click and clears with the clear button', async () => {
    globalThis.fetch = vi.fn(() => ok(THAKUR))
    const { box } = setup()
    type(box, 'Thakur')
    await tick(DEBOUNCE_MS)
    await screen.findAllByRole('option')
    fireEvent.pointerDown(document.body)
    expect(screen.queryByRole('option')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Clear search' }))
    expect(box).toHaveValue('')
    expect(box).toHaveFocus()
  })

  it('shows empty-result and network-error states', async () => {
    globalThis.fetch = vi.fn(() => ok({ features: [] }))
    const { box } = setup()
    type(box, 'zzqqxx')
    await tick(DEBOUNCE_MS)
    expect(await screen.findByText(/No places found for “zzqqxx”/)).toBeInTheDocument()
    globalThis.fetch = vi.fn(() => Promise.reject(new TypeError('Failed to fetch')))
    type(box, 'zzqqxy')
    await tick(DEBOUNCE_MS)
    expect(await screen.findByText(/Place search is unavailable/)).toBeInTheDocument()
  })

  it('a late response for an older query never replaces newer results, and is aborted', async () => {
    const signals = []
    let releaseOld
    globalThis.fetch = vi.fn((url, { signal }) => {
      signals.push(signal)
      if (String(url).includes('q=Ban&')) return new Promise((r) => (releaseOld = () => r({ ok: true, status: 200, json: () => Promise.resolve({ features: [feat(1, 'Banaras')] }) })))
      return ok({ features: [feat(2, 'Bandra West', { osm_key: 'place', osm_value: 'suburb' })] })
    })
    const { box } = setup()
    type(box, 'Ban')
    await tick(DEBOUNCE_MS)
    type(box, 'Bandra')
    await tick(DEBOUNCE_MS)
    expect(await screen.findByRole('option', { name: /Bandra West/ })).toBeInTheDocument()
    expect(signals[0].aborted).toBe(true)
    await act(async () => releaseOld())
    await tick(10)
    expect(screen.queryByRole('option', { name: /Banaras/ })).not.toBeInTheDocument()
    expect(screen.getByRole('option', { name: /Bandra West/ })).toBeInTheDocument()
  })
})

describe('Photon mapping', () => {
  it('maps categories and drops transit-stop duplicates', () => {
    const stop = (id) => feat(id, 'Bandra', { osm_key: 'railway', osm_value: 'stop', locality: 'Khar' })
    const list = [
      feat(10, 'Bandra', { osm_key: 'railway', osm_value: 'station', locality: 'Khar' }),
      stop(11), stop(12), stop(13),
      feat(14, 'Bandra West', { osm_key: 'place', osm_value: 'suburb', locality: undefined }),
      feat(15, 'Chhatrapati Shivaji Maharaj International Airport', { osm_key: 'aeroway', osm_value: 'aerodrome' }),
    ].map(fromPhoton)
    const r = rankSuggestions(list)
    expect(r.map((s) => [s.name, s.kind])).toEqual([
      ['Bandra', 'rail'],
      ['Bandra West', 'area'],
      ['Chhatrapati Shivaji Maharaj International Airport', 'airport'],
    ])
    expect(fromPhoton({ properties: { name: 'x' }, geometry: { coordinates: [] } })).toBeNull()
  })
})
