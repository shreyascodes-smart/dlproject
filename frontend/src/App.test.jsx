/**
 * Integration tests for the Milestone 5 app. fetch is mocked at the network boundary with the
 * same JSON shapes the FastAPI backend and Open-Meteo return.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import { clearWeatherCache } from './lib/weather'
import { airResponse, assessment, forecastResponse, liveAssessment, samples, status } from './test/fixtures'

const json = (body, ok = true, code = 200, headers = {}) =>
  Promise.resolve({ ok, status: code, json: () => Promise.resolve(body), headers: { get: (k) => headers[k] ?? null } })

// Photon (autocomplete) GeoJSON, same shape as photon.komoot.io/api
const photon = (id, name, state, country, lng, lat, key = 'place', value = 'city') => ({
  type: 'Feature',
  geometry: { type: 'Point', coordinates: [lng, lat] },
  properties: { osm_type: 'R', osm_id: id, osm_key: key, osm_value: value, name, state, country },
})
const PLACES = {
  type: 'FeatureCollection',
  features: [photon(1, 'Patna', 'Bihar', 'India', 85.14, 25.61), photon(2, 'Dhaka', 'Dhaka Division', 'Bangladesh', 90.41, 23.81)],
}

const READY = { ready: true, checks: { api: 'ok', segmentation_checkpoint: 'present', model: 'loaded', model_version: 'segmentation_v2', outcome_policy: 'present', satellite_credentials: 'configured' }, error: null }
const RESEARCH = {
  active_model: 'segmentation_v2',
  policy: { version: 'segmentation_v2-val-20261028', none_below_pct: 1, inconclusive_ceiling_pct: { tiled: 6.5, resize: 4.2 }, min_assessable_land_frac: 0.25, water_dominated_frac: 0.5 },
  protocol: 'Full-resolution inference on the frozen event-disjoint splits.',
  models: [
    { version: 'segmentation_v1', installed: true, active: false, training: { encoder_name: 'efficientnet-b2' }, val: { tiled: { pixel_metrics_excluding_known_water: { flood_iou: 0.2524, flood_precision: 0.4, flood_recall: 0.4, flood_dice: 0.4 } } }, val_n_samples: 24 },
    { version: 'segmentation_v2', installed: true, active: true, training: { bands: ['B02', 'B03'] }, val: { tiled: { pixel_metrics_excluding_known_water: { flood_iou: 0.4512, flood_precision: 0.5, flood_recall: 0.7, flood_dice: 0.6 } } }, val_n_samples: 24 },
  ],
}

function mockBackend({ statusBody = status, assess, weather = 'ok', air = 'ok' } = {}) {
  const calls = { assess: [], weather: 0, air: 0, samples: 0 }
  globalThis.fetch = vi.fn((url, opts = {}) => {
    const u = String(url)
    if (u.endsWith('/api/flood/model-status')) return json(statusBody)
    if (u.endsWith('/api/flood/demo-samples')) {
      calls.samples += 1
      return json(samples)
    }
    if (u.endsWith('/api/flood/research')) return json(RESEARCH)
    if (u.startsWith('/ready')) return json(READY)
    if (u.endsWith('/api/flood/assess')) {
      const body = JSON.parse(opts.body)
      calls.assess.push(body)
      if (assess) return assess(body, opts)
      if (body.mode === 'live') return json(liveAssessment({ lat: body.latitude, lng: body.longitude }))
      return json(assessment(samples.samples.find((s) => s.id === body.sample_id), body.method))
    }
    if (u.includes('photon.komoot.io/api')) return json(PLACES)
    if (u.includes('nominatim.openstreetmap.org/reverse')) return json({ address: { city: 'Somewhere', country: 'Testland' } })
    if (u.includes('air-quality-api.open-meteo.com')) {
      calls.air += 1
      return air === 'ok' ? json(airResponse()) : json({}, false, 500)
    }
    if (u.includes('api.open-meteo.com')) {
      calls.weather += 1
      return weather === 'ok' ? json(forecastResponse()) : json({}, false, 500)
    }
    return Promise.reject(new Error(`unexpected ${u}`))
  })
  return calls
}

async function pickPlace(user, name = /Patna/) {
  const box = screen.getByRole('combobox', { name: /Search for a place/ })
  await user.clear(box)
  await user.type(box, 'Pa')
  await user.click(await screen.findByRole('option', { name }))
}

const primary = () => screen.getByRole('button', { name: /Analyse this area|Analyse again|Analysing/ })

async function go(user, name) {
  const nav = screen.getAllByRole('navigation', { name: 'Primary' })[0]
  await user.click(within(nav).getByRole('link', { name }))
}

const BANNED = [
  'This is a preliminary estimate from an experimental model, not a confirmed flood event or an official warning.',
  'Preliminary model estimate.',
  'Results appear here. They show what the model sees in one satellite image — not an official warning.',
  'What you can do',
]

beforeEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
  clearWeatherCache()
  window.location.hash = ''
  document.documentElement.removeAttribute('data-theme')
})

afterEach(() => {
  window.location.hash = ''
})

describe('App shell and navigation', () => {
  it('has primary navigation to every page and marks the current one', async () => {
    const user = userEvent.setup()
    mockBackend()
    render(<App />)
    const nav = screen.getAllByRole('navigation', { name: 'Primary' })[0]
    for (const name of ['Flood analysis', 'Weather & air', 'Demo scenes', 'Research', 'Methodology']) {
      expect(within(nav).getByRole('link', { name })).toBeInTheDocument()
    }
    expect(within(nav).getByRole('link', { name: 'Flood analysis' })).toHaveAttribute('aria-current', 'page')
    await go(user, 'Methodology')
    expect(await screen.findByRole('heading', { name: 'How Disaster Insights works' })).toBeInTheDocument()
    expect(within(nav).getByRole('link', { name: 'Methodology' })).toHaveAttribute('aria-current', 'page')
    expect(document.title).toMatch(/Methodology/)
  })

  it('shows the service status and recovers from offline', async () => {
    globalThis.fetch = vi.fn(() => Promise.reject(new TypeError('Failed to fetch')))
    render(<App />)
    expect(await screen.findByText('Service offline')).toBeInTheDocument()
    expect(screen.getByText(/analysis service is unavailable right now/)).toBeInTheDocument()
    mockBackend()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Reconnect to the analysis service' }))
    expect(await screen.findByText('Online')).toBeInTheDocument()
  })

  it('switches between dark and light and remembers the choice', async () => {
    const user = userEvent.setup()
    mockBackend()
    render(<App />)
    const html = document.documentElement
    expect(html.dataset.theme).toBe('dark')
    await user.click(screen.getByRole('button', { name: 'Switch to light mode' }))
    expect(html.dataset.theme).toBe('light')
    expect(localStorage.getItem('di-theme')).toBe('light')
    await user.click(screen.getByRole('button', { name: 'Switch to dark mode' }))
    expect(html.dataset.theme).toBe('dark')
  })
})

describe('Flood analysis', () => {
  it('keeps the controls outside the scrolling results area', async () => {
    mockBackend()
    render(<App />)
    const controls = screen.getByRole('heading', { name: 'Flood analysis' }).closest('section')
    const results = screen.getByRole('region', { name: 'Results' })
    expect(controls).toHaveClass('panel__controls')
    expect(results).toHaveClass('panel__scroll')
    expect(results.contains(controls)).toBe(false)
    expect(controls.contains(screen.getByRole('combobox', { name: /Search for a place/ }))).toBe(true)
  })

  it('requires a location, then analyses it and presents the result in reading order', async () => {
    const user = userEvent.setup()
    const calls = mockBackend()
    render(<App />)
    await waitFor(() => expect(screen.getByText('Choose a location to analyse.')).toBeInTheDocument())
    expect(primary()).toBeDisabled()
    await pickPlace(user)
    await waitFor(() => expect(primary()).toBeEnabled())
    await user.click(primary())
    await waitFor(() => expect(calls.assess).toHaveLength(1))
    expect(calls.assess[0]).toEqual({ mode: 'live', latitude: 25.61, longitude: 85.14, method: 'tiled' })

    const result = await screen.findByRole('article')
    // 1 location + time, 2 outcome, 3 coverage, 4 image, 5 map, 6 context, 7 method
    const order = ['Patna, Bihar', 'Outcome', 'Coverage', 'Image', 'Map', 'Environmental context', 'How this result was produced']
    const text = result.textContent
    const idx = order.map((s) => text.indexOf(s))
    expect(idx.every((i) => i >= 0)).toBe(true)
    expect([...idx].sort((a, b) => a - b)).toEqual(idx)
    expect(within(result).getByText(/Analysed/)).toBeInTheDocument()
    expect(within(result).getByRole('heading', { name: 'Inconclusive' })).toBeInTheDocument()
    // coverage states its denominator
    expect(within(result).getByText('4.7%')).toBeInTheDocument()
    expect(within(result).getByText(/of 21\.4 km² valid land/)).toBeInTheDocument()
    expect(within(result).getByText(/10,000 of 213,750 pixels/)).toBeInTheDocument()
    expect(within(result).getByText('18 Oct 2026')).toBeInTheDocument()
    expect(within(result).getByText(/8 days old/)).toBeInTheDocument()
    expect(within(result).getByText('Sentinel-2 scene classification (Sen2Cor)')).toBeInTheDocument()
  })

  it('never shows banned copy, probabilities, risk scores or "safe" wording', async () => {
    const user = userEvent.setup()
    mockBackend({ assess: (b) => json(liveAssessment({ lat: b.latitude, lng: b.longitude }, 'none_detected')) })
    render(<App />)
    await pickPlace(user)
    await user.click(primary())
    await screen.findByRole('heading', { name: 'No flood-like patterns detected' })
    const panel = document.querySelector('.workspace__panel').textContent
    for (const b of BANNED) expect(panel).not.toContain(b)
    const checked = panel.replace(/not a probability, confidence or risk/gi, '').replace(/chance/g, '')
    expect(checked).not.toMatch(/probability|risk score|high risk|low risk|\bsafe\b|confidence/i)
  })

  it.each([
    ['flood_like', 'Flood-like patterns detected', true],
    ['uncertain_water', 'Inconclusive — mostly permanent water', false],
    ['insufficient_coverage', 'Insufficient valid coverage', false],
  ])('shows the %s outcome', async (level, headline, hasShare) => {
    const user = userEvent.setup()
    mockBackend({ assess: (b) => json(liveAssessment({ lat: b.latitude, lng: b.longitude }, level)) })
    render(<App />)
    await pickPlace(user)
    await user.click(primary())
    const result = await screen.findByRole('article')
    expect(within(result).getByRole('heading', { name: headline })).toBeInTheDocument()
    expect(within(result).queryByText(/of the valid land analysed is flood-like water/) != null).toBe(hasShare)
    if (!hasShare) expect(within(result).getByText(/No flood-like share is reported for this outcome/)).toBeInTheDocument()
  })

  it.each([
    [404, { detail: 'No Sentinel-2 acquisition found for this area in the last 45 days. Try again later.' }, 'Imagery unavailable', /last 45 days/],
    [422, { detail: 'The latest image covers only 12% of this area.' }, 'Insufficient valid coverage', /covers only 12%/],
    [502, { detail: 'The Copernicus satellite service could not provide imagery right now.' }, 'Satellite service unavailable', /Copernicus/],
    [503, { detail: 'The segmentation_v2 checkpoint is not installed on this server.' }, 'Service unavailable', /not installed/],
    [429, { detail: 'Too many assessment requests.' }, 'Too many requests', /about 30 s/],
  ])('maps HTTP %s to an outcome state with a retry', async (code, body, title, msg) => {
    const user = userEvent.setup()
    let fail = true
    mockBackend({ assess: (b) => (fail ? json(body, false, code, { 'Retry-After': '30' }) : json(liveAssessment({ lat: b.latitude, lng: b.longitude }))) })
    render(<App />)
    await pickPlace(user)
    await user.click(primary())
    const alert = await screen.findByRole('alert')
    expect(within(alert).getByRole('heading', { name: title })).toBeInTheDocument()
    expect(within(alert).getByText(msg)).toBeInTheDocument()
    fail = false
    await user.click(within(alert).getByRole('button', { name: /Try again/ }))
    expect(await screen.findByRole('article')).toBeInTheDocument()
  })

  it('reports an unreachable service without leaking the API address', async () => {
    const user = userEvent.setup()
    mockBackend({ assess: () => Promise.reject(new TypeError('Failed to fetch')) })
    render(<App />)
    await pickPlace(user)
    await user.click(primary())
    const alert = await screen.findByRole('alert')
    expect(within(alert).getByRole('heading', { name: 'Analysis service unreachable' })).toBeInTheDocument()
    expect(alert.textContent).not.toMatch(/localhost|127\.0\.0\.1|:8000/)
  })

  it('clears the result when the location changes and ignores late responses', async () => {
    const user = userEvent.setup()
    let release
    let slow = false
    mockBackend({
      assess: (b) =>
        slow
          ? new Promise((resolve) => {
              release = () => resolve({ ok: true, status: 200, headers: { get: () => null }, json: () => Promise.resolve(liveAssessment({ lat: b.latitude, lng: b.longitude }, 'flood_like')) })
            })
          : json(liveAssessment({ lat: b.latitude, lng: b.longitude })),
    })
    render(<App />)
    await pickPlace(user)
    await user.click(primary())
    await screen.findByRole('article')
    await pickPlace(user, /Dhaka/)
    expect(screen.queryByRole('article')).not.toBeInTheDocument()
    expect(primary()).toHaveTextContent('Analyse this area')

    slow = true
    await user.click(primary())
    await waitFor(() => expect(release).toBeTypeOf('function'))
    await pickPlace(user, /Patna/)
    await act(async () => {
      release()
      await new Promise((r) => setTimeout(r, 20))
    })
    expect(screen.queryByRole('heading', { name: 'Flood-like patterns detected' })).not.toBeInTheDocument()
  })

  it('remembers the selected location across reloads', async () => {
    const user = userEvent.setup()
    mockBackend()
    const { unmount } = render(<App />)
    await pickPlace(user)
    expect(JSON.parse(localStorage.getItem('di-location'))).toMatchObject({ lat: 25.61, lng: 85.14 })
    unmount()
    render(<App />)
    expect(await screen.findByText('Patna, Bihar')).toBeInTheDocument()
  })

  it('points to the demo scenes when live imagery is not configured', async () => {
    const user = userEvent.setup()
    mockBackend({ statusBody: { ...status, satellite: 'not_configured' } })
    render(<App />)
    await pickPlace(user)
    expect(await screen.findByText(/Live satellite imagery is not configured/)).toBeInTheDocument()
    expect(primary()).toBeDisabled()
    await user.click(screen.getByRole('link', { name: 'Explore archived demo scenes' }))
    expect(await screen.findByRole('heading', { name: 'Demo scenes' })).toBeInTheDocument()
  })

  it('shows a weather and air-quality snapshot that is labelled as context', async () => {
    const user = userEvent.setup()
    mockBackend()
    render(<App />)
    await pickPlace(user)
    const ctx = await screen.findByRole('region', { name: 'Environmental context' })
    expect(within(ctx).getByText(/not used by the flood model/)).toBeInTheDocument()
    expect(await within(ctx).findByText('166')).toBeInTheDocument()
    expect(within(ctx).getByText('89')).toBeInTheDocument()
    expect(within(ctx).getByText('US AQI')).toBeInTheDocument()
    expect(within(ctx).getByText('European AQI')).toBeInTheDocument()
  })
})

describe('Weather & air quality page', () => {
  it('renders current conditions, hourly, daily and both AQI scales separately', async () => {
    const user = userEvent.setup()
    mockBackend()
    localStorage.setItem('di-location', JSON.stringify({ lat: 25.61, lng: 85.14, name: 'Patna, Bihar, India' }))
    render(<App />)
    await go(user, 'Weather & air')
    expect(await screen.findByRole('heading', { name: 'Patna, Bihar' })).toBeInTheDocument()
    const now = (await screen.findByRole('heading', { name: 'Current conditions' })).closest('section')
    expect(await within(now).findByText('Overcast')).toBeInTheDocument()
    for (const label of ['Humidity', 'Wind', 'Gusts', 'Pressure', 'Visibility', 'UV index', 'Sunrise', 'Sunset']) {
      expect(within(now).getByText(label)).toBeInTheDocument()
    }
    expect(within(now).getByText('7.9 km')).toBeInTheDocument()
    expect(within(now).getByText('05:55')).toBeInTheDocument()
    expect(within(now).getByText(/from N/)).toBeInTheDocument()

    const air = screen.getByRole('heading', { name: 'Air quality' }).closest('section')
    expect(within(air).getByText('US AQI')).toBeInTheDocument()
    expect(within(air).getByText('European AQI')).toBeInTheDocument()
    expect(within(air).getByText('Unhealthy')).toBeInTheDocument()
    expect(within(air).getByText('Very poor')).toBeInTheDocument()
    expect(within(air).getByText(/not comparable/)).toBeInTheDocument()

    const daily = screen.getByRole('heading', { name: '7-day forecast' }).closest('section')
    expect(within(daily).getAllByRole('row')).toHaveLength(8) // header + 7 days from today
    expect(within(daily).getByRole('rowheader', { name: 'Today' })).toBeInTheDocument()

    const pollutants = screen.getByRole('heading', { name: 'Pollutants' }).closest('section')
    expect(within(pollutants).getByText('PM2.5')).toBeInTheDocument()
    expect(within(pollutants).getAllByText('μg/m³').length).toBeGreaterThan(3)
    expect(within(pollutants).getByText(/only available for Europe/)).toBeInTheDocument()

    expect(screen.getByRole('group', { name: /Hourly temperature chart/ })).toBeInTheDocument()
    expect(screen.getByRole('group', { name: /Hourly US and European AQI chart/ })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Hourly data table' }))
    expect(screen.getByRole('table', { name: 'Hourly weather data' })).toBeInTheDocument()
  })

  it('keeps weather and air quality independent when one service fails', async () => {
    const user = userEvent.setup()
    const calls = mockBackend({ weather: 'fail' })
    localStorage.setItem('di-location', JSON.stringify({ lat: 25.61, lng: 85.14, name: 'Patna, Bihar, India' }))
    render(<App />)
    await go(user, 'Weather & air')
    const air = (await screen.findByRole('heading', { name: 'Air quality' })).closest('section')
    expect(await within(air).findByText('Unhealthy')).toBeInTheDocument()
    const now = screen.getByRole('heading', { name: 'Current conditions' }).closest('section')
    expect(await within(now).findByRole('alert')).toHaveTextContent(/error \(500\)/)
    // retry recovers weather without re-fetching air quality
    expect(calls.air).toBe(1)
    const after = mockBackend()
    await user.click(within(now).getByRole('button', { name: /Retry/ }))
    expect(await within(now).findByText('Overcast')).toBeInTheDocument()
    expect(after.weather).toBe(1)
    expect(after.air).toBe(0)
  })

  it('asks for a location when none is selected', async () => {
    const user = userEvent.setup()
    mockBackend()
    render(<App />)
    await go(user, 'Weather & air')
    expect(await screen.findByRole('heading', { name: 'No location selected' })).toBeInTheDocument()
  })
})

describe('Demo scenes', () => {
  it('loads the catalogue only when opened and runs a scene by id', async () => {
    const user = userEvent.setup()
    const calls = mockBackend()
    render(<App />)
    await waitFor(() => expect(screen.getByText('Online')).toBeInTheDocument())
    expect(calls.samples).toBe(0)
    await go(user, 'Demo scenes')
    expect(await screen.findByText('Demonstration mode')).toBeInTheDocument()
    await user.click(await screen.findByRole('button', { name: /Southnormanton/ }))
    await waitFor(() => expect(calls.assess.at(-1)).toEqual({ mode: 'demo', sample_id: 'EMSR342_07SOUTHNORMANTON_DEL_MONIT03_v2', method: 'tiled' }))
    const result = await screen.findByRole('article')
    expect(within(result).getByText(/Archived demo scene/)).toBeInTheDocument()
    expect(within(result).getByRole('heading', { name: 'Flood-like patterns detected' })).toBeInTheDocument()
    expect(within(result).getByRole('heading', { name: 'Agreement with the EMS flood map' })).toBeInTheDocument()
    expect(within(result).getByText('0.43')).toBeInTheDocument()
    expect(within(result).getByText(/archived event/)).toBeInTheDocument()
    expect(calls.samples).toBe(1)
  })

  it('keeps the flood page free of demo samples', async () => {
    mockBackend()
    render(<App />)
    await waitFor(() => expect(screen.getByText('Online')).toBeInTheDocument())
    expect(screen.queryByText(/Southnormanton|Sample assessments/)).not.toBeInTheDocument()
  })
})

describe('Research dashboard', () => {
  it('shows readiness, model comparison and diagnostics of the last result with a method re-run', async () => {
    const user = userEvent.setup()
    const calls = mockBackend()
    render(<App />)
    await go(user, 'Demo scenes')
    await user.click(await screen.findByRole('button', { name: /Southnormanton/ }))
    await screen.findByRole('article')
    await go(user, 'Research')
    expect(await screen.findByRole('heading', { name: 'Research dashboard' }, { timeout: 8000 })).toBeInTheDocument()
    expect(await screen.findByText('Ready')).toBeInTheDocument()
    expect(screen.getByText('satellite credentials')).toBeInTheDocument()
    expect(await screen.findByText('0.451')).toBeInTheDocument()
    expect(screen.getByText('0.252')).toBeInTheDocument()
    const diag = screen.getByRole('heading', { name: 'Last assessment diagnostics' }).closest('section')
    expect(within(diag).getByText('0.432')).toBeInTheDocument()
    await user.click(within(diag).getAllByRole('radio', { name: 'Resize' })[0])
    await waitFor(() => expect(calls.assess.at(-1)).toMatchObject({ mode: 'demo', method: 'resize' }))
  }, 30000)
})
