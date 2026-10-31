/**
 * Browser smoke test against the REAL running app and backend (Milestone 5).
 *
 * Prerequisites (two terminals):
 *   backend:  cd backend;  ..\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
 *   frontend: cd frontend; npm run dev          (proxies /api, /ready, /health to the backend)
 * Run:       cd frontend; node e2e/smoke.mjs [chromium|firefox|webkit|all]
 *
 * Engines: Chromium = installed Microsoft Edge/Chrome; Firefox and WebKit = Playwright builds
 * (`npx playwright-core install firefox webkit`). WebKit is the engine behind Safari, but the
 * Playwright WebKit build on Windows is NOT Safari.
 *
 * Copernicus is not called by default: live analysis requests are intercepted in the browser and
 * answered with contract-shaped fixtures (several outcomes and an error). Everything else — model
 * status, demo catalogue, real demo inference, readiness, research summary, Nominatim, Open-Meteo
 * and map tiles — is real. Set LIVE=1 to send ONE real live request (Chromium only).
 *
 * Accessibility: axe-core (WCAG 2.0/2.1/2.2 A + AA rules) on every page in both themes.
 */
import { chromium, firefox, webkit } from 'playwright-core'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const APP = process.env.APP_URL || 'http://localhost:5173'
const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '../../docs/figures/ui_m5')
mkdirSync(OUT, { recursive: true })
const AXE = readFileSync(createRequire(import.meta.url).resolve('axe-core/axe.min.js'), 'utf8')
const DEMO_SCENE = process.env.DEMO_SCENE || 'Calayan' // smallest validation scene (266 × 218 px): fast even on CPU

const edge = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Google/Chrome/Application/chrome.exe'].find(existsSync)
const ENGINES = {
  chromium: () => chromium.launch({ executablePath: edge, headless: true }),
  firefox: () => firefox.launch({ headless: true }),
  webkit: () => webkit.launch({ headless: true }),
}
const which = process.argv[2] || 'all'
const engines = which === 'all' ? Object.keys(ENGINES) : [which]

const BANNED = [
  'This is a preliminary estimate from an experimental model, not a confirmed flood event or an official warning.',
  'Preliminary model estimate.',
  'Results appear here. They show what the model sees in one satellite image — not an official warning.',
  'What you can do',
]

const HEADLINE = {
  flood_like: 'Flood-like patterns detected',
  none_detected: 'No flood-like patterns detected',
  inconclusive: 'Inconclusive',
  uncertain_water: 'Inconclusive — mostly permanent water',
  insufficient_coverage: 'Insufficient valid coverage',
}

// 1 × 1 transparent PNG: fixture overlays are invisible, so screenshots show the basemap and the analysed area
const PIXEL = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='

/** Contract-shaped live response (Milestone 5) for a 500 × 500 px, 5 × 5 km area. */
function liveFixture(lat, lng, level) {
  const d = 0.0225
  const eligible = level === 'insufficient_coverage' ? 25000 : 225000
  const land = level === 'uncertain_water' ? 30000 : Math.round(eligible * 0.95)
  const flood = { flood_like: 40000, inconclusive: 9000, none_detected: 400, uncertain_water: 500, insufficient_coverage: 0 }[level]
  const share = Math.round((1e6 * flood) / land) / 1e4
  return {
    status: 'success', mode: 'live', method: 'tiled', method_note: '256 px tiles at native resolution', experimental: true, sample: null,
    imagery: {
      source: 'Copernicus Sentinel-2 Level-1C', processing_level: 'L1C', acquisition_date: '2026-10-18', acquisition_datetime: '2026-10-18T05:32:11Z',
      resolution_m: 10, width: 500, height: 500, crs: 'EPSG:4326', center: { lat, lng },
      bbox: { west: lng - d, south: lat - d, east: lng + d, north: lat + d }, tile_cloud_cover_pct: 3.1, cloud_mask_source: 'sen2cor_scl',
      bands: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'], location_note: 'A 5 × 5 km area centred on the selected point.',
    },
    images: { rgb: PIXEL, overlay: PIXEL, reference: null, error: null, display_width: 500, display_height: 500, legend: { flood: '#00aaff' } },
    prediction: {
      target: 'surface_water', raw_label: 'surface water (including permanent water)', raw_coverage_pct: 10, flood_coverage_pct: 10,
      coverage_definition: '…', n_flood_pixels: flood + 2000, n_eligible_pixels: eligible, n_pixels: 250000, eligible_pct: (100 * eligible) / 250000,
      nodata_pct: 0, eligibility_source: 'validity_mask+nodata', threshold: 0.5, threshold_note: 'The score is not a calibrated probability or confidence.', n_tiles: 4, inference_s: 1,
    },
    reference: { available: false, note: 'No EMS reference mask exists for live imagery; accuracy cannot be measured.' },
    quality: { status: 'ok', warnings: ['Experimental: live imagery has no ground-truth reference, so this prediction is unvalidated.'] },
    model: { version: 'segmentation_v2', architecture: 'U-Net', encoder: 'efficientnet-b2', epoch: 7, target: 'surface_water', bands: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'], device: 'cuda' },
    assessment: {
      level, headline: HEADLINE[level], summary: 'Fixture summary for the browser test.',
      reasons: level === 'insufficient_coverage' ? ['Only 10% of the image is usable; the rest is cloud, shadow or missing data.'] : [],
      limitations: [], uncertainty: [], suggestion: level === 'uncertain_water' ? 'Choose a point further inland to assess land areas.' : null, next_steps: [],
      flood_like_outside_known_water_pct: share,
      basis: { kind: 'descriptive_model_outcome', is_risk_rating: false, is_probability: false, method: 'tiled', none_below_pct: 1, inconclusive_ceiling_pct: 6.5, policy_version: 'fixture' },
    },
    water_context: {
      available: true, n_eligible: eligible, n_eligible_outside_known_water: land, n_flood_like_outside_known_water: flood, n_flood_like_on_known_water: 2000,
      flood_like_outside_known_water_pct: share, raw_flood_like_pct: 10, known_water_pct_of_eligible: 5, sea_pct_of_eligible: 0, source: 'JRC GSW', reference_status: 'ok',
    },
    map: {
      crs: 'EPSG:4326', bounds: [[lat - d, lng - d], [lat + d, lng + d]], satellite: PIXEL, flood: PIXEL, set_aside: PIXEL, width: 500, height: 500,
    },
    freshness: { acquisition_date: '2026-10-18', age_days: 8, archived: false },
    limitations: ['Fixture limitation.'], disclaimer: 'Satellite-image flood extent analysis from a research model. Not an official warning; it does not forecast floods.',
    processing_s: 1, cached: false,
  }
}

async function axe(page, label, results) {
  await page.waitForTimeout(600) // let entry animations (opacity) finish so colours are measured at rest
  await page.addScriptTag({ content: AXE })
  const r = await page.evaluate(async () => {
    // eslint-disable-next-line no-undef
    const res = await axe.run(document, { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'] } })
    return res.violations.map((v) => ({ id: v.id, impact: v.impact, n: v.nodes.length, target: v.nodes[0]?.target?.join(' ') }))
  })
  results.axe.push({ label, violations: r })
  return r
}

const nav = (page, name) => page.locator('nav.topnav').getByRole('link', { name }).click()

async function runEngine(name) {
  const results = { checks: [], axe: [] }
  const check = (n, ok, info = '') => {
    results.checks.push({ n, ok })
    console.log(`[${name}] ${ok ? 'PASS' : 'FAIL'}  ${n}${info ? `  — ${info}` : ''}`)
  }
  const browser = await ENGINES[name]()
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    colorScheme: 'dark',
    ...(name === 'chromium' ? { geolocation: { latitude: 25.61, longitude: 85.14 }, permissions: ['geolocation'] } : {}),
  })
  const page = await ctx.newPage()
  const errors = []
  const failed = []
  let cutoff = null
  page.on('console', (m) => {
    if (m.type() !== 'error') return
    // The intentional 404 fixture for "imagery unavailable" is logged by the browser as a failed resource
    if (/status of 404/.test(m.text()) && /\/api\/flood\/assess$/.test(m.location()?.url || '')) return
    errors.push(`${m.text()} @ ${(m.location()?.url || '').slice(0, 80)}`)
  })
  page.on('pageerror', (e) => errors.push(String(e)))
  page.on('requestfailed', (r) => failed.push(`${r.url().slice(0, 90)} ${r.failure()?.errorText}`))
  const assessCalls = []
  page.on('request', (r) => r.url().endsWith('/api/flood/assess') && assessCalls.push(JSON.parse(r.postData() || '{}')))

  let liveMode = 'inconclusive'
  const realLive = process.env.LIVE === '1' && name === 'chromium'
  if (!realLive) {
    await page.route('**/api/flood/assess', async (route) => {
      const body = JSON.parse(route.request().postData() || '{}')
      if (body.mode !== 'live') return route.continue()
      await new Promise((r) => setTimeout(r, 700)) // visible loading state
      if (liveMode === 'error404') {
        return route.fulfill({ status: 404, json: { detail: 'No Sentinel-2 acquisition found for this area in the last 45 days. Try again later.' } })
      }
      return route.fulfill({ json: liveFixture(body.latitude, body.longitude, liveMode) })
    })
  }

  try {
    // 1. Flood page, nothing selected
    await page.goto(APP, { waitUntil: 'domcontentloaded' })
    await page.getByRole('heading', { name: 'Flood analysis', level: 1 }).waitFor({ timeout: 60000 })
    await page.getByText('Choose a location to analyse.').waitFor({ timeout: 90000 })
    check('primary action disabled until a location is chosen', await page.getByRole('button', { name: 'Analyse this area' }).isDisabled())
    const startText = await page.locator('.workspace__panel').evaluate((el) => el.textContent)
    check('no sample list or developer metrics on the flood page', !/Southnormanton|Sample assessments|IoU|epoch|cuda/.test(startText))
    const mapShare = await page.evaluate(() => {
      const m = document.querySelector('.workspace__map').getBoundingClientRect()
      return Math.round(((m.width * m.height) / (window.innerWidth * window.innerHeight)) * 100)
    })
    check('map takes most of the screen', mapShare > 55, `${mapShare}% of viewport`)
    await page.screenshot({ path: `${OUT}/${name}_01_flood_start_dark.png` })
    await axe(page, 'flood start (dark)', results)

    // 2. Keyboard: skip link + "/" search shortcut
    await page.keyboard.press('Tab')
    const first = await page.evaluate(() => `${document.activeElement?.tagName}:${document.activeElement?.textContent || document.activeElement?.getAttribute('aria-label')}`)
    if (name === 'webkit') check('first Tab stop is a control (WebKit skips links by default)', /^(BUTTON|INPUT):/.test(first) || first.startsWith('A:'), first)
    else check('skip link is the first focus stop', first === 'A:Skip to content', first)
    await page.locator('body').click({ position: { x: 5, y: 300 } }).catch(() => {})
    await page.keyboard.press('/')
    check('"/" focuses place search', (await page.evaluate(() => document.activeElement?.getAttribute('role'))) === 'combobox')

    // 3. Real Photon autocomplete → choose place with the keyboard (Enter searches now, second Enter selects)
    await page.keyboard.type('Patna')
    await page.keyboard.press('Enter')
    await page.getByRole('option').first().waitFor({ timeout: 20000 })
    await page.keyboard.press('Enter')
    await page.locator('.place-chip').waitFor()
    check('place chosen from real search results', true, (await page.locator('.place-chip strong').innerText()).slice(0, 40))
    check('5 × 5 km analysis area drawn on the map', (await page.locator('path.fmap__area').count()) === 1)
    check('location remembered', (await page.evaluate(() => localStorage.getItem('di-location')))?.includes('"lat"'))

    // 4. Live analysis
    await page.getByRole('button', { name: 'Analyse this area' }).click()
    const prog = await page.locator('.progress').isVisible().catch(() => false)
    check('progress shown while analysing', prog || realLive)
    await page.locator('article.result').waitFor({ timeout: realLive ? 240000 : 20000 })
    const res = await page.locator('article.result').evaluate((el) => el.textContent)
    const order = ['Outcome', 'Coverage', 'Image', 'Map', 'Environmental context', 'How this result was produced']
    const idx = order.map((s) => res.indexOf(s))
    check('result sections in the required order', idx.every((v, i) => v >= 0 && (i === 0 || v > idx[i - 1])), idx.join(','))
    check('coverage states its denominator', /of .+ km² valid land/.test(res) || realLive)
    check('image date and age shown', /\d{1,2} \w{3} \d{4}/.test(res) && /days? old|today/.test(res))
    check('live request carries coordinates only', assessCalls.at(-1)?.mode === 'live' && !assessCalls.at(-1)?.sample_id)
    const panelText = await page.locator('.workspace__panel').evaluate((el) => el.textContent)
    check('banned copy absent', BANNED.every((b) => !panelText.includes(b)))
    await page.waitForTimeout(700)
    const fit = await page.evaluate(() => {
      const m = document.querySelector('.workspace__map').getBoundingClientRect()
      const imgs = [...document.querySelectorAll('.workspace__map img.leaflet-image-layer')].map((i) => i.getBoundingClientRect())
      if (!imgs.length) return { n: 0, share: 0, inside: false }
      const r = imgs[0]
      return {
        n: imgs.length,
        share: Math.round(((r.width * r.height) / (m.width * m.height)) * 100),
        inside: r.left >= m.left - 2 && r.right <= m.right + 2 && r.top >= m.top - 2 && r.bottom <= m.bottom + 2,
      }
    })
    check('map fitted to the analysed raster bounds', fit.n >= 2 && fit.inside && fit.share > 25, `${fit.n} overlays, ${fit.share}% of map`)
    await page.screenshot({ path: `${OUT}/${name}_02_live_result_dark.png` })
    await axe(page, 'live result (dark)', results)

    // 5. Panel scrolling is covered in depth by e2e/scroll.mjs (wheel, scrollbar, keyboard, async updates, 6 viewports)

    // 6. Layer panel and basemap
    await page.getByRole('button', { name: 'Layers' }).click()
    const ov = await page.locator('.workspace__map img.leaflet-image-layer').count()
    await page.getByLabel('Satellite image', { exact: true }).uncheck()
    check('satellite layer toggle works', (await page.locator('.workspace__map img.leaflet-image-layer').count()) === ov - 1)
    await page.getByLabel('Satellite image', { exact: true }).check()
    await page.locator('.map-panel__seg label', { hasText: 'Streets' }).click()
    await page.waitForTimeout(300)
    check('basemap switch works', (await page.locator('.tiles--streets').count()) === 1)
    await page.locator('.map-panel__seg label', { hasText: 'Minimal' }).click()
    await page.getByRole('button', { name: 'Close' }).click()

    // 7. Other outcomes and an error state (fixtures)
    if (!realLive) {
      liveMode = 'uncertain_water'
      await page.getByRole('button', { name: /Analyse again/ }).click()
      await page.getByRole('heading', { name: HEADLINE.uncertain_water }).waitFor({ timeout: 20000 })
      check('permanent-water outcome shows no flood-like share', (await page.getByText('of the valid land analysed is flood-like water').count()) === 0)
      liveMode = 'error404'
      await page.getByRole('button', { name: /Analyse again/ }).click()
      await page.getByRole('heading', { name: 'Imagery unavailable' }).waitFor({ timeout: 20000 })
      check('missing imagery shown as an outcome with retry', await page.getByRole('button', { name: /Try again/ }).isVisible())
      liveMode = 'flood_like'
      await page.getByRole('button', { name: /Try again/ }).click()
      await page.getByRole('heading', { name: HEADLINE.flood_like }).waitFor({ timeout: 20000 })
      check('retry recovers to a result', true)
    }

    // 8. Changing location clears the old result
    await page.locator('.workspace__map .leaflet-container').click({ position: { x: 300, y: 600 } })
    check('changing location removes the previous result', (await page.locator('article.result').count()) === 0)

    // 9. Weather & air quality (real Open-Meteo)
    await nav(page, 'Weather & air')
    await page.getByRole('heading', { name: 'Current conditions' }).waitFor({ timeout: 20000 })
    await page.locator('.tile').first().waitFor({ timeout: 30000 })
    await page.locator('.aqi').first().waitFor({ timeout: 30000 })
    const wx = await page.locator('.wx-grid').evaluate((el) => el.textContent)
    check('weather: current conditions with wind, gusts, visibility, sunrise/sunset', ['Wind', 'Gusts', 'Visibility', 'Pressure', 'Sunrise', 'Sunset', 'UV index'].every((t) => wx.includes(t)))
    await page.locator('.aqi').first().waitFor({ timeout: 30000 })
    check('air quality: US AQI and European AQI shown separately', wx.includes('US AQI') && wx.includes('European AQI'))
    check('hourly and AQI trend charts rendered', (await page.locator('.chart svg path').count()) >= 2)
    check('7-day table rendered', (await page.locator('table.daily tbody tr').count()) === 7)
    await page.getByRole('radio', { name: 'Rain' }).click()
    check('hourly chart view switch works', (await page.locator('.chart__bars, .chart svg path').count()) >= 1)
    await page.getByRole('button', { name: 'Hourly data table' }).click()
    check('chart data table alternative', (await page.getByRole('table', { name: 'Hourly weather data' }).count()) === 1)
    await page.getByRole('button', { name: 'Hourly data table' }).click()
    await page.screenshot({ path: `${OUT}/${name}_03_weather_dark.png`, fullPage: true })
    await axe(page, 'weather (dark)', results)

    // 10. Demo scenes: real inference on an archived scene + georeferenced overlays
    await nav(page, 'Demo scenes')
    await page.getByText('Demonstration mode').waitFor({ timeout: 20000 })
    await page.getByRole('button', { name: new RegExp(DEMO_SCENE) }).first().click()
    await page.locator('article.result').waitFor({ timeout: 240000 })
    const dres = await page.locator('article.result').evaluate((el) => el.textContent)
    check('demo result labelled archived', /Archived demo scene/i.test(dres) && dres.includes('archived event'))
    check('demo result shows EMS agreement', dres.includes('Agreement with the EMS flood map'))
    await page.waitForTimeout(800)
    check('demo overlays placed on the map', (await page.locator('.workspace__map img.leaflet-image-layer').count()) >= 2)
    await page.screenshot({ path: `${OUT}/${name}_04_demo_dark.png` })
    await axe(page, 'demo (dark)', results)

    // 11. Research dashboard: readiness, comparison, diagnostics with a real method re-run
    await nav(page, 'Research')
    await page.getByRole('heading', { name: 'Research dashboard' }).waitFor({ timeout: 20000 })
    await page.locator('.checks').waitFor({ timeout: 30000 })
    check('readiness checks shown', (await page.locator('.check').count()) >= 5)
    await page.locator('table.dtable--cmp').waitFor({ timeout: 30000 })
    check('model comparison table shown', (await page.locator('table.dtable--cmp tbody tr').count()) >= 6)
    const diag = page.locator('section', { has: page.getByRole('heading', { name: 'Last assessment diagnostics' }) })
    const n0 = assessCalls.length
    await diag.getByRole('radio', { name: 'Resize' }).first().click()
    await page.waitForFunction(() => document.querySelector('.card__meta')?.closest('section') && [...document.querySelectorAll('.card__meta')].some((e) => e.textContent.includes('method resize')), null, { timeout: 240000 })
    check('method switch re-runs real inference', assessCalls.length > n0 && assessCalls.at(-1).method === 'resize')
    await page.screenshot({ path: `${OUT}/${name}_05_research_dark.png`, fullPage: true })
    await axe(page, 'research (dark)', results)

    // 12. Methodology
    await nav(page, 'Methodology')
    await page.getByRole('heading', { name: 'How Disaster Insights works' }).waitFor({ timeout: 20000 })
    check('methodology lists every outcome state', (await page.locator('.outcome-list__item').count()) === 7)
    await axe(page, 'methodology (dark)', results)

    // 13. Light theme on every page
    await page.getByRole('button', { name: 'Switch to light mode' }).click()
    check('light theme applied', (await page.evaluate(() => document.documentElement.dataset.theme)) === 'light')
    await page.reload({ waitUntil: 'domcontentloaded' })
    check('theme persists after reload', (await page.evaluate(() => document.documentElement.dataset.theme)) === 'light')
    await page.getByRole('heading', { name: 'How Disaster Insights works' }).waitFor({ timeout: 20000 })
    await axe(page, 'methodology (light)', results)
    for (const [link, heading, shot] of [
      ['Flood analysis', 'Flood analysis', '06_flood_light'],
      ['Weather & air', 'Current conditions', '07_weather_light'],
      ['Demo scenes', 'Demo scenes', '08_demo_light'],
      ['Research', 'Research dashboard', '09_research_light'],
    ]) {
      await nav(page, link)
      await page.getByRole('heading', { name: heading }).first().waitFor({ timeout: 20000 })
      await page.waitForTimeout(1500)
      await page.screenshot({ path: `${OUT}/${name}_${shot}.png` })
      await axe(page, `${link} (light)`, results)
    }

    // 14. Responsive: tablet + mobile (tab bar instead of top navigation)
    await nav(page, 'Flood analysis')
    for (const [w, h, tag] of [[1024, 768, 'tablet'], [390, 844, 'mobile']]) {
      await page.setViewportSize({ width: w, height: h })
      await page.waitForTimeout(500)
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
      check(`${tag} ${w}px: no horizontal overflow`, overflow <= 1, `${overflow}px`)
      await page.screenshot({ path: `${OUT}/${name}_10_${tag}.png`, fullPage: tag === 'mobile' })
    }
    check('mobile shows the bottom tab bar', await page.locator('nav.tabbar').isVisible())
    await page.locator('nav.tabbar').getByRole('link', { name: 'Weather' }).click()
    await page.getByRole('heading', { name: 'Current conditions' }).waitFor({ timeout: 20000 })
    const overflowW = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    check('mobile weather page: no horizontal overflow', overflowW <= 1, `${overflowW}px`)
    await page.screenshot({ path: `${OUT}/${name}_11_mobile_weather.png`, fullPage: true })
    await axe(page, 'mobile weather (light)', results)
    await page.setViewportSize({ width: 1440, height: 900 })

    // 15. Failure handling (errors after this point are expected)
    cutoff = { errors: errors.length, failed: failed.length }
    await page.route('**/air-quality-api.open-meteo.com/**', (r) => r.abort())
    await page.reload({ waitUntil: 'domcontentloaded' })
    await page.locator('.card--aqi .card__error').waitFor({ timeout: 30000 }).catch(() => {})
    await page.locator('.card--now .tile').first().waitFor({ timeout: 30000 }).catch(() => {})
    const aqErr = await page.locator('.card--aqi .card__error').count()
    const wxOk = await page.locator('.card--now .tile').count()
    check('weather still shown when air quality fails (independent states)', aqErr === 1 && wxOk > 5, `aq error ${aqErr}, weather tiles ${wxOk}`)
    await page.unroute('**/air-quality-api.open-meteo.com/**')
    await page.route('**/api/**', (r) => r.abort())
    await nav(page, 'Flood analysis')
    await page.reload({ waitUntil: 'domcontentloaded' })
    await page.getByText('Service offline').waitFor({ timeout: 20000 })
    check('offline state shown when the API is unreachable', true)
    await page.unroute('**/api/**')
  } catch (e) {
    check('workflow completed without exception', false, e.message.split('\n')[0])
    await page.screenshot({ path: `${OUT}/${name}_zz_failure.png` }).catch(() => {})
  }

  const realErrors = errors.slice(0, cutoff?.errors ?? errors.length)
  check('no runtime console errors', realErrors.length === 0, realErrors.slice(0, 2).join(' | '))
  // ERR_ABORTED / NS_BINDING_ABORTED = requests the app cancelled on purpose (superseded, StrictMode dev double-run, map tiles)
  const realFailed = failed.slice(0, cutoff?.failed ?? failed.length).filter((f) => !/ERR_ABORTED|NS_BINDING_ABORTED|cancelled|Load request cancelled/i.test(f))
  check('no unexpected failed requests', realFailed.length === 0, realFailed.slice(0, 2).join(' | '))
  const axeTotal = results.axe.reduce((s, a) => s + a.violations.length, 0)
  for (const a of results.axe) {
    console.log(`[${name}] axe ${a.label}: ${a.violations.length ? a.violations.map((v) => `${v.id}(${v.impact},${v.n}) ${v.target}`).join('; ') : 'no violations'}`)
  }
  check('axe-core: no WCAG A/AA violations on checked screens', axeTotal === 0, `${axeTotal} rule violations`)
  await browser.close()
  const passed = results.checks.filter((c) => c.ok).length
  console.log(`[${name}] ${passed}/${results.checks.length} checks passed\n`)
  return passed === results.checks.length
}

let ok = true
for (const e of engines) ok = (await runEngine(e)) && ok
process.exit(ok ? 0 : 1)
