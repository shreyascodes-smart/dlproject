/**
 * Scrolling regression test for the Flood analysis page (real app, real browser input).
 * Prerequisites: backend on :8000 and `npm run dev` on :5173.  Run: node e2e/scroll.mjs [chromium|firefox|webkit|all]
 *
 * The live analysis is answered with a contract-shaped fixture (so it is fast and repeatable);
 * weather / air quality (Open-Meteo), Nominatim and map tiles are real. Open-Meteo is delayed by 2.5 s
 * so its content arrives WHILE the user is scrolled down (async-update check).
 */
import { chromium, firefox, webkit } from 'playwright-core'
import { existsSync } from 'node:fs'

const APP = process.env.APP_URL || 'http://localhost:5173'
const edge = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Google/Chrome/Application/chrome.exe'].find(existsSync)
const ENGINES = {
  // HEADED=1: a real window. Headless Chromium draws overlay scrollbars with no draggable track,
  // so the scrollbar-drag check only runs headed.
  chromium: () => chromium.launch({ executablePath: edge, headless: process.env.HEADED !== '1' }),
  firefox: () => firefox.launch({ headless: true }),
  webkit: () => webkit.launch({ headless: true }),
}
const VIEWPORTS = [[1920, 1080], [1440, 900], [1366, 768], [1024, 768], [768, 1024], [390, 844]]
const LOC = { lat: 19.1836, lng: 72.9017, name: 'P/S Ward, Mumbai, Maharashtra' }
const PIXEL = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
const SECTIONS = ['Outcome', 'Coverage', 'Compared with a year earlier', 'Image', 'Map', 'Environmental context', 'How this result was produced']

function fixture(lat, lng) {
  const d = 0.0225
  const byClass = { permanent: { n_eligible: 1761, n_water_predicted: 1761 }, sea: { n_eligible: 0, n_water_predicted: 0 },
    seasonal: { n_eligible: 38, n_water_predicted: 19 }, mapped: { n_eligible: 11381, n_water_predicted: 6012 } }
  return {
    status: 'success', mode: 'live', method: 'tiled', method_note: '…', experimental: true, sample: null,
    imagery: { source: 'Copernicus Sentinel-2 Level-1C', processing_level: 'L1C', acquisition_date: '2026-10-28', resolution_m: 10, width: 500, height: 500,
      center: { lat, lng }, bbox: { west: lng - d, south: lat - d, east: lng + d, north: lat + d }, tile_cloud_cover_pct: 4, cloud_mask_source: 'sen2cor_scl' },
    images: { rgb: PIXEL, overlay: PIXEL, reference: null, error: null, display_width: 500, display_height: 500, legend: { flood: '#00aaff' } },
    prediction: { target: 'surface_water', raw_label: 'surface water', raw_coverage_pct: 4.3, flood_coverage_pct: 4.3, coverage_definition: '…', n_flood_pixels: 10600,
      n_eligible_pixels: 249500, n_pixels: 250000, eligible_pct: 99.8, nodata_pct: 0, eligibility_source: 'validity_mask+nodata', threshold: 0.5, threshold_note: '…', n_tiles: 4,
      inference_s: 1, haze_index: 0.09, hazy: false },
    reference: { available: false, note: '…' },
    quality: { status: 'ok', warnings: ['Experimental: live imagery has no ground-truth reference.', 'Clouds were masked with Sen2Cor.'] },
    model: { version: 'segmentation_v2', architecture: 'U-Net', encoder: 'efficientnet-b2', bands: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'] },
    assessment: { level: 'none_detected', headline: 'No flood-like patterns detected', summary: 'The water found on normally dry land was already there a year earlier.',
      reasons: ['Almost all of the flood-like water was already present on 2025-11-27.'], limitations: [], suggestion: null, flood_like_outside_known_water_pct: 1.3,
      basis: { kind: 'descriptive_model_outcome', is_probability: false, none_below_pct: 1, inconclusive_ceiling_pct: 28.57 } },
    water_context: { available: true, by_class: byClass, n_eligible: 249500, n_eligible_outside_known_water: 236320, n_flood_like_outside_known_water: 3072,
      flood_like_outside_known_water_pct: 1.3, source: 'JRC GSW; OSM', layers: { jrc: 'ok', osm: 'ok' } },
    change: { status: 'available', baseline_date: '2025-11-27', compared_pct_of_usable: 95, water_then_pct_of_dry_land: 1.1, water_now_pct_of_dry_land: 1.3,
      new_water_pct_of_dry_land: 0.2, persistent_share_of_candidate_pct: 70 },
    map: { crs: 'EPSG:4326', bounds: [[lat - d, lng - d], [lat + d, lng + d]], satellite: PIXEL, flood: PIXEL, set_aside: PIXEL, width: 500, height: 500 },
    freshness: { acquisition_date: '2026-10-28', age_days: 3, archived: false },
    limitations: ['Fixture limitation one.', 'Fixture limitation two.'], disclaimer: 'Satellite-image flood extent analysis from a research model.', cached: false,
  }
}

async function run(name) {
  const browser = await ENGINES[name]()
  let pass = 0
  let fail = 0
  const check = (vp, n, ok, info = '') => {
    if (ok) pass++
    else fail++
    console.log(`[${name} ${vp}] ${ok ? 'PASS' : 'FAIL'}  ${n}${info ? `  — ${info}` : ''}`)
  }
  for (const [w, h] of VIEWPORTS) {
    const vp = `${w}x${h}`
    const desktop = w >= 1024
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, colorScheme: 'dark' })
    await ctx.addInitScript((l) => localStorage.setItem('di-location', JSON.stringify(l)), LOC)
    const page = await ctx.newPage()
    const errors = []
    page.on('pageerror', (e) => errors.push(String(e)))
    await page.route('**/api/flood/assess', async (r) => {
      const b = JSON.parse(r.request().postData() || '{}')
      await new Promise((res) => setTimeout(res, 600))
      return r.fulfill({ json: fixture(b.latitude, b.longitude) })
    })
    await page.route('**/*open-meteo.com/**', async (r) => {
      await new Promise((res) => setTimeout(res, 2500))
      return r.continue()
    })
    try {
      await page.goto(`${APP}/#/`, { waitUntil: 'domcontentloaded' })
      const analyse = page.getByRole('button', { name: /Analyse this area/ })
      await analyse.waitFor({ timeout: 60000 })
      await page.waitForFunction(() => !document.querySelector('button.btn--primary')?.disabled, null, { timeout: 60000 })
      await analyse.click()
      await page.locator('article.result').waitFor({ timeout: 20000 })

      // The scroller: the side panel on desktop, the document on tablet portrait / mobile
      const scroller = desktop ? '.workspace__panel' : null
      const st = () => page.evaluate((s) => (s ? document.querySelector(s).scrollTop : window.scrollY), scroller)
      const max = () => page.evaluate((s) => {
        const el = s ? document.querySelector(s) : document.scrollingElement
        return el.scrollHeight - el.clientHeight
      }, scroller)
      const mapBox = () => page.locator('.workspace__map').boundingBox()

      if (desktop) {
        const doc = await page.evaluate(() => document.scrollingElement.scrollHeight - window.innerHeight)
        check(vp, 'no document-level scrolling on desktop', doc <= 1, `${doc}px`)
      }
      const overflowX = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
      check(vp, 'no horizontal overflow', overflowX <= 1, `${overflowX}px`)

      // 1. Wheel scroll top -> bottom: monotonic, no jumps, reaches the end, every section seen
      // Desktop: pointer over the side panel. Single column: pointer mid-screen, so the full-width map
      // passes under it while the page scrolls (the trap that used to stop page scrolling).
      const panel = await page.locator(desktop ? '.workspace__panel' : '.panel__controls').boundingBox()
      await page.mouse.move(panel.x + Math.min(160, panel.width / 2), desktop ? panel.y + panel.height / 2 : h / 2)
      const map0 = desktop ? await mapBox() : null
      const seen = new Set()
      let prev = await st()
      let monotonic = true
      let maxJump = 0
      let weatherArrivedMidScroll = false
      for (let i = 0; i < 80; i++) {
        await page.mouse.wheel(0, 120)
        await page.waitForTimeout(70)
        const y = await st()
        if (y < prev - 1) monotonic = false
        maxJump = Math.max(maxJump, y - prev)
        prev = y
        for (const s of SECTIONS) {
          const vis = await page.evaluate((t) => [...document.querySelectorAll('article.result h3, article.result .outcome__eyebrow, article.result .disclosure__title')]
            .some((e) => e.textContent.trim().startsWith(t) && e.getBoundingClientRect().bottom > 0 && e.getBoundingClientRect().top < window.innerHeight), s)
          if (vis) seen.add(s)
        }
        if (!weatherArrivedMidScroll && y > 200) weatherArrivedMidScroll = (await page.locator('.ctx__big').count()) === 0
        if (y >= (await max()) - 1) break
      }
      const bottom = await st()
      const mx = await max()
      check(vp, 'wheel scrolling reaches the very bottom', bottom >= mx - 1, `scrollTop ${bottom} / max ${mx}`)
      check(vp, 'scroll position only moves in the wheel direction (no jumps back)', monotonic)
      check(vp, 'no step larger than the wheel input', maxJump <= 240, `largest step ${maxJump}px`)
      check(vp, 'every result section is reachable', seen.size === SECTIONS.length, [...SECTIONS].filter((s) => !seen.has(s)).join(', ') || 'all')
      const footer = await page.locator('.result__foot').boundingBox()
      const tabbar = await page.locator('nav.tabbar').boundingBox()
      const limit = tabbar && (await page.locator('nav.tabbar').isVisible()) ? tabbar.y : h
      check(vp, 'last content fully visible (not behind fixed navigation)', footer && footer.y + footer.height <= limit + 1, `footer bottom ${Math.round(footer.y + footer.height)} / limit ${Math.round(limit)}`)
      if (desktop) {
        const map1 = await mapBox()
        check(vp, 'map did not move or resize while scrolling', JSON.stringify(map0) === JSON.stringify(map1))
      }

      // 2. Async update while scrolled: weather/AQ arrive late; position must stay
      await page.locator('.ctx__big').first().waitFor({ timeout: 30000 })
      if (desktop) {
        const target = Math.round((await max()) / 2)
        await page.evaluate(([s, y]) => (document.querySelector(s).scrollTop = y), [scroller, target])
        await page.waitForTimeout(200)
        const before = await st()
        // Re-run the analysis from the result footer (JS click: no scrollIntoView side effects)
        await page.evaluate(() => [...document.querySelectorAll('.result__foot button')].find((b) => /Run again/.test(b.textContent))?.click())
        await page.waitForTimeout(250)
        const during = await st()
        await page.locator('article.result:not(.is-stale)').waitFor({ timeout: 20000 })
        await page.waitForTimeout(400)
        const after = await st()
        check(vp, 'assessment re-run keeps the scroll position', Math.abs(during - before) <= 2 && Math.abs(after - before) <= 2, `${before} -> ${during} -> ${after}`)
      }
      // Weather/AQ (delayed 2.5 s) loading while the user was scrolled down: the monotonic/no-jump checks
      // above cover it; this line only records whether it actually happened during this run.
      console.log(`[${name} ${vp}] INFO  weather/AQ arrived ${weatherArrivedMidScroll ? 'while scrolled down' : 'before scrolling'}`)

      // 3. Back to the top with the wheel; the search and Analyse controls are back in place
      for (let i = 0; i < 120 && (await st()) > 0; i++) {
        await page.mouse.wheel(0, -160)
        await page.waitForTimeout(40)
      }
      check(vp, 'wheel scrolling returns to the top', (await st()) === 0)
      const ctl = await page.getByRole('button', { name: /Analyse again/ }).boundingBox()
      check(vp, 'controls visible at the top again', ctl && ctl.y > 0 && ctl.y < h)

      if (desktop) {
        // 4. Scrollbar dragging (Chromium/Edge render a draggable scrollbar on the panel's right edge)
        const gutter = await page.locator('.workspace__panel').evaluate((el) => el.offsetWidth - el.clientWidth - 1)
        if (gutter <= 0 || name !== 'chromium' || process.env.HEADED !== '1') {
          console.log(`[${name} ${vp}] SKIP  scrollbar drag — needs HEADED=1 Chromium (headless engines draw overlay scrollbars with no draggable track)`)
        } else {
          const p = await page.locator('.workspace__panel').boundingBox()
          const x = p.x + p.width - 1 - gutter / 2  // middle of the scrollbar track (inside the 1 px border)
          const hit = await page.evaluate(([x, y]) => { const t = document.elementFromPoint(x, y); return `${t?.tagName}.${String(t?.className).slice(0, 30)}` }, [x, p.y + 30])
          await page.mouse.move(x, p.y + 30)
          await page.mouse.down()
          await page.mouse.move(x, p.y + p.height * 0.6, { steps: 12 })
          await page.mouse.up()
          const dragged = await st()
          check(vp, 'scrollbar drag scrolls the panel', dragged > 50, `scrollTop ${dragged}, gutter ${gutter}, hit ${hit}, headless ${process.env.HEADED !== '1'}`)
          await page.evaluate((s) => (document.querySelector(s).scrollTop = 0), scroller)
        }
        // 5. Keyboard scrolling when the panel is focused
        await page.locator('.workspace__panel').evaluate((el) => { el.tabIndex = -1; el.focus() })
        await page.waitForTimeout(150)
        await page.keyboard.press('PageDown')
        await page.waitForFunction((s) => document.querySelector(s).scrollTop > 100, scroller, { timeout: 2000 }).catch(() => {})
        check(vp, 'keyboard PageDown scrolls the panel', (await st()) > 100, `scrollTop ${await st()}`)
        // PageDown animates (smooth scroll): wait until it has settled before resetting
        for (let last = -1, i = 0; i < 30; i++) {
          const y = await st()
          if (y === last) break
          last = y
          await page.waitForTimeout(80)
        }
        await page.evaluate((s) => document.querySelector(s).scrollTo({ top: 0, behavior: 'instant' }), scroller)
        await page.waitForTimeout(150)

        // 6. Map: wheel over the map zooms the MAP and does not scroll the panel; drag pans it
        const m = await mapBox()
        // Zoom level = z of the newest OpenStreetMap tile requests (…/{z}/{x}/{y}.png)
        const tileZ = () => page.evaluate(() => Math.max(-1, ...[...document.querySelectorAll('.leaflet-tile-pane img')]
          .map((i) => Number((i.getAttribute('src') || '').split('/').slice(-3)[0])).filter(Number.isFinite)))
        const zoom0 = await tileZ()
        // a point on the map away from the analysed raster, legend, chip and controls
        await page.mouse.move(m.x + m.width * 0.5, m.y + m.height * 0.12)
        for (let i = 0; i < 3; i++) {
          await page.mouse.wheel(0, -200)
          await page.waitForTimeout(120)
        }
        await page.waitForFunction((z) => Math.max(...[...document.querySelectorAll('.leaflet-tile-pane img')]
          .map((i) => Number((i.getAttribute('src') || '').split('/').slice(-3)[0])).filter(Number.isFinite)) > z, zoom0,
        { timeout: 4000 }).catch(() => {})
        const zoom1 = await tileZ()
        check(vp, 'wheel over the map zooms the map, not the panel', zoom1 > zoom0 && (await st()) === 0, `tile zoom ${zoom0} -> ${zoom1}, panel ${await st()}`)
        await page.waitForTimeout(800) // let the zoom animation finish: Leaflet ignores drags while animating
        const pane0 = await page.evaluate(() => document.querySelector('.leaflet-map-pane').style.transform)
        await page.mouse.move(m.x + m.width / 2, m.y + m.height / 2)
        await page.mouse.down()
        await page.mouse.move(m.x + m.width / 2 + 120, m.y + m.height / 2 + 60, { steps: 8 })
        await page.mouse.up()
        await page.waitForTimeout(300)
        const pane1 = await page.evaluate(() => document.querySelector('.leaflet-map-pane').style.transform)
        check(vp, 'map drag pans the map', pane0 !== pane1)
        const leafletSize = await page.evaluate(() => {
          const c = document.querySelector('.leaflet-container').getBoundingClientRect()
          const w = document.querySelector('.workspace__map').getBoundingClientRect()
          return Math.abs(c.width - w.width) + Math.abs(c.height - w.height)
        })
        check(vp, 'Leaflet fills its container exactly', leafletSize <= 1, `${leafletSize}px difference`)
        check(vp, 'layer control reachable', await page.getByRole('button', { name: 'Layers' }).isVisible())
      }

      // 7. Leave and come back: page starts at the top with the controls showing
      if (desktop) await page.evaluate((s) => (document.querySelector(s).scrollTop = 400), scroller)
      else await page.evaluate(() => window.scrollTo(0, 400))
      await page.evaluate(() => (window.location.hash = '#/weather'))
      await page.getByRole('heading', { name: 'Current conditions' }).waitFor({ timeout: 30000 })
      await page.evaluate(() => (window.location.hash = '#/'))
      await page.locator('article.result').waitFor({ timeout: 20000 })
      await page.waitForTimeout(300)
      check(vp, 'returning to Flood analysis starts at the top', (await st()) === 0 && (await page.evaluate(() => window.scrollY)) === 0)
      check(vp, 'no runtime errors', errors.length === 0, errors[0] || '')
    } catch (e) {
      check(vp, 'test completed without exception', false, e.message.split('\n')[0])
    }
    await ctx.close()
  }
  await browser.close()
  console.log(`[${name}] ${pass}/${pass + fail} checks passed\n`)
  return fail === 0
}

const which = process.argv[2] || 'chromium'
let ok = true
for (const e of which === 'all' ? Object.keys(ENGINES) : [which]) ok = (await run(e)) && ok
process.exit(ok ? 0 : 1)
