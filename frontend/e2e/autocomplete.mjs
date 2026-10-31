/**
 * Real-browser check of place autocomplete (real Photon, real Leaflet map). Needs `npm run dev` (+ backend).
 * Run: node e2e/autocomplete.mjs
 */
import { chromium } from 'playwright-core'
import { existsSync } from 'node:fs'

const APP = process.env.APP_URL || 'http://localhost:5173'
const edge = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Google/Chrome/Application/chrome.exe'].find(existsSync)
let pass = 0
let fail = 0
const check = (tag, n, ok, info = '') => {
  if (ok) pass++
  else fail++
  console.log(`[${tag}] ${ok ? 'PASS' : 'FAIL'}  ${n}${info ? `  — ${info}` : ''}`)
}

const browser = await chromium.launch({ executablePath: edge, headless: true })
for (const [w, h, theme] of [[1440, 900, 'dark'], [390, 844, 'light']]) {
  const tag = `${w}x${h} ${theme}`
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, colorScheme: theme })
  await ctx.addInitScript((t) => localStorage.setItem('di-theme', t), theme)
  const page = await ctx.newPage()
  const photon = []
  page.on('request', (r) => r.url().includes('photon.komoot.io') && photon.push(r.url()))
  const errors = []
  page.on('pageerror', (e) => errors.push(String(e)))
  try {
    await page.goto(`${APP}/#/`, { waitUntil: 'domcontentloaded' })
    const box = page.getByRole('combobox', { name: 'Search for a place' })
    await box.waitFor({ timeout: 30000 })
    const scrollBefore = await page.evaluate(() => [document.querySelector('.workspace__panel')?.scrollTop ?? 0, window.scrollY])

    await box.click()
    await box.pressSequentially('Thakur College', { delay: 40 })
    const first = page.getByRole('option').first()
    await first.waitFor({ timeout: 20000 })
    check(tag, 'one debounced request for 14 keystrokes', photon.length === 1, `${photon.length} Photon requests`)
    const names = await page.locator('.lsearch__name').allTextContents()
    check(tag, 'real suggestions for "Thakur College"', names.some((n) => /Thakur College of Science/i.test(n)), names.slice(0, 3).join(' | '))

    // Dropdown overlays content below (not clipped, no layout shift)
    const menu = await page.locator('.lsearch__menu').boundingBox()
    const top = await page.evaluate(([x, y]) => document.elementFromPoint(x, y)?.closest('.lsearch__menu') != null, [menu.x + menu.width / 2, menu.y + menu.height - 6])
    check(tag, 'dropdown fully visible above surrounding content', top && menu.y + menu.height <= h, `bottom ${Math.round(menu.y + menu.height)}`)

    // Keyboard: ↓ then Enter picks the 2nd suggestion
    await page.keyboard.press('ArrowDown')
    const picked = (await page.locator('.lsearch__opt.is-active .lsearch__name').textContent()).trim()
    await page.keyboard.press('Enter')
    await page.locator('.place-chip').waitFor({ timeout: 5000 })
    check(tag, 'Enter selects the highlighted suggestion', (await box.inputValue()) === picked, picked)
    check(tag, 'dropdown closed after selection', (await page.locator('.lsearch__opt').count()) === 0)
    const loc = await page.evaluate(() => JSON.parse(localStorage.getItem('di-location')))
    check(tag, 'exact provider coordinates stored', Math.abs(loc.lat - 19.206) < 0.01 && Math.abs(loc.lng - 72.87) < 0.01, `${loc.lat}, ${loc.lng}`)

    // Map moved there and the 5 × 5 km area is drawn around the point
    await page.waitForTimeout(1200)
    const area = await page.locator('path.fmap__area').boundingBox()
    const pin = await page.locator('.map-pin__dot').boundingBox()
    const inside = area && pin && pin.x > area.x && pin.x + pin.width < area.x + area.width && pin.y > area.y && pin.y + pin.height < area.y + area.height
    check(tag, 'map moved: 5 × 5 km area drawn around the selected point', !!inside)
    check(tag, 'nothing analysed automatically', !(await page.locator('article.result').count()))
    const scrollAfter = await page.evaluate(() => [document.querySelector('.workspace__panel')?.scrollTop ?? 0, window.scrollY])
    check(tag, 'selection did not move the panel/page scroll', JSON.stringify(scrollAfter) === JSON.stringify(scrollBefore), `${scrollBefore} -> ${scrollAfter}`)

    // Escape closes, and a second query works ("Bandra": no duplicate station stops)
    await box.fill('')
    await box.pressSequentially('Bandra', { delay: 40 })
    await page.getByRole('option').first().waitFor({ timeout: 20000 })
    const bandra = await page.locator('.lsearch__name').allTextContents()
    check(tag, '"Bandra" suggestions are distinct', new Set(bandra).size === bandra.length && bandra.length >= 2, bandra.join(' | '))
    await page.keyboard.press('Escape')
    check(tag, 'Escape closes the dropdown', (await page.locator('.lsearch__opt').count()) === 0)

    if (w < 600) {
      // touch: tap a suggestion
      await box.fill('')
      await box.pressSequentially('Mumbai Airport', { delay: 40 })
      const opt = page.getByRole('option', { name: /International Airport/ }).first()
      await opt.waitFor({ timeout: 20000 })
      await opt.tap().catch(() => opt.click())
      check(tag, 'tap selects a suggestion on mobile', /Airport/.test(await box.inputValue()), await box.inputValue())
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
      check(tag, 'no horizontal overflow', overflow <= 1, `${overflow}px`)
    }
    check(tag, 'no runtime errors', errors.length === 0, errors[0] || '')
  } catch (e) {
    check(tag, 'completed without exception', false, e.message.split('\n')[0])
  }
  await ctx.close()
}
await browser.close()
console.log(`${pass}/${pass + fail} checks passed`)
process.exit(fail ? 1 : 0)
