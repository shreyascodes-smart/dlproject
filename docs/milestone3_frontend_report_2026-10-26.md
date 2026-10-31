# Milestone 3 — Frontend Integration & UI Redesign

> Date: 2026-10-26
> Status: **complete — verified against the running FastAPI backend in a real browser.**
> No retraining. Checkpoints, raw data, masks, manifest and Phase H reports unchanged (sizes/dates checked). Nothing committed.

## Outcome

The React app now uses the U-Net segmentation API end to end. It replaces the EfficientNet-B0 / Grad-CAM / "confidence" screens with a three-pane analysis workspace, and keeps weather, air quality, the Leaflet/OpenStreetMap map, geolocation and reverse geocoding.

| Check | Result |
|---|---|
| Frontend unit + integration tests (`npm test`) | **23 passed** |
| Lint (`npm run lint`, oxlint) | **0 warnings, 0 errors** |
| Production build (`npm run build`) | passes (444 kB JS, 137 kB gzip) |
| `npm audit` | **0 vulnerabilities** |
| Browser smoke test vs real backend (`node e2e/smoke.mjs`, Edge headless) | **35 / 35 checks passed**, 0 console errors, 0 failed requests |
| Backend + ML regression (`pytest backend/tests ml/tests`) | **176 passed** |

## Stack decisions

Kept React 19 + Vite + Leaflet/react-leaflet. Added only:

| Package | Why |
|---|---|
| `lucide-react` 1.61.0 | One consistent icon set (replaces emoji icons) |
| `@fontsource-variable/inter` 5.3.0 | Self-hosted Inter; no Google Fonts request |
| dev: `vitest` 5.0.5, `jsdom` 30.1.3, `@testing-library/*` | Frontend tests |
| dev: `playwright-core` 1.64.1 | Browser smoke test using the installed Edge/Chrome (no browser download) |

Removed `axios` (replaced by `fetch` with timeouts/abort; its `follow-redirects` dependency had a high-severity advisory). `npm audit fix` updated Vite 8.3.1 → 8.3.5 (dev-server advisories). No animation library: transitions are CSS, which keeps them cheap and makes `prefers-reduced-motion` handling simple. No TypeScript migration: it would not change the demo and would cost time before the deadline.

## What the user sees

**App bar:** brand, two sections (Flood analysis, Weather & air), location chip (opens map picker), live model status (`Model ready · GPU` / connecting / offline with reconnect), About dialog (model, held-out test metrics, limits).

**Flood analysis (desktop: scenes | viewer | results):**

- **Scene list:** 46 archived scenes grouped by event; search (event, place, date); All / Validation / Test filter; Val/Test badges; EMS reference flood % per scene; keyboard navigation (↑/↓/Home/End/Enter). Default selection is the first catalogue entry (not hand-picked).
- **Archive / Live switch:** Live is labelled *Experimental* and disabled with an explanation if Copernicus credentials are missing.
- **Viewer layers:** Satellite, Prediction, Reference, Errors, Compare (keys 1–5).
  - Compare: draggable before/after divider (mouse, touch, arrow keys), "Prediction ↔ Reference" or "Satellite ↔ Prediction".
  - Zoom 1–8× (wheel around the cursor, buttons, +/−/0 keys, double-click), drag to pan, full screen.
  - Zoom is kept when switching layers so the same area can be compared.
- **Method switch:** Tiled / Resize, each with a tooltip explaining the trade-off.
- **Legend** per layer, using the colours sent by the API; the reference layer states "Reference data, not a model output".
- **Loading:** progress overlay with elapsed seconds and a note that large scenes take up to ~40 s the first time; the previous result stays visible but dimmed.
- **Results panel:**
  - Scene name, split, event, imagery date, resolution, source, and location (opens a map showing the scene footprint).
  - Note that the scene shows the event's own location.
  - Predicted coverage % of eligible pixels, with its definition and a bar against the EMS reference %.
  - IoU / Dice / precision / recall with plain-English tooltips, and a correct / false alarm / missed pixel bar.
  - Expandable image quality, processing details (model, method, tiles, flood rule, device, and whether the result was computed, served from the server cache or from this session) and model limitations.
  - Disclaimer.

**Weather & air:** inline Leaflet map (click to move the location), My location, reverse-geocoded place name, current conditions, US/EU AQI with scale and pollutants, 7-day forecast with temperature-range and rain bars. A note states this data is not used by the flood model.

**Responsive:** 3-pane at ≥ 1024 px, narrower panes ≤ 1279 px; below 1024 px the scene list becomes a drawer, the viewer sits on top and the results below; no horizontal overflow at 390 px.

**Accessibility:**
- Semantic radio groups, listbox, slider, disclosure buttons.
- Native `<dialog>` (focus trap, Escape).
- Visible focus rings, skip link, ARIA labels on icon buttons, tooltips on focus as well as hover.
- `prefers-reduced-motion` respected.

## Engineering notes

- `src/lib/api.js` checks every response for the fields the UI depends on; network, timeout, HTTP and schema errors get distinct messages.
- `useAssessment` aborts superseded requests and ignores stale responses (tested), plus an 8-entry in-browser cache so revisiting a scene is instant.
- Weather/AQI results are stored with the location they belong to, so old data never shows for a new location.
- Images are the backend's ≤ 1024 px JPEGs; nothing large is re-encoded in the browser.
- No credentials in frontend code. The API base URL can be changed with `VITE_API_URL`.

## Fixes found by testing the running app

| Issue found in the browser | Fix |
|---|---|
| 32 px horizontal overflow on a 390 px phone | Constrained the grid column; compact app bar |
| React duplicate-key warning in the compare legend | Legend entries keyed by label |
| Leaflet's default white box around the custom pin | Override `.leaflet-div-icon` styles |
| Scene-footprint map opened over-zoomed (fit computed while the dialog was 0 × 0) | Re-measure, then fit bounds after the dialog opens |
| Long source text clipped in the results panel | Allow wrapping |
| Live result over the sea at Mumbai reported 75.7 % "flood" | Model limitation, not a UI bug. Added limitation: permanent water (sea, lakes, rivers) can be marked as flood. Cached results now always show current limitation wording. |

## Commands run

```powershell
cd frontend
npm test                    # 23 passed
npm run lint                # clean
npm run build               # ok
npm audit                   # 0 vulnerabilities
node e2e/smoke.mjs          # with backend + `npm run dev` running; LIVE=1 adds a real Copernicus request
cd ..
.\venv\Scripts\python.exe -m pytest backend/tests ml/tests   # 176 passed
```

The smoke test covers:
- initial load with an explicit `sample_id`
- 46-scene catalogue
- EMSR342 tiled IoU 0.432 matching the backend
- reference, error and compare layers, slider drag and keyboard, zoom
- resize method (IoU 0.266)
- split filter (22 test) and search
- test-split badge
- scene-map dialog with OSM tiles and the footprint fully visible; Escape closes it
- About dialog
- geolocation, weather, AQI, 7-day forecast, reverse geocoding (Mumbai), map-click relocation
- live mode: experimental label, reference layers disabled, no accuracy shown
- mobile drawer and overflow
- 1024 / 1280 px toolbars not clipped
- console errors and failed requests

Screenshots: `docs/figures/ui/` (desktop, compare, errors, map dialog, weather, live, mobile, drawer, tablet).

## Not verified / known limitations

- Tested in Chromium (Edge) only; Firefox and Safari not tested. Touch was emulated by viewport only, not on a real device. No screen-reader session was run.
- In development mode React StrictMode sends the first status/catalogue requests twice and cancels one; production builds do not.
- Reverse geocoding returns no name over open sea; the app shows coordinates instead.
- Live smoke runs made two real Copernicus requests with the credentials in `backend/.env` (only coordinates and dates sent).

## Repository notes (decide before committing)

- `frontend/dist/` was committed in an earlier commit even though `frontend/.gitignore` ignores it. `npm run build` therefore shows the old tracked files as modified/deleted. Recommended: stop tracking it (`git rm -r --cached frontend/dist`) when you next commit. I did not change git state.
- New screenshots in `docs/figures/ui/` are ~6.8 MB; `docs/figures/seg_inspection_*` are ~17 MB.

## Running the demo

```powershell
# terminal 1
cd "D:\superiorparab\final year DL project\backend"
..\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
# terminal 2
cd "D:\superiorparab\final year DL project\frontend"
npm run dev
# open http://localhost:5173
```

Remaining for Milestone 5: README and `backend/requirements.txt` updates.
