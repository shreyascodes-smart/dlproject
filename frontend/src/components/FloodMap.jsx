import { useEffect, useId, useState } from 'react'
import { ImageOverlay, MapContainer, Marker, Rectangle, TileLayer, ZoomControl, useMap, useMapEvents } from 'react-leaflet'
import { Layers, X } from 'lucide-react'
import 'leaflet/dist/leaflet.css'
import { DEFAULT_CENTER, DEFAULT_ZOOM, pinIcon, wrapLng } from '../lib/map'

const OSM_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
const OSM_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
// Keyless basemaps only (CARTO raster tiles now answer "API key required").
// "Minimal" is the OpenStreetMap standard style, muted with a CSS filter on the tile layer only
// (dark: inverted; light: desaturated) — analysis overlays are in another pane and keep their colours.
const BASEMAPS = {
  minimal: { label: 'Minimal', url: OSM_URL, attribution: OSM_ATTR, className: 'tiles--minimal' },
  streets: { label: 'Streets', url: OSM_URL, attribution: OSM_ATTR, className: 'tiles--streets' },
}

const SENTINEL_ATTR = 'Contains modified Copernicus Sentinel data'
const JRC_ATTR = 'Permanent water: EC JRC / Google'

/**
 * In the single-column layout (< 1024 px) the page itself scrolls and the full-width map passes under
 * the pointer: Leaflet's wheel zoom would then swallow the wheel and trap the page. There, wheel input
 * is left to the page; the map still zooms with +/−, pinch and double-click, and still drags.
 * On desktop the map sits beside its own scroll panel, so wheel zoom stays on.
 */
const SINGLE_COLUMN = '(max-width: 1023px)'

function WheelPolicy() {
  const map = useMap()
  useEffect(() => {
    const mql = window.matchMedia?.(SINGLE_COLUMN)
    if (!mql) return undefined
    const apply = () => (mql.matches ? map.scrollWheelZoom.disable() : map.scrollWheelZoom.enable())
    apply()
    mql.addEventListener('change', apply)
    return () => mql.removeEventListener('change', apply)
  }, [map])
  return null
}

function Clicks({ onPick }) {
  useMapEvents({ click: (e) => onPick?.({ lat: e.latlng.lat, lng: wrapLng(e.latlng.lng) }) })
  return null
}

function SizeWatcher() {
  const map = useMap()
  useEffect(() => {
    const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => map.invalidateSize({ pan: false }))
    ro?.observe(map.getContainer())
    return () => {
      ro?.disconnect()
      // Leaflet 1.9 finishes an animated zoom in an untracked 250 ms timeout; if the map is removed
      // meanwhile (e.g. the user switches page), that timeout throws "_leaflet_pos of undefined".
      // Marking the animation as done makes the pending callback a no-op. (Do not call map.stop()
      // here: the map may already be removed, and stopping a pan animation then throws.)
      map._animatingZoom = false
    }
  }, [map])
  return null
}

/**
 * Fit the view when the analysed raster / pending area changes or when asked to (`nonce`).
 * Never on every render, so the user can pan and zoom freely.
 */
function Fit({ bounds, center, nonce }) {
  const map = useMap()
  const key = bounds
    ? bounds.flat().map((v) => v.toFixed(5)).join(',')
    : center
      ? `${center[0].toFixed(5)},${center[1].toFixed(5)}`
      : ''
  useEffect(() => {
    if (!key) return undefined
    const nums = key.split(',').map(Number)
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    const t = setTimeout(() => {
      map.invalidateSize()
      if (nums.length === 4) {
        map.fitBounds(
          [
            [nums[0], nums[1]],
            [nums[2], nums[3]],
          ],
          { padding: [28, 28], animate: !reduce, maxZoom: 15 },
        )
      } else map.setView(nums, Math.max(map.getZoom(), 12), { animate: !reduce })
    }, 50)
    return () => clearTimeout(t)
  }, [key, nonce, map])
  return null
}

function LayerPanel({ basemap, onBasemap, show, setShow, opacity, setOpacity, layers }) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const toggle = (k) => (e) => setShow((s) => ({ ...s, [k]: e.target.checked }))
  return (
    <div className={`map-panel ${open ? 'is-open' : ''}`}>
      <div className="map-panel__head">
        {open && <span className="map-panel__title">Map layers</span>}
        <button type="button" className="map-panel__toggle" aria-expanded={open} aria-controls={id} onClick={() => setOpen((v) => !v)}>
          {open ? <X size={15} aria-hidden="true" /> : <Layers size={15} aria-hidden="true" />}
          <span>{open ? 'Close' : 'Layers'}</span>
        </button>
      </div>
      <div id={id} className="map-panel__body" hidden={!open}>
        <fieldset>
          <legend>Basemap</legend>
          <div className="map-panel__seg">
            {Object.entries(BASEMAPS).map(([k, b]) => (
              <label key={k} className={basemap === k ? 'is-active' : ''}>
                <input type="radio" name={`${id}-basemap`} value={k} checked={basemap === k} onChange={() => onBasemap(k)} />
                {b.label}
              </label>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>Analysis layers</legend>
          <label className={`map-check ${!layers?.flood ? 'is-disabled' : ''}`}>
            <input type="checkbox" checked={show.flood} onChange={toggle('flood')} disabled={!layers?.flood} />
            <span className="swatch swatch--flood" aria-hidden="true" /> Flood-like water
          </label>
          <label className={`map-check ${!layers?.set_aside ? 'is-disabled' : ''}`}>
            <input type="checkbox" checked={show.setAside} onChange={toggle('setAside')} disabled={!layers?.set_aside} />
            <span className="swatch swatch--aside" aria-hidden="true" /> Known water (set aside)
          </label>
          <label className={`map-check ${!layers?.satellite ? 'is-disabled' : ''}`}>
            <input type="checkbox" checked={show.satellite} onChange={toggle('satellite')} disabled={!layers?.satellite} />
            <span className="swatch swatch--sat" aria-hidden="true" /> Satellite image
          </label>
          <label className="map-check">
            <input type="checkbox" checked={show.area} onChange={toggle('area')} />
            <span className="swatch swatch--area" aria-hidden="true" /> Analysis area outline
          </label>
        </fieldset>
        <label className="map-range">
          <span>
            Satellite image opacity <output className="num">{Math.round(opacity * 100)}%</output>
          </span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            disabled={!layers?.satellite || !show.satellite}
          />
        </label>
      </div>
    </div>
  )
}

/**
 * Assessment map: theme-aware minimal or street basemap, the analysed 5 × 5 km area, and (after an
 * assessment) the georeferenced satellite image, flood-like water and set-aside known water.
 * The view fits the analysed raster's real bounds.
 */
export default function FloodMap({
  marker,
  area,
  layers,
  onPick,
  busy,
  label,
  theme,
  basemap,
  onBasemap,
  fitNonce = 0,
  chip,
  children,
}) {
  const [show, setShow] = useState({ flood: true, setAside: true, satellite: true, area: true })
  const [opacity, setOpacity] = useState(1)
  const bm = BASEMAPS[basemap] || BASEMAPS.minimal
  const tileKey = `${basemap}-${theme}`
  const resultBounds = layers?.bounds || null
  const fitBounds = resultBounds || area || null
  const outline = area || resultBounds

  return (
    <div className={`fmap ${busy ? 'is-busy' : ''}`} role="region" aria-label={label || 'Map'}>
      <MapContainer
        center={marker ? [marker.lat, marker.lng] : DEFAULT_CENTER}
        zoom={marker ? 12 : DEFAULT_ZOOM}
        minZoom={2}
        worldCopyJump
        zoomControl={false}
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer key={tileKey} url={bm.url} attribution={bm.attribution} className={bm.className} maxZoom={19} />
        <ZoomControl position="bottomright" />
        <SizeWatcher />
        <WheelPolicy />
        <Fit bounds={fitBounds} center={marker && !fitBounds ? [marker.lat, marker.lng] : null} nonce={fitNonce} />
        {onPick && <Clicks onPick={onPick} />}
        {layers?.satellite && show.satellite && (
          <ImageOverlay url={layers.satellite} bounds={resultBounds} opacity={opacity} zIndex={2} attribution={SENTINEL_ATTR} />
        )}
        {layers?.set_aside && show.setAside && (
          <ImageOverlay url={layers.set_aside} bounds={resultBounds} opacity={0.9} zIndex={3} attribution={JRC_ATTR} />
        )}
        {layers?.flood && show.flood && <ImageOverlay url={layers.flood} bounds={resultBounds} opacity={0.88} zIndex={4} />}
        {outline && show.area && (
          <Rectangle
            bounds={outline}
            pathOptions={{ color: 'var(--map-area)', weight: 2, fill: false, dashArray: busy || !layers ? '6 6' : null }}
            className={busy ? 'fmap__area is-busy' : 'fmap__area'}
            interactive={false}
          />
        )}
        {marker && <Marker position={[marker.lat, marker.lng]} icon={pinIcon} keyboard={false} interactive={false} />}
      </MapContainer>

      <LayerPanel
        basemap={basemap}
        onBasemap={onBasemap}
        show={show}
        setShow={setShow}
        opacity={opacity}
        setOpacity={setOpacity}
        layers={layers}
      />

      {chip && <div className="map-chip">{chip}</div>}

      {layers && (
        <ul className="map-legend" aria-label="Map legend">
          {show.flood && (
            <li>
              <span className="swatch swatch--flood" aria-hidden="true" /> Flood-like water
            </li>
          )}
          {layers.set_aside && show.setAside && (
            <li>
              <span className="swatch swatch--aside" aria-hidden="true" /> Known water / sea (set aside)
            </li>
          )}
          {show.area && (
            <li>
              <span className="swatch swatch--area" aria-hidden="true" /> Analysed area
            </li>
          )}
        </ul>
      )}
      {children}
    </div>
  )
}
