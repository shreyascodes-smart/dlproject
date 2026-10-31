import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ImageOff, Maximize2, Minimize2, Minus, Plus, RotateCcw, Satellite } from 'lucide-react'
import { Spinner, Tooltip } from './ui'
import { LAYER_LABEL } from '../lib/layers'

const MIN_ZOOM = 1
const MAX_ZOOM = 8

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

function Layer({ src, name, style, onError, failed }) {
  if (!src || failed) {
    return (
      <div className="viewer__missing" style={style}>
        <ImageOff size={22} aria-hidden="true" />
        <span>{LAYER_LABEL[name]} unavailable</span>
      </div>
    )
  }
  return (
    <img
      key={src.length + src.slice(-32)}
      className="viewer__layer"
      src={src}
      alt={LAYER_LABEL[name]}
      style={style}
      draggable={false}
      onError={onError}
    />
  )
}

/**
 * Satellite scene viewer.
 *   - single layer, or a draggable comparison slider between two layers
 *   - wheel / buttons / keyboard zoom (1x–8x) around the cursor, drag to pan
 *   - fullscreen via the Fullscreen API
 * Zoom is kept when switching layers (to compare one area) and reset when the scene changes.
 */
export default function SceneViewer({ images, view, pair, sceneKey, loading, startedAt, loadingLabel, overlayInfo }) {
  const stageRef = useRef(null)
  const rootRef = useRef(null)
  const [box, setBox] = useState({ w: 0, h: 0 })
  const [t, setT] = useState({ k: 1, x: 0, y: 0 })
  const [split, setSplit] = useState(0.5)
  const [failed, setFailed] = useState({})
  const [isFs, setIsFs] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const [prevScene, setPrevScene] = useState(sceneKey)
  const drag = useRef(null)

  // Reset zoom and failed-image flags when the scene changes (adjusting state during render)
  if (prevScene !== sceneKey) {
    setPrevScene(sceneKey)
    setT({ k: 1, x: 0, y: 0 })
    setFailed({})
  }

  const iw = images?.display_width || 0
  const ih = images?.display_height || 0

  // Fit the image into the stage
  useLayoutEffect(() => {
    const el = stageRef.current
    if (!el) return undefined
    const measure = () => setBox({ w: el.clientWidth, h: el.clientHeight })
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const scale = iw && ih && box.w && box.h ? Math.min(box.w / iw, box.h / ih) : 0
  const fw = iw * scale
  const fh = ih * scale
  const ox = (box.w - fw) / 2
  const oy = (box.h - fh) / 2

  const clampT = useCallback(
    (n) => {
      const k = clamp(n.k, MIN_ZOOM, MAX_ZOOM)
      return { k, x: clamp(n.x, fw * (1 - k), 0), y: clamp(n.y, fh * (1 - k), 0) }
    },
    [fw, fh],
  )

  const zoomAt = useCallback(
    (factor, cx = fw / 2, cy = fh / 2) => {
      setT((p) => {
        const k = clamp(p.k * factor, MIN_ZOOM, MAX_ZOOM)
        const r = k / p.k
        return clampT({ k, x: cx - (cx - p.x) * r, y: cy - (cy - p.y) * r })
      })
    },
    [clampT, fw, fh],
  )

  // Wheel zoom (non-passive so the page does not scroll while zooming the image)
  useEffect(() => {
    const el = stageRef.current
    if (!el) return undefined
    const onWheel = (e) => {
      if (!scale) return
      e.preventDefault()
      const r = el.getBoundingClientRect()
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left - ox, e.clientY - r.top - oy)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [zoomAt, scale, ox, oy])

  useEffect(() => {
    const onFs = () => setIsFs(document.fullscreenElement === rootRef.current)
    document.addEventListener('fullscreenchange', onFs)
    return () => document.removeEventListener('fullscreenchange', onFs)
  }, [])

  useEffect(() => {
    if (!loading || !startedAt) return undefined
    const id = setInterval(() => setNow(Date.now()), 500)
    return () => clearInterval(id)
  }, [loading, startedAt])
  const elapsed = startedAt ? Math.max(0, Math.floor((now - startedAt) / 1000)) : 0

  const compare = view === 'compare' && pair?.length === 2
  const [leftName, rightName] = compare ? pair : [view, view]

  const stagePoint = (e) => {
    const r = stageRef.current.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }

  const onPointerDown = (e) => {
    if (e.button !== 0 || !scale) return
    const onHandle = e.target.closest?.('[data-split-handle]')
    const p = stagePoint(e)
    drag.current = onHandle ? { type: 'split' } : { type: 'pan', sx: p.x, sy: p.y, tx: t.x, ty: t.y }
    stageRef.current.setPointerCapture?.(e.pointerId)
  }
  const onPointerMove = (e) => {
    const d = drag.current
    if (!d) return
    const p = stagePoint(e)
    if (d.type === 'split') setSplit(clamp((p.x - ox) / fw, 0, 1))
    else if (t.k > 1) setT(clampT({ k: t.k, x: d.tx + p.x - d.sx, y: d.ty + p.y - d.sy }))
  }
  const onPointerUp = (e) => {
    drag.current = null
    stageRef.current?.releasePointerCapture?.(e.pointerId)
  }

  const onSplitKey = (e) => {
    const step = e.shiftKey ? 0.1 : 0.02
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') setSplit((s) => clamp(s - step, 0, 1))
    else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') setSplit((s) => clamp(s + step, 0, 1))
    else if (e.key === 'Home') setSplit(0)
    else if (e.key === 'End') setSplit(1)
    else return
    e.preventDefault()
  }

  const onStageKey = (e) => {
    if (e.target !== e.currentTarget) return
    const pan = 40
    if (e.key === '+' || e.key === '=') zoomAt(1.25)
    else if (e.key === '-') zoomAt(0.8)
    else if (e.key === '0') setT({ k: 1, x: 0, y: 0 })
    else if (e.key === 'ArrowLeft') setT((p) => clampT({ ...p, x: p.x + pan }))
    else if (e.key === 'ArrowRight') setT((p) => clampT({ ...p, x: p.x - pan }))
    else if (e.key === 'ArrowUp') setT((p) => clampT({ ...p, y: p.y + pan }))
    else if (e.key === 'ArrowDown') setT((p) => clampT({ ...p, y: p.y - pan }))
    else return
    e.preventDefault()
  }

  const toggleFs = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen()
      else await rootRef.current?.requestFullscreen?.()
    } catch {
      /* fullscreen not permitted: ignore */
    }
  }

  // Split position in image space (handles zoom/pan)
  const splitScreenX = ox + split * fw
  const clipPct = fw ? clamp(((split * fw - t.x) / t.k / fw) * 100, 0, 100) : 50
  const layerTransform = { transform: `translate(${t.x}px, ${t.y}px) scale(${t.k})` }
  const markFail = (name) => () => setFailed((f) => ({ ...f, [name]: true }))
  const canFs = typeof document !== 'undefined' && document.fullscreenEnabled

  return (
    <div ref={rootRef} className={`viewer ${isFs ? 'is-fullscreen' : ''}`}>
      <div
        ref={stageRef}
        className={`viewer__stage ${t.k > 1 ? 'is-zoomed' : ''} ${loading ? 'is-loading' : ''}`}
        tabIndex={images ? 0 : -1}
        role="group"
        aria-label={
          images
            ? `${compare ? `${LAYER_LABEL[leftName]} compared with ${LAYER_LABEL[rightName]}` : LAYER_LABEL[view]}. Use plus and minus to zoom, arrow keys to pan, 0 to reset.`
            : 'Scene viewer'
        }
        onKeyDown={onStageKey}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={(e) => {
          if (e.target.closest?.('[data-split-handle]') || !scale) return
          const p = stagePoint(e)
          zoomAt(t.k >= 4 ? 1 / t.k : 2, p.x - ox, p.y - oy)
        }}
      >
        {!images && !loading && (
          <div className="viewer__empty">
            <Satellite size={28} aria-hidden="true" />
            <p>Select a scene to analyse</p>
          </div>
        )}

        {images && scale > 0 && (
          <div className="viewer__canvas" style={{ left: ox, top: oy, width: fw, height: fh }}>
            <Layer
              name={rightName}
              src={images[rightName]}
              failed={failed[rightName]}
              onError={markFail(rightName)}
              style={layerTransform}
            />
            {compare && (
              <Layer
                name={leftName}
                src={images[leftName]}
                failed={failed[leftName]}
                onError={markFail(leftName)}
                style={{ ...layerTransform, clipPath: `inset(0 ${100 - clipPct}% 0 0)` }}
              />
            )}
          </div>
        )}

        {compare && images && scale > 0 && (
          <>
            <span className="viewer__tag viewer__tag--left" style={{ left: ox + 10, top: oy + 10 }}>
              {LAYER_LABEL[leftName]}
            </span>
            <span className="viewer__tag viewer__tag--right" style={{ right: box.w - ox - fw + 10, top: oy + 10 }}>
              {LAYER_LABEL[rightName]}
            </span>
            <div
              data-split-handle
              className="viewer__split"
              style={{ left: splitScreenX, top: oy, height: fh }}
              role="slider"
              tabIndex={0}
              aria-label={`Comparison position: left shows ${LAYER_LABEL[leftName]}, right shows ${LAYER_LABEL[rightName]}`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(split * 100)}
              aria-valuetext={`${Math.round(split * 100)}% ${LAYER_LABEL[leftName]}`}
              onKeyDown={onSplitKey}
            >
              <span className="viewer__split-knob" data-split-handle aria-hidden="true">
                <svg viewBox="0 0 16 16" width="14" height="14">
                  <path d="M6 4 2 8l4 4M10 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.6" />
                </svg>
              </span>
            </div>
          </>
        )}

        {loading && (
          <div className="viewer__progress" role="status" aria-live="polite">
            <div className="progress-card">
              <Spinner size={18} />
              <div>
                <strong>{loadingLabel || 'Analysing scene'}</strong>
                <span>
                  {elapsed}s elapsed
                  {elapsed >= 4 && ' · large scenes take up to ~40 s the first time, then load instantly'}
                </span>
              </div>
            </div>
            <div className="progress-bar" aria-hidden="true">
              <span />
            </div>
          </div>
        )}

        {overlayInfo && !loading && <div className="viewer__info">{overlayInfo}</div>}
      </div>

      <div className="viewer__zoom" aria-label="Zoom controls">
        <Tooltip content="Zoom in (+)">
          <button type="button" className="icon-btn" onClick={() => zoomAt(1.4)} disabled={!images || t.k >= MAX_ZOOM} aria-label="Zoom in">
            <Plus size={16} aria-hidden="true" />
          </button>
        </Tooltip>
        <span className="viewer__zoom-level" aria-live="polite">
          {t.k.toFixed(t.k < 10 ? 1 : 0)}×
        </span>
        <Tooltip content="Zoom out (−)">
          <button type="button" className="icon-btn" onClick={() => zoomAt(1 / 1.4)} disabled={!images || t.k <= MIN_ZOOM} aria-label="Zoom out">
            <Minus size={16} aria-hidden="true" />
          </button>
        </Tooltip>
        <Tooltip content="Reset view (0)">
          <button type="button" className="icon-btn" onClick={() => setT({ k: 1, x: 0, y: 0 })} disabled={!images || t.k === 1} aria-label="Reset zoom">
            <RotateCcw size={15} aria-hidden="true" />
          </button>
        </Tooltip>
        {canFs && (
          <Tooltip content={isFs ? 'Exit full screen' : 'Full screen'}>
            <button type="button" className="icon-btn" onClick={toggleFs} disabled={!images} aria-label={isFs ? 'Exit full screen' : 'Full screen'}>
              {isFs ? <Minimize2 size={15} aria-hidden="true" /> : <Maximize2 size={15} aria-hidden="true" />}
            </button>
          </Tooltip>
        )}
      </div>
    </div>
  )
}
