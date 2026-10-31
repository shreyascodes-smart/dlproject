import { useCallback, useId, useLayoutEffect, useMemo, useRef, useState } from 'react'

/** Width of an element, kept up to date with a ResizeObserver. */
function useElementWidth(fallback = 640) {
  const ref = useRef(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return undefined
    const measure = () => setWidth(el.clientWidth || el.getBoundingClientRect?.().width || fallback)
    measure()
    if (typeof ResizeObserver === 'undefined') return undefined
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [fallback])
  return [ref, width || fallback]
}

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)

function extent(values, { min, max, pad = 0.08, floor } = {}) {
  const nums = values.filter(isNum)
  let lo = min ?? (nums.length ? Math.min(...nums) : 0)
  let hi = max ?? (nums.length ? Math.max(...nums) : 1)
  if (lo === hi) {
    lo -= 1
    hi += 1
  }
  const span = hi - lo
  if (min == null) lo -= span * pad
  if (max == null) hi += span * pad
  if (floor != null) lo = Math.max(floor, lo)
  return [lo, hi]
}

function ticks([lo, hi], n = 4) {
  const raw = (hi - lo) / n
  const mag = 10 ** Math.floor(Math.log10(Math.abs(raw) || 1))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw
  const out = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(6)))
  return out
}

/**
 * Time-series chart (SVG, no dependencies).
 *  - series: [{ key, label, values, color, type: 'line' | 'bar', axis: 'left' | 'right', format }]
 *  - Hover or focus + arrow keys inspect individual points; values are announced to screen readers.
 *  - Every chart is paired with a data table alternative by the caller.
 */
export function TimeChart({ times, series, nowIndex = null, height = 200, leftDomain, rightDomain, xLabel, label, formatTime }) {
  const [wrapRef, width] = useElementWidth()
  const [hover, setHover] = useState(null)
  const descId = useId()
  const n = times?.length || 0
  const hasRight = series.some((s) => s.axis === 'right')
  const pad = { l: 40, r: hasRight ? 40 : 14, t: 14, b: 26 }
  const plotW = Math.max(10, width - pad.l - pad.r)
  const plotH = Math.max(10, height - pad.t - pad.b)
  const step = n > 1 ? plotW / (n - 1) : plotW
  const x = useCallback((i) => pad.l + (n > 1 ? i * step : plotW / 2), [pad.l, n, step, plotW])

  const left = useMemo(
    () => extent(series.filter((s) => s.axis !== 'right').flatMap((s) => s.values), leftDomain || {}),
    [series, leftDomain],
  )
  const right = useMemo(
    () => (hasRight ? extent(series.filter((s) => s.axis === 'right').flatMap((s) => s.values), rightDomain || {}) : null),
    [series, rightDomain, hasRight],
  )
  const y = (v, axis) => {
    const [lo, hi] = axis === 'right' ? right : left
    return pad.t + plotH - ((v - lo) / (hi - lo)) * plotH
  }

  const xTicks = useMemo(() => {
    const out = []
    const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(plotW / 64))))
    // Hourly ISO times: align ticks to round hours (00, 06, 12, 18 …) so midnight gets the day label
    const hours = (times || []).map((t) => (typeof t === 'string' && t.length >= 13 ? Number(t.slice(11, 13)) : NaN))
    if (hours.length && hours.every(Number.isFinite)) {
      const stepH = [1, 2, 3, 6, 12, 24].find((s) => s >= every) || 24
      hours.forEach((h, i) => h % stepH === 0 && out.push(i))
      return out
    }
    for (let i = 0; i < n; i += every) out.push(i)
    return out
  }, [n, plotW, times])

  if (!n) return <p className="muted small">No data available.</p>

  const fmtT = formatTime || ((t) => t)
  const pick = (clientX) => {
    const r = wrapRef.current?.getBoundingClientRect()
    if (!r) return
    const i = Math.round((clientX - r.left - pad.l) / step)
    setHover(Math.max(0, Math.min(n - 1, i)))
  }
  const onKey = (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault()
      setHover((h) => {
        const base = h ?? nowIndex ?? 0
        return Math.max(0, Math.min(n - 1, base + (e.key === 'ArrowRight' ? 1 : -1)))
      })
    } else if (e.key === 'Home') setHover(0)
    else if (e.key === 'End') setHover(n - 1)
    else if (e.key === 'Escape') setHover(null)
  }
  const barW = Math.max(2, Math.min(18, step * 0.62))
  const tipLeft = hover != null ? Math.min(Math.max(x(hover) + 10, 4), width - 170) : 0
  const flip = hover != null && x(hover) > width - 190

  return (
    <div className="chart" ref={wrapRef}>
      <div
        className="chart__plot"
        tabIndex={0}
        role="group"
        aria-label={`${label}. Use the left and right arrow keys to read values.`}
        aria-describedby={descId}
        onPointerMove={(e) => pick(e.clientX)}
        onPointerLeave={() => setHover(null)}
        onKeyDown={onKey}
        onBlur={() => setHover(null)}
      >
        <svg width={width} height={height} aria-hidden="true" focusable="false">
          {ticks(left).map((v) => (
            <g key={`l${v}`}>
              <line className="chart__grid" x1={pad.l} x2={pad.l + plotW} y1={y(v, 'left')} y2={y(v, 'left')} />
              <text className="chart__axis" x={pad.l - 6} y={y(v, 'left') + 4} textAnchor="end">
                {series.find((s) => s.axis !== 'right')?.tick?.(v) ?? v}
              </text>
            </g>
          ))}
          {right &&
            ticks(right).map((v) => (
              <text key={`r${v}`} className="chart__axis" x={pad.l + plotW + 6} y={y(v, 'right') + 4} textAnchor="start">
                {series.find((s) => s.axis === 'right')?.tick?.(v) ?? v}
              </text>
            ))}
          {series
            .filter((s) => s.type === 'bar')
            .map((s) => (
              <g key={s.key} fill={s.color} className="chart__bars">
                {s.values.map((v, i) => {
                  if (!isNum(v) || v <= 0) return null
                  const top = y(v, s.axis)
                  const base = y((s.axis === 'right' ? right : left)[0], s.axis)
                  return <rect key={i} x={x(i) - barW / 2} y={top} width={barW} height={Math.max(0, base - top)} rx={1.5} />
                })}
              </g>
            ))}
          {series
            .filter((s) => s.type !== 'bar')
            .map((s) => {
              let d = ''
              let pen = false
              s.values.forEach((v, i) => {
                if (!isNum(v)) {
                  pen = false
                  return
                }
                d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v, s.axis).toFixed(1)}`
                pen = true
              })
              return <path key={s.key} d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
            })}
          {nowIndex != null && nowIndex >= 0 && nowIndex < n && (
            <g className="chart__now">
              <line x1={x(nowIndex)} x2={x(nowIndex)} y1={pad.t} y2={pad.t + plotH} />
              <text x={x(nowIndex) + 4} y={pad.t + 10}>Now</text>
            </g>
          )}
          {xTicks.map((i) => (
            <text key={`x${i}`} className="chart__axis" x={x(i)} y={height - 8} textAnchor="middle">
              {fmtT(times[i], i, 'tick')}
            </text>
          ))}
          {hover != null && (
            <g className="chart__hover">
              <line x1={x(hover)} x2={x(hover)} y1={pad.t} y2={pad.t + plotH} />
              {series
                .filter((s) => s.type !== 'bar' && isNum(s.values[hover]))
                .map((s) => (
                  <circle key={s.key} cx={x(hover)} cy={y(s.values[hover], s.axis)} r={4} fill={s.color} />
                ))}
            </g>
          )}
        </svg>
        {hover != null && (
          <div className={`chart__tip ${flip ? 'is-flipped' : ''}`} style={flip ? { left: x(hover) - 170, top: 8 } : { left: tipLeft, top: 8 }}>
            <strong>{fmtT(times[hover], hover, 'full')}</strong>
            {series.map((s) => (
              <span key={s.key}>
                <i style={{ background: s.color }} aria-hidden="true" />
                {s.label}: {s.format ? s.format(s.values[hover]) : (s.values[hover] ?? '—')}
              </span>
            ))}
          </div>
        )}
      </div>
      <p id={descId} className="sr-only" aria-live="polite">
        {hover != null
          ? `${fmtT(times[hover], hover, 'full')}: ${series.map((s) => `${s.label} ${s.format ? s.format(s.values[hover]) : (s.values[hover] ?? 'no data')}`).join(', ')}`
          : ''}
      </p>
      <ul className="chart__legend" aria-hidden="true">
        {series.map((s) => (
          <li key={s.key}>
            <i className={s.type === 'bar' ? 'is-bar' : ''} style={{ background: s.color }} />
            {s.label}
          </li>
        ))}
        {xLabel && <li className="chart__xlabel">{xLabel}</li>}
      </ul>
    </div>
  )
}

/** Accessible data table alternative for a chart. */
export function DataTable({ caption, columns, rows, highlight }) {
  return (
    <div className="dtable-wrap" tabIndex={0} role="region" aria-label={caption}>
      <table className="dtable">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.id ?? i} className={highlight === i ? 'is-now' : ''}>
              {columns.map((c, j) =>
                j === 0 ? (
                  <th key={c.key} scope="row">
                    {r[c.key]}
                  </th>
                ) : (
                  <td key={c.key}>{r[c.key]}</td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Horizontal range bar for the daily min–max temperature rows. */
export function RangeBar({ min, max, lo, hi, label }) {
  if (!isNum(min) || !isNum(max) || !isNum(lo) || !isNum(hi) || hi <= lo) return <span className="rangebar" aria-hidden="true" />
  const a = ((min - lo) / (hi - lo)) * 100
  const b = ((max - lo) / (hi - lo)) * 100
  return (
    <span className="rangebar" role="img" aria-label={label}>
      <span className="rangebar__fill" style={{ left: `${a}%`, width: `${Math.max(3, b - a)}%` }} />
    </span>
  )
}

/** Semi-circular gauge for an index value (e.g. AQI). `bands`: [{ max, color }] */
export function Gauge({ value, max, bands, label, size = 132 }) {
  const r = size / 2 - 10
  const cx = size / 2
  const cy = size / 2
  const arc = (from, to) => {
    const a0 = Math.PI * (1 - from)
    const a1 = Math.PI * (1 - to)
    const p0 = [cx + r * Math.cos(a0), cy - r * Math.sin(a0)]
    const p1 = [cx + r * Math.cos(a1), cy - r * Math.sin(a1)]
    return `M${p0[0].toFixed(2)},${p0[1].toFixed(2)} A${r},${r} 0 0 1 ${p1[0].toFixed(2)},${p1[1].toFixed(2)}`
  }
  const toOf = (b) => Math.min(1, (Number.isFinite(b.max) ? b.max : max) / max)
  const segs = bands.map((b, i) => ({ from: i ? toOf(bands[i - 1]) : 0, to: toOf(b), color: b.color })).filter((s) => s.to > s.from)
  const v = isNum(value) ? Math.max(0, Math.min(1, value / max)) : null
  const ang = v != null ? Math.PI * (1 - v) : null
  return (
    <svg className="gauge" width={size} height={size / 2 + 12} viewBox={`0 0 ${size} ${size / 2 + 12}`} role="img" aria-label={label}>
      {segs.map((s, i) => (
        <path key={i} d={arc(s.from + 0.004, s.to - 0.004)} stroke={s.color} strokeWidth={9} fill="none" strokeLinecap="butt" opacity={0.9} />
      ))}
      {ang != null && (
        <g>
          <circle cx={cx + r * Math.cos(ang)} cy={cy - r * Math.sin(ang)} r={7} className="gauge__knob" />
        </g>
      )}
    </svg>
  )
}
