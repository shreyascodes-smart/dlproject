import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown, Info, LoaderCircle, X } from 'lucide-react'

/* ───────────── Tooltip ─────────────
 * Hover or keyboard focus shows it; Escape hides it. Rendered in a portal and
 * positioned against the viewport so it is never clipped by scroll areas. */
export function Tooltip({ content, children, side = 'top' }) {
  const id = useId()
  const anchor = useRef(null)
  const tip = useRef(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState(null)

  useLayoutEffect(() => {
    if (!open || !anchor.current || !tip.current) return
    const a = anchor.current.getBoundingClientRect()
    const t = tip.current.getBoundingClientRect()
    const gap = 8
    let top = side === 'bottom' ? a.bottom + gap : a.top - t.height - gap
    if (top < 8) top = a.bottom + gap
    if (top + t.height > window.innerHeight - 8) top = a.top - t.height - gap
    let left = a.left + a.width / 2 - t.width / 2
    left = Math.max(8, Math.min(left, window.innerWidth - t.width - 8))
    setPos({ top, left })
  }, [open, side, content])

  useEffect(() => {
    if (!open) return undefined
    const onKey = (e) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  if (!content) return children
  return (
    <span
      ref={anchor}
      className="tip-anchor"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      aria-describedby={open ? id : undefined}
    >
      {children}
      {open &&
        createPortal(
          <span
            ref={tip}
            id={id}
            role="tooltip"
            className="tooltip"
            style={pos ? { top: pos.top, left: pos.left } : { visibility: 'hidden', top: 0, left: 0 }}
          >
            {content}
          </span>,
          document.body,
        )}
    </span>
  )
}

/** Small focusable "i" button that explains a term. */
export function InfoTip({ label, children }) {
  return (
    <Tooltip content={children}>
      <button type="button" className="info-tip" aria-label={`About ${label}`}>
        <Info size={13} aria-hidden="true" />
      </button>
    </Tooltip>
  )
}

/* ───────────── Segmented control (radio group with arrow-key navigation) ───────────── */
export function Segmented({ label, options, value, onChange, size = 'md', className = '' }) {
  const refs = useRef([])
  const enabled = options.filter((o) => !o.disabled)
  const onKey = (e, i) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) return
    e.preventDefault()
    const order = options.map((o, idx) => ({ ...o, idx })).filter((o) => !o.disabled)
    const cur = order.findIndex((o) => o.idx === i)
    let next = cur
    if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = order.length - 1
    else next = (cur + (e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 1) + order.length) % order.length
    const target = order[next]
    onChange(target.value)
    refs.current[target.idx]?.focus()
  }
  return (
    <div role="radiogroup" aria-label={label} className={`segmented segmented--${size} ${className}`}>
      {options.map((o, i) => {
        const checked = o.value === value
        const btn = (
          <button
            key={o.value}
            ref={(el) => (refs.current[i] = el)}
            type="button"
            role="radio"
            aria-checked={checked}
            disabled={o.disabled}
            tabIndex={checked || (!enabled.some((x) => x.value === value) && i === 0) ? 0 : -1}
            className={`segmented__item ${checked ? 'is-active' : ''}`}
            onClick={() => onChange(o.value)}
            onKeyDown={(e) => onKey(e, i)}
          >
            {o.icon}
            <span>{o.label}</span>
          </button>
        )
        return o.tooltip ? (
          <Tooltip key={o.value} content={o.tooltip}>
            {btn}
          </Tooltip>
        ) : (
          btn
        )
      })}
    </div>
  )
}

/* ───────────── Disclosure ───────────── */
/** lazy: render (and therefore fetch) the content only while open. */
export function Disclosure({ title, meta, defaultOpen = false, tone, lazy = false, children }) {
  const [open, setOpen] = useState(defaultOpen)
  const id = useId()
  return (
    <section className={`disclosure ${open ? 'is-open' : ''} ${tone ? `disclosure--${tone}` : ''}`}>
      <h3 className="disclosure__heading">
        <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen((v) => !v)}>
          <span className="disclosure__title">{title}</span>
          {meta != null && <span className="disclosure__meta">{meta}</span>}
          <ChevronDown size={16} className="disclosure__chev" aria-hidden="true" />
        </button>
      </h3>
      <div id={id} className="disclosure__body" hidden={!open}>
        {lazy && !open ? null : children}
      </div>
    </section>
  )
}

/* ───────────── Dialog (native <dialog>: focus trap + Escape for free) ───────────── */
export function Dialog({ open, onClose, title, children, footer, variant = 'center', className = '' }) {
  const ref = useRef(null)
  const titleId = useId()

  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (open && !el.open) {
      if (typeof el.showModal === 'function') el.showModal()
      else el.setAttribute('open', '')
    } else if (!open && el.open) {
      if (typeof el.close === 'function') el.close()
      else el.removeAttribute('open')
    }
  }, [open])

  const onCancel = useCallback(
    (e) => {
      e.preventDefault()
      onClose()
    },
    [onClose],
  )

  return (
    <dialog
      ref={ref}
      className={`dialog dialog--${variant} ${className}`}
      aria-labelledby={titleId}
      onCancel={onCancel}
      onClick={(e) => {
        if (e.target === ref.current) onClose()
      }}
    >
      {open && (
        <div className="dialog__panel">
          <header className="dialog__header">
            <h2 id={titleId}>{title}</h2>
            <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
              <X size={18} aria-hidden="true" />
            </button>
          </header>
          <div className="dialog__body">{children}</div>
          {footer && <footer className="dialog__footer">{footer}</footer>}
        </div>
      )}
    </dialog>
  )
}

export function Spinner({ size = 16, label }) {
  return (
    <LoaderCircle
      size={size}
      className="spin"
      role={label ? 'status' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : 'true'}
    />
  )
}

export function Badge({ tone = 'neutral', children, title }) {
  return (
    <span className={`badge badge--${tone}`} title={title}>
      {children}
    </span>
  )
}

export function SplitBadge({ split }) {
  return split === 'test' ? (
    <Badge tone="test" title="Held-out test sample: illustration only, never used for tuning">
      Test
    </Badge>
  ) : (
    <Badge tone="val" title="Validation sample">
      Val
    </Badge>
  )
}
