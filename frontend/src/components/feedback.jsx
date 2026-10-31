import { useEffect, useState } from 'react'
import { Spinner } from './ui'

/** Progress notice with elapsed seconds (no invented stages or percentages). */
export function Elapsed({ startedAt, label, hint }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500)
    return () => clearInterval(id)
  }, [])
  const s = startedAt ? Math.max(0, Math.floor((now - startedAt) / 1000)) : 0
  return (
    <div className="progress" role="status" aria-live="polite">
      <Spinner size={16} />
      <div>
        <strong>{label}</strong>
        <span className="num">
          {s}s{s >= 6 && hint ? ` · ${hint}` : ''}
        </span>
      </div>
      <div className="progress__bar" aria-hidden="true">
        <span />
      </div>
    </div>
  )
}

export function ResultSkeleton() {
  return (
    <div className="skeleton-card" aria-hidden="true">
      <span className="sk sk--sm" />
      <span className="sk sk--lg" />
      <span className="sk" />
      <span className="sk sk--block" />
      <span className="sk" />
    </div>
  )
}
