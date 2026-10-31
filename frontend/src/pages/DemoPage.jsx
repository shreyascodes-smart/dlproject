import { useMemo, useState } from 'react'
import { Archive, RefreshCw, Search, X } from 'lucide-react'
import FloodMap from '../components/FloodMap'
import ResultPanel, { OutcomeNotice } from '../components/ResultPanel'
import { Badge, Spinner } from '../components/ui'
import { Elapsed, ResultSkeleton } from '../components/feedback'
import { describeError, freshness } from '../lib/assessment'
import { bboxToBounds } from '../lib/map'
import { fmtDate } from '../lib/format'
import { filterSamples, placeName } from '../lib/samples'

function SamplePicker({ catalogue, selectedId, loadingId, onSelect }) {
  const [q, setQ] = useState('')
  const visible = useMemo(() => filterSamples(catalogue.samples, { query: q }), [catalogue.samples, q])
  if (catalogue.state === 'loading' || catalogue.state === 'idle') {
    return (
      <p className="samples__state">
        <Spinner size={15} /> Loading demo scenes…
      </p>
    )
  }
  if (catalogue.state === 'error') {
    return (
      <div className="samples__state" role="alert">
        <span>{catalogue.error?.message || 'Demo scenes could not be loaded.'}</span>
        <button type="button" className="btn btn--sm" onClick={catalogue.retry}>
          <RefreshCw size={14} aria-hidden="true" /> Retry
        </button>
      </div>
    )
  }
  return (
    <div className="samples">
      <label className="field field--sm">
        <Search size={14} aria-hidden="true" />
        <span className="sr-only">Filter demo scenes</span>
        <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter by place, event or date" />
        {q && (
          <button type="button" className="field__clear" onClick={() => setQ('')} aria-label="Clear filter">
            <X size={13} aria-hidden="true" />
          </button>
        )}
      </label>
      <ul className="samples__list" aria-label="Demo scenes">
        {visible.map((s) => (
          <li key={s.id}>
            <button type="button" className={`sample ${s.id === selectedId ? 'is-selected' : ''}`} aria-pressed={s.id === selectedId} onClick={() => onSelect(s.id)}>
              <span className="sample__name">{placeName(s)}</span>
              <span className="sample__meta">
                {s.event_id} · {fmtDate(s.s2_date)}
              </span>
              {loadingId === s.id ? <Spinner size={13} /> : <Badge tone={s.split === 'test' ? 'test' : 'val'}>{s.split === 'test' ? 'Test' : 'Val'}</Badge>}
            </button>
          </li>
        ))}
        {visible.length === 0 && <li className="samples__state">No scenes match “{q}”.</li>}
      </ul>
    </div>
  )
}

/** Explicit demonstration mode: archived WorldFloods scenes of past events (validation + held-out test). */
export default function DemoPage({ demo, catalogue, theme, basemap, onBasemap, navigate, status }) {
  const [sampleId, setSampleId] = useState(() => demo.request?.sample_id || null)
  const [fitNonce, setFitNonce] = useState(0)
  const loading = demo.status === 'loading'
  const sample = catalogue.samples.find((s) => s.id === sampleId) || null
  const data = demo.data?.mode === 'demo' && demo.data.sample?.id === sampleId ? demo.data : null
  const error = demo.status === 'error' && demo.request?.sample_id === sampleId ? describeError(demo.error, { live: false }) : null
  const run = (id) => demo.run({ mode: 'demo', sample_id: id, method: 'tiled' }, { cacheKey: `${id}|tiled` })
  const select = (id) => {
    setSampleId(id)
    run(id)
  }
  const fresh = data ? freshness(data.freshness) : null
  const notReady = status.state !== 'ready' && status.state !== 'loading'

  return (
    <div className="workspace workspace--demo">
      <div className="workspace__panel">
        <section className="panel__controls" aria-labelledby="demo-title">
          <p className="mode-banner">
            <Archive size={14} aria-hidden="true" /> Demonstration mode
          </p>
          <h1 className="panel__title" id="demo-title">
            Demo scenes
          </h1>
          <p className="panel__lead">
            Archived Sentinel-2 scenes of past flood events, with Copernicus EMS flood maps for comparison. Each scene shows its own location, not
            the place you selected.
          </p>
          {notReady && <p className="panel__blocker">The analysis service is unavailable right now. It is re-checked automatically.</p>}
          <SamplePicker catalogue={catalogue} selectedId={sampleId} loadingId={loading ? demo.request?.sample_id : null} onSelect={select} />
        </section>
        <section className="panel__scroll" aria-label="Demo result" id="demo-results" tabIndex={-1}>
          {error && <OutcomeNotice outcome={error} onRetry={() => run(sampleId)} />}
          {loading && !data && <ResultSkeleton />}
          {data && <ResultPanel data={data} stale={loading} onShowOnMap={() => setFitNonce((n) => n + 1)} onRefresh={() => run(sampleId)} onNavigate={navigate} />}
          {!data && !error && !loading && (
            <section className="howto">
              <h2>Pick a scene</h2>
              <p className="muted">
                Validation scenes were used to choose the model and its outcome thresholds. Held-out test scenes were never used for tuning and are
                shown for illustration only.
              </p>
            </section>
          )}
        </section>
      </div>
      <section className="workspace__map" aria-label="Map">
        <FloodMap
          marker={sample?.center || null}
          area={!data?.map && sample ? bboxToBounds(sample.bbox) : null}
          layers={data?.map || null}
          busy={loading}
          theme={theme}
          basemap={basemap}
          onBasemap={onBasemap}
          fitNonce={fitNonce}
          chip={
            data ? (
              <>
                <Archive size={13} aria-hidden="true" /> Archived · {fresh.date}
              </>
            ) : null
          }
          label={sample ? `Map of the demo scene ${placeName(sample)}` : 'Map of demo scenes'}
        >
          {loading && (
            <div className="map-float map-float--top">
              <Elapsed startedAt={demo.startedAt} label="Analysing demo scene" hint="large scenes take up to ~40 s the first time" />
            </div>
          )}
        </FloodMap>
      </section>
    </div>
  )
}
