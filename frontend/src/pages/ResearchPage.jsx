import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, Check, Columns2, Copy, FlaskConical, RefreshCw, TriangleAlert } from 'lucide-react'
import SceneViewer from '../components/SceneViewer'
import Inspector from '../components/Inspector'
import { Badge, Segmented, Spinner } from '../components/ui'
import { getReadiness, getResearch } from '../lib/api'
import { LAYER_LABEL, REFERENCE_LAYERS } from '../lib/layers'
import { fmtInt, fmtPct, fmtRatio } from '../lib/format'

function useLoad(fn, deps) {
  const [s, setS] = useState({ status: 'loading', data: null, error: null })
  const [nonce, setNonce] = useState(0)
  useEffect(() => {
    const ctrl = new AbortController()
    setS((x) => ({ ...x, status: 'loading', error: null }))
    fn({ signal: ctrl.signal })
      .then((data) => setS({ status: 'success', data, error: null }))
      .catch((e) => e?.name !== 'AbortError' && setS({ status: 'error', data: null, error: e }))
    return () => ctrl.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce, ...deps])
  return { ...s, reload: useCallback(() => setNonce((n) => n + 1), []) }
}

const CHECK_TONE = (v) => {
  const s = String(v)
  if (/^(ok|present|loaded|configured|reachable)$/.test(s)) return 'good'
  if (/^(skipped|not_configured)$/.test(s) || s.startsWith('segmentation_')) return 'neutral'
  return 'bad'
}

function Readiness() {
  const [deep, setDeep] = useState(false)
  const r = useLoad((o) => getReadiness({ ...o, deep }), [deep])
  return (
    <section className="card" aria-labelledby="ready-title">
      <header className="card__head">
        <h2 id="ready-title">Service readiness</h2>
        {r.data && <Badge tone={r.data.ready ? 'good' : 'bad'}>{r.data.ready ? 'Ready' : 'Not ready'}</Badge>}
        <div className="card__actions">
          <button type="button" className="btn btn--xs" onClick={() => (deep ? r.reload() : setDeep(true))} disabled={r.status === 'loading'}>
            {r.status === 'loading' ? <Spinner size={12} /> : <Activity size={12} aria-hidden="true" />} {deep ? 'Re-run external checks' : 'Check external services'}
          </button>
        </div>
      </header>
      {r.status === 'error' && (
        <p className="card__error" role="alert">
          <TriangleAlert size={15} aria-hidden="true" /> {r.error?.message}
        </p>
      )}
      {r.data && (
        <dl className="checks">
          {Object.entries(r.data.checks || {}).map(([k, v]) => (
            <div key={k} className={`check check--${CHECK_TONE(v)}`}>
              <dt>{k.replaceAll('_', ' ')}</dt>
              <dd>{String(v).replaceAll('_', ' ')}</dd>
            </div>
          ))}
        </dl>
      )}
      {r.data?.error && <p className="muted small">{r.data.error}</p>}
    </section>
  )
}

const METRICS = [
  ['flood_iou', 'Flood IoU'],
  ['flood_precision', 'Precision'],
  ['flood_recall', 'Recall'],
  ['flood_dice', 'Dice'],
]

function perEventMean(perEvent) {
  const vals = Object.values(perEvent || {})
    .map((m) => m?.flood_iou)
    .filter((v) => typeof v === 'number')
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
}

function ModelComparison({ research }) {
  const [method, setMethod] = useState('tiled')
  const models = research.data?.models || []
  const cols = models.flatMap((m) => ['val', 'test'].map((split) => ({ m, split, s: m[split]?.[method] })))
  return (
    <section className="card card--wide" aria-labelledby="cmp-title">
      <header className="card__head">
        <h2 id="cmp-title">Model comparison</h2>
        <span className="card__meta">Full-resolution, product protocol (positives outside JRC permanent water vs EMS flood)</span>
        <Segmented
          label="Inference method"
          size="sm"
          value={method}
          onChange={setMethod}
          options={[
            { value: 'tiled', label: 'Tiled' },
            { value: 'resize', label: 'Resize' },
          ]}
        />
      </header>
      {research.status === 'loading' && (
        <p className="card__loading">
          <Spinner size={15} /> Loading evaluation summaries…
        </p>
      )}
      {research.status === 'error' && (
        <p className="card__error" role="alert">
          <TriangleAlert size={15} aria-hidden="true" /> {research.error?.message}
        </p>
      )}
      {research.data && (
        <>
          <div className="dtable-wrap" tabIndex={0} role="region" aria-label="Model comparison table">
            <table className="dtable dtable--cmp">
              <thead>
                <tr>
                  <th scope="col">Metric</th>
                  {cols.map(({ m, split }) => (
                    <th key={`${m.version}${split}`} scope="col">
                      {m.version.replace('segmentation_', '')} · {split}
                      {m.active && split === 'val' && <span className="dtable__tag">active</span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRICS.map(([k, label]) => (
                  <tr key={k}>
                    <th scope="row">{label}</th>
                    {cols.map(({ m, split, s }) => (
                      <td key={`${m.version}${split}`} className="num">
                        {fmtRatio(s?.pixel_metrics_excluding_known_water?.[k], 3)}
                      </td>
                    ))}
                  </tr>
                ))}
                <tr>
                  <th scope="row">Flood IoU, raw (no water set aside)</th>
                  {cols.map(({ m, split, s }) => (
                    <td key={`${m.version}${split}`} className="num">
                      {fmtRatio(s?.pixel_metrics_raw?.flood_iou, 3)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <th scope="row">Mean per-event IoU</th>
                  {cols.map(({ m, split, s }) => (
                    <td key={`${m.version}${split}`} className="num">
                      {fmtRatio(perEventMean(s?.per_event_excluding_known_water), 3)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <th scope="row">Max share on no-flood scenes</th>
                  {cols.map(({ m, split, s }) => (
                    <td key={`${m.version}${split}`} className="num">
                      {fmtPct(s?.negative_outside_water_pct?.max, 2)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <th scope="row">Scenes</th>
                  {cols.map(({ m, split }) => (
                    <td key={`${m.version}${split}`} className="num">
                      {fmtInt(m[`${split}_n_samples`])}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <p className="muted small">{research.data.protocol} Cells show “n/a” where an evaluation has not been run.</p>
          <div className="model-cards">
            {models.map((m) => (
              <div key={m.version} className={`model-card ${m.active ? 'is-active' : ''}`}>
                <p>
                  <strong>{m.version}</strong> {m.active && <Badge tone="good">Active</Badge>} {!m.installed && <Badge tone="neutral">Not installed</Badge>}
                </p>
                <dl className="kv">
                  {Object.entries(m.training || {}).map(([k, v]) => (
                    <div key={k} className="kv__row">
                      <dt>{k.replaceAll('_', ' ')}</dt>
                      <dd>{Array.isArray(v) ? v.join(', ') : typeof v === 'number' ? (Number.isInteger(v) ? fmtInt(v) : v.toFixed(4)) : String(v)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

function Policy({ research }) {
  const p = research.data?.policy
  return (
    <section className="card" aria-labelledby="pol-title">
      <header className="card__head">
        <h2 id="pol-title">Outcome policy</h2>
        {p?.version && <span className="card__meta">{p.version}</span>}
      </header>
      {p ? (
        <dl className="kv">
          <div className="kv__row">
            <dt>“None” below</dt>
            <dd>{p.none_below_pct}% flood-like share</dd>
          </div>
          <div className="kv__row">
            <dt>Inconclusive up to</dt>
            <dd>{Object.entries(p.inconclusive_ceiling_pct || {}).map(([k, v]) => `${v}% (${k})`).join(' · ')}</dd>
          </div>
          <div className="kv__row">
            <dt>Minimum usable land</dt>
            <dd>{Math.round(p.min_assessable_land_frac * 100)}% of the image</dd>
          </div>
          <div className="kv__row">
            <dt>Water-dominated</dt>
            <dd>more than {Math.round(p.water_dominated_frac * 100)}% known water</dd>
          </div>
          {p.provenance && (
            <div className="kv__row">
              <dt>Provenance</dt>
              <dd>{p.provenance}</dd>
            </div>
          )}
        </dl>
      ) : (
        <p className="muted small">{research.status === 'loading' ? 'Loading…' : 'No validated policy for the active model: outcomes are reported as inconclusive.'}</p>
      )}
    </section>
  )
}

const PAIRS = { 'overlay|reference': ['overlay', 'reference'], 'rgb|overlay': ['rgb', 'overlay'] }

function stripImages(d) {
  if (!d) return null
  const { images, map, ...rest } = d
  return {
    ...rest,
    images: images && { ...Object.fromEntries(Object.entries(images).map(([k, v]) => [k, typeof v === 'string' && v.startsWith('data:') ? `<${v.length} chars>` : v])) },
    map: map && { ...map, satellite: map.satellite && '<image>', flood: map.flood && '<image>', set_aside: map.set_aside && '<image>' },
  }
}

function Diagnostics({ data, loading, startedAt, onRerun, onShowOnMap }) {
  const [viewChoice, setView] = useState('overlay')
  const [pairChoice, setPair] = useState('overlay|reference')
  const [copied, setCopied] = useState(false)
  const hasRef = !!data?.reference?.available
  const view = !hasRef && REFERENCE_LAYERS.has(viewChoice) ? 'overlay' : viewChoice
  const pairKey = !hasRef && pairChoice === 'overlay|reference' ? 'rgb|overlay' : pairChoice
  const json = useMemo(() => JSON.stringify(stripImages(data), null, 2), [data])
  const wc = data?.water_context
  const basis = data?.assessment?.basis

  if (!data) {
    return (
      <section className="card card--wide" aria-labelledby="diag-title">
        <header className="card__head">
          <h2 id="diag-title">Last assessment diagnostics</h2>
        </header>
        <p className="muted">Run a flood analysis or open a demo scene; its raw prediction, reference comparison and processing details appear here.</p>
      </section>
    )
  }
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(json)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable */
    }
  }
  return (
    <section className="card card--wide" aria-labelledby="diag-title">
      <header className="card__head">
        <h2 id="diag-title">Last assessment diagnostics</h2>
        <span className="card__meta">
          {data.mode === 'live' ? 'Live image' : `Demo · ${data.sample?.id}`} · method {data.method}
        </span>
        <Segmented
          label="Inference method"
          size="sm"
          value={data.method}
          onChange={onRerun}
          options={[
            { value: 'tiled', label: 'Tiled', tooltip: '256 px tiles at native 10 m resolution (training scale)' },
            { value: 'resize', label: 'Resize', tooltip: 'Whole scene shrunk to 256 × 256 px (Phase H protocol)' },
          ]}
        />
      </header>
      <div className="diag">
        <div className="diag__viewer">
          <div className="diag__bar">
            <Segmented
              label="Layer"
              value={view}
              onChange={setView}
              size="sm"
              options={[
                { value: 'rgb', label: 'Satellite' },
                { value: 'overlay', label: 'Raw prediction' },
                { value: 'reference', label: 'EMS reference', disabled: !hasRef },
                { value: 'error', label: 'Errors', disabled: !hasRef },
                { value: 'compare', label: 'Compare', icon: <Columns2 size={13} aria-hidden="true" /> },
              ]}
            />
            {view === 'compare' && (
              <Segmented
                label="Compare layers"
                size="sm"
                value={pairKey}
                onChange={setPair}
                options={[
                  { value: 'overlay|reference', label: 'Prediction ↔ Reference', disabled: !hasRef },
                  { value: 'rgb|overlay', label: 'Satellite ↔ Prediction' },
                ]}
              />
            )}
          </div>
          <div className="diag__stage">
            <SceneViewer
              images={data.images}
              view={view}
              pair={PAIRS[pairKey]}
              sceneKey={`${data.mode}|${data.sample?.id || data.imagery?.center?.lat}`}
              loading={loading}
              startedAt={startedAt}
              loadingLabel={`Running the model (${data.method})`}
            />
          </div>
          <p className="muted small">
            {LAYER_LABEL[view] || 'Compare'} · raw model output (before setting aside known water) · scroll or +/− to zoom, drag to pan
          </p>
        </div>
        <div className="diag__side">
          <section className="dev-block">
            <h3>Assessment rule</h3>
            <dl className="kv">
              {[
                ['Level', data.assessment?.level],
                ['Flood-like outside known water', fmtPct(wc?.flood_like_outside_known_water_pct, 2)],
                ['Raw positives (all eligible px)', fmtPct(wc?.raw_flood_like_pct, 2)],
                ['Positives on known water / sea', fmtPct(wc?.flood_like_on_known_water_share_pct, 1)],
                ['Known water in scene', fmtPct(wc?.known_water_pct_of_eligible, 1)],
                ['Policy', basis?.policy_version || 'none'],
                ['Water reference', wc?.source || 'not available'],
              ].map(([k, v]) => (
                <div key={k} className="kv__row">
                  <dt>{k}</dt>
                  <dd>{v ?? '—'}</dd>
                </div>
              ))}
            </dl>
          </section>
          <Inspector data={data} stale={loading} onShowOnMap={onShowOnMap} />
        </div>
      </div>
      <details className="rawjson">
        <summary>Response JSON (images omitted)</summary>
        <button type="button" className="btn btn--xs" onClick={copy}>
          {copied ? <Check size={12} aria-hidden="true" /> : <Copy size={12} aria-hidden="true" />} {copied ? 'Copied' : 'Copy'}
        </button>
        <pre tabIndex={0}>{json}</pre>
      </details>
    </section>
  )
}

/** Developer / research dashboard: readiness, model registry and evaluations, diagnostics of the last result. */
export default function ResearchPage({ status, last, loading, startedAt, onRerun, onShowOnMap }) {
  const research = useLoad((o) => getResearch(o), [])
  const st = status.data
  const t = st?.model?.reported_test_metrics
  return (
    <div className="page page--research">
      <header className="page__head">
        <div className="page__title">
          <p className="eyebrow">
            <FlaskConical size={13} aria-hidden="true" /> Developer &amp; research
          </p>
          <h1>Research dashboard</h1>
          <p className="page__sub">Model registry, evaluation summaries and raw diagnostics. Intended for developers and reviewers.</p>
        </div>
        <div className="page__tools">
          <button type="button" className="btn btn--sm" onClick={() => { status.retry(); research.reload() }}>
            <RefreshCw size={14} aria-hidden="true" /> Reload
          </button>
        </div>
      </header>
      <div className="rs-grid">
        <Readiness />
        <section className="card" aria-labelledby="active-title">
          <header className="card__head">
            <h2 id="active-title">Active model</h2>
            {st?.model_version && <span className="card__meta">{st.model_version}</span>}
          </header>
          {st?.model ? (
            <dl className="kv">
              {[
                ['Architecture', `${st.model.architecture} · ${st.model.encoder} · epoch ${st.model.epoch}`],
                ['Target', st.model.target],
                ['Bands', (st.model.bands || []).join(', ')],
                ['Preprocessing', st.model.preprocessing],
                ['Device', st.model.device],
                ['Policy', st.policy_version || 'none'],
                ['Held-out test IoU', t ? `${fmtRatio(t.flood_iou)} (P ${fmtRatio(t.flood_precision)}, R ${fmtRatio(t.flood_recall)}, ${t.n_samples} scenes)` : 'not evaluated'],
                ['Legacy classifier', st.model.legacy_classifier],
              ].map(([k, v]) => (
                <div key={k} className="kv__row">
                  <dt>{k}</dt>
                  <dd>{v ?? '—'}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="muted small">{status.state === 'loading' ? 'Loading…' : status.data?.error || status.error?.message || 'Model not loaded.'}</p>
          )}
        </section>
        <Policy research={research} />
        <ModelComparison research={research} />
        <Diagnostics data={last} loading={loading} startedAt={startedAt} onRerun={onRerun} onShowOnMap={onShowOnMap} />
      </div>
    </div>
  )
}
