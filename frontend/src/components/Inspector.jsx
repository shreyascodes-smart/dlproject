import { CircleCheck, Clock, Database, FlaskConical, MapPin, TriangleAlert } from 'lucide-react'
import { Badge, Disclosure, InfoTip, SplitBadge, Tooltip } from './ui'
import { placeName } from '../lib/samples'
import { fmtCoord, fmtDate, fmtInt, fmtPct, fmtRatio, fmtSeconds, SPLIT_LABEL } from '../lib/format'

const METRIC_HELP = {
  flood_iou:
    'Intersection over Union: overlap between predicted flood and EMS-mapped flood, divided by their combined area. 1 is perfect, 0 is no overlap.',
  flood_dice: 'Dice / F1 score: like IoU but weights the overlap twice. Always ≥ IoU.',
  flood_precision: 'Of the pixels the model called flood, the share that EMS also maps as flood. Low precision = many false alarms.',
  flood_recall: 'Of the pixels EMS maps as flood, the share the model found. Low recall = missed flood.',
}

function Meta({ label, children }) {
  return (
    <div className="meta">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  )
}

function Bar({ value, max = 100, color, label }) {
  const pct = typeof value === 'number' ? Math.max(0, Math.min(100, (value / max) * 100)) : 0
  return (
    <div className="hbar" role="img" aria-label={`${label}: ${fmtPct(value)}`}>
      <span className="hbar__fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  )
}

function SourceNote({ data }) {
  const s = data._source
  if (s === 'browser') return <span className="source-note"><Database size={13} aria-hidden="true" /> Shown from this session</span>
  if (s === 'server-cache') return <span className="source-note"><Database size={13} aria-hidden="true" /> Loaded from server cache</span>
  return (
    <span className="source-note">
      <Clock size={13} aria-hidden="true" /> Computed in {fmtSeconds(data.processing_s)}
    </span>
  )
}

function ConfusionBar({ m, legend }) {
  const total = (m.tp || 0) + (m.fp || 0) + (m.fn || 0)
  if (!total) return <p className="muted small">No flood predicted or mapped in the supervised area.</p>
  const parts = [
    { k: 'tp', label: 'Correct flood', color: legend?.true_positive, n: m.tp },
    { k: 'fp', label: 'False alarm', color: legend?.false_positive, n: m.fp },
    { k: 'fn', label: 'Missed flood', color: legend?.false_negative, n: m.fn },
  ]
  return (
    <div className="confusion">
      <div className="confusion__bar" role="img" aria-label={parts.map((p) => `${p.label} ${fmtInt(p.n)} pixels`).join(', ')}>
        {parts.map((p) => (
          <span key={p.k} style={{ flexGrow: p.n, background: p.color }} />
        ))}
      </div>
      <ul className="confusion__legend">
        {parts.map((p) => (
          <li key={p.k}>
            <span className="swatch" style={{ background: p.color }} aria-hidden="true" />
            <span>{p.label}</span>
            <span className="num">{fmtInt(p.n)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function Inspector({ data, onShowOnMap, stale }) {
  if (!data) return null
  const { sample, imagery, prediction, reference, quality, model } = data
  const live = data.mode === 'live'
  const legend = data.images?.legend || {}
  const m = reference?.metrics
  const warnings = quality?.warnings || []
  const title = live ? 'Live Sentinel-2 scene' : placeName(sample)
  const center = imagery?.center

  return (
    <div className={`inspector ${stale ? 'is-stale' : ''}`} aria-busy={stale || undefined}>
      {/* ── Scene ── */}
      <section className="insp-block insp-scene">
        <div className="insp-scene__title">
          <h2>{title}</h2>
          <div className="insp-scene__badges">
            {live ? (
              <Badge tone="exp">
                <FlaskConical size={12} aria-hidden="true" /> Experimental
              </Badge>
            ) : (
              <SplitBadge split={sample.split} />
            )}
          </div>
        </div>
        <p className="insp-scene__sub">
          {live
            ? 'Least-cloudy mosaic around the selected location'
            : `${sample.event_id} · ${sample.event_type || 'Flood'} · ${SPLIT_LABEL[sample.split]}`}
        </p>
        <dl className="meta-grid">
          <Meta label="Imagery date">{live ? imagery?.date_range || '—' : fmtDate(imagery?.acquisition_date)}</Meta>
          <Meta label="Resolution">
            {imagery?.resolution_m ? `${imagery.resolution_m} m · ${fmtInt(imagery.width)}×${fmtInt(imagery.height)} px` : '—'}
          </Meta>
          <Meta label="Source">
            <span>{imagery?.source || '—'}</span>
          </Meta>
          <Meta label="Location">
            {center ? (
              <button type="button" className="link-btn" onClick={() => onShowOnMap(data)}>
                <MapPin size={13} aria-hidden="true" />
                {fmtCoord(center.lat, center.lng, 2)}
              </button>
            ) : (
              '—'
            )}
          </Meta>
        </dl>
        {!live && <p className="note">{imagery?.location_note}</p>}
      </section>

      {/* ── Coverage ── */}
      <section className="insp-block">
        <div className="insp-head">
          <h3>Raw model coverage</h3>
          <InfoTip label="raw model coverage">{prediction.coverage_definition}</InfoTip>
        </div>
        <div className="kpi">
          <span className="kpi__value num">{fmtPct(prediction.flood_coverage_pct, 1)}</span>
          <span className="kpi__text">of eligible pixels · {prediction.raw_label || 'model positives'}</span>
        </div>
        <div className="compare-bars">
          <div className="compare-bars__row">
            <span>Model prediction</span>
            <Bar value={prediction.flood_coverage_pct} color={legend.flood} label="Model prediction" />
            <span className="num">{fmtPct(prediction.flood_coverage_pct)}</span>
          </div>
          {reference?.available && (
            <div className="compare-bars__row">
              <Tooltip content="Share of supervised (cloud-free, mapped) pixels that Copernicus EMS marks as flood. Reference data, not a model output.">
                <span className="dotted">EMS reference</span>
              </Tooltip>
              <Bar value={reference.flood_pct_of_supervised} color="var(--ref)" label="EMS reference" />
              <span className="num">{fmtPct(reference.flood_pct_of_supervised)}</span>
            </div>
          )}
        </div>
        <p className="small muted">
          {fmtInt(prediction.n_flood_pixels)} of {fmtInt(prediction.n_eligible_pixels)} eligible pixels ·{' '}
          {fmtPct(prediction.eligible_pct, 0)} of the scene is eligible
          {prediction.eligibility_source === 'nodata_only' && ' (no cloud mask available)'}
        </p>
      </section>

      {/* ── Reference comparison ── */}
      <section className="insp-block">
        <div className="insp-head">
          <h3>Agreement with EMS reference</h3>
          {reference?.available && <InfoTip label="reference comparison">{reference.metrics_note}</InfoTip>}
        </div>
        {reference?.available && m ? (
          <>
            <div className="metric-grid">
              {[
                ['flood_iou', 'IoU'],
                ['flood_dice', 'Dice'],
                ['flood_precision', 'Precision'],
                ['flood_recall', 'Recall'],
              ].map(([k, label]) => (
                <Tooltip key={k} content={METRIC_HELP[k]}>
                  <div className="metric" tabIndex={0}>
                    <span className="metric__label">{label}</span>
                    <span className="metric__value num">{fmtRatio(m[k])}</span>
                  </div>
                </Tooltip>
              ))}
            </div>
            <ConfusionBar m={m} legend={legend} />
            <p className="small muted">{reference.split_note}</p>
          </>
        ) : (
          <p className="empty-note">
            {reference?.note || 'No reference mask is available for this imagery, so accuracy cannot be measured.'}
          </p>
        )}
      </section>

      {/* ── Progressive disclosure ── */}
      <div className="insp-disclosures">
        <Disclosure
          title={
            <>
              {quality?.status === 'ok' && !live ? (
                <CircleCheck size={15} className="tone-good" aria-hidden="true" />
              ) : (
                <TriangleAlert size={15} className="tone-warn" aria-hidden="true" />
              )}
              Image quality
            </>
          }
          meta={`${warnings.length} ${warnings.length === 1 ? 'note' : 'notes'}`}
          defaultOpen={live || quality?.status !== 'ok'}
        >
          <p className="small">
            Status: <strong>{quality?.status === 'ok' ? 'OK' : quality?.status?.replaceAll('_', ' ') || 'unknown'}</strong>
          </p>
          {warnings.length ? (
            <ul className="bullets">
              {warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          ) : (
            <p className="small muted">No quality warnings for this scene.</p>
          )}
        </Disclosure>

        <Disclosure title="Processing details" meta={data.method === 'tiled' ? 'Tiled' : 'Resize'}>
          <dl className="kv">
            <dt>Model</dt>
            <dd>
              {model?.architecture} · {model?.encoder} · epoch {model?.epoch}
            </dd>
            <dt>Method</dt>
            <dd>{data.method_note}</dd>
            {prediction.n_tiles != null && (
              <>
                <dt>Tiles</dt>
                <dd>{fmtInt(prediction.n_tiles)} × 256 px</dd>
              </>
            )}
            <dt>Flood rule</dt>
            <dd>{prediction.threshold_note}</dd>
            <dt>Device</dt>
            <dd>{model?.device}</dd>
            <dt>Result</dt>
            <dd>
              <SourceNote data={data} />
            </dd>
          </dl>
        </Disclosure>

        <Disclosure title="Model limitations" meta={`${data.limitations?.length || 0}`}>
          <ul className="bullets">
            {(data.limitations || []).map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </Disclosure>
      </div>

      <p className="disclaimer">{data.disclaimer}</p>
    </div>
  )
}
