import {
  Archive,
  ArrowRight,
  CalendarClock,
  CircleAlert,
  CircleCheck,
  CloudOff,
  Crosshair,
  Droplets,
  ImageOff,
  MapPin,
  RefreshCw,
  Satellite,
  ServerCrash,
  TriangleAlert,
} from 'lucide-react'
import { Disclosure, InfoTip } from './ui'
import { coverage, fmtKm2, freshness, MODEL_TARGET_LABEL, outcomeUi, showsShare } from '../lib/assessment'
import { fmtCoord, fmtInt, fmtPct, fmtRatio } from '../lib/format'
import { placeName } from '../lib/samples'

const TONE_ICON = {
  flood: Droplets,
  clear: CircleCheck,
  neutral: CircleAlert,
  muted: CloudOff,
  error: ServerCrash,
}

export function OutcomeIcon({ level, size = 20 }) {
  const ui = outcomeUi(level)
  const Icon = level === 'imagery_unavailable' ? ImageOff : TONE_ICON[ui.tone] || CircleAlert
  return <Icon size={size} aria-hidden="true" />
}

function timeText(ms) {
  if (!ms) return null
  return new Date(ms).toLocaleString('en-GB', { hour: '2-digit', minute: '2-digit', day: 'numeric', month: 'short' })
}

/** Error or "no result" outcome in the same visual slot as a real outcome. */
export function OutcomeNotice({ outcome, onRetry, children }) {
  const ui = outcomeUi(outcome.level)
  return (
    <section className={`outcome outcome--${ui.tone}`} role="alert">
      <span className="outcome__icon">
        <OutcomeIcon level={outcome.level} />
      </span>
      <div className="outcome__body">
        <p className="outcome__eyebrow">{ui.short}</p>
        <h2 className="outcome__title">{outcome.title}</h2>
        <p className="outcome__summary">{outcome.message}</p>
        {children}
        {outcome.retry && onRetry && (
          <button type="button" className="btn btn--sm outcome__retry" onClick={onRetry}>
            <RefreshCw size={14} aria-hidden="true" /> Try again
          </button>
        )}
      </div>
    </section>
  )
}

function CompositionBar({ cov }) {
  const total = cov.total || (cov.land ?? 0) + (cov.knownWater ?? 0) + (cov.unusable ?? 0)
  if (!total) return null
  const flood = cov.floodLike ?? 0
  const bc = cov.byClass
  const km2 = (n) => (typeof n === 'number' ? (n * cov.pxM2) / 1e6 : null)
  const waterParts = bc
    ? [
        { k: 'water', label: 'Permanent water / sea (set aside)', n: (bc.permanent?.n_eligible ?? 0) + (bc.sea?.n_eligible ?? 0) },
        { k: 'mapped', label: 'Mapped lakes, ponds, rivers, canals (set aside)', n: bc.mapped?.n_eligible ?? 0 },
        { k: 'seasonal', label: 'Seasonally wet land (set aside)', n: bc.seasonal?.n_eligible ?? 0 },
      ].filter((p) => p.n > 0 || p.k === 'water')
    : [{ k: 'water', label: 'Known water / sea (set aside)', n: cov.knownWater ?? 0 }]
  const parts = [
    { k: 'flood', label: 'Flood-like water on normally dry land', n: flood, area: cov.areaKm2.floodLike },
    { k: 'land', label: 'Other normally dry land', n: Math.max(0, (cov.land ?? 0) - flood), area: cov.areaKm2.land != null ? cov.areaKm2.land - (cov.areaKm2.floodLike ?? 0) : null },
    ...waterParts.map((p) => ({ ...p, area: km2(p.n) })),
    { k: 'unusable', label: 'Cloud, shadow or no data', n: cov.unusable ?? 0, area: cov.areaKm2.unusable },
  ]
  return (
    <div className="compo">
      <div className="compo__bar" role="img" aria-label={parts.map((p) => `${p.label} ${fmtPct((100 * p.n) / total)}`).join(', ')}>
        {parts.map((p) => (p.n > 0 ? <span key={p.k} className={`compo__seg compo__seg--${p.k}`} style={{ flexGrow: p.n }} /> : null))}
      </div>
      <ul className="compo__legend">
        {parts.map((p) => (
          <li key={p.k}>
            <span className={`compo__sw compo__seg--${p.k}`} aria-hidden="true" />
            <span className="compo__label">{p.label}</span>
            <span className="num">{fmtPct((100 * p.n) / total)}</span>
            <span className="num muted">{fmtKm2(p.area)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Fact({ label, children, warn }) {
  return (
    <div className={`fact ${warn ? 'fact--warn' : ''}`}>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  )
}

function ChangeSection({ change }) {
  const ok = change.status === 'available'
  return (
    <section className="rsec" aria-labelledby="chg-title">
      <div className="rsec__head">
        <h3 id="chg-title">Compared with a year earlier</h3>
        <InfoTip label="year-earlier comparison">
          The same 5 × 5 km area, model and bands, from the same season about one year earlier. Only pixels usable in both images and outside known
          water are compared. Water that was already there then is likely ordinary water.
        </InfoTip>
      </div>
      {ok ? (
        <>
          <dl className="facts">
            <Fact label="Earlier image">{fmtDateShort(change.baseline_date)}</Fact>
            <Fact label="Compared area">{fmtPct(change.compared_pct_of_usable, 0)} of usable pixels</Fact>
            <Fact label="Water on dry land then">{fmtPct(change.water_then_pct_of_dry_land, 1)}</Fact>
            <Fact label="Water on dry land now">{fmtPct(change.water_now_pct_of_dry_land, 1)}</Fact>
            <Fact label="New since then">{fmtPct(change.new_water_pct_of_dry_land, 1)} of dry land</Fact>
            <Fact label="Already there then">{fmtPct(change.persistent_share_of_candidate_pct, 0)} of today’s flood-like water</Fact>
          </dl>
        </>
      ) : (
        <p className="rsec__note">{change.reason || 'No earlier image could be compared.'}</p>
      )}
    </section>
  )
}

function fmtDateShort(iso) {
  if (!iso) return '—'
  const d = new Date(`${iso.slice(0, 10)}T00:00:00Z`)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
}

const CLOUD_MASK = {
  sen2cor_scl: 'Sentinel-2 scene classification (Sen2Cor)',
  unavailable: 'Not available',
}

/**
 * Assessment result in reading order: location & time, outcome, coverage (with its denominator),
 * image date & quality, map, environmental context, method.
 */
export default function ResultPanel({ data, stale, placeLabel, onShowOnMap, onRefresh, context, onNavigate }) {
  const a = data.assessment || {}
  const ui = outcomeUi(a.level)
  const live = data.mode === 'live'
  const cov = coverage(data)
  const fresh = freshness(data.freshness)
  const im = data.imagery || {}
  const wc = data.water_context || {}
  const p = data.prediction || {}
  const basis = a.basis || {}
  const model = data.model || {}
  const center = im.center
  const where = live ? placeLabel || 'Selected location' : placeName(data.sample)
  const share = showsShare(a.level) ? cov.share : null
  const warnings = (data.quality?.warnings || []).filter((w) => !w.startsWith('Experimental: live imagery'))
  const ref = data.reference
  const m = ref?.metrics

  return (
    <article className={`result ${stale ? 'is-stale' : ''}`} aria-busy={stale || undefined} aria-labelledby="result-title">
      {/* 1 · Location and time */}
      <header className="result__head">
        <p className="result__kind">
          {live ? (
            <>
              <Satellite size={13} aria-hidden="true" /> Latest Sentinel-2 image
            </>
          ) : (
            <>
              <Archive size={13} aria-hidden="true" /> Archived demo scene · {data.sample?.event_id}
            </>
          )}
        </p>
        <h2 className="result__place" id="result-title">
          <MapPin size={16} aria-hidden="true" />
          <span>{where}</span>
        </h2>
        <p className="result__sub">
          {center && <span className="num">{fmtCoord(center.lat, center.lng, 3)}</span>}
          {data._receivedAt && <span>Analysed {timeText(data._receivedAt)}</span>}
        </p>
      </header>

      {/* 2 · Outcome */}
      <section className={`outcome outcome--${ui.tone}`} aria-live="polite">
        <span className="outcome__icon">
          <OutcomeIcon level={a.level} />
        </span>
        <div className="outcome__body">
          <p className="outcome__eyebrow">Outcome</p>
          <h3 className="outcome__title">{a.headline || ui.short}</h3>
          {a.summary && <p className="outcome__summary">{a.summary}</p>}
          {a.reasons?.length > 0 && (
            <ul className="outcome__reasons">
              {a.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}
          {a.suggestion && <p className="outcome__hint">{a.suggestion}</p>}
        </div>
      </section>

      {/* 3 · Coverage, with its denominator */}
      <section className="rsec" aria-labelledby="cov-title">
        <div className="rsec__head">
          <h3 id="cov-title">Coverage</h3>
          <InfoTip label="coverage">
            Share of valid land pixels (cloud-free, with image data, outside known permanent water and sea) that the
            model marked as water. A pixel ratio from one image — not a probability, confidence or risk.
          </InfoTip>
        </div>
        {share != null ? (
          <div className="kpi">
            <span className="kpi__value num">{fmtPct(share, share < 10 ? 1 : 0)}</span>
            <span className="kpi__text">
              of the valid land analysed is flood-like water
              <span className="kpi__denom num">
                ≈ {fmtKm2(cov.areaKm2.floodLike)} of {fmtKm2(cov.areaKm2.land)} valid land ({fmtInt(cov.floodLike)} of {fmtInt(cov.land)} pixels)
              </span>
            </span>
          </div>
        ) : (
          <p className="rsec__note">
            No flood-like share is reported for this outcome
            {cov.usablePct != null && <> — only {fmtPct(cov.usablePct, 0)} of the image was usable</>}.
          </p>
        )}
        <CompositionBar cov={cov} />
        {!cov.waterReference && (
          <p className="rsec__warn">
            <TriangleAlert size={14} aria-hidden="true" /> Permanent-water reference unavailable: lakes, rivers or sea may be counted as flood-like.
          </p>
        )}
      </section>

      {/* 3b · Change since the same season a year earlier (live only) */}
      {live && data.change && <ChangeSection change={data.change} />}

      {/* 4 · Image date and quality */}
      <section className="rsec" aria-labelledby="img-title">
        <div className="rsec__head">
          <h3 id="img-title">Image</h3>
        </div>
        <dl className="facts">
          <Fact label="Acquired" warn={fresh.stale}>
            <CalendarClock size={13} aria-hidden="true" /> {fresh.date}
            {fresh.age && <span className="muted"> · {fresh.age}</span>}
          </Fact>
          <Fact label="Usable area">{fmtPct(p.eligible_pct, 0)} of the image</Fact>
          <Fact label="Cloud mask">{p.eligibility_source === 'nodata_only' ? 'None available' : CLOUD_MASK[im.cloud_mask_source] || (live ? 'Validity mask' : 'Dataset cloud/validity mask')}</Fact>
          {im.tile_cloud_cover_pct != null && <Fact label="Tile cloud cover">{fmtPct(im.tile_cloud_cover_pct, 0)}</Fact>}
          {p.haze_index != null && (
            <Fact label="Haze" warn={p.hazy}>
              {p.hazy ? 'Hazier than any training image' : 'Within the training range'}
              <span className="muted num"> · index {p.haze_index.toFixed(3)}</span>
            </Fact>
          )}
          <Fact label="Source">{live ? `Copernicus Sentinel-2 ${im.processing_level || 'L1C'} · ${im.resolution_m || 10} m` : 'WorldFloods v2 archive (Sentinel-2 L1C)'}</Fact>
          <Fact label="Grid">
            {fmtInt(im.width)} × {fmtInt(im.height)} px
          </Fact>
        </dl>
        {fresh.stale && <p className="rsec__warn">Conditions may have changed since this image was taken.</p>}
        {!live && <p className="rsec__note">{im.location_note}</p>}
        {warnings.length > 0 && (
          <Disclosure title="Image notes" meta={warnings.length}>
            <ul className="bullets">
              {warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          </Disclosure>
        )}
      </section>

      {/* 5 · Map */}
      <section className="rsec rsec--row" aria-labelledby="map-title">
        <div>
          <h3 id="map-title">Map</h3>
          <p className="rsec__note">Flood-like water outside known water bodies is drawn in blue; known water that was set aside in grey.</p>
        </div>
        {onShowOnMap && (
          <button type="button" className="btn btn--sm" onClick={onShowOnMap}>
            <Crosshair size={14} aria-hidden="true" /> Fit map
          </button>
        )}
      </section>

      {/* Demo scenes have an EMS reference: show agreement */}
      {ref?.available && m && (
        <section className="rsec" aria-labelledby="ref-title">
          <div className="rsec__head">
            <h3 id="ref-title">Agreement with the EMS flood map</h3>
            <InfoTip label="reference agreement">{ref.metrics_note}</InfoTip>
          </div>
          <div className="mini-metrics">
            {[
              ['flood_iou', 'IoU'],
              ['flood_precision', 'Precision'],
              ['flood_recall', 'Recall'],
              ['flood_dice', 'Dice'],
            ].map(([k, l]) => (
              <div key={k}>
                <span>{l}</span>
                <strong className="num">{fmtRatio(m[k], 2)}</strong>
              </div>
            ))}
          </div>
          <p className="rsec__note">
            Mapped flood covers {fmtPct(ref.flood_pct_of_supervised, 1)} of the supervised area. {ref.split_note}
          </p>
        </section>
      )}

      {/* 6 · Environmental context */}
      {context}

      {/* 7 · Method */}
      <section className="rsec rsec--method">
        <Disclosure title="How this result was produced" meta={model.version}>
          <dl className="kv">
            <dt>Model</dt>
            <dd>
              {model.architecture} ({model.encoder}) · {model.version}
            </dd>
            <dt>Detects</dt>
            <dd>{MODEL_TARGET_LABEL[p.target] || p.raw_label || '—'}</dd>
            <dt>Bands</dt>
            <dd>{(model.bands || []).join(', ') || '—'}</dd>
            <dt>Rule</dt>
            <dd>
              {basis.none_below_pct != null
                ? `“None” below ${basis.none_below_pct}%; “inconclusive” up to ${basis.inconclusive_ceiling_pct}% (largest share marked on validation scenes without mapped flooding).`
                : 'No validated outcome thresholds for this model.'}
            </dd>
            <dt>Known water</dt>
            <dd>{wc.source || 'Not available for this image'}</dd>
          </dl>
          <ul className="bullets small">
            {(data.limitations || []).map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
          {onNavigate && (
            <p className="rsec__links">
              <button type="button" className="link-btn" onClick={() => onNavigate('methodology')}>
                Methodology <ArrowRight size={13} aria-hidden="true" />
              </button>
              <button type="button" className="link-btn" onClick={() => onNavigate('research')}>
                Research diagnostics <ArrowRight size={13} aria-hidden="true" />
              </button>
            </p>
          )}
        </Disclosure>
      </section>

      <footer className="result__foot">
        <p className="result__disclaimer">{data.disclaimer}</p>
        {onRefresh && (
          <button type="button" className="btn btn--sm btn--ghost" onClick={onRefresh} disabled={stale}>
            <RefreshCw size={14} aria-hidden="true" /> {live ? 'Run again' : 'Reload'}
          </button>
        )}
      </footer>
    </article>
  )
}
