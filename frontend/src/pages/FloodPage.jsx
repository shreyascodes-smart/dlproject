import { useState } from 'react'
import { MapPin, Satellite, X } from 'lucide-react'
import FloodMap from '../components/FloodMap'
import LocationSearch from '../components/LocationSearch'
import ResultPanel, { OutcomeNotice } from '../components/ResultPanel'
import ContextSnapshot from '../components/ContextSnapshot'
import { Elapsed, ResultSkeleton } from '../components/feedback'
import { Spinner } from '../components/ui'
import { describeError, freshness } from '../lib/assessment'
import { areaBounds } from '../lib/map'
import { fmtCoord } from '../lib/format'

function HowItWorks({ hasLocation }) {
  return (
    <section className="howto" aria-labelledby="howto-title">
      <h2 id="howto-title">{hasLocation ? 'Ready to analyse' : 'How it works'}</h2>
      <ol>
        <li className={hasLocation ? 'is-done' : ''}>
          <strong>Choose a place</strong>
          <span>Search, use your location, or click the map.</span>
        </li>
        <li>
          <strong>Analyse the latest image</strong>
          <span>The most recent clear Sentinel-2 image (last 45 days) is checked for water outside known lakes, rivers and sea.</span>
        </li>
        <li>
          <strong>Read the outcome</strong>
          <span>Each result shows its coverage, image date and quality, and how it was produced.</span>
        </li>
      </ol>
    </section>
  )
}

const sameLocation = (a, b) => a && b && Math.abs(a.lat - b.lat) < 1e-6 && Math.abs(a.lng - b.lng) < 1e-6

/** Live flood analysis: persistent controls, scrollable results, map fitted to the analysed raster. */
export default function FloodPage({
  location,
  onLocation,
  place,
  live,
  status,
  theme,
  basemap,
  onBasemap,
  navigate,
  searchRef,
  onLocateMe,
  locating,
}) {
  const [fitNonce, setFitNonce] = useState(0)
  const loading = live.status === 'loading'
  const req = live.request
  const forHere = req?.mode === 'live' && location && sameLocation({ lat: req.latitude, lng: req.longitude }, location)
  const data = live.data?.mode === 'live' && forHere ? live.data : null
  const error = live.status === 'error' && forHere ? describeError(live.error, { live: true }) : null
  const modelReady = status.state === 'ready'
  const liveConfigured = status.data?.satellite === 'configured'

  const blocker = !modelReady
    ? status.state === 'loading'
      ? 'Connecting to the analysis service…'
      : 'The analysis service is unavailable right now. It is re-checked automatically.'
    : !liveConfigured
      ? 'Live satellite imagery is not configured on this server.'
      : !location
        ? 'Choose a location to analyse.'
        : null

  const runLive = () => location && live.run({ mode: 'live', latitude: location.lat, longitude: location.lng, method: 'tiled' })
  const placeLabel = place.name || (place.status === 'loading' ? 'Finding place name…' : location ? fmtCoord(location.lat, location.lng, 3) : null)
  const area = location && !data?.map ? areaBounds(location.lat, location.lng) : null
  const fresh = data ? freshness(data.freshness) : null
  const chip = data ? (
    <>
      <Satellite size={13} aria-hidden="true" /> Sentinel-2 · {fresh.date}
      {fresh.age && <span className="muted"> · {fresh.age}</span>}
    </>
  ) : null

  return (
    <div className="workspace">
      <div className="workspace__panel">
        <section className="panel__controls" aria-labelledby="flood-title">
          <h1 className="panel__title" id="flood-title">
            Flood analysis
          </h1>
          <p className="panel__lead">Check the latest Sentinel-2 image of a 5 × 5 km area for flood-like water.</p>
          <LocationSearch onSelect={onLocation} onLocateMe={onLocateMe} locating={locating} inputRef={searchRef} id="flood-search" near={location} />
          {location && (
            <div className="place-chip">
              <MapPin size={15} aria-hidden="true" />
              <div>
                <strong>{placeLabel}</strong>
                <span className="num">{fmtCoord(location.lat, location.lng, 4)} · 5 × 5 km area</span>
              </div>
              <button type="button" className="icon-btn icon-btn--sm" onClick={() => onLocation(null)} aria-label="Clear selected location">
                <X size={14} aria-hidden="true" />
              </button>
            </div>
          )}
          <button
            type="button"
            className="btn btn--primary btn--lg btn--block"
            onClick={runLive}
            disabled={!!blocker || loading}
            aria-describedby={blocker ? 'live-blocker' : undefined}
          >
            {loading ? <Spinner size={16} /> : <Satellite size={16} aria-hidden="true" />}
            {loading ? 'Analysing…' : data ? 'Analyse again' : 'Analyse this area'}
          </button>
          {blocker && (
            <p id="live-blocker" className="panel__blocker">
              {blocker}
              {modelReady && !liveConfigured && (
                <>
                  {' '}
                  <a className="link-btn" href="#/demo">
                    Explore archived demo scenes
                  </a>
                </>
              )}
            </p>
          )}
        </section>

        <section className="panel__scroll" aria-label="Results" id="results" tabIndex={-1}>
          {error && <OutcomeNotice outcome={error} onRetry={runLive} />}
          {loading && !data && <ResultSkeleton />}
          {data && (
            <ResultPanel
              data={data}
              stale={loading}
              placeLabel={place.name}
              onShowOnMap={() => setFitNonce((n) => n + 1)}
              onRefresh={runLive}
              onNavigate={navigate}
              context={<ContextSnapshot location={location} onOpen={() => navigate('weather')} />}
            />
          )}
          {!data && !error && !loading && (
            <>
              <HowItWorks hasLocation={!!location} />
              {location && <ContextSnapshot location={location} onOpen={() => navigate('weather')} />}
            </>
          )}
        </section>
      </div>

      <section className="workspace__map" aria-label="Map">
        <FloodMap
          marker={location}
          area={area}
          layers={data?.map || null}
          onPick={onLocation}
          busy={loading}
          theme={theme}
          basemap={basemap}
          onBasemap={onBasemap}
          fitNonce={fitNonce}
          chip={chip}
          label={
            location
              ? `Map around ${placeLabel}. Click the map to move the location.`
              : 'Map. Click to choose a location.'
          }
        >
          {loading && (
            <div className="map-float map-float--top">
              <Elapsed startedAt={live.startedAt} label="Finding and analysing the latest image" hint="satellite downloads can take up to a minute" />
            </div>
          )}
          {!location && !loading && (
            <div className="map-float map-float--hint" aria-hidden="true">
              Click the map to choose a location
            </div>
          )}
        </FloodMap>
      </section>
    </div>
  )
}
