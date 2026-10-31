import { useEffect, useMemo, useState } from 'react'
import {
  CloudRain,
  Droplets,
  Eye,
  Gauge as GaugeIcon,
  MapPin,
  Navigation2,
  RefreshCw,
  Sun,
  Sunrise,
  Sunset,
  Thermometer,
  TriangleAlert,
  Umbrella,
  Wind,
} from 'lucide-react'
import LocationSearch from '../components/LocationSearch'
import WeatherIcon from '../components/WeatherIcon'
import { DataTable, Gauge, RangeBar, TimeChart } from '../components/charts'
import { Disclosure, Segmented, Spinner } from '../components/ui'
import { useAirQuality, useWeather } from '../hooks/useEnvironment'
import { fmtCoord } from '../lib/format'
import {
  aqiBand,
  compass,
  currentHourIndex,
  dayLength,
  describeWeather,
  EU_AQI,
  EU_AQI_MAX,
  localClock,
  localDay,
  POLLEN,
  POLLUTANTS,
  rainSummary,
  sumRange,
  todayIndex,
  US_AQI,
  US_AQI_MAX,
  uvCategory,
} from '../lib/weather'

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
const fmt = (v, d = 0) => (isNum(v) ? v.toFixed(d) : '—')

function useNow(ms = 30000) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(id)
  }, [ms])
  return now
}

function ago(ms, now) {
  if (!ms) return null
  const m = Math.max(0, Math.round((now - ms) / 60000))
  return m < 1 ? 'just now' : m === 1 ? '1 min ago' : m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`
}

/** "2026-10-28T03:00" -> tick "03:00" (or the day name at midnight), full "Wed 28 Oct, 03:00" */
function hourFormatter(t, _i, kind) {
  if (!t) return '—'
  const day = localDay(t)
  if (kind === 'full') return `${localDay(t, { long: true })}, ${localClock(t)}`
  return t.slice(11, 13) === '00' ? day : localClock(t)
}

function Card({ id, title, meta, className = '', children, state, onRetry, actions }) {
  return (
    <section className={`card ${className}`} aria-labelledby={id} aria-busy={state?.status === 'loading' || undefined}>
      <header className="card__head">
        <h2 id={id}>{title}</h2>
        {meta && <span className="card__meta">{meta}</span>}
        {actions}
      </header>
      {state?.status === 'loading' && !state.data && (
        <div className="card__loading">
          <Spinner size={16} /> Loading…
        </div>
      )}
      {(state?.status === 'error' || (state?.status === 'stale-error' && !state.data)) && (
        <div className="card__error" role="alert">
          <TriangleAlert size={16} aria-hidden="true" />
          <span>{state.error}</span>
          <button type="button" className="btn btn--xs" onClick={onRetry}>
            <RefreshCw size={12} aria-hidden="true" /> Retry
          </button>
        </div>
      )}
      {state?.status === 'stale-error' && state.data && (
        <p className="card__stale" role="status">
          <TriangleAlert size={13} aria-hidden="true" /> Refresh failed; showing earlier data.
        </p>
      )}
      {(!state || state.data) && children}
    </section>
  )
}

function Tile({ icon, label, value, sub }) {
  return (
    <div className="tile">
      <span className="tile__label">
        {icon}
        {label}
      </span>
      <span className="tile__value num">{value}</span>
      {sub && <span className="tile__sub">{sub}</span>}
    </div>
  )
}

function CurrentCard({ wx }) {
  const d = wx.data
  const c = d?.current
  const u = d?.current_units || {}
  const daily = d?.daily
  const ti = todayIndex(daily?.time, c?.time)
  const w = c ? describeWeather(c.weather_code, c.is_day) : null
  const rain = rainSummary(d)
  const past2 = daily ? sumRange(daily.precipitation_sum, 0, ti) : null
  const uv = uvCategory(c?.uv_index)
  return (
    <Card id="wx-now" title="Current conditions" meta={c ? `Local ${localClock(c.time)} ${d.timezone_abbreviation || ''}` : null} className="card--now" state={wx} onRetry={wx.refresh}>
      {c && (
        <>
          <div className="now">
            <WeatherIcon name={w.icon} size={56} className="now__icon" />
            <div className="now__temp">
              <span className="now__value num">
                {fmt(c.temperature_2m)}
                <small>{u.temperature_2m || '°C'}</small>
              </span>
              <span className="now__desc">{w.text}</span>
              <span className="now__sub num">
                Feels like {fmt(c.apparent_temperature)}
                {u.apparent_temperature || '°C'}
                {daily && (
                  <>
                    {' '}
                    · H {fmt(daily.temperature_2m_max?.[ti])}° L {fmt(daily.temperature_2m_min?.[ti])}°
                  </>
                )}
              </span>
            </div>
          </div>
          <div className="tiles">
            <Tile icon={<Droplets size={14} aria-hidden="true" />} label="Humidity" value={`${fmt(c.relative_humidity_2m)}%`} />
            <Tile
              icon={<Wind size={14} aria-hidden="true" />}
              label="Wind"
              value={`${fmt(c.wind_speed_10m)} ${u.wind_speed_10m || 'km/h'}`}
              sub={
                <span className="winddir">
                  <Navigation2 size={12} aria-hidden="true" style={{ transform: `rotate(${(c.wind_direction_10m ?? 0) + 180}deg)` }} />
                  from {compass(c.wind_direction_10m)} ({fmt(c.wind_direction_10m)}°)
                </span>
              }
            />
            <Tile icon={<Wind size={14} aria-hidden="true" />} label="Gusts" value={`${fmt(c.wind_gusts_10m)} ${u.wind_gusts_10m || 'km/h'}`} />
            <Tile icon={<GaugeIcon size={14} aria-hidden="true" />} label="Pressure" value={`${fmt(c.pressure_msl)} ${u.pressure_msl || 'hPa'}`} sub="at sea level" />
            <Tile
              icon={<Eye size={14} aria-hidden="true" />}
              label="Visibility"
              value={isNum(c.visibility) ? (c.visibility >= 10000 ? `${fmt(c.visibility / 1000)} km` : `${fmt(c.visibility / 1000, 1)} km`) : '—'}
            />
            <Tile icon={<Thermometer size={14} aria-hidden="true" />} label="Cloud cover" value={`${fmt(c.cloud_cover)}%`} />
            <Tile icon={<Sun size={14} aria-hidden="true" />} label="UV index" value={fmt(c.uv_index, 1)} sub={uv} />
            <Tile
              icon={<Umbrella size={14} aria-hidden="true" />}
              label="Precipitation"
              value={`${fmt(c.precipitation, 1)} ${u.precipitation || 'mm'}`}
              sub={isNum(c.precipitation_probability) ? `${c.precipitation_probability}% chance this hour` : null}
            />
            <Tile icon={<CloudRain size={14} aria-hidden="true" />} label="Rain, last 24 h" value={`${fmt(rain.past24, 1)} mm`} sub={past2 != null ? `${fmt(past2, 1)} mm in the 2 previous days` : null} />
            <Tile
              icon={<CloudRain size={14} aria-hidden="true" />}
              label="Rain, next 24 h"
              value={`${fmt(rain.next24, 1)} mm`}
              sub={rain.nextProbMax != null ? `up to ${rain.nextProbMax}% chance` : null}
            />
            <Tile icon={<Sunrise size={14} aria-hidden="true" />} label="Sunrise" value={localClock(daily?.sunrise?.[ti])} />
            <Tile icon={<Sunset size={14} aria-hidden="true" />} label="Sunset" value={localClock(daily?.sunset?.[ti])} sub={dayLength(daily?.sunrise?.[ti], daily?.sunset?.[ti])} />
          </div>
        </>
      )}
    </Card>
  )
}

function AqiScale({ name, value, scale, max }) {
  const band = aqiBand(scale, value)
  return (
    <div className="aqi">
      <p className="aqi__name">{name}</p>
      <Gauge value={value} max={max} bands={scale} label={`${name} ${isNum(value) ? Math.round(value) : 'unavailable'}${band ? `, ${band.label}` : ''}`} />
      <p className="aqi__value num">{isNum(value) ? Math.round(value) : '—'}</p>
      <p className="aqi__band">
        <i style={{ background: band?.color }} aria-hidden="true" /> {band?.label || 'Unavailable'}
      </p>
    </div>
  )
}

function AirNowCard({ aq }) {
  const a = aq.data?.current
  const us = aqiBand(US_AQI, a?.us_aqi)
  const eu = aqiBand(EU_AQI, a?.european_aqi)
  return (
    <Card id="aq-now" title="Air quality" meta={a ? `Local ${localClock(a.time)}` : null} className="card--aqi" state={aq} onRetry={aq.refresh}>
      {a && (
        <>
          <div className="aqi-pair">
            <AqiScale name="US AQI" value={a.us_aqi} scale={US_AQI} max={US_AQI_MAX} />
            <AqiScale name="European AQI" value={a.european_aqi} scale={EU_AQI} max={EU_AQI_MAX} />
          </div>
          <div className="guidance">
            <h3>Health guidance</h3>
            {us && (
              <p>
                <strong>US AQI · {us.label}:</strong> {us.advice}
              </p>
            )}
            {eu && (
              <p>
                <strong>European AQI · {eu.label}:</strong> {eu.advice}
              </p>
            )}
            <p className="muted small">The two indices use different pollutants, breakpoints and scales, so their numbers are not comparable. General guidance only, not medical advice.</p>
          </div>
        </>
      )}
    </Card>
  )
}

const HOURLY_VIEWS = [
  { value: 'temp', label: 'Temperature' },
  { value: 'rain', label: 'Rain' },
  { value: 'wind', label: 'Wind' },
]

function HourlyCard({ wx }) {
  const [view, setView] = useState('temp')
  const h = wx.data?.hourly
  const units = wx.data?.hourly_units
  const now = h ? currentHourIndex(h.time, wx.data.current?.time) : null
  const series = useMemo(() => {
    if (!h) return []
    const hu = units || {}
    if (view === 'rain')
      return [
        { key: 'p', label: `Precipitation (${hu.precipitation || 'mm'})`, values: h.precipitation, color: 'var(--c-rain)', type: 'bar', format: (v) => `${fmt(v, 1)} mm` },
        { key: 'pp', label: 'Chance of precipitation', values: h.precipitation_probability, color: 'var(--c-prob)', axis: 'right', format: (v) => `${fmt(v)}%`, tick: (v) => `${v}%` },
      ]
    if (view === 'wind')
      return [{ key: 'w', label: `Wind (${hu.wind_speed_10m || 'km/h'})`, values: h.wind_speed_10m, color: 'var(--c-wind)', format: (v) => `${fmt(v)} km/h` }]
    return [{ key: 't', label: `Temperature (${hu.temperature_2m || '°C'})`, values: h.temperature_2m, color: 'var(--c-temp)', format: (v) => `${fmt(v, 1)}°`, tick: (v) => `${v}°` }]
  }, [h, units, view])
  const rows = h
    ? h.time.map((t, i) => ({
        id: t,
        time: hourFormatter(t, i, 'full'),
        temp: `${fmt(h.temperature_2m?.[i], 1)}°`,
        prob: `${fmt(h.precipitation_probability?.[i])}%`,
        rain: `${fmt(h.precipitation?.[i], 1)} mm`,
        wind: `${fmt(h.wind_speed_10m?.[i])} km/h`,
        hum: `${fmt(h.relative_humidity_2m?.[i])}%`,
        sky: describeWeather(h.weather_code?.[i], 1).text,
      }))
    : []
  return (
    <Card
      id="wx-hourly"
      title="Hourly"
      meta="Past 24 h and next 48 h"
      className="card--wide"
      state={wx}
      onRetry={wx.refresh}
      actions={h ? <Segmented label="Hourly chart" size="sm" value={view} onChange={setView} options={HOURLY_VIEWS} /> : null}
    >
      {h && (
        <>
          <TimeChart
            times={h.time}
            series={series}
            nowIndex={now}
            height={210}
            label={`Hourly ${HOURLY_VIEWS.find((v) => v.value === view).label.toLowerCase()} chart`}
            formatTime={hourFormatter}
            leftDomain={view === 'rain' ? { min: 0 } : view === 'wind' ? { min: 0 } : undefined}
            rightDomain={{ min: 0, max: 100 }}
          />
          <Disclosure title="Hourly data table" lazy>
            <DataTable
              caption="Hourly weather data"
              highlight={now}
              columns={[
                { key: 'time', label: 'Time (local)' },
                { key: 'sky', label: 'Conditions' },
                { key: 'temp', label: 'Temp.' },
                { key: 'prob', label: 'Chance' },
                { key: 'rain', label: 'Precip.' },
                { key: 'wind', label: 'Wind' },
                { key: 'hum', label: 'Humidity' },
              ]}
              rows={rows}
            />
          </Disclosure>
        </>
      )}
    </Card>
  )
}

function DailyCard({ wx }) {
  const d = wx.data?.daily
  const du = wx.data?.daily_units || {}
  const ti = todayIndex(d?.time, wx.data?.current?.time)
  const idx = d ? d.time.map((_, i) => i).filter((i) => i >= ti) : []
  const lo = d ? Math.min(...idx.map((i) => d.temperature_2m_min[i]).filter(isNum)) : 0
  const hi = d ? Math.max(...idx.map((i) => d.temperature_2m_max[i]).filter(isNum)) : 1
  return (
    <Card id="wx-daily" title="7-day forecast" className="card--daily" state={wx} onRetry={wx.refresh}>
      {d && (
        <div className="daily-wrap" tabIndex={0} role="region" aria-label="7-day forecast table">
          <table className="daily">
            <thead>
              <tr>
                <th scope="col">Day</th>
                <th scope="col">Conditions</th>
                <th scope="col">
                  <span className="sr-only">Chance of </span>Rain
                </th>
                <th scope="col">Low – high</th>
                <th scope="col">Wind</th>
                <th scope="col">UV</th>
                <th scope="col">Sun</th>
              </tr>
            </thead>
            <tbody>
              {idx.map((i, k) => {
                const w = describeWeather(d.weather_code[i], 1)
                return (
                  <tr key={d.time[i]}>
                    <th scope="row">{k === 0 ? 'Today' : k === 1 ? 'Tomorrow' : localDay(d.time[i])}</th>
                    <td>
                      <span className="daily__sky">
                        <WeatherIcon name={w.icon} size={18} /> {w.text}
                      </span>
                    </td>
                    <td className="num">
                      {fmt(d.precipitation_probability_max?.[i])}%<span className="daily__sub">{fmt(d.precipitation_sum?.[i], 1)} {du.precipitation_sum || 'mm'}</span>
                    </td>
                    <td>
                      <span className="daily__range num">
                        <span>{fmt(d.temperature_2m_min[i])}°</span>
                        <RangeBar min={d.temperature_2m_min[i]} max={d.temperature_2m_max[i]} lo={lo} hi={hi} label={`${fmt(d.temperature_2m_min[i])} to ${fmt(d.temperature_2m_max[i])} degrees`} />
                        <span>{fmt(d.temperature_2m_max[i])}°</span>
                      </span>
                    </td>
                    <td className="num">
                      {fmt(d.wind_speed_10m_max?.[i])} {du.wind_speed_10m_max || 'km/h'} {compass(d.wind_direction_10m_dominant?.[i])}
                      <span className="daily__sub">gusts {fmt(d.wind_gusts_10m_max?.[i])}</span>
                    </td>
                    <td className="num">
                      {fmt(d.uv_index_max?.[i], 1)}
                      <span className="daily__sub">{uvCategory(d.uv_index_max?.[i])}</span>
                    </td>
                    <td className="num daily__sun">
                      <span>↑ {localClock(d.sunrise?.[i])}</span>
                      <span>↓ {localClock(d.sunset?.[i])}</span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

function PollutantsCard({ aq }) {
  const a = aq.data?.current
  const u = aq.data?.current_units || {}
  const pollen = POLLEN.filter(([k]) => isNum(a?.[k]))
  return (
    <Card id="aq-pollutants" title="Pollutants" meta="Surface concentrations" className="card--pollutants" state={aq} onRetry={aq.refresh}>
      {a && (
        <>
          <div className="pollutants">
            {POLLUTANTS.map(([k, abbr, name]) => (
              <div key={k} className="pollutant">
                <span className="pollutant__abbr">{abbr}</span>
                <span className="pollutant__value num">
                  {fmt(a[k], a[k] < 10 ? 1 : 0)} <small>{u[k] || 'μg/m³'}</small>
                </span>
                <span className="pollutant__name">{name}</span>
              </div>
            ))}
          </div>
          <h3 className="card__sub">Pollen</h3>
          {pollen.length ? (
            <div className="pollutants pollutants--pollen">
              {pollen.map(([k, name]) => (
                <div key={k} className="pollutant">
                  <span className="pollutant__abbr">{name}</span>
                  <span className="pollutant__value num">
                    {fmt(a[k], 1)} <small>{u[k] || 'grains/m³'}</small>
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted small">Pollen forecasts are only available for Europe.</p>
          )}
        </>
      )}
    </Card>
  )
}

function AqiTrendCard({ aq }) {
  const h = aq.data?.hourly
  const now = h ? currentHourIndex(h.time, aq.data.current?.time) : null
  const series = h
    ? [
        { key: 'us', label: 'US AQI', values: h.us_aqi, color: 'var(--c-us)', format: (v) => fmt(v) },
        { key: 'eu', label: 'European AQI', values: h.european_aqi, color: 'var(--c-eu)', format: (v) => fmt(v) },
      ]
    : []
  return (
    <Card id="aq-trend" title="Air-quality trend" meta="Past 24 h and next 3 days" className="card--wide" state={aq} onRetry={aq.refresh}>
      {h && (
        <>
          <TimeChart times={h.time} series={series} nowIndex={now} height={190} label="Hourly US and European AQI chart" formatTime={hourFormatter} leftDomain={{ min: 0 }} />
          <Disclosure title="Air-quality data table" lazy>
            <DataTable
              caption="Hourly air-quality data"
              highlight={now}
              columns={[
                { key: 'time', label: 'Time (local)' },
                { key: 'us', label: 'US AQI' },
                { key: 'eu', label: 'European AQI' },
                { key: 'pm', label: 'PM2.5 (μg/m³)' },
              ]}
              rows={h.time.map((t, i) => ({ id: t, time: hourFormatter(t, i, 'full'), us: fmt(h.us_aqi?.[i]), eu: fmt(h.european_aqi?.[i]), pm: fmt(h.pm2_5?.[i], 1) }))}
            />
          </Disclosure>
        </>
      )}
    </Card>
  )
}

/** Full weather and air-quality dashboard for the selected location (independent of the flood model). */
export default function WeatherPage({ location, onLocation, place, onLocateMe, locating, searchRef }) {
  const wx = useWeather(location, !!location)
  const aq = useAirQuality(location, !!location)
  const now = useNow()
  const updated = Math.min(...[wx.fetchedAt, aq.fetchedAt].filter(Boolean))
  const busy = wx.status === 'loading' || wx.status === 'refreshing' || aq.status === 'loading' || aq.status === 'refreshing'
  const refresh = () => {
    wx.refresh()
    aq.refresh()
  }
  return (
    <div className="page page--weather">
      <header className="page__head">
        <div className="page__title">
          <p className="eyebrow">Weather &amp; air quality</p>
          <h1>{location ? place.name || fmtCoord(location.lat, location.lng, 2) : 'Choose a location'}</h1>
          {location && (
            <p className="page__sub">
              <MapPin size={13} aria-hidden="true" /> <span className="num">{fmtCoord(location.lat, location.lng, 3)}</span>
              {wx.data?.timezone && <span> · {wx.data.timezone.replace('_', ' ')}</span>}
              {Number.isFinite(updated) && <span> · Updated {ago(updated, now)}</span>}
            </p>
          )}
        </div>
        <div className="page__tools">
          <LocationSearch onSelect={onLocation} onLocateMe={onLocateMe} locating={locating} inputRef={searchRef} hint={null} id="weather-search" near={location} />
          {location && (
            <button type="button" className="btn btn--sm" onClick={refresh} disabled={busy}>
              {busy ? <Spinner size={14} /> : <RefreshCw size={14} aria-hidden="true" />} Refresh
            </button>
          )}
        </div>
      </header>

      {!location ? (
        <div className="empty-page">
          <MapPin size={28} aria-hidden="true" />
          <h2>No location selected</h2>
          <p>Search for a place or use your location to see current weather, the hourly and 7-day forecast, and air quality.</p>
        </div>
      ) : (
        <div className="wx-grid">
          <CurrentCard wx={wx} />
          <AirNowCard aq={aq} />
          <HourlyCard wx={wx} />
          <DailyCard wx={wx} />
          <PollutantsCard aq={aq} />
          <AqiTrendCard aq={aq} />
          <p className="sources">
            Weather: <a href="https://open-meteo.com/" target="_blank" rel="noreferrer">Open-Meteo</a> (CC BY 4.0). Air quality: Open-Meteo Air Quality API using{' '}
            <a href="https://atmosphere.copernicus.eu/" target="_blank" rel="noreferrer">Copernicus CAMS</a> forecasts. Model data, not station measurements. Times are local to the location.
          </p>
        </div>
      )}
    </div>
  )
}
