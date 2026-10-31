import { ArrowRight, CloudRain, RefreshCw, Wind } from 'lucide-react'
import WeatherIcon from './WeatherIcon'
import { useAirQuality, useWeather } from '../hooks/useEnvironment'
import { aqiBand, compass, describeWeather, EU_AQI, rainSummary, US_AQI } from '../lib/weather'

const fmt = (v, d = 0) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—')

function Failed({ what, onRetry, message }) {
  return (
    <div className="ctx__state" role="status">
      <span>{message || `${what} unavailable.`}</span>
      <button type="button" className="btn btn--xs" onClick={onRetry}>
        <RefreshCw size={12} aria-hidden="true" /> Retry
      </button>
    </div>
  )
}

function Skeleton() {
  return (
    <div className="ctx__skel" aria-hidden="true">
      <span />
      <span />
    </div>
  )
}

/**
 * Weather and air-quality snapshot for the selected location. Context only: it is not used by the
 * flood model. Weather and air quality load (and fail) independently of each other.
 */
export default function ContextSnapshot({ location, onOpen }) {
  const wx = useWeather(location, !!location)
  const aq = useAirQuality(location, !!location)
  if (!location) return null
  const c = wx.data?.current
  const u = wx.data?.current_units || {}
  const w = c ? describeWeather(c.weather_code, c.is_day) : null
  const rain = rainSummary(wx.data)
  const a = aq.data?.current
  const us = aqiBand(US_AQI, a?.us_aqi)
  const eu = aqiBand(EU_AQI, a?.european_aqi)

  return (
    <section className="rsec ctx" aria-labelledby="ctx-title">
      <div className="rsec__head">
        <h3 id="ctx-title">Environmental context</h3>
        <span className="rsec__meta">Context only · not used by the flood model</span>
      </div>
      <div className="ctx__grid">
        <div className="ctx__card" aria-busy={wx.status === 'loading' || undefined}>
          <p className="ctx__label">Weather now</p>
          {wx.status === 'loading' && <Skeleton />}
          {(wx.status === 'error' || (wx.status === 'stale-error' && !c)) && <Failed what="Weather" message={wx.error} onRetry={wx.refresh} />}
          {c && (
            <>
              <div className="ctx__main">
                <WeatherIcon name={w.icon} size={26} />
                <span className="ctx__big num">
                  {fmt(c.temperature_2m)}
                  {u.temperature_2m || '°C'}
                </span>
                <span className="ctx__sub">{w.text}</span>
              </div>
              <ul className="ctx__facts">
                <li>
                  <CloudRain size={13} aria-hidden="true" /> Rain last 24 h <strong className="num">{fmt(rain.past24, 1)} mm</strong>
                </li>
                <li>
                  <CloudRain size={13} aria-hidden="true" /> Next 24 h <strong className="num">{fmt(rain.next24, 1)} mm</strong>
                  {rain.nextProbMax != null && <span className="muted">(chance up to {rain.nextProbMax}%)</span>}
                </li>
                <li>
                  <Wind size={13} aria-hidden="true" /> Wind <strong className="num">{fmt(c.wind_speed_10m)} {u.wind_speed_10m || 'km/h'}</strong>{' '}
                  {compass(c.wind_direction_10m)}
                </li>
              </ul>
            </>
          )}
        </div>
        <div className="ctx__card" aria-busy={aq.status === 'loading' || undefined}>
          <p className="ctx__label">Air quality now</p>
          {aq.status === 'loading' && <Skeleton />}
          {(aq.status === 'error' || (aq.status === 'stale-error' && !a)) && <Failed what="Air quality" message={aq.error} onRetry={aq.refresh} />}
          {a && (
            <div className="ctx__aqi">
              <div>
                <span className="ctx__scale">US AQI</span>
                <span className="ctx__big num">{fmt(a.us_aqi)}</span>
                <span className="ctx__band">
                  <i style={{ background: us?.color }} aria-hidden="true" /> {us?.label || 'Unavailable'}
                </span>
              </div>
              <div>
                <span className="ctx__scale">European AQI</span>
                <span className="ctx__big num">{fmt(a.european_aqi)}</span>
                <span className="ctx__band">
                  <i style={{ background: eu?.color }} aria-hidden="true" /> {eu?.label || 'Unavailable'}
                </span>
              </div>
              <p className="ctx__pm">
                PM2.5 <strong className="num">{fmt(a.pm2_5, 1)}</strong> {aq.data?.current_units?.pm2_5 || 'μg/m³'}
              </p>
            </div>
          )}
        </div>
      </div>
      {onOpen && (
        <button type="button" className="link-btn ctx__open" onClick={onOpen}>
          Full weather &amp; air-quality dashboard <ArrowRight size={14} aria-hidden="true" />
        </button>
      )}
    </section>
  )
}
