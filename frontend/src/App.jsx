import { useState, useCallback, useEffect } from 'react'
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet'
import axios from 'axios'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import './index.css'

delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

const API = 'http://localhost:8000'
const WEATHER = 'https://api.open-meteo.com/v1/forecast'
const AQI_URL = 'https://air-quality-api.open-meteo.com/v1/air-quality'
const NOMINATIM = 'https://nominatim.openstreetmap.org/reverse'

function MapClickHandler({ onSelect }) {
  useMapEvents({
    click: (e) => onSelect({ lat: e.latlng.lat, lng: e.latlng.lng }),
  })
  return null
}

export default function App() {
  const [loc, setLoc] = useState(null)
  const [placeName, setPlaceName] = useState(null)
  const [assessment, setAssessment] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelStatus, setModelStatus] = useState('checking')
  const [weather, setWeather] = useState(null)
  const [aqi, setAqi] = useState(null)
  const [expanded, setExpanded] = useState(null)
  const [geoLoading, setGeoLoading] = useState(false)
  const [mapOpen, setMapOpen] = useState(false)
  const [mapSelection, setMapSelection] = useState(null)
  const [activeTab, setActiveTab] = useState('flood')

  useEffect(() => {
    axios.get(`${API}/api/flood/model-status`)
      .then(r => setModelStatus(r.data.status))
      .catch(() => setModelStatus('offline'))
  }, [])

  const reverseGeocode = async (lat, lng) => {
    try {
      const r = await axios.get(NOMINATIM, {
        params: { lat, lon: lng, format: 'json', zoom: 16, addressdetails: 1 },
        headers: { 'Accept-Language': 'en' },
      })
      const a = r.data.address
      const parts = []
      if (a.suburb || a.neighbourhood) parts.push(a.suburb || a.neighbourhood)
      if (a.city || a.town || a.village || a.county) parts.push(a.city || a.town || a.village || a.county)
      if (a.state) parts.push(a.state)
      if (a.country) parts.push(a.country)
      setPlaceName(parts.join(', ') || r.data.display_name)
    } catch {
      setPlaceName(`${lat.toFixed(4)}°N, ${lng.toFixed(4)}°E`)
    }
  }

  const fetchWeather = async (lat, lng) => {
    try {
      const [w, a] = await Promise.all([
        axios.get(WEATHER, {
          params: {
            latitude: lat, longitude: lng,
            current: 'temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,cloud_cover',
            daily: 'temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code',
            timezone: 'auto', forecast_days: 5,
          },
        }),
        axios.get(AQI_URL, {
          params: {
            latitude: lat, longitude: lng,
            current: 'pm10,pm2_5,nitrogen_dioxide,ozone,us_aqi',
          },
        }),
      ])
      setWeather(w.data)
      setAqi(a.data)
    } catch (e) {
      console.error('Weather fetch error:', e)
    }
  }

  const applyLocation = useCallback((lat, lng) => {
    setLoc({ lat, lng })
    setAssessment(null)
    setError(null)
    setPlaceName(null)
    reverseGeocode(lat, lng)
    fetchWeather(lat, lng)
  }, [])

  const useMyLocation = () => {
    if (!navigator.geolocation) {
      setError('Geolocation is not supported by your browser')
      return
    }
    setGeoLoading(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        applyLocation(pos.coords.latitude, pos.coords.longitude)
        setGeoLoading(false)
      },
      () => {
        setError('Location permission denied. Please select on the map instead.')
        setGeoLoading(false)
      },
      { enableHighAccuracy: true, timeout: 10000 },
    )
  }

  const openMap = () => {
    setMapSelection(null)
    setMapOpen(true)
  }

  const confirmMapSelection = () => {
    if (mapSelection) {
      applyLocation(mapSelection.lat, mapSelection.lng)
      setMapOpen(false)
      setMapSelection(null)
    }
  }

  const runAssessment = async () => {
    if (!loc) return
    setLoading(true)
    setError(null)
    setAssessment(null)
    try {
      const r = await axios.post(`${API}/api/flood/assess`, {
        latitude: loc.lat,
        longitude: loc.lng,
        mode: 'demo',
      })
      setAssessment(r.data)
      setActiveTab('flood')
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const wxIcon = (c) =>
    c <= 3 ? '☀️' : c <= 48 ? '🌫️' : c <= 67 ? '🌧️' : c <= 77 ? '🌨️' : c <= 82 ? '🌧️' : '⛈️'

  const aqiInfo = (v) =>
    v <= 50
      ? { label: 'Good', color: '#34d399' }
      : v <= 100
        ? { label: 'Moderate', color: '#fbbf24' }
        : v <= 150
          ? { label: 'Unhealthy (Sensitive)', color: '#fb923c' }
          : { label: 'Unhealthy', color: '#f87171' }

  return (
    <div className="app">
      {/* ─── Header ─── */}
      <header className="header">
        <div className="header-brand">
          <div className="brand-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          </div>
          <div>
            <h1>Disaster Insights</h1>
            <p>Climate-Induced Natural Disaster Assessment</p>
          </div>
        </div>
        <div className={`status-pill ${modelStatus === 'ready' ? 'online' : ''}`}>
          <span className="pulse" />
          {modelStatus === 'ready' ? 'AI Model Online' : modelStatus === 'checking' ? 'Connecting...' : 'Offline'}
        </div>
      </header>

      {/* ─── Version Notice ─── */}
      <div className="version-bar">
        <span>v1.0 — This application is exclusively designed for <strong>Climate-Induced Natural Disasters</strong>. This is the first working release with many updates oncoming.</span>
      </div>

      {/* ─── Main Content ─── */}
      <main className="main">
        {!loc ? (
          /* ── Welcome ── */
          <div className="welcome">
            <div className="welcome-inner">
              <div className="welcome-glow" />
              <div className="welcome-content">
                <h2>Disaster Insights</h2>
                <p className="welcome-tagline">
                  AI-powered assessment platform for climate-induced natural disasters.
                  Leveraging Sentinel-2 satellite imagery, EfficientNet-B0 deep learning,
                  and Grad-CAM explainability.
                </p>
                <div className="welcome-chips">
                  <span className="chip">🌊 Flood Detection</span>
                  <span className="chip">🌡️ Weather Analysis</span>
                  <span className="chip">💨 Air Quality</span>
                  <span className="chip">🔬 Explainable AI</span>
                </div>
                <div className="welcome-actions">
                  <button className="btn-hero" onClick={useMyLocation} disabled={geoLoading}>
                    {geoLoading ? 'Detecting location...' : 'Use My Location'}
                  </button>
                  <button className="btn-secondary" onClick={openMap}>
                    Select on Map
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* ── Dashboard ── */
          <div className="dashboard">
            {/* Location Header */}
            <div className="loc-header glass-card">
              <div className="loc-info">
                <div className="loc-pin">📍</div>
                <div>
                  <h3 className="loc-name">{placeName || 'Resolving...'}</h3>
                  <span className="loc-coords">{loc.lat.toFixed(4)}°N, {loc.lng.toFixed(4)}°E</span>
                </div>
              </div>
              <div className="loc-actions">
                <button className="btn-sm" onClick={useMyLocation} disabled={geoLoading}>
                  {geoLoading ? '...' : '📱 Relocate'}
                </button>
                <button className="btn-sm" onClick={openMap}>🗺️ Map</button>
              </div>
            </div>

            {/* Tabs */}
            <div className="tabs">
              <button className={`tab-btn ${activeTab === 'flood' ? 'active' : ''}`} onClick={() => setActiveTab('flood')}>
                Flood Assessment
              </button>
              <button className={`tab-btn ${activeTab === 'climate' ? 'active' : ''}`} onClick={() => setActiveTab('climate')}>
                Weather & AQI
              </button>
            </div>

            {/* Tab Panels */}
            <div className="tab-panel">
              {/* ── Flood Tab ── */}
              {activeTab === 'flood' && (
                <>
                  {!assessment && !loading && (
                    <div className="glass-card action-prompt">
                      <h3>Satellite Flood Analysis</h3>
                      <p>Analyze Sentinel-2 satellite imagery using deep learning to detect flood patterns and assess risk at your selected location.</p>
                      <button className="btn-hero" onClick={runAssessment} disabled={loading || modelStatus !== 'ready'}>
                        Run Flood Assessment
                      </button>
                      {modelStatus !== 'ready' && modelStatus !== 'checking' && (
                        <p className="warn-text">Backend server is offline. Start it to enable assessments.</p>
                      )}
                    </div>
                  )}

                  {loading && (
                    <div className="glass-card" style={{ textAlign: 'center', padding: '3rem' }}>
                      <div className="loader" />
                      <p style={{ marginTop: '1rem', color: 'var(--text-2)' }}>Analyzing satellite imagery...</p>
                    </div>
                  )}

                  {error && <div className="alert-card">{error}</div>}

                  {assessment && (
                    <>
                      <div className="demo-tag">
                        Demo Mode — Analyzed archived WorldFloods image: <strong>{assessment.location.demo_image}</strong>
                      </div>

                      {/* Risk Result */}
                      <div className="glass-card risk-card">
                        <div className="risk-row">
                          <div className={`risk-circle ${assessment.risk.level}`}>
                            <span className="risk-pct">{(assessment.prediction.flood_probability * 100).toFixed(0)}%</span>
                            <span className="risk-txt">{assessment.risk.label}</span>
                          </div>
                          <div className="risk-meta">
                            <div className="meta-row"><span>Classification</span><strong>{assessment.prediction.class_name === 'flood' ? '🌊 Flood Detected' : '✅ No Flood'}</strong></div>
                            <div className="meta-row"><span>Confidence</span><strong>{(assessment.prediction.flood_probability * 100).toFixed(1)}%</strong></div>
                            <div className="meta-row"><span>Model</span><strong>EfficientNet-B0</strong></div>
                            <div className="meta-row"><span>Explainability</span><strong>Grad-CAM</strong></div>
                          </div>
                        </div>
                      </div>

                      {/* Grad-CAM */}
                      {(assessment.image_base64 || assessment.gradcam_base64) && (
                        <div className="glass-card">
                          <h4 className="section-title">Model Explainability (Grad-CAM)</h4>
                          <div className="img-grid">
                            {assessment.image_base64 && (
                              <figure className="img-figure" onClick={() => setExpanded(`data:image/png;base64,${assessment.image_base64}`)}>
                                <img src={`data:image/png;base64,${assessment.image_base64}`} alt="Sentinel-2 RGB" />
                                <figcaption>Sentinel-2 RGB</figcaption>
                              </figure>
                            )}
                            {assessment.gradcam_base64 && (
                              <figure className="img-figure" onClick={() => setExpanded(`data:image/png;base64,${assessment.gradcam_base64}`)}>
                                <img src={`data:image/png;base64,${assessment.gradcam_base64}`} alt="Grad-CAM" />
                                <figcaption>Grad-CAM Heatmap</figcaption>
                              </figure>
                            )}
                          </div>
                          <p className="caption">Click to enlarge · Red = high model attention</p>
                        </div>
                      )}

                      <button className="btn-secondary" style={{ width: '100%' }} onClick={() => setAssessment(null)}>
                        ↻ New Assessment
                      </button>
                      <p className="disclaimer-text">{assessment.disclaimer}</p>
                    </>
                  )}
                </>
              )}

              {/* ── Climate Tab ── */}
              {activeTab === 'climate' && (
                <>
                  {!weather ? (
                    <div className="glass-card" style={{ textAlign: 'center', padding: '3rem' }}>
                      <div className="loader" />
                      <p style={{ marginTop: '1rem', color: 'var(--text-2)' }}>Loading weather data...</p>
                    </div>
                  ) : (
                    <>
                      {weather.current && (
                        <div className="glass-card">
                          <h4 className="section-title">{wxIcon(weather.current.weather_code)} Current Conditions</h4>
                          <div className="temp-hero">
                            <span className="temp-big">{weather.current.temperature_2m}°C</span>
                            <span className="temp-sub">Feels like {weather.current.apparent_temperature}°C</span>
                          </div>
                          <div className="metrics-row">
                            <div className="metric"><strong>{weather.current.relative_humidity_2m}%</strong><span>Humidity</span></div>
                            <div className="metric"><strong>{weather.current.wind_speed_10m}</strong><span>Wind km/h</span></div>
                            <div className="metric"><strong>{weather.current.precipitation}</strong><span>Rain mm</span></div>
                            <div className="metric"><strong>{weather.current.cloud_cover}%</strong><span>Clouds</span></div>
                          </div>
                        </div>
                      )}

                      {weather.daily && (
                        <div className="glass-card">
                          <h4 className="section-title">5-Day Forecast</h4>
                          <div className="forecast">
                            {weather.daily.time.map((d, i) => (
                              <div key={d} className="fc-row">
                                <span className="fc-day">{new Date(d).toLocaleDateString('en', { weekday: 'short', month: 'short', day: 'numeric' })}</span>
                                <span>{wxIcon(weather.daily.weather_code[i])}</span>
                                <span className="fc-temp">{Math.round(weather.daily.temperature_2m_max[i])}° / {Math.round(weather.daily.temperature_2m_min[i])}°</span>
                                <span className="fc-rain">{weather.daily.precipitation_sum[i]}mm</span>
                                <span className="fc-prob">{weather.daily.precipitation_probability_max[i]}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {aqi?.current && (() => {
                        const info = aqiInfo(aqi.current.us_aqi)
                        return (
                          <div className="glass-card">
                            <h4 className="section-title">Air Quality Index</h4>
                            <div className="aqi-ring" style={{ '--aqi-color': info.color }}>
                              <span className="aqi-val">{aqi.current.us_aqi}</span>
                              <span className="aqi-label">{info.label}</span>
                            </div>
                            <div className="metrics-row two-col">
                              <div className="metric"><strong>{aqi.current.pm2_5?.toFixed(1)}</strong><span>PM2.5 µg/m³</span></div>
                              <div className="metric"><strong>{aqi.current.pm10?.toFixed(1)}</strong><span>PM10 µg/m³</span></div>
                              <div className="metric"><strong>{aqi.current.ozone?.toFixed(1)}</strong><span>O₃ µg/m³</span></div>
                              <div className="metric"><strong>{aqi.current.nitrogen_dioxide?.toFixed(1)}</strong><span>NO₂ µg/m³</span></div>
                            </div>
                          </div>
                        )
                      })()}
                    </>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </main>

      {/* ─── Map Modal ─── */}
      {mapOpen && (
        <div className="modal-overlay" onClick={() => setMapOpen(false)}>
          <div className="modal-window" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Select Location</h3>
              <button className="modal-close" onClick={() => setMapOpen(false)}>✕</button>
            </div>
            <div className="modal-map">
              <MapContainer
                center={loc ? [loc.lat, loc.lng] : [20.5, 78.9]}
                zoom={loc ? 10 : 5}
                style={{ height: '100%', width: '100%' }}
              >
                <TileLayer
                  attribution='&copy; OpenStreetMap'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <MapClickHandler onSelect={setMapSelection} />
                {mapSelection && <Marker position={[mapSelection.lat, mapSelection.lng]} />}
                {!mapSelection && loc && <Marker position={[loc.lat, loc.lng]} />}
              </MapContainer>
            </div>
            <div className="modal-footer">
              {mapSelection ? (
                <p className="modal-coords">Selected: {mapSelection.lat.toFixed(4)}°N, {mapSelection.lng.toFixed(4)}°E</p>
              ) : (
                <p className="modal-coords">Click on the map to pick a location</p>
              )}
              <button className="btn-hero" onClick={confirmMapSelection} disabled={!mapSelection}>
                Confirm Location
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ─── Image Lightbox ─── */}
      {expanded && (
        <div className="modal-overlay" onClick={() => setExpanded(null)}>
          <img src={expanded} alt="Enlarged" className="lightbox-img" />
        </div>
      )}
    </div>
  )
}
