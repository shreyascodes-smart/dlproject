import { useState, useCallback, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, Marker, useMapEvents, useMap } from 'react-leaflet'
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

function MapClick({ onClick }) {
  useMapEvents({ click: e => onClick(e.latlng) })
  return null
}

function FlyTo({ lat, lng }) {
  const map = useMap()
  useEffect(() => { if (lat && lng) map.flyTo([lat, lng], 12) }, [lat, lng, map])
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
  const [showMap, setShowMap] = useState(false)
  const [activeTab, setActiveTab] = useState('flood')

  useEffect(() => {
    axios.get(`${API}/api/flood/model-status`).then(r => setModelStatus(r.data.status)).catch(() => setModelStatus('offline'))
  }, [])

  // Reverse geocoding
  const reverseGeocode = async (lat, lng) => {
    try {
      const r = await axios.get(NOMINATIM, {
        params: { lat, lon: lng, format: 'json', zoom: 16, addressdetails: 1 },
        headers: { 'Accept-Language': 'en' }
      })
      const a = r.data.address
      const parts = []
      if (a.suburb || a.neighbourhood) parts.push(a.suburb || a.neighbourhood)
      if (a.city || a.town || a.village || a.county) parts.push(a.city || a.town || a.village || a.county)
      if (a.state) parts.push(a.state)
      if (a.country) parts.push(a.country)
      setPlaceName(parts.join(', ') || r.data.display_name)
    } catch { setPlaceName(null) }
  }

  const fetchWeather = async (lat, lng) => {
    try {
      const [w, a] = await Promise.all([
        axios.get(WEATHER, { params: {
          latitude: lat, longitude: lng,
          current: 'temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m,cloud_cover,surface_pressure',
          daily: 'temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code',
          timezone: 'auto', forecast_days: 5,
        }}),
        axios.get(AQI_URL, { params: {
          latitude: lat, longitude: lng,
          current: 'pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi',
        }})
      ])
      setWeather(w.data)
      setAqi(a.data)
    } catch (e) { console.error('Weather error:', e) }
  }

  const selectLocation = useCallback((lat, lng) => {
    setLoc({ lat, lng })
    setAssessment(null)
    setError(null)
    setPlaceName(null)
    reverseGeocode(lat, lng)
    fetchWeather(lat, lng)
  }, [])

  const handleMapClick = useCallback((latlng) => {
    selectLocation(latlng.lat, latlng.lng)
    setShowMap(false) // auto-close map after selection
  }, [selectLocation])

  const useMyLocation = () => {
    if (!navigator.geolocation) { setError('Geolocation not supported by your browser'); return }
    setGeoLoading(true)
    navigator.geolocation.getCurrentPosition(
      pos => { selectLocation(pos.coords.latitude, pos.coords.longitude); setGeoLoading(false) },
      () => { setError('Location permission denied. Please allow location access or select on map.'); setGeoLoading(false) },
      { enableHighAccuracy: true, timeout: 10000 }
    )
  }

  const runAssessment = async () => {
    if (!loc) return
    setLoading(true); setError(null); setAssessment(null)
    try {
      const r = await axios.post(`${API}/api/flood/assess`, {
        latitude: loc.lat, longitude: loc.lng, mode: 'demo'
      })
      setAssessment(r.data)
      setActiveTab('flood')
    } catch (e) { setError(e.response?.data?.detail || e.message) }
    finally { setLoading(false) }
  }

  const wxIcon = (c) => c <= 3 ? '☀️' : c <= 48 ? '🌫️' : c <= 67 ? '🌧️' : c <= 77 ? '🌨️' : c <= 82 ? '🌧️' : '⛈️'
  const aqiLevel = (v) => v <= 50 ? { l: 'Good', c: '#10b981' } : v <= 100 ? { l: 'Moderate', c: '#f59e0b' } : v <= 150 ? { l: 'Unhealthy (Sensitive)', c: '#f97316' } : { l: 'Unhealthy', c: '#ef4444' }

  const statusText = modelStatus === 'ready' ? 'Model Online' : modelStatus === 'checking' ? 'Loading...' : 'Offline'

  return (
    <div className="app">
      {/* ─── Header ─── */}
      <header className="header">
        <div className="header-left">
          <span className="header-logo">🌍</span>
          <div>
            <h1 className="header-title">ClimaSight</h1>
            <span className="header-sub">Climate-Induced Natural Disaster Assessment</span>
          </div>
        </div>
        <div className={`status ${modelStatus === 'ready' ? 'online' : 'offline'}`}>
          <span className="status-dot"></span>{statusText}
        </div>
      </header>

      <main className="main-layout">
        {/* ─── Top: Location Bar ─── */}
        <section className="location-bar">
          <div className="location-bar-inner">
            <div className="location-info">
              <span className="location-icon">📍</span>
              {loc ? (
                <div className="location-text">
                  <span className="location-name">{placeName || 'Resolving location...'}</span>
                  <span className="location-coords">{loc.lat.toFixed(4)}°N, {loc.lng.toFixed(4)}°E</span>
                </div>
              ) : (
                <span className="location-placeholder">No location selected</span>
              )}
            </div>
            <div className="location-buttons">
              <button className="btn btn-accent" onClick={useMyLocation} disabled={geoLoading}>
                {geoLoading ? '⏳ Detecting...' : '📱 Use My Location'}
              </button>
              <button className="btn btn-ghost" onClick={() => setShowMap(!showMap)}>
                {showMap ? '✕ Close Map' : '🗺️ Select on Map'}
              </button>
            </div>
          </div>

          {/* Collapsible Map */}
          {showMap && (
            <div className="map-drawer">
              <MapContainer center={loc ? [loc.lat, loc.lng] : [20.5, 78.9]} zoom={loc ? 12 : 5} style={{ height: '100%', width: '100%' }}>
                <TileLayer attribution='&copy; OpenStreetMap' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                <MapClick onClick={handleMapClick} />
                {loc && <Marker position={[loc.lat, loc.lng]} />}
                {loc && <FlyTo lat={loc.lat} lng={loc.lng} />}
              </MapContainer>
              <div className="map-hint">Click anywhere on the map to select a location</div>
            </div>
          )}
        </section>

        {/* ─── Content Area ─── */}
        <section className="content-area">
          {!loc ? (
            /* Welcome State */
            <div className="welcome">
              <div className="welcome-card">
                <div className="welcome-icon">🛰️</div>
                <h2>Welcome to ClimaSight</h2>
                <p>AI-powered assessment of climate-induced natural disasters using Sentinel-2 satellite imagery, deep learning (EfficientNet-B0), and Grad-CAM explainability.</p>
                <div className="welcome-features">
                  <div className="feature"><span>🌊</span><span>Flood Detection</span></div>
                  <div className="feature"><span>🌤️</span><span>Weather Data</span></div>
                  <div className="feature"><span>🌬️</span><span>Air Quality</span></div>
                  <div className="feature"><span>🔬</span><span>Grad-CAM AI</span></div>
                </div>
                <button className="btn btn-primary btn-lg" onClick={useMyLocation} disabled={geoLoading}>
                  {geoLoading ? '⏳ Detecting your location...' : '📱 Get Started — Use My Location'}
                </button>
                <button className="btn btn-ghost" onClick={() => setShowMap(true)} style={{ marginTop: '0.5rem' }}>
                  🗺️ Or select a location on the map
                </button>
              </div>
            </div>
          ) : (
            /* Results Tabs */
            <>
              {/* Tab Bar */}
              <div className="tab-bar">
                <button className={`tab ${activeTab === 'flood' ? 'active' : ''}`} onClick={() => setActiveTab('flood')}>
                  🌊 Flood Assessment
                </button>
                <button className={`tab ${activeTab === 'climate' ? 'active' : ''}`} onClick={() => setActiveTab('climate')}>
                  🌤️ Weather & AQI
                </button>
              </div>

              {/* ─── Flood Tab ─── */}
              {activeTab === 'flood' && (
                <div className="tab-content">
                  {!assessment && (
                    <div className="card action-card">
                      <h3 className="card-title">🛰️ Satellite Flood Analysis</h3>
                      <p className="card-desc">Run EfficientNet-B0 deep learning model on Sentinel-2 satellite imagery to detect flood patterns at this location.</p>
                      <button className="btn btn-primary btn-full" onClick={runAssessment} disabled={loading || modelStatus !== 'ready'}>
                        {loading ? (<><span className="spinner"></span> Analyzing satellite imagery...</>) : '🛰️ Run Flood Assessment'}
                      </button>
                      {modelStatus !== 'ready' && (
                        <p className="alert alert-warn">
                          {modelStatus === 'offline' ? '⚠️ Backend server is offline. Start it first.' : '⏳ Model is loading...'}
                        </p>
                      )}
                    </div>
                  )}

                  {error && <div className="alert alert-error">⚠️ {error}</div>}

                  {assessment && (
                    <>
                      <div className="demo-notice">
                        📡 <strong>Demo Mode</strong> — Analyzed archived Sentinel-2 image: <em>{assessment.location.demo_image}</em>
                      </div>

                      {/* Risk Card */}
                      <div className="card">
                        <div className="risk-row">
                          <div className={`risk-badge risk-${assessment.risk.level}`}>
                            <span className="risk-pct">{(assessment.prediction.flood_probability * 100).toFixed(0)}%</span>
                            <span className="risk-lbl">{assessment.risk.label}</span>
                          </div>
                          <div className="risk-details">
                            <div className="detail"><span>Classification</span><strong>{assessment.prediction.class_name === 'flood' ? '🌊 Flood Detected' : '✅ No Flood'}</strong></div>
                            <div className="detail"><span>Confidence</span><strong>{(assessment.prediction.flood_probability * 100).toFixed(1)}%</strong></div>
                            <div className="detail"><span>Model</span><strong>EfficientNet-B0</strong></div>
                            <div className="detail"><span>Explainability</span><strong>Grad-CAM</strong></div>
                          </div>
                        </div>
                      </div>

                      {/* Grad-CAM Images */}
                      {(assessment.image_base64 || assessment.gradcam_base64) && (
                        <div className="card">
                          <h3 className="card-title">🔬 Model Explainability (Grad-CAM)</h3>
                          <div className="image-pair">
                            {assessment.image_base64 && (
                              <div className="img-box" onClick={() => setExpanded(`data:image/png;base64,${assessment.image_base64}`)}>
                                <img src={`data:image/png;base64,${assessment.image_base64}`} alt="Sentinel-2 RGB" />
                                <span>Sentinel-2 RGB</span>
                              </div>
                            )}
                            {assessment.gradcam_base64 && (
                              <div className="img-box" onClick={() => setExpanded(`data:image/png;base64,${assessment.gradcam_base64}`)}>
                                <img src={`data:image/png;base64,${assessment.gradcam_base64}`} alt="Grad-CAM" />
                                <span>Grad-CAM Heatmap</span>
                              </div>
                            )}
                          </div>
                          <p className="hint">Click images to enlarge · Red = high attention areas</p>
                        </div>
                      )}

                      <button className="btn btn-ghost btn-full" onClick={() => setAssessment(null)} style={{ marginTop: '0.25rem' }}>
                        ↻ Run New Assessment
                      </button>

                      <p className="disclaimer">⚠️ {assessment.disclaimer}</p>
                    </>
                  )}
                </div>
              )}

              {/* ─── Climate Tab ─── */}
              {activeTab === 'climate' && (
                <div className="tab-content">
                  {!weather ? (
                    <div className="empty-state"><span className="spinner"></span><p>Loading weather data...</p></div>
                  ) : (
                    <>
                      {/* Current Weather */}
                      {weather.current && (
                        <div className="card">
                          <h3 className="card-title">{wxIcon(weather.current.weather_code)} Current Weather</h3>
                          <div className="wx-hero">
                            <span className="wx-temp">{weather.current.temperature_2m}°C</span>
                            <span className="wx-feels">Feels like {weather.current.apparent_temperature}°C</span>
                          </div>
                          <div className="stat-grid-4">
                            <div className="stat"><span className="stat-val">{weather.current.relative_humidity_2m}%</span><span className="stat-lbl">Humidity</span></div>
                            <div className="stat"><span className="stat-val">{weather.current.wind_speed_10m}</span><span className="stat-lbl">Wind km/h</span></div>
                            <div className="stat"><span className="stat-val">{weather.current.precipitation}</span><span className="stat-lbl">Rain mm</span></div>
                            <div className="stat"><span className="stat-val">{weather.current.cloud_cover}%</span><span className="stat-lbl">Clouds</span></div>
                          </div>
                        </div>
                      )}

                      {/* 5-Day Forecast */}
                      {weather.daily && (
                        <div className="card">
                          <h3 className="card-title">📅 5-Day Forecast</h3>
                          <div className="forecast-list">
                            {weather.daily.time.map((d, i) => (
                              <div key={d} className="forecast-row">
                                <span className="fc-day">{new Date(d).toLocaleDateString('en', { weekday: 'short', month: 'short', day: 'numeric' })}</span>
                                <span className="fc-icon">{wxIcon(weather.daily.weather_code[i])}</span>
                                <span className="fc-temp">{Math.round(weather.daily.temperature_2m_max[i])}° / {Math.round(weather.daily.temperature_2m_min[i])}°</span>
                                <span className="fc-rain">💧 {weather.daily.precipitation_sum[i]}mm</span>
                                <span className="fc-prob">{weather.daily.precipitation_probability_max[i]}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* AQI */}
                      {aqi?.current && (
                        <div className="card">
                          <h3 className="card-title">🌬️ Air Quality Index</h3>
                          <div className="aqi-hero" style={{ borderColor: aqiLevel(aqi.current.us_aqi).c }}>
                            <span className="aqi-num" style={{ color: aqiLevel(aqi.current.us_aqi).c }}>{aqi.current.us_aqi}</span>
                            <span className="aqi-lbl" style={{ color: aqiLevel(aqi.current.us_aqi).c }}>{aqiLevel(aqi.current.us_aqi).l}</span>
                          </div>
                          <div className="stat-grid-2">
                            <div className="stat"><span className="stat-val">{aqi.current.pm2_5?.toFixed(1)}</span><span className="stat-lbl">PM2.5 µg/m³</span></div>
                            <div className="stat"><span className="stat-val">{aqi.current.pm10?.toFixed(1)}</span><span className="stat-lbl">PM10 µg/m³</span></div>
                            <div className="stat"><span className="stat-val">{aqi.current.ozone?.toFixed(1)}</span><span className="stat-lbl">Ozone µg/m³</span></div>
                            <div className="stat"><span className="stat-val">{aqi.current.nitrogen_dioxide?.toFixed(1)}</span><span className="stat-lbl">NO₂ µg/m³</span></div>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </>
          )}
        </section>
      </main>

      {/* Image Modal */}
      {expanded && (
        <div className="modal-backdrop" onClick={() => setExpanded(null)}>
          <img src={expanded} alt="Enlarged" className="modal-img" />
          <p className="modal-hint">Click anywhere to close</p>
        </div>
      )}
    </div>
  )
}
