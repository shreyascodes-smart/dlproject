import { useState, useCallback, useEffect } from 'react'
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet'
import axios from 'axios'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import './index.css'

// Fix Leaflet default marker icon
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

const API_URL = 'http://localhost:8000'
const WEATHER_API = 'https://api.open-meteo.com/v1/forecast'
const AQI_API = 'https://air-quality-api.open-meteo.com/v1/air-quality'

function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click: (e) => {
      onMapClick(e.latlng)
    },
  })
  return null
}

function App() {
  const [selectedLocation, setSelectedLocation] = useState(null)
  const [assessment, setAssessment] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelStatus, setModelStatus] = useState('checking')
  const [demoImages, setDemoImages] = useState([])
  const [selectedImageIdx, setSelectedImageIdx] = useState(null)
  const [weather, setWeather] = useState(null)
  const [airQuality, setAirQuality] = useState(null)
  const [weatherLoading, setWeatherLoading] = useState(false)
  const [expandedImage, setExpandedImage] = useState(null)

  // Check model status on mount
  useEffect(() => {
    checkModelStatus()
    fetchDemoImages()
  }, [])

  const checkModelStatus = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/flood/model-status`)
      setModelStatus(res.data.status)
    } catch {
      setModelStatus('offline')
    }
  }

  const fetchDemoImages = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/flood/demo-images`)
      setDemoImages(res.data.images || [])
    } catch {
      // Backend may not be running yet
    }
  }

  const fetchWeatherData = async (lat, lng) => {
    setWeatherLoading(true)
    try {
      // Fetch weather
      const weatherRes = await axios.get(WEATHER_API, {
        params: {
          latitude: lat,
          longitude: lng,
          current: 'temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m,surface_pressure',
          daily: 'temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,precipitation_probability_max',
          timezone: 'auto',
          forecast_days: 3,
        }
      })
      setWeather(weatherRes.data)

      // Fetch AQI
      const aqiRes = await axios.get(AQI_API, {
        params: {
          latitude: lat,
          longitude: lng,
          current: 'pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi',
        }
      })
      setAirQuality(aqiRes.data)
    } catch (err) {
      console.error('Weather/AQI fetch error:', err)
    } finally {
      setWeatherLoading(false)
    }
  }

  const handleMapClick = useCallback((latlng) => {
    setSelectedLocation({ lat: latlng.lat, lng: latlng.lng })
    setAssessment(null)
    setError(null)
    setSelectedImageIdx(null)
    fetchWeatherData(latlng.lat, latlng.lng)
  }, [])

  const runAssessment = async () => {
    if (!selectedLocation) return

    setLoading(true)
    setError(null)
    setAssessment(null)

    try {
      const res = await axios.post(`${API_URL}/api/flood/assess`, {
        latitude: selectedLocation.lat,
        longitude: selectedLocation.lng,
        mode: 'demo',
        demo_image_index: selectedImageIdx,
      })
      setAssessment(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Assessment failed')
    } finally {
      setLoading(false)
    }
  }

  const getWeatherIcon = (code) => {
    if (code <= 3) return '☀️'
    if (code <= 48) return '🌫️'
    if (code <= 57) return '🌧️'
    if (code <= 67) return '🌧️'
    if (code <= 77) return '🌨️'
    if (code <= 82) return '🌧️'
    if (code <= 86) return '🌨️'
    if (code <= 99) return '⛈️'
    return '🌤️'
  }

  const getAqiLevel = (aqi) => {
    if (aqi <= 50) return { label: 'Good', color: 'var(--risk-low)', bg: 'rgba(16,185,129,0.15)' }
    if (aqi <= 100) return { label: 'Moderate', color: 'var(--accent-amber)', bg: 'rgba(245,158,11,0.15)' }
    if (aqi <= 150) return { label: 'Unhealthy (Sensitive)', color: '#ff6b35', bg: 'rgba(255,107,53,0.15)' }
    if (aqi <= 200) return { label: 'Unhealthy', color: 'var(--accent-red)', bg: 'rgba(239,68,68,0.15)' }
    return { label: 'Hazardous', color: 'var(--risk-very-high)', bg: 'rgba(185,28,28,0.15)' }
  }

  const getStatusBadge = () => {
    const statusMap = {
      ready: { dot: 'ready', text: 'Model Ready' },
      not_ready: { dot: 'loading', text: 'Model Loading' },
      checking: { dot: 'loading', text: 'Checking...' },
      offline: { dot: 'error', text: 'Backend Offline' },
    }
    const s = statusMap[modelStatus] || statusMap.offline
    return (
      <div className="status-badge">
        <span className={`status-dot ${s.dot}`}></span>
        {s.text}
      </div>
    )
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo">
          <div className="logo-icon">🌍</div>
          <div>
            <h1>ClimaSight</h1>
            <span className="header-subtitle">Climate Disaster Assessment</span>
          </div>
        </div>
        {getStatusBadge()}
      </header>

      {/* Main Content */}
      <main className="app-main">
        {/* Map Panel */}
        <div className="map-panel">
          <MapContainer
            center={[20.5937, 78.9629]}
            zoom={5}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <MapClickHandler onMapClick={handleMapClick} />
            {selectedLocation && (
              <Marker position={[selectedLocation.lat, selectedLocation.lng]} />
            )}
          </MapContainer>
          {!selectedLocation && (
            <div className="map-instructions">
              Click anywhere on the map to select a location for disaster assessment
            </div>
          )}
        </div>

        {/* Side Panel */}
        <div className="side-panel">
          {/* Location Card */}
          {selectedLocation ? (
            <div className="card animate-fade-in">
              <div className="card-header">
                <div className="icon" style={{ background: 'rgba(59, 130, 246, 0.15)', color: 'var(--accent-blue)' }}>📍</div>
                <div>
                  <h3>Selected Location</h3>
                  <span className="subtitle">Click map to change</span>
                </div>
              </div>
              <div className="detail-row">
                <span className="label">Latitude</span>
                <span className="value">{selectedLocation.lat.toFixed(4)}°</span>
              </div>
              <div className="detail-row">
                <span className="label">Longitude</span>
                <span className="value">{selectedLocation.lng.toFixed(4)}°</span>
              </div>

              {/* Demo Image Selector */}
              {demoImages.length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <label className="label" style={{ fontSize: '0.75rem', display: 'block', marginBottom: '0.375rem' }}>
                    Select satellite image for analysis:
                  </label>
                  <select
                    className="demo-select"
                    value={selectedImageIdx ?? ''}
                    onChange={(e) => setSelectedImageIdx(e.target.value === '' ? null : parseInt(e.target.value))}
                  >
                    <option value="">Auto (based on coordinates)</option>
                    {demoImages.map((img) => (
                      <option key={img.index} value={img.index}>
                        {img.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <button
                className="btn btn-primary btn-full"
                onClick={runAssessment}
                disabled={loading || modelStatus !== 'ready'}
                style={{ marginTop: '0.75rem' }}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing satellite imagery...
                  </>
                ) : (
                  '🛰️ Run Flood Assessment'
                )}
              </button>
              {modelStatus !== 'ready' && modelStatus !== 'checking' && (
                <p style={{ fontSize: '0.75rem', color: 'var(--accent-amber)', marginTop: '0.5rem', textAlign: 'center' }}>
                  {modelStatus === 'offline' ? 'Start the backend server first' : 'Model is loading...'}
                </p>
              )}
            </div>
          ) : (
            <div className="card welcome-state">
              <div className="emoji">🛰️</div>
              <h2>ClimaSight</h2>
              <p>
                AI-powered climate disaster assessment using Sentinel-2 satellite
                imagery, deep learning, and explainable AI (Grad-CAM).
              </p>
              <p style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Click on the map to begin analysis.
              </p>
            </div>
          )}

          {/* Weather & Climate Card */}
          {weather && weather.current && (
            <div className="card animate-fade-in">
              <div className="card-header">
                <div className="icon" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)' }}>
                  {getWeatherIcon(weather.current.weather_code)}
                </div>
                <div>
                  <h3>Weather & Climate</h3>
                  <span className="subtitle">Current conditions at location</span>
                </div>
              </div>
              <div className="weather-grid">
                <div className="weather-item">
                  <span className="weather-value">{weather.current.temperature_2m}°C</span>
                  <span className="weather-label">Temperature</span>
                </div>
                <div className="weather-item">
                  <span className="weather-value">{weather.current.relative_humidity_2m}%</span>
                  <span className="weather-label">Humidity</span>
                </div>
                <div className="weather-item">
                  <span className="weather-value">{weather.current.wind_speed_10m} km/h</span>
                  <span className="weather-label">Wind Speed</span>
                </div>
                <div className="weather-item">
                  <span className="weather-value">{weather.current.precipitation} mm</span>
                  <span className="weather-label">Precipitation</span>
                </div>
              </div>
              {weather.daily && (
                <div style={{ marginTop: '0.75rem' }}>
                  <span className="label" style={{ fontSize: '0.6875rem', display: 'block', marginBottom: '0.375rem' }}>3-Day Forecast</span>
                  <div className="forecast-grid">
                    {weather.daily.time.map((date, i) => (
                      <div key={date} className="forecast-item">
                        <span className="forecast-date">{new Date(date).toLocaleDateString('en', { weekday: 'short' })}</span>
                        <span className="forecast-temp">
                          {Math.round(weather.daily.temperature_2m_max[i])}° / {Math.round(weather.daily.temperature_2m_min[i])}°
                        </span>
                        <span className="forecast-rain">🌧 {weather.daily.precipitation_sum[i]}mm</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* AQI Card */}
          {airQuality && airQuality.current && (
            <div className="card animate-fade-in">
              <div className="card-header">
                <div className="icon" style={{
                  background: getAqiLevel(airQuality.current.us_aqi).bg,
                  color: getAqiLevel(airQuality.current.us_aqi).color
                }}>🌬️</div>
                <div>
                  <h3>Air Quality Index</h3>
                  <span className="subtitle">{getAqiLevel(airQuality.current.us_aqi).label}</span>
                </div>
              </div>
              <div className="aqi-display">
                <div className="aqi-circle" style={{
                  borderColor: getAqiLevel(airQuality.current.us_aqi).color,
                  color: getAqiLevel(airQuality.current.us_aqi).color,
                }}>
                  <span className="aqi-value">{airQuality.current.us_aqi}</span>
                  <span className="aqi-label">US AQI</span>
                </div>
              </div>
              <div className="aqi-grid">
                <div className="detail-row">
                  <span className="label">PM2.5</span>
                  <span className="value">{airQuality.current.pm2_5?.toFixed(1)} µg/m³</span>
                </div>
                <div className="detail-row">
                  <span className="label">PM10</span>
                  <span className="value">{airQuality.current.pm10?.toFixed(1)} µg/m³</span>
                </div>
                <div className="detail-row">
                  <span className="label">Ozone (O₃)</span>
                  <span className="value">{airQuality.current.ozone?.toFixed(1)} µg/m³</span>
                </div>
                <div className="detail-row">
                  <span className="label">NO₂</span>
                  <span className="value">{airQuality.current.nitrogen_dioxide?.toFixed(1)} µg/m³</span>
                </div>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="card animate-slide-up" style={{ borderColor: 'rgba(239, 68, 68, 0.3)' }}>
              <div className="card-header">
                <div className="icon" style={{ background: 'rgba(239, 68, 68, 0.15)', color: 'var(--accent-red)' }}>⚠️</div>
                <h3>Error</h3>
              </div>
              <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>{error}</p>
            </div>
          )}

          {/* Assessment Result */}
          {assessment && (
            <>
              {/* Demo Mode Banner */}
              <div className="demo-banner animate-slide-up">
                <strong>📡 Demo Mode</strong> — Analyzing real WorldFloods satellite image:
                <em> {assessment.location.demo_image}</em>.
                This is a pre-captured Sentinel-2 image from the Copernicus EMS flood archive,
                not live satellite data from the selected coordinates.
              </div>

              {/* Risk Card */}
              <div className="card animate-slide-up">
                <div className="card-header">
                  <div className="icon" style={{ background: 'rgba(59, 130, 246, 0.15)', color: 'var(--accent-blue)' }}>🎯</div>
                  <div>
                    <h3>Flood Risk Assessment</h3>
                    <span className="subtitle">EfficientNet-B0 · Grad-CAM</span>
                  </div>
                </div>
                <div className="risk-display">
                  <div className={`risk-circle risk-${assessment.risk.level}`}>
                    <span className="probability">
                      {(assessment.prediction.flood_probability * 100).toFixed(0)}%
                    </span>
                    <span className="label">{assessment.risk.label}</span>
                  </div>
                </div>
                <div className="detail-row">
                  <span className="label">Classification</span>
                  <span className="value" style={{ textTransform: 'uppercase' }}>
                    {assessment.prediction.class_name === 'flood' ? '🌊 Flood Detected' : '✅ No Flood'}
                  </span>
                </div>
                <div className="detail-row">
                  <span className="label">Flood Probability</span>
                  <span className="value">{(assessment.prediction.flood_probability * 100).toFixed(1)}%</span>
                </div>
                <div className="detail-row">
                  <span className="label">Source Image</span>
                  <span className="value" style={{ fontSize: '0.6875rem', maxWidth: '200px', textAlign: 'right', wordBreak: 'break-all' }}>
                    {assessment.location.demo_image}
                  </span>
                </div>
              </div>

              {/* Grad-CAM Images — FULL WIDTH, PROPER SIZING */}
              {(assessment.image_base64 || assessment.gradcam_base64) && (
                <div className="card animate-slide-up" style={{ animationDelay: '0.1s' }}>
                  <div className="card-header">
                    <div className="icon" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)' }}>🔬</div>
                    <div>
                      <h3>Model Explainability</h3>
                      <span className="subtitle">Grad-CAM — what the model focused on</span>
                    </div>
                  </div>
                  <div className="image-stack">
                    {assessment.image_base64 && (
                      <div className="image-card-full" onClick={() => setExpandedImage(`data:image/png;base64,${assessment.image_base64}`)}>
                        <img
                          src={`data:image/png;base64,${assessment.image_base64}`}
                          alt="Sentinel-2 satellite imagery (RGB)"
                        />
                        <div className="image-label">🛰️ Sentinel-2 Satellite Image (RGB True Color)</div>
                      </div>
                    )}
                    {assessment.gradcam_base64 && (
                      <div className="image-card-full" onClick={() => setExpandedImage(`data:image/png;base64,${assessment.gradcam_base64}`)}>
                        <img
                          src={`data:image/png;base64,${assessment.gradcam_base64}`}
                          alt="Grad-CAM heatmap overlay showing regions that influenced the prediction"
                        />
                        <div className="image-label">🔥 Grad-CAM Heatmap — Red = high attention, Blue = low</div>
                      </div>
                    )}
                  </div>
                  <p className="image-hint">Click images to enlarge</p>
                </div>
              )}

              {/* Disclaimer */}
              <div className="disclaimer animate-slide-up" style={{ animationDelay: '0.2s' }}>
                ⚠️ {assessment.disclaimer}
              </div>
            </>
          )}

          {/* Loading weather indicator */}
          {weatherLoading && selectedLocation && !weather && (
            <div className="card" style={{ textAlign: 'center', padding: '1rem' }}>
              <span className="spinner" style={{ margin: '0 auto' }}></span>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>Fetching weather data...</p>
            </div>
          )}
        </div>
      </main>

      {/* Expanded Image Modal */}
      {expandedImage && (
        <div className="image-modal" onClick={() => setExpandedImage(null)}>
          <div className="image-modal-content">
            <img src={expandedImage} alt="Enlarged view" />
            <p className="image-modal-hint">Click anywhere to close</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
