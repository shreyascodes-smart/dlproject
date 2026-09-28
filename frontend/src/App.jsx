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

  const handleMapClick = useCallback((latlng) => {
    setSelectedLocation({ lat: latlng.lat, lng: latlng.lng })
    setAssessment(null)
    setError(null)
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
      })
      setAssessment(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Assessment failed')
    } finally {
      setLoading(false)
    }
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
          <div className="logo-icon">🌊</div>
          <div>
            <h1>Flood Assessment</h1>
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
              Click anywhere on the map to select a location for flood assessment
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
              <button
                className="btn btn-primary btn-full"
                onClick={runAssessment}
                disabled={loading || modelStatus !== 'ready'}
                style={{ marginTop: '1rem' }}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  '🔍 Run Flood Assessment'
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
              <h2>Flood Assessment Tool</h2>
              <p>
                Click on the map to select a location, then run the deep learning model
                to assess flood risk using Sentinel-2 satellite imagery analysis.
              </p>
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
              {/* Risk Card */}
              <div className="card animate-slide-up">
                <div className="card-header">
                  <div className="icon" style={{ background: 'rgba(59, 130, 246, 0.15)', color: 'var(--accent-blue)' }}>🎯</div>
                  <div>
                    <h3>Flood Risk Assessment</h3>
                    <span className="subtitle">EfficientNet-B0 prediction</span>
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
                    {assessment.prediction.class_name === 'flood' ? '🌊 Flood' : '✅ Non-Flood'}
                  </span>
                </div>
                <div className="detail-row">
                  <span className="label">Flood Probability</span>
                  <span className="value">{(assessment.prediction.flood_probability * 100).toFixed(1)}%</span>
                </div>
                <div className="detail-row">
                  <span className="label">Mode</span>
                  <span className="value" style={{ textTransform: 'capitalize' }}>{assessment.mode}</span>
                </div>
                {assessment.location.demo_image && (
                  <div className="detail-row">
                    <span className="label">Demo Image</span>
                    <span className="value" style={{ fontSize: '0.6875rem' }}>{assessment.location.demo_image}</span>
                  </div>
                )}
              </div>

              {/* Grad-CAM Images */}
              {(assessment.image_base64 || assessment.gradcam_base64) && (
                <div className="card animate-slide-up" style={{ animationDelay: '0.1s' }}>
                  <div className="card-header">
                    <div className="icon" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)' }}>🔬</div>
                    <div>
                      <h3>Model Explainability</h3>
                      <span className="subtitle">Grad-CAM visualization</span>
                    </div>
                  </div>
                  <div className="image-display">
                    {assessment.image_base64 && (
                      <div className="image-card">
                        <img
                          src={`data:image/png;base64,${assessment.image_base64}`}
                          alt="Satellite imagery"
                        />
                        <div className="image-label">Satellite Image</div>
                      </div>
                    )}
                    {assessment.gradcam_base64 && (
                      <div className="image-card">
                        <img
                          src={`data:image/png;base64,${assessment.gradcam_base64}`}
                          alt="Grad-CAM heatmap overlay"
                        />
                        <div className="image-label">Grad-CAM Heatmap</div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Disclaimer */}
              <div className="disclaimer animate-slide-up" style={{ animationDelay: '0.2s' }}>
                ⚠️ {assessment.disclaimer}
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
