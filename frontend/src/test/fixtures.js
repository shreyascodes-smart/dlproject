/** Response fixtures shaped like the FastAPI backend (Milestone 5 contract) and Open-Meteo. */
export const IMG = 'data:image/jpeg;base64,/9j/AAAA'
export const WEBP = 'data:image/webp;base64,UklGRgAAAAA='

export const status = {
  status: 'ready',
  model_version: 'segmentation_v2',
  model: {
    version: 'segmentation_v2',
    task: 'pixel-level segmentation',
    target: 'surface_water',
    bands: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
    preprocessing: 'segmentation_v2:B02-B03-B04-B08-B11-B12:standardize',
    architecture: 'U-Net',
    encoder: 'efficientnet-b2',
    epoch: 7,
    checkpoint: 'best_seg_model.pt',
    input_size: 256,
    device: 'cuda',
    reported_test_metrics: { split: 'test', n_samples: 22, flood_iou: 0.41, flood_dice: 0.58, flood_precision: 0.52, flood_recall: 0.66 },
    legacy_classifier: 'not used',
  },
  policy_version: 'segmentation_v2-val-20261028',
  methods: { tiled: '256 px tiles…', resize: 'Whole scene…' },
  default_method: 'tiled',
  demo_samples: 3,
  satellite: 'configured',
}

const sample = (id, split, extra = {}) => ({
  id,
  label: `${id.split('_')[0]} · X`,
  event_id: id.split('_')[0],
  aoi_code: id.split('_')[1],
  split,
  split_note: split === 'test' ? 'Held-out test sample. … illustration only.' : 'Validation sample. …',
  event_type: 'Flash flood',
  s2_date: '2019-02-19',
  center: { lat: -18.2, lng: 140.9 },
  bbox: { west: 140.6, east: 141.2, south: -18.5, north: -17.9 },
  width: 5469,
  height: 5579,
  reference_flood_pct: 59.27,
  label_status: 'positive',
  ...extra,
})

export const samples = {
  count: 3,
  default_method: 'tiled',
  methods: ['tiled', 'resize'],
  samples: [
    sample('EMSR264_08VATOMANDRY_DEL_v2', 'val', { reference_flood_pct: 0.54 }),
    sample('EMSR342_07SOUTHNORMANTON_DEL_MONIT03_v2', 'val'),
    sample('EMSR9284_01YLITORNIONORTHERN_DEL_MONIT01_v1', 'test', { reference_flood_pct: 0.54 }),
  ],
  excluded: [],
  note: '…',
}

const LIMITATIONS = [
  "On held-out test events this model's flood IoU was 0.41 (precision 0.52, recall 0.66), so many flood pixels are missed and some non-flood pixels are flagged.",
  'Flood coverage is a ratio of image pixels, not a measured area in km².',
]

const HEADLINES = {
  insufficient_coverage: 'Insufficient valid coverage',
  uncertain_water: 'Inconclusive — mostly permanent water',
  none_detected: 'No flood-like patterns detected',
  inconclusive: 'Inconclusive',
  flood_like: 'Flood-like patterns detected',
}

const SUMMARIES = {
  insufficient_coverage: 'There are not enough clear pixels in this image to assess the area.',
  uncertain_water: 'Most of the usable area is permanent water or sea, where flood water cannot be told apart from normal water in a single image.',
  none_detected: 'No flood-like water was found outside known water bodies in the valid analysed pixels. Flooding under cloud, vegetation or after the image date would not be visible.',
  inconclusive: 'Some flood-like water was found, but no more than this model also marks on reference scenes without mapped flooding.',
  flood_like: 'More flood-like water was found outside known water bodies than this model marked on any reference scene without mapped flooding. Not verified against ground observations.',
}

export function assessmentBlock(level, value, method = 'tiled') {
  const suggestion = { insufficient_coverage: 'Try again when a clearer image is available.', uncertain_water: 'Choose a point further inland to assess land areas.' }[level] || null
  return {
    level,
    headline: HEADLINES[level],
    summary: SUMMARIES[level],
    reasons: level === 'insufficient_coverage' ? ['Only 10% of the image is usable; the rest is cloud, shadow or missing data.'] : [],
    limitations: [],
    uncertainty: [],
    suggestion,
    next_steps: suggestion ? [suggestion] : [],
    flood_like_outside_known_water_pct: value,
    basis: {
      kind: 'descriptive_model_outcome', is_risk_rating: false, is_probability: false, method,
      none_below_pct: 1, inconclusive_ceiling_pct: method === 'tiled' ? 6.5 : 4.2,
      min_assessable_land_frac: 0.25, water_dominated_frac: 0.5, policy_version: 'segmentation_v2-val-20261028',
      explanation: 'Outcomes compare the flood-like share outside known water bodies with …',
    },
  }
}

const mapBlock = (b) => ({
  crs: 'EPSG:4326',
  bounds: [[b.south, b.west], [b.north, b.east]],
  satellite: WEBP, flood: WEBP, set_aside: WEBP, width: 800, height: 800,
})

const modelBlock = {
  version: 'segmentation_v2', architecture: 'U-Net', encoder: 'efficientnet-b2', epoch: 7, checkpoint: 'best_seg_model.pt',
  target: 'surface_water', bands: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'], preprocessing: 'segmentation_v2:…', policy_version: 'segmentation_v2-val-20261028', device: 'cuda',
}

function waterContext(nPixels, eligible, land, floodLand, floodWater) {
  const pct = (a, b) => (b ? Math.round((1e6 * a) / b) / 1e4 : null)
  return {
    available: true,
    n_eligible: eligible,
    n_eligible_outside_known_water: land,
    known_water_pct_of_eligible: pct(eligible - land, eligible),
    sea_pct_of_eligible: 0,
    n_flood_like_raw: floodLand + floodWater,
    n_flood_like_on_known_water: floodWater,
    n_flood_like_outside_known_water: floodLand,
    raw_flood_like_pct: pct(floodLand + floodWater, eligible),
    flood_like_on_known_water_share_pct: pct(floodWater, floodLand + floodWater),
    flood_like_outside_known_water_pct: pct(floodLand, land),
  }
}

export function assessment(sampleObj, method = 'tiled', overrides = {}) {
  const wc = waterContext(30512151, 29700000, 29000000, method === 'tiled' ? 12600000 : 6800000, 300000)
  const share = wc.flood_like_outside_known_water_pct
  return {
    status: 'success',
    mode: 'demo',
    method,
    method_note: method === 'tiled' ? '256 px tiles at native 10 m resolution' : 'Whole scene shrunk to 256 x 256 px',
    experimental: false,
    sample: sampleObj,
    imagery: {
      source: 'WorldFloods v2 archive — Copernicus Sentinel-2 Level-1C',
      acquisition_date: sampleObj.s2_date,
      resolution_m: 10, width: sampleObj.width, height: sampleObj.height,
      center: sampleObj.center, bbox: sampleObj.bbox,
      location_note: 'This archived scene shows the event’s own location, not the location selected on the map.',
    },
    images: {
      rgb: IMG, overlay: IMG, reference: IMG, error: IMG,
      display_width: 1004, display_height: 1024,
      legend: { flood: '#00aaff', ignored: '#8c5a32', true_positive: '#00c8ff', false_positive: '#ff3ca0', false_negative: '#ffd200' },
    },
    prediction: {
      target: 'surface_water', raw_label: 'surface water (including permanent water)',
      raw_coverage_pct: wc.raw_flood_like_pct, flood_coverage_pct: wc.raw_flood_like_pct,
      coverage_definition: 'Raw model positives … divided by eligible pixels …',
      n_flood_pixels: wc.n_flood_like_raw, n_eligible_pixels: 29700000, n_pixels: 30512151,
      eligible_pct: 97.3, nodata_pct: 0.8, eligibility_source: 'validity_mask+nodata',
      threshold: 0.5, threshold_note: 'A pixel is positive when the model’s softmax score exceeds 0.5. The score is not a calibrated probability or confidence.',
      n_tiles: method === 'tiled' ? 840 : null, inference_s: 30.1,
    },
    reference: {
      available: true,
      source: 'Copernicus EMS mapped flood polygons… (reference data, not a model output)',
      flood_pct_of_supervised: 59.2668, supervised_pct: 98.1,
      metrics: { tp: 9180000, fp: 3700000, fn: 8500000, tn: 8300000, flood_iou: method === 'tiled' ? 0.4324 : 0.2663, flood_dice: 0.6037, flood_precision: 0.7127, flood_recall: 0.5233, pixel_accuracy: 0.59 },
      metrics_raw: { flood_iou: 0.41 },
      metrics_note: 'Pixel comparison of the reported flood-like water … with the EMS reference for this one sample …',
      split_note: sampleObj.split_note,
    },
    quality: { status: 'ok', warnings: [] },
    model: modelBlock,
    assessment: assessmentBlock(share > 6.5 ? 'flood_like' : 'inconclusive', share, method),
    water_context: { ...wc, source: 'JRC Global Surface Water (WorldFloods PERMANENTWATERJRC, class 3 = permanent)' },
    map: mapBlock(sampleObj.bbox),
    freshness: { acquisition_date: sampleObj.s2_date, age_days: 2806, archived: true, label: 'Archived demonstration image' },
    limitations: LIMITATIONS,
    disclaimer: 'Satellite-image flood extent analysis from a research model. Not an official warning; it does not forecast floods.',
    processing_s: 31.4,
    cached: false,
    ...overrides,
  }
}

/** Live result: 500 × 500 px (25 km²) with 10 % unusable and 5 % known water. */
export function liveAssessment({ lat, lng }, level = 'inconclusive', overrides = {}) {
  const d = 0.0225
  const bbox = { west: lng - d, south: lat - d, east: lng + d, north: lat + d }
  const floodLand = { flood_like: 40000, inconclusive: 10000, none_detected: 500, uncertain_water: 1000, insufficient_coverage: 0 }[level]
  const wc = waterContext(250000, 225000, 213750, floodLand, 2000)
  const value = ['uncertain_water', 'insufficient_coverage'].includes(level) ? wc.flood_like_outside_known_water_pct : wc.flood_like_outside_known_water_pct
  return {
    status: 'success', mode: 'live', method: 'tiled', method_note: '256 px tiles…', experimental: true, sample: null,
    imagery: {
      source: 'Copernicus Sentinel-2 Level-1C', processing_level: 'L1C', acquisition_date: '2026-10-18', acquisition_datetime: '2026-10-18T05:32:11Z',
      resolution_m: 10, width: 500, height: 500, crs: 'EPSG:4326', center: { lat, lng }, bbox, tile_cloud_cover_pct: 3.1,
      cloud_mask_source: 'sen2cor_scl', bands: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'], location_note: 'A 5 × 5 km area…',
    },
    images: { rgb: IMG, overlay: IMG, reference: null, error: null, display_width: 500, display_height: 500, legend: { flood: '#00aaff' } },
    prediction: {
      target: 'surface_water', raw_label: 'surface water (including permanent water)', raw_coverage_pct: wc.raw_flood_like_pct,
      flood_coverage_pct: wc.raw_flood_like_pct, coverage_definition: '…', n_flood_pixels: wc.n_flood_like_raw, n_eligible_pixels: 225000, n_pixels: 250000,
      eligible_pct: 90, nodata_pct: 0, eligibility_source: 'validity_mask+nodata', threshold: 0.5,
      threshold_note: 'The score is not a calibrated probability or confidence.', n_tiles: 4, inference_s: 1.1,
    },
    reference: { available: false, note: 'No EMS reference mask exists for live imagery; accuracy cannot be measured.' },
    quality: { status: 'ok', warnings: ['Experimental: live imagery has no ground-truth reference, so this prediction is unvalidated.', 'Clouds and cloud shadows were masked with the Sentinel-2 scene classification (Sen2Cor); thin cloud and haze can still be missed.'] },
    model: modelBlock,
    assessment: assessmentBlock(level, value),
    water_context: { ...wc, source: 'JRC Global Surface Water (test)', reference_status: 'ok' },
    map: mapBlock(bbox),
    freshness: { acquisition_date: '2026-10-18', age_days: 8, archived: false, label: 'Image is 8 days old' },
    limitations: LIMITATIONS,
    disclaimer: 'Satellite-image flood extent analysis from a research model. Not an official warning; it does not forecast floods.',
    processing_s: 22.5,
    cached: false,
    ...overrides,
  }
}

/* ───────────── Open-Meteo ───────────── */
function hours(startIso, n) {
  const out = []
  const t0 = Date.parse(`${startIso}:00Z`)
  for (let i = 0; i < n; i++) out.push(new Date(t0 + i * 3600e3).toISOString().slice(0, 16))
  return out
}

export function forecastResponse() {
  const time = hours('2026-10-27T03:00', 72)
  const days = ['2026-10-26', '2026-10-27', '2026-10-28', '2026-10-29', '2026-10-30', '2026-10-31', '2026-11-01', '2026-11-02', '2026-11-03']
  return {
    latitude: 25.6, longitude: 85.14, timezone: 'Asia/Kolkata', timezone_abbreviation: 'GMT+5:30', utc_offset_seconds: 19800,
    current_units: { temperature_2m: '°C', apparent_temperature: '°C', wind_speed_10m: 'km/h', wind_gusts_10m: 'km/h', pressure_msl: 'hPa', precipitation: 'mm', visibility: 'm' },
    current: {
      time: '2026-10-28T03:15', temperature_2m: 26.4, relative_humidity_2m: 63, apparent_temperature: 28.1, is_day: 0, precipitation: 0,
      precipitation_probability: 3, weather_code: 3, cloud_cover: 94, pressure_msl: 1013, wind_speed_10m: 6.1, wind_direction_10m: 360,
      wind_gusts_10m: 10.4, visibility: 7900, uv_index: 0,
    },
    hourly_units: { temperature_2m: '°C', precipitation: 'mm', wind_speed_10m: 'km/h' },
    hourly: {
      time,
      temperature_2m: time.map((_, i) => 24 + 6 * Math.sin(i / 4)),
      precipitation_probability: time.map((_, i) => (i % 12) * 5),
      precipitation: time.map((_, i) => (i >= 24 && i < 30 ? 0.1 : i < 24 ? 0.25 : 0)),
      weather_code: time.map(() => 3),
      wind_speed_10m: time.map(() => 8),
      relative_humidity_2m: time.map(() => 70),
    },
    daily_units: { precipitation_sum: 'mm', wind_speed_10m_max: 'km/h' },
    daily: {
      time: days,
      weather_code: days.map((_, i) => [3, 61, 3, 51, 95, 51, 3, 51, 1][i]),
      temperature_2m_max: days.map(() => 31),
      temperature_2m_min: days.map(() => 22),
      sunrise: days.map((d) => `${d}T05:55`),
      sunset: days.map((d) => `${d}T17:11`),
      uv_index_max: days.map(() => 4.7),
      precipitation_sum: days.map((_, i) => (i === 4 ? 43.5 : 1)),
      precipitation_probability_max: days.map(() => 40),
      wind_speed_10m_max: days.map(() => 12),
      wind_gusts_10m_max: days.map(() => 30),
      wind_direction_10m_dominant: days.map(() => 90),
    },
  }
}

export function airResponse() {
  const time = hours('2026-10-27T00:00', 96)
  return {
    timezone: 'Asia/Kolkata',
    current_units: { pm2_5: 'μg/m³', pm10: 'μg/m³', carbon_monoxide: 'μg/m³', nitrogen_dioxide: 'μg/m³', ozone: 'μg/m³', sulphur_dioxide: 'μg/m³', dust: 'μg/m³', us_aqi: 'USAQI', european_aqi: 'EAQI' },
    current: {
      time: '2026-10-28T03:00', us_aqi: 166, european_aqi: 89, pm2_5: 112.3, pm10: 113, nitrogen_dioxide: 54, ozone: 34, sulphur_dioxide: 32,
      carbon_monoxide: 594, dust: 3, alder_pollen: null, birch_pollen: null, grass_pollen: null, ragweed_pollen: null,
    },
    hourly: { time, us_aqi: time.map(() => 160), european_aqi: time.map(() => 85), pm2_5: time.map(() => 100) },
  }
}
