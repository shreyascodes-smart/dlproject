export const LAYER_LABEL = {
  rgb: 'Satellite image',
  overlay: 'Model prediction',
  reference: 'EMS reference',
  error: 'Error view',
}

/** Layers that only exist when the API returns a reference mask (archived scenes). */
export const REFERENCE_LAYERS = new Set(['reference', 'error'])
