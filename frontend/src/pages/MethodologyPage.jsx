import { BookOpen } from 'lucide-react'
import { OUTCOMES } from '../lib/assessment'
import { OutcomeIcon } from '../components/ResultPanel'
import { fmtRatio } from '../lib/format'

const OUTCOME_TEXT = {
  flood_like: 'More flood-like water outside known water bodies than the model marked on any validation scene without mapped flooding.',
  none_detected: 'Less than 1% of the valid land is flood-like. Flooding under cloud, dense vegetation or after the image date is not visible.',
  inconclusive: 'Some flood-like water, but within the range the model also marks on validation scenes without mapped flooding.',
  uncertain_water: 'Most of the usable area is permanent water or sea, where flood water cannot be told apart from normal water in one image.',
  insufficient_coverage: 'Less than a quarter of the image is usable (cloud, shadow or missing data), or no land remains after setting aside known water.',
  imagery_unavailable: 'No usable Sentinel-2 acquisition was found for the area in the search window, or the satellite service could not deliver it.',
  service_unavailable: 'The analysis service or the model is not available, or the request limit was reached.',
}

/** Concise, accurate description of the method, data, outcome rules, limitations and sources. */
export default function MethodologyPage({ status }) {
  const m = status.data?.model
  const t = m?.reported_test_metrics
  return (
    <div className="page page--prose">
      <header className="page__head">
        <div className="page__title">
          <p className="eyebrow">
            <BookOpen size={13} aria-hidden="true" /> Methodology
          </p>
          <h1>How Disaster Insights works</h1>
          <p className="page__sub">A research prototype for flood extent analysis in single Sentinel-2 images. Not an official warning; it does not forecast floods.</p>
        </div>
      </header>

      <article className="prose">
        <section>
          <h2>What one analysis does</h2>
          <ol>
            <li>Finds the most recent clear Copernicus Sentinel-2 Level-1C acquisition of a 5 × 5 km area (searching the last 45 days) at 10 m resolution.</li>
            <li>Masks clouds and cloud shadows with the Sentinel-2 scene classification (Sen2Cor). Masked and missing pixels are excluded from every number.</li>
            <li>
              Runs a U-Net segmentation model ({m ? `${m.version}, ${m.encoder} encoder, bands ${(m.bands || []).join(', ')}` : 'the active model'}) on 256 px tiles at native
              resolution.
            </li>
            <li>Sets aside pixels on known permanent water and sea (JRC Global Surface Water). What remains on land is reported as flood-like water.</li>
            <li>
              Checks haze: if the image is hazier than every training image (darkest 5% of blue reflectance above 0.140), only water that is also dark in
              short-wave infrared is counted and the outcome is reported as inconclusive.
            </li>
            <li>Turns the flood-like share into a descriptive outcome using thresholds derived on the validation split only.</li>
          </ol>
        </section>

        <section>
          <h2>Model and training data</h2>
          {m?.target === 'ems_flood' ? (
            <p>
              The active model ({m.version}) marks <strong>EMS-style flood extent</strong> from the red, green and blue bands only. Pixels it marks on the JRC
              permanent-water layer are set aside before the flood-like share is computed.
            </p>
          ) : (
            <p>
              The model detects <strong>surface water</strong> from six bands (blue, green, red, near-infrared and two short-wave infrared). Water is separated
              from land mainly in the infrared bands, which the earlier RGB-only model could not use. Flood-like water is surface water outside the JRC
              permanent-water layer (as in the WorldFloods / ml4floods work), because one image cannot show whether water is normally there.
            </p>
          )}
          <p>
            Training uses the WorldFloods v2 dataset (Sentinel-2 scenes of Copernicus EMS flood activations). Validation and test events are frozen and event-disjoint:
            no event in them is used for training. Validation chose the checkpoint and the outcome thresholds; the test split was evaluated once.
          </p>
          {t && (
            <p className="callout">
              Held-out test result for the active model: flood IoU {fmtRatio(t.flood_iou, 2)}, precision {fmtRatio(t.flood_precision, 2)}, recall {fmtRatio(t.flood_recall, 2)}
              {t.n_samples ? ` on ${t.n_samples} scenes` : ''}. Pixel-level agreement with mapped floods is limited; treat results as indicative.
            </p>
          )}
        </section>

        <section>
          <h2>Coverage</h2>
          <p>
            The flood-like share is <em>flood-like pixels ÷ valid land pixels</em>. Valid land pixels have image data, are cloud-free and lie outside known permanent
            water and sea. Areas in km² are pixel counts × pixel area (≈100 m² at 10 m) and are approximate. The share is a pixel ratio from one image: it is not a
            probability, a confidence or a risk level.
          </p>
        </section>

        <section>
          <h2>Outcomes</h2>
          <ul className="outcome-list">
            {Object.entries(OUTCOMES).map(([k, v]) => (
              <li key={k} className={`outcome-list__item outcome--${v.tone}`}>
                <span className="outcome__icon">
                  <OutcomeIcon level={k} size={16} />
                </span>
                <div>
                  <strong>{v.short}</strong>
                  <p>{OUTCOME_TEXT[k]}</p>
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section>
          <h2>Limitations</h2>
          <ul>
            <li>One optical image: flooding under cloud, under dense vegetation, in narrow streets or after the acquisition date is not visible.</li>
            <li>Wet soil, shadows, dark roofs and burn scars can look like water; thin cloud and haze can be missed by the cloud mask. Heavy haze is detected only when it exceeds the training range.</li>
            <li>
              The “flood-like” threshold is the largest share the model marked on a validation scene without mapped flooding
              {m?.version === 'segmentation_v2' ? ' (about 40% of the valid land for the current model)' : ''}, so many real floods read as inconclusive
              rather than detected.
            </li>
            <li>The permanent-water layer is a long-term reference; new reservoirs, river changes or seasonal wetlands can be counted as flood-like.</li>
            <li>Live images have no reference flood map, so a live result cannot be scored. Demo scenes show agreement with Copernicus EMS maps for illustration.</li>
            <li>Outcome thresholds come from a small validation set (24 scenes) and describe this model only.</li>
          </ul>
        </section>

        <section>
          <h2>Weather and air quality</h2>
          <p>
            Weather and air-quality data come from Open-Meteo (forecast models, and CAMS for air quality). They are shown for context only and are not inputs to the
            flood model. US AQI and European AQI use different scales and are shown separately.
          </p>
        </section>

        <section>
          <h2>Sources and licences</h2>
          <ul className="sources-list">
            <li>Contains modified Copernicus Sentinel data, via the Copernicus Data Space Ecosystem.</li>
            <li>Permanent water: JRC Global Surface Water (Source: EC JRC/Google).</li>
            <li>Training and demo scenes: WorldFloods v2 (CC BY-NC 4.0), with Copernicus EMS flood maps as reference.</li>
            <li>Weather: Open-Meteo.com (CC BY 4.0). Air quality: Open-Meteo using Copernicus Atmosphere Monitoring Service forecasts.</li>
            <li>Maps: © OpenStreetMap contributors (ODbL). Place search: Photon by Komoot (OpenStreetMap data); place names for clicked points: OpenStreetMap Nominatim.</li>
          </ul>
        </section>
      </article>
    </div>
  )
}
