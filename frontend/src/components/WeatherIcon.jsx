import { Cloud, CloudDrizzle, CloudFog, CloudLightning, CloudMoon, CloudRain, CloudSnow, CloudSun, Moon, Sun } from 'lucide-react'

const ICONS = {
  sun: Sun,
  moon: Moon,
  'cloud-sun': CloudSun,
  'cloud-moon': CloudMoon,
  cloud: Cloud,
  fog: CloudFog,
  drizzle: CloudDrizzle,
  rain: CloudRain,
  snow: CloudSnow,
  storm: CloudLightning,
}

/** Decorative weather icon (the condition is always also given as text). */
export default function WeatherIcon({ name, size = 20, className = '' }) {
  const Icon = ICONS[name] || Cloud
  return <Icon size={size} className={`wx-icon wx-icon--${name} ${className}`} aria-hidden="true" />
}
