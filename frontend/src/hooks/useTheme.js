import { useCallback, useEffect, useState } from 'react'

const KEY = 'di-theme'

function systemTheme() {
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

function stored() {
  try {
    const v = localStorage.getItem(KEY)
    return v === 'light' || v === 'dark' ? v : null
  } catch {
    return null
  }
}

/** Light/dark theme. Follows the OS until the user picks one; the choice is remembered. */
export function useTheme() {
  const [theme, setTheme] = useState(() => stored() || systemTheme())

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'light' ? '#f4f7fb' : '#070b14')
  }, [theme])

  useEffect(() => {
    if (stored()) return undefined
    const mql = window.matchMedia?.('(prefers-color-scheme: light)')
    if (!mql) return undefined
    const onChange = () => !stored() && setTheme(mql.matches ? 'light' : 'dark')
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [])

  const toggle = useCallback(() => {
    setTheme((t) => {
      const next = t === 'dark' ? 'light' : 'dark'
      try {
        localStorage.setItem(KEY, next)
      } catch {
        /* storage unavailable */
      }
      return next
    })
  }, [])

  return { theme, toggle }
}
