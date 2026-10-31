import { Component } from 'react'
import { RefreshCw, TriangleAlert } from 'lucide-react'

/**
 * Last-resort guard for a page: an unexpected rendering error shows a recoverable message instead
 * of a blank screen. The rest of the app (navigation, other pages) keeps working.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.warn('Page error caught by ErrorBoundary:', error, info?.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="empty-page" role="alert">
        <TriangleAlert size={28} aria-hidden="true" />
        <h2>This page ran into a problem</h2>
        <p>Your selected location and results are kept. Reload the page section to continue.</p>
        <button type="button" className="btn btn--sm" onClick={() => this.setState({ error: null })}>
          <RefreshCw size={14} aria-hidden="true" /> Try again
        </button>
      </div>
    )
  }
}
