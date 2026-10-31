import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// The UI calls the API with relative URLs (/api, /ready, /health). In development and
// `vite preview` these are proxied to the FastAPI backend; in production put both behind
// one origin (reverse proxy) or build with VITE_API_URL=https://api.example.org.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000'
  const proxy = Object.fromEntries(['/api', '/ready', '/health'].map((p) => [p, { target, changeOrigin: true }]))
  return {
    plugins: [react()],
    server: { proxy },
    preview: { proxy },
    build: {
      chunkSizeWarningLimit: 700,
    },
    test: {
      environment: 'jsdom',
      setupFiles: ['./src/test/setup.js'],
      css: false,
      include: ['src/**/*.test.{js,jsx}'],
      restoreMocks: true,
      // Integration tests type into inputs and lazy-load pages; allow for a busy machine
      testTimeout: 20000,
    },
  }
})
