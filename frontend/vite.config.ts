import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Port and proxy target are configurable via env so the same image runs
// locally (`npm run dev`) and inside Docker Compose.
//   VITE_PORT           — port Vite listens on (default 8086)
//   VITE_PROXY_TARGET   — where `/api/*` gets proxied
//                         local dev: http://localhost:8080
//                         compose:   http://orchestrator:8080
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const port = Number(env.VITE_PORT ?? process.env.VITE_PORT ?? 8086)
  const proxyTarget =
    env.VITE_PROXY_TARGET ??
    process.env.VITE_PROXY_TARGET ??
    'http://localhost:8080'

  return {
    plugins: [react(), tailwindcss()],
    server: {
      host: '0.0.0.0',
      port,
      // Allow external hostnames (nattech.fib.upc.edu) to hit the dev server.
      allowedHosts: ['nattech.fib.upc.edu', 'localhost', '.fib.upc.edu'],
      proxy: {
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
