import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import fs from 'node:fs'
import https from 'node:https'

// Port and proxy target are configurable via env so the same image runs
// locally (`npm run dev`) and inside Docker Compose.
//   VITE_PORT           — port Vite listens on (default 8086)
//   VITE_PROXY_TARGET   — where `/api/*` gets proxied
//                         dev:                http://localhost:8080
//                         compose:            http://orchestrator:8080
//                         compose+TLS (prod): https://orchestrator:8080
//   VITE_TLS_ENABLED    — "true" to serve HTTPS using /certs/frontend.{pem,key}
//                         and verify the proxy target against /certs/ca.pem.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const port = Number(env.VITE_PORT ?? process.env.VITE_PORT ?? 8086)
  const proxyTarget =
    env.VITE_PROXY_TARGET ??
    process.env.VITE_PROXY_TARGET ??
    'http://localhost:8080'

  const tlsEnabled =
    (env.VITE_TLS_ENABLED ?? process.env.VITE_TLS_ENABLED) === 'true'

  // Build optional HTTPS / proxy-CA config only when TLS is on. Reading the
  // files at config time keeps Vite startup loud-failing if a cert is missing,
  // instead of producing confusing TLS errors later.
  const httpsConfig = tlsEnabled
    ? {
        cert: fs.readFileSync('/certs/frontend.pem'),
        key: fs.readFileSync('/certs/frontend.key'),
      }
    : undefined

  // For the proxy hop into the orchestrator we need both ends of mTLS:
  //   - verify the orchestrator cert against our CA  (ca)
  //   - present our own client cert                  (cert + key on the agent)
  // Vite forwards `agent` to http-proxy, which uses it for the upstream request.
  const proxyTlsConfig = tlsEnabled
    ? {
        secure: true,
        agent: new https.Agent({
          ca: fs.readFileSync('/certs/ca.pem'),
          cert: fs.readFileSync('/certs/frontend.pem'),
          key: fs.readFileSync('/certs/frontend.key'),
        }),
      }
    : { secure: false }

  return {
    plugins: [react(), tailwindcss()],
    server: {
      host: '0.0.0.0',
      port,
      https: httpsConfig,
      // Allow external hostnames (nattech.fib.upc.edu) to hit the dev server.
      allowedHosts: ['nattech.fib.upc.edu', 'localhost', '.fib.upc.edu'],
      proxy: {
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
          ...proxyTlsConfig,
        },
      },
    },
  }
})
