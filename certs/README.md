# A2A TLS certificates

Self-signed certificates for mTLS between agents in the production deployment.

## Generation

The `cert-init` service in `docker-compose.production.yml` runs `generate.sh`
automatically before any other service starts. To regenerate manually:

```bash
# From inside Docker (production deployment):
docker compose -f docker-compose.yml -f docker-compose.production.yml run --rm cert-init

# From a WSL/Linux host (advanced — for ad-hoc inspection):
sh certs/generate.sh   # writes to /certs by default; override with CERT_DIR=$(pwd)/certs
```

The script is idempotent: existing certs valid for ≥7 days are left untouched.

## CA root: `PTI_12.1`

Import `ca.pem` into your browser/OS trust store to make HTTPS connections to
`https://nattech.fib.upc.edu:40530` (orchestrator) and `:40536` (frontend)
trusted without warnings.

- **Chrome/Edge (Windows)**: Settings → Privacy & security → Security → Manage
  certificates → Trusted Root Certification Authorities → Import → `ca.pem`.
- **Firefox**: Preferences → Privacy & Security → Certificates → View
  Certificates → Authorities → Import → `ca.pem`. Check "Trust this CA to
  identify websites".

## Files

```
ca.pem, ca.key                     ← Root CA (10 years)
orchestrator.pem, orchestrator.key ← Server + client cert per service (1 year)
normalizer.pem, normalizer.key
ae1.pem, ae1.key
ae2.pem, ae2.key
ae3.pem, ae3.key
feedback.pem, feedback.key
mcp-tools.pem, mcp-tools.key       ← Used by Caddy sidecar in front of MCP
mcp-tls.pem, mcp-tls.key           ← Caddy frontend's own cert (alias)
frontend.pem, frontend.key
prometheus.pem, prometheus.key     ← Used by Prometheus to scrape agents (mTLS client)
grafana.pem, grafana.key           ← Grafana HTTPS server cert
```

Each service cert carries both `serverAuth` and `clientAuth` Extended Key
Usages, so the same cert is used in both directions of any mTLS handshake.

## Security notes

- Private keys (`*.key`) are gitignored — never commit them.
- `ca.key` lives only inside the `cert-init` volume; once certs are issued
  it is not needed for runtime operation. Treat as sensitive.
- For real production (not academic demo): replace the self-signed CA with
  one from your corporate PKI or Let's Encrypt for public-facing services.
