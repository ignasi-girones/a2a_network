#!/bin/sh
# Generate self-signed CA + per-service certs for the A2A network.
#
# Idempotent: if a cert already exists and is valid for at least 7 more days,
# it is left untouched. Run from inside the cert-init container (alpine + openssl)
# or directly on a Linux/WSL host: `sh certs/generate.sh`.
#
# Output files written to $CERT_DIR (defaults to /certs):
#   ca.pem, ca.key             — root CA (CN=PTI_12.1)
#   <service>.pem, <service>.key — per-service cert with serverAuth+clientAuth EKU
#
# Each service cert has SANs for: <service-name>, localhost, 127.0.0.1, ::1.
# Orchestrator and frontend additionally include nattech.fib.upc.edu for
# external access via the UPC FIB VM tunnel.

set -eu

CERT_DIR="${CERT_DIR:-/certs}"
CA_CN="${CA_CN:-PTI_12.1}"

mkdir -p "$CERT_DIR"

SERVICES="orchestrator normalizer ae1 ae2 ae3 feedback mcp-tools mcp-tls frontend prometheus grafana"

ext_for() {
  case "$1" in
    orchestrator|frontend) echo "nattech.fib.upc.edu" ;;
    *) echo "" ;;
  esac
}

cert_still_valid() {
  cert="$1"
  min_days="$2"
  [ -f "$cert" ] && openssl x509 -in "$cert" -noout -checkend $((min_days * 86400)) >/dev/null 2>&1
}

# 1. Root CA
# Python's ssl module (stricter than curl) requires the CA to carry a
# keyUsage extension and a CA:TRUE basicConstraints — without them
# verification fails with "CA cert does not include key usage extension".
if ! cert_still_valid "$CERT_DIR/ca.pem" 30; then
  echo "Generating root CA (CN=$CA_CN)..."
  openssl genrsa -out "$CERT_DIR/ca.key" 4096 >/dev/null 2>&1
  chmod 600 "$CERT_DIR/ca.key"
  openssl req -x509 -new -nodes -key "$CERT_DIR/ca.key" -sha256 -days 3650 \
    -subj "/CN=$CA_CN/O=A2A Network/OU=TFG UPC FIB" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$CERT_DIR/ca.pem"
else
  echo "Root CA still valid, keeping existing $CERT_DIR/ca.pem"
fi

# 2. Per-service certs
for svc in $SERVICES; do
  cert="$CERT_DIR/$svc.pem"
  key="$CERT_DIR/$svc.key"

  if cert_still_valid "$cert" 7; then
    echo "$svc: cert valid >7d, skip"
    continue
  fi

  ext_dns="$(ext_for "$svc")"
  san="DNS:$svc,DNS:localhost,IP:127.0.0.1,IP:0:0:0:0:0:0:0:1"
  if [ -n "$ext_dns" ]; then
    san="DNS:$ext_dns,$san"
  fi

  cnf="/tmp/openssl-$svc.cnf"
  cat > "$cnf" <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no
[req_distinguished_name]
CN = $svc
[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = $san
EOF

  echo "Generating cert for $svc (SAN: $san)..."
  openssl genrsa -out "$key" 2048 >/dev/null 2>&1
  chmod 600 "$key"
  openssl req -new -key "$key" -config "$cnf" -out "/tmp/$svc.csr" >/dev/null 2>&1
  openssl x509 -req -in "/tmp/$svc.csr" \
    -CA "$CERT_DIR/ca.pem" -CAkey "$CERT_DIR/ca.key" -CAcreateserial \
    -days 365 -sha256 \
    -extensions v3_req -extfile "$cnf" \
    -out "$cert" >/dev/null 2>&1
  chmod 644 "$cert"
  rm -f "/tmp/$svc.csr" "$cnf"
done

echo ""
echo "All certs ready in $CERT_DIR:"
ls -la "$CERT_DIR" | awk '/\.(pem|key)$/ {print "  " $NF}'

# The agent containers run as uid 1000 (USER app in the Dockerfile). Hand
# every cert+key over to that uid so the non-root processes can read their
# own private keys (mode 600 → readable only by owner).
chown -R 1000:1000 "$CERT_DIR"
