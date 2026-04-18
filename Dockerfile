# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the A2A debate network backend.
#
# Stage 1 (builder) installs dependencies into /opt/venv. Stage 2 copies that
# venv into a slim runtime image that never sees pip/build toolchains. The
# final image runs as a non-root user and uses tini as PID 1 so SIGTERM is
# forwarded cleanly to Python on `docker stop`.

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time deps for any wheels that need to compile (grpc/protobuf, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtualenv so the runtime stage can copy it wholesale.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

# tini: proper PID-1 so SIGTERM/SIGINT reach uvicorn and drain connections.
# curl: used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for defense-in-depth. UID 1000 matches the default on most
# Debian/Ubuntu hosts, which keeps bind-mounted volumes writable.
RUN groupadd --system --gid 1000 app \
 && useradd --system --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy only what we actually run. Tests/frontend/docs stay out of the image.
COPY --chown=app:app common/ common/
COPY --chown=app:app agents/ agents/

USER app

# Default to the orchestrator. docker-compose overrides `command:` per service.
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "agents.orchestrator"]

# Each agent exposes /.well-known/agent.json (A2A discovery). A 200 there
# means the ASGI app is up and serving. Individual services override this
# healthcheck in docker-compose with their own port — the default here
# targets the orchestrator on 8080 (matches the nattech:40530 tunnel).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8080/.well-known/agent.json || exit 1
