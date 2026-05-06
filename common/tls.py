"""TLS configuration helpers for the A2A network.

Production deploys mTLS between every agent. To keep the call sites in
``__main__.py``, ``registry_client.py`` and ``a2a_helpers.py`` clean, this
module centralises:

  - ``uvicorn_tls_kwargs(service)`` — a dict ready to splat into
    ``uvicorn.run(...)`` so the agent serves HTTPS and *requires* a client
    cert signed by our CA (mTLS).
  - ``httpx_tls_kwargs(service)`` — kwargs for ``httpx.AsyncClient(**)``
    so any outgoing call presents the agent's own client cert and trusts
    only our CA.
  - ``url_scheme()`` — ``"https"`` or ``"http"`` according to ``TLS_ENABLED``.

When ``TLS_ENABLED=false`` (the dev default), every helper returns an
empty/no-op value so the existing HTTP-only flow is untouched.
"""

from __future__ import annotations

import logging
import os
import ssl

from common.config import settings

logger = logging.getLogger(__name__)


def _service_name(default: str | None = None) -> str:
    """Pick the service name used to look up our cert/key files.

    Priority: explicit caller arg → ``TLS_SERVICE_NAME`` env → ``SELF_HOST``
    env (Compose convention). A blank result raises so we fail loudly
    instead of mounting the wrong cert.
    """
    name = (
        default
        or os.environ.get("TLS_SERVICE_NAME")
        or os.environ.get("SELF_HOST")
    )
    if not name:
        raise RuntimeError(
            "TLS enabled but neither service arg nor TLS_SERVICE_NAME / "
            "SELF_HOST is set — cannot pick a cert."
        )
    return name


def _cert_paths(service: str) -> tuple[str, str, str]:
    """Return (cert, key, ca) absolute paths for a service."""
    base = settings.tls_cert_dir.rstrip("/")
    return (
        f"{base}/{service}.pem",
        f"{base}/{service}.key",
        f"{base}/ca.pem",
    )


def url_scheme() -> str:
    return "https" if settings.tls_enabled else "http"


def uvicorn_tls_kwargs(service: str | None = None) -> dict:
    """Return kwargs to pass to ``uvicorn.run(...)`` for mTLS.

    Returns an empty dict when TLS is disabled, so callers can simply do:
        uvicorn.run(app, host=..., port=..., **uvicorn_tls_kwargs("orchestrator"))
    """
    if not settings.tls_enabled:
        return {}
    name = _service_name(service)
    cert, key, ca = _cert_paths(name)
    logger.info("TLS enabled for %s — cert=%s, ca=%s", name, cert, ca)
    return {
        "ssl_keyfile": key,
        "ssl_certfile": cert,
        "ssl_ca_certs": ca,
        "ssl_cert_reqs": ssl.CERT_REQUIRED,  # mTLS — client must present cert
    }


def httpx_tls_kwargs(service: str | None = None) -> dict:
    """Return kwargs for ``httpx.AsyncClient(**kwargs)`` doing mTLS.

    The returned dict carries:
      - ``verify``: path to our CA so we trust peer server certs
      - ``cert``: tuple of (cert, key) we present as a client to peers

    Empty dict when TLS is off.
    """
    if not settings.tls_enabled:
        return {}
    name = _service_name(service)
    cert, key, ca = _cert_paths(name)
    return {
        "verify": ca,
        "cert": (cert, key),
    }
