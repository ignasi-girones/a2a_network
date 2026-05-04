"""Starlette ASGI middleware that records HTTP request metrics.

Added to each agent's app exactly like ``CORSMiddleware`` — the agent
code never touches this module.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from common.telemetry.metrics import HTTP_DURATION, HTTP_REQUESTS

# Patterns that would explode label cardinality if left as-is.
# We collapse them to template placeholders.
_DYNAMIC_PREFIXES = ("/registry/by-skill/", "/registry/")


def _normalize_path(path: str) -> str:
    """Collapse dynamic path segments to keep Prometheus cardinality bounded.

    ``/registry/analyst`` → ``/registry/{id}``
    ``/registry/by-skill/role_analyst`` → ``/registry/by-skill/{skill}``
    """
    for prefix in _DYNAMIC_PREFIXES:
        if path.startswith(prefix) and len(path) > len(prefix):
            suffix_label = "skill" if "by-skill" in prefix else "id"
            return f"{prefix.rstrip('/')}/{{{suffix_label}}}"
    return path


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Counts requests and records latency histograms per agent."""

    def __init__(self, app, agent_id: str = "unknown") -> None:  # noqa: ANN001
        super().__init__(app)
        self.agent_id = agent_id

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        method = request.method
        path = _normalize_path(request.url.path)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        status = str(response.status_code)
        HTTP_REQUESTS.labels(
            agent_id=self.agent_id, method=method, path=path, status=status
        ).inc()
        HTTP_DURATION.labels(
            agent_id=self.agent_id, method=method, path=path
        ).observe(elapsed)

        return response
