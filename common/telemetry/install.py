"""Single-call telemetry installer.

``install_telemetry(app, agent_id)`` wires the Prometheus middleware,
mounts the ``/metrics`` endpoint, and installs the LLM/MCP function
hooks. When ``settings.telemetry_enabled`` is ``False``, this is a
complete no-op — no middleware, no routes, no hooks.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from common.config import settings

if TYPE_CHECKING:
    from starlette.applications import Starlette

logger = logging.getLogger(__name__)


def install_telemetry(app: "Starlette", agent_id: str) -> None:
    """Wire all Prometheus telemetry into a Starlette app.

    Called once per agent process, right before ``uvicorn.run()``.
    """
    if not settings.telemetry_enabled:
        logger.info("Telemetry disabled (TELEMETRY_ENABLED=false)")
        return

    from prometheus_client import make_asgi_app
    from starlette.routing import Mount

    from common.telemetry.hooks import install_llm_hooks, install_mcp_hooks
    from common.telemetry.metrics import AGENT_INFO, PROCESS_START, REGISTRY
    from common.telemetry.middleware import PrometheusMiddleware

    # 1. ASGI middleware — counts requests and records latency.
    app.add_middleware(PrometheusMiddleware, agent_id=agent_id)

    # 2. /metrics endpoint — Prometheus scrapes this.
    metrics_app = make_asgi_app(registry=REGISTRY)
    app.routes.insert(0, Mount("/metrics", app=metrics_app))

    # 3. Function hooks — wrap llm_complete and _call_mcp_tool.
    install_llm_hooks(agent_id)
    install_mcp_hooks(agent_id)

    # 4. Static agent info metric.
    AGENT_INFO.labels(agent_id=agent_id).info({"version": "0.5.0"})
    PROCESS_START.labels(agent_id=agent_id).set(time.time())

    logger.info("Telemetry installed for agent %s", agent_id)
