"""Telemetry package — Prometheus metrics for A2A debate network.

Fully decoupled from agent business logic:
- Agents never import prometheus_client directly.
- A single ``install_telemetry(app, agent_id)`` call wires everything.
- When ``settings.telemetry_enabled`` is False, this is a complete no-op.
"""

from common.telemetry.install import install_telemetry

__all__ = ["install_telemetry"]
