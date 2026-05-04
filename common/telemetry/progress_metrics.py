"""MetricsProgressCallback — captures debate lifecycle events as Prometheus
metrics without modifying the event payload.

Inserted into the callback chain between ``_BeliefRecordingProgress`` and
``SSEProgressCallback``, following the same wrapper pattern established in
``agentic_orchestrator.py``.
"""

from __future__ import annotations

import time
from typing import Any

from agents.orchestrator.plan_executor import ProgressCallback
from common.telemetry.metrics import (
    APORIA_DETECTIONS,
    BELIEF_DELTA_ABS,
    BELIEF_UPDATES,
    DEBATES_ACTIVE,
    DEBATES_TOTAL,
    DELIBERATION_DURATION,
    DELIBERATION_ROUNDS,
    REGISTRY_WORKERS,
    SUBTASK_DISPATCH,
    WORKER_CONFIGURE_DURATION,
    WORKER_SPAWNS,
)


class MetricsProgressCallback(ProgressCallback):
    """Records Prometheus metrics from debate lifecycle events.

    Every event is forwarded unchanged to ``upstream``.
    """

    def __init__(self, upstream: ProgressCallback, agent_id: str) -> None:
        self.upstream = upstream
        self.agent_id = agent_id
        self._debate_start: float | None = None
        self._dispatch_timers: dict[str, float] = {}

    async def on_progress(
        self, stage: str, message: str, data: dict[str, Any] | None = None
    ) -> None:
        self._record(stage, data)
        await self.upstream.on_progress(stage, message, data)

    def _record(self, stage: str, data: dict[str, Any] | None) -> None:
        d = data or {}

        if stage == "discover":
            workers = d.get("workers") or d.get("catalog") or []
            if isinstance(workers, list):
                REGISTRY_WORKERS.labels(agent_id=self.agent_id).set(len(workers))

        elif stage == "plan_ready":
            DEBATES_TOTAL.labels(agent_id=self.agent_id).inc()
            DEBATES_ACTIVE.labels(agent_id=self.agent_id).inc()
            self._debate_start = time.perf_counter()

        elif stage in ("subtask_dispatch", "round_dispatch"):
            role_id = d.get("role_id") or "unknown"
            SUBTASK_DISPATCH.labels(agent_id=self.agent_id, role_id=role_id).inc()
            # Start timer for configure latency
            task_id = d.get("subtask_id") or ""
            if task_id:
                self._dispatch_timers[task_id] = time.perf_counter()

        elif stage == "subtask_done":
            task_id = d.get("subtask_id") or ""
            started = self._dispatch_timers.pop(task_id, None)
            if started is not None:
                role_id = d.get("role_id") or "unknown"
                WORKER_CONFIGURE_DURATION.labels(
                    agent_id=self.agent_id, role_id=role_id
                ).observe(time.perf_counter() - started)

        elif stage == "subtask_failed":
            task_id = d.get("subtask_id") or ""
            self._dispatch_timers.pop(task_id, None)

        elif stage == "belief_update":
            role_id = d.get("role_id") or "unknown"
            phase = d.get("phase") or "unknown"
            BELIEF_UPDATES.labels(
                agent_id=self.agent_id, role_id=role_id, phase=phase
            ).inc()
            delta = d.get("delta")
            if delta is not None:
                BELIEF_DELTA_ABS.labels(
                    agent_id=self.agent_id, role_id=role_id
                ).observe(abs(float(delta)))

        elif stage == "aporia_detected":
            if d.get("detected", False):
                APORIA_DETECTIONS.labels(agent_id=self.agent_id).inc()

        elif stage == "round_start":
            DELIBERATION_ROUNDS.labels(agent_id=self.agent_id).inc()

        elif stage in ("spawn", "spawn_ok"):
            role = d.get("role") or d.get("skill") or "unknown"
            WORKER_SPAWNS.labels(agent_id=self.agent_id, role=role).inc()

        elif stage == "deliberation_complete":
            if self._debate_start is not None:
                elapsed = time.perf_counter() - self._debate_start
                reason = d.get("terminated_reason") or "unknown"
                DELIBERATION_DURATION.labels(
                    agent_id=self.agent_id, terminated_reason=reason
                ).observe(elapsed)
                self._debate_start = None
            DEBATES_ACTIVE.labels(agent_id=self.agent_id).dec()
