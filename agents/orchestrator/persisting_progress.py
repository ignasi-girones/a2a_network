"""PersistingProgressCallback — writes every event to the DebateStore.

Sits at the OUTERMOST layer of the progress callback chain:

    PersistingProgressCallback ⊃ MetricsProgressCallback ⊃ SSEProgressCallback

We persist *before* relaying so the SSE listener never sees an event that
isn't yet in the DB. If the F5 client then refetches via /events?since=N
it sees the same sequence of events (no holes, no duplicates).

A persistence failure is logged and swallowed: the live stream still
reaches the client, the debate continues. We don't want a transient disk
issue to abort an in-progress deliberation.
"""

from __future__ import annotations

import logging

from agents.orchestrator.debate_store import DebateStore
from agents.orchestrator.plan_executor import ProgressCallback

logger = logging.getLogger(__name__)


class PersistingProgressCallback(ProgressCallback):
    """Wraps an inner ProgressCallback, persisting each event before relaying.

    Args:
        inner: the next callback in the chain (typically the metrics+SSE pair).
        store: shared DebateStore instance.
        debate_id: which debate row this run is associated with.
    """

    def __init__(
        self,
        inner: ProgressCallback,
        store: DebateStore,
        debate_id: str,
    ) -> None:
        self.inner = inner
        self.store = store
        self.debate_id = debate_id

    async def on_progress(
        self, stage: str, message: str, data: dict | None = None
    ) -> None:
        try:
            await self.store.record_event(
                self.debate_id, stage, message, data
            )
        except Exception as e:
            # Don't kill the run over a write error — log and keep going.
            logger.warning(
                "Failed to persist event stage=%s for debate %s: %s",
                stage,
                self.debate_id,
                e,
            )
        # Always relay, even if persistence failed: the user looking at the
        # live page should still see the event happen, even if F5 will lose it.
        await self.inner.on_progress(stage, message, data)
