"""Tests for PersistingProgressCallback: persist-before-relay invariant,
graceful degradation if persistence fails, sequence stability under load."""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import pytest

from agents.orchestrator.debate_store import DebateStore
from agents.orchestrator.persisting_progress import PersistingProgressCallback
from agents.orchestrator.plan_executor import ProgressCallback


class _RecordingInner(ProgressCallback):
    """Captures every relayed event in call order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []

    async def on_progress(
        self, stage: str, message: str, data: dict | None = None
    ) -> None:
        self.calls.append((stage, message, data))


class _BrokenStore:
    """Pretends to be a DebateStore but raises on record_event."""

    async def record_event(self, *_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("disk full")


@pytest.fixture
async def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = DebateStore(path)
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()
        os.unlink(path)


# ── persist-before-relay ordering ──────────────────────────────────────────


async def test_persist_runs_before_inner_on_progress(store):
    """The DB row must exist before the inner callback fires.

    Why: if the live SSE listener saw an event that wasn't yet persisted,
    a competing GET /events?since=N call could miss it (the listener saw
    seq=K but the SELECT returns up to K-1).
    """
    debate_id = await store.create_debate("test")
    observed_state: list[int] = []

    class _Inner(ProgressCallback):
        async def on_progress(
            self, stage: str, message: str, data: dict | None = None
        ) -> None:
            # When this fires, the row must already be in the DB.
            events = await store.get_events(debate_id)
            observed_state.append(len(events))

    cb = PersistingProgressCallback(_Inner(), store, debate_id)
    await cb.on_progress("a", "msg", {"k": 1})
    await cb.on_progress("b", "msg", None)

    assert observed_state == [1, 2]  # at each relay, all prior events persisted


async def test_inner_called_even_when_persist_fails(store):
    """A persistence failure must NOT swallow the live event."""
    inner = _RecordingInner()
    cb = PersistingProgressCallback(inner, _BrokenStore(), "fake-id")
    await cb.on_progress("a", "msg", {"x": 1})
    await cb.on_progress("b", "msg2", None)
    assert [c[0] for c in inner.calls] == ["a", "b"]
    assert inner.calls[0] == ("a", "msg", {"x": 1})


async def test_persist_failure_logged_not_raised(store, caplog):
    """Bug taxonomy: even with a persist exception, no exception bubbles up."""
    inner = _RecordingInner()
    cb = PersistingProgressCallback(inner, _BrokenStore(), "fake-id")
    # Should NOT raise:
    await cb.on_progress("stage", "msg", None)
    # And the warning is logged:
    assert any(
        "Failed to persist event" in r.message for r in caplog.records
    )


# ── seq monotonicity end-to-end ─────────────────────────────────────────────


async def test_seq_matches_relay_order_under_concurrency(store):
    """Many concurrent on_progress calls — seqs in DB must form a permutation
    of 0..N-1, with no duplicates and no gaps."""
    debate_id = await store.create_debate("test")
    inner = _RecordingInner()
    cb = PersistingProgressCallback(inner, store, debate_id)

    N = 25
    await asyncio.gather(
        *[cb.on_progress(f"stage{i}", f"msg{i}", {"i": i}) for i in range(N)]
    )

    events = await store.get_events(debate_id)
    seqs = [e.seq for e in events]
    assert sorted(seqs) == list(range(N))
    assert len(events) == N
    # And every relay also fired (the inner saw N calls).
    assert len(inner.calls) == N


async def test_data_payload_roundtrips_through_persistence(store):
    """The event readable from /events should match what was passed in."""
    debate_id = await store.create_debate("test")
    cb = PersistingProgressCallback(_RecordingInner(), store, debate_id)
    payload = {
        "plan": {"subtasks": [{"id": "t1", "depends_on": []}]},
        "score": 0.83,
        "list": [1, 2, 3],
    }
    await cb.on_progress("plan_ready", "Plan listo", payload)
    events = await store.get_events(debate_id)
    assert len(events) == 1
    assert events[0].stage == "plan_ready"
    assert events[0].message == "Plan listo"
    assert events[0].data == payload
