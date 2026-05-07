"""Tests for DebateStore: persistence + concurrency + pub/sub."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from agents.orchestrator.debate_store import (
    DebateAlreadyActiveError,
    DebateStore,
)


@pytest.fixture
async def store():
    """Fresh on-disk store per test (NamedTemporaryFile so concurrent tests
    don't share a connection — aiosqlite + ":memory:" is per-connection)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = DebateStore(path)
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()
        os.unlink(path)


# ── Lifecycle ──────────────────────────────────────────────────────────────


async def test_create_debate_returns_id_and_running_status(store):
    debate_id = await store.create_debate("¿Vale la pena la IA?")
    assert isinstance(debate_id, str) and len(debate_id) >= 32  # uuid4

    row = await store.get_debate(debate_id)
    assert row is not None
    assert row.status == "running"
    assert row.prompt == "¿Vale la pena la IA?"
    assert row.verdict is None and row.error is None


async def test_create_debate_rejects_when_one_active(store):
    await store.create_debate("primero")
    with pytest.raises(DebateAlreadyActiveError):
        await store.create_debate("segundo")


async def test_mark_completed_unblocks_creating_new_debate(store):
    first = await store.create_debate("primero")
    await store.mark_completed(first, "Veredicto final.")
    second = await store.create_debate("segundo")
    assert second != first

    first_row = await store.get_debate(first)
    assert first_row.status == "completed"
    assert first_row.verdict == "Veredicto final."


async def test_mark_failed_records_error_and_unblocks(store):
    first = await store.create_debate("primero")
    await store.mark_failed(first, "boom: timeout")
    row = await store.get_debate(first)
    assert row.status == "failed"
    assert row.error == "boom: timeout"

    # And we can start a new one
    await store.create_debate("segundo")


async def test_get_active_returns_running_or_none(store):
    assert await store.get_active() is None
    debate_id = await store.create_debate("activo")
    active = await store.get_active()
    assert active is not None and active.id == debate_id

    await store.mark_completed(debate_id, "fin")
    assert await store.get_active() is None


async def test_list_debates_orders_by_created_at_desc(store):
    a = await store.create_debate("primero")
    await store.mark_completed(a, "A")
    b = await store.create_debate("segundo")
    await store.mark_completed(b, "B")
    c = await store.create_debate("tercero")
    rows = await store.list_debates()
    assert [r.id for r in rows] == [c, b, a]


# ── Events ──────────────────────────────────────────────────────────────────


async def test_record_event_assigns_monotonic_seq(store):
    debate_id = await store.create_debate("test")
    s0 = await store.record_event(debate_id, "discover", "msg", {"k": 1})
    s1 = await store.record_event(debate_id, "plan", None, None)
    s2 = await store.record_event(debate_id, "subtask_dispatch", "go", {"id": "t1"})
    assert (s0, s1, s2) == (0, 1, 2)


async def test_get_events_since_filters_correctly(store):
    debate_id = await store.create_debate("test")
    for stage in ("a", "b", "c", "d", "e"):
        await store.record_event(debate_id, stage, None, None)

    all_events = await store.get_events(debate_id)
    assert [e.stage for e in all_events] == ["a", "b", "c", "d", "e"]
    assert [e.seq for e in all_events] == [0, 1, 2, 3, 4]

    tail = await store.get_events(debate_id, since_seq=2)
    assert [e.stage for e in tail] == ["d", "e"]
    assert [e.seq for e in tail] == [3, 4]


async def test_record_event_roundtrips_data_as_json(store):
    debate_id = await store.create_debate("test")
    payload = {"plan": {"subtasks": [{"id": "t1", "deps": ["t0"]}]}, "n": 7}
    await store.record_event(debate_id, "plan_ready", "msg", payload)
    events = await store.get_events(debate_id)
    assert len(events) == 1
    assert events[0].data == payload


async def test_completed_emits_synthetic_verdict_event(store):
    debate_id = await store.create_debate("test")
    await store.record_event(debate_id, "discover", None, None)
    await store.mark_completed(debate_id, "El verdict.")
    events = await store.get_events(debate_id)
    # 1 real event + 1 synthetic verdict = 2
    assert len(events) == 2
    assert events[-1].stage == "verdict"
    assert events[-1].data == {"text": "El verdict."}
    assert events[-1].seq == 1  # monotonic continuation


async def test_failed_emits_synthetic_failed_event(store):
    debate_id = await store.create_debate("test")
    await store.mark_failed(debate_id, "boom")
    events = await store.get_events(debate_id)
    assert events[-1].stage == "failed"
    assert events[-1].data == {"error": "boom"}


# ── Concurrency ────────────────────────────────────────────────────────────


async def test_record_event_seq_stays_monotonic_under_concurrency(store):
    """20 concurrent record_event calls — seq must be 0..19 with no gaps."""
    debate_id = await store.create_debate("test")
    seqs = await asyncio.gather(
        *[
            store.record_event(debate_id, f"stage{i}", None, {"i": i})
            for i in range(20)
        ]
    )
    assert sorted(seqs) == list(range(20))


# ── Pub/Sub ────────────────────────────────────────────────────────────────


async def test_subscribe_receives_live_events(store):
    debate_id = await store.create_debate("test")

    async def consume(events_out):
        async with store.subscribe(debate_id) as q:
            for _ in range(3):
                events_out.append(await q.get())

    received: list[dict] = []
    consumer = asyncio.create_task(consume(received))
    await asyncio.sleep(0)  # let consumer subscribe

    await store.record_event(debate_id, "a", None, None)
    await store.record_event(debate_id, "b", None, {"k": 1})
    await store.record_event(debate_id, "c", None, None)
    await asyncio.wait_for(consumer, timeout=2.0)

    assert [e["stage"] for e in received] == ["a", "b", "c"]
    assert [e["seq"] for e in received] == [0, 1, 2]
    assert received[1]["data"] == {"k": 1}


async def test_subscribe_unregisters_on_exit(store):
    debate_id = await store.create_debate("test")
    async with store.subscribe(debate_id):
        assert debate_id in store._subscribers
        assert len(store._subscribers[debate_id]) == 1
    # On exit the queue is removed and the empty set is dropped.
    assert debate_id not in store._subscribers


async def test_mark_completed_publishes_synthetic_verdict(store):
    debate_id = await store.create_debate("test")

    async def consume():
        async with store.subscribe(debate_id) as q:
            return await asyncio.wait_for(q.get(), timeout=2.0)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await store.mark_completed(debate_id, "x")
    payload = await consumer
    assert payload["stage"] == "verdict"
    assert payload["data"] == {"text": "x"}


# ── Cleanup ────────────────────────────────────────────────────────────────


async def test_cleanup_orphans_marks_running_as_failed(store):
    a = await store.create_debate("first")
    n = await store.cleanup_orphans()
    assert n == 1
    row = await store.get_debate(a)
    assert row.status == "failed"
    assert row.error == "orchestrator restart"
    # And we can create a new debate
    await store.create_debate("after-cleanup")


async def test_cleanup_orphans_noop_when_none_running(store):
    a = await store.create_debate("done")
    await store.mark_completed(a, "v")
    assert await store.cleanup_orphans() == 0


async def test_cleanup_orphans_handles_multiple_via_failsafe(store):
    """Should never happen in practice (one_running unique idx prevents
    multiple running rows), but verify the cleanup still works if the
    invariant ever degrades — defence in depth."""
    a = await store.create_debate("orphan")
    # Force a second running row by bypassing create_debate (simulate db
    # corruption). Since the unique index would block this, we drop it
    # first and reinsert.
    await store.db.execute("DROP INDEX IF EXISTS one_running")
    await store.db.execute(
        "INSERT INTO debates (id, prompt, status) VALUES ('zzz', 'fake', 'running')"
    )
    await store.db.commit()
    n = await store.cleanup_orphans()
    assert n == 2
    assert (await store.get_debate(a)).status == "failed"
