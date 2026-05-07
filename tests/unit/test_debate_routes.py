"""Tests for the /debates HTTP routes.

Uses Starlette's TestClient (sync) and async fixtures for the store. The
`POST /debates` background task runs `AgenticOrchestrator.run`, which we
do NOT exercise here (covered by integration tests elsewhere) — this
suite mocks the run by invoking the store directly to assert that the
HTTP layer matches expectations: shape, status codes, dedup, replay.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from agents.orchestrator.debate_routes import debate_routes
from agents.orchestrator.debate_store import DebateStore


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


@pytest.fixture
def app(store):
    """Bare Starlette app with the debate routes mounted + store on state."""
    app = Starlette(routes=debate_routes)
    app.state.debate_store = store
    return app


# ── GET /debates (list) ──────────────────────────────────────────────────


def test_list_debates_empty(app):
    with TestClient(app) as client:
        r = client.get("/debates")
        assert r.status_code == 200
        assert r.json() == {"count": 0, "debates": []}


async def test_list_debates_after_create(store, app):
    a = await store.create_debate("primero")
    await store.mark_completed(a, "verdict A")
    b = await store.create_debate("segundo")
    with TestClient(app) as client:
        r = client.get("/debates")
        body = r.json()
        assert body["count"] == 2
        ids = [d["id"] for d in body["debates"]]
        assert ids == [b, a]  # newest first
        assert body["debates"][0]["status"] == "running"
        assert body["debates"][1]["status"] == "completed"


# ── GET /debates/active ──────────────────────────────────────────────────


def test_active_returns_null_when_none(app):
    with TestClient(app) as client:
        r = client.get("/debates/active")
        assert r.status_code == 200
        assert r.json() is None


async def test_active_returns_running_row(store, app):
    debate_id = await store.create_debate("activo")
    with TestClient(app) as client:
        r = client.get("/debates/active")
        assert r.status_code == 200
        body = r.json()
        assert body is not None
        assert body["id"] == debate_id
        assert body["status"] == "running"


# ── GET /debates/<id> ────────────────────────────────────────────────────


def test_get_debate_404(app):
    with TestClient(app) as client:
        r = client.get("/debates/nonexistent")
        assert r.status_code == 404
        assert r.json() == {"error": "not_found"}


async def test_get_debate_returns_metadata(store, app):
    debate_id = await store.create_debate("test")
    with TestClient(app) as client:
        r = client.get(f"/debates/{debate_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == debate_id
        assert body["prompt"] == "test"


# ── POST /debates ────────────────────────────────────────────────────────


def test_post_debates_rejects_missing_prompt(app):
    with TestClient(app) as client:
        r = client.post("/debates", json={})
        assert r.status_code == 400
        assert r.json()["error"] == "missing_prompt"


def test_post_debates_rejects_invalid_json(app):
    with TestClient(app) as client:
        r = client.post("/debates", content="not json")
        assert r.status_code == 400


async def test_post_debates_returns_409_when_active_exists(store, app):
    """When a debate is already running, POST must reject with 409.

    We don't actually run the background task — we just simulate "there's
    already a running debate" by directly inserting one via the store.
    """
    await store.create_debate("first one")
    with TestClient(app) as client:
        r = client.post("/debates", json={"prompt": "second one"})
        assert r.status_code == 409
        body = r.json()
        assert body["error"] == "debate_active"
        assert body["active"] is not None
        assert body["active"]["prompt"] == "first one"


# ── GET /debates/<id>/events ─────────────────────────────────────────────


async def test_events_404_for_unknown_debate(app):
    with TestClient(app) as client:
        r = client.get("/debates/missing/events")
        assert r.status_code == 404


async def test_events_returns_full_replay(store, app):
    debate_id = await store.create_debate("test")
    for stage in ("a", "b", "c"):
        await store.record_event(debate_id, stage, None, {"s": stage})
    with TestClient(app) as client:
        r = client.get(f"/debates/{debate_id}/events")
        body = r.json()
        assert body["count"] == 3
        assert [e["stage"] for e in body["events"]] == ["a", "b", "c"]
        assert [e["seq"] for e in body["events"]] == [0, 1, 2]
        assert body["events"][0]["data"] == {"s": "a"}


async def test_events_since_filter(store, app):
    debate_id = await store.create_debate("test")
    for i, stage in enumerate(("a", "b", "c", "d", "e")):
        await store.record_event(debate_id, stage, None, None)
    with TestClient(app) as client:
        r = client.get(f"/debates/{debate_id}/events?since=2")
        body = r.json()
        assert [e["stage"] for e in body["events"]] == ["d", "e"]
        assert [e["seq"] for e in body["events"]] == [3, 4]


# ── GET /debates/<id>/stream ─────────────────────────────────────────────


async def test_stream_404_for_unknown_debate(app):
    with TestClient(app) as client:
        r = client.get("/debates/missing/stream")
        assert r.status_code == 404


def _parse_sse(body_text: str) -> list[dict]:
    """Parse SSE response body into a list of payload dicts."""
    out: list[dict] = []
    for chunk in body_text.split("\n\n"):
        for line in chunk.split("\n"):
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload and payload != "":
                    try:
                        out.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass
    return out


async def test_stream_emits_catchup_then_terminal(store, app):
    """Pre-populate a completed debate, then GET /stream — every event
    must be emitted (including the synthetic 'verdict') and the response
    closes cleanly afterwards."""
    debate_id = await store.create_debate("test")
    await store.record_event(debate_id, "discover", "msg", None)
    await store.record_event(debate_id, "plan_ready", "msg", {"n": 3})
    await store.mark_completed(debate_id, "veredicto final")

    with TestClient(app) as client:
        with client.stream("GET", f"/debates/{debate_id}/stream") as resp:
            body = b"".join(resp.iter_bytes()).decode("utf-8")

    events = _parse_sse(body)
    stages = [e["stage"] for e in events]
    # discover, plan_ready, verdict (synthetic) — possibly duplicated
    # if the terminal-loop emits the synthetic event a second time, dedup:
    assert "discover" in stages
    assert "plan_ready" in stages
    assert stages[-1] == "verdict" or "verdict" in stages
    # Verify verdict payload
    verdict_evs = [e for e in events if e["stage"] == "verdict"]
    assert verdict_evs[-1]["data"] == {"text": "veredicto final"}


async def test_stream_with_since_skips_old_events(store, app):
    """Stream from the middle of an already-terminal debate."""
    debate_id = await store.create_debate("test")
    await store.record_event(debate_id, "a", None, None)  # seq 0
    await store.record_event(debate_id, "b", None, None)  # seq 1
    await store.record_event(debate_id, "c", None, None)  # seq 2
    await store.mark_failed(debate_id, "boom")  # seq 3 (synthetic 'failed')

    with TestClient(app) as client:
        with client.stream("GET", f"/debates/{debate_id}/stream?since=1") as resp:
            body = b"".join(resp.iter_bytes()).decode("utf-8")

    events = _parse_sse(body)
    # Should NOT include 'a' (seq 0) or 'b' (seq 1)
    stages = [e["stage"] for e in events]
    assert "a" not in stages
    assert "b" not in stages
    assert "c" in stages
    assert "failed" in stages
