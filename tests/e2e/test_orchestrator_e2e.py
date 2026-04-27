"""End-to-end tests that exercise the orchestrator's full HTTP surface in-process.

Strategy: build a Starlette app mirroring the production `agents.orchestrator`
layout (registry routes + planner routes), then drive it through Starlette's
TestClient. The externals the orchestrator would call out to — the LLM and the
A2A worker dispatch — are mocked at their module boundaries.

What this covers:
- Registry HTTP: register, list, deregister, by-skill
- Planner HTTP: POST /orchestrator/plan returns a plan
- Plan Executor HTTP: POST /orchestrator/plan/execute runs the DAG
- Agentic run HTTP: POST /orchestrator/agentic/run runs Planner → capacity →
  PlanExecutor → synthesize and returns the verdict

What this does NOT cover (intentionally, to stay CI-friendly):
- Real LLM providers (Groq, Gemini, etc.)
- Real subprocess spawning of worker processes
- Real A2A SDK streaming

Those are validated by the manual verification steps documented in the
implementation notes.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

import json

from agents.orchestrator import agent_registry as ar_module
from agents.orchestrator import deliberation_loop as dl_module
from agents.orchestrator import plan_executor as pe_module
from agents.orchestrator import planner as planner_module
from agents.orchestrator import planner_routes as pr_module
from agents.orchestrator import worker_spawner as ws_module
from agents.orchestrator.agent_registry import AgentRegistry
from common.models import SubTask, TaskPlan


# Responses the fake workers will return for any A2A call that hits their URL.
# Phase 4: one URL per canonical role worker.
WORKER_RESPONSES = {
    "http://localhost:9081": "Análisis factual del tema.",
    "http://localhost:9082": "Evidencia de búsqueda externa.",
    "http://localhost:9083": "Contraargumento adversarial.",
    "http://localhost:9084": "Crítica empírica.",
    "http://localhost:9085": "Perspectiva pragmática.",
    "http://localhost:9086": "## Final Verdict\nBoth sides have merit.",
}

# Ports used by the 6 role workers in the e2e fixture
_ROLE_PORTS = {
    "analyst": 9081,
    "seeker": 9082,
    "devils_advocate": 9083,
    "empiricist": 9084,
    "pragmatist": 9085,
    "synthesizer": 9086,
}


class FakeClient:
    def __init__(self, url: str):
        self.url = url

    async def close(self):
        pass


@pytest.fixture
def orchestrator_app(monkeypatch) -> Iterator[Starlette]:
    """Build a Starlette app with the same routes as agents.orchestrator."""
    # 1. Fresh registry, patched into every module that uses the singleton.
    registry = AgentRegistry()
    monkeypatch.setattr(ar_module, "registry", registry)
    monkeypatch.setattr(pr_module, "registry", registry)

    # 2. Fresh spawner bound to this registry.
    monkeypatch.setattr(ws_module, "_spawner", None)

    # 3. Assemble the routes exactly as __main__ does.
    app = Starlette(routes=ar_module.registry_routes + pr_module.planner_routes)
    yield app


@pytest.fixture
def client(orchestrator_app) -> Iterator[TestClient]:
    with TestClient(orchestrator_app) as c:
        yield c


@pytest.fixture
def seed_workers(client):
    """Register one worker per canonical role via the HTTP registry endpoint."""
    payloads = [
        {
            "agent_id": role,
            "url": f"http://localhost:{port}",
            "card": {"skills": [{"id": f"role_{role}", "name": role.capitalize(), "tags": []}]},
        }
        for role, port in _ROLE_PORTS.items()
    ]
    for p in payloads:
        r = client.post("/registry/register", json=p)
        assert r.status_code == 200
    return payloads


@pytest.fixture
def mock_llm_and_a2a(monkeypatch):
    """Patch Planner's LLM + PlanExecutor/DeliberationLoop's A2A layer.

    Phase 4 changes:
    - The Planner LLM now only extracts {goal, claim} — the sextet DAG is
      built deterministically. The fake LLM returns a JSON with those fields.
    - The DeliberationLoop dispatches workers sequentially; we patch
      send_and_get_text in plan_executor so every worker returns a canned
      response keyed by URL.
    - We also stub DeliberationLoop.run so the e2e test doesn't have to spin
      up full A2A streaming — it just asserts the HTTP surface works.
    """
    # Planner calls llm_complete expecting {"goal": ..., "claim": ...}
    fake_extraction = json.dumps({"goal": "E2E test", "claim": "X es mejor que Y"})

    async def fake_llm(**kwargs):
        return fake_extraction

    monkeypatch.setattr(planner_module, "llm_complete", fake_llm)

    # Stub A2A helpers used by PlanExecutor / dispatch_one
    async def fake_create_a2a(url):
        return FakeClient(url), object()

    async def fake_send(client, prompt, **kwargs):
        return WORKER_RESPONSES.get(client.url, "Respuesta genérica.")

    monkeypatch.setattr(pe_module, "create_a2a_client", fake_create_a2a)
    monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

    # Stub DeliberationLoop.run so we don't need real A2A streaming in e2e
    async def fake_loop_run(self, *, claim, goal):
        from common.models import DiscussionLedger
        ledger = DiscussionLedger(claim=claim, goal=goal, max_rounds=1)
        verdict = "## Final Verdict\nBoth sides have merit."
        return ledger, verdict

    monkeypatch.setattr(dl_module.DeliberationLoop, "run", fake_loop_run)

    return fake_extraction


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRegistryHTTP:
    def test_initial_registry_empty(self, client):
        r = client.get("/registry/agents")
        assert r.status_code == 200
        assert r.json() == {"count": 0, "workers": []}

    def test_register_and_list(self, client, seed_workers):
        r = client.get("/registry/agents")
        assert r.json()["count"] == 6  # one per canonical role
        ids = {w["agent_id"] for w in r.json()["workers"]}
        assert ids == set(_ROLE_PORTS.keys())

    def test_by_skill(self, client, seed_workers):
        r = client.get("/registry/by-skill/role_analyst")
        assert r.json()["count"] == 1

        r = client.get("/registry/by-skill/nonexistent")
        assert r.json()["count"] == 0

    def test_deregister(self, client, seed_workers):
        r = client.delete("/registry/analyst")
        assert r.status_code == 200
        r = client.get("/registry/agents")
        assert r.json()["count"] == 5


class TestPlannerHTTP:
    def test_preview_plan(self, client, seed_workers, mock_llm_and_a2a):
        r = client.post(
            "/orchestrator/plan",
            json={"input": "Should we adopt X?"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["plan"]["goal"] == "E2E test"
        # Phase 4: canonical sextet (6 subtasks)
        assert len(body["plan"]["subtasks"]) == 6

    def test_preview_rejects_empty_input(self, client):
        r = client.post("/orchestrator/plan", json={"input": ""})
        assert r.status_code == 400

    def test_preview_rejects_bad_json(self, client):
        r = client.post(
            "/orchestrator/plan",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400


class TestPlanExecuteHTTP:
    def test_execute_returns_outputs(self, client, seed_workers, mock_llm_and_a2a):
        """Phase 4: /plan/execute runs the sextet DAG one-shot and returns all 6 results."""
        r = client.post(
            "/orchestrator/plan/execute",
            json={"input": "Test prompt"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        # Sextet: 6 subtasks (t1–t6)
        assert len(body["results"]) == 6
        assert "t1" in body["results"]
        assert "t6" in body["results"]


class TestAgenticRunHTTP:
    def test_full_agentic_flow(self, client, seed_workers, mock_llm_and_a2a):
        r = client.post(
            "/orchestrator/agentic/run",
            json={"input": "Compare A and B"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        # Verdict comes from the stubbed DeliberationLoop
        assert body["verdict"].startswith("## Final Verdict")

        # Phase 4 lifecycle stages
        stages = [e["stage"] for e in body["events"]]
        for required in ("discover", "plan", "plan_ready"):
            assert required in stages, f"Missing stage {required!r} in {stages}"

        # No spawn events: all 6 role workers are pre-seeded
        assert not any(s.startswith("spawn") for s in stages)

    def test_agentic_run_rejects_empty_input(self, client, seed_workers):
        r = client.post("/orchestrator/agentic/run", json={"input": ""})
        assert r.status_code == 400

    def test_missing_skill_fails_gracefully(
        self, client, mock_llm_and_a2a
    ):
        """No workers registered → agentic run fails with 500 and a clear error."""
        r = client.post(
            "/orchestrator/agentic/run",
            json={"input": "whatever"},
        )
        # The planner catalog is empty so create_plan raises; the endpoint
        # converts exceptions into 500 with a detail message.
        assert r.status_code == 500
        body = r.json()
        assert "detail" in body
