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

from agents.orchestrator import agent_registry as ar_module
from agents.orchestrator import plan_executor as pe_module
from agents.orchestrator import planner as planner_module
from agents.orchestrator import planner_routes as pr_module
from agents.orchestrator import worker_spawner as ws_module
from agents.orchestrator.agent_registry import AgentRegistry
from common.models import SubTask, TaskPlan


# Responses the fake workers will return for any A2A call that hits their URL.
WORKER_RESPONSES = {
    "http://localhost:9001": "NORMALIZED_OUTPUT",
    "http://localhost:9002": "PRO_ARGUMENT",
    "http://localhost:9003": "CON_ARGUMENT",
    "http://localhost:9004": "## Final Verdict\nBoth sides have merit.",
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
    """Register the 4 base workers via the HTTP registry endpoint."""
    payloads = [
        {
            "agent_id": "normalizer",
            "url": "http://localhost:9001",
            "card": {"skills": [{"id": "normalize_input", "name": "N", "tags": []}]},
        },
        {
            "agent_id": "ae1",
            "url": "http://localhost:9002",
            "card": {"skills": [{"id": "debate", "name": "D", "tags": []}]},
        },
        {
            "agent_id": "ae2",
            "url": "http://localhost:9003",
            "card": {"skills": [{"id": "debate", "name": "D", "tags": []}]},
        },
        {
            "agent_id": "feedback",
            "url": "http://localhost:9004",
            "card": {"skills": [{"id": "format_verdict", "name": "F", "tags": []}]},
        },
    ]
    for p in payloads:
        r = client.post("/registry/register", json=p)
        assert r.status_code == 200
    return payloads


@pytest.fixture
def mock_llm_and_a2a(monkeypatch):
    """Patch Planner's LLM + PlanExecutor's A2A layer."""
    canonical_plan = TaskPlan(
        goal="E2E test",
        subtasks=[
            SubTask(id="t1", description="Normalize", required_skill="normalize_input"),
            SubTask(
                id="t2",
                description="Pro",
                required_skill="debate",
                depends_on=["t1"],
                perspective="pro",
            ),
            SubTask(
                id="t3",
                description="Con",
                required_skill="debate",
                depends_on=["t1"],
                perspective="con",
            ),
            SubTask(
                id="t4",
                description="Verdict",
                required_skill="format_verdict",
                depends_on=["t2", "t3"],
            ),
        ],
    ).model_dump_json()

    async def fake_llm(**kwargs):
        return canonical_plan

    monkeypatch.setattr(planner_module, "llm_complete", fake_llm)

    async def fake_create_a2a(url):
        return FakeClient(url), object()

    async def fake_send(client, prompt, **kwargs):
        return WORKER_RESPONSES[client.url]

    monkeypatch.setattr(pe_module, "create_a2a_client", fake_create_a2a)
    monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

    return canonical_plan


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRegistryHTTP:
    def test_initial_registry_empty(self, client):
        r = client.get("/registry/agents")
        assert r.status_code == 200
        assert r.json() == {"count": 0, "workers": []}

    def test_register_and_list(self, client, seed_workers):
        r = client.get("/registry/agents")
        assert r.json()["count"] == 4
        ids = {w["agent_id"] for w in r.json()["workers"]}
        assert ids == {"normalizer", "ae1", "ae2", "feedback"}

    def test_by_skill(self, client, seed_workers):
        r = client.get("/registry/by-skill/debate")
        assert r.json()["count"] == 2

        r = client.get("/registry/by-skill/nonexistent")
        assert r.json()["count"] == 0

    def test_deregister(self, client, seed_workers):
        r = client.delete("/registry/ae1")
        assert r.status_code == 200
        r = client.get("/registry/agents")
        assert r.json()["count"] == 3


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
        assert len(body["plan"]["subtasks"]) == 4

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
        r = client.post(
            "/orchestrator/plan/execute",
            json={"input": "Test prompt"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["results"]["t1"] == "NORMALIZED_OUTPUT"
        assert body["results"]["t2"] == "PRO_ARGUMENT"
        assert body["results"]["t3"] == "CON_ARGUMENT"
        assert body["results"]["t4"].startswith("## Final Verdict")


class TestAgenticRunHTTP:
    def test_full_agentic_flow(self, client, seed_workers, mock_llm_and_a2a):
        r = client.post(
            "/orchestrator/agentic/run",
            json={"input": "Compare A and B"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        # Verdict is the t4 output (single sink)
        assert body["verdict"].startswith("## Final Verdict")

        # Trace contains the expected lifecycle stages
        stages = [e["stage"] for e in body["events"]]
        for required in ("discover", "plan", "plan_ready", "plan_start", "plan_complete", "synthesize"):
            assert required in stages, f"Missing stage {required!r} in {stages}"

        # No spawn events since supply meets demand
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
