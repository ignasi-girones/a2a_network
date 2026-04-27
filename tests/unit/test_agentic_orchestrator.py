"""Unit tests for AgenticOrchestrator (Phase 3).

Focus: peak-demand calculation under the canonical quartet, role-aware
capacity gap detection, sink-based synthesis (the synthesizer is always
the sink), and the end-to-end orchestration logic with everything mocked.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agents.orchestrator import agentic_orchestrator as ao_module
from agents.orchestrator import plan_executor as pe_module
from agents.orchestrator.agent_registry import AgentRegistry
from agents.orchestrator.agentic_orchestrator import (
    AgenticOrchestrator,
    _peak_concurrent_demand,
    _role_from_skill,
)
from agents.orchestrator.plan_executor import ProgressCallback
from common.models import CANONICAL_ROLES, SubTask, TaskPlan, WorkerEntry


pytestmark = pytest.mark.asyncio


class CollectingProgress(ProgressCallback):
    def __init__(self):
        self.events: list[tuple[str, str, dict | None]] = []

    async def on_progress(self, stage, message, data=None):
        self.events.append((stage, message, data))


def _patch_httpx_configure_ok(monkeypatch):
    """Stub out the /internal/configure POST so it always succeeds."""

    class _DummyResponse:
        def raise_for_status(self):
            pass

    class _DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, _url, json=None):
            return _DummyResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _DummyClient)


# ── _role_from_skill ─────────────────────────────────────────────────────────


class TestRoleFromSkill:
    def test_recognises_canonical_roles(self):
        for r in CANONICAL_ROLES:
            assert _role_from_skill(f"role_{r}") == r

    def test_unknown_role_prefix_returns_none(self):
        assert _role_from_skill("role_hallucinated") is None

    def test_non_role_skill_returns_none(self):
        assert _role_from_skill("debate") is None
        assert _role_from_skill("") is None


# ── _peak_concurrent_demand ──────────────────────────────────────────────────


class TestPeakConcurrentDemand:
    def test_canonical_quartet(self, sample_debate_plan):
        """Each role is dispatched once → peak is 1 per role-skill."""
        peak = _peak_concurrent_demand(sample_debate_plan)
        assert peak == {f"role_{r}": 1 for r in CANONICAL_ROLES}

    def test_single_task(self):
        plan = TaskPlan(
            goal="x",
            subtasks=[
                SubTask(
                    id="t1",
                    description="a",
                    role_id="analyst",
                    required_skill="role_analyst",
                )
            ],
        )
        assert _peak_concurrent_demand(plan) == {"role_analyst": 1}

    def test_fully_parallel_same_role(self):
        """3 independent devils_advocate tasks (DRTAG-style) → peak=3."""
        plan = TaskPlan(
            goal="x",
            subtasks=[
                SubTask(
                    id=f"t{i}",
                    description="a",
                    role_id="devils_advocate",
                    required_skill="role_devils_advocate",
                )
                for i in range(3)
            ],
        )
        assert _peak_concurrent_demand(plan) == {"role_devils_advocate": 3}

    def test_deadlock_returns_partial(self):
        """A cyclic DAG causes the loop to bail early; we get the partial peak."""
        plan = TaskPlan(
            goal="x",
            subtasks=[
                SubTask(
                    id="t1",
                    description="a",
                    role_id="analyst",
                    required_skill="role_analyst",
                    depends_on=["t2"],
                ),
                SubTask(
                    id="t2",
                    description="b",
                    role_id="seeker",
                    required_skill="role_seeker",
                    depends_on=["t1"],
                ),
            ],
        )
        assert _peak_concurrent_demand(plan) == {}


# ── _ensure_capacity ─────────────────────────────────────────────────────────


class FakeSpawner:
    """A WorkerSpawner stub that records spawn calls and registers fake workers."""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.spawned: list[tuple[str, str]] = []  # (agent_id, role)
        self.torn_down: list[str] = []

    async def spawn(self, agent_id: str, role: str | None = None, **_kwargs):
        # In real usage the WorkerSpawner expects a role; the FakeSpawner
        # mirrors that contract.
        assert role is not None, "Phase 3 spawn must include a role"
        self.spawned.append((agent_id, role))
        await self.registry.register(
            WorkerEntry(
                agent_id=agent_id,
                url=f"http://localhost/{agent_id}",
                card={
                    "skills": [
                        {"id": f"role_{role}", "name": role.title(), "tags": []}
                    ]
                },
            )
        )
        from types import SimpleNamespace
        return SimpleNamespace(agent_id=agent_id, role=role)

    async def teardown(self, agent_id: str):
        self.torn_down.append(agent_id)
        return await self.registry.deregister(agent_id)


class TestEnsureCapacity:
    async def test_no_spawn_when_supply_sufficient(
        self, registry_with_workers, sample_debate_plan
    ):
        spawner = FakeSpawner(registry_with_workers)
        ao = AgenticOrchestrator(
            registry=registry_with_workers, spawner=spawner
        )
        await ao._ensure_capacity(sample_debate_plan)
        assert spawner.spawned == []

    async def test_spawns_missing_roles_with_correct_role(
        self, fresh_registry, sample_debate_plan
    ):
        """No workers registered → spawn 4, one per canonical role."""
        spawner = FakeSpawner(fresh_registry)
        ao = AgenticOrchestrator(registry=fresh_registry, spawner=spawner)
        await ao._ensure_capacity(sample_debate_plan)

        assert len(spawner.spawned) == len(CANONICAL_ROLES)
        spawned_roles = {role for _aid, role in spawner.spawned}
        assert spawned_roles == set(CANONICAL_ROLES)

        # Post-spawn, the registry advertises one worker per role.
        for role in CANONICAL_ROLES:
            workers = await fresh_registry.find_by_skill(f"role_{role}")
            assert len(workers) == 1

    async def test_partial_coverage_only_spawns_missing(
        self, fresh_registry, sample_debate_plan
    ):
        """Half the workers exist → only the missing roles get spawned."""
        for role in ["analyst", "seeker"]:
            await fresh_registry.register(
                WorkerEntry(
                    agent_id=role,
                    url=f"http://localhost/{role}",
                    card={"skills": [{"id": f"role_{role}"}]},
                )
            )

        spawner = FakeSpawner(fresh_registry)
        ao = AgenticOrchestrator(registry=fresh_registry, spawner=spawner)
        await ao._ensure_capacity(sample_debate_plan)

        spawned_roles = {role for _aid, role in spawner.spawned}
        assert spawned_roles == {"devils_advocate", "empiricist", "pragmatist", "synthesizer"}


# ── synthesize + sink detection ──────────────────────────────────────────────


class TestSynthesize:
    async def test_synthesizer_is_the_sink(
        self, fresh_registry, sample_debate_plan
    ):
        """The canonical sextet has t6 (synthesizer) as the unique sink."""
        spawner = FakeSpawner(fresh_registry)
        ao = AgenticOrchestrator(registry=fresh_registry, spawner=spawner)
        results = {
            "t1": "factual baseline",
            "t2": "external evidence",
            "t3": "adversarial critique",
            "t4": "empirical challenge",
            "t5": "practical cases",
            "t6": "FINAL VERDICT",
        }
        verdict = await ao._synthesize("user", sample_debate_plan, results)
        assert verdict == "FINAL VERDICT"

    async def test_multiple_sinks_invokes_llm(
        self, fresh_registry, monkeypatch
    ):
        """An ad-hoc plan with 2 sinks → calls llm_complete to combine them."""
        plan = TaskPlan(
            goal="x",
            subtasks=[
                SubTask(
                    id="t1",
                    description="a",
                    role_id="analyst",
                    required_skill="role_analyst",
                ),
                SubTask(
                    id="t2",
                    description="b",
                    role_id="seeker",
                    required_skill="role_seeker",
                ),
            ],
        )

        async def fake_complete(**kwargs):
            return "SYNTHESIZED"

        monkeypatch.setattr(ao_module, "llm_complete", fake_complete)

        spawner = FakeSpawner(fresh_registry)
        ao = AgenticOrchestrator(registry=fresh_registry, spawner=spawner)
        verdict = await ao._synthesize("user", plan, {"t1": "A", "t2": "B"})
        assert verdict == "SYNTHESIZED"


# ── Full run() with mocks ────────────────────────────────────────────────────


class TestRun:
    async def test_end_to_end_no_spawn(
        self, registry_with_workers, monkeypatch
    ):
        """run() goes: plan (mocked LLM) → execute (mocked A2A) → synthesize."""
        # 1. Stub the Planner LLM call — returns goal+claim JSON
        async def fake_llm_complete(**kwargs):
            return json.dumps(
                {"goal": "Microservicios o monolito", "claim": "Microservicios > monolito"}
            )

        from agents.orchestrator import planner as planner_module
        monkeypatch.setattr(planner_module, "llm_complete", fake_llm_complete)

        # 2. Stub the worker /internal/configure POST.
        _patch_httpx_configure_ok(monkeypatch)

        # 3. Stub the A2A dispatch layer used by PlanExecutor.
        class FakeClient:
            def __init__(self, url):
                self.url = url

            async def close(self):
                pass

        async def fake_create(url):
            return FakeClient(url), object()

        # registry_with_workers maps roles to specific URLs (see conftest).
        url_to_output = {
            "http://localhost:9002": "ANALYST",       # analyst
            "http://localhost:9003": "SEEKER",        # seeker
            "http://localhost:9087": "DEVILS",        # devils_advocate
            "http://localhost:9089": "EMPIRICIST",    # empiricist
            "http://localhost:9090": "PRAGMATIST",    # pragmatist
            "http://localhost:9088": "SYNTHESIS",     # synthesizer (sink)
        }

        async def fake_send(client, prompt, **kwargs):
            return url_to_output[client.url]

        monkeypatch.setattr(pe_module, "create_a2a_client", fake_create)
        monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

        spawner = FakeSpawner(registry_with_workers)
        progress = CollectingProgress()
        ao = AgenticOrchestrator(
            registry=registry_with_workers,
            spawner=spawner,
            progress=progress,
        )

        verdict = await ao.run("Microservicios vs monolito")

        assert verdict == "SYNTHESIS"
        assert spawner.spawned == []
        stages = [e[0] for e in progress.events]
        assert "plan_ready" in stages
        # Phase 4: plan_complete is replaced by deliberation_complete
        assert "deliberation_start" in stages
        assert "deliberation_complete" in stages

    async def test_run_triggers_spawn_and_teardown_when_role_missing(
        self, fresh_registry, monkeypatch
    ):
        """No workers registered → spawn 4 (one per role), teardown after run."""

        async def fake_llm_complete(**kwargs):
            return json.dumps({"goal": "g", "claim": "c"})

        from agents.orchestrator import planner as planner_module
        monkeypatch.setattr(planner_module, "llm_complete", fake_llm_complete)

        _patch_httpx_configure_ok(monkeypatch)

        class FakeClient:
            def __init__(self, url):
                self.url = url

            async def close(self):
                pass

        async def fake_create(url):
            return FakeClient(url), object()

        async def fake_send(client, prompt, **kwargs):
            # The synthesizer's output is the verdict; mark its url specially.
            return f"out-from-{client.url}"

        monkeypatch.setattr(pe_module, "create_a2a_client", fake_create)
        monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

        spawner = FakeSpawner(fresh_registry)
        ao = AgenticOrchestrator(registry=fresh_registry, spawner=spawner)

        verdict = await ao.run("user")

        # Should have spawned 6 workers (one per role)
        assert len(spawner.spawned) == len(CANONICAL_ROLES)
        spawned_roles = {role for _aid, role in spawner.spawned}
        assert spawned_roles == set(CANONICAL_ROLES)
        # And torn them down at the end
        spawned_ids = {aid for aid, _r in spawner.spawned}
        assert set(spawner.torn_down) == spawned_ids
        # Verdict comes from the synthesizer sink
        assert verdict.startswith("out-from-")
