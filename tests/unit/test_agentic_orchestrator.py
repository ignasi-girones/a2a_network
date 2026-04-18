"""Unit tests for AgenticOrchestrator.

Focus: the pieces that don't require real workers — peak-demand calculation,
capacity gap detection, sink-based synthesis, and the end-to-end orchestration
logic with everything mocked.
"""

from __future__ import annotations

import pytest

from agents.orchestrator import agentic_orchestrator as ao_module
from agents.orchestrator import plan_executor as pe_module
from agents.orchestrator.agent_registry import AgentRegistry
from agents.orchestrator.agentic_orchestrator import (
    AgenticOrchestrator,
    _peak_concurrent_demand,
)
from agents.orchestrator.plan_executor import ProgressCallback
from common.models import SubTask, TaskPlan, WorkerEntry


pytestmark = pytest.mark.asyncio


class CollectingProgress(ProgressCallback):
    def __init__(self):
        self.events: list[tuple[str, str, dict | None]] = []

    async def on_progress(self, stage, message, data=None):
        self.events.append((stage, message, data))


# ── _peak_concurrent_demand ──────────────────────────────────────────────────


class TestPeakConcurrentDemand:
    def test_debate_dag(self, sample_debate_plan):
        peak = _peak_concurrent_demand(sample_debate_plan)
        assert peak == {
            "normalize_input": 1,
            "debate": 2,
            "format_verdict": 1,
        }

    def test_single_task(self):
        plan = TaskPlan(
            goal="x",
            subtasks=[SubTask(id="t1", description="a", required_skill="s")],
        )
        assert _peak_concurrent_demand(plan) == {"s": 1}

    def test_fully_parallel(self):
        """3 independent debate tasks → peak=3."""
        plan = TaskPlan(
            goal="x",
            subtasks=[
                SubTask(id=f"t{i}", description="a", required_skill="debate")
                for i in range(3)
            ],
        )
        assert _peak_concurrent_demand(plan) == {"debate": 3}

    def test_sequential_same_skill(self):
        """3 sequential debate tasks → peak=1 (no concurrency)."""
        plan = TaskPlan(
            goal="x",
            subtasks=[
                SubTask(id="t1", description="a", required_skill="debate"),
                SubTask(
                    id="t2",
                    description="a",
                    required_skill="debate",
                    depends_on=["t1"],
                ),
                SubTask(
                    id="t3",
                    description="a",
                    required_skill="debate",
                    depends_on=["t2"],
                ),
            ],
        )
        assert _peak_concurrent_demand(plan) == {"debate": 1}

    def test_deadlock_returns_partial(self):
        """A cyclic DAG causes the loop to bail early; we get the partial peak."""
        plan = TaskPlan(
            goal="x",
            subtasks=[
                SubTask(
                    id="t1",
                    description="a",
                    required_skill="debate",
                    depends_on=["t2"],
                ),
                SubTask(
                    id="t2",
                    description="b",
                    required_skill="debate",
                    depends_on=["t1"],
                ),
            ],
        )
        # Returns empty since no task is ever ready
        assert _peak_concurrent_demand(plan) == {}


# ── _ensure_capacity ─────────────────────────────────────────────────────────


class FakeSpawner:
    """A WorkerSpawner stub that records spawn calls and registers fake workers."""

    def __init__(self, registry: AgentRegistry, auto_register_skill: str = "debate"):
        self.registry = registry
        self.auto_register_skill = auto_register_skill
        self.spawned: list[str] = []
        self.torn_down: list[str] = []

    async def spawn(self, agent_id: str, **_kwargs):
        self.spawned.append(agent_id)
        await self.registry.register(
            WorkerEntry(
                agent_id=agent_id,
                url=f"http://localhost/{agent_id}",
                card={
                    "skills": [
                        {"id": self.auto_register_skill, "name": "x", "tags": []}
                    ]
                },
            )
        )
        # The orchestrator ignores the return value; a SimpleNamespace is fine.
        from types import SimpleNamespace
        return SimpleNamespace(agent_id=agent_id)

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

    async def test_spawns_difference(self, fresh_registry):
        """Plan needs 3 debate workers, registry has 1 → spawn 2."""
        await fresh_registry.register(
            WorkerEntry(
                agent_id="ae1",
                url="http://ae1",
                card={"skills": [{"id": "debate"}]},
            )
        )

        plan = TaskPlan(
            goal="x",
            subtasks=[
                SubTask(id=f"t{i}", description="a", required_skill="debate")
                for i in range(3)
            ],
        )
        spawner = FakeSpawner(fresh_registry)
        ao = AgenticOrchestrator(registry=fresh_registry, spawner=spawner)
        await ao._ensure_capacity(plan)

        assert len(spawner.spawned) == 2
        # Post-spawn, registry should have 3 debate workers
        debate_workers = await fresh_registry.find_by_skill("debate")
        assert len(debate_workers) == 3


# ── synthesize + sink detection ──────────────────────────────────────────────


class TestSynthesize:
    async def test_single_sink_returns_output_directly(
        self, fresh_registry, sample_debate_plan
    ):
        """sample_debate_plan has t4 as single sink → its output becomes verdict."""
        spawner = FakeSpawner(fresh_registry)
        ao = AgenticOrchestrator(registry=fresh_registry, spawner=spawner)
        results = {
            "t1": "norm",
            "t2": "pro",
            "t3": "con",
            "t4": "FINAL VERDICT",
        }
        verdict = await ao._synthesize("user", sample_debate_plan, results)
        assert verdict == "FINAL VERDICT"

    async def test_multiple_sinks_invokes_llm(
        self, fresh_registry, monkeypatch
    ):
        """Plan with 2 sinks → must call llm_complete to combine them."""
        plan = TaskPlan(
            goal="x",
            subtasks=[
                SubTask(id="t1", description="a", required_skill="debate"),
                SubTask(id="t2", description="b", required_skill="debate"),
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
        self, registry_with_workers, sample_debate_plan, monkeypatch
    ):
        """run() goes: plan (mocked) → execute (mocked A2A) → synthesize (sink)."""
        # 1. Stub the Planner LLM call
        async def fake_llm_complete(**kwargs):
            # The planner asks for JSON; return our sample plan as JSON.
            return sample_debate_plan.model_dump_json()

        # The Planner LLM is imported into planner_module; patch THAT binding.
        from agents.orchestrator import planner as planner_module
        monkeypatch.setattr(planner_module, "llm_complete", fake_llm_complete)

        # 2. Stub the A2A dispatch layer used by PlanExecutor.
        class FakeClient:
            def __init__(self, url):
                self.url = url

            async def close(self):
                pass

        async def fake_create(url):
            return FakeClient(url), object()

        response_by_url = {
            "http://localhost:9001": "NORM",
            "http://localhost:9002": "PRO",
            "http://localhost:9003": "CON",
            "http://localhost:9004": "FINAL",
        }

        async def fake_send(client, prompt, **kwargs):
            return response_by_url[client.url]

        monkeypatch.setattr(pe_module, "create_a2a_client", fake_create)
        monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

        # 3. No spawn should happen (peak demand 2 = available)
        spawner = FakeSpawner(registry_with_workers)
        progress = CollectingProgress()
        ao = AgenticOrchestrator(
            registry=registry_with_workers,
            spawner=spawner,
            progress=progress,
        )

        verdict = await ao.run("user input")

        assert verdict == "FINAL"
        assert spawner.spawned == []
        stages = [e[0] for e in progress.events]
        assert "plan_ready" in stages
        assert "plan_complete" in stages

    async def test_run_triggers_spawn_and_teardown(
        self, fresh_registry, monkeypatch
    ):
        """Plan demanding 3 parallel debates with only 1 debate worker → spawn 2, teardown after."""
        # Seed one debate worker + normalizer + feedback so synthesis finds a sink
        await fresh_registry.register(
            WorkerEntry(
                agent_id="ae1",
                url="http://localhost:9002",
                card={"skills": [{"id": "debate"}]},
            )
        )
        await fresh_registry.register(
            WorkerEntry(
                agent_id="normalizer",
                url="http://localhost:9001",
                card={"skills": [{"id": "normalize_input"}]},
            )
        )
        await fresh_registry.register(
            WorkerEntry(
                agent_id="feedback",
                url="http://localhost:9004",
                card={"skills": [{"id": "format_verdict"}]},
            )
        )

        # Plan with 3 parallel debate subtasks (t2/t3/t4) after t1, then t5 feedback
        plan_json = TaskPlan(
            goal="g",
            subtasks=[
                SubTask(id="t1", description="norm", required_skill="normalize_input"),
                SubTask(
                    id="t2", description="a", required_skill="debate",
                    depends_on=["t1"],
                ),
                SubTask(
                    id="t3", description="b", required_skill="debate",
                    depends_on=["t1"],
                ),
                SubTask(
                    id="t4", description="c", required_skill="debate",
                    depends_on=["t1"],
                ),
                SubTask(
                    id="t5", description="v", required_skill="format_verdict",
                    depends_on=["t2", "t3", "t4"],
                ),
            ],
        ).model_dump_json()

        async def fake_llm_complete(**kwargs):
            return plan_json

        from agents.orchestrator import planner as planner_module
        monkeypatch.setattr(planner_module, "llm_complete", fake_llm_complete)

        class FakeClient:
            def __init__(self, url):
                self.url = url

            async def close(self):
                pass

        async def fake_create(url):
            return FakeClient(url), object()

        async def fake_send(client, prompt, **kwargs):
            return f"out-from-{client.url}"

        monkeypatch.setattr(pe_module, "create_a2a_client", fake_create)
        monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

        spawner = FakeSpawner(fresh_registry, auto_register_skill="debate")
        ao = AgenticOrchestrator(registry=fresh_registry, spawner=spawner)

        verdict = await ao.run("user")

        # Should have spawned 2 extra debate workers
        assert len(spawner.spawned) == 2
        # And torn them down at the end
        assert set(spawner.torn_down) == set(spawner.spawned)
        # Verdict comes from the t5 sink
        assert verdict.startswith("out-from-")
