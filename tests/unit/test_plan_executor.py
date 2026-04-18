"""Unit tests for PlanExecutor.

The A2A client layer is mocked at the `create_a2a_client` / `send_and_get_text`
boundary so no real HTTP traffic is generated. The focus here is the DAG
walking logic, round-robin assignment, and error propagation.
"""

from __future__ import annotations

import pytest

from agents.orchestrator import plan_executor as pe_module
from agents.orchestrator.plan_executor import (
    PlanExecutionError,
    PlanExecutor,
    ProgressCallback,
    _build_subtask_prompt,
)
from common.models import SubTask, TaskPlan


pytestmark = pytest.mark.asyncio


# ── _build_subtask_prompt ────────────────────────────────────────────────────


class TestBuildSubtaskPrompt:
    def test_no_dependencies(self):
        t = SubTask(id="t1", description="Do X", required_skill="debate")
        prompt = _build_subtask_prompt(t, {}, "Goal G")
        assert "Goal: Goal G" in prompt
        assert "Task: Do X" in prompt
        assert "Context" not in prompt

    def test_with_perspective(self):
        t = SubTask(
            id="t1", description="Do X", required_skill="debate", perspective="pro"
        )
        prompt = _build_subtask_prompt(t, {}, "G")
        assert "Perspective: pro" in prompt

    def test_with_dependencies(self):
        t = SubTask(
            id="t2",
            description="Respond",
            required_skill="debate",
            depends_on=["t1"],
        )
        prompt = _build_subtask_prompt(t, {"t1": "prior output"}, "G")
        assert "[t1]" in prompt
        assert "prior output" in prompt


# ── _assign_workers (round-robin) ────────────────────────────────────────────


class CollectingProgress(ProgressCallback):
    """Progress sink that records every event for assertions."""

    def __init__(self):
        self.events: list[tuple[str, str, dict | None]] = []

    async def on_progress(self, stage, message, data=None):
        self.events.append((stage, message, data))


class TestAssignWorkers:
    async def test_round_robin_debate(self, registry_with_workers):
        """Two parallel 'debate' tasks land on ae1 and ae2 (not both on ae1)."""
        executor = PlanExecutor(registry=registry_with_workers)
        tasks = [
            SubTask(id="t2", description="pro", required_skill="debate"),
            SubTask(id="t3", description="con", required_skill="debate"),
        ]
        assigned = await executor._assign_workers(tasks)
        assert {assigned["t2"].agent_id, assigned["t3"].agent_id} == {"ae1", "ae2"}

    async def test_round_robin_wraps(self, registry_with_workers):
        """Three parallel debate tasks with 2 workers → wrap-around to first."""
        executor = PlanExecutor(registry=registry_with_workers)
        tasks = [
            SubTask(id=f"t{i}", description="x", required_skill="debate")
            for i in range(3)
        ]
        assigned = await executor._assign_workers(tasks)
        ids = [assigned[t.id].agent_id for t in tasks]
        # First two map 1:1 to ae1/ae2; third wraps back to whichever workers[0] is
        assert len({ids[0], ids[1]}) == 2
        assert ids[2] in {"ae1", "ae2"}

    async def test_missing_skill_raises(self, registry_with_workers):
        executor = PlanExecutor(registry=registry_with_workers)
        with pytest.raises(PlanExecutionError, match="no_such_skill"):
            await executor._assign_workers(
                [SubTask(id="t1", description="x", required_skill="no_such_skill")]
            )


# ── End-to-end DAG execution (mocked A2A layer) ──────────────────────────────


class FakeClient:
    async def close(self):
        pass


def make_mock_a2a(responses: dict[str, str], monkeypatch):
    """Patch create_a2a_client + send_and_get_text to serve canned responses.

    `responses` maps worker URL → response text.
    """
    received_prompts: list[tuple[str, str]] = []  # (url, prompt)

    async def fake_create(url):
        client = FakeClient()
        client._url = url
        return client, object()

    async def fake_send(client, prompt, **kwargs):
        url = client._url
        received_prompts.append((url, prompt))
        return responses[url]

    monkeypatch.setattr(pe_module, "create_a2a_client", fake_create)
    monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)
    return received_prompts


class TestExecute:
    async def test_happy_dag(
        self, registry_with_workers, sample_debate_plan, monkeypatch
    ):
        """Full 4-subtask DAG executes with canned responses, respecting deps."""
        responses = {
            "http://localhost:9001": "NORMALIZED",
            "http://localhost:9002": "PRO_ARG",
            "http://localhost:9003": "CON_ARG",
            "http://localhost:9004": "VERDICT",
        }
        prompts = make_mock_a2a(responses, monkeypatch)

        progress = CollectingProgress()
        executor = PlanExecutor(registry=registry_with_workers, progress=progress)
        results = await executor.execute(sample_debate_plan)

        assert results == {
            "t1": "NORMALIZED",
            "t2": "PRO_ARG",
            "t3": "CON_ARG",
            "t4": "VERDICT",
        }

        # t4's prompt must carry both t2 and t3 outputs (dep propagation)
        feedback_prompt = next(p for url, p in prompts if url == "http://localhost:9004")
        assert "PRO_ARG" in feedback_prompt
        assert "CON_ARG" in feedback_prompt

        # Expected progress stages fired
        stages = [e[0] for e in progress.events]
        assert "plan_start" in stages
        assert "plan_complete" in stages
        assert stages.count("subtask_done") == 4

    async def test_subtask_error_propagates(
        self, registry_with_workers, sample_debate_plan, monkeypatch
    ):
        async def fake_create(url):
            return FakeClient(), object()

        async def fake_send(client, prompt, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(pe_module, "create_a2a_client", fake_create)
        monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

        executor = PlanExecutor(registry=registry_with_workers)
        with pytest.raises(PlanExecutionError, match="boom"):
            await executor.execute(sample_debate_plan)

    async def test_cycle_detected(self, registry_with_workers, monkeypatch):
        """A DAG with a cycle raises PlanExecutionError ('deadlock')."""
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

        async def fake_create(url):
            return FakeClient(), object()

        async def fake_send(client, prompt, **kwargs):
            return "x"

        monkeypatch.setattr(pe_module, "create_a2a_client", fake_create)
        monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

        executor = PlanExecutor(registry=registry_with_workers)
        with pytest.raises(PlanExecutionError, match="deadlock"):
            await executor.execute(plan)
