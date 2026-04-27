"""Unit tests for PlanExecutor (Phase 3).

The A2A client layer is mocked at the `create_a2a_client` /
`send_and_get_text` boundary so no real HTTP traffic is generated. The
``/internal/configure`` POST is mocked at the `httpx.AsyncClient`
boundary so persona configuration succeeds without a live worker.
"""

from __future__ import annotations

import httpx
import pytest

from agents.orchestrator import plan_executor as pe_module
from agents.orchestrator.plan_executor import (
    PlanExecutionError,
    PlanExecutor,
    ProgressCallback,
    _build_subtask_prompt,
)
from common.models import SubTask, TaskPlan, WorkerEntry


pytestmark = pytest.mark.asyncio


# ── Helpers ─────────────────────────────────────────────────────────────────


def _patch_httpx_configure_ok(monkeypatch):
    """Stub /internal/configure POSTs so persona configuration succeeds."""

    captured = {"calls": []}

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

        async def post(self, url, json=None):
            captured["calls"].append({"url": url, "payload": json})
            return _DummyResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _DummyClient)
    return captured


# ── _build_subtask_prompt ────────────────────────────────────────────────────


class TestBuildSubtaskPrompt:
    def test_no_dependencies(self):
        t = SubTask(
            id="t1",
            description="Do X",
            role_id="analyst",
            required_skill="role_analyst",
        )
        prompt = _build_subtask_prompt(t, {}, "Goal G")
        assert "Goal: Goal G" in prompt
        assert "Task: Do X" in prompt
        assert "Context" not in prompt

    def test_with_perspective(self):
        t = SubTask(
            id="t1",
            description="Do X",
            role_id="analyst",
            required_skill="role_analyst",
            perspective="pro",
        )
        prompt = _build_subtask_prompt(t, {}, "G")
        assert "Perspective: pro" in prompt

    def test_with_dependencies(self):
        t = SubTask(
            id="t2",
            description="Respond",
            role_id="seeker",
            required_skill="role_seeker",
            depends_on=["t1"],
        )
        prompt = _build_subtask_prompt(t, {"t1": "prior output"}, "G")
        assert "[t1]" in prompt
        assert "prior output" in prompt


# ── _assign_workers (round-robin per skill) ─────────────────────────────────


class CollectingProgress(ProgressCallback):
    """Progress sink that records every event for assertions."""

    def __init__(self):
        self.events: list[tuple[str, str, dict | None]] = []

    async def on_progress(self, stage, message, data=None):
        self.events.append((stage, message, data))


@pytest.fixture
async def registry_with_two_devils_advocates(fresh_registry):
    """Two devils_advocate workers — exercises round-robin per skill."""
    await fresh_registry.register(
        WorkerEntry(
            agent_id="da1",
            url="http://localhost:9087",
            card={"skills": [{"id": "role_devils_advocate"}]},
        )
    )
    await fresh_registry.register(
        WorkerEntry(
            agent_id="da2",
            url="http://localhost:9088",
            card={"skills": [{"id": "role_devils_advocate"}]},
        )
    )
    return fresh_registry


class TestAssignWorkers:
    async def test_round_robin_same_role(self, registry_with_two_devils_advocates):
        """Two parallel devils_advocate tasks land on da1 and da2."""
        executor = PlanExecutor(registry=registry_with_two_devils_advocates)
        tasks = [
            SubTask(
                id="t2",
                description="att1",
                role_id="devils_advocate",
                required_skill="role_devils_advocate",
            ),
            SubTask(
                id="t3",
                description="att2",
                role_id="devils_advocate",
                required_skill="role_devils_advocate",
            ),
        ]
        assigned = await executor._assign_workers(tasks)
        assert {assigned["t2"].agent_id, assigned["t3"].agent_id} == {"da1", "da2"}

    async def test_round_robin_wraps(self, registry_with_two_devils_advocates):
        """Three parallel devils_advocate tasks with 2 workers → wrap-around."""
        executor = PlanExecutor(registry=registry_with_two_devils_advocates)
        tasks = [
            SubTask(
                id=f"t{i}",
                description="x",
                role_id="devils_advocate",
                required_skill="role_devils_advocate",
            )
            for i in range(3)
        ]
        assigned = await executor._assign_workers(tasks)
        ids = [assigned[t.id].agent_id for t in tasks]
        assert len({ids[0], ids[1]}) == 2
        assert ids[2] in {"da1", "da2"}

    async def test_missing_skill_raises(self, registry_with_workers):
        executor = PlanExecutor(registry=registry_with_workers)
        with pytest.raises(PlanExecutionError, match="no_such_skill"):
            await executor._assign_workers(
                [
                    SubTask(
                        id="t1",
                        description="x",
                        required_skill="no_such_skill",
                    )
                ]
            )


# ── End-to-end DAG execution (mocked A2A layer + httpx configure) ──────────


class FakeClient:
    async def close(self):
        pass


def make_mock_a2a(responses: dict[str, str], monkeypatch):
    """Patch create_a2a_client + send_and_get_text to serve canned responses."""
    received_prompts: list[tuple[str, str]] = []

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
    async def test_happy_quartet(
        self, registry_with_workers, sample_debate_plan, monkeypatch
    ):
        """Full 6-subtask sextet executes, respecting deps + persona configure."""
        responses = {
            "http://localhost:9002": "ANALYST_OUT",       # analyst
            "http://localhost:9003": "SEEKER_OUT",        # seeker
            "http://localhost:9087": "DEVILS_OUT",        # devils_advocate
            "http://localhost:9089": "EMPIRICIST_OUT",    # empiricist
            "http://localhost:9090": "PRAGMATIST_OUT",    # pragmatist
            "http://localhost:9088": "VERDICT",           # synthesizer
        }
        prompts = make_mock_a2a(responses, monkeypatch)
        configure_calls = _patch_httpx_configure_ok(monkeypatch)

        progress = CollectingProgress()
        executor = PlanExecutor(registry=registry_with_workers, progress=progress)
        results = await executor.execute(sample_debate_plan)

        assert results == {
            "t1": "ANALYST_OUT",
            "t2": "SEEKER_OUT",
            "t3": "DEVILS_OUT",
            "t4": "EMPIRICIST_OUT",
            "t5": "PRAGMATIST_OUT",
            "t6": "VERDICT",
        }

        # The synthesizer (t6) prompt should carry all upstream outputs.
        synth_prompt = next(
            p for url, p in prompts if url == "http://localhost:9088"
        )
        assert "ANALYST_OUT" in synth_prompt
        assert "SEEKER_OUT" in synth_prompt
        assert "DEVILS_OUT" in synth_prompt

        # Each subtask triggered one /internal/configure POST with its persona.
        assert len(configure_calls["calls"]) == 6
        roles_configured = {
            c["payload"]["persona"]["role_id"] for c in configure_calls["calls"]
        }
        assert roles_configured == {
            "analyst", "seeker", "devils_advocate", "empiricist", "pragmatist", "synthesizer"
        }
        # The claim from the plan flows into every config payload.
        for call in configure_calls["calls"]:
            assert call["payload"]["claim"] == sample_debate_plan.claim

        # Expected progress stages
        stages = [e[0] for e in progress.events]
        assert "plan_start" in stages
        assert "plan_complete" in stages
        assert stages.count("subtask_done") == 6

    async def test_subtask_error_propagates(
        self, registry_with_workers, sample_debate_plan, monkeypatch
    ):
        _patch_httpx_configure_ok(monkeypatch)

        async def fake_create(url):
            return FakeClient(), object()

        async def fake_send(client, prompt, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(pe_module, "create_a2a_client", fake_create)
        monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

        executor = PlanExecutor(registry=registry_with_workers)
        with pytest.raises(PlanExecutionError, match="boom"):
            await executor.execute(sample_debate_plan)

    async def test_configure_failure_aborts_run(
        self, registry_with_workers, sample_debate_plan, monkeypatch
    ):
        """If /internal/configure raises, the plan fails fast (no stale persona)."""

        class _DummyClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def post(self, _url, json=None):
                raise RuntimeError("configure exploded")

        monkeypatch.setattr(httpx, "AsyncClient", _DummyClient)

        async def fake_create(url):
            return FakeClient(), object()

        async def fake_send(client, prompt, **kwargs):
            return "should not reach"

        monkeypatch.setattr(pe_module, "create_a2a_client", fake_create)
        monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

        executor = PlanExecutor(registry=registry_with_workers)
        with pytest.raises(PlanExecutionError, match="configure"):
            await executor.execute(sample_debate_plan)

    async def test_cycle_detected(self, registry_with_workers, monkeypatch):
        """A DAG with a cycle raises PlanExecutionError ('deadlock')."""
        _patch_httpx_configure_ok(monkeypatch)
        plan = TaskPlan(
            goal="x",
            claim="c",
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

        async def fake_create(url):
            return FakeClient(), object()

        async def fake_send(client, prompt, **kwargs):
            return "x"

        monkeypatch.setattr(pe_module, "create_a2a_client", fake_create)
        monkeypatch.setattr(pe_module, "send_and_get_text", fake_send)

        executor = PlanExecutor(registry=registry_with_workers)
        with pytest.raises(PlanExecutionError, match="deadlock"):
            await executor.execute(plan)
