"""Unit tests for the telemetry package.

Tests the three decoupled layers:
1. PrometheusMiddleware — HTTP request metrics
2. MetricsProgressCallback — debate lifecycle metrics
3. Function hooks — LLM/MCP wrapping
4. install_telemetry — the single entrypoint
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from agents.orchestrator.plan_executor import ProgressCallback
from common.telemetry.metrics import (
    APORIA_DETECTIONS,
    BELIEF_UPDATES,
    DEBATES_TOTAL,
    DELIBERATION_ROUNDS,
    HTTP_DURATION,
    HTTP_REQUESTS,
    LLM_CALLS,
    MCP_CALLS,
    SUBTASK_DISPATCH,
    WORKER_SPAWNS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _counter_value(counter, **labels) -> float:
    """Read the current value of a Prometheus counter."""
    try:
        return counter.labels(**labels)._value.get()
    except Exception:
        return 0.0


def _gauge_value(gauge, **labels) -> float:
    try:
        return gauge.labels(**labels)._value.get()
    except Exception:
        return 0.0


class _NullProgress(ProgressCallback):
    """No-op progress callback for testing the metrics wrapper."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []

    async def on_progress(self, stage, message, data=None):
        self.calls.append((stage, message, data))


# ── Middleware tests ───────────────────────────────────────────────────────────


class TestPrometheusMiddleware:
    @pytest.fixture
    def app_with_middleware(self):
        def homepage(request):
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/", homepage)])

        from common.telemetry.middleware import PrometheusMiddleware

        app.add_middleware(PrometheusMiddleware, agent_id="test_agent")
        return app

    def test_middleware_increments_counter(self, app_with_middleware):
        before = _counter_value(
            HTTP_REQUESTS,
            agent_id="test_agent",
            method="GET",
            path="/",
            status="200",
        )
        client = TestClient(app_with_middleware)
        client.get("/")
        after = _counter_value(
            HTTP_REQUESTS,
            agent_id="test_agent",
            method="GET",
            path="/",
            status="200",
        )
        assert after == before + 1

    def test_middleware_records_duration(self, app_with_middleware):
        client = TestClient(app_with_middleware)
        client.get("/")
        # Duration histogram should have at least one observation
        sample_count = HTTP_DURATION.labels(
            agent_id="test_agent", method="GET", path="/"
        )._sum.get()
        assert sample_count > 0


class TestPathNormalization:
    def test_registry_id_collapsed(self):
        from common.telemetry.middleware import _normalize_path

        assert _normalize_path("/registry/analyst") == "/registry/{id}"
        assert _normalize_path("/registry/devils_advocate") == "/registry/{id}"

    def test_by_skill_collapsed(self):
        from common.telemetry.middleware import _normalize_path

        assert (
            _normalize_path("/registry/by-skill/role_analyst")
            == "/registry/by-skill/{skill}"
        )

    def test_static_paths_unchanged(self):
        from common.telemetry.middleware import _normalize_path

        assert _normalize_path("/metrics") == "/metrics"
        assert _normalize_path("/orchestrator/plan") == "/orchestrator/plan"
        assert _normalize_path("/registry") == "/registry"


# ── MetricsProgressCallback tests ──────────────────────────────────────────────


class TestMetricsProgressCallback:
    @pytest.fixture
    def setup(self):
        from common.telemetry.progress_metrics import MetricsProgressCallback

        null = _NullProgress()
        cb = MetricsProgressCallback(null, agent_id="test_orch")
        return cb, null

    @pytest.mark.asyncio
    async def test_forwards_all_events(self, setup):
        cb, null = setup
        await cb.on_progress("plan_ready", "Plan listo", {"plan": {}})
        assert len(null.calls) == 1
        assert null.calls[0][0] == "plan_ready"

    @pytest.mark.asyncio
    async def test_plan_ready_increments_debates(self, setup):
        cb, _ = setup
        before = _counter_value(DEBATES_TOTAL, agent_id="test_orch")
        await cb.on_progress("plan_ready", "Plan listo", {})
        after = _counter_value(DEBATES_TOTAL, agent_id="test_orch")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_subtask_dispatch_by_role(self, setup):
        cb, _ = setup
        before = _counter_value(
            SUBTASK_DISPATCH, agent_id="test_orch", role_id="analyst"
        )
        await cb.on_progress(
            "subtask_dispatch", "Dispatching", {"role_id": "analyst", "subtask_id": "t1"}
        )
        after = _counter_value(
            SUBTASK_DISPATCH, agent_id="test_orch", role_id="analyst"
        )
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_belief_update_records_delta(self, setup):
        cb, _ = setup
        before = _counter_value(
            BELIEF_UPDATES,
            agent_id="test_orch",
            role_id="seeker",
            phase="post_initial",
        )
        await cb.on_progress(
            "belief_update",
            "Updated",
            {"role_id": "seeker", "delta": 0.75, "phase": "post_initial"},
        )
        after = _counter_value(
            BELIEF_UPDATES,
            agent_id="test_orch",
            role_id="seeker",
            phase="post_initial",
        )
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_aporia_detected_counted(self, setup):
        cb, _ = setup
        before = _counter_value(APORIA_DETECTIONS, agent_id="test_orch")
        await cb.on_progress(
            "aporia_detected", "Aporia!", {"detected": True}
        )
        after = _counter_value(APORIA_DETECTIONS, agent_id="test_orch")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_aporia_not_counted_when_false(self, setup):
        cb, _ = setup
        before = _counter_value(APORIA_DETECTIONS, agent_id="test_orch")
        await cb.on_progress(
            "aporia_detected", "No aporia", {"detected": False}
        )
        after = _counter_value(APORIA_DETECTIONS, agent_id="test_orch")
        assert after == before  # unchanged

    @pytest.mark.asyncio
    async def test_round_start_increments_rounds(self, setup):
        cb, _ = setup
        before = _counter_value(DELIBERATION_ROUNDS, agent_id="test_orch")
        await cb.on_progress("round_start", "Round 1")
        after = _counter_value(DELIBERATION_ROUNDS, agent_id="test_orch")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_spawn_increments_spawns(self, setup):
        cb, _ = setup
        before = _counter_value(
            WORKER_SPAWNS, agent_id="test_orch", role="analyst"
        )
        await cb.on_progress("spawn", "Spawning", {"role": "analyst"})
        after = _counter_value(
            WORKER_SPAWNS, agent_id="test_orch", role="analyst"
        )
        assert after == before + 1


# ── Hook tests ─────────────────────────────────────────────────────────────────


class TestLLMHooks:
    @pytest.mark.asyncio
    async def test_llm_hook_increments_counter(self, monkeypatch):
        import common.llm_provider as llm_mod

        # Save and replace with a fake
        async def fake_llm(**kwargs):
            return "response text"

        monkeypatch.setattr(llm_mod, "llm_complete", fake_llm)

        from common.telemetry.hooks import install_llm_hooks

        install_llm_hooks("test_hook_agent")

        before = _counter_value(
            LLM_CALLS,
            agent_id="test_hook_agent",
            model="test/model",
            status="ok",
        )

        result = await llm_mod.llm_complete(
            model="test/model",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result == "response text"

        after = _counter_value(
            LLM_CALLS,
            agent_id="test_hook_agent",
            model="test/model",
            status="ok",
        )
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_llm_hook_counts_errors(self, monkeypatch):
        import common.llm_provider as llm_mod

        async def failing_llm(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(llm_mod, "llm_complete", failing_llm)

        from common.telemetry.hooks import install_llm_hooks

        install_llm_hooks("test_hook_err")

        before = _counter_value(
            LLM_CALLS,
            agent_id="test_hook_err",
            model="err/model",
            status="error",
        )

        with pytest.raises(RuntimeError):
            await llm_mod.llm_complete(
                model="err/model",
                messages=[{"role": "user", "content": "hi"}],
            )

        after = _counter_value(
            LLM_CALLS,
            agent_id="test_hook_err",
            model="err/model",
            status="error",
        )
        assert after == before + 1


class TestMCPHooks:
    @pytest.mark.asyncio
    async def test_mcp_hook_increments_counter(self, monkeypatch):
        import agents.specialized.executor as spec_mod

        async def fake_mcp(tool, args, *, whitelist):
            return "result"

        monkeypatch.setattr(spec_mod, "_call_mcp_tool", fake_mcp)

        from common.telemetry.hooks import install_mcp_hooks

        install_mcp_hooks("test_mcp_agent")

        before = _counter_value(
            MCP_CALLS,
            agent_id="test_mcp_agent",
            tool="web_search",
            status="ok",
        )

        result = await spec_mod._call_mcp_tool(
            "web_search", {"query": "test"}, whitelist=["web_search"]
        )
        assert result == "result"

        after = _counter_value(
            MCP_CALLS,
            agent_id="test_mcp_agent",
            tool="web_search",
            status="ok",
        )
        assert after == before + 1


# ── install_telemetry tests ────────────────────────────────────────────────────


class TestInstallTelemetry:
    def test_install_adds_metrics_route(self, monkeypatch):
        monkeypatch.setattr("common.config.settings.telemetry_enabled", True)

        app = Starlette(routes=[Route("/", lambda r: PlainTextResponse("ok"))])

        from common.telemetry.install import install_telemetry

        install_telemetry(app, "test_install")

        client = TestClient(app)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "a2a_http_requests_total" in r.text

    def test_install_noop_when_disabled(self, monkeypatch):
        monkeypatch.setattr("common.config.settings.telemetry_enabled", False)

        app = Starlette(routes=[Route("/", lambda r: PlainTextResponse("ok"))])

        from common.telemetry.install import install_telemetry

        install_telemetry(app, "test_disabled")

        client = TestClient(app)
        r = client.get("/metrics")
        # No /metrics route was mounted → 404 or 405
        assert r.status_code in (404, 405)
