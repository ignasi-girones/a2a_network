"""HTTP routes that expose the Planner for debugging / observability.

These routes are mounted alongside the A2A Starlette app (same as
`agent_registry.registry_routes`). They are infrastructure, not A2A
protocol: the frontend can call them to preview the plan the orchestrator
would generate for a given prompt, before any worker is invoked.

Endpoints:
    POST /orchestrator/plan
        body: {"input": "free-text user request"}
        returns: the TaskPlan JSON plus the worker catalog used.

    POST /orchestrator/plan/execute   (debug-only)
        body: {"input": "free-text user request"}
        returns: the plan + per-subtask outputs after running PlanExecutor.
        Note: this is a debug affordance — the production debate path is
        still FlowManager. Sub-phase 2c will replace FlowManager with
        AgenticOrchestrator and retire this endpoint.
"""

from __future__ import annotations

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents.orchestrator.agent_registry import registry
from agents.orchestrator.agentic_orchestrator import AgenticOrchestrator
from agents.orchestrator.plan_executor import (
    PlanExecutionError,
    PlanExecutor,
    ProgressCallback,
)
from agents.orchestrator.planner import Planner
from agents.orchestrator.worker_spawner import get_spawner
from common.config import settings

logger = logging.getLogger(__name__)


def _serialize_worker_catalog() -> list[dict]:
    """Flatten registry.all_workers() into the shape the Planner expects.

    Planner reads `skills` directly off each entry, so we lift them out of
    `card` here rather than making the Planner know about WorkerEntry.
    """
    # NOTE: this is sync-serialized — caller is async and awaits registry first.
    raise NotImplementedError  # placeholder, overridden by async helper below


async def _catalog() -> list[dict]:
    workers = await registry.all_workers()
    return [
        {
            "agent_id": w.agent_id,
            "url": w.url,
            "skills": w.card.get("skills") or [],
        }
        for w in workers
    ]


async def preview_plan(request: Request) -> JSONResponse:
    """POST /orchestrator/plan — returns the generated plan, does not execute."""
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            {"status": "error", "detail": f"Invalid JSON body: {e}"},
            status_code=400,
        )

    user_input = (body or {}).get("input", "").strip()
    if not user_input:
        return JSONResponse(
            {"status": "error", "detail": "Missing 'input' field"},
            status_code=400,
        )

    catalog = await _catalog()
    planner = Planner(model=settings.orchestrator_model)

    try:
        plan = await planner.create_plan(user_input, catalog)
    except Exception as e:
        logger.exception("Planner failed")
        return JSONResponse(
            {"status": "error", "detail": f"Planner failed: {e}"},
            status_code=500,
        )

    return JSONResponse({
        "status": "ok",
        "input": user_input,
        "catalog": catalog,
        "plan": plan.model_dump(),
    })


async def execute_plan(request: Request) -> JSONResponse:
    """POST /orchestrator/plan/execute — plan + run it. Debug only."""
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            {"status": "error", "detail": f"Invalid JSON body: {e}"},
            status_code=400,
        )

    user_input = (body or {}).get("input", "").strip()
    if not user_input:
        return JSONResponse(
            {"status": "error", "detail": "Missing 'input' field"},
            status_code=400,
        )

    catalog = await _catalog()
    planner = Planner(model=settings.orchestrator_model)

    try:
        plan = await planner.create_plan(user_input, catalog)
    except Exception as e:
        logger.exception("Planner failed")
        return JSONResponse(
            {"status": "error", "detail": f"Planner failed: {e}"},
            status_code=500,
        )

    executor = PlanExecutor(registry=registry)
    try:
        results = await executor.execute(plan)
    except PlanExecutionError as e:
        return JSONResponse(
            {
                "status": "error",
                "detail": str(e),
                "plan": plan.model_dump(),
            },
            status_code=500,
        )

    return JSONResponse({
        "status": "ok",
        "input": user_input,
        "plan": plan.model_dump(),
        "results": results,
    })


class _CollectingProgress(ProgressCallback):
    """ProgressCallback that buffers every stage into a list (for debug endpoint)."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def on_progress(
        self, stage: str, message: str, data: dict | None = None
    ) -> None:
        self.events.append({"stage": stage, "message": message, "data": data})


async def agentic_run(request: Request) -> JSONResponse:
    """POST /orchestrator/agentic/run — non-streaming debug endpoint.

    Invokes AgenticOrchestrator (Planner → ensure-capacity → PlanExecutor →
    synthesize) and returns the final verdict plus a trace of every progress
    event. Useful for verifying sub-phase 2c without going through the SDK's
    streaming client.
    """
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            {"status": "error", "detail": f"Invalid JSON body: {e}"},
            status_code=400,
        )

    user_input = (body or {}).get("input", "").strip()
    if not user_input:
        return JSONResponse(
            {"status": "error", "detail": "Missing 'input' field"},
            status_code=400,
        )

    progress = _CollectingProgress()
    orchestrator = AgenticOrchestrator(
        registry=registry,
        spawner=get_spawner(),
        progress=progress,
    )
    try:
        verdict = await orchestrator.run(user_input)
    except Exception as e:
        logger.exception("Agentic run failed")
        return JSONResponse(
            {
                "status": "error",
                "detail": f"Agentic run failed: {e}",
                "events": progress.events,
            },
            status_code=500,
        )

    return JSONResponse({
        "status": "ok",
        "input": user_input,
        "verdict": verdict,
        "events": progress.events,
    })


planner_routes = [
    Route("/orchestrator/plan", preview_plan, methods=["POST"]),
    Route("/orchestrator/plan/execute", execute_plan, methods=["POST"]),
    Route("/orchestrator/agentic/run", agentic_run, methods=["POST"]),
]
