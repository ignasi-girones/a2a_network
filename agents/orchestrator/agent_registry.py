"""In-memory Agent Registry hosted on the orchestrator.

Workers (normalizer, ae1, ae2, feedback, future dynamic workers) POST their
Agent Card here on startup so the orchestrator can discover available
capabilities at plan time — replacing the hardcoded-port approach.

This is **infrastructure**, not A2A protocol: it runs alongside the orchestrator's
A2A Starlette app as extra routes, like `config_api.py` does for specialized agents.

Exposed endpoints:
    POST   /registry/register       — worker announces itself (body: WorkerEntry JSON)
    GET    /registry/agents         — list all registered workers
    DELETE /registry/{agent_id}     — worker announces shutdown
    GET    /registry/by-skill/{id}  — filter workers that advertise a given skill id
"""

from __future__ import annotations

import asyncio
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from common.models import WorkerEntry

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Thread-safe in-memory store of registered worker agents."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerEntry] = {}
        self._lock = asyncio.Lock()

    async def register(self, entry: WorkerEntry) -> None:
        async with self._lock:
            self._workers[entry.agent_id] = entry
        logger.info(
            "Registered worker: %s @ %s (skills=%s)",
            entry.agent_id,
            entry.url,
            [s.get("id") for s in entry.card.get("skills", [])],
        )

    async def deregister(self, agent_id: str) -> bool:
        async with self._lock:
            removed = self._workers.pop(agent_id, None)
        if removed:
            logger.info("Deregistered worker: %s", agent_id)
            return True
        return False

    async def all_workers(self) -> list[WorkerEntry]:
        async with self._lock:
            return list(self._workers.values())

    async def find_by_skill(self, skill_id: str) -> list[WorkerEntry]:
        async with self._lock:
            return [
                w
                for w in self._workers.values()
                if any(s.get("id") == skill_id for s in w.card.get("skills", []))
            ]

    async def find_by_agent_id(self, agent_id: str) -> WorkerEntry | None:
        async with self._lock:
            return self._workers.get(agent_id)


# Module-level singleton — one registry per orchestrator process.
registry = AgentRegistry()


# ── HTTP route handlers ──────────────────────────────────────────────────────


async def register_worker(request: Request) -> JSONResponse:
    """POST /registry/register — body: {agent_id, url, card}."""
    try:
        body = await request.json()
        entry = WorkerEntry(**body)
    except Exception as e:
        logger.warning("Invalid registration payload: %s", e)
        return JSONResponse(
            {"status": "error", "detail": f"Invalid payload: {e}"},
            status_code=400,
        )

    await registry.register(entry)
    return JSONResponse(
        {"status": "registered", "agent_id": entry.agent_id},
        status_code=200,
    )


async def deregister_worker(request: Request) -> JSONResponse:
    """DELETE /registry/{agent_id}."""
    agent_id = request.path_params["agent_id"]
    removed = await registry.deregister(agent_id)
    if removed:
        return JSONResponse({"status": "deregistered", "agent_id": agent_id})
    return JSONResponse(
        {"status": "not_found", "agent_id": agent_id},
        status_code=404,
    )


async def list_workers(request: Request) -> JSONResponse:
    """GET /registry/agents — list all registered workers."""
    workers = await registry.all_workers()
    return JSONResponse(
        {
            "count": len(workers),
            "workers": [w.model_dump() for w in workers],
        }
    )


async def list_by_skill(request: Request) -> JSONResponse:
    """GET /registry/by-skill/{skill_id} — workers that advertise a skill."""
    skill_id = request.path_params["skill_id"]
    matches = await registry.find_by_skill(skill_id)
    return JSONResponse(
        {
            "skill_id": skill_id,
            "count": len(matches),
            "workers": [w.model_dump() for w in matches],
        }
    )


# Routes to be mounted into the orchestrator's Starlette app.
registry_routes = [
    Route("/registry/register", register_worker, methods=["POST"]),
    Route("/registry/agents", list_workers, methods=["GET"]),
    Route("/registry/by-skill/{skill_id}", list_by_skill, methods=["GET"]),
    Route("/registry/{agent_id}", deregister_worker, methods=["DELETE"]),
]
