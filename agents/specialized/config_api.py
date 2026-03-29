"""Internal configuration API for Specialized Agents.

This is NOT part of the A2A protocol. It's an infrastructure endpoint
that the orchestrator uses to dynamically configure agent roles before
each debate flow.
"""

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents.specialized.agent_state import AgentState
from common.models import AgentRoleConfig

logger = logging.getLogger(__name__)


async def configure_agent(request: Request) -> JSONResponse:
    """POST /internal/configure - Set the agent's role and skills."""
    state: AgentState = request.app.state.agent_state
    body = await request.json()

    try:
        config = AgentRoleConfig(**body)
        await state.configure(config)
        logger.info(
            "Agent %s configured as: %s (%s)",
            state.agent_id,
            config.role,
            config.perspective,
        )
        return JSONResponse({
            "status": "configured",
            "agent_id": state.agent_id,
            "role": config.role,
        })
    except Exception as e:
        logger.error("Configuration failed: %s", e)
        return JSONResponse(
            {"status": "error", "detail": str(e)},
            status_code=400,
        )


async def get_state(request: Request) -> JSONResponse:
    """GET /internal/state - Get the agent's current configuration."""
    state: AgentState = request.app.state.agent_state
    return JSONResponse({
        "agent_id": state.agent_id,
        "role": await state.get_role(),
        "ready": await state.is_ready(),
        "skills": [s.model_dump() for s in await state.get_skills()],
    })


config_routes = [
    Route("/internal/configure", configure_agent, methods=["POST"]),
    Route("/internal/state", get_state, methods=["GET"]),
]
