"""Internal configuration API for Specialized Agents.

This is NOT part of the A2A protocol. It's an infrastructure endpoint
the orchestrator uses to dynamically configure a worker's PersonaContract
before each deliberative run.
"""

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents.specialized.agent_state import AgentState
from common.models import AgentRoleConfig

logger = logging.getLogger(__name__)


async def configure_agent(request: Request) -> JSONResponse:
    """POST /internal/configure — set the agent's persona and skills."""
    state: AgentState = request.app.state.agent_state
    body = await request.json()

    try:
        config = AgentRoleConfig(**body)
        await state.configure(config)
        logger.info(
            "Agent %s configured as role=%s (%s) stratagem=%s",
            state.agent_id,
            config.persona.role_id,
            config.persona.display_name,
            config.persona.eristic_stratagem_id,
        )
        return JSONResponse({
            "status": "configured",
            "agent_id": state.agent_id,
            "role_id": config.persona.role_id,
            "display_name": config.persona.display_name,
        })
    except Exception as e:
        logger.error("Configuration failed: %s", e)
        return JSONResponse(
            {"status": "error", "detail": str(e)},
            status_code=400,
        )


async def get_state(request: Request) -> JSONResponse:
    """GET /internal/state — current configuration."""
    state: AgentState = request.app.state.agent_state
    return JSONResponse({
        "agent_id": state.agent_id,
        "role_id": await state.get_role_id(),
        "display_name": await state.get_display_name(),
        "ready": await state.is_ready(),
        "skills": [s.model_dump() for s in await state.get_skills()],
    })


config_routes = [
    Route("/internal/configure", configure_agent, methods=["POST"]),
    Route("/internal/state", get_state, methods=["GET"]),
]
