import argparse
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

from agents.specialized.agent_state import AgentState
from agents.specialized.config_api import config_routes
from agents.specialized.executor import SpecializedExecutor
from common.a2a_helpers import build_agent_card, build_skill
from common.config import settings
from common.registry_client import (
    agent_card_to_dict,
    deregister_self,
    register_self_with_orchestrator,
)

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Specialized Agent (AE)")
    parser.add_argument("--port", type=int, required=True)
    # --agent-id accepts arbitrary IDs so WorkerSpawner can launch
    # dynamic workers like "dyn_debate_3". The base agents still use "ae1"/"ae2".
    parser.add_argument("--agent-id", type=str, required=True)
    args = parser.parse_args()

    port = args.port
    agent_id = args.agent_id
    url = settings.own_url(port)

    # Select model based on agent ID
    model = settings.ae1_model if agent_id == "ae1" else settings.ae2_model

    # Create mutable state
    state = AgentState(agent_id=agent_id)

    # Build initial agent card (will be dynamically modified via card_modifier)
    initial_card = build_agent_card(
        name=f"Specialized Agent ({agent_id.upper()})",
        description="Awaiting role assignment from orchestrator.",
        url=url,
        skills=[
            build_skill(
                skill_id="debate",
                name="Debate",
                description="Engage in structured debate on assigned topic.",
                tags=["debate", "analysis"],
            )
        ],
        streaming=True,
    )

    # Dynamic card modifier: reads from AgentState to serve current role
    async def card_modifier(card: AgentCard) -> AgentCard:
        if not await state.is_ready():
            return card

        role = await state.get_role()
        skills_defs = await state.get_skills()

        # Build dynamic skills from state
        from common.a2a_helpers import build_skill as bs
        dynamic_skills = [
            bs(skill_id=s.id, name=s.name, description=s.description, tags=s.tags)
            for s in skills_defs
        ]

        card.name = f"Specialized Agent ({agent_id.upper()}) - {role}"
        card.description = f"Agent configured as: {role}"
        del card.skills[:]
        card.skills.extend(dynamic_skills)
        return card

    executor = SpecializedExecutor(state=state, model=model)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=initial_card,
        http_handler=handler,
        card_modifier=card_modifier,
    )

    # Serialize the card once for registry use.
    card_dict = agent_card_to_dict(
        name=initial_card.name,
        description=initial_card.description,
        skills=initial_card.skills,
        streaming=True,
    )

    # Lifespan hook — register on startup, deregister on shutdown.
    # Starlette 1.0 removed add_event_handler; lifespan is the supported API.
    @asynccontextmanager
    async def lifespan(_app):
        asyncio.create_task(
            register_self_with_orchestrator(
                agent_id=agent_id, url=url, card=card_dict
            )
        )
        try:
            yield
        finally:
            await deregister_self(agent_id)

    starlette_app = app.build(lifespan=lifespan)

    # Mount internal config API routes
    starlette_app.state.agent_state = state
    for route in config_routes:
        starlette_app.routes.insert(0, route)

    print(f"Specialized Agent ({agent_id.upper()}) starting on {url} with model {model}")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
