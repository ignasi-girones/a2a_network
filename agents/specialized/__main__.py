import argparse
import logging

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

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Specialized Agent (AE)")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--agent-id", type=str, required=True, choices=["ae1", "ae2"])
    args = parser.parse_args()

    port = args.port
    agent_id = args.agent_id
    url = f"http://localhost:{port}"

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
        streaming=False,
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

    starlette_app = app.build()

    # Mount internal config API routes
    starlette_app.state.agent_state = state
    for route in config_routes:
        starlette_app.routes.insert(0, route)

    print(f"Specialized Agent ({agent_id.upper()}) starting on {url} with model {model}")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
