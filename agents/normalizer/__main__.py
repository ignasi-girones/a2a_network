import asyncio
from contextlib import asynccontextmanager

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from agents.normalizer.executor import NormalizerExecutor
from common.a2a_helpers import build_agent_card, build_skill
from common.config import settings
from common.registry_client import (
    agent_card_to_dict,
    deregister_self,
    register_self_with_orchestrator,
)

AGENT_ID = "normalizer"


def main():
    port = settings.normalizer_port
    url = settings.own_url(port)

    skills = [
        build_skill(
            skill_id="normalize_input",
            name="Normalize Input",
            description=(
                "Transforms free-text user input into a structured JSON "
                "format with topic, domain, question type, constraints, "
                "and suggested debate perspectives."
            ),
            tags=["normalization", "parsing", "structuring"],
        )
    ]

    card = build_agent_card(
        name="Normalizer Agent",
        description=(
            "Analyzes and normalizes user prompts into structured data "
            "for downstream processing by specialized debate agents."
        ),
        url=url,
        skills=skills,
        streaming=False,
    )

    handler = DefaultRequestHandler(
        agent_executor=NormalizerExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=handler,
    )

    # Auto-register with the orchestrator's AgentRegistry.
    card_dict = agent_card_to_dict(
        name=card.name,
        description=card.description,
        skills=card.skills,
        streaming=False,
    )

    @asynccontextmanager
    async def lifespan(_app):
        asyncio.create_task(
            register_self_with_orchestrator(
                agent_id=AGENT_ID, url=url, card=card_dict
            )
        )
        try:
            yield
        finally:
            await deregister_self(AGENT_ID)

    starlette_app = app.build(lifespan=lifespan)

    print(f"Normalizer Agent starting on {url}")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
