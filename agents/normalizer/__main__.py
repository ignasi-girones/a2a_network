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
                "Converts a raw free-text user prompt into a structured JSON "
                "object with: topic, domain, question type, explicit "
                "constraints, and suggested contrasting perspectives.\n"
                "WHEN TO USE: as the FIRST step of any plan whose user input "
                "arrives as raw natural language. Downstream subtasks should "
                "depend on this one so they receive structured context "
                "instead of having to parse the free text themselves.\n"
                "INPUT: the user's original prompt (no extra context needed).\n"
                "OUTPUT: a single JSON object — pass its id as a `depends_on` "
                "to any subtask that benefits from structured context."
            ),
            tags=["normalization", "parsing", "preprocessing", "first-step"],
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

    from common.telemetry import install_telemetry
    install_telemetry(starlette_app, "normalizer")

    print(f"Normalizer Agent starting on {url}")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
