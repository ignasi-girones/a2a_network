import asyncio
from contextlib import asynccontextmanager

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from agents.feedback.executor import FeedbackExecutor
from common.a2a_helpers import build_agent_card, build_skill
from common.config import settings
from common.tls import uvicorn_tls_kwargs
from common.registry_client import (
    agent_card_to_dict,
    deregister_self,
    register_self_with_orchestrator,
)

AGENT_ID = "feedback"


def main():
    port = settings.feedback_port
    url = settings.own_url(port)

    skills = [
        build_skill(
            skill_id="format_verdict",
            name="Format Verdict",
            description=(
                "Produces the final human-readable report of a deliberation. "
                "Output is structured Markdown in Spanish with sections: "
                "executive summary, participants, key arguments, points of "
                "agreement, remaining disagreements, final verdict, and an "
                "explicit debate-state label (Consenso alcanzado / Consenso "
                "parcial / Sin consenso).\n"
                "WHEN TO USE: as the LAST step of any plan that needs a "
                "polished user-facing answer. The orchestrator may also "
                "append it automatically after a consensus loop, so include "
                "it here only if you want this skill to be the explicit sink "
                "of your plan; otherwise the orchestrator will add it.\n"
                "INPUT: depend on every subtask whose output should be "
                "summarised — typically the last debate round of every agent.\n"
                "OUTPUT: the final answer the user will see."
            ),
            tags=["feedback", "formatting", "final-step", "reporting"],
        )
    ]

    card = build_agent_card(
        name="Feedback Agent",
        description=(
            "Transforms structured debate results into clear, "
            "well-formatted human-readable reports."
        ),
        url=url,
        skills=skills,
        streaming=False,
    )

    handler = DefaultRequestHandler(
        agent_executor=FeedbackExecutor(),
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
    install_telemetry(starlette_app, "feedback")

    print(f"Feedback Agent starting on {url}")
    uvicorn.run(
        starlette_app,
        host="0.0.0.0",
        port=port,
        **uvicorn_tls_kwargs("feedback"),
    )


if __name__ == "__main__":
    main()
