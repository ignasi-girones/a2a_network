import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from starlette.middleware.cors import CORSMiddleware

from agents.orchestrator.executor import OrchestratorExecutor
from common.a2a_helpers import build_agent_card, build_skill
from common.config import settings

logging.basicConfig(level=logging.INFO)


def main():
    port = settings.orchestrator_port
    url = f"http://localhost:{port}"

    skills = [
        build_skill(
            skill_id="debate_orchestration",
            name="Debate Orchestration",
            description=(
                "Receives a user prompt and orchestrates a full structured "
                "debate between two dynamically-configured specialized agents. "
                "Returns a formatted verdict with analysis."
            ),
            tags=["orchestration", "debate", "multi-agent"],
        )
    ]

    card = build_agent_card(
        name="Orchestrator Agent",
        description=(
            "Central coordinator of the A2A debate network. Normalizes input, "
            "assigns roles to specialized agents, manages debate rounds, "
            "evaluates consensus, and delivers the final verdict."
        ),
        url=url,
        skills=skills,
        streaming=True,
    )

    handler = DefaultRequestHandler(
        agent_executor=OrchestratorExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=handler,
    )

    starlette_app = app.build()

    # Add CORS for frontend (localhost:3000)
    starlette_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    print(f"Orchestrator Agent starting on {url}")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
