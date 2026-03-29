import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from agents.normalizer.executor import NormalizerExecutor
from common.a2a_helpers import build_agent_card, build_skill
from common.config import settings


def main():
    port = settings.normalizer_port
    url = f"http://localhost:{port}"

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

    print(f"Normalizer Agent starting on {url}")
    uvicorn.run(app.build(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
