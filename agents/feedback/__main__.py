import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from agents.feedback.executor import FeedbackExecutor
from common.a2a_helpers import build_agent_card, build_skill
from common.config import settings


def main():
    port = settings.feedback_port
    url = f"http://localhost:{port}"

    skills = [
        build_skill(
            skill_id="format_verdict",
            name="Format Verdict",
            description=(
                "Takes a structured debate summary (JSON) and produces "
                "a human-readable report with executive summary, key "
                "arguments, verdict, and confidence level."
            ),
            tags=["feedback", "formatting", "reporting"],
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

    print(f"Feedback Agent starting on {url}")
    uvicorn.run(app.build(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
