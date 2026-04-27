"""HTTP routes that expose the configured LLM models per agent.

These routes are mounted alongside the A2A Starlette app (same as
`registry_routes` and `planner_routes`). They let the frontend show the
real model each agent is using — read from `settings`, which pydantic
populates from the `.env` file at startup.

Endpoint:
    GET /models
        returns: {
          "orchestrator": "groq/llama-3.3-70b-versatile",
          "normalizer": "gemini/gemini-2.5-flash",
          "ae1": "mistral/mistral-large-latest",
          "ae2": "cerebras/qwen-3-235b-a22b-instruct-2507",
          "ae3": "groq/llama-3.1-8b-instant",
          "feedback": "ollama/qwen2.5:14b"
        }
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from common.config import settings


async def get_models(_request: Request) -> JSONResponse:
    """Return the LLM model configured for each agent role."""
    return JSONResponse(
        {
            "orchestrator": settings.orchestrator_model,
            "normalizer": settings.normalizer_model,
            "ae1": settings.ae1_model,
            "ae2": settings.ae2_model,
            "ae3": settings.ae3_model,
            "feedback": settings.feedback_model,
        }
    )


models_routes = [
    Route("/models", get_models, methods=["GET"]),
]
