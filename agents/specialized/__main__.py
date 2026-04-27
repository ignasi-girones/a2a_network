"""Entrypoint for a Specialized worker (Phase 3 — dialectic role binding).

Each worker process is bound to **one** dialectic role at startup via the
mandatory ``--role`` flag. The role determines:

  - which LLM model the worker uses (so the system stays multi-provider);
  - which skill id the worker advertises in the registry so the planner can
    route the right subtask to it (``role_analyst``, ``role_seeker``,
    ``role_devils_advocate``, ``role_synthesizer``);
  - the initial display name shown until the orchestrator POSTs the full
    ``PersonaContract`` via ``/internal/configure``.

The persona itself (system prompt template, tool whitelist, eristic
stratagem, temperature) is *runtime-configured* by the orchestrator — not
baked here — so the same worker process can run with a different stratagem
on the next dispatch (e.g. for DRTAG aporia disruption).
"""

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
from common.models import CANONICAL_ROLES
from common.registry_client import (
    agent_card_to_dict,
    deregister_self,
    register_self_with_orchestrator,
)

logging.basicConfig(level=logging.INFO)


# ── Role → model mapping ────────────────────────────────────────────────────
# Phase A keeps the Phase 2 multi-provider story alive: each role lands on a
# distinct provider where possible. Synthesizer/analyst share Mistral because
# both want stability; Devil's Advocate gets Cerebras (high-throughput, good
# for high-temperature combative variety); Seeker gets Groq (fast turnaround
# during the search-and-refine cycle).
ROLE_MODELS: dict[str, str] = {
    "analyst":         settings.ae1_model,         # mistral large
    "seeker":          settings.orchestrator_model, # groq llama-3.3-70b
    "devils_advocate": settings.ae2_model,          # cerebras qwen
    "empiricist":      settings.orchestrator_model, # groq llama-3.3-70b
    "pragmatist":      settings.ae1_model,          # mistral large
    "synthesizer":     settings.ae1_model,          # mistral large
}

# ── Role → display label / skill description ──────────────────────────────
ROLE_LABELS: dict[str, str] = {
    "analyst":         "Analista",
    "seeker":          "Buscador",
    "devils_advocate": "Abogado del Diablo",
    "empiricist":      "Empírico",
    "pragmatist":      "Pragmático",
    "synthesizer":     "Sintetizador",
}

ROLE_SKILL_DESCRIPTIONS: dict[str, str] = {
    "analyst": (
        "Produce an impartial factual baseline for a deliberative panel: "
        "facts, definitions, stakeholders, and explicit uncertainty markers."
    ),
    "seeker": (
        "Enrich the analyst's baseline with external evidence, using web "
        "search to fill in gaps the analyst could not have known."
    ),
    "devils_advocate": (
        "Adversarially attack the easy conclusion using a Schopenhauer "
        "eristic stratagem assigned by the orchestrator."
    ),
    "empiricist": (
        "Apply Popperian falsificationism to the panel: challenge methodology, "
        "demand testable predictions, search arxiv for contradicting evidence."
    ),
    "pragmatist": (
        "Ground abstract claims in documented real-world cases — companies, "
        "projects, historical precedents — citing concrete outcomes."
    ),
    "synthesizer": (
        "Apply Habermasian validity-claims (truth, rightness, sincerity, "
        "comprehensibility) to the panel outputs and produce a final verdict."
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Specialized Agent (Phase 3 dialectic worker)")
    parser.add_argument("--port", type=int, required=True)
    # --agent-id accepts arbitrary IDs so WorkerSpawner can launch
    # dynamic workers like "dyn_devils_advocate_2" for DRTAG.
    parser.add_argument("--agent-id", type=str, required=True)
    parser.add_argument(
        "--role",
        type=str,
        required=True,
        choices=list(CANONICAL_ROLES),
        help=(
            "Dialectic role this worker binds to at startup. Determines "
            "the advertised skill and the LLM model. Persona contract "
            "(system prompt, stratagem) is set per-run by the orchestrator."
        ),
    )
    args = parser.parse_args()

    port = args.port
    agent_id = args.agent_id
    role = args.role
    url = settings.own_url(port)
    model = ROLE_MODELS[role]
    label = ROLE_LABELS[role]

    # Worker state (persona arrives later via /internal/configure)
    state = AgentState(agent_id=agent_id)

    # Initial agent card — advertises the role-specific skill so the planner
    # can route subtasks to the right worker via find_by_skill().
    skill_id = f"role_{role}"
    initial_card = build_agent_card(
        name=f"Specialized Agent ({agent_id}) — {label}",
        description=(
            f"Dialectic role: {label}. "
            f"Awaiting persona configuration from orchestrator."
        ),
        url=url,
        skills=[
            build_skill(
                skill_id=skill_id,
                name=label,
                description=ROLE_SKILL_DESCRIPTIONS[role],
                tags=["dialectic", role],
            )
        ],
        streaming=True,
    )

    async def card_modifier(card: AgentCard) -> AgentCard:
        """Update name/description once the orchestrator has configured a persona.

        We don't rewrite the skill list here: the role-binding skill is fixed
        at startup and used for routing. Stratagem variations and persona
        details are surfaced through the orchestrator's progress events, not
        through the agent card.
        """
        if not await state.is_ready():
            return card

        display_name = await state.get_display_name()
        role_id = await state.get_role_id()
        card.name = f"Specialized Agent ({agent_id}) — {display_name}"
        card.description = (
            f"Configured persona: {display_name} (role={role_id})."
        )
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

    # Serialize the (initial) card once for registry use.
    card_dict = agent_card_to_dict(
        name=initial_card.name,
        description=initial_card.description,
        skills=initial_card.skills,
        streaming=True,
    )

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

    print(
        f"Specialized Agent ({agent_id}) starting on {url} | "
        f"role={role} | model={model}"
    )
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
