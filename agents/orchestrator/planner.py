"""Planner — Phase 3 deliberative-quartet plan emitter.

The Phase 2 planner asked the LLM to generate the entire DAG (nodes,
dependencies, skills) from scratch. That gave hallucinated skills and
inconsistent debate structures. Phase 3 inverts the contract:

  - The DAG topology is **fixed**: the canonical RAPID-D quartet
    (Analista → {Buscador, Abogado del Diablo} → Sintetizador). The Planner
    no longer chooses the structure.
  - The LLM is restricted to extracting two pieces of context from the
    user prompt: (a) a one-sentence ``goal`` restatement and (b) a
    testable ``claim`` that anchors the panel's debate (and, in Phase B,
    every agent's BeliefState log_odds).
  - Per-role subtask descriptions are deterministic templates filled with
    ``goal`` + ``claim``. The role-specific prompt detail lives in
    ``persona_catalog`` (system prompts) — the SubTask description here
    is a thin instruction for the executor's prompt builder.

This guarantees:
  - Every prompt produces the same quartet, so the frontend graph and the
    TFG memoria can cite a stable architecture.
  - The Planner cannot invent skills outside ``role_<role_id>``.
  - LLM failure degrades gracefully to a hard-coded quartet using
    ``user_input`` as both goal and claim.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from common.llm_provider import llm_complete
from common.models import CANONICAL_ROLES, SubTask, TaskPlan

logger = logging.getLogger(__name__)


# ── Canonical sextet topology ───────────────────────────────────────────────
#
# Phase 1 (parallel):
#   t1 (analyst, no deps)
#
# Phase 2 (parallel after t1):
#   t2 (seeker,          depends_on=[t1])
#   t3 (devils_advocate, depends_on=[t1])
#
# Phase 3 (parallel after t2):
#   t4 (empiricist,  depends_on=[t1, t2])  ← challenges evidence quality
#   t5 (pragmatist,  depends_on=[t1, t2])  ← real-world cases
#
# Phase 4 (after all):
#   t6 (synthesizer, depends_on=[t1, t2, t3, t4, t5])
#
# t3 (DA) can run in parallel with t2 since it only needs t1;
# the PlanExecutor's readiness check handles this automatically.
_SEXTET_DAG: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("t1", "analyst",         ()),
    ("t2", "seeker",          ("t1",)),
    ("t3", "devils_advocate", ("t1",)),
    ("t4", "empiricist",      ("t1", "t2")),
    ("t5", "pragmatist",      ("t1", "t2")),
    ("t6", "synthesizer",     ("t1", "t2", "t3", "t4", "t5")),
)


# Per-role description templates. The {claim} placeholder receives the
# planner-extracted central proposition; descriptions stay short because the
# real role-specific prompt detail lives in persona_catalog system prompts.
_ROLE_DESCRIPTIONS: dict[str, str] = {
    "analyst": (
        "Produce an impartial factual baseline for the central claim "
        "({claim!r}): definitions, stakeholders, verifiable facts, and "
        "explicit [DISPUTED] markers where the literature is contested."
    ),
    "seeker": (
        "Identify 2-3 sub-questions the Analyst left unanswered about "
        "{claim!r} and search for external evidence (web_search) to fill "
        "them. Cite sources visibly and flag failed lookups."
    ),
    "devils_advocate": (
        "Attack the easy conclusion that the Analyst's baseline implies "
        "about {claim!r}. Apply the Schopenhauer eristic stratagem assigned "
        "in your persona — embody it without announcing it."
    ),
    "empiricist": (
        "Interrogate the Analyst and Seeker outputs about {claim!r} with "
        "Popperian falsificationism: demand testable predictions, challenge "
        "methodology, search arxiv for peer-reviewed falsifying evidence."
    ),
    "pragmatist": (
        "Ground the panel's abstract discussion of {claim!r} in concrete "
        "documented cases — companies, projects, or historical precedents "
        "where the claim was tested in practice. Cite outcomes."
    ),
    "synthesizer": (
        "Evaluate all five panel contributions (Analyst, Seeker, Devil's "
        "Advocate, Empiricist, Pragmatist) under Habermasian validity claims "
        "(truth, rightness, sincerity, comprehensibility) and produce a final "
        "verdict on {claim!r}."
    ),
}


_PLANNER_SYSTEM_PROMPT = """\
You are the Planner of a deliberative agentic orchestrator.

The DAG of subtasks is FIXED — you do not choose nodes or dependencies.
A Spanish-language deliberative panel of six roles (Analista, Buscador,
Abogado del Diablo, Empírico, Pragmático, Sintetizador) will run on every
user prompt.

Your single responsibility is to extract two strings from the user's
free-text request:

  1. `goal`  — a one-sentence restatement of what the user wants the panel
               to deliberate about.
  2. `claim` — a single TESTABLE proposition (the form "X is Y because Z"
               or "we should/should not do W"). The dialectic panel will
               argue for and against this claim, so it must be specific
               enough to be supported or refuted with evidence. If the user
               input is open-ended, choose the strongest defensible claim
               implied by it.

Return ONLY valid JSON of the shape:
{
  "goal": "...",
  "claim": "..."
}

Both strings should be in Spanish, between 5 and 30 words. No commentary,
no markdown, no extra fields.
"""


def _format_worker_catalog(workers: list[dict[str, Any]]) -> str:
    """Render the worker catalog purely for diagnostic logging.

    Phase 3 doesn't pass the catalog to the LLM (the DAG is fixed), but we
    still log it so a missing role-worker is visible at plan time.
    """
    if not workers:
        return "(no workers currently registered)"
    lines = []
    for w in workers:
        skills = w.get("skills") or []
        skill_ids = [s.get("id") for s in skills]
        lines.append(f"- {w.get('agent_id')}: {skill_ids}")
    return "\n".join(lines)


def _validate_role_coverage(
    workers: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Return (covered_roles, missing_roles) according to the registry.

    A role is "covered" if at least one worker advertises ``role_<role_id>``
    as a skill id. Missing roles will need to be spawned by the orchestrator
    (sub-phase A.6) before the plan executes.
    """
    advertised: set[str] = set()
    for w in workers:
        for s in w.get("skills") or []:
            sid = s.get("id")
            if isinstance(sid, str):
                advertised.add(sid)
    covered = {r for r in CANONICAL_ROLES if f"role_{r}" in advertised}
    missing = set(CANONICAL_ROLES) - covered
    return covered, missing


def _build_sextet_plan(goal: str, claim: str) -> TaskPlan:
    """Assemble the canonical 6-node TaskPlan deterministically."""
    subtasks: list[SubTask] = []
    for tid, role, deps in _SEXTET_DAG:
        description = _ROLE_DESCRIPTIONS[role].format(claim=claim or goal)
        subtasks.append(
            SubTask(
                id=tid,
                description=description,
                role_id=role,  # type: ignore[arg-type]
                required_skill=f"role_{role}",
                depends_on=list(deps),
                perspective=None,
            )
        )
    return TaskPlan(
        goal=goal,
        claim=claim,
        subtasks=subtasks,
        max_workers=len(_SEXTET_DAG),
    )


# Keep backward-compat alias so any external callers still compile.
_build_quartet_plan = _build_sextet_plan


def _parse_extraction(raw: str) -> tuple[str, str]:
    """Pull (goal, claim) out of the LLM's JSON response.

    Raises ValueError on malformed JSON or missing fields. We don't accept
    partial responses — both fields must be present and non-empty for the
    extraction to count as successful.
    """
    data = json.loads(raw)
    goal = (data.get("goal") or "").strip()
    claim = (data.get("claim") or "").strip()
    if not goal or not claim:
        raise ValueError(
            f"Planner response missing required fields "
            f"(goal={goal!r}, claim={claim!r})"
        )
    return goal, claim


class Planner:
    """Phase 3 Planner — extracts goal+claim, emits the canonical quartet.

    The LLM is consulted only for natural-language extraction. The DAG
    structure is hard-coded and never depends on the user input or the
    worker catalog — that's the deterministic backbone the TFG memoria
    can cite as the dialectic architecture.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def create_plan(
        self,
        user_input: str,
        workers: list[dict[str, Any]],
    ) -> TaskPlan:
        """Extract goal+claim from `user_input`, return canonical quartet.

        ``workers`` is logged but not passed to the LLM. Coverage gaps
        (missing roles) are surfaced via warnings; the orchestrator is
        responsible for spawning missing workers via WorkerSpawner before
        executing.
        """
        catalog = _format_worker_catalog(workers)
        covered, missing = _validate_role_coverage(workers)
        logger.info("Planner sees registry:\n%s", catalog)
        if missing:
            logger.warning(
                "Planner: roles missing from registry, orchestrator must "
                "spawn them: %s (covered: %s)",
                sorted(missing), sorted(covered),
            )

        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User request:\n{user_input}\n\n"
                    "Emit the JSON {goal, claim} now."
                ),
            },
        ]

        for attempt in range(3):
            try:
                raw = await llm_complete(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                goal, claim = _parse_extraction(raw)
                logger.info(
                    "Planner extracted goal=%r claim=%r", goal, claim
                )
                return _build_sextet_plan(goal, claim)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Plan extraction failed (attempt %d): %s",
                    attempt + 1, e,
                )
                messages.append({
                    "role": "user",
                    "content": (
                        f"That response was invalid: {e}. Return ONLY a "
                        'JSON object with non-empty "goal" and "claim" '
                        "fields, both Spanish strings."
                    ),
                })

        # Hard fallback: the dialectic backbone is too important to skip
        # because of LLM JSON errors. Use the raw user input as both goal
        # and claim — the Analyst's baseline will absorb the ambiguity.
        logger.warning(
            "Planner falling back to user_input as goal+claim after retries"
        )
        fallback_text = user_input.strip()[:200] or "Tema sin especificar"
        return _build_sextet_plan(fallback_text, fallback_text)

    async def replan(
        self,
        original: TaskPlan,
        failed_task: SubTask,
        error: str,
        workers: list[dict[str, Any]],
    ) -> TaskPlan:
        """Rebuild the canonical quartet (preserving goal+claim).

        Phase 3 doesn't reshape the DAG on failure — the four roles are
        non-negotiable. Replanning instead means re-emitting the same
        plan; the orchestrator (caller) is expected to retry the failed
        subtask, possibly after spawning a fresh worker. We accept the
        failed-task and error arguments to preserve the API surface for
        DRTAG (Phase C), which will use them to choose a new stratagem.
        """
        logger.info(
            "Planner.replan: keeping canonical quartet (failed=%s, error=%s)",
            failed_task.id, error,
        )
        return _build_sextet_plan(original.goal, original.claim)
