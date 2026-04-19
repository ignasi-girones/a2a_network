"""Planner — LLM-driven decomposition of a user request into a TaskPlan.

Given (a) the user's free-text input and (b) the skills currently advertised
by workers in the AgentRegistry, the Planner emits a DAG of SubTasks.

The plan is consumed by PlanExecutor, which runs ready subtasks in parallel,
dispatching each one to a worker via A2A based on `required_skill`.

Design notes:
- The prompt bakes in the "Plan-and-Execute" pattern: one LLM pass generates
  the full DAG up front; we don't re-plan between steps (that's what `replan`
  is for, invoked only on failure).
- We pass the live skill catalog into the prompt so the planner can only
  request skills it knows exist. Any mismatch is caught post-parse and,
  in sub-phase 2c, will trigger `WorkerSpawner`.
- JSON-mode is requested via `response_format={"type": "json_object"}` for
  providers that support it (Groq does). Even so, we validate and do one
  retry with a corrective message if parsing fails.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from common.llm_provider import llm_complete
from common.models import SubTask, TaskPlan

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """\
You are the Planner of an agentic orchestrator. Your job is to decompose a
user request into a small DAG (2-6 nodes) of sub-tasks that can be delegated
to specialized worker agents.

You will receive a catalog of available workers with their skills. Every
`required_skill` in your plan MUST match an `id` from that catalog.

Return ONLY valid JSON with this exact shape:
{
  "goal": "one-sentence restatement of the user's intent",
  "subtasks": [
    {
      "id": "t1",
      "description": "concrete instruction for the worker",
      "required_skill": "<skill id from catalog>",
      "depends_on": [],
      "perspective": null
    },
    {
      "id": "t2",
      "description": "...",
      "required_skill": "<skill id from catalog>",
      "depends_on": ["t1"],
      "perspective": "pro"
    }
  ],
  "max_workers": 3
}

Rules:
- IDs are short strings ("t1", "t2", ...), unique within the plan.
- `depends_on` lists earlier IDs whose output this subtask needs as context.
- `perspective` is only meaningful for debate-style workers; use null otherwise.
- Prefer a pipeline where a normalization/structuring step runs first, then
  analytical workers run in parallel on its output, then a synthesis step.
- CRITICAL: `required_skill` MUST be copied VERBATIM from the catalog's skill
  `id` field. Never invent, translate, or generalize a skill name. If the
  catalog offers `normalize_input`, `debate`, `format_verdict` then those are
  the ONLY valid values. Skills like `research`, `analysis`, `summarize`,
  `search`, etc. DO NOT EXIST unless they appear in the catalog — using them
  will cause the plan to fail immediately.
- Keep descriptions concrete and actionable — a worker should be able to act
  on the `description` alone (plus the outputs of its `depends_on`).
"""


REPLAN_SYSTEM_PROMPT = """\
A previous plan failed on a specific subtask. Produce a revised plan that
avoids the failure while still achieving the original goal. Same JSON shape
as before. You may drop or rewrite the failed subtask, or route around it.
"""


def _format_worker_catalog(workers: list[dict[str, Any]]) -> str:
    """Render the available-workers catalog for the planner prompt.

    `workers` is a list of dicts with keys {agent_id, url, skills} where each
    skill is {id, name, description, tags}. We keep this compact because it
    goes into every planning call.
    """
    if not workers:
        return "(no workers currently registered)"

    lines = []
    for w in workers:
        skills = w.get("skills") or []
        if not skills:
            lines.append(f"- agent_id={w.get('agent_id')}: (no skills advertised)")
            continue
        skill_lines = [
            f"    • id={s.get('id')!r} name={s.get('name')!r} "
            f"tags={s.get('tags') or []}"
            for s in skills
        ]
        lines.append(
            f"- agent_id={w.get('agent_id')}:\n" + "\n".join(skill_lines)
        )
    return "\n".join(lines)


def _parse_plan(raw: str, known_skills: set[str] | None = None) -> TaskPlan:
    """Parse and validate a planner LLM response into a TaskPlan.

    Raises ValueError on malformed JSON or invalid plan structure: unknown
    dependencies, duplicate IDs, empty subtasks list, or — when
    `known_skills` is provided — `required_skill` values that aren't in
    the catalog.

    The skill check lives here (not in the executor) so the Planner can
    detect hallucinated skills and re-prompt the LLM with a corrective
    message before burning a full execution attempt.
    """
    data = json.loads(raw)
    plan = TaskPlan(**data)

    ids = [t.id for t in plan.subtasks]
    if not ids:
        raise ValueError("Plan contains no subtasks")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate subtask IDs: {ids}")
    known = set(ids)
    for t in plan.subtasks:
        for dep in t.depends_on:
            if dep not in known:
                raise ValueError(
                    f"Subtask {t.id!r} depends on unknown id {dep!r}"
                )

    if known_skills is not None:
        invented = {
            t.required_skill for t in plan.subtasks
            if t.required_skill not in known_skills
        }
        if invented:
            raise ValueError(
                f"Plan references unknown skills {sorted(invented)}. "
                f"Available skills: {sorted(known_skills)}"
            )
    return plan


class Planner:
    """Wraps the LLM call that turns user input + worker catalog → TaskPlan."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.4,
        max_tokens: int = 900,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def create_plan(
        self,
        user_input: str,
        workers: list[dict[str, Any]],
    ) -> TaskPlan:
        """Generate a TaskPlan for `user_input` using `workers` as the catalog.

        `workers` is the serialized output of the AgentRegistry — each element
        is expected to carry at least {agent_id, url, skills}.
        """
        catalog = _format_worker_catalog(workers)
        known_skills = {
            s.get("id")
            for w in workers
            for s in (w.get("skills") or [])
            if s.get("id")
        }
        user_prompt = (
            f"Available workers:\n{catalog}\n\n"
            f"Valid `required_skill` values (choose ONLY from this set): "
            f"{sorted(known_skills)}\n\n"
            f"User request:\n{user_input}\n\n"
            "Emit the JSON plan now."
        )
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(3):
            raw = await llm_complete(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            try:
                plan = _parse_plan(raw, known_skills=known_skills)
                logger.info(
                    "Planner produced %d subtasks for goal=%r",
                    len(plan.subtasks),
                    plan.goal,
                )
                return plan
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Plan parse failed (attempt %d): %s", attempt + 1, e
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That response was invalid: {e}. "
                        f"Remember: `required_skill` MUST be one of "
                        f"{sorted(known_skills)}. Return ONLY valid JSON."
                    ),
                })

        # Last-resort fallback: linear normalize → debate(pro) || debate(con) →
        # format_verdict, IF those skills exist in the catalog. Otherwise raise.
        skill_ids = {
            s.get("id")
            for w in workers
            for s in (w.get("skills") or [])
        }
        if {"normalize_input", "debate", "format_verdict"} <= skill_ids:
            logger.warning("Falling back to default debate plan")
            return TaskPlan(
                goal=user_input[:120],
                subtasks=[
                    SubTask(
                        id="t1",
                        description=(
                            "Normalize the user request into structured JSON "
                            "with topic, domain, and perspectives."
                        ),
                        required_skill="normalize_input",
                    ),
                    SubTask(
                        id="t2",
                        description=(
                            "Argue in favor of the proposal using the "
                            "normalized topic as context."
                        ),
                        required_skill="debate",
                        depends_on=["t1"],
                        perspective="pro",
                    ),
                    SubTask(
                        id="t3",
                        description=(
                            "Argue against the proposal using the "
                            "normalized topic as context."
                        ),
                        required_skill="debate",
                        depends_on=["t1"],
                        perspective="con",
                    ),
                    SubTask(
                        id="t4",
                        description=(
                            "Produce a human-readable verdict synthesizing "
                            "both sides of the debate."
                        ),
                        required_skill="format_verdict",
                        depends_on=["t2", "t3"],
                    ),
                ],
                max_workers=3,
            )
        raise RuntimeError("Planner failed to produce a valid plan")

    async def replan(
        self,
        original: TaskPlan,
        failed_task: SubTask,
        error: str,
        workers: list[dict[str, Any]],
    ) -> TaskPlan:
        """Produce a revised plan after a subtask failed."""
        catalog = _format_worker_catalog(workers)
        known_skills = {
            s.get("id")
            for w in workers
            for s in (w.get("skills") or [])
            if s.get("id")
        }
        user_prompt = (
            f"Original goal: {original.goal}\n\n"
            f"Original plan JSON:\n{original.model_dump_json(indent=2)}\n\n"
            f"Failed subtask id: {failed_task.id}\n"
            f"Failure reason: {error}\n\n"
            f"Current worker catalog:\n{catalog}\n\n"
            f"Valid `required_skill` values: {sorted(known_skills)}\n\n"
            "Emit the revised JSON plan now."
        )
        messages = [
            {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw = await llm_complete(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return _parse_plan(raw, known_skills=known_skills)
