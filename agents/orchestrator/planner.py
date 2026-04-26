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
user request into a DAG of sub-tasks that can be delegated to specialized
worker agents.

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
    }
  ],
  "max_workers": 3
}

═══════ GENERAL RULES ═══════
- IDs are short strings ("t1", "t2", ...), unique within the plan.
- `depends_on` lists earlier IDs whose output this subtask needs as context.
- CRITICAL: `required_skill` MUST be copied VERBATIM from the catalog's skill
  `id` field. Never invent, translate, or generalize a skill name. If the
  catalog offers `normalize_input`, `debate`, `format_verdict` then those are
  the ONLY valid values. Skills like `research`, `analysis`, `summarize`,
  `search`, etc. DO NOT EXIST unless they appear in the catalog — using them
  will cause the plan to fail immediately.
- Keep descriptions concrete and actionable — a worker should be able to act
  on the `description` alone (plus the outputs of its `depends_on`).

═══════ PATTERN: deliberative questions (opinions, decisions, comparisons) ═══════
When the user asks something with multiple defensible answers (should we, which
is better, is X right, etc.), generate an ITERATIVE DEBATE PLAN with rounds.

Required structure:
1. ONE `normalize_input` subtask first (no deps).
2. TWO parallel `debate` subtasks for INITIAL OPINIONS (deps: [t1]):
   - One with `perspective` starting with "ae1: <role + stance>"
   - One with `perspective` starting with "ae2: <opposing role + stance>"
   - The two roles MUST be CONTRASTING (e.g. DevOps Engineer vs Team Lead).
3. ALTERNATING ROUNDS of `debate` subtasks. For each round R from 1 to N (N=2):
   - First an `ae2` subtask (deps: both opinions + previous round's outputs)
   - Then an `ae1` subtask (deps: the `ae2` round-R subtask + everything before)
4. ONE `format_verdict` subtask depending on ALL debate subtasks.

For each `debate` subtask, the `description` MUST include:
   - "ROLE: <the role>"
   - "PERSPECTIVE: <the stance>"
   - "ROUND: <opening | round 1 | round 2 of 2 (FINAL — synthesize)>"
   - "GOAL: <argue your stance / find common ground / propose unified answer>"
   - "Format: AGREEMENTS: ... / REFINEMENT: ..."

CRITICAL — `perspective` field convention for debate subtasks:
   - Initial opinion AE1: "ae1: <role>, <stance>"
   - Initial opinion AE2: "ae2: <role>, <stance>"
   - Round-R AE1: "ae1: round <R>"
   - Round-R AE2: "ae2: round <R>"
   The "ae1:" / "ae2:" prefix is MANDATORY — the executor uses it to label
   each agent's own prior arguments vs the opponent's when assembling context.

The FINAL round's `description` MUST instruct the agent to drop the
adversarial framing and propose a single unified answer ("FINAL — synthesize").

═══════ PATTERN: factual / single-answer questions ═══════
If the request has one obvious answer (definitions, calculations, lookups),
do NOT use the debate pattern. Use a short pipeline: normalize → one or two
analytic steps → format_verdict. No rounds.

═══════ EXAMPLE: deliberative plan with 2 rounds ═══════
{
  "goal": "Compare remote vs in-person work for software teams",
  "subtasks": [
    {"id":"t1","description":"Normalize the request into structured JSON","required_skill":"normalize_input","depends_on":[],"perspective":null},
    {"id":"t2","description":"ROLE: DevOps Engineer. PERSPECTIVE: Remote work boosts productivity. ROUND: opening. GOAL: state your initial position. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t1"],"perspective":"ae1: DevOps Engineer, pro-remote"},
    {"id":"t3","description":"ROLE: Team Lead. PERSPECTIVE: In-person work strengthens cohesion. ROUND: opening. GOAL: state your initial position. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t1"],"perspective":"ae2: Team Lead, pro-onsite"},
    {"id":"t4","description":"ROLE: Team Lead. ROUND: round 1 of 2. GOAL: respond to AE1's opening — concede valid points first, then refine. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t2","t3"],"perspective":"ae2: round 1"},
    {"id":"t5","description":"ROLE: DevOps Engineer. ROUND: round 1 of 2. GOAL: respond to AE2's round 1 — concede valid points first, then refine. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t2","t3","t4"],"perspective":"ae1: round 1"},
    {"id":"t6","description":"ROLE: Team Lead. ROUND: round 2 of 2 (FINAL — synthesize). GOAL: drop the adversarial framing and propose a unified answer integrating both sides. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t2","t3","t4","t5"],"perspective":"ae2: round 2"},
    {"id":"t7","description":"ROLE: DevOps Engineer. ROUND: round 2 of 2 (FINAL — synthesize). GOAL: drop the adversarial framing and propose a unified answer integrating both sides. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t2","t3","t4","t5","t6"],"perspective":"ae1: round 2"},
    {"id":"t8","description":"Format the final verdict synthesizing all debate rounds","required_skill":"format_verdict","depends_on":["t2","t3","t4","t5","t6","t7"],"perspective":null}
  ],
  "max_workers": 2
}
"""


REPLAN_SYSTEM_PROMPT = """\
A previous plan failed on a specific subtask. Produce a revised plan that
avoids the failure while still achieving the original goal. Same JSON shape
as before. You may drop or rewrite the failed subtask, or route around it.
"""


EXTEND_FOR_CONSENSUS_SYSTEM_PROMPT = """\
A debate plan has finished executing but the two agents have NOT yet reached
consensus. You must produce a SHORT EXTENSION PLAN with 2-4 additional
subtasks that continue the debate from where it left off and force convergence.

Return ONLY valid JSON with the same shape as a normal plan. Rules for the
extension:
- IDs must NOT collide with the original plan. Use prefix "x" (x1, x2, ...).
- All `debate` subtasks in the extension MUST be SYNTHESIS rounds. The
  description must instruct the agent to drop adversarial framing, lead
  with shared ground, and propose a unified answer.
- Use `perspective` with the "ae1: ..." / "ae2: ..." convention exactly as
  before, with round labels like "ae1: synthesis" / "ae2: synthesis".
- `depends_on` for each new subtask MUST include the latest existing debate
  subtask IDs from the original plan so context flows in (you will be told
  which IDs those are).
- End with a `format_verdict` subtask that depends on all NEW debate subtasks
  AND the original final-round subtasks.
- The original plan's normalize step must NOT be repeated.
- Keep the extension small (4-6 subtasks max).
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

    async def extend_for_consensus(
        self,
        original: TaskPlan,
        results: dict[str, str],
        workers: list[dict[str, Any]],
        consensus_reason: str,
    ) -> TaskPlan:
        """Produce a small extension plan to push two agents toward consensus.

        The extension is a fresh DAG that the orchestrator will execute on top
        of the original plan's results. Subtask IDs in the extension must not
        collide with the original.
        """
        catalog = _format_worker_catalog(workers)
        known_skills = {
            s.get("id")
            for w in workers
            for s in (w.get("skills") or [])
            if s.get("id")
        }
        # Collect the latest debate subtasks (per-agent) so the extension can
        # depend on them and the agents see their own most recent positions.
        debate_tasks = [t for t in original.subtasks if t.required_skill == "debate"]
        latest_per_agent: dict[str, str] = {}
        for t in debate_tasks:
            persp = (t.perspective or "").lower()
            if persp.startswith("ae1"):
                latest_per_agent["ae1"] = t.id
            elif persp.startswith("ae2"):
                latest_per_agent["ae2"] = t.id
        latest_ids = list(latest_per_agent.values())
        existing_ids = [t.id for t in original.subtasks]

        # Build a compressed view of the latest positions so the planner has context.
        recent_excerpt = "\n\n".join(
            f"[{tid} — {next((t.perspective for t in debate_tasks if t.id == tid), '')}]\n"
            f"{(results.get(tid, '') or '')[:600]}"
            for tid in latest_ids
        )

        user_prompt = (
            f"Original goal: {original.goal}\n\n"
            f"Why consensus has not been reached:\n{consensus_reason}\n\n"
            f"Latest positions (truncated):\n{recent_excerpt}\n\n"
            f"Existing subtask IDs (do NOT reuse, prefix new IDs with 'x'):\n"
            f"{existing_ids}\n\n"
            f"IDs of the most recent debate subtask per agent — your new "
            f"subtasks should depend on these so context propagates:\n"
            f"{latest_per_agent}\n\n"
            f"Worker catalog:\n{catalog}\n\n"
            f"Valid `required_skill` values: {sorted(known_skills)}\n\n"
            "Emit the JSON extension plan now."
        )
        messages = [
            {"role": "system", "content": EXTEND_FOR_CONSENSUS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(2):
            raw = await llm_complete(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            try:
                ext = _parse_plan(raw, known_skills=known_skills)
                # Reject ID collisions with the original plan.
                clash = {t.id for t in ext.subtasks} & set(existing_ids)
                if clash:
                    raise ValueError(
                        f"Extension reuses existing IDs {sorted(clash)}; "
                        f"prefix new IDs with 'x'."
                    )
                logger.info(
                    "Planner produced %d extension subtasks for consensus push",
                    len(ext.subtasks),
                )
                return ext
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Extension plan parse failed (attempt %d): %s", attempt + 1, e)
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": f"That response was invalid: {e}. Return ONLY valid JSON.",
                })

        raise RuntimeError("Planner failed to produce a valid extension plan")

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
