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
is better, is X right, etc.), generate a MINIMAL OPENING PLAN. The orchestrator
runs a separate consensus loop that evaluates the agents' positions after each
exchange and asks you for SYNTHESIS EXTENSIONS only if convergence has not
been reached. This means:
  - You do NOT plan multiple rounds up front.
  - You do NOT include a `format_verdict` step. The orchestrator synthesizes
    the final answer itself from the latest positions of all three agents once
    the consensus loop ends (either by convergence or by hitting the round cap).

Required structure (exactly 4 subtasks):
1. ONE `normalize_input` subtask first (no deps).
2. THREE parallel `debate` subtasks for INITIAL OPINIONS (deps: [t1]):
   - AE1: an advocate for one side ("ae1: <role + stance>")
   - AE2: an advocate for the opposing side ("ae2: <opposing role + stance>")
   - AE3: an INDEPENDENT EVALUATOR who does NOT have a stance assigned
     ("ae3: Independent evaluator").
   The first two roles MUST be CONTRASTING (e.g. DevOps Engineer vs Team
   Lead). AE3 is NOT a mediator looking for a middle ground — AE3 is an
   evidence-driven evaluator who will side with whichever position has the
   strongest arguments, even if that means fully endorsing AE1 or AE2.

For each `debate` subtask, the `description` MUST include:
   - "ROLE: <the role>"
   - "PERSPECTIVE: <stance for AE1/AE2; 'independent evaluator, sides with the
      strongest arguments' for AE3>"
   - "ROUND: opening"
   - "GOAL: <for AE1/AE2> argue your initial position with the strongest
      evidence you can muster, but stay genuinely open to changing your mind
      if the opposing side presents stronger evidence.
      <for AE3> read both opening positions when they arrive (this is the
      opening round so write your initial assessment of the tradeoffs).
      Do NOT default to splitting the difference — if you already see one
      side as more grounded, say so."
   - "Format: AGREEMENTS: ... / REFINEMENT: ..."

CRITICAL — `perspective` field convention for debate subtasks:
   - Initial opinion AE1: "ae1: <role>, <stance>"
   - Initial opinion AE2: "ae2: <role>, <stance>"
   - Initial opinion AE3: "ae3: Independent evaluator"
   The "ae1:" / "ae2:" / "ae3:" prefix is MANDATORY — the executor uses it to
   label each agent's own prior arguments vs the others' when assembling
   context, and the consensus loop uses it to identify each agent's latest
   position.

═══════ PATTERN: factual / single-answer questions ═══════
If the request has one obvious answer (definitions, calculations, lookups),
do NOT use the debate pattern. Use a short pipeline: normalize → one or two
analytic steps → format_verdict. No rounds.

═══════ EXAMPLE: deliberative opening plan (4 subtasks) ═══════
{
  "goal": "Compare remote vs in-person work for software teams",
  "subtasks": [
    {"id":"t1","description":"Normalize the request into structured JSON","required_skill":"normalize_input","depends_on":[],"perspective":null},
    {"id":"t2","description":"ROLE: DevOps Engineer. PERSPECTIVE: Remote work boosts productivity. ROUND: opening. GOAL: state your initial position clearly so the others can respond to it — but stay open to revising it in later rounds. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t1"],"perspective":"ae1: DevOps Engineer, pro-remote"},
    {"id":"t3","description":"ROLE: Team Lead. PERSPECTIVE: In-person work strengthens cohesion. ROUND: opening. GOAL: state your initial position clearly so the others can respond to it — but stay open to revising it in later rounds. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t1"],"perspective":"ae2: Team Lead, pro-onsite"},
    {"id":"t4","description":"ROLE: Independent evaluator. PERSPECTIVE: no assigned stance — sides with the strongest arguments. ROUND: opening. GOAL: write your initial assessment of the tradeoffs based on what you currently know about the topic. Do NOT default to splitting the difference; if one position is already more clearly grounded in evidence, say so. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t1"],"perspective":"ae3: Independent evaluator"}
  ],
  "max_workers": 3
}
"""


REPLAN_SYSTEM_PROMPT = """\
A previous plan failed on a specific subtask. Produce a revised plan that
avoids the failure while still achieving the original goal. Same JSON shape
as before. You may drop or rewrite the failed subtask, or route around it.
"""


EXTEND_FOR_CONSENSUS_SYSTEM_PROMPT = """\
A debate has just finished its latest exchange and the THREE agents (ae1, ae2,
ae3) have NOT yet reached consensus. You must produce ONE more synthesis round
as a SMALL EXTENSION PLAN — exactly THREE additional `debate` subtasks (one
per agent) that continue from the latest positions and push toward convergence.

Return ONLY valid JSON with EXACTLY this shape (all top-level fields required):
{
  "goal": "<restate the original deliberation goal>",
  "subtasks": [ ...three debate subtasks... ],
  "max_workers": 3
}

Rules:
- IDs must NOT collide with existing IDs. Use prefix "x" (e.g. x1, x2, x3).
- Produce EXACTLY THREE subtasks, all `required_skill = "debate"`. No
  `normalize_input`, no `format_verdict` — the orchestrator handles the final
  synthesis itself once consensus is reached or the round budget is exhausted.
- All three subtasks MUST be HONEST RE-EVALUATION rounds. Convergence is the
  goal, but FORCED CENTRISM IS NOT. The instructions you write must:
  1. Tell each agent to genuinely re-weigh both sides on the merits of the
     evidence presented so far, not to defend the role they were assigned.
  2. Make explicit that CHANGING SIDES is encouraged when warranted: if AE1
     finds AE2's evidence stronger overall, AE1 should say "You changed my
     mind — I now agree that <X>" and shift its position toward AE2's stance.
     The same applies in reverse.
  3. Forbid vague "both sides have a point" language used to avoid
     confrontation. If one side is clearly stronger, the agents should say so
     and converge TOWARD that side, not toward the centre.
  4. AE3 must NOT default to mediating between AE1 and AE2. AE3 weighs the
     evidence independently and explicitly endorses the side it finds
     better-grounded — including a full endorsement of AE1 or AE2 when
     warranted. Saying "they are both right in their own way" is a failure
     mode for AE3 unless the evidence genuinely is balanced.
  5. The desired outcome is an *evidence-driven* convergence: ideally the
     three agents end up close to each other AND close to whichever stance
     the strongest evidence supports — not at 0.5 by default.
- Use `perspective` with the "ae1: ..." / "ae2: ..." / "ae3: ..." convention.
  Use round labels like "ae1: synthesis 1", "ae2: synthesis 1",
  "ae3: synthesis 1" (or 2, 3, ...) reflecting how many synthesis rounds have
  already happened.

CRITICAL — PARALLEL DEPS, NOT SEQUENTIAL:
- The three new subtasks must run IN PARALLEL within this round. Therefore
  `depends_on` for ALL THREE new subtasks MUST be the SAME set: the latest
  debate subtask IDs for each of the three agents from the PREVIOUS round
  (you will be told exactly which IDs those are).
- Do NOT make x2 depend on x1, or x3 depend on x1/x2. That would force a
  sequential chain inside the same round, which means the first agent never
  sees what the others say in this round and only the last agent sees
  everyone. We want every agent to react to the same shared snapshot of the
  previous round.
- The orchestrator will tell you the consensus reason — address that gap
  explicitly in each subtask's description.

═══════ EXAMPLE: re-evaluation extension (3 subtasks, all parallel) ═══════
Given the original plan ended with debate IDs t2 (ae1), t3 (ae2), t4 (ae3),
and the consensus reason was "AE2's evidence on metric X looks stronger but
AE1 has not yet engaged with it":
{
  "goal": "Reach an evidence-driven answer on remote vs in-person work",
  "subtasks": [
    {"id":"x1","description":"ROUND: re-evaluation 1. You entered as DevOps Engineer (pro-remote) but your task now is HONEST re-evaluation, not advocacy. Re-read AE2 and AE3's arguments from the previous round. Specifically engage with AE2's evidence on metric X — if it is stronger than your initial counter, say 'You changed my mind on X' and shift your overall stance toward AE2's. If after honest re-weighing you still find your initial side stronger, defend it with the new evidence. Do NOT settle for vague middle-ground language. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t2","t3","t4"],"perspective":"ae1: synthesis 1"},
    {"id":"x2","description":"ROUND: re-evaluation 1. You entered as Team Lead (pro-onsite) but your task now is HONEST re-evaluation, not advocacy. Re-read AE1 and AE3's arguments from the previous round. If AE1's evidence has overturned any of your claims, concede explicitly and shift your stance. If you still find your initial side stronger after honest re-weighing, defend it. Do NOT settle for vague middle-ground language. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t2","t3","t4"],"perspective":"ae2: synthesis 1"},
    {"id":"x3","description":"ROUND: re-evaluation 1. You are an independent evaluator, NOT a mediator. Weigh AE1's vs AE2's arguments strictly on evidence quality. If AE2's evidence on metric X is decisively stronger, endorse AE2's position — even fully — and explain why. Equally, if AE1's case is stronger, endorse AE1. Only stay near the middle if the evidence is genuinely balanced. Avoid 'both have a point' language unless you can defend it specifically. Format: AGREEMENTS: / REFINEMENT:","required_skill":"debate","depends_on":["t2","t3","t4"],"perspective":"ae3: synthesis 1"}
  ],
  "max_workers": 3
}

Notice all three `depends_on` are identical — the previous round's outputs
only. That is deliberate: it gives every agent the same starting context.
Notice also that NO description re-asserts the agent's initial role/stance
as if it were still binding — the agents are explicitly told the previous
role was a starting point, not a position to defend.
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


def _parse_plan(
    raw: str,
    known_skills: set[str] | None = None,
    default_goal: str | None = None,
    external_ids: set[str] | None = None,
) -> TaskPlan:
    """Parse and validate a planner LLM response into a TaskPlan.

    Raises ValueError on malformed JSON or invalid plan structure: unknown
    dependencies, duplicate IDs, empty subtasks list, or — when
    `known_skills` is provided — `required_skill` values that aren't in
    the catalog.

    `default_goal` is used to backfill a missing `goal` field — useful for
    extension plans, which reuse the original plan's goal and where models
    sometimes omit the field.

    `external_ids` are subtask IDs that exist in a previous plan and are
    therefore valid `depends_on` targets even though they aren't declared
    in this plan — used when validating extension plans whose new subtasks
    depend on the original plan's debate outputs.

    The skill check lives here (not in the executor) so the Planner can
    detect hallucinated skills and re-prompt the LLM with a corrective
    message before burning a full execution attempt.
    """
    data = json.loads(raw)
    if default_goal is not None and not data.get("goal"):
        data["goal"] = default_goal
    plan = TaskPlan(**data)

    ids = [t.id for t in plan.subtasks]
    if not ids:
        raise ValueError("Plan contains no subtasks")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate subtask IDs: {ids}")
    known = set(ids) | (external_ids or set())
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
            for tag in ("ae1", "ae2", "ae3"):
                if persp.startswith(tag):
                    latest_per_agent[tag] = t.id
                    break
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
                ext = _parse_plan(
                    raw,
                    known_skills=known_skills,
                    default_goal=original.goal,
                    external_ids=set(existing_ids),
                )
                # Reject ID collisions with the original plan.
                clash = {t.id for t in ext.subtasks} & set(existing_ids)
                if clash:
                    raise ValueError(
                        f"Extension reuses existing IDs {sorted(clash)}; "
                        f"prefix new IDs with 'x'."
                    )
                # Reject sequential chains inside the extension. The new
                # subtasks must run in parallel — i.e. none of them can depend
                # on another new subtask. This guarantees every agent reacts
                # to the same shared snapshot of the previous round.
                new_ids = {t.id for t in ext.subtasks}
                for t in ext.subtasks:
                    chained = set(t.depends_on) & new_ids
                    if chained:
                        raise ValueError(
                            f"Subtask {t.id!r} depends on other new "
                            f"subtask(s) {sorted(chained)}. The extension "
                            f"must be PARALLEL: every new subtask must "
                            f"depend ONLY on the previous round's outputs, "
                            f"not on other new subtasks."
                        )
                logger.info(
                    "Planner produced %d extension subtasks for consensus push",
                    len(ext.subtasks),
                )
                return ext
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Extension plan parse failed (attempt %d): %s\nRaw response: %s",
                    attempt + 1, e, raw[:1500],
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That response was invalid: {e}. Return ONLY valid JSON "
                        f"with the exact top-level shape "
                        f'{{"goal": "...", "subtasks": [...], "max_workers": <int>}}. '
                        f"All three top-level fields are MANDATORY. Do NOT wrap the "
                        f"plan in any other object."
                    ),
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
