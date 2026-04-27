"""AgenticOrchestrator — the agentic replacement for FlowManager.

This is the sub-phase 2c production path. Instead of a hardcoded
normalize→roles→debate pipeline, we:

1. Consult the AgentRegistry to discover currently-available workers.
2. Ask the Planner LLM to decompose the user request into a TaskPlan (DAG).
3. Pre-flight: for each skill the plan requires, if the peak concurrent
   demand exceeds the number of registered workers for that skill, spawn
   extra specialized workers via WorkerSpawner until demand is covered.
4. Hand the plan to PlanExecutor, which walks the DAG and dispatches each
   subtask to a worker (round-robin when a skill has several workers).
5. After execution, if the plan was a multi-round debate, evaluate consensus
   between the final positions of each agent. If not reached, ask the
   Planner for an EXTENSION PLAN (more synthesis rounds) and execute it.
   Repeat up to MAX_CONSENSUS_EXTENSIONS times. This makes the iteration
   itself an agentic decision rather than a hardcoded loop.
6. Return the final synthesized verdict — by convention, the output of
   whichever subtask has no successors is treated as the verdict.

Progress streams via a ProgressCallback so the existing SSE path in
`executor.py` continues to feed the frontend timeline.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from agents.orchestrator.agent_registry import AgentRegistry
from agents.orchestrator.plan_executor import PlanExecutor, ProgressCallback
from agents.orchestrator.planner import Planner
from agents.orchestrator.worker_spawner import WorkerSpawner
from common.config import settings
from common.llm_provider import llm_complete
from common.models import SubTask, TaskPlan

MAX_CONSENSUS_EXTENSIONS = 3

logger = logging.getLogger(__name__)


SYNTHESIZE_PROMPT = """\
You are the orchestrator summarizing the results of an agentic task plan.
Produce a clear, human-readable final answer to the user's original request,
drawing on the outputs of all subtasks.

Original request:
{user_input}

Subtask outputs:
{subtask_outputs}

Write the final answer in Spanish, using markdown where helpful. Do not
mention the subtask IDs — present the answer as if the reader never saw
the plan."""


CONSENSUS_CHECK_PROMPT = """\
Three debate agents have just exchanged their latest arguments in a structured
deliberation:
 - AE1 opened by advocating one side
 - AE2 opened by advocating the opposing side
 - AE3 is an independent evaluator with no assigned stance

Your job is to evaluate whether they have substantively converged on a
shared, evidence-driven answer — and to score each agent's CURRENT position
on a continuous axis where:
   0.0 = endorses AE1's opening stance
   1.0 = endorses AE2's opening stance
   0.5 = no clear position either way (genuinely balanced or evasive)

CRITICAL — read this before scoring:
1. Score where each agent IS NOW, not where they started. If AE1 has
   genuinely changed sides and now endorses AE2's stance, AE1's position
   should be HIGH (close to 1.0), not low. The same applies in reverse.
2. Look for explicit side-shift markers like "You changed my mind on X",
   "I now agree with...", "the evidence on X is decisive". Treat these as
   strong signals to move that agent toward the side they shifted to.
3. Do NOT reward forced centrism. Two patterns to flag:
   a. "Both have a point" with no specific commitments — this is evasion,
      not consensus. agreement_score should be LOW even if all three sit
      near 0.5, because there is no real shared position.
   b. Wishy-washy "common ground" that doesn't actually answer the
      question — same treatment.
4. Reward HONEST CONVERGENCE TOWARD A SIDE. If the evidence presented
   clearly favours one side and all three agents have moved toward it
   (positions clustered near 0.0 OR clustered near 1.0), that is a high
   agreement_score — possibly higher than a clustering near 0.5.
5. Only assign agreement_score >= 0.75 when the three latest positions
   actually agree on a SUBSTANTIVE answer to the original question, not
   just on procedural tone. The shared_points list must contain concrete
   claims, not platitudes.

HARD CONSISTENCY RULE — agreement_score MUST track positions:
The `agreement_score` is the geometric clustering of `positions`, NOT a
separate "tone" or "vibes" metric. Use this scale, anchored to the spread
between the highest and lowest position values (max − min):
   spread <= 0.15  →  agreement_score in [0.85, 1.00]   (tight cluster)
   spread <= 0.25  →  agreement_score in [0.70, 0.85]
   spread <= 0.40  →  agreement_score in [0.50, 0.70]
   spread <= 0.60  →  agreement_score in [0.30, 0.50]
   spread >  0.60  →  agreement_score in [0.00, 0.30]   (clearly apart)
If you see one agent at 0.05 and another at 0.92 (spread > 0.85), the
agreement_score CANNOT be high — it must be in [0.00, 0.30] no matter how
politely worded the messages were. Polite tone is not consensus.

You may bias slightly within each band based on the *quality* of the
agreement (concrete shared_points push toward the high end of the band;
vague platitudes push toward the low end). But you must stay inside the
band the spread dictates.

AE1 latest position ({ae1_perspective}):
{ae1_text}

AE2 latest position ({ae2_perspective}):
{ae2_text}

AE3 latest position ({ae3_perspective}):
{ae3_text}

Return ONLY valid JSON with this exact shape:
{{
  "agreement_score": <float 0.0 to 1.0 — overall convergence across the three>,
  "positions": {{
    "ae1": <float 0.0 to 1.0>,
    "ae2": <float 0.0 to 1.0>,
    "ae3": <float 0.0 to 1.0>
  }},
  "shared_points": ["concrete claim 1", "..."],
  "remaining_disagreements": ["concrete disagreement 1", "..."],
  "reason": "<one-sentence rationale that mentions whether convergence is toward AE1's side, AE2's side, the centre, or unclear>"
}}

agreement_score interpretation:
- >= 0.75: substantive consensus on the actual question, no further rounds needed
- 0.50 - 0.74: partial convergence, one re-evaluation round could close the gap
- < 0.50: still meaningfully apart, OR all three are using vague centrist
  language without committing to a real answer (vagueness is NOT consensus)

The `positions` object is what the UI uses to plot how the agents are moving
across rounds — be diligent and consistent. A position score reflects
endorsement of a side based on the agent's current claims; movement across
rounds is what the user is watching for."""


def _peak_concurrent_demand(plan: TaskPlan) -> dict[str, int]:
    """Estimate how many workers of each skill the plan may need at once.

    A simple upper bound: walk the DAG one ready-set at a time and track
    the max count of any single skill in any ready set. This matches
    PlanExecutor's execution pattern.
    """
    results_ready: set[str] = set()
    pending = list(plan.subtasks)
    peak: Counter[str] = Counter()

    while pending:
        ready = [t for t in pending if all(d in results_ready for d in t.depends_on)]
        if not ready:
            # Deadlock — bail and let PlanExecutor raise the real error.
            return dict(peak)
        batch_counts = Counter(t.required_skill for t in ready)
        for skill, c in batch_counts.items():
            peak[skill] = max(peak[skill], c)
        for t in ready:
            results_ready.add(t.id)
            pending.remove(t)

    return dict(peak)


class AgenticOrchestrator:
    """Plan → (spawn if needed) → execute → synthesize."""

    def __init__(
        self,
        registry: AgentRegistry,
        spawner: WorkerSpawner,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.registry = registry
        self.spawner = spawner
        self.progress = progress or ProgressCallback()
        self.planner = Planner(model=settings.orchestrator_model)
        self.executor = PlanExecutor(registry=registry, progress=self.progress)
        # Track workers we spawned for this run so we can tear them down.
        self._spawned_this_run: list[str] = []

    async def run(self, user_input: str) -> str:
        """Execute one agentic run; returns the final answer text."""
        try:
            # 1. Discover workers
            await self.progress.on_progress(
                "discover", "Consultando registry de workers..."
            )
            workers = await self.registry.all_workers()
            catalog = [
                {
                    "agent_id": w.agent_id,
                    "url": w.url,
                    "skills": w.card.get("skills") or [],
                }
                for w in workers
            ]

            # 2. Plan
            await self.progress.on_progress(
                "plan", "Generando plan de subtareas con LLM..."
            )
            plan = await self.planner.create_plan(user_input, catalog)
            await self.progress.on_progress(
                "plan_ready",
                f"Plan generado: {len(plan.subtasks)} subtareas",
                {"plan": plan.model_dump()},
            )

            # 3. Pre-flight: spawn workers if peak concurrent demand exceeds supply
            await self._ensure_capacity(plan)

            # 4. Execute the DAG
            results = await self.executor.execute(plan)

            # 4b. Consensus check + agentic extension loop. Only meaningful for
            # multi-round debate plans; a no-op for factual pipelines.
            plan, results = await self._consensus_loop(plan, results, catalog)

            # 4c. If the deliberative plan ended without a format_verdict step
            # (Option-2 default), append one now and dispatch it to the
            # feedback agent. This makes the feedback worker visible in the
            # frontend timeline and produces a single canonical sink for
            # synthesis.
            plan, results = await self._finalize_with_feedback(plan, results)

            # 5. Synthesize
            return await self._synthesize(user_input, plan, results)

        finally:
            # Best-effort cleanup of workers we spawned for this run.
            for agent_id in self._spawned_this_run:
                try:
                    await self.spawner.teardown(agent_id)
                except Exception as e:
                    logger.warning(
                        "Failed to teardown spawned worker %s: %s", agent_id, e
                    )
            self._spawned_this_run.clear()

    async def _consensus_loop(
        self,
        plan: TaskPlan,
        results: dict[str, str],
        catalog: list[dict],
    ) -> tuple[TaskPlan, dict[str, str]]:
        """Evaluate consensus on the latest debate exchange; extend if needed.

        Returns the (possibly augmented) plan and merged results. The plan's
        subtasks list is concatenated with extension subtasks so downstream
        synthesis sees the full DAG.
        """
        # Identify the LATEST debate subtask per agent in the current plan.
        latest = self._latest_debate_per_agent(plan)
        if not (latest.get("ae1") and latest.get("ae2")):
            # Plan was not a debate plan — nothing to evaluate.
            return plan, results

        merged_plan = plan
        merged_results = results
        # Preserve DAG order: ordered list, plus a lookup by id.
        merged_ordered: list[SubTask] = list(plan.subtasks)
        all_subtasks: dict[str, SubTask] = {t.id: t for t in plan.subtasks}

        for attempt in range(MAX_CONSENSUS_EXTENSIONS):
            agent_texts: dict[str, str] = {}
            agent_perspectives: dict[str, str] = {}
            for tag in ("ae1", "ae2", "ae3"):
                tid = latest.get(tag)
                if not tid:
                    continue
                task = all_subtasks[tid]
                agent_texts[tag] = merged_results.get(tid, "")
                agent_perspectives[tag] = task.perspective or tag.upper()

            score, reason, positions, shared_points, disagreements = (
                await self._check_consensus(agent_texts, agent_perspectives)
            )
            # Emit a dedicated event so the frontend can plot how the agents
            # have moved on the AE1↔AE2 axis after this round, including the
            # substantive points the model identified as shared / unresolved.
            await self.progress.on_progress(
                "agent_positions",
                f"Posiciones tras ronda {attempt}",
                {
                    "round": attempt,
                    "positions": positions,
                    "agreement_score": score,
                    "shared_points": shared_points,
                    "remaining_disagreements": disagreements,
                    "subtask_ids": {
                        tag: latest.get(tag) for tag in ("ae1", "ae2", "ae3")
                    },
                },
            )
            await self.progress.on_progress(
                "consensus_check",
                f"Consenso evaluado: score={score:.2f}",
                {
                    "agreement_score": score,
                    "reason": reason,
                    "extension_attempt": attempt,
                    "positions": positions,
                    "shared_points": shared_points,
                    "remaining_disagreements": disagreements,
                    "round": attempt,
                },
            )

            if score >= 0.75:
                await self.progress.on_progress(
                    "consensus", f"Consenso alcanzado (score={score:.2f})"
                )
                return merged_plan, merged_results

            if attempt + 1 >= MAX_CONSENSUS_EXTENSIONS:
                await self.progress.on_progress(
                    "no_consensus",
                    f"Máximo de extensiones alcanzado sin consenso (score={score:.2f})",
                )
                return merged_plan, merged_results

            # Ask the planner for an extension plan that pushes for synthesis.
            await self.progress.on_progress(
                "extend_plan",
                f"Sin consenso (score={score:.2f}); pidiendo plan de síntesis al planner...",
                {"agreement_score": score, "reason": reason},
            )
            try:
                extension = await self.planner.extend_for_consensus(
                    original=merged_plan,
                    results=merged_results,
                    workers=catalog,
                    consensus_reason=reason,
                )
            except Exception as e:
                logger.warning("Extension planning failed: %s", e)
                await self.progress.on_progress(
                    "extend_failed",
                    f"No se pudo extender el plan: {e}",
                    {"error": str(e)},
                )
                return merged_plan, merged_results

            # Build the merged plan FIRST and emit it, so the frontend sees
            # the new extension nodes (as pending) before any subtask_dispatch
            # event references them. Without this the graph can't render the
            # newly arriving x* nodes.
            merged_ordered_pending = list(merged_ordered) + list(extension.subtasks)
            merged_plan = TaskPlan(
                goal=merged_plan.goal,
                subtasks=list(merged_ordered_pending),
                max_workers=max(merged_plan.max_workers, extension.max_workers),
            )
            await self.progress.on_progress(
                "plan_ready",
                f"Plan extendido: ahora {len(merged_plan.subtasks)} subtareas",
                {"plan": merged_plan.model_dump()},
            )

            # Make sure we have the workers the extension needs.
            await self._ensure_capacity(extension)

            # Execute extension on top of existing context.
            ext_results = await self.executor.execute(
                extension,
                prior_results=merged_results,
                prior_subtasks=all_subtasks,
            )
            merged_results = {**merged_results, **ext_results}
            for t in extension.subtasks:
                all_subtasks[t.id] = t
                merged_ordered.append(t)

            latest = self._latest_debate_per_agent(merged_plan)
            if not (latest.get("ae1") and latest.get("ae2")):
                return merged_plan, merged_results

        return merged_plan, merged_results

    @staticmethod
    def _latest_debate_per_agent(plan: TaskPlan) -> dict[str, str]:
        """Return {'ae1': id, 'ae2': id, 'ae3': id} for the plan.

        "Latest" is the last debate subtask whose perspective starts with
        the agent tag in DAG order (assumed to be the order in plan.subtasks,
        which mirrors the planner's output). Missing agents are absent from
        the returned dict.
        """
        out: dict[str, str] = {}
        for t in plan.subtasks:
            if t.required_skill != "debate":
                continue
            persp = (t.perspective or "").strip().lower()
            for tag in ("ae1", "ae2", "ae3"):
                if persp == tag or persp.startswith(f"{tag}:") or persp.startswith(f"{tag} "):
                    out[tag] = t.id
                    break
        return out

    async def _check_consensus(
        self,
        agent_texts: dict[str, str],
        agent_perspectives: dict[str, str],
    ) -> tuple[float, str, dict[str, float], list[str], list[str]]:
        """LLM-graded consensus score, per-agent positions, and the
        substantive points the model identified as shared / unresolved.

        Returns (agreement_score, reason, positions, shared_points,
        remaining_disagreements). `positions` maps each agent tag to a float
        in [0, 1] representing where that agent currently sits on the AE1↔AE2
        axis (0 = AE1 stance, 1 = AE2 stance, 0.5 = neutral). Missing
        positions are filled with 0.5 as a safe default.
        """
        prompt = CONSENSUS_CHECK_PROMPT.format(
            ae1_perspective=agent_perspectives.get("ae1", "AE1"),
            ae2_perspective=agent_perspectives.get("ae2", "AE2"),
            ae3_perspective=agent_perspectives.get("ae3", "AE3"),
            ae1_text=agent_texts.get("ae1", ""),
            ae2_text=agent_texts.get("ae2", ""),
            ae3_text=agent_texts.get("ae3", ""),
        )
        try:
            raw = await llm_complete(
                model=settings.orchestrator_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw)
            score = float(data.get("agreement_score", 0.0))
            reason = str(data.get("reason", "")).strip() or "no reason provided"
            raw_positions = data.get("positions") or {}
            positions: dict[str, float] = {}
            for tag in ("ae1", "ae2", "ae3"):
                v = raw_positions.get(tag)
                try:
                    positions[tag] = max(0.0, min(1.0, float(v))) if v is not None else 0.5
                except (TypeError, ValueError):
                    positions[tag] = 0.5
            shared = [str(p).strip() for p in (data.get("shared_points") or []) if p]
            disagreements = [
                str(p).strip() for p in (data.get("remaining_disagreements") or []) if p
            ]
            score = max(0.0, min(1.0, score))

            # Backstop: enforce that agreement_score actually tracks the
            # geometric spread of positions, even if the LLM ignored the
            # consistency rule in the prompt. Otherwise the user sees
            # contradictions like "agreement 0.92" while AE1 sits at 0.05
            # and AE2 at 0.92 (positions clearly apart).
            pos_values = [v for v in positions.values() if v is not None]
            if len(pos_values) >= 2:
                spread = max(pos_values) - min(pos_values)
                # Same bands as the prompt; we cap the LLM's score to the
                # top of the band the spread dictates so it can never claim
                # more agreement than the geometry shows.
                if spread <= 0.15:
                    cap = 1.00
                elif spread <= 0.25:
                    cap = 0.85
                elif spread <= 0.40:
                    cap = 0.70
                elif spread <= 0.60:
                    cap = 0.50
                else:
                    cap = 0.30
                if score > cap:
                    logger.info(
                        "Capping agreement_score %.2f → %.2f because position "
                        "spread is %.2f (positions=%s)",
                        score, cap, spread, positions,
                    )
                    score = cap
                    reason = (
                        f"{reason} [score capped to {cap:.2f} by orchestrator "
                        f"because position spread {spread:.2f} is too wide for "
                        f"a higher agreement score]"
                    )

            return (
                score,
                reason,
                positions,
                shared,
                disagreements,
            )
        except Exception as e:
            logger.warning("Consensus check failed, defaulting to no-consensus: %s", e)
            return (
                0.0,
                f"consensus check error: {e}",
                {"ae1": 0.0, "ae2": 1.0, "ae3": 0.5},
                [],
                [],
            )

    async def _ensure_capacity(self, plan: TaskPlan) -> None:
        """Spawn extra workers if peak concurrent demand exceeds supply."""
        peak = _peak_concurrent_demand(plan)
        for skill, needed in peak.items():
            have = await self.registry.find_by_skill(skill)
            missing = needed - len(have)
            if missing <= 0:
                continue
            await self.progress.on_progress(
                "spawn",
                f"Spawneando {missing} worker(s) extra para skill '{skill}'",
                {"skill": skill, "needed": needed, "have": len(have)},
            )
            for i in range(missing):
                # Reuse the "specialized" worker type. Specialized advertises
                # the `debate` skill, which is the only skill our current plans
                # ever over-demand. If future plans need new skill types, we'd
                # need a way to tell the spawner which worker *module* to launch.
                suffix = len(have) + i + 1
                agent_id = f"dyn_{skill}_{suffix}"
                try:
                    await self.spawner.spawn(agent_id)
                    self._spawned_this_run.append(agent_id)
                except Exception as e:
                    logger.exception("Spawn failed for %s", agent_id)
                    await self.progress.on_progress(
                        "spawn_failed",
                        f"No se pudo spawnear worker {agent_id}: {e}",
                        {"agent_id": agent_id, "error": str(e)},
                    )
                    raise

    async def _finalize_with_feedback(
        self,
        plan: TaskPlan,
        results: dict[str, str],
    ) -> tuple[TaskPlan, dict[str, str]]:
        """Append + execute a final format_verdict subtask if a feedback worker
        is available and the plan doesn't already end with one.

        This brings the feedback agent back into the deliberative path under
        Option 2 (where extensions don't include format_verdict themselves)
        so it shows up as a real node in the frontend graph and timeline.
        """
        # Skip if the plan already has a format_verdict step (e.g. factual
        # pipelines, or a planner that emitted one explicitly).
        if any(t.required_skill == "format_verdict" for t in plan.subtasks):
            return plan, results

        # Need a worker that advertises format_verdict.
        feedback_workers = await self.registry.find_by_skill("format_verdict")
        if not feedback_workers:
            return plan, results

        # Only run the feedback step when the plan actually had a debate; for
        # purely factual/single-answer plans the existing _synthesize path is
        # already sufficient.
        latest = self._latest_debate_per_agent(plan)
        if not (latest.get("ae1") and latest.get("ae2")):
            return plan, results

        # Pick a unique id that doesn't collide with existing ones.
        existing_ids = {t.id for t in plan.subtasks}
        final_id = "final_verdict"
        suffix = 1
        while final_id in existing_ids:
            suffix += 1
            final_id = f"final_verdict_{suffix}"

        # Depend on every agent that participated, including the neutral
        # mediator if it was part of the debate.
        final_deps = [latest[tag] for tag in ("ae1", "ae2", "ae3") if latest.get(tag)]

        final_task = SubTask(
            id=final_id,
            description=(
                "Sintetiza un veredicto final claro y bien estructurado a "
                "partir de los argumentos finales de los agentes. Resalta "
                "puntos de acuerdo, desacuerdos residuales y la conclusión "
                "unificada en castellano."
            ),
            required_skill="format_verdict",
            depends_on=final_deps,
            perspective=None,
        )

        final_plan_segment = TaskPlan(
            goal=plan.goal,
            subtasks=[final_task],
            max_workers=plan.max_workers,
        )

        # Update the merged plan view + emit so the frontend shows the new
        # feedback node before it starts running.
        merged_plan = TaskPlan(
            goal=plan.goal,
            subtasks=[*plan.subtasks, final_task],
            max_workers=plan.max_workers,
        )
        await self.progress.on_progress(
            "plan_ready",
            f"Plan finalizado: {len(merged_plan.subtasks)} subtareas",
            {"plan": merged_plan.model_dump()},
        )

        # Execute just the final subtask, threading prior context so
        # format_verdict sees the latest debate outputs as deps. If the
        # feedback agent fails (rate limit, timeout, etc.), fall back to
        # the multi-sink synthesizer so the user still gets an answer.
        prior_subtasks = {t.id: t for t in plan.subtasks}
        try:
            final_results = await self.executor.execute(
                final_plan_segment,
                prior_results=results,
                prior_subtasks=prior_subtasks,
            )
        except Exception as e:
            logger.warning("Final format_verdict step failed: %s", e)
            await self.progress.on_progress(
                "finalize_failed",
                f"Feedback agent falló, usaré síntesis interna: {e}",
                {"error": str(e)},
            )
            return plan, results
        return merged_plan, {**results, **final_results}

    async def _synthesize(
        self,
        user_input: str,
        plan: TaskPlan,
        results: dict[str, str],
    ) -> str:
        """Collapse subtask outputs into a single user-facing answer.

        Strategy: if any subtask has no successors, its output IS the final
        answer (common when the plan ends with a format_verdict step).
        Otherwise we ask the orchestrator's LLM to synthesize from all
        subtask outputs.
        """
        # Find subtasks that are not depended on by anyone = sinks.
        depended_on: set[str] = {
            d for t in plan.subtasks for d in t.depends_on
        }
        sinks = [t for t in plan.subtasks if t.id not in depended_on]

        if len(sinks) == 1:
            sink = sinks[0]
            await self.progress.on_progress(
                "synthesize",
                f"Veredicto final tomado de subtarea '{sink.id}'",
                {"source_subtask": sink.id},
            )
            return results.get(sink.id, "").strip() or self._fallback_summary(results)

        # Multiple sinks: ask the LLM to combine them.
        await self.progress.on_progress(
            "synthesize", "Sintetizando veredicto final..."
        )
        formatted = "\n\n".join(
            f"### {tid}\n{results[tid]}"
            for tid in results
        )
        prompt = SYNTHESIZE_PROMPT.format(
            user_input=user_input, subtask_outputs=formatted
        )
        return await llm_complete(
            model=settings.orchestrator_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1200,
        )

    @staticmethod
    def _fallback_summary(results: dict[str, str]) -> str:
        """Bare-bones fallback if the sink subtask produced empty text."""
        return "\n\n".join(f"**{k}**\n{v}" for k, v in results.items())
