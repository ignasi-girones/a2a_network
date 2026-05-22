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
from agents.orchestrator.consensus_metrics import (
    ConsensusMetrics,
    compute_metrics,
)
from agents.orchestrator.plan_executor import PlanExecutor, ProgressCallback
from agents.orchestrator.planner import Planner
from agents.orchestrator.worker_spawner import WorkerSpawner
from common.config import settings
from common.llm_provider import llm_complete, llm_embed
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


CONSENSUS_TEXTS_PROMPT = """\
Three debate agents have just exchanged their latest arguments in a
structured deliberation. Your ONLY job is to extract two short bullet lists
of concrete textual claims from the latest exchange:
  - shared_points: substantive points the agents have come to agree on,
    using their actual claims (not platitudes like "both have a point").
  - remaining_disagreements: concrete claims where the agents still disagree.

You are NOT scoring convergence. The orchestrator computes the agreement
score and per-agent positions empirically from the texts (cosine similarity
of embeddings to the AE1/AE2 opening anchors, dispersion, movement, and
explicit concession markers). Do NOT speculate on numerical scores here.

AE1 latest position ({ae1_perspective}):
{ae1_text}

AE2 latest position ({ae2_perspective}):
{ae2_text}

AE3 latest position ({ae3_perspective}):
{ae3_text}

Return ONLY valid JSON with EXACTLY this shape:
{{
  "shared_points": ["concrete shared claim 1", "concrete shared claim 2"],
  "remaining_disagreements": ["concrete disagreement 1", "concrete disagreement 2"],
  "reason": "<one short sentence summarising what the agents agreed on and what they still disagree about>"
}}

Use plain Spanish in the bullet content (translate from English if the
debate happened in English). If a list is empty (genuinely no shared points
or no remaining disagreements), return an empty array."""


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

    async def run(self, user_input: str, *, extra_context: str | None = None) -> str:
        """Execute one agentic run; returns the final answer text.

        ``extra_context`` carries text extracted from user-uploaded attachments.
        It is threaded to the planner and plan executor so agents can reference
        the material during the debate.
        """
        self._extra_context = extra_context
        self.executor.extra_context = extra_context
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

        # Capture the AE1 / AE2 OPENING texts as anchors for the empirical
        # axis. By definition AE1's opening = position 0.0 and AE2's opening
        # = position 1.0; every subsequent text is placed by cosine similarity
        # against those two anchors. The anchors are computed exactly ONCE
        # and reused across rounds so the axis stays stable.
        ae1_anchor_text = merged_results.get(latest["ae1"], "")
        ae2_anchor_text = merged_results.get(latest["ae2"], "")
        ae1_anchor_emb: list[float] = []
        ae2_anchor_emb: list[float] = []
        await self.progress.on_progress(
            "embedding",
            f"Anclando eje con {settings.embedding_model} (aperturas AE1/AE2)",
            {
                "phase": "anchors",
                "model": settings.embedding_model,
                "n_texts": 2,
            },
        )
        try:
            anchor_embs = await llm_embed([ae1_anchor_text, ae2_anchor_text])
            ae1_anchor_emb, ae2_anchor_emb = anchor_embs[0], anchor_embs[1]
            logger.info(
                "Consensus anchors embedded with model %s (dim=%d)",
                settings.embedding_model,
                len(ae1_anchor_emb),
            )
            await self.progress.on_progress(
                "embedding_done",
                f"Anclas listas (dim={len(ae1_anchor_emb)})",
                {
                    "phase": "anchors",
                    "model": settings.embedding_model,
                    "dim": len(ae1_anchor_emb),
                },
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.warning(
                "Embedding the consensus anchors failed using model %s — %s. "
                "Empirical metrics will degrade to default positions and score 0.",
                settings.embedding_model,
                err,
            )
            await self.progress.on_progress(
                "embedding_failed",
                f"Embeddings cayeron con {settings.embedding_model}",
                {
                    "phase": "anchors",
                    "model": settings.embedding_model,
                    "error": err,
                },
            )

        prev_positions: dict[str, float] | None = None

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

            metrics, shared_points, disagreements, reason = await self._check_consensus(
                agent_texts=agent_texts,
                agent_perspectives=agent_perspectives,
                ae1_anchor_emb=ae1_anchor_emb,
                ae2_anchor_emb=ae2_anchor_emb,
                prev_positions=prev_positions,
            )
            score = metrics.agreement_score
            positions = metrics.positions
            prev_positions = dict(positions)

            # Emit a dedicated event so the frontend can plot how the agents
            # have moved on the AE1↔AE2 axis after this round, including the
            # raw component metrics for full auditability.
            await self.progress.on_progress(
                "agent_positions",
                f"Posiciones tras ronda {attempt}",
                {
                    "round": attempt,
                    "positions": positions,
                    "agreement_score": score,
                    "shared_points": shared_points,
                    "remaining_disagreements": disagreements,
                    "components": metrics.components,
                    "movement": metrics.movement,
                    "concessions": metrics.concessions,
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
                    "components": metrics.components,
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
        *,
        agent_texts: dict[str, str],
        agent_perspectives: dict[str, str],
        ae1_anchor_emb: list[float],
        ae2_anchor_emb: list[float],
        prev_positions: dict[str, float] | None,
    ) -> tuple[ConsensusMetrics, list[str], list[str], str]:
        """Compute the empirical consensus metrics for the latest exchange
        plus an LLM-extracted summary of shared/disagreement points.

        Returns (metrics, shared_points, remaining_disagreements, reason).

        The numeric components — `agreement_score`, per-agent positions,
        dispersion, movement, similarity, concession_score — all come from
        `consensus_metrics.compute_metrics`, which is reproducible and
        independent of any LLM judgement. The LLM is only invoked to extract
        the textual lists of shared points and remaining disagreements,
        because those need natural-language understanding and have no closed
        algorithmic form.

        If embeddings or the LLM call fail, we degrade gracefully: positions
        fall back to anchor-defaults (AE1=0, AE2=1, AE3=0.5), score is 0,
        and the textual lists end up empty.
        """
        # 1. Numeric metrics from embeddings. Only feasible if we have anchors.
        metrics: ConsensusMetrics | None = None
        if ae1_anchor_emb and ae2_anchor_emb:
            try:
                tags = [t for t in ("ae1", "ae2", "ae3") if t in agent_texts]
                texts_in_order = [agent_texts[t] for t in tags]
                await self.progress.on_progress(
                    "embedding",
                    f"Embedeando {len(tags)} texto(s) con {settings.embedding_model}",
                    {
                        "phase": "round",
                        "model": settings.embedding_model,
                        "n_texts": len(tags),
                        "agent_tags": tags,
                    },
                )
                embs = await llm_embed(texts_in_order)
                embeddings_by_tag = {tag: embs[i] for i, tag in enumerate(tags)}
                await self.progress.on_progress(
                    "embedding_done",
                    f"Embeddings listos para {len(tags)} agente(s)",
                    {
                        "phase": "round",
                        "model": settings.embedding_model,
                        "dim": len(embs[0]) if embs else 0,
                    },
                )
                metrics = compute_metrics(
                    embeddings=embeddings_by_tag,
                    texts={t: agent_texts[t] for t in tags},
                    ae1_anchor_emb=ae1_anchor_emb,
                    ae2_anchor_emb=ae2_anchor_emb,
                    previous_positions=prev_positions,
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                logger.warning(
                    "Empirical consensus metrics failed (%s). "
                    "Falling back to default positions and zero score.",
                    err,
                )
                await self.progress.on_progress(
                    "embedding_failed",
                    f"Embeddings cayeron en evaluación: {err}",
                    {
                        "phase": "round",
                        "model": settings.embedding_model,
                        "error": err,
                    },
                )

        if metrics is None:
            # Fallback: anchor-default positions (AE1=0, AE2=1, AE3=0.5),
            # zero score. The deliberation will exhaust its round budget
            # without ever claiming consensus, which is the safe default.
            fallback_positions = {
                "ae1": 0.0,
                "ae2": 1.0,
                "ae3": 0.5,
            }
            fallback_positions = {
                k: v for k, v in fallback_positions.items() if k in agent_texts
            }
            metrics = ConsensusMetrics(
                positions=fallback_positions,
                dispersion=1.0,
                pairwise_similarity=0.0,
                movement={k: 0.0 for k in fallback_positions},
                movement_score=0.0,
                concessions={k: 0 for k in fallback_positions},
                concession_score=0.0,
                agreement_score=0.0,
                components={"fallback": True},
            )

        # 2. LLM call ONLY for the textual summary — shared_points,
        # remaining_disagreements, and a brief reason. The score and
        # positions are NOT requested here; they come from `metrics`.
        shared: list[str] = []
        disagreements: list[str] = []
        reason = ""
        try:
            prompt = CONSENSUS_TEXTS_PROMPT.format(
                ae1_perspective=agent_perspectives.get("ae1", "AE1"),
                ae2_perspective=agent_perspectives.get("ae2", "AE2"),
                ae3_perspective=agent_perspectives.get("ae3", "AE3"),
                ae1_text=agent_texts.get("ae1", ""),
                ae2_text=agent_texts.get("ae2", ""),
                ae3_text=agent_texts.get("ae3", ""),
            )
            raw = await llm_complete(
                model=settings.orchestrator_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw)
            shared = [
                str(p).strip()
                for p in (data.get("shared_points") or [])
                if p
            ]
            disagreements = [
                str(p).strip()
                for p in (data.get("remaining_disagreements") or [])
                if p
            ]
            reason = str(data.get("reason", "")).strip()
        except Exception as e:
            logger.warning("Textual consensus extraction failed: %s", e)
            reason = f"textual extraction error: {e}"

        # Compose a short auto-explanation of the score from the components,
        # so the UI can show it and the user can audit which factor dominates.
        if not reason:
            reason = "no reason provided"
        comp = metrics.components
        explanation_parts = []
        if "dispersion" in comp:
            explanation_parts.append(
                f"dispersión={comp['dispersion']:.2f}"
            )
        if "pairwise_similarity" in comp:
            explanation_parts.append(
                f"similitud={comp['pairwise_similarity']:.2f}"
            )
        if "movement_score" in comp:
            explanation_parts.append(
                f"movimiento={comp['movement_score']:.2f}"
            )
        if "concession_score" in comp:
            explanation_parts.append(
                f"concesiones={comp['concession_score']:.2f}"
            )
        if explanation_parts:
            reason = f"{reason} [score={metrics.agreement_score:.2f} • " + ", ".join(explanation_parts) + "]"

        return metrics, shared, disagreements, reason

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
