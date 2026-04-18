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
5. Return the final synthesized verdict — by convention, the output of
   whichever subtask has no successors is treated as the verdict.

Progress streams via a ProgressCallback so the existing SSE path in
`executor.py` continues to feed the frontend timeline.
"""

from __future__ import annotations

import logging
from collections import Counter

from agents.orchestrator.agent_registry import AgentRegistry
from agents.orchestrator.plan_executor import PlanExecutor, ProgressCallback
from agents.orchestrator.planner import Planner
from agents.orchestrator.worker_spawner import WorkerSpawner
from common.config import settings
from common.llm_provider import llm_complete
from common.models import TaskPlan

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
