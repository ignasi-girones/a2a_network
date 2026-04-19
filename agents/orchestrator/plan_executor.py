"""PlanExecutor — walks a TaskPlan DAG and dispatches subtasks to workers.

Execution model:
- Every tick, select all subtasks whose dependencies are already complete
  ("ready set").
- Dispatch the whole ready set in parallel via `asyncio.gather`.
- When the set is empty but `pending` isn't, raise — that means the DAG has
  a cycle or an unresolvable dependency.

Dispatch per subtask:
1. Look up a worker for `required_skill` in the AgentRegistry.
2. If the ready set contains N > 1 tasks sharing a skill, distribute them
   round-robin across all workers that advertise that skill (so two parallel
   "debate" tasks naturally go to AE1 and AE2 rather than both hitting AE1).
3. Build a prompt combining the SubTask description with the results of its
   dependencies (context propagation — the only way one subtask's output
   reaches another).
4. A2A-call the worker via `send_and_get_text` and store the response as
   `results[subtask.id]`.

Progress is streamed out through a ProgressCallback (same shape as the one
in `flow_manager.py`).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from agents.orchestrator.agent_registry import AgentRegistry
from common.a2a_helpers import create_a2a_client, send_and_get_text
from common.models import SubTask, TaskPlan, WorkerEntry

logger = logging.getLogger(__name__)


class ProgressCallback:
    """Minimal progress-event sink. Matches `flow_manager.ProgressCallback`."""

    async def on_progress(
        self, stage: str, message: str, data: dict | None = None
    ) -> None:
        pass


class PlanExecutionError(RuntimeError):
    """Raised when a plan cannot be executed (cycle, missing skill, etc.)."""


def _build_subtask_prompt(
    task: SubTask, dep_results: dict[str, str], goal: str
) -> str:
    """Compose the prompt a worker will actually receive.

    Includes:
    - The global plan goal (useful for debate workers that need framing).
    - The subtask's own description.
    - Perspective hint if present.
    - Formatted outputs of all `depends_on` predecessors.
    """
    parts = [f"Goal: {goal}", f"Task: {task.description}"]
    if task.perspective:
        parts.append(f"Perspective: {task.perspective}")
    if task.depends_on:
        parts.append("\nContext from previous steps:")
        for dep_id in task.depends_on:
            text = dep_results.get(dep_id, "").strip()
            parts.append(f"\n[{dep_id}]\n{text}")
    return "\n".join(parts)


class PlanExecutor:
    """Executes a TaskPlan against workers discovered in the AgentRegistry."""

    def __init__(
        self,
        registry: AgentRegistry,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.registry = registry
        self.progress = progress or ProgressCallback()

    async def execute(self, plan: TaskPlan) -> dict[str, str]:
        """Run the plan; return {subtask_id: text_output} for all subtasks."""
        results: dict[str, str] = {}
        pending: list[SubTask] = list(plan.subtasks)

        await self.progress.on_progress(
            "plan_start",
            f"Ejecutando plan con {len(pending)} subtareas",
            {"goal": plan.goal, "subtask_count": len(pending)},
        )

        while pending:
            ready = [
                t for t in pending if all(dep in results for dep in t.depends_on)
            ]
            if not ready:
                blocked = [t.id for t in pending]
                raise PlanExecutionError(
                    f"DAG deadlock: no subtasks ready but pending={blocked}"
                )

            await self.progress.on_progress(
                "plan_batch",
                f"Lanzando {len(ready)} subtareas en paralelo",
                {"subtask_ids": [t.id for t in ready]},
            )

            # Assign workers to ready tasks, round-robin per skill so parallel
            # same-skill tasks land on different workers.
            assignments = await self._assign_workers(ready)
            batch_outputs = await asyncio.gather(
                *[
                    self._execute_subtask(t, assignments[t.id], results, plan.goal)
                    for t in ready
                ],
                return_exceptions=True,
            )

            for task, output in zip(ready, batch_outputs):
                if isinstance(output, Exception):
                    await self.progress.on_progress(
                        "subtask_failed",
                        f"Subtarea {task.id} falló: {output}",
                        {
                            "subtask_id": task.id,
                            "required_skill": task.required_skill,
                            "perspective": task.perspective,
                            "error": str(output),
                        },
                    )
                    raise PlanExecutionError(
                        f"Subtask {task.id} failed: {output}"
                    ) from output
                results[task.id] = output
                pending.remove(task)
                # Full output travels in `text`; `output_preview` kept for
                # log views that want a single-line summary.
                await self.progress.on_progress(
                    "subtask_done",
                    f"Subtarea {task.id} completada",
                    {
                        "subtask_id": task.id,
                        "required_skill": task.required_skill,
                        "perspective": task.perspective,
                        "text": output,
                        "output_preview": output[:200],
                    },
                )

        await self.progress.on_progress(
            "plan_complete",
            "Plan completo",
            # Per-subtask full text is already in the subtask_done events;
            # here we just mark completion without re-sending it all.
            {"subtask_ids": list(results.keys())},
        )
        return results

    async def _execute_subtask(
        self,
        task: SubTask,
        worker: WorkerEntry,
        dep_results: dict[str, str],
        goal: str,
    ) -> str:
        """Build the prompt, A2A-call the assigned worker, return the response."""
        prompt = _build_subtask_prompt(task, dep_results, goal)

        # Dispatch event carries everything the frontend needs to render the
        # node transitioning to "running": worker id, required skill,
        # perspective (pro/con/null for debate tasks), full description, and
        # the dependency inputs this subtask was given (useful for tracing).
        await self.progress.on_progress(
            "subtask_dispatch",
            f"→ {worker.agent_id}: {task.description[:80]}",
            {
                "subtask_id": task.id,
                "worker_id": worker.agent_id,
                "required_skill": task.required_skill,
                "perspective": task.perspective,
                "description": task.description,
                "depends_on": task.depends_on,
            },
        )

        client, _card = await create_a2a_client(worker.url)
        try:
            return await send_and_get_text(
                client,
                prompt,
                on_intermediate=self._relay_intermediate,
            )
        finally:
            await client.close()

    async def _assign_workers(
        self, ready: list[SubTask]
    ) -> dict[str, WorkerEntry]:
        """Assign a WorkerEntry to each ready subtask via round-robin per skill.

        Raises PlanExecutionError if any required skill has zero workers.
        """
        # Group ready tasks by skill.
        by_skill: dict[str, list[SubTask]] = defaultdict(list)
        for t in ready:
            by_skill[t.required_skill].append(t)

        assignments: dict[str, WorkerEntry] = {}
        for skill, tasks in by_skill.items():
            workers = await self.registry.find_by_skill(skill)
            if not workers:
                raise PlanExecutionError(
                    f"No worker in registry advertises skill {skill!r} "
                    f"(needed by subtasks {[t.id for t in tasks]})"
                )
            for i, t in enumerate(tasks):
                assignments[t.id] = workers[i % len(workers)]
        return assignments

    async def _relay_intermediate(self, metadata: dict[str, Any]) -> None:
        """Forward intermediate tool-use events from workers to the SSE stream."""
        stage = metadata.get("stage", "info")
        message = metadata.get("message", "")
        data = metadata.get("data")
        await self.progress.on_progress(stage, message, data)
