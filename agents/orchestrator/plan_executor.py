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


DEBATE_AGENT_TAGS = ("ae1", "ae2", "ae3")


def _own_agent_id(perspective: str | None) -> str | None:
    """Extract 'ae1', 'ae2', or 'ae3' from a perspective string like
    'ae1: round 1'.

    Returns None if the perspective doesn't follow the aeN:/ convention,
    which signals this isn't a per-agent debate task.
    """
    if not perspective:
        return None
    p = perspective.strip().lower()
    for tag in DEBATE_AGENT_TAGS:
        if p == tag or p.startswith(f"{tag}:") or p.startswith(f"{tag} "):
            return tag
    return None


def _build_subtask_prompt(
    task: SubTask,
    dep_results: dict[str, str],
    goal: str,
    plan_subtasks: dict[str, SubTask] | None = None,
) -> str:
    """Compose the prompt a worker will actually receive.

    For non-debate subtasks: a flat list of dep outputs prefixed by their id.
    For debate subtasks (skill='debate' with an 'ae1:' / 'ae2:' perspective):
    deps are split into "Your previous arguments" vs "Opponent's previous
    arguments" so each agent receives the structured trajectory of the
    deliberation — this is what enables convergence across rounds.
    """
    own_agent = (
        _own_agent_id(task.perspective)
        if task.required_skill == "debate"
        else None
    )

    parts = [f"Goal: {goal}", f"Task: {task.description}"]
    if task.perspective:
        parts.append(f"Perspective: {task.perspective}")

    if not task.depends_on:
        return "\n".join(parts)

    if own_agent is None or plan_subtasks is None:
        # Generic dep dump for non-debate tasks (or debate tasks the planner
        # built without the ae1/ae2 convention — degraded but still works).
        parts.append("\nContext from previous steps:")
        for dep_id in task.depends_on:
            text = dep_results.get(dep_id, "").strip()
            parts.append(f"\n[{dep_id}]\n{text}")
        return "\n".join(parts)

    # Debate task: classify each dep as own / other-agent / shared context.
    # With 3 agents, "opponent" is plural — every other ae* agent counts.
    own_block: list[str] = []
    others_blocks: dict[str, list[str]] = {}
    shared_block: list[str] = []
    for dep_id in task.depends_on:
        dep_task = plan_subtasks.get(dep_id)
        text = dep_results.get(dep_id, "").strip()
        if not text:
            continue
        dep_agent = _own_agent_id(dep_task.perspective) if dep_task else None
        label = (dep_task.perspective or dep_id) if dep_task else dep_id
        block_line = f"\n[{label}]\n{text}"
        if dep_agent is None:
            shared_block.append(block_line)
        elif dep_agent == own_agent:
            own_block.append(block_line)
        else:
            others_blocks.setdefault(dep_agent, []).append(block_line)

    if shared_block:
        parts.append("\n[Shared context]")
        parts.extend(shared_block)
    if own_block:
        parts.append("\n[Your previous arguments]")
        parts.extend(own_block)
    for tag in DEBATE_AGENT_TAGS:
        if tag == own_agent:
            continue
        block = others_blocks.get(tag)
        if not block:
            continue
        parts.append(
            f"\n[{tag.upper()}'s arguments — respond to these]"
        )
        parts.extend(block)

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

    async def execute(
        self,
        plan: TaskPlan,
        prior_results: dict[str, str] | None = None,
        prior_subtasks: dict[str, SubTask] | None = None,
    ) -> dict[str, str]:
        """Run the plan; return {subtask_id: text_output} for all subtasks.

        `prior_results` and `prior_subtasks` let the executor resolve
        `depends_on` IDs that belong to a previous plan (used for extension
        plans that continue an unfinished debate). They are merged into the
        local context but only the subtasks of THIS plan are scheduled.
        """
        # Merge previous-plan context so deps in extension plans resolve.
        results: dict[str, str] = dict(prior_results or {})
        all_subtasks: dict[str, SubTask] = dict(prior_subtasks or {})
        for t in plan.subtasks:
            all_subtasks[t.id] = t
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
            # same-skill tasks land on different workers. For debate tasks
            # respect the ae1/ae2 perspective tag so each agent receives its
            # own per-round subtasks (the round-robin default would shuffle
            # them and break trajectory).
            assignments = await self._assign_workers(ready)
            batch_outputs = await asyncio.gather(
                *[
                    self._execute_subtask(
                        t, assignments[t.id], results, plan.goal, all_subtasks,
                    )
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
        all_subtasks: dict[str, SubTask] | None = None,
    ) -> str:
        """Build the prompt, A2A-call the assigned worker, return the response."""
        prompt = _build_subtask_prompt(task, dep_results, goal, all_subtasks)

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
        """Assign a WorkerEntry to each ready subtask.

        Default rule: round-robin across all workers advertising the skill.
        Override for debate tasks: if `perspective` starts with one of the
        recognised debate tags ('ae1:' / 'ae2:' / 'ae3:'), pin the task to
        the worker with that agent_id when available — this keeps each
        agent's trajectory on a single LLM/provider across rounds.

        Raises PlanExecutionError if any required skill has zero workers.
        """
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
            by_id = {w.agent_id: w for w in workers}
            unpinned: list[SubTask] = []
            for t in tasks:
                tag = _own_agent_id(t.perspective) if skill == "debate" else None
                if tag and tag in by_id:
                    assignments[t.id] = by_id[tag]
                else:
                    unpinned.append(t)
            for i, t in enumerate(unpinned):
                assignments[t.id] = workers[i % len(workers)]
        return assignments

    async def _relay_intermediate(self, metadata: dict[str, Any]) -> None:
        """Forward intermediate tool-use events from workers to the SSE stream."""
        stage = metadata.get("stage", "info")
        message = metadata.get("message", "")
        data = metadata.get("data")
        await self.progress.on_progress(stage, message, data)
