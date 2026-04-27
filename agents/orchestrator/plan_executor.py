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

import httpx

from agents.orchestrator.agent_registry import AgentRegistry
from agents.orchestrator.persona_catalog import build_persona
from agents.specialized.eristic import random_stratagem
from common.a2a_helpers import create_a2a_client, send_and_get_text
from common.models import AgentRoleConfig, SubTask, TaskPlan, WorkerEntry

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


def build_round_prompt(
    *,
    goal: str,
    task_description: str,
    round_number: int,
    max_rounds: int,
    ledger_view: str,
    your_last_text: str,
) -> str:
    """Phase 4: build the per-turn user prompt for a round-based dispatch.

    The orchestrator's DeliberationLoop calls this once per speaker per
    round. The format mirrors `_build_subtask_prompt` so the worker's
    `_split_topic_and_peers` keeps working unchanged: ``Goal:`` is
    extracted, ``Discussion ledger so far:`` becomes the peers_outputs
    block. The round/self-position sections are surfaced separately and
    flow into the persona system prompt via dedicated placeholders.
    """
    return (
        f"Goal: {goal}\n"
        f"Task: {task_description}\n"
        f"Round: {round_number} / {max_rounds}\n\n"
        f"Your previous position:\n{your_last_text}\n\n"
        f"Discussion ledger so far:\n{ledger_view}"
    )


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
        return await self.execute_partial(plan, seed_results=None)

    async def execute_partial(
        self,
        plan: TaskPlan,
        seed_results: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Run ``plan`` but treat any id present in ``seed_results`` as done.

        Used by DRTAG (Phase C) to splice a small two-node disruption sub-plan
        on top of an already-completed canonical quartet without re-dispatching
        the analyst / seeker / original devil's advocate. Dependencies on
        seeded ids are satisfied with the cached output; the rest of the
        DAG executes normally.
        """
        results: dict[str, str] = dict(seed_results or {})
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
                    self._execute_subtask(t, assignments[t.id], results, plan)
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

    async def dispatch_one(
        self,
        *,
        role_id: str,
        prompt: str,
        plan: TaskPlan,
        round_number: int,
        max_rounds: int,
        stratagem=None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Phase 4 single-task dispatch: configure + send, no DAG walk.

        Used by ``DeliberationLoop`` to invoke one role per turn within a
        round. Round-aware persona rendering happens *server-side* in the
        worker via the round_number/ledger_view extracted from the prompt;
        we just need to:
          1. Pick a worker advertising ``role_<role_id>`` from the registry.
          2. POST a fresh PersonaContract (the stratagem may rotate per round
             for the Devil's Advocate).
          3. Send the prompt and return the text + persona metadata.

        Returns ``(text, persona_meta)`` so the caller can build a LedgerEntry.
        """
        skill = f"role_{role_id}"
        workers = await self.registry.find_by_skill(skill)
        if not workers:
            raise PlanExecutionError(
                f"No worker advertises {skill!r} for round {round_number}"
            )
        # Round-robin across workers of the same role: pick by round number
        # so two consecutive rounds dispatching the same role pick different
        # workers if available (helps when DRTAG spawned a disruptor).
        worker = workers[round_number % len(workers)]

        # Build a SubTask shell so we can reuse _configure_worker.
        from common.models import SubTask  # local import to avoid cycle

        synthetic_task = SubTask(
            id=f"r{round_number}_{role_id}",
            description=f"Round {round_number} intervention as {role_id}",
            role_id=role_id,  # type: ignore[arg-type]
            required_skill=skill,
            depends_on=[],
        )
        persona_meta = await self._configure_worker(
            worker, synthetic_task, plan, stratagem_override=stratagem
        )

        await self.progress.on_progress(
            "round_dispatch",
            f"→ {worker.agent_id} (ronda {round_number})",
            {
                "role_id": role_id,
                "round_number": round_number,
                "worker_id": worker.agent_id,
                "persona": persona_meta,
            },
        )

        client, _card = await create_a2a_client(worker.url)
        try:
            text = await send_and_get_text(
                client,
                prompt,
                on_intermediate=self._relay_intermediate,
            )
        finally:
            await client.close()
        return text, persona_meta

    async def _execute_subtask(
        self,
        task: SubTask,
        worker: WorkerEntry,
        dep_results: dict[str, str],
        plan: TaskPlan,
    ) -> str:
        """Configure the worker's persona, then A2A-dispatch the prompt.

        Phase 3 inserts a synchronous configure step before every dispatch:
        the orchestrator builds the role's PersonaContract via
        ``persona_catalog.build_persona`` and POSTs it to the worker's
        ``/internal/configure`` endpoint. The worker's executor then reads
        this persona to render the system prompt and enforce its tool
        whitelist for this run.
        """
        # 1. Configure persona (skipped only for legacy non-role subtasks).
        persona_meta = await self._configure_worker(worker, task, plan)

        # 2. Build the user-facing prompt (goal/task/context-from-deps).
        prompt = _build_subtask_prompt(task, dep_results, plan.goal)

        # Dispatch event carries everything the frontend needs to render the
        # node transitioning to "running": worker id, role/skill, persona
        # display name + stratagem (for the Devil's Advocate badge), full
        # description, and the dependency inputs this subtask was given.
        await self.progress.on_progress(
            "subtask_dispatch",
            f"→ {worker.agent_id}: {task.description[:80]}",
            {
                "subtask_id": task.id,
                "worker_id": worker.agent_id,
                "role_id": task.role_id,
                "required_skill": task.required_skill,
                "perspective": task.perspective,
                "description": task.description,
                "depends_on": task.depends_on,
                "persona": persona_meta,
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

    async def _configure_worker(
        self,
        worker: WorkerEntry,
        task: SubTask,
        plan: TaskPlan,
        *,
        stratagem_override=None,
    ) -> dict[str, Any] | None:
        """POST the role's PersonaContract to ``worker``'s configure endpoint.

        Returns a small dict (display_name, role_id, stratagem_id) suitable for
        embedding in the dispatch progress event so the frontend can label
        the node before the worker emits any text. Returns ``None`` if the
        subtask has no role_id (defensive: should not happen in Phase 3).
        """
        if task.role_id is None:
            logger.warning(
                "Subtask %s has no role_id — skipping persona configure",
                task.id,
            )
            return None

        # Stratagem is only meaningful for the Devil's Advocate. Picking a
        # random one per dispatch gives the qualitative variety the TFG
        # memoria cites; DRTAG (Phase C) will pass an explicit stratagem to
        # ensure the disruptor uses a *different* one from the original.
        if stratagem_override is not None:
            stratagem = stratagem_override
        elif task.role_id == "devils_advocate":
            stratagem = random_stratagem()
        else:
            stratagem = None
        persona = build_persona(task.role_id, stratagem=stratagem)

        config = AgentRoleConfig(
            persona=persona,
            skills=[],  # Phase 3: persona drives identity, skills are advisory
            claim=plan.claim,
        )
        configure_url = f"{worker.url.rstrip('/')}/internal/configure"

        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                response = await http.post(
                    configure_url, json=config.model_dump()
                )
                response.raise_for_status()
        except Exception as e:
            # A failed configure means the worker would run with a stale
            # persona — better to fail loudly than silently produce a wrong
            # role's output.
            raise PlanExecutionError(
                f"Failed to configure {worker.agent_id} as "
                f"{task.role_id}: {e}"
            ) from e

        return {
            "display_name": persona.display_name,
            "role_id": persona.role_id,
            "stratagem_id": persona.eristic_stratagem_id,
            "tool_whitelist": list(persona.tool_whitelist),
            "temperature": persona.temperature,
        }

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
