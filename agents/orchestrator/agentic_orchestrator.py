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
import time
from collections import Counter, defaultdict
from typing import Any

from agents.orchestrator.agent_registry import AgentRegistry
from agents.orchestrator.deliberation_loop import DeliberationLoop
from agents.orchestrator.plan_executor import PlanExecutionError, PlanExecutor, ProgressCallback
from agents.orchestrator.planner import Planner
from agents.orchestrator.worker_spawner import WorkerSpawner
from agents.specialized.eristic import random_stratagem
from common.config import settings
from common.llm_provider import llm_complete
from common.models import CANONICAL_ROLES, RoleId, SubTask, TaskPlan

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


# ── DRTAG / aporia thresholds ──────────────────────────────────────────────
# An aporia is declared when the panel's debate-side roles converge to nearly
# the same posterior AND the magnitudes of their belief updates are small.
# Both conditions must hold so that *legitimate* convergence (where every
# agent moved a lot but ended in agreement) is not misclassified.
APORIA_LOG_ODDS_SPREAD_EPS = 0.30   # |max - min| of final log_odds
APORIA_TOTAL_MOVEMENT_EPS = 0.40    # mean per-agent total |delta_log_odds|

# Roles whose belief trajectories enter the aporia decision. We exclude the
# Synthesizer because by construction it does not adopt a position.
_DEBATE_SIDE_ROLES: tuple[RoleId, ...] = (
    "analyst",
    "seeker",
    "devils_advocate",
)


class _BeliefRecordingProgress(ProgressCallback):
    """Wraps an upstream ProgressCallback and aggregates Phase 3 metadata.

    Captures, while still forwarding every event downstream:
      - belief_update events into per-agent sample logs (for aporia
        detection and post-hoc DRTAG disruption);
      - subtask_dispatch events that carry a stratagem_id, so DRTAG can
        pick a *different* stratagem for the disruptor.

    The wrapper has no business logic; aporia analysis lives in
    ``_detect_aporia`` so it can be unit-tested without spinning up the
    full orchestrator.
    """

    def __init__(self, upstream: ProgressCallback) -> None:
        self.upstream = upstream
        # agent_id → list of {log_odds, delta, rationale, phase, role_id}
        self.beliefs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # stratagem_ids that the orchestrator has already used this run
        self.used_stratagems: set[int] = set()
        # Phase 4: optional reference to the DeliberationLoop so belief
        # updates project into its compact summary as they stream in.
        self.loop: "DeliberationLoop | None" = None

    async def on_progress(
        self, stage: str, message: str, data: dict | None = None
    ) -> None:
        if stage == "belief_update" and data and data.get("agent"):
            self.beliefs[data["agent"]].append(
                {
                    "log_odds": data.get("log_odds", 0.0),
                    "delta": data.get("delta", 0.0),
                    "rationale": data.get("rationale", ""),
                    "phase": data.get("phase", ""),
                    "role_id": data.get("role_id"),
                }
            )
            # Phase 4: project into the loop's belief summary so speaker
            # selection and aporia detection see fresh numbers.
            if self.loop is not None:
                role_id = data.get("role_id")
                if role_id is not None:
                    self.loop.update_belief_from_event(
                        role_id,
                        log_odds=float(data.get("log_odds", 0.0)),
                        delta=float(data.get("delta", 0.0)),
                    )
        if stage in ("subtask_dispatch", "round_dispatch") and data:
            persona = data.get("persona") or {}
            sid = persona.get("stratagem_id")
            if isinstance(sid, int):
                self.used_stratagems.add(sid)
        await self.upstream.on_progress(stage, message, data)


def _detect_aporia(
    beliefs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Decide whether the panel's belief trajectories indicate aporia.

    An aporia is the dialectic stuck state: roles converged without
    actually moving (sycophancy / shared-blind-spot patterns). Returns a
    diagnostic dict regardless of outcome — the frontend renders the
    numbers even on a negative call so the academic memoria can show the
    detector's calibration.
    """
    debate_logs = [
        (agent, samples)
        for agent, samples in beliefs.items()
        if samples
        # Filter to debate-side roles. The first sample carries role_id.
        and samples[0].get("role_id") in _DEBATE_SIDE_ROLES
    ]

    if len(debate_logs) < 2:
        return {
            "detected": False,
            "reason": "insufficient_debate_signal",
            "n_agents": len(debate_logs),
            "spread": None,
            "mean_total_movement": None,
        }

    final_log_odds = [samples[-1]["log_odds"] for _agent, samples in debate_logs]
    spread = max(final_log_odds) - min(final_log_odds)

    total_movements = [
        sum(abs(s["delta"]) for s in samples)
        for _agent, samples in debate_logs
    ]
    mean_movement = sum(total_movements) / len(total_movements)

    detected = (
        spread < APORIA_LOG_ODDS_SPREAD_EPS
        and mean_movement < APORIA_TOTAL_MOVEMENT_EPS
    )
    return {
        "detected": detected,
        "reason": "flat_and_close" if detected else "movement_or_spread",
        "n_agents": len(debate_logs),
        "spread": spread,
        "mean_total_movement": mean_movement,
        "agent_finals": {
            agent: samples[-1]["log_odds"]
            for agent, samples in debate_logs
        },
    }


_HABERMAS_DELIMITER = "---HABERMAS-JSON---"


def _split_habermas_table(
    sink_text: str,
) -> tuple[str, list[dict[str, Any]] | None]:
    """Split a synthesizer output into (verdict, validity_claims).

    The Synthesizer prompt asks for two parts separated by
    ``---HABERMAS-JSON---``. If the delimiter is present we try to parse
    the JSON tail; on any malformed input we fall back to (full_text, None)
    so the user still gets the verdict.
    """
    if _HABERMAS_DELIMITER not in sink_text:
        return sink_text, None

    head, _, tail = sink_text.partition(_HABERMAS_DELIMITER)
    head = head.strip()
    tail = tail.strip()

    # Strip a leading ```json fence if the model added one.
    if tail.startswith("```"):
        tail = tail.split("\n", 1)[1] if "\n" in tail else tail[3:]
        if tail.endswith("```"):
            tail = tail[: -3]
        tail = tail.strip()

    import json as _json
    try:
        data = _json.loads(tail)
    except _json.JSONDecodeError:
        logger.warning("Habermas table JSON parse failed; verdict shown as-is.")
        return head, None

    claims = data.get("validity_claims")
    if not isinstance(claims, list):
        return head, None

    # Lightweight schema sanity: drop entries missing the required keys.
    cleaned: list[dict[str, Any]] = []
    for entry in claims:
        if not isinstance(entry, dict):
            continue
        if "agent" not in entry:
            continue
        cleaned.append(entry)
    return head, cleaned or None


def _role_from_skill(skill: str) -> RoleId | None:
    """Parse ``role_<role_id>`` skills back to a canonical role.

    Returns ``None`` for non-role skills (legacy/foreign), letting the
    caller decide whether that's acceptable. Phase 3 plans are pure
    role-skills, but we keep the parser tolerant for the test suite
    which sometimes uses synthetic skill ids.
    """
    if not skill.startswith("role_"):
        return None
    candidate = skill[len("role_"):]
    if candidate in CANONICAL_ROLES:
        return candidate  # type: ignore[return-value]
    return None


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
        # Wrap the upstream progress so we can passively aggregate belief
        # samples + stratagem usage during execute() without coupling the
        # PlanExecutor to DRTAG concerns.
        self._recorder = _BeliefRecordingProgress(self.progress)
        self.executor = PlanExecutor(
            registry=registry, progress=self._recorder
        )
        # Track workers we spawned for this run so we can tear them down.
        self._spawned_this_run: list[str] = []

    async def run(self, user_input: str) -> str:
        """Execute one agentic run; returns the final answer text.

        Phase 4 flow:
          1. Discover workers from the registry.
          2. Planner extracts {goal, claim} from the user input.
          3. Pre-flight capacity check (one worker per canonical role).
          4. DeliberationLoop runs N rounds of round-robin interventions
             over a shared blackboard (DiscussionLedger).
          5. The Synthesizer (final round) emits the verdict + Habermas
             validity table. We split the table out of the verdict and
             emit ``habermas_table`` so the frontend can render it.
        """
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

            # 2. Plan (extracts goal + claim only — the DAG is fixed)
            await self.progress.on_progress(
                "plan", "Extrayendo objetivo + claim del prompt..."
            )
            plan = await self.planner.create_plan(user_input, catalog)
            await self.progress.on_progress(
                "plan_ready",
                f"Plan: '{plan.goal}' · claim '{plan.claim}'",
                {"plan": plan.model_dump()},
            )

            # 3. Pre-flight: spawn workers if any canonical role is missing
            await self._ensure_capacity(plan)

            # 4. Multi-round deliberation
            loop = DeliberationLoop(
                executor=self.executor,
                progress=self._recorder,
                max_rounds=settings.deliberation_max_rounds,
            )
            self._recorder.loop = loop  # wire belief projection
            ledger, raw_verdict = await loop.run(
                claim=plan.claim, goal=plan.goal
            )

            # 5. Split verdict and Habermas table; emit table separately.
            verdict, habermas = _split_habermas_table(raw_verdict)
            if habermas is not None:
                await self.progress.on_progress(
                    "habermas_table",
                    "Tabla de pretensiones de validez del Sintetizador.",
                    {"validity_claims": habermas},
                )
            return verdict or raw_verdict

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
        """Spawn extra workers if peak concurrent demand exceeds supply.

        Phase 3: every required_skill is of the form ``role_<role_id>``;
        we derive the dialectic role from the skill id and pass it to the
        spawner so the new subprocess advertises the right role-skill on
        registration.
        """
        peak = _peak_concurrent_demand(plan)
        for skill, needed in peak.items():
            have = await self.registry.find_by_skill(skill)
            missing = needed - len(have)
            if missing <= 0:
                continue

            role = _role_from_skill(skill)
            if role is None:
                # No legacy fallback in Phase 3 — every plan node is a role.
                raise PlanExecutionError(
                    f"Cannot spawn worker for skill {skill!r}: not a "
                    f"role_<role_id> skill (Phase 3 requires role-bound workers)"
                )

            await self.progress.on_progress(
                "spawn",
                f"Spawneando {missing} worker(s) extra para rol '{role}'",
                {
                    "skill": skill,
                    "role": role,
                    "needed": needed,
                    "have": len(have),
                },
            )
            for i in range(missing):
                suffix = len(have) + i + 1
                agent_id = f"dyn_{role}_{suffix}"
                try:
                    await self.spawner.spawn(agent_id, role=role)
                    self._spawned_this_run.append(agent_id)
                except Exception as e:
                    logger.exception("Spawn failed for %s", agent_id)
                    await self.progress.on_progress(
                        "spawn_failed",
                        f"No se pudo spawnear worker {agent_id}: {e}",
                        {
                            "agent_id": agent_id,
                            "role": role,
                            "error": str(e),
                        },
                    )
                    raise

    async def _maybe_apply_drtag(
        self,
        plan: TaskPlan,
        results: dict[str, str],
    ) -> dict[str, str]:
        """If the panel is in aporia, spawn a disruptor and re-synthesize.

        Phase 3 / Pillar 3:
          1. Inspect the per-agent belief log captured by ``_recorder``.
          2. If aporia is declared by ``_detect_aporia``, emit
             ``aporia_detected`` so the frontend can render the band.
          3. Spawn a fresh ``dyn_disruptor_<n>`` worker bound to
             ``devils_advocate``, configured with a stratagem that was NOT
             used in the original run (drawn from the catalog).
          4. Run a one-off Disruptor subtask (depends on the analyst output
             only — that's what the original Devil's Advocate consumed).
          5. Append the disruptor's output to the synthesis context and
             ask the synthesizer to re-evaluate. The new synthesis output
             replaces ``results[t4]`` so the verdict is the post-DRTAG one.

        On any failure inside the disruption pipeline we log + emit
        ``aporia_recovered_failed`` and return the original results
        unchanged — DRTAG is informative, never blocking.
        """
        diagnosis = _detect_aporia(self._recorder.beliefs)
        if not diagnosis["detected"]:
            return results

        await self.progress.on_progress(
            "aporia_detected",
            (
                "Aporía detectada: las trayectorias bayesianas no se "
                f"separan (spread={diagnosis['spread']:.2f}) y el "
                f"movimiento total es bajo "
                f"(mean={diagnosis['mean_total_movement']:.2f})."
            ),
            diagnosis,
        )

        # Find the synthesizer subtask in the original plan so we can
        # reuse its dependency chain after augmenting with the disruptor.
        synth_task = next(
            (t for t in plan.subtasks if t.role_id == "synthesizer"), None
        )
        analyst_task = next(
            (t for t in plan.subtasks if t.role_id == "analyst"), None
        )
        if synth_task is None or analyst_task is None:
            await self.progress.on_progress(
                "aporia_recovered_failed",
                "Plan sin Sintetizador/Analista — no se puede ejecutar DRTAG.",
                None,
            )
            return results

        # Pick a stratagem distinct from any used so far.
        try:
            new_stratagem = random_stratagem(
                exclude=set(self._recorder.used_stratagems)
            )
        except Exception as e:
            logger.warning("DRTAG random_stratagem failed: %s", e)
            return results

        await self.progress.on_progress(
            "drtag_dispatch",
            (
                f"Spawneando disruptor con estratagema #{new_stratagem.id} "
                f"({new_stratagem.name})."
            ),
            {
                "stratagem_id": new_stratagem.id,
                "stratagem_name": new_stratagem.name,
                "excluded": sorted(self._recorder.used_stratagems),
            },
        )

        disruptor_id = f"dyn_disruptor_{int(time.time())}"
        try:
            await self.spawner.spawn(disruptor_id, role="devils_advocate")
            self._spawned_this_run.append(disruptor_id)
        except Exception as e:
            logger.warning("DRTAG disruptor spawn failed: %s", e)
            await self.progress.on_progress(
                "aporia_recovered_failed",
                f"No se pudo spawnear el disruptor: {e}",
                {"agent_id": disruptor_id, "error": str(e)},
            )
            return results

        # Build a tiny two-node sub-plan: disruptor → re-synthesis.
        # We use the original plan's claim / goal so the persona templates
        # render with the same context.
        disruptor_subtask = SubTask(
            id="t3p",
            description=(
                "Counter-attack the panel's emerging consensus on "
                f"{plan.claim!r} using a fresh eristic stratagem (#"
                f"{new_stratagem.id} {new_stratagem.name}). Surface "
                "assumptions the original Devil's Advocate missed."
            ),
            role_id="devils_advocate",
            required_skill="role_devils_advocate",
            depends_on=["t1"],
        )
        resynth_subtask = SubTask(
            id="t4p",
            description=(
                "Re-evaluate the panel under Habermasian validity claims, "
                "now including the disruptor's counter-attack."
            ),
            role_id="synthesizer",
            required_skill="role_synthesizer",
            depends_on=["t1", "t2", "t3", "t3p"],
        )
        drtag_plan = TaskPlan(
            goal=plan.goal,
            claim=plan.claim,
            subtasks=[
                # Carry over the original analyst/seeker/devil's advocate
                # tasks unchanged so the executor's dep_results map can
                # reuse the cached outputs without re-dispatching.
                # (The executor only re-runs nodes whose ids are absent
                # from `results`, so we need to seed `results` accordingly.)
                disruptor_subtask,
                resynth_subtask,
            ],
            max_workers=2,
        )

        # Override the spawner's _ensure_capacity for this sub-plan: we
        # already have synthesizer + devils_advocate workers; the disruptor
        # is the new spawn. PlanExecutor's _assign_workers will pick the
        # disruptor *or* the original devils_advocate via round-robin —
        # either is fine since both advertise role_devils_advocate.

        try:
            sub_results = await self.executor.execute_partial(
                drtag_plan, seed_results=results
            )
        except PlanExecutionError as e:
            logger.warning("DRTAG sub-plan failed: %s", e)
            await self.progress.on_progress(
                "aporia_recovered_failed",
                f"DRTAG sub-plan falló: {e}",
                {"error": str(e)},
            )
            return results

        merged = dict(results)
        merged["t3p"] = sub_results.get("t3p", "")
        # t4p is the new verdict. Replace t4 so _synthesize picks it as the
        # sink-output naturally; also keep t4p so the frontend can show both.
        if "t4p" in sub_results:
            merged["t4"] = sub_results["t4p"]
            merged["t4p"] = sub_results["t4p"]

        await self.progress.on_progress(
            "drtag_resynthesized",
            "Veredicto reemitido tras la disrupción.",
            {
                "stratagem_id": new_stratagem.id,
                "stratagem_name": new_stratagem.name,
            },
        )
        return merged

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
            raw_sink_output = results.get(sink.id, "").strip()
            verdict, habermas_table = _split_habermas_table(raw_sink_output)
            if habermas_table is not None:
                await self.progress.on_progress(
                    "habermas_table",
                    "Tabla de pretensiones de validez del Sintetizador.",
                    {"validity_claims": habermas_table},
                )
            return verdict or self._fallback_summary(results)

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
