"""DeliberationLoop — Phase 4 multi-round deliberation orchestrator.

Replaces the one-shot DAG walk of Phase 3 with a turn-based blackboard:

  Round 1 (opening):  every debate-side role intervenes once.
  Round 2..N:         a subset of roles speaks (speaker selection); each
                      sees the full ledger of all prior interventions.
  Final round:        the Synthesizer reads the entire ledger and emits
                      the verdict + Habermasian validity table.

Termination criteria (any one ends the loop):
  - Consensus: belief log-odds spread across debate roles is small AND
    every role moved meaningfully across the run (legitimate convergence).
  - Aporia: spread is small AND nobody moved (panel stuck → DRTAG fires
    a disruptor with a fresh stratagem and the loop continues for one
    more round before checking again).
  - max_rounds reached.

The loop publishes ``round_start``, ``ledger_entry``, ``round_end`` and
``deliberation_complete`` events so the frontend can render the chat-style
threaded view in real time without waiting for the final synthesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agents.orchestrator.discussion_ledger import (
    append as ledger_append,
    build_self_position_section,
    format_for_prompt,
    role_label,
)
from agents.orchestrator.plan_executor import (
    PlanExecutor,
    ProgressCallback,
    build_round_prompt,
)
from agents.specialized.eristic import random_stratagem
from common.models import (
    DiscussionLedger,
    LedgerEntry,
    RoleId,
    SubTask,
    TaskPlan,
)

logger = logging.getLogger(__name__)


# Roles that take a position on the claim (everyone except the Synthesizer).
DEBATE_ROLES: tuple[RoleId, ...] = (
    "analyst",
    "seeker",
    "devils_advocate",
    "empiricist",
    "pragmatist",
)

# A speaker is selected for a follow-up round if its belief moved less than
# this threshold in its last turn (i.e. it's stuck or holding firm).
STUCK_DELTA_THRESHOLD = 0.4

# Aporia thresholds (mirror the Phase 3 DRTAG calibration).
APORIA_LOG_ODDS_SPREAD_EPS = 0.30
APORIA_TOTAL_MOVEMENT_EPS = 0.40

# Hard cap to keep rate-limited providers happy. Each agent may speak at
# most this many times across the whole deliberation (opening + follow-ups).
MAX_TURNS_PER_AGENT = 4


@dataclass
class _Belief:
    """Compact projection of an agent's running BeliefState for selection logic."""
    log_odds: float = 0.0
    total_movement: float = 0.0
    last_delta: float = 0.0
    turns: int = 0


class DeliberationLoop:
    """Multi-round deliberation orchestrator over a shared blackboard.

    The loop owns a ``DiscussionLedger`` and a per-role ``_Belief``
    summary derived from belief_update events relayed by the workers.
    It does NOT replicate the full BeliefState — the ground truth lives
    inside each worker; the loop only needs enough signal to pick the
    next round's speakers and detect aporia.
    """

    def __init__(
        self,
        executor: PlanExecutor,
        progress: ProgressCallback,
        max_rounds: int = 3,
    ) -> None:
        self.executor = executor
        self.progress = progress
        self.max_rounds = max_rounds
        self._beliefs: dict[RoleId, _Belief] = {}
        # Stratagems already used by any Devil's Advocate in this run.
        self._used_stratagems: set[int] = set()

    # ── Public entrypoint ───────────────────────────────────────────────

    async def run(self, *, claim: str, goal: str) -> tuple[DiscussionLedger, str]:
        """Run the deliberation; return (ledger, final_verdict_text)."""
        ledger = DiscussionLedger(
            claim=claim, goal=goal, max_rounds=self.max_rounds
        )
        plan = self._make_synthetic_plan(claim, goal)

        await self.progress.on_progress(
            "deliberation_start",
            f"Iniciando deliberación · {self.max_rounds} rondas máximo",
            {"claim": claim, "goal": goal, "max_rounds": self.max_rounds},
        )

        # ── Round 1: opening statements (all debate-side roles) ──
        ledger.current_round = 1
        await self._dispatch_round(ledger, list(DEBATE_ROLES), plan, round_number=1)

        # ── Rounds 2..N: speaker-selected follow-ups + termination ──
        for r in range(2, self.max_rounds + 1):
            ledger.current_round = r

            speakers = self._select_speakers(ledger)
            if not speakers:
                ledger.terminated_reason = "consensus"
                await self.progress.on_progress(
                    "deliberation_terminated",
                    f"Consenso alcanzado tras la ronda {r - 1}.",
                    {"reason": "consensus", "rounds_used": r - 1},
                )
                break

            await self._dispatch_round(ledger, speakers, plan, round_number=r)

            # Mid-deliberation aporia check
            diagnosis = self._detect_aporia()
            if diagnosis["detected"]:
                handled = await self._fire_disruptor(ledger, plan, r, diagnosis)
                if not handled:
                    ledger.terminated_reason = "aporia_unhandled"
                    break
                # If disruptor fired, give the panel one more round to react
                # before re-checking termination — but only if we haven't run
                # out of rounds.
        else:
            ledger.terminated_reason = "max_rounds"
            await self.progress.on_progress(
                "deliberation_terminated",
                "Máximo de rondas alcanzado.",
                {"reason": "max_rounds", "rounds_used": self.max_rounds},
            )

        # ── Final round: synthesizer ──
        verdict = await self._run_synthesis(ledger, plan)
        await self.progress.on_progress(
            "deliberation_complete",
            "Deliberación concluida.",
            {
                "rounds_used": ledger.current_round,
                "terminated_reason": ledger.terminated_reason,
                "n_entries": len(ledger.entries),
            },
        )
        return ledger, verdict

    # ── Round mechanics ─────────────────────────────────────────────────

    async def _dispatch_round(
        self,
        ledger: DiscussionLedger,
        speakers: list[RoleId],
        plan: TaskPlan,
        *,
        round_number: int,
    ) -> None:
        """Dispatch every role in `speakers` once, in sequence (not parallel).

        Sequential dispatch is intentional: each speaker should see the
        ledger updates from earlier speakers in the *same* round, not just
        from prior rounds. This is what makes the panel react to itself
        within a round instead of feeling like N independent monologues.
        """
        await self.progress.on_progress(
            "round_start",
            f"Ronda {round_number} · participan: "
            + ", ".join(role_label(r) for r in speakers),
            {
                "round_number": round_number,
                "speakers": list(speakers),
                "max_rounds": self.max_rounds,
            },
        )

        for role in speakers:
            entry = await self._dispatch_one(
                ledger, role, plan, round_number=round_number
            )
            if entry is None:
                # Worker error already surfaced via plan_executor's progress.
                continue
            await self.progress.on_progress(
                "ledger_entry",
                f"[turno {entry.turn}] {role_label(role)} habló",
                {
                    "entry": entry.model_dump(),
                    "round_number": round_number,
                },
            )

        await self.progress.on_progress(
            "round_end",
            f"Ronda {round_number} cerrada · {len(ledger.entries)} entries en total",
            {
                "round_number": round_number,
                "n_entries": len(ledger.entries),
                "beliefs": {
                    r: {
                        "log_odds": b.log_odds,
                        "total_movement": b.total_movement,
                        "last_delta": b.last_delta,
                        "turns": b.turns,
                    }
                    for r, b in self._beliefs.items()
                },
            },
        )

    async def _dispatch_one(
        self,
        ledger: DiscussionLedger,
        role_id: RoleId,
        plan: TaskPlan,
        *,
        round_number: int,
    ) -> LedgerEntry | None:
        """Run one speaker for one round and append to the ledger."""
        ledger_view = format_for_prompt(ledger, viewer_role=role_id)
        your_last_text = build_self_position_section(ledger, role_id=role_id)

        description = (
            f"Intervén como {role_label(role_id)} en la deliberación sobre "
            f"{ledger.claim!r}."
        )
        prompt = build_round_prompt(
            goal=ledger.goal,
            task_description=description,
            round_number=round_number,
            max_rounds=ledger.max_rounds,
            ledger_view=ledger_view,
            your_last_text=your_last_text,
        )

        # Devil's Advocate: rotate stratagem each time it speaks so the
        # disruption stays fresh across rounds. Other roles ignore stratagem.
        stratagem = None
        if role_id == "devils_advocate":
            try:
                stratagem = random_stratagem(exclude=set(self._used_stratagems))
                self._used_stratagems.add(stratagem.id)
            except Exception:
                stratagem = random_stratagem()  # fallback if we exhausted them

        try:
            text, _persona_meta = await self.executor.dispatch_one(
                role_id=role_id,
                prompt=prompt,
                plan=plan,
                round_number=round_number,
                max_rounds=self.max_rounds,
                stratagem=stratagem,
            )
        except Exception as e:
            logger.warning(
                "DeliberationLoop: dispatch %s round %d failed: %s",
                role_id, round_number, e,
            )
            return None

        # The worker emits belief_update events; our caller (orchestrator)
        # threads them through `_BeliefRecordingProgress` so we can read
        # the latest belief here. Until that wiring lands, fall back to a
        # neutral 0 — the ledger entry stays useful even without belief.
        belief = self._beliefs.get(role_id)
        belief_after = belief.log_odds if belief else None
        delta = belief.last_delta if belief else None

        entry = ledger_append(
            ledger,
            role_id=role_id,
            agent_id=role_id,  # one canonical worker per role; refine if dyn workers
            text=text,
            round_number=round_number,
            belief_after=belief_after,
            delta=delta,
        )
        return entry

    # ── Speaker selection ───────────────────────────────────────────────

    def _select_speakers(self, ledger: DiscussionLedger) -> list[RoleId]:
        """Decide who speaks in the next round.

        Rules (in order):
          1. The Devil's Advocate speaks every round (constant pressure).
          2. Any role whose last belief delta < STUCK_DELTA_THRESHOLD
             AND who hasn't hit MAX_TURNS_PER_AGENT speaks again.
          3. A role that was *referenced* in another role's most recent
             entry gets priority (it was challenged → should respond).
          4. Roles that already moved a lot in the last round are skipped
             this round to give space.

        If the resulting speaker list is empty, the loop terminates with
        ``consensus`` reason — nobody has anything new to say.
        """
        if not ledger.entries:
            return list(DEBATE_ROLES)

        last_round = max(e.round_number for e in ledger.entries)
        speakers: list[RoleId] = []

        # Rule 1: DA always speaks (unless cap reached)
        da_belief = self._beliefs.get("devils_advocate")
        if not da_belief or da_belief.turns < MAX_TURNS_PER_AGENT:
            speakers.append("devils_advocate")

        # Rule 2 + 3: stuck or directly challenged roles
        last_round_entries = [
            e for e in ledger.entries if e.round_number == last_round
        ]
        challenged_roles = {
            ledger.entries[ref].role_id
            for entry in last_round_entries
            for ref in entry.references
            if 0 <= ref < len(ledger.entries)
        }

        for role in DEBATE_ROLES:
            if role == "devils_advocate":
                continue
            b = self._beliefs.get(role)
            if b and b.turns >= MAX_TURNS_PER_AGENT:
                continue
            stuck = b is not None and abs(b.last_delta) < STUCK_DELTA_THRESHOLD
            challenged = role in challenged_roles
            # In Round 2 we want broad participation; afterwards only
            # stuck/challenged roles.
            broadly_active = last_round == 1
            if broadly_active or stuck or challenged:
                speakers.append(role)

        # De-duplicate while preserving order
        seen = set()
        ordered: list[RoleId] = []
        for r in speakers:
            if r not in seen:
                ordered.append(r)
                seen.add(r)
        return ordered

    # ── Termination / aporia ────────────────────────────────────────────

    def _detect_aporia(self) -> dict[str, Any]:
        """Same calibration as the Phase 3 DRTAG detector, applied per-round."""
        debate_beliefs = [
            b for r, b in self._beliefs.items() if r in DEBATE_ROLES and b.turns > 0
        ]
        if len(debate_beliefs) < 2:
            return {"detected": False, "reason": "insufficient_signal"}

        log_odds = [b.log_odds for b in debate_beliefs]
        spread = max(log_odds) - min(log_odds)
        mean_movement = sum(b.total_movement for b in debate_beliefs) / len(
            debate_beliefs
        )
        detected = (
            spread < APORIA_LOG_ODDS_SPREAD_EPS
            and mean_movement < APORIA_TOTAL_MOVEMENT_EPS
        )
        return {
            "detected": detected,
            "reason": "flat_and_close" if detected else "movement_or_spread",
            "spread": spread,
            "mean_total_movement": mean_movement,
            "n_agents": len(debate_beliefs),
        }

    async def _fire_disruptor(
        self,
        ledger: DiscussionLedger,
        plan: TaskPlan,
        round_number: int,
        diagnosis: dict[str, Any],
    ) -> bool:
        """Add an extra Devil's Advocate turn with an unused stratagem.

        Returns True if a disruption was issued, False if we couldn't pick
        a fresh stratagem (extremely unlikely).
        """
        await self.progress.on_progress(
            "aporia_detected",
            (
                "Aporía detectada en mitad de la deliberación: spread="
                f"{diagnosis.get('spread', 0):.2f}, movimiento medio="
                f"{diagnosis.get('mean_total_movement', 0):.2f}."
            ),
            diagnosis,
        )
        try:
            stratagem = random_stratagem(exclude=set(self._used_stratagems))
        except Exception as e:
            logger.warning("Disruptor stratagem selection failed: %s", e)
            return False
        self._used_stratagems.add(stratagem.id)

        await self.progress.on_progress(
            "drtag_dispatch",
            (
                f"Disruptor con estratagema #{stratagem.id} ({stratagem.name}) "
                f"interviene fuera de turno."
            ),
            {"stratagem_id": stratagem.id, "stratagem_name": stratagem.name},
        )
        await self._dispatch_one(
            ledger, "devils_advocate", plan, round_number=round_number
        )
        return True

    # ── Synthesis ───────────────────────────────────────────────────────

    async def _run_synthesis(
        self, ledger: DiscussionLedger, plan: TaskPlan
    ) -> str:
        """Run a single Synthesizer turn that reads the entire ledger."""
        ledger_view = format_for_prompt(ledger, viewer_role="synthesizer")
        description = (
            "Evalúa todas las intervenciones del panel bajo las pretensiones "
            "de validez habermasianas y emite el veredicto final sobre "
            f"{ledger.claim!r}."
        )
        prompt = build_round_prompt(
            goal=ledger.goal,
            task_description=description,
            round_number=ledger.current_round + 1,
            max_rounds=ledger.max_rounds,
            ledger_view=ledger_view,
            your_last_text="(no aplicable — eres el Sintetizador)",
        )

        await self.progress.on_progress(
            "synthesis_start",
            "Sintetizador deliberando sobre todas las intervenciones…",
            {"n_entries": len(ledger.entries)},
        )

        try:
            text, _meta = await self.executor.dispatch_one(
                role_id="synthesizer",
                prompt=prompt,
                plan=plan,
                round_number=ledger.current_round + 1,
                max_rounds=self.max_rounds,
                stratagem=None,
            )
        except Exception as e:
            logger.exception("Synthesis dispatch failed: %s", e)
            return f"[Synthesis failed: {e}]"

        # The synthesizer doesn't take a position, but we still log its turn
        # in the ledger so the frontend can render it under "Final round".
        ledger_append(
            ledger,
            role_id="synthesizer",
            agent_id="synthesizer",
            text=text,
            round_number=ledger.current_round + 1,
            belief_after=None,
            delta=None,
        )
        return text

    # ── Belief tracking (called by AgenticOrchestrator on each event) ──

    def update_belief_from_event(
        self,
        role_id: RoleId,
        log_odds: float,
        delta: float,
    ) -> None:
        """Project a worker's belief_update into the loop's compact summary."""
        b = self._beliefs.setdefault(role_id, _Belief())
        b.log_odds = log_odds
        b.last_delta = delta
        b.total_movement += abs(delta)
        b.turns += 1

    # ── Internal helpers ────────────────────────────────────────────────

    def _make_synthetic_plan(self, claim: str, goal: str) -> TaskPlan:
        """A minimal plan shell so PlanExecutor.dispatch_one has a `plan` arg.

        We don't actually execute this plan — we just need its `goal` and
        `claim` for persona configuration. The subtasks list is empty so
        no DAG walk happens.
        """
        return TaskPlan(
            goal=goal,
            claim=claim,
            subtasks=[
                # Placeholder — never executed; just makes the model valid.
                SubTask(
                    id="placeholder",
                    description="placeholder",
                    role_id="analyst",
                    required_skill="role_analyst",
                )
            ],
            max_workers=len(DEBATE_ROLES) + 1,
        )
