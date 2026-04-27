"""Unit tests for the Phase 4 DeliberationLoop.

We monkey-patch PlanExecutor.dispatch_one so no real LLM/A2A traffic
happens, then exercise the loop's mechanics: speaker selection, round
sequencing, aporia detection, max_rounds enforcement, and the final
synthesis turn.
"""

from __future__ import annotations

import pytest

from agents.orchestrator.deliberation_loop import (
    DEBATE_ROLES,
    MAX_TURNS_PER_AGENT,
    DeliberationLoop,
)
from agents.orchestrator.plan_executor import PlanExecutor, ProgressCallback


pytestmark = pytest.mark.asyncio


class _CollectingProgress(ProgressCallback):
    def __init__(self):
        self.events: list[tuple[str, str, dict | None]] = []

    async def on_progress(self, stage, message, data=None):
        self.events.append((stage, message, data))


class _StubExecutor(PlanExecutor):
    """PlanExecutor that doesn't talk to the network — returns canned text."""

    def __init__(self):
        # Skip super().__init__ — we don't need a registry for these tests.
        self.calls: list[dict] = []
        self._answers_by_role: dict[str, list[str]] = {}

    def queue(self, role_id: str, *texts: str) -> None:
        self._answers_by_role.setdefault(role_id, []).extend(texts)

    async def dispatch_one(self, *, role_id, prompt, plan, round_number, max_rounds, stratagem=None):
        self.calls.append(
            {
                "role_id": role_id,
                "round_number": round_number,
                "stratagem": stratagem,
                "prompt": prompt,
            }
        )
        queue = self._answers_by_role.get(role_id, [])
        text = queue.pop(0) if queue else f"[{role_id} round {round_number}]"
        meta = {
            "display_name": role_id,
            "role_id": role_id,
            "stratagem_id": stratagem.id if stratagem else None,
            "tool_whitelist": [],
            "temperature": 0.5,
        }
        return text, meta


class TestRound1Opening:
    async def test_all_debate_roles_speak_in_round_one(self):
        executor = _StubExecutor()
        loop = DeliberationLoop(executor, _CollectingProgress(), max_rounds=1)
        # Force every belief to "moved a lot" so no follow-up is needed,
        # making this a 1-round test (round 2 would terminate via consensus).
        ledger, verdict = await loop.run(claim="X", goal="g")

        # 5 debate roles + 1 synthesizer = 6 calls
        assert len(executor.calls) == 6
        roles_called = [c["role_id"] for c in executor.calls]
        # First 5 are the debate-side roles in canonical order
        assert set(roles_called[:5]) == set(DEBATE_ROLES)
        # Synthesizer is last
        assert roles_called[-1] == "synthesizer"
        assert "synthesizer" in verdict or verdict.startswith("[")

    async def test_ledger_records_every_intervention(self):
        executor = _StubExecutor()
        executor.queue("analyst", "ANALYST_TEXT")
        loop = DeliberationLoop(executor, _CollectingProgress(), max_rounds=1)
        ledger, _ = await loop.run(claim="X", goal="g")
        # 5 debate + 1 synthesizer = 6 ledger entries
        assert len(ledger.entries) == 6
        analyst_entries = [e for e in ledger.entries if e.role_id == "analyst"]
        assert analyst_entries[0].text == "ANALYST_TEXT"


class TestSpeakerSelection:
    async def test_da_always_speaks_in_followup_rounds(self):
        executor = _StubExecutor()
        progress = _CollectingProgress()
        loop = DeliberationLoop(executor, progress, max_rounds=2)
        # Project beliefs that make every non-DA role "satisfied" (moved a lot)
        for role in DEBATE_ROLES:
            loop.update_belief_from_event(role, log_odds=2.0, delta=2.0)
        # But with one non-DA stuck so we can prove DA + stuck both speak
        loop.update_belief_from_event("analyst", log_odds=2.0, delta=0.1)

        ledger, _ = await loop.run(claim="X", goal="g")
        # Round 1 always has 5 debate. Round 2 may have DA + analyst.
        round_2_calls = [c for c in executor.calls if c["round_number"] == 2]
        round_2_roles = [c["role_id"] for c in round_2_calls]
        assert "devils_advocate" in round_2_roles

    async def test_max_turns_per_agent_caps_da(self):
        executor = _StubExecutor()
        loop = DeliberationLoop(executor, _CollectingProgress(), max_rounds=10)
        # Pretend DA already hit the cap before the loop ran further rounds.
        for _ in range(MAX_TURNS_PER_AGENT):
            loop.update_belief_from_event("devils_advocate", log_odds=-1.0, delta=0.0)
        # Selection should now exclude DA even though it's "stuck".
        speakers = loop._select_speakers(_make_filled_ledger())
        assert "devils_advocate" not in speakers


def _make_filled_ledger():
    """Helper: ledger with one entry so _select_speakers doesn't see it as empty."""
    from common.models import DiscussionLedger, LedgerEntry

    ledger = DiscussionLedger(claim="X", goal="g", max_rounds=3)
    ledger.entries.append(
        LedgerEntry(turn=0, round_number=1, role_id="analyst", agent_id="a", text="t")
    )
    return ledger


class TestAporia:
    async def test_aporia_fires_disruptor(self):
        executor = _StubExecutor()
        progress = _CollectingProgress()
        loop = DeliberationLoop(executor, progress, max_rounds=2)
        # Set every debate role at log_odds≈0 with delta≈0 → aporia
        for role in DEBATE_ROLES:
            loop.update_belief_from_event(role, log_odds=0.0, delta=0.0)
        # Run round 2 selection / dispatch (Round 1 will fill in too)
        ledger, _ = await loop.run(claim="X", goal="g")
        stages = [e[0] for e in progress.events]
        # The detector should fire after round 2
        assert "aporia_detected" in stages or "drtag_dispatch" in stages


class TestSynthesisRunsLast:
    async def test_synthesizer_sees_full_ledger(self):
        executor = _StubExecutor()
        loop = DeliberationLoop(executor, _CollectingProgress(), max_rounds=1)
        ledger, _ = await loop.run(claim="claim X", goal="goal G")
        synth_call = next(c for c in executor.calls if c["role_id"] == "synthesizer")
        # The synthesizer's prompt must include the full ledger (every prior role)
        for role in ("Analista", "Buscador", "Empírico", "Pragmático"):
            assert role in synth_call["prompt"], f"missing {role}"
        # And the central claim
        assert "claim X" in synth_call["prompt"]


class TestBeliefProjection:
    async def test_update_belief_from_event_accumulates(self):
        loop = DeliberationLoop(
            _StubExecutor(), _CollectingProgress(), max_rounds=1
        )
        loop.update_belief_from_event("analyst", log_odds=0.5, delta=0.5)
        loop.update_belief_from_event("analyst", log_odds=0.2, delta=-0.3)
        b = loop._beliefs["analyst"]
        assert b.log_odds == 0.2
        assert b.last_delta == -0.3
        # total_movement is sum of |delta|
        assert b.total_movement == pytest.approx(0.8)
        assert b.turns == 2
