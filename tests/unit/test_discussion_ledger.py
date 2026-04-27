"""Unit tests for the Phase 4 DiscussionLedger and helpers."""

from __future__ import annotations

from agents.orchestrator.discussion_ledger import (
    append,
    build_self_position_section,
    detect_references,
    entries_in_round,
    filter_by_roles,
    format_for_prompt,
    latest_for_role,
    role_label,
    total_movement_per_role,
)
from common.models import DiscussionLedger, LedgerEntry


def _make_ledger() -> DiscussionLedger:
    return DiscussionLedger(claim="X is Y", goal="decide X", max_rounds=3)


class TestAppend:
    def test_assigns_monotonic_turn_ids(self):
        ledger = _make_ledger()
        e0 = append(ledger, role_id="analyst", agent_id="a", text="t0", round_number=1)
        e1 = append(ledger, role_id="seeker", agent_id="b", text="t1", round_number=1)
        assert e0.turn == 0
        assert e1.turn == 1
        assert ledger.entries == [e0, e1]

    def test_propagates_belief_fields(self):
        ledger = _make_ledger()
        e = append(
            ledger,
            role_id="analyst",
            agent_id="a",
            text="hello",
            round_number=1,
            belief_after=0.7,
            delta=0.3,
        )
        assert e.belief_after == 0.7
        assert e.delta == 0.3


class TestDetectReferences:
    def test_explicit_turn_mention(self):
        prior = [
            LedgerEntry(turn=0, round_number=1, role_id="analyst", agent_id="a", text="x"),
            LedgerEntry(turn=1, round_number=1, role_id="seeker", agent_id="b", text="y"),
        ]
        refs = detect_references("Como dijo en el turno 0, creo que…", prior)
        assert refs == [0]

    def test_role_label_mention(self):
        prior = [
            LedgerEntry(turn=0, round_number=1, role_id="analyst", agent_id="a", text="x"),
            LedgerEntry(turn=1, round_number=1, role_id="seeker", agent_id="b", text="y"),
            LedgerEntry(turn=2, round_number=2, role_id="analyst", agent_id="a", text="z"),
        ]
        # "Analista" should resolve to the most recent analyst turn (turn 2)
        refs = detect_references("El Analista dijo algo importante.", prior)
        assert 2 in refs

    def test_out_of_range_turn_ignored(self):
        prior = [
            LedgerEntry(turn=0, round_number=1, role_id="analyst", agent_id="a", text="x"),
        ]
        refs = detect_references("Ver turno 99", prior)
        assert refs == []

    def test_empty_text_returns_empty(self):
        assert detect_references("", []) == []


class TestLookups:
    def test_latest_for_role(self):
        ledger = _make_ledger()
        append(ledger, role_id="analyst", agent_id="a", text="first", round_number=1)
        append(ledger, role_id="analyst", agent_id="a", text="second", round_number=2)
        latest = latest_for_role(ledger, "analyst")
        assert latest is not None
        assert latest.text == "second"

    def test_latest_for_role_none(self):
        ledger = _make_ledger()
        append(ledger, role_id="seeker", agent_id="s", text="x", round_number=1)
        assert latest_for_role(ledger, "analyst") is None

    def test_entries_in_round(self):
        ledger = _make_ledger()
        append(ledger, role_id="analyst", agent_id="a", text="r1", round_number=1)
        append(ledger, role_id="seeker", agent_id="s", text="r1", round_number=1)
        append(ledger, role_id="analyst", agent_id="a", text="r2", round_number=2)
        r1 = entries_in_round(ledger, 1)
        assert len(r1) == 2

    def test_filter_by_roles(self):
        ledger = _make_ledger()
        append(ledger, role_id="analyst", agent_id="a", text="x", round_number=1)
        append(ledger, role_id="seeker", agent_id="s", text="y", round_number=1)
        append(ledger, role_id="devils_advocate", agent_id="d", text="z", round_number=1)
        out = filter_by_roles(ledger, ["analyst", "seeker"])
        assert {e.role_id for e in out} == {"analyst", "seeker"}


class TestTotalMovement:
    def test_sums_absolute_deltas_per_role(self):
        ledger = _make_ledger()
        append(ledger, role_id="seeker", agent_id="s", text="t", round_number=1, delta=0.5)
        append(ledger, role_id="seeker", agent_id="s", text="t", round_number=2, delta=-0.3)
        append(ledger, role_id="analyst", agent_id="a", text="t", round_number=1, delta=0.1)
        out = total_movement_per_role(ledger)
        assert out["seeker"] == 0.8
        assert out["analyst"] == 0.1

    def test_ignores_none_deltas(self):
        ledger = _make_ledger()
        append(ledger, role_id="synthesizer", agent_id="s", text="x", round_number=1)
        out = total_movement_per_role(ledger)
        assert out == {}


class TestFormatForPrompt:
    def test_empty_ledger(self):
        out = format_for_prompt(_make_ledger())
        assert "aún no hay" in out

    def test_includes_role_label_and_round(self):
        ledger = _make_ledger()
        append(
            ledger,
            role_id="analyst",
            agent_id="a",
            text="punto importante",
            round_number=1,
            delta=0.4,
        )
        out = format_for_prompt(ledger)
        assert "Analista" in out
        assert "Ronda 1" in out
        assert "punto importante" in out
        assert "+0.40" in out  # delta annotation

    def test_marks_viewers_own_entries(self):
        ledger = _make_ledger()
        append(ledger, role_id="analyst", agent_id="a", text="ana texto", round_number=1)
        append(ledger, role_id="seeker", agent_id="s", text="sek texto", round_number=1)
        out = format_for_prompt(ledger, viewer_role="analyst")
        # Find the analyst's entry and confirm it has a marker
        assert "►" in out

    def test_truncates_old_entries(self):
        ledger = _make_ledger()
        long_text = "X" * 2000
        append(ledger, role_id="analyst", agent_id="a", text=long_text, round_number=1)
        append(ledger, role_id="seeker", agent_id="s", text="recent", round_number=2)
        out = format_for_prompt(ledger, max_chars_per_entry=100)
        # Old analyst is truncated; recent seeker is not
        assert "…" in out
        assert "recent" in out


class TestBuildSelfPositionSection:
    def test_no_prior_returns_new_marker(self):
        ledger = _make_ledger()
        out = build_self_position_section(ledger, role_id="analyst")
        assert "nuevo" in out

    def test_returns_prior_text_with_round_marker(self):
        ledger = _make_ledger()
        append(ledger, role_id="analyst", agent_id="a", text="mi tesis", round_number=1)
        out = build_self_position_section(ledger, role_id="analyst")
        assert "ronda 1" in out
        assert "mi tesis" in out


class TestRoleLabel:
    def test_canonical_roles_have_spanish_labels(self):
        assert role_label("analyst") == "Analista"
        assert role_label("devils_advocate") == "Abogado del Diablo"
        assert role_label("synthesizer") == "Sintetizador"
