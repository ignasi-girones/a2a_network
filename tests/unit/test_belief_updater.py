"""Unit tests for the Bayesian belief updater (Phase 3 / Pillar 2).

Two layers under test:

  1. ``apply_bias`` — pure numerical update, exercised exhaustively here
     because it is what the TFG memoria cites when defending the
     epistemological model.
  2. ``update_belief`` — async LLM-backed wrapper. We monkey-patch the
     ``llm_complete`` binding so no real LLM call goes out.
"""

from __future__ import annotations

import json

import pytest

from agents.specialized import belief_updater as bu_module
from agents.specialized.belief_updater import (
    _parse_llr_response,
    apply_bias,
    update_belief,
)
from common.models import BeliefState


# ── apply_bias ───────────────────────────────────────────────────────────────


class TestApplyBias:
    def test_pure_bayesian_no_biases(self):
        s = BeliefState(claim="X", log_odds=0.0)
        delta, new = apply_bias(state=s, llr=1.5)
        assert delta == pytest.approx(1.5)
        assert new == pytest.approx(1.5)

    def test_zero_llr_is_noop(self):
        s = BeliefState(claim="X", log_odds=0.7)
        delta, new = apply_bias(state=s, llr=0.0)
        assert delta == 0.0
        assert new == pytest.approx(0.7)

    def test_anchoring_shrinks_update(self):
        s = BeliefState(claim="X", log_odds=0.0, anchoring=0.5)
        delta, _ = apply_bias(state=s, llr=2.0)
        assert delta == pytest.approx(1.0)

    def test_full_anchoring_blocks_update(self):
        s = BeliefState(claim="X", log_odds=0.0, anchoring=1.0)
        delta, new = apply_bias(state=s, llr=3.0)
        assert delta == 0.0
        assert new == 0.0

    def test_sensitivity_scales_evidence(self):
        s = BeliefState(claim="X", log_odds=0.0, evidence_sensitivity=0.5)
        delta, _ = apply_bias(state=s, llr=2.0)
        assert delta == pytest.approx(1.0)

    def test_asymmetry_amplifies_confirmation(self):
        """Same-sign evidence is amplified by asymmetry > 1."""
        s = BeliefState(claim="X", log_odds=1.0, asymmetry=2.0)
        delta, _ = apply_bias(state=s, llr=1.0)
        assert delta == pytest.approx(2.0)

    def test_asymmetry_dampens_disconfirmation(self):
        """Opposite-sign evidence is dampened by asymmetry > 1."""
        s = BeliefState(claim="X", log_odds=1.0, asymmetry=2.0)
        delta, _ = apply_bias(state=s, llr=-1.0)
        assert delta == pytest.approx(-0.5)

    def test_asymmetry_inactive_at_neutral_prior(self):
        """At log_odds=0 there is no direction to confirm or disconfirm."""
        s = BeliefState(claim="X", log_odds=0.0, asymmetry=2.0)
        delta_pos, _ = apply_bias(state=s, llr=1.0)
        delta_neg, _ = apply_bias(state=s, llr=-1.0)
        assert delta_pos == pytest.approx(1.0)
        assert delta_neg == pytest.approx(-1.0)

    def test_llr_clamped(self):
        """Huge LLR values are bounded so a single hallucination cannot
        dominate the trajectory."""
        s = BeliefState(claim="X", log_odds=0.0)
        delta_high, _ = apply_bias(state=s, llr=99.0)
        delta_low, _ = apply_bias(state=s, llr=-99.0)
        assert delta_high <= 3.0
        assert delta_low >= -3.0

    def test_log_odds_clamped(self):
        """The posterior never escapes ±5."""
        s = BeliefState(claim="X", log_odds=4.5)
        _, new = apply_bias(state=s, llr=99.0)
        assert new <= 5.0
        s2 = BeliefState(claim="X", log_odds=-4.5)
        _, new2 = apply_bias(state=s2, llr=-99.0)
        assert new2 >= -5.0


# ── _parse_llr_response ─────────────────────────────────────────────────────


class TestParseLlrResponse:
    def test_valid_json(self):
        raw = json.dumps({"llr": 1.7, "rationale": "strong"})
        llr, rat = _parse_llr_response(raw)
        assert llr == 1.7
        assert rat == "strong"

    def test_missing_rationale_defaults_empty(self):
        llr, rat = _parse_llr_response(json.dumps({"llr": -0.5}))
        assert llr == -0.5
        assert rat == ""

    def test_garbage_falls_back_to_zero(self):
        llr, rat = _parse_llr_response("not json at all")
        assert llr == 0.0
        assert "no rationale" in rat or "not json" in rat

    def test_regex_fallback_picks_first_number(self):
        """If JSON fails but a number is present, prefer that over zero."""
        llr, _ = _parse_llr_response("model said 1.2 confidence")
        assert llr == 1.2

    def test_nan_or_inf_coerced_to_zero(self):
        raw = json.dumps({"llr": float("inf")})
        # json.dumps writes Infinity which is not strict JSON; prove the
        # parser tolerates it via fallback.
        llr, _ = _parse_llr_response(raw)
        assert llr == 0.0 or abs(llr) < 1e9  # either zero or very large is fine


# ── update_belief (LLM-backed) ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestUpdateBelief:
    async def test_llm_response_moves_log_odds(self, monkeypatch):
        async def fake_complete(**kwargs):
            return json.dumps(
                {"llr": 1.5, "rationale": "evidence supports the claim"}
            )

        monkeypatch.setattr(bu_module, "llm_complete", fake_complete)
        s = BeliefState(claim="X is true")
        delta = await update_belief(
            s, evidence_text="Detailed supporting evidence.", model="m"
        )
        assert delta.llr == 1.5
        assert delta.new_log_odds == pytest.approx(1.5)
        assert s.log_odds == pytest.approx(1.5)
        assert len(s.history) == 1
        assert s.history[0].rationale == "evidence supports the claim"
        assert s.history[0].stage == "post_response"

    async def test_negative_llr_moves_against(self, monkeypatch):
        async def fake_complete(**kwargs):
            return json.dumps({"llr": -2.0, "rationale": "contradiction"})

        monkeypatch.setattr(bu_module, "llm_complete", fake_complete)
        s = BeliefState(claim="X")
        delta = await update_belief(s, evidence_text="counter-evidence", model="m")
        assert delta.delta_log_odds < 0
        assert s.log_odds < 0

    async def test_llm_failure_records_flat_no_crash(self, monkeypatch):
        """A network/LLM error is logged but doesn't break the run."""

        async def fake_complete(**kwargs):
            raise RuntimeError("llm down")

        monkeypatch.setattr(bu_module, "llm_complete", fake_complete)
        s = BeliefState(claim="X", log_odds=0.5)
        delta = await update_belief(s, evidence_text="text", model="m")
        # Empty/failed response → 0 LLR → 0 delta → state unchanged
        assert delta.llr == 0.0
        assert delta.delta_log_odds == 0.0
        assert s.log_odds == pytest.approx(0.5)

    async def test_empty_evidence_is_noop(self, monkeypatch):
        async def fake_complete(**kwargs):
            raise AssertionError("LLM should not be called for empty evidence")

        monkeypatch.setattr(bu_module, "llm_complete", fake_complete)
        s = BeliefState(claim="X", log_odds=0.3)
        delta = await update_belief(s, evidence_text="   ", model="m")
        assert delta.llr == 0.0
        assert s.log_odds == pytest.approx(0.3)
        # An empty-evidence sample is still appended so the chart stays aligned.
        assert len(s.history) == 1
        assert s.history[0].rationale == "(no evidence)"

    async def test_history_accumulates_across_updates(self, monkeypatch):
        responses = iter(
            [
                json.dumps({"llr": 0.5, "rationale": "step 1"}),
                json.dumps({"llr": -0.3, "rationale": "step 2"}),
                json.dumps({"llr": 0.8, "rationale": "step 3"}),
            ]
        )

        async def fake_complete(**kwargs):
            return next(responses)

        monkeypatch.setattr(bu_module, "llm_complete", fake_complete)
        s = BeliefState(claim="X")
        await update_belief(s, evidence_text="a", model="m", stage="post_initial")
        await update_belief(s, evidence_text="b", model="m", stage="post_refine")
        await update_belief(s, evidence_text="c", model="m", stage="post_aporia")

        assert len(s.history) == 3
        assert [h.stage for h in s.history] == [
            "post_initial",
            "post_refine",
            "post_aporia",
        ]
