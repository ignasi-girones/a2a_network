"""Unit tests for Phase 3 / Pillar 3 — DRTAG aporia detection.

The detector consumes the per-agent belief log captured by
``_BeliefRecordingProgress`` and decides whether the panel is in the
"flat-and-close" stuck state. These tests pin the calibration: legitimate
debates (with movement) must NOT trip it, and stuck debates MUST.
"""

from __future__ import annotations

from agents.orchestrator.agentic_orchestrator import (
    APORIA_LOG_ODDS_SPREAD_EPS,
    APORIA_TOTAL_MOVEMENT_EPS,
    _detect_aporia,
    _split_habermas_table,
)


# ── _detect_aporia ──────────────────────────────────────────────────────────


class TestDetectAporiaPositive:
    def test_flat_and_close_triggers(self):
        beliefs = {
            "analyst": [
                {"log_odds": 0.05, "delta": 0.05, "role_id": "analyst"},
            ],
            "seeker": [
                {"log_odds": 0.10, "delta": 0.10, "role_id": "seeker"},
            ],
            "devils_advocate": [
                {"log_odds": 0.0, "delta": 0.0, "role_id": "devils_advocate"},
            ],
        }
        out = _detect_aporia(beliefs)
        assert out["detected"] is True
        assert out["reason"] == "flat_and_close"
        assert out["n_agents"] == 3
        assert out["spread"] < APORIA_LOG_ODDS_SPREAD_EPS
        assert out["mean_total_movement"] < APORIA_TOTAL_MOVEMENT_EPS


class TestDetectAporiaNegative:
    def test_wide_spread_blocks_aporia(self):
        """If beliefs spread far apart the panel disagreed — not stuck."""
        beliefs = {
            "analyst": [
                {"log_odds": 1.5, "delta": 1.5, "role_id": "analyst"},
            ],
            "devils_advocate": [
                {"log_odds": -1.5, "delta": -1.5, "role_id": "devils_advocate"},
            ],
        }
        out = _detect_aporia(beliefs)
        assert out["detected"] is False

    def test_high_movement_blocks_aporia(self):
        """All agents moved >> threshold even though they ended close —
        legitimate convergence, not a stuck state."""
        beliefs = {
            "analyst": [
                {"log_odds": 1.0, "delta": 1.0, "role_id": "analyst"},
                {"log_odds": 0.1, "delta": -0.9, "role_id": "analyst"},
            ],
            "seeker": [
                {"log_odds": -0.8, "delta": -0.8, "role_id": "seeker"},
                {"log_odds": 0.2, "delta": 1.0, "role_id": "seeker"},
            ],
        }
        out = _detect_aporia(beliefs)
        assert out["detected"] is False
        assert out["spread"] < APORIA_LOG_ODDS_SPREAD_EPS  # close…
        assert out["mean_total_movement"] >= APORIA_TOTAL_MOVEMENT_EPS  # …but moved

    def test_synthesizer_role_excluded_from_decision(self):
        """The Synthesizer doesn't take a position — its samples must not
        count toward aporia.

        Here the analyst is moving wildly (legitimate); only the
        synthesizer is flat. The detector must look past the synthesizer
        and report movement / spread from the analyst alone."""
        beliefs = {
            "synthesizer": [
                {"log_odds": 0.0, "delta": 0.0, "role_id": "synthesizer"},
            ],
            "analyst": [
                {"log_odds": 2.5, "delta": 2.5, "role_id": "analyst"},
            ],
        }
        out = _detect_aporia(beliefs)
        # Only one debate-side agent → "insufficient_debate_signal"
        assert out["detected"] is False
        assert out["n_agents"] == 1
        assert out["reason"] == "insufficient_debate_signal"

    def test_empty_beliefs_safe(self):
        out = _detect_aporia({})
        assert out["detected"] is False
        assert out["n_agents"] == 0


# ── _split_habermas_table ───────────────────────────────────────────────────


class TestSplitHabermasTable:
    def test_no_delimiter_returns_whole_text(self):
        verdict, claims = _split_habermas_table("Plain markdown verdict.")
        assert verdict == "Plain markdown verdict."
        assert claims is None

    def test_well_formed_split(self):
        text = (
            "Final markdown verdict.\n\n"
            "---HABERMAS-JSON---\n"
            '{"validity_claims": ['
            '{"agent": "Analista", "truth": 0.9, "rightness": 0.8, '
            '"sincerity": 0.85, "comprehensibility": 1.0, "admitted": true,'
            '"note": "ok"}]}'
        )
        verdict, claims = _split_habermas_table(text)
        assert verdict == "Final markdown verdict."
        assert claims is not None
        assert len(claims) == 1
        assert claims[0]["agent"] == "Analista"
        assert claims[0]["admitted"] is True

    def test_handles_code_fence(self):
        text = (
            "Markdown.\n---HABERMAS-JSON---\n"
            "```json\n"
            '{"validity_claims": [{"agent": "X"}]}\n'
            "```"
        )
        verdict, claims = _split_habermas_table(text)
        assert verdict == "Markdown."
        assert claims is not None
        assert claims[0]["agent"] == "X"

    def test_malformed_json_falls_back_gracefully(self):
        text = (
            "Markdown verdict.\n---HABERMAS-JSON---\nnot a json object"
        )
        verdict, claims = _split_habermas_table(text)
        assert verdict == "Markdown verdict."
        assert claims is None

    def test_missing_required_keys_drops_entries(self):
        text = (
            "Verdict.\n---HABERMAS-JSON---\n"
            '{"validity_claims": ['
            '{"agent": "A", "truth": 1},'
            '{"truth": 0.5}]}'  # second entry has no "agent"
        )
        verdict, claims = _split_habermas_table(text)
        assert verdict == "Verdict."
        assert claims is not None
        assert len(claims) == 1
        assert claims[0]["agent"] == "A"
