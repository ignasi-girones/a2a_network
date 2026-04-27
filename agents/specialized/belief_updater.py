"""Bayesian belief updater — Phase 3 / Pillar 2.

Each Specialized Agent maintains a ``BeliefState`` (log-odds of a central
claim). After every LLM intervention (initial argument, refined response,
DRTAG retry), this module:

  1. Calls a lateral LLM to evaluate the *log-likelihood ratio* of the
     latest evidence under the claim (positive = supports claim,
     negative = supports its negation).
  2. Combines that LLR with the persona's bias parameters (evidence
     sensitivity, anchoring, asymmetry) to produce a delta_log_odds.
  3. Appends a BeliefSample to ``state.history`` so the frontend's
     ``BeliefTrajectoryChart`` can draw the per-agent time series.

The "rationale" returned by the lateral LLM is preserved in the sample
so a hover on the frontend reveals *why* the belief moved, satisfying
the auditability requirement of the TFG memoria.

Cost note: the lateral call uses a small token budget (≤200) and a
fast model — it is called at most twice per agent per run. Total
overhead is O(8) extra LLM calls per debate, well within rate limits.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass

from common.llm_provider import llm_complete
from common.models import BeliefSample, BeliefState

logger = logging.getLogger(__name__)


# ── Numerical bounds ────────────────────────────────────────────────────────
# log_odds=±5 corresponds to ≈99.3% / 0.7% probability — beyond this point
# additional evidence cannot meaningfully move the posterior, so we clamp.
_LOG_ODDS_MIN = -5.0
_LOG_ODDS_MAX = 5.0
# A single LLM call should not produce a swing larger than this; protects
# against outlier hallucinations.
_LLR_CLAMP = 3.0


# ── Lateral LLM prompt ──────────────────────────────────────────────────────

_BELIEF_SYSTEM_PROMPT = """\
You are a Bayesian evidence-evaluator embedded in a multi-agent debate.

You will receive a CENTRAL CLAIM and a piece of TEXT (an agent's
contribution). Estimate the log-likelihood ratio of the TEXT under the
hypothesis "the claim is TRUE" versus "the claim is FALSE":

    llr = ln P(text | claim true) - ln P(text | claim false)

Return ONLY a JSON object of the form:
{
  "llr": <number between -3 and +3>,
  "rationale": "<one short Spanish sentence explaining your estimate>"
}

Calibration:
  +2.0 to +3.0 — text presents strong, specific, novel evidence FOR the claim.
  +0.5 to +1.5 — text presents weak or known evidence FOR the claim.
   0.0         — text is neutral, restates the claim, or is non-evidential.
  -0.5 to -1.5 — text presents weak evidence AGAINST the claim.
  -2.0 to -3.0 — text presents strong, specific evidence AGAINST the claim.

Do not flatter the text. Do not echo its tone. Be quantitative."""


@dataclass(frozen=True)
class BeliefDelta:
    """Result of a single belief update — what was applied and why."""
    llr: float           # raw log-likelihood ratio from the lateral LLM
    delta_log_odds: float  # signed amount actually applied (after bias factors)
    new_log_odds: float
    rationale: str


# ── Pure (numerical) update — testable without an LLM ──────────────────────


def apply_bias(
    *,
    state: BeliefState,
    llr: float,
) -> tuple[float, float]:
    """Apply persona bias factors to a raw LLR; return (delta, new_log_odds).

    Pure function, no I/O — testable with sample inputs.

    Model:
      effective_llr = clamp(llr, ±LLR_CLAMP) * evidence_sensitivity
      if effective_llr * sign(state.log_odds) > 0:
          # confirmatory evidence → amplify by asymmetry (≥1)
          effective_llr *= asymmetry
      else:
          # disconfirmatory → dampen
          effective_llr /= asymmetry
      delta = effective_llr * (1 - anchoring)  # anchoring shrinks updates
      new_log_odds = clamp(state.log_odds + delta, ±LOG_ODDS_MAX)
    """
    bounded = max(-_LLR_CLAMP, min(_LLR_CLAMP, llr))
    effective = bounded * state.evidence_sensitivity

    # Asymmetry only kicks in once the agent has *some* prior direction.
    # At the neutral prior (log_odds=0) the agent updates symmetrically.
    if state.log_odds != 0.0 and state.asymmetry > 0:
        same_sign = (effective * state.log_odds) > 0
        if same_sign:
            effective *= state.asymmetry
        else:
            effective /= state.asymmetry

    delta = effective * max(0.0, 1.0 - state.anchoring)
    new_log_odds = max(
        _LOG_ODDS_MIN, min(_LOG_ODDS_MAX, state.log_odds + delta)
    )
    return delta, new_log_odds


# ── LLM-backed update ───────────────────────────────────────────────────────


def _parse_llr_response(raw: str) -> tuple[float, str]:
    """Pull (llr, rationale) out of the lateral LLM's JSON response.

    Falls back to (0.0, raw) on malformed input — the executor would
    rather log a flat update than crash a debate run.
    """
    try:
        data = json.loads(raw)
        llr = float(data.get("llr", 0.0))
        rationale = str(data.get("rationale", "")).strip()
        if not math.isfinite(llr):
            llr = 0.0
        return llr, rationale
    except (json.JSONDecodeError, TypeError, ValueError):
        # Try a regex fallback: pull the first signed float we can find.
        m = re.search(r"-?\d+(\.\d+)?", raw or "")
        if m:
            try:
                return float(m.group(0)), raw[:160].strip()
            except ValueError:
                pass
        logger.warning("Belief updater could not parse LLR from %r", raw[:80])
        return 0.0, "(no rationale parsed)"


async def update_belief(
    state: BeliefState,
    *,
    evidence_text: str,
    model: str,
    stage: str = "post_response",
) -> BeliefDelta:
    """Update ``state`` in place after observing new evidence.

    Calls the lateral LLM (small budget) to evaluate the LLR, applies
    the persona bias factors, and appends a ``BeliefSample`` to
    ``state.history``.

    Returns the BeliefDelta so the executor can emit a frontend event.
    """
    if not evidence_text.strip():
        # Empty evidence is a no-op; record a flat sample so the chart
        # still ticks.
        sample = BeliefSample(
            log_odds=state.log_odds,
            delta=0.0,
            rationale="(no evidence)",
            stage=stage,
        )
        state.history.append(sample)
        return BeliefDelta(
            llr=0.0,
            delta_log_odds=0.0,
            new_log_odds=state.log_odds,
            rationale="(no evidence)",
        )

    user_prompt = (
        f"CENTRAL CLAIM:\n{state.claim}\n\n"
        f"TEXT TO EVALUATE:\n{evidence_text[:1800]}\n\n"
        "Return the JSON now."
    )
    try:
        raw = await llm_complete(
            model=model,
            messages=[
                {"role": "system", "content": _BELIEF_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # we want stable, calibrated numbers
            max_tokens=200,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning("Belief LLM call failed: %s", e)
        raw = "{}"

    llr, rationale = _parse_llr_response(raw)
    delta, new_log_odds = apply_bias(state=state, llr=llr)
    state.log_odds = new_log_odds
    state.history.append(
        BeliefSample(
            log_odds=new_log_odds,
            delta=delta,
            rationale=rationale,
            stage=stage,
        )
    )
    return BeliefDelta(
        llr=llr,
        delta_log_odds=delta,
        new_log_odds=new_log_odds,
        rationale=rationale,
    )
