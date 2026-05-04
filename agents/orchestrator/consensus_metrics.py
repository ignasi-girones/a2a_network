"""Empirical consensus metrics.

This module replaces the LLM-graded `agreement_score` with reproducible,
quantitative measurements derived directly from the agents' texts.

Why empirical?
    The previous approach asked an LLM to grade convergence and assign each
    agent a position on a 0..1 axis. The LLM was inconsistent — sometimes
    AE1 (which opened with stance A) ended up scored higher than AE2 (which
    opened with stance B) just because the LLM re-anchored the axis on each
    call. Embedding-based anchors fix that: AE1's opening text IS the
    geometric definition of position 0.0; AE2's opening IS position 1.0.
    Every subsequent text is placed by its cosine similarity relative to
    those two anchors.

Components of `agreement_score`:
    1. Dispersion           — geometric spread of positions on the AE1↔AE2 axis
    2. Pairwise similarity  — semantic similarity across the latest texts
    3. Movement             — how much each agent moved since the last round
    4. Concession markers   — explicit "you changed my mind" signals

Each component is in [0, 1]. The combined score is a weighted sum.

The LLM is still used SEPARATELY to extract `shared_points` and
`remaining_disagreements` (a textual summary) — there is no good algorithm
for that. But the score itself is no longer subjective.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Weights for the combined agreement_score (sum to 1.0) ──────────────────
W_DISPERSION = 0.40   # tighter cluster on the axis → higher score
W_SIMILARITY = 0.35   # texts more similar to each other → higher score
W_MOVEMENT = 0.15     # agents reacting to each other → higher score
W_CONCESSIONS = 0.10  # explicit concessions → higher score


# ── Concession markers (regex). Lower-case, ASCII-folded comparison. ───────
# Designed to fire on explicit signals that an agent has shifted stance.
# Kept conservative on purpose — false positives would inflate the score.
_CONCESSION_PATTERNS_ES = [
    r"\btienes?\s+raz[oó]n\b",
    r"\bme\s+has\s+convencido\b",
    r"\bcambias?t?e?\s+mi\s+opini[oó]n\b",
    r"\bahora\s+(?:estoy\s+de\s+acuerdo|coincido|comparto)\b",
    r"\bconcedo\b",
    r"\bconcedido\b",
    r"\bdebo\s+admitir\b",
    r"\bme\s+retracto\b",
    r"\bacepto\s+que\b",
    r"\breconozco\s+que\b",
]
_CONCESSION_PATTERNS_EN = [
    r"\byou\s+changed\s+my\s+mind\b",
    r"\byou'?re\s+right\b",
    r"\bi\s+(?:now\s+)?agree\b",
    r"\bi\s+concede\b",
    r"\bi\s+stand\s+corrected\b",
    r"\bi\s+was\s+wrong\b",
    r"\bi\s+have\s+to\s+admit\b",
    r"\bgood\s+point\b",
]
_CONCESSION_REGEX = re.compile(
    "|".join(_CONCESSION_PATTERNS_ES + _CONCESSION_PATTERNS_EN),
    re.IGNORECASE,
)


# ── Public dataclass ───────────────────────────────────────────────────────


@dataclass
class ConsensusMetrics:
    """Container for one round's empirical evaluation."""

    positions: dict[str, float]                  # ae1/ae2/ae3 → [0, 1]
    dispersion: float                            # max(positions) - min(positions)
    pairwise_similarity: float                   # mean cosine sim across pairs
    movement: dict[str, float]                   # ae* → distance moved since prev round
    movement_score: float                        # 0..1 normalised aggregate of movement
    concessions: dict[str, int]                  # ae* → count of concession markers
    concession_score: float                      # 0..1 aggregate
    agreement_score: float                       # combined 0..1
    components: dict[str, float] = field(default_factory=dict)  # raw component values, for the UI


# ── Vector helpers ────────────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Standard cosine similarity in [-1, 1]. Handles zero vectors safely."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ── Position anchoring ────────────────────────────────────────────────────


def position_on_ae1_ae2_axis(
    text_emb: list[float],
    ae1_anchor_emb: list[float],
    ae2_anchor_emb: list[float],
) -> float:
    """Place a text on the AE1↔AE2 axis using cosine similarity to anchors.

    Definition:
        position = sim_to_AE2 / (sim_to_AE1 + sim_to_AE2)

    With this formula:
        - A text identical to AE1's opening → position close to 0.0.
        - A text identical to AE2's opening → position close to 1.0.
        - A text equally close to both     → position ≈ 0.5.
        - A text closer to AE2 than to AE1 → position > 0.5  (i.e. has
          drifted toward AE2's side, regardless of which agent emitted it).

    Cosine similarities are clamped to [0, 1] before combining (negative
    similarities, rare with these models, would otherwise distort the
    ratio). If both anchors are equidistant we return 0.5.
    """
    s1 = max(0.0, cosine_similarity(text_emb, ae1_anchor_emb))
    s2 = max(0.0, cosine_similarity(text_emb, ae2_anchor_emb))
    total = s1 + s2
    if total == 0:
        return 0.5
    return _clamp01(s2 / total)


# ── Per-component metrics ─────────────────────────────────────────────────


def dispersion(positions: dict[str, float]) -> float:
    """`max(positions) - min(positions)`. 0 = identical positions; 1 = polar."""
    if len(positions) < 2:
        return 0.0
    vals = list(positions.values())
    return max(vals) - min(vals)


def pairwise_similarity_mean(embeddings: dict[str, list[float]]) -> float:
    """Mean cosine similarity across every pair of agents.

    Clamped to [0, 1] before averaging. Returns 0 if fewer than 2 agents.
    """
    keys = list(embeddings.keys())
    if len(keys) < 2:
        return 0.0
    sims: list[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sims.append(
                max(0.0, cosine_similarity(embeddings[keys[i]], embeddings[keys[j]]))
            )
    return _clamp01(sum(sims) / len(sims))


def movement_per_agent(
    current: dict[str, float],
    previous: dict[str, float] | None,
) -> dict[str, float]:
    """Absolute distance each agent moved on the axis since the previous round.

    If there is no previous snapshot, every movement is 0.0.
    """
    if not previous:
        return {tag: 0.0 for tag in current}
    return {
        tag: abs(current[tag] - previous.get(tag, current[tag]))
        for tag in current
    }


def movement_score(movement: dict[str, float]) -> float:
    """Aggregate movement into a single 0..1 score.

    Rationale: any movement at all is a weak positive signal that agents
    are reacting to each other (the alternative is a stuck debate).
    Movement of ~0.25 already saturates the score — beyond that it doesn't
    matter how much further they swung. Returns 0.0 on the opening round.
    """
    if not movement:
        return 0.0
    avg = sum(movement.values()) / len(movement)
    return _clamp01(avg / 0.25)


def count_concessions(text: str) -> int:
    """Count concession markers in a single text."""
    if not text:
        return 0
    return len(_CONCESSION_REGEX.findall(text))


def concession_score(concessions: dict[str, int]) -> float:
    """Map raw concession counts into [0, 1].

    A single explicit concession (count = 1 from any agent) already gives
    a meaningful signal; we saturate around an average of 2 markers per
    agent so a chatty model doesn't dominate the score.
    """
    if not concessions:
        return 0.0
    avg = sum(concessions.values()) / len(concessions)
    return _clamp01(avg / 2.0)


# ── Top-level computation ─────────────────────────────────────────────────


def compute_metrics(
    *,
    embeddings: dict[str, list[float]],
    texts: dict[str, str],
    ae1_anchor_emb: list[float],
    ae2_anchor_emb: list[float],
    previous_positions: dict[str, float] | None = None,
) -> ConsensusMetrics:
    """Compute all empirical metrics for one round.

    `embeddings` and `texts` must share the same keys (agent tags such as
    "ae1", "ae2", "ae3"). Anchors are the embeddings of AE1 and AE2's
    OPENING texts — captured once at the start of the deliberation and
    reused for every subsequent round so the axis stays stable.
    """
    # 1. Anchored positions on the AE1↔AE2 axis.
    positions = {
        tag: position_on_ae1_ae2_axis(emb, ae1_anchor_emb, ae2_anchor_emb)
        for tag, emb in embeddings.items()
    }

    # 2. Dispersion of positions.
    disp = dispersion(positions)
    disp_score = _clamp01(1.0 - disp)

    # 3. Pairwise semantic similarity across agents.
    sim = pairwise_similarity_mean(embeddings)

    # 4. Movement since previous round.
    movement = movement_per_agent(positions, previous_positions)
    mov_score = movement_score(movement)

    # 5. Concession markers in the texts.
    concessions = {tag: count_concessions(t) for tag, t in texts.items()}
    conc_score = concession_score(concessions)

    # Combined score.
    agreement = (
        W_DISPERSION * disp_score
        + W_SIMILARITY * sim
        + W_MOVEMENT * mov_score
        + W_CONCESSIONS * conc_score
    )
    agreement = _clamp01(agreement)

    components = {
        "dispersion": disp,
        "dispersion_score": disp_score,
        "pairwise_similarity": sim,
        "movement_score": mov_score,
        "concession_score": conc_score,
        "weights": {
            "dispersion": W_DISPERSION,
            "similarity": W_SIMILARITY,
            "movement": W_MOVEMENT,
            "concessions": W_CONCESSIONS,
        },
    }

    return ConsensusMetrics(
        positions=positions,
        dispersion=disp,
        pairwise_similarity=sim,
        movement=movement,
        movement_score=mov_score,
        concessions=concessions,
        concession_score=conc_score,
        agreement_score=agreement,
        components=components,
    )
