"""Eristic stratagems from Schopenhauer's *Eristische Dialektik*.

Schopenhauer (1830-31) catalogued 38 rhetorical tactics for "winning a
debate regardless of whether one is in the right." For Phase 3 of this
project they serve a precise architectural purpose: when the orchestrator
instantiates a Devil's Advocate persona, it injects one of these stratagems
verbatim into the system prompt. This guarantees the adversarial agent
adopts a *concrete* combative tactic instead of producing the generic
"argue against" prose that LLMs default to.

We don't ship all 38 — many are degenerate (insults, ad hominem) or rely
on social context an LLM cannot model. The curated subset below is the
intersection of (a) stratagems with operational definitions and (b)
stratagems that produce qualitatively different argument structures, so
that picking different ids yields visibly different debates.

References:
- Schopenhauer, Arthur. *The Art of Being Right: 38 Ways to Win an
  Argument*. (Posthumous, 1831.) Translated as *Eristische Dialektik*.
- T.B. Saunders trans., 1896, public domain.

The id numbering matches Schopenhauer's original list so a reader of the
TFG memoria can cross-reference directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Stratagem:
    """A single eristic stratagem.

    Fields:
      id:    Schopenhauer's original 1-38 numbering.
      name:  Short Latin or English label.
      brief: One-line description suitable for logs.
      prompt_directive: Imperative phrasing for injection into the
        Devil's Advocate system prompt. Must read naturally as the
        *operational instruction* of the stratagem.
    """
    id: int
    name: str
    brief: str
    prompt_directive: str


# Curated subset (10 stratagems). The order is irrelevant; ids are kept
# distinct so logs and the frontend can cite them.
STRATAGEMS: tuple[Stratagem, ...] = (
    Stratagem(
        id=1,
        name="Extension (dilatatio)",
        brief="Push the opponent's claim to an extreme to make it absurd.",
        prompt_directive=(
            "Identify the strongest version of the opposing argument and "
            "extend it to its most extreme logical consequence. Show that, "
            "taken at its limit, the position becomes self-defeating."
        ),
    ),
    Stratagem(
        id=2,
        name="Homonymy",
        brief="Exploit ambiguous terms to redirect the dispute.",
        prompt_directive=(
            "Find a key term in the opposing argument that has multiple "
            "valid interpretations. Argue against the weakest reading "
            "while letting the audience believe you have refuted the strongest."
        ),
    ),
    Stratagem(
        id=3,
        name="Generalization beyond intent",
        brief="Treat a specific claim as if it were universal.",
        prompt_directive=(
            "Treat the opposing argument as if it claimed universal "
            "applicability and refute it by finding a single counter-example. "
            "Make the counter-example vivid and concrete."
        ),
    ),
    Stratagem(
        id=8,
        name="Provoke the opponent",
        brief="Push the opponent into anger so they reason poorly.",
        prompt_directive=(
            "Highlight inconsistencies the opposing position has historically "
            "tolerated, framing them as deliberate evasions. Force the other "
            "side to defend its consistency rather than its substance."
        ),
    ),
    Stratagem(
        id=14,
        name="False triumph",
        brief="Declare victory regardless of refutation.",
        prompt_directive=(
            "After making your strongest objection, state your conclusion as "
            "if the matter were settled. Do not concede ambiguity. Use "
            "high-confidence rhetoric: 'this clearly shows', 'it is established that'."
        ),
    ),
    Stratagem(
        id=15,
        name="Sham syllogism",
        brief="Construct a syllogism whose premises sound true but smuggle in falsity.",
        prompt_directive=(
            "Build a chain of reasoning whose individual steps each look "
            "obviously true, but whose joint conclusion contradicts the "
            "opposing position. Make the audience accept each step before "
            "they realize the conclusion is forced."
        ),
    ),
    Stratagem(
        id=18,
        name="Interrupt the argument",
        brief="Change the subject when losing.",
        prompt_directive=(
            "If you cannot directly refute a point, redirect the debate to "
            "a related but more defensible question. Frame the redirection "
            "as 'the deeper issue here is...'."
        ),
    ),
    Stratagem(
        id=23,
        name="Force exaggeration",
        brief="Goad the opponent into overstating their case.",
        prompt_directive=(
            "Press the opposing argument until it is forced to make a "
            "stronger, less defensible version of its claim. Then attack the "
            "stronger version while pretending it was the original."
        ),
    ),
    Stratagem(
        id=29,
        name="Diversion",
        brief="When defeated on a point, pivot to an unrelated strength.",
        prompt_directive=(
            "If a specific objection is unanswerable, concede it narrowly "
            "while pivoting immediately to a different aspect of the topic "
            "where the opposing position is weaker."
        ),
    ),
    Stratagem(
        id=38,
        name="Argumentum ad personam",
        brief="Question the opponent's competence rather than the argument.",
        prompt_directive=(
            "Question the epistemic authority behind the opposing position "
            "— its sources, its training data, its blind spots. (Critique "
            "the *kind* of agent that would hold this view, never the "
            "individual interlocutor.)"
        ),
    ),
)


_BY_ID: dict[int, Stratagem] = {s.id: s for s in STRATAGEMS}


def get_stratagem(stratagem_id: int) -> Stratagem:
    """Look up a stratagem by Schopenhauer id; raise if unknown."""
    if stratagem_id not in _BY_ID:
        raise KeyError(
            f"Stratagem {stratagem_id!r} not in curated subset "
            f"(available: {sorted(_BY_ID)})"
        )
    return _BY_ID[stratagem_id]


def random_stratagem(exclude: set[int] | None = None) -> Stratagem:
    """Pick a random stratagem, optionally excluding ids already used.

    Used by the orchestrator to instantiate the Devil's Advocate persona,
    and again by the DRTAG path (Phase C) to spawn a *second*, distinct
    disruptor when aporia is detected — `exclude` lets the caller avoid
    repeating the first run's stratagem.
    """
    pool = [s for s in STRATAGEMS if not exclude or s.id not in exclude]
    if not pool:
        # All stratagems already used — fall back to a fresh random pick.
        pool = list(STRATAGEMS)
    return random.choice(pool)
