"""Pure operations over the Phase 4 DiscussionLedger.

The ledger is a shared blackboard — every agent reads the entire history
on each round. These helpers keep all serialisation / view-building logic
in one place so the DeliberationLoop, the prompt builder, and the test
suite agree on the format.

Nothing here does I/O. Nothing here mutates the ledger except the explicit
``append`` helper. All other functions return derived views.
"""

from __future__ import annotations

import re
from typing import Iterable

from common.models import DiscussionLedger, LedgerEntry, RoleId


# ── Role display labels (kept here to avoid circular imports with the
# persona catalog at module load time) ─────────────────────────────────────
_ROLE_LABELS: dict[RoleId, str] = {
    "analyst": "Analista",
    "seeker": "Buscador",
    "devils_advocate": "Abogado del Diablo",
    "empiricist": "Empírico",
    "pragmatist": "Pragmático",
    "synthesizer": "Sintetizador",
}


def role_label(role_id: RoleId) -> str:
    """Spanish display label for a canonical role."""
    return _ROLE_LABELS.get(role_id, role_id)


def append(
    ledger: DiscussionLedger,
    *,
    role_id: RoleId,
    agent_id: str,
    text: str,
    round_number: int,
    belief_after: float | None = None,
    delta: float | None = None,
) -> LedgerEntry:
    """Append a new entry, computing turn id + reference resolution."""
    entry = LedgerEntry(
        turn=len(ledger.entries),
        round_number=round_number,
        role_id=role_id,
        agent_id=agent_id,
        text=text,
        belief_after=belief_after,
        delta=delta,
        references=detect_references(text, ledger.entries),
    )
    ledger.entries.append(entry)
    return entry


def detect_references(text: str, prior_entries: list[LedgerEntry]) -> list[int]:
    """Heuristically detect which prior turns this text reacts to.

    Two cheap signals:
      1. Explicit ``turno N`` / ``turn N`` mention → that turn id.
      2. Mentions of a prior author's role label (case-insensitive),
         e.g. "como dijo el Buscador..." → most recent Buscador turn.

    The result is best-effort — it powers the frontend's clickable cross-
    references but is not load-bearing for any backend logic.
    """
    refs: set[int] = set()
    if not text or not prior_entries:
        return []

    # Explicit "turno N"
    for m in re.finditer(r"\bturno\s+(\d+)\b", text, flags=re.IGNORECASE):
        try:
            n = int(m.group(1))
            if 0 <= n < len(prior_entries):
                refs.add(n)
        except ValueError:
            continue

    # Role mentions → latest entry by that role
    for label, role_id in [(v, k) for k, v in _ROLE_LABELS.items()]:
        if re.search(rf"\b{re.escape(label)}\b", text, flags=re.IGNORECASE):
            latest = next(
                (e for e in reversed(prior_entries) if e.role_id == role_id),
                None,
            )
            if latest is not None:
                refs.add(latest.turn)

    return sorted(refs)


def latest_for_role(
    ledger: DiscussionLedger, role_id: RoleId
) -> LedgerEntry | None:
    """The most recent entry written by any agent of the given role."""
    for e in reversed(ledger.entries):
        if e.role_id == role_id:
            return e
    return None


def entries_in_round(
    ledger: DiscussionLedger, round_number: int
) -> list[LedgerEntry]:
    """All entries written in a given round (chronological)."""
    return [e for e in ledger.entries if e.round_number == round_number]


def total_movement_per_role(
    ledger: DiscussionLedger,
) -> dict[RoleId, float]:
    """Sum of |delta| per role across all rounds.

    Used by the speaker-selection algorithm: an agent that hasn't moved
    over the last round is a good candidate to speak again because either
    (a) it's stuck and has more to say, or (b) it's anchored and needs to
    be poked by a fresh peer's intervention.
    """
    totals: dict[RoleId, float] = {}
    for e in ledger.entries:
        if e.delta is None:
            continue
        totals[e.role_id] = totals.get(e.role_id, 0.0) + abs(e.delta)
    return totals


def format_for_prompt(
    ledger: DiscussionLedger,
    *,
    viewer_role: RoleId | None = None,
    max_chars_per_entry: int = 600,
) -> str:
    """Render the ledger as a single block of text suitable for an LLM prompt.

    Each entry is rendered as:
      [turno N · Ronda R · Rol] (▲ +0.3) <— marker only when delta is set
      <text, truncated>

    Parameters
    ----------
    viewer_role
        If provided, the viewer's own entries are marked with a ``►``
        prefix so the LLM can locate its own prior position immediately.
    max_chars_per_entry
        Older entries get truncated to keep the context window in budget;
        the most recent entry is never truncated.
    """
    if not ledger.entries:
        return "(aún no hay intervenciones)"

    lines: list[str] = []
    last_idx = len(ledger.entries) - 1
    for i, e in enumerate(ledger.entries):
        marker = "►" if (viewer_role is not None and e.role_id == viewer_role) else " "
        delta_tag = ""
        if e.delta is not None:
            sign = "+" if e.delta >= 0 else ""
            delta_tag = f" (Δ log-odds {sign}{e.delta:.2f})"
        header = (
            f"{marker} [turno {e.turn} · Ronda {e.round_number} · "
            f"{role_label(e.role_id)}]{delta_tag}"
        )

        text = e.text.strip()
        # The most recent entry is never truncated — agents must respond to
        # the freshest signal in full.
        if i != last_idx and len(text) > max_chars_per_entry:
            text = text[:max_chars_per_entry].rstrip() + "…"
        lines.append(header)
        lines.append(text)
        lines.append("")  # blank separator
    return "\n".join(lines).rstrip()


def build_self_position_section(
    ledger: DiscussionLedger,
    *,
    role_id: RoleId,
    max_chars: int = 800,
) -> str:
    """Render the viewer's own prior-round entry, or a 'new' placeholder."""
    entry = latest_for_role(ledger, role_id)
    if entry is None:
        return "(eres nuevo en este turno — no has intervenido aún)"
    text = entry.text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return f"(de la ronda {entry.round_number}, turno {entry.turn})\n{text}"


def participating_roles(ledger: DiscussionLedger) -> set[RoleId]:
    """Which roles have intervened at least once."""
    return {e.role_id for e in ledger.entries}


def filter_by_roles(
    ledger: DiscussionLedger, roles: Iterable[RoleId]
) -> list[LedgerEntry]:
    """All entries authored by any of the given roles."""
    role_set = set(roles)
    return [e for e in ledger.entries if e.role_id in role_set]
