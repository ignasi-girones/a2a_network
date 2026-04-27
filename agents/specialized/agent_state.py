"""Mutable state for a Specialized Agent.

Phase 3: the legacy {role, perspective, system_prompt} bundle is replaced by
a structured `PersonaContract` set by the orchestrator before each run.
The agent card and executor read from this state via async accessors.

Phase 3 / Pillar 2 adds an externalised ``BeliefState`` per agent, anchored
to a central claim. The state is reseeded at every ``configure()`` call so
log-odds trajectories reflect a single deliberation, not history across runs.

Backward compatibility is intentionally NOT preserved — the orchestrator
must send a persona, and the worker refuses to operate without one.
"""

import asyncio

from common.models import (
    AgentRoleConfig,
    BeliefState,
    PersonaContract,
    RoleId,
    SkillDefinition,
)


# ── Per-role default belief priors ──────────────────────────────────────────
# Bias parameters are shaped to the dialectic role:
#   - Analista: pure Bayesian (no priors, no anchoring, no asymmetry).
#   - Buscador: slightly volatile (sensitivity 1.2) — tends to update
#     more on external evidence.
#   - Abogado del Diablo: starts at log_odds=-1 (mild scepticism), high
#     asymmetry (1.6) — confirmation bias toward its assigned negative
#     stance, modeling the eristic role.
#   - Sintetizador: pure Bayesian, but with mild anchoring (0.2) so
#     the final verdict isn't whipsawed by the most recent input.
_ROLE_BELIEF_DEFAULTS: dict[RoleId, dict[str, float]] = {
    "analyst": {
        # Pure Bayesian — no priors, no anchoring, no asymmetry.
        "log_odds": 0.0,
        "evidence_sensitivity": 1.0,
        "anchoring": 0.0,
        "asymmetry": 1.0,
    },
    "seeker": {
        # Standard Bayesian — updates cleanly on external evidence.
        # evidence_sensitivity kept at 1.0 (not 1.2) to avoid immediately
        # maxing out the chart on the first search result.
        "log_odds": 0.0,
        "evidence_sensitivity": 1.0,
        "anchoring": 0.0,
        "asymmetry": 1.0,
    },
    "devils_advocate": {
        # Starts mildly sceptical (−0.5 instead of −1.0) and has moderate
        # confirmation bias (asymmetry 1.2 instead of 1.6) so the trajectory
        # shows movement rather than pinning immediately at −5.
        "log_odds": -0.5,
        "evidence_sensitivity": 1.0,
        "anchoring": 0.1,
        "asymmetry": 1.2,
    },
    "empiricist": {
        # Popperian falsificationist: starts neutral, updates slowly
        # (evidence_sensitivity 0.8 = stubborn), slight anti-confirmation
        # bias (asymmetry 0.9 < 1 dampens confirmatory updates).
        "log_odds": 0.0,
        "evidence_sensitivity": 0.85,
        "anchoring": 0.15,
        "asymmetry": 0.9,
    },
    "pragmatist": {
        # Case-study pragmatist: slight positive prior (real-world solutions
        # usually exist), normal sensitivity, mild anchoring on first cases.
        "log_odds": 0.2,
        "evidence_sensitivity": 1.0,
        "anchoring": 0.1,
        "asymmetry": 1.0,
    },
    "synthesizer": {
        # Mild anchoring so the verdict isn't whipsawed by the last input.
        "log_odds": 0.0,
        "evidence_sensitivity": 1.0,
        "anchoring": 0.2,
        "asymmetry": 1.0,
    },
}


def _seed_belief(*, claim: str, role_id: RoleId) -> BeliefState:
    """Construct a fresh BeliefState with role-shaped bias parameters."""
    params = _ROLE_BELIEF_DEFAULTS.get(role_id, {})
    return BeliefState(
        claim=claim,
        log_odds=params.get("log_odds", 0.0),
        evidence_sensitivity=params.get("evidence_sensitivity", 1.0),
        anchoring=params.get("anchoring", 0.0),
        asymmetry=params.get("asymmetry", 1.0),
        history=[],
    )


class AgentState:
    """Mutable per-worker state, updated via /internal/configure."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._lock = asyncio.Lock()
        self.persona: PersonaContract | None = None
        self.skills: list[SkillDefinition] = []
        self.claim: str | None = None
        self.belief: BeliefState | None = None
        self.ready: bool = False

    async def configure(self, config: AgentRoleConfig) -> None:
        """Set the persona for the next run.

        The system prompt is NOT pre-rendered here — it is rendered per-call
        in the executor (with topic and peers_outputs from the dispatched
        prompt) using `persona_catalog.render_system_prompt`.

        A fresh BeliefState is also seeded from the configured claim, with
        role-shaped bias parameters (see ``_ROLE_BELIEF_DEFAULTS``).
        """
        async with self._lock:
            self.persona = config.persona
            self.skills = config.skills
            self.claim = config.claim
            self.ready = True
            if config.claim:
                self.belief = _seed_belief(
                    claim=config.claim, role_id=config.persona.role_id
                )
            else:
                self.belief = None

    async def get_persona(self) -> PersonaContract:
        async with self._lock:
            if self.persona is None:
                raise RuntimeError(
                    f"Agent {self.agent_id} has no persona — orchestrator "
                    f"must POST /internal/configure before dispatching work"
                )
            return self.persona

    async def get_role_id(self) -> str:
        async with self._lock:
            return self.persona.role_id if self.persona else "unassigned"

    async def get_display_name(self) -> str:
        async with self._lock:
            return (
                self.persona.display_name if self.persona else "Unassigned"
            )

    async def get_skills(self) -> list[SkillDefinition]:
        async with self._lock:
            return list(self.skills)

    async def get_claim(self) -> str | None:
        async with self._lock:
            return self.claim

    async def get_belief(self) -> BeliefState | None:
        """Return the live BeliefState (mutated in place by belief_updater).

        Returns ``None`` when no claim has been configured (e.g. legacy
        run path that pre-dates Pillar 2).
        """
        async with self._lock:
            return self.belief

    async def snapshot_belief(self) -> BeliefState | None:
        """Return a deep copy of the BeliefState for safe external use."""
        async with self._lock:
            if self.belief is None:
                return None
            return self.belief.model_copy(deep=True)

    async def is_ready(self) -> bool:
        async with self._lock:
            return self.ready
