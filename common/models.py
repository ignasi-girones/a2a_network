from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Phase 3: dialectic role catalog ─────────────────────────────────────────
# These four roles map to the RAPID-D pipeline (Coinbase) extended with
# Schopenhauer's eristic dialectic and Habermas's validity claims. The Planner
# emits exactly this quartet for every prompt; AE1/AE2 as symmetric pro/con
# debaters are retired in this phase.
RoleId = Literal[
    "analyst",
    "seeker",
    "devils_advocate",
    "empiricist",
    "pragmatist",
    "synthesizer",
]
CANONICAL_ROLES: tuple[RoleId, ...] = (
    "analyst",
    "seeker",
    "devils_advocate",
    "empiricist",
    "pragmatist",
    "synthesizer",
)


class WorkerEntry(BaseModel):
    """A worker agent registered with the orchestrator's AgentRegistry.

    The `card` field holds a serialized subset of the worker's AgentCard
    (name, description, skills, capabilities) that the orchestrator's Planner
    can consult to match sub-tasks to workers by skill id or tag.
    """
    agent_id: str
    url: str
    card: dict[str, Any] = Field(default_factory=dict)
    registered_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class NormalizedPrompt(BaseModel):
    """Output of the Normalizer agent."""
    topic: str
    domain: str
    question_type: str = Field(description="e.g. 'opinion', 'decision', 'analysis'")
    constraints: list[str] = Field(default_factory=list)
    suggested_perspectives: list[str] = Field(default_factory=list)


class SkillDefinition(BaseModel):
    """A dynamically generated skill description."""
    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class BeliefSample(BaseModel):
    """A single point on an agent's BeliefState trajectory.

    Stored after every belief update. The frontend's
    ``BeliefTrajectoryChart`` graphs the per-agent series of these.
    """
    log_odds: float = Field(
        description="Posterior log-odds of the central claim, after the update."
    )
    delta: float = Field(
        description="Magnitude of the most recent update (signed).",
    )
    rationale: str = Field(
        default="",
        description="Short LLM-supplied justification for the update direction.",
    )
    stage: str = Field(
        default="post_response",
        description="Lifecycle marker: prior | post_initial | post_refine | post_aporia",
    )


class BeliefState(BaseModel):
    """Phase 3 / Pillar 2 — externalised, auditable epistemic posture.

    The agent's stance no longer lives in the LLM's free-text output; it
    lives in a scalar log-odds value over a single central claim, updated
    by a lateral LLM call after every intervention. The trajectory is
    what the frontend graphs as the "epistemic dynamics" of the debate.

    Bias parameters (Pearl/Kahneman):
      evidence_sensitivity: how strongly the agent updates on new evidence
        (1.0 = pure Bayesian; <1 = stubborn; >1 = volatile).
      anchoring: shrinks updates toward 0 to model anchoring on prior
        beliefs (0 = no anchoring; 1 = ignore evidence completely).
      asymmetry: confirmation bias — when >1 the agent amplifies evidence
        that confirms its current direction and dampens disconfirming
        evidence; when 1 updates are symmetric.
    """
    claim: str
    log_odds: float = 0.0  # 0 = 50/50 prior
    evidence_sensitivity: float = 1.0
    anchoring: float = 0.0
    asymmetry: float = 1.0
    history: list[BeliefSample] = Field(default_factory=list)


class PersonaContract(BaseModel):
    """The dialectic identity assigned to a worker for a single run.

    This is the unit of identity in Phase 3. A worker is no longer "AE1 pro"
    or "AE2 con" — it is one of four canonical dialectic roles, each with a
    distinct system-prompt template, tool whitelist, and (for the Devil's
    Advocate) an eristic stratagem from Schopenhauer.

    Fields:
      role_id: Which of the four roles this persona plays.
      display_name: Human-friendly label rendered in the frontend graph.
      system_prompt_template: Format string supporting {topic}, {peers_outputs},
        {stratagem}, {claim}. Unused placeholders are dropped silently.
      tool_whitelist: MCP tool ids this persona may call. Enforced in the
        worker executor — calling anything outside this list raises.
      eristic_stratagem_id: 1-38, only meaningful for `devils_advocate`. None
        elsewhere.
      temperature: LLM sampling temperature for this run.
    """
    role_id: RoleId
    display_name: str
    system_prompt_template: str
    tool_whitelist: list[str] = Field(default_factory=list)
    eristic_stratagem_id: int | None = None
    temperature: float = 0.7


class AgentRoleConfig(BaseModel):
    """Configuration sent to a Specialized Agent to set its role.

    Phase 3: the legacy `role` (free string) + `perspective` (pro/con) pair is
    replaced by a structured `PersonaContract`. The `skills` list is kept so
    the worker can advertise rol-specific skill ids in its Agent Card.
    """
    persona: PersonaContract
    skills: list[SkillDefinition] = Field(default_factory=list)
    # The user's normalized claim — anchors belief state and prompt context.
    # Optional during transition; required when Phase B is active.
    claim: str | None = None


class RoleDecision(BaseModel):
    """Orchestrator's decision on how to configure the debate."""
    ae1_config: AgentRoleConfig
    ae2_config: AgentRoleConfig
    max_rounds: int = Field(ge=2, le=5)


class DebateRound(BaseModel):
    """A single round of the debate."""
    round_number: int
    ae1_argument: str
    ae2_argument: str


class DebateState(BaseModel):
    """Full state of an ongoing debate."""
    topic: str
    ae1_role: str
    ae2_role: str
    initial_opinions: dict[str, str] = Field(default_factory=dict)
    rounds: list[DebateRound] = Field(default_factory=list)
    consensus_reached: bool = False
    max_rounds: int = 5


class FlowResult(BaseModel):
    """Final output passed to the Feedback agent."""
    topic: str
    ae1_role: str
    ae2_role: str
    ae1_initial_opinion: str
    ae2_initial_opinion: str
    debate_rounds: list[DebateRound]
    consensus_reached: bool
    summary: str


class SubTask(BaseModel):
    """A single unit of work in a TaskPlan.

    The Planner emits these as nodes of a DAG. Phase 3 adds `role_id` as the
    primary routing key (analyst / seeker / devils_advocate / synthesizer);
    `required_skill` is kept for backward compatibility with the registry
    matcher and is set by the orchestrator to `f"role_{role_id}"`.
    """
    id: str = Field(description="Short unique id inside the plan, e.g. 't1'")
    description: str = Field(
        description="Concrete instruction this sub-agent should carry out"
    )
    role_id: RoleId | None = Field(
        default=None,
        description="Phase 3 dialectic role; the orchestrator derives "
        "required_skill from this when present.",
    )
    required_skill: str = Field(
        description="Skill id (AgentSkill.id) to look up in the registry"
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of subtasks whose outputs are needed before this runs",
    )
    perspective: str | None = Field(
        default=None,
        description="Free-text perspective tag (legacy; informational only)",
    )


# ── Phase 4: Multi-round deliberation with shared blackboard ──────────────
# A LedgerEntry is one intervention by one agent, in one round. The ledger
# is the shared discussion context all agents read on every round; it
# replaces the dep-output map of Phase 3 as the primary inter-agent channel.


class LedgerEntry(BaseModel):
    """A single intervention in the discussion ledger.

    Phase 4 turns the panel from a one-shot DAG into a turn-based
    blackboard: every agent sees every entry of every prior round and may
    refer to other turns by index.
    """
    turn: int = Field(description="Monotonic turn id across the whole run, starting at 0.")
    round_number: int = Field(description="1-indexed round; round 1 is the opening statements.")
    role_id: RoleId
    agent_id: str
    text: str
    belief_after: float | None = Field(
        default=None,
        description="Author's BeliefState.log_odds after writing this entry (None for synthesizer).",
    )
    delta: float | None = Field(
        default=None,
        description="Belief movement caused by this turn (signed).",
    )
    references: list[int] = Field(
        default_factory=list,
        description="Turn-ids the author explicitly responds to (rough heuristic).",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class DiscussionLedger(BaseModel):
    """The shared blackboard for a single deliberation run.

    Lives in orchestrator memory during execution. Serialised in
    ``ledger_entry`` SSE events so the frontend can build a chat-style
    threaded view in real time.
    """
    claim: str
    goal: str
    entries: list[LedgerEntry] = Field(default_factory=list)
    max_rounds: int = 3
    current_round: int = 0
    terminated_reason: str | None = Field(
        default=None,
        description="'consensus' | 'aporia_resolved' | 'max_rounds' | 'cap_reached'",
    )


class TaskPlan(BaseModel):
    """The LLM's decomposition of a user request into a DAG of SubTasks."""
    goal: str = Field(description="Restated user intent in one sentence")
    claim: str = Field(
        default="",
        description="Phase 3: the central proposition the dialectic panel "
        "argues about. Anchors the BeliefState log_odds across roles; the "
        "orchestrator forwards it to every worker via PersonaContract.",
    )
    subtasks: list[SubTask] = Field(default_factory=list)
    max_workers: int = Field(
        default=4,
        description="Upper bound of workers this plan may need concurrently",
    )
