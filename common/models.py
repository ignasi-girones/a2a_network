from pydantic import BaseModel, Field


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


class AgentRoleConfig(BaseModel):
    """Configuration sent to a Specialized Agent to set its role."""
    role: str
    perspective: str
    skills: list[SkillDefinition] = Field(default_factory=list)


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
