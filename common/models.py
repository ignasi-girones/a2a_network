from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


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


class SubTask(BaseModel):
    """A single unit of work in a TaskPlan.

    The Planner emits these as nodes of a DAG: each SubTask declares what it
    needs (`required_skill`) and what must finish first (`depends_on`). The
    PlanExecutor matches `required_skill` against worker AgentCard skills in
    the AgentRegistry at execution time.
    """
    id: str = Field(description="Short unique id inside the plan, e.g. 't1'")
    description: str = Field(
        description="Concrete instruction this sub-agent should carry out"
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
        description="For debate-style workers: 'pro' / 'con' / 'neutral' / free text",
    )


class TaskPlan(BaseModel):
    """The LLM's decomposition of a user request into a DAG of SubTasks."""
    goal: str = Field(description="Restated user intent in one sentence")
    subtasks: list[SubTask] = Field(default_factory=list)
    max_workers: int = Field(
        default=4,
        description="Upper bound of workers this plan may need concurrently",
    )
