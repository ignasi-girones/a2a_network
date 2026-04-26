import asyncio

from common.models import AgentRoleConfig, SkillDefinition


class AgentState:
    """Mutable state for a Specialized Agent.

    Updated by the orchestrator via the internal config API before each
    debate flow. The A2A agent card and executor read from this state.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a debate participant in a structured deliberation. Each task "
        "you receive will assign you a ROLE and PERSPECTIVE in its description, "
        "and tell you the round number and goal (argue / converge / synthesize).\n\n"
        "Your goal is NOT to win — it is to reach the best joint answer "
        "through dialogue. Convergence on a shared verdict across rounds is "
        "the success criterion, not defending the initial position.\n\n"
        "Format every response in TWO labeled sections:\n"
        "AGREEMENTS: bullet-list the points from the opponent's argument you "
        "accept as valid. Use 'None yet' only when truly nothing can be conceded.\n"
        "REFINEMENT: state where you still disagree or want to nuance, with "
        "reasoning. Keep this shorter than AGREEMENTS as rounds progress.\n\n"
        "Concede explicitly when the opponent presents stronger evidence — "
        "say 'You changed my mind on X'. Be concise (3-5 paragraphs total). "
        "Follow the per-round goal stated under [Goal for this round]."
    )

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._lock = asyncio.Lock()
        self.role: str = "Unassigned"
        self.perspective: str = ""
        self.skills: list[SkillDefinition] = []
        self.system_prompt: str = self.DEFAULT_SYSTEM_PROMPT
        self.ready: bool = False

    async def configure(self, config: AgentRoleConfig) -> None:
        async with self._lock:
            self.role = config.role
            self.perspective = config.perspective
            self.skills = config.skills
            self.system_prompt = (
                f"You are a {config.role} participating in a structured deliberation. "
                f"Your perspective: {config.perspective}.\n\n"
                "Your goal is NOT to win — it is to reach the best joint answer "
                "through dialogue. Convergence on a shared verdict is the success "
                "criterion across rounds, not defending your initial position.\n\n"
                "Format every response in TWO labeled sections:\n"
                "AGREEMENTS: bullet-list the points from the opponent's argument "
                "you accept as valid. Use 'None yet' only if you genuinely cannot "
                "concede anything.\n"
                "REFINEMENT: state where you still disagree or want to nuance, "
                "with reasoning. Keep this shorter than AGREEMENTS as rounds progress.\n\n"
                "Concede explicitly when the opponent presents stronger evidence — "
                "say 'You changed my mind on X'. Be concise (3-5 paragraphs total). "
                "Follow any per-round instruction the orchestrator includes under "
                "[Goal for this round]."
            )
            self.ready = True

    async def get_system_prompt(self) -> str:
        async with self._lock:
            return self.system_prompt

    async def get_role(self) -> str:
        async with self._lock:
            return self.role

    async def get_skills(self) -> list[SkillDefinition]:
        async with self._lock:
            return list(self.skills)

    async def is_ready(self) -> bool:
        async with self._lock:
            return self.ready
