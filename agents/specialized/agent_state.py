import asyncio

from common.models import AgentRoleConfig, SkillDefinition


class AgentState:
    """Mutable state for a Specialized Agent.

    Updated by the orchestrator via the internal config API before each
    debate flow. The A2A agent card and executor read from this state.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._lock = asyncio.Lock()
        self.role: str = "Unassigned"
        self.perspective: str = ""
        self.skills: list[SkillDefinition] = []
        self.system_prompt: str = "You are a debate agent. Await configuration."
        self.ready: bool = False

    async def configure(self, config: AgentRoleConfig) -> None:
        async with self._lock:
            self.role = config.role
            self.perspective = config.perspective
            self.skills = config.skills
            self.system_prompt = (
                f"You are a {config.role} participating in a structured debate. "
                f"Your perspective: {config.perspective}. "
                f"Make clear, well-reasoned arguments. Be concise (3-5 paragraphs max). "
                f"Engage directly with the opposing argument when responding."
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
