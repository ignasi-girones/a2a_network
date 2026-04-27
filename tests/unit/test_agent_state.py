"""Unit tests for AgentState — Phase 3 PersonaContract-based state.

Phase 3 replaced the legacy {role, perspective, system_prompt} bundle with
a `PersonaContract` set per-run by the orchestrator. Tests below match the
new API.
"""

from __future__ import annotations

import pytest

from agents.orchestrator.persona_catalog import build_persona
from agents.specialized.agent_state import AgentState
from common.models import AgentRoleConfig, SkillDefinition


pytestmark = pytest.mark.asyncio


def _config_for(role: str, *, claim: str | None = None, skills=None) -> AgentRoleConfig:
    return AgentRoleConfig(
        persona=build_persona(role),  # type: ignore[arg-type]
        skills=skills or [],
        claim=claim,
    )


class TestAgentState:
    async def test_initial_state_unconfigured(self):
        s = AgentState(agent_id="analyst")
        assert s.agent_id == "analyst"
        assert await s.is_ready() is False
        assert await s.get_role_id() == "unassigned"
        assert await s.get_display_name() == "Unassigned"
        assert await s.get_skills() == []
        assert await s.get_claim() is None

    async def test_get_persona_raises_before_configure(self):
        s = AgentState(agent_id="analyst")
        with pytest.raises(RuntimeError, match="no persona"):
            await s.get_persona()

    async def test_configure_sets_persona(self):
        s = AgentState(agent_id="analyst")
        config = _config_for(
            "analyst",
            claim="Microservicios > monolito",
            skills=[SkillDefinition(id="role_analyst", name="Analista")],
        )
        await s.configure(config)

        assert await s.is_ready() is True
        assert await s.get_role_id() == "analyst"
        assert await s.get_display_name() == "Analista"
        assert await s.get_claim() == "Microservicios > monolito"

        persona = await s.get_persona()
        assert persona.role_id == "analyst"
        # Phase C heterogeneity: analyst pulls from Wikipedia.
        assert "wikipedia" in persona.tool_whitelist
        assert persona.eristic_stratagem_id is None

        skills = await s.get_skills()
        assert len(skills) == 1
        assert skills[0].id == "role_analyst"

    async def test_devils_advocate_carries_stratagem(self):
        s = AgentState(agent_id="da")
        await s.configure(_config_for("devils_advocate"))
        persona = await s.get_persona()
        assert persona.role_id == "devils_advocate"
        assert persona.eristic_stratagem_id is not None
        assert "Abogado del Diablo" in persona.display_name

    async def test_reconfigure_overwrites(self):
        """A second configure() replaces persona+claim, doesn't merge."""
        s = AgentState(agent_id="ae1")
        await s.configure(_config_for("analyst", claim="C1"))
        await s.configure(_config_for("seeker", claim="C2"))
        assert await s.get_role_id() == "seeker"
        assert await s.get_display_name() == "Buscador"
        assert await s.get_claim() == "C2"

    async def test_get_skills_returns_copy(self):
        """Mutating the returned skills list must not affect internal state."""
        s = AgentState(agent_id="analyst")
        await s.configure(
            _config_for(
                "analyst",
                skills=[SkillDefinition(id="a", name="A")],
            )
        )
        out = await s.get_skills()
        out.append(SkillDefinition(id="hacked", name="X"))
        again = await s.get_skills()
        assert len(again) == 1
        assert again[0].id == "a"
