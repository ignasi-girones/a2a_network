"""Unit tests for AgentState — the mutable state held by Specialized agents."""

from __future__ import annotations

import pytest

from agents.specialized.agent_state import AgentState
from common.models import AgentRoleConfig, SkillDefinition


pytestmark = pytest.mark.asyncio


class TestAgentState:
    async def test_initial_state(self):
        s = AgentState(agent_id="ae1")
        assert s.agent_id == "ae1"
        assert await s.is_ready() is False
        assert await s.get_role() == "Unassigned"
        # Default system prompt is non-empty
        assert len(await s.get_system_prompt()) > 0
        assert await s.get_skills() == []

    async def test_configure_sets_fields(self):
        s = AgentState(agent_id="ae1")
        config = AgentRoleConfig(
            role="Devil's Advocate",
            perspective="skeptical",
            skills=[SkillDefinition(id="critique", name="Critique")],
        )
        await s.configure(config)

        assert await s.is_ready() is True
        assert await s.get_role() == "Devil's Advocate"
        skills = await s.get_skills()
        assert len(skills) == 1
        assert skills[0].id == "critique"

        prompt = await s.get_system_prompt()
        assert "Devil's Advocate" in prompt
        assert "skeptical" in prompt

    async def test_reconfigure_overwrites(self):
        """A second configure() replaces all fields, doesn't merge."""
        s = AgentState(agent_id="ae1")
        await s.configure(
            AgentRoleConfig(role="R1", perspective="P1", skills=[])
        )
        await s.configure(
            AgentRoleConfig(role="R2", perspective="P2", skills=[])
        )
        assert await s.get_role() == "R2"
        assert "P2" in await s.get_system_prompt()
        assert "P1" not in await s.get_system_prompt()

    async def test_get_skills_returns_copy(self):
        """Mutating the returned skills list must not affect internal state."""
        s = AgentState(agent_id="ae1")
        await s.configure(
            AgentRoleConfig(
                role="R",
                perspective="P",
                skills=[SkillDefinition(id="a", name="A")],
            )
        )
        out = await s.get_skills()
        out.append(SkillDefinition(id="hacked", name="X"))
        # Internal skills unchanged
        again = await s.get_skills()
        assert len(again) == 1
        assert again[0].id == "a"
