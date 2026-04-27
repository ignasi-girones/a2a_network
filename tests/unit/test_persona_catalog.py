"""Unit tests for ``agents.orchestrator.persona_catalog``.

The catalog is the single source of truth for the four canonical persona
templates (Phase 3). These tests pin down:

  - every canonical role builds a valid PersonaContract,
  - tool whitelists match the dialectic design (Synthesizer has none, etc.),
  - render_system_prompt fills placeholders and gracefully drops unused ones,
  - the Devil's Advocate's stratagem is injected verbatim into the prompt.
"""

from __future__ import annotations

import pytest

from agents.orchestrator.persona_catalog import build_persona, render_system_prompt
from agents.specialized.eristic import STRATAGEMS, get_stratagem
from common.models import CANONICAL_ROLES, PersonaContract


class TestBuildPersona:
    def test_each_role_builds(self):
        for role in CANONICAL_ROLES:
            p = build_persona(role)
            assert isinstance(p, PersonaContract)
            assert p.role_id == role
            assert p.display_name
            assert p.system_prompt_template

    def test_synthesizer_has_no_tools(self):
        p = build_persona("synthesizer")
        assert p.tool_whitelist == []
        assert p.eristic_stratagem_id is None

    def test_analyst_uses_wikipedia_plus_calculator(self):
        """Phase C: Analyst pulls baseline facts from Wikipedia."""
        p = build_persona("analyst")
        assert "wikipedia" in p.tool_whitelist
        assert "calculator" in p.tool_whitelist
        assert "web_search" not in p.tool_whitelist
        assert p.eristic_stratagem_id is None

    def test_seeker_has_web_search(self):
        p = build_persona("seeker")
        assert "web_search" in p.tool_whitelist

    def test_empiricist_uses_arxiv_and_calculator(self):
        p = build_persona("empiricist")
        assert "arxiv" in p.tool_whitelist
        assert "calculator" in p.tool_whitelist
        assert "web_search" not in p.tool_whitelist
        assert p.eristic_stratagem_id is None

    def test_pragmatist_uses_web_search_and_wikipedia(self):
        p = build_persona("pragmatist")
        assert "web_search" in p.tool_whitelist
        assert "wikipedia" in p.tool_whitelist
        assert "arxiv" not in p.tool_whitelist
        assert p.eristic_stratagem_id is None

    def test_devils_advocate_carries_stratagem(self):
        p = build_persona("devils_advocate")
        assert p.eristic_stratagem_id is not None
        assert p.eristic_stratagem_id in {s.id for s in STRATAGEMS}

    def test_devils_advocate_high_temperature(self):
        """High temp is part of the academic argument: combative variety."""
        p = build_persona("devils_advocate")
        assert p.temperature >= 0.8

    def test_explicit_stratagem_override(self):
        s = get_stratagem(15)  # Sham syllogism
        p = build_persona("devils_advocate", stratagem=s)
        assert p.eristic_stratagem_id == 15
        assert "Sham syllogism" in p.display_name

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError, match="Unknown role_id"):
            build_persona("hallucinated")  # type: ignore[arg-type]


class TestRenderSystemPrompt:
    def test_topic_substituted(self):
        p = build_persona("analyst")
        out = render_system_prompt(
            p, topic="microservicios", claim="X", peers_outputs=""
        )
        assert "microservicios" in out

    def test_devils_advocate_stratagem_injected(self):
        p = build_persona("devils_advocate", stratagem=get_stratagem(1))  # Extension
        out = render_system_prompt(
            p, topic="t", claim="c", peers_outputs="prev"
        )
        # Stratagem id and name appear in the rendered prompt.
        assert "Extension" in out or "extension" in out
        assert "1" in out
        # The directive itself appears verbatim somewhere.
        directive = get_stratagem(1).prompt_directive
        # Use the first 30 chars as a stable substring (the template wraps it).
        assert directive[:30] in out

    def test_missing_placeholders_dont_crash(self):
        """Templates use {topic}/{claim}/{peers_outputs}; missing keys default
        to empty strings rather than raising KeyError."""
        p = build_persona("seeker")
        out = render_system_prompt(p, topic="x")
        assert isinstance(out, str)
        assert len(out) > 0

    def test_synthesizer_fills_peers_outputs(self):
        p = build_persona("synthesizer")
        out = render_system_prompt(
            p, topic="t", claim="c", peers_outputs="ANALYST: ...\nSEEKER: ..."
        )
        assert "ANALYST" in out
        assert "SEEKER" in out
