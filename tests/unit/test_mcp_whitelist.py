"""Phase 3 / Pillar 3 — MCP tool whitelist enforcement.

Each persona advertises a set of MCP tools it is allowed to call. The
SpecializedExecutor's ``_call_mcp_tool`` enforces the whitelist before
dialing the MCP server. These tests pin both:

  - the whitelist entries on each canonical persona (heterogeneity guard),
  - the runtime enforcement (a worker that *configurationally* should not
    call a tool gets a hard error instead of a silent leak).
"""

from __future__ import annotations

import pytest

from agents.orchestrator.persona_catalog import build_persona
from agents.specialized.executor import ToolNotAllowedError, _call_mcp_tool


# ── Whitelist heterogeneity ────────────────────────────────────────────────


class TestPersonaWhitelists:
    def test_analyst_uses_wikipedia(self):
        p = build_persona("analyst")
        assert "wikipedia" in p.tool_whitelist
        # Analyst doesn't get free-text web search — its baseline must come
        # from encyclopedic sources to model the RAPID-D analyst role.
        assert "web_search" not in p.tool_whitelist

    def test_seeker_uses_web_search_and_arxiv(self):
        p = build_persona("seeker")
        assert "web_search" in p.tool_whitelist
        assert "arxiv" in p.tool_whitelist

    def test_devils_advocate_no_arxiv_but_web_search(self):
        p = build_persona("devils_advocate")
        # Devil's Advocate is a generalist attacker, not an academic researcher.
        assert "web_search" in p.tool_whitelist
        assert "arxiv" not in p.tool_whitelist

    def test_synthesizer_has_no_tools(self):
        p = build_persona("synthesizer")
        assert p.tool_whitelist == []

    def test_no_two_roles_share_full_whitelist(self):
        """Heterogeneity hard-rule: at least one tool differs between any
        two roles. Otherwise the panel collapses back to monoculture."""
        # Only debate-side agents (synthesizer has empty list, excluded to avoid
        # it matching any role that might also get empty list by accident).
        debate_roles = ("analyst", "seeker", "devils_advocate", "empiricist", "pragmatist")
        ws = {
            r: tuple(sorted(build_persona(r).tool_whitelist))
            for r in debate_roles
        }
        assert len(set(ws.values())) == len(ws)


# ── Runtime enforcement ────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestCallMcpToolEnforcement:
    async def test_unauthorised_tool_raises(self):
        """An agent whose persona does not whitelist arxiv must not be able
        to call it — even by name. The exception type is part of the
        contract; tests downstream can match on it."""
        with pytest.raises(ToolNotAllowedError):
            await _call_mcp_tool(
                "arxiv",
                {"query": "x"},
                whitelist=["wikipedia", "calculator"],
            )

    async def test_authorised_tool_does_not_raise_on_whitelist_check(
        self, monkeypatch
    ):
        """When the tool IS whitelisted, the function must NOT raise the
        whitelist exception — it may still return None on network error,
        but that's a different code path.

        We monkey-patch the streamablehttp_client so no real MCP traffic is
        attempted; the test asserts that we get past the whitelist check
        without raising ToolNotAllowedError.
        """
        from agents.specialized import executor as ex_module

        class _DummyCtx:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                # Force the inner block to bail with a network error so we
                # exercise the resilience path without hitting MCP.
                raise OSError("simulated network error")

            async def __aexit__(self, *_):
                return False

        monkeypatch.setattr(
            ex_module, "streamablehttp_client", lambda *a, **kw: _DummyCtx()
        )

        result = await _call_mcp_tool(
            "wikipedia",
            {"title": "Bayesian inference"},
            whitelist=["wikipedia", "calculator"],
        )
        # Network error → returns None (resilient path), not raises.
        assert result is None
