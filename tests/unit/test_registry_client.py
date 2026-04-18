"""Unit tests for common.registry_client — client-side helpers."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from common import registry_client
from common.registry_client import (
    agent_card_to_dict,
    deregister_self,
    register_self_with_orchestrator,
)


pytestmark = pytest.mark.asyncio


# ── agent_card_to_dict ──────────────────────────────────────────────────────


class TestAgentCardToDict:
    def test_basic(self):
        skills = [
            SimpleNamespace(
                id="debate", name="Debate", description="d", tags=["a", "b"]
            )
        ]
        out = agent_card_to_dict(
            name="N", description="D", skills=skills, streaming=True
        )
        assert out["name"] == "N"
        assert out["description"] == "D"
        assert out["streaming"] is True
        assert out["skills"] == [
            {"id": "debate", "name": "Debate", "description": "d", "tags": ["a", "b"]}
        ]

    def test_skill_missing_description_defaulted(self):
        """`description` attribute may be absent; it defaults to empty string."""
        s = SimpleNamespace(id="x", name="X", tags=[])
        out = agent_card_to_dict(name="n", description="d", skills=[s])
        # Missing description becomes "" (empty string is kept — the filter
        # only drops None values). id/name/tags remain untouched.
        assert out["skills"][0]["description"] == ""
        assert out["skills"][0]["id"] == "x"
        assert out["skills"][0]["name"] == "X"
        assert out["skills"][0]["tags"] == []

    def test_empty_skills_list(self):
        out = agent_card_to_dict(name="n", description="d", skills=[])
        assert out["skills"] == []


# ── register_self_with_orchestrator ─────────────────────────────────────────


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class FakeAsyncClient:
    """Context-manager-compatible fake httpx.AsyncClient.

    Given a sequence of responses (or exception instances), returns/raises
    each in turn for successive post() calls.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.post_calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json):
        self.post_calls.append((url, json))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def delete(self, url):
        self.post_calls.append((url, None))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestRegisterSelf:
    async def test_succeeds_on_first_attempt(self, monkeypatch):
        fake = FakeAsyncClient([FakeResponse(200)])
        monkeypatch.setattr(
            registry_client.httpx, "AsyncClient", lambda **kw: fake
        )
        ok = await register_self_with_orchestrator(
            agent_id="ae1",
            url="http://localhost:9002",
            card={"skills": []},
            max_retries=1,
        )
        assert ok is True
        assert len(fake.post_calls) == 1

    async def test_retries_on_connect_error_then_succeeds(self, monkeypatch):
        fake = FakeAsyncClient(
            [httpx.ConnectError("refused"), FakeResponse(200)]
        )
        monkeypatch.setattr(
            registry_client.httpx, "AsyncClient", lambda **kw: fake
        )
        ok = await register_self_with_orchestrator(
            agent_id="ae1",
            url="http://x",
            card={},
            max_retries=2,
            retry_delay=0.01,
        )
        assert ok is True
        assert len(fake.post_calls) == 2

    async def test_gives_up_after_exhausting_retries(self, monkeypatch):
        fake = FakeAsyncClient(
            [httpx.ConnectError("down")] * 3
        )
        monkeypatch.setattr(
            registry_client.httpx, "AsyncClient", lambda **kw: fake
        )
        ok = await register_self_with_orchestrator(
            agent_id="ae1",
            url="http://x",
            card={},
            max_retries=3,
            retry_delay=0.01,
        )
        assert ok is False

    async def test_non_200_status_is_logged_and_retried(self, monkeypatch):
        fake = FakeAsyncClient(
            [FakeResponse(500, text="err"), FakeResponse(200)]
        )
        monkeypatch.setattr(
            registry_client.httpx, "AsyncClient", lambda **kw: fake
        )
        ok = await register_self_with_orchestrator(
            agent_id="ae1",
            url="http://x",
            card={},
            max_retries=2,
            retry_delay=0.01,
        )
        assert ok is True


# ── deregister_self ─────────────────────────────────────────────────────────


class TestDeregisterSelf:
    async def test_ignores_errors(self, monkeypatch):
        """deregister_self is best-effort; connection errors must be swallowed."""
        fake = FakeAsyncClient([httpx.ConnectError("no orch")])
        monkeypatch.setattr(
            registry_client.httpx, "AsyncClient", lambda **kw: fake
        )
        # Should NOT raise even though the orchestrator is unreachable.
        await deregister_self("ae1")

    async def test_happy_path(self, monkeypatch):
        fake = FakeAsyncClient([FakeResponse(200)])
        monkeypatch.setattr(
            registry_client.httpx, "AsyncClient", lambda **kw: fake
        )
        await deregister_self("ae1")
        assert len(fake.post_calls) == 1
        assert "ae1" in fake.post_calls[0][0]
