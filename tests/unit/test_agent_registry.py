"""Unit tests for the AgentRegistry and its HTTP routes."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from agents.orchestrator.agent_registry import AgentRegistry
from common.models import WorkerEntry


pytestmark = pytest.mark.asyncio


# ── Core registry logic ──────────────────────────────────────────────────────


class TestAgentRegistry:
    async def test_empty_registry(self, fresh_registry: AgentRegistry):
        assert await fresh_registry.all_workers() == []
        assert await fresh_registry.find_by_skill("debate") == []
        assert await fresh_registry.find_by_agent_id("ae1") is None

    async def test_register_and_find(self, fresh_registry: AgentRegistry):
        entry = WorkerEntry(
            agent_id="ae1",
            url="http://localhost:9002",
            card={"skills": [{"id": "debate", "name": "Debate", "tags": []}]},
        )
        await fresh_registry.register(entry)

        workers = await fresh_registry.all_workers()
        assert len(workers) == 1
        assert workers[0].agent_id == "ae1"

        assert await fresh_registry.find_by_agent_id("ae1") == entry
        matches = await fresh_registry.find_by_skill("debate")
        assert len(matches) == 1
        assert matches[0].agent_id == "ae1"

    async def test_register_same_id_replaces(self, fresh_registry: AgentRegistry):
        """Re-registering the same agent_id overwrites the previous entry."""
        await fresh_registry.register(
            WorkerEntry(agent_id="ae1", url="http://x", card={})
        )
        await fresh_registry.register(
            WorkerEntry(agent_id="ae1", url="http://y", card={})
        )
        workers = await fresh_registry.all_workers()
        assert len(workers) == 1
        assert workers[0].url == "http://y"

    async def test_deregister(self, fresh_registry: AgentRegistry):
        await fresh_registry.register(
            WorkerEntry(agent_id="ae1", url="http://x", card={})
        )
        assert await fresh_registry.deregister("ae1") is True
        assert await fresh_registry.all_workers() == []
        assert await fresh_registry.deregister("ae1") is False

    async def test_find_by_skill_role_match(
        self, registry_with_workers: AgentRegistry
    ):
        """Phase 3: each canonical role advertises its own role_<id> skill."""
        for role in ("analyst", "seeker", "devils_advocate", "synthesizer"):
            matches = await registry_with_workers.find_by_skill(f"role_{role}")
            assert {m.agent_id for m in matches} == {role}

    async def test_find_by_skill_no_match(self, registry_with_workers: AgentRegistry):
        assert await registry_with_workers.find_by_skill("no_such_skill") == []

    async def test_skill_lookup_handles_missing_skills(
        self, fresh_registry: AgentRegistry
    ):
        """A worker with no `skills` in its card never matches."""
        await fresh_registry.register(
            WorkerEntry(agent_id="bare", url="http://x", card={"name": "Bare"})
        )
        assert await fresh_registry.find_by_skill("role_analyst") == []

    async def test_concurrent_register(self, fresh_registry: AgentRegistry):
        """The asyncio.Lock serializes concurrent register calls cleanly."""
        import asyncio

        async def register(i: int):
            await fresh_registry.register(
                WorkerEntry(agent_id=f"w{i}", url=f"http://x:{i}", card={})
            )

        await asyncio.gather(*(register(i) for i in range(20)))
        workers = await fresh_registry.all_workers()
        assert len(workers) == 20


# ── HTTP routes ──────────────────────────────────────────────────────────────


@pytest.fixture
def registry_app(monkeypatch, fresh_registry):
    """Starlette app exposing the registry routes, with the singleton patched out."""
    from agents.orchestrator import agent_registry as ar_module

    monkeypatch.setattr(ar_module, "registry", fresh_registry)
    app = Starlette(routes=ar_module.registry_routes)
    return app


class TestRegistryRoutes:
    def test_register_happy_path(self, registry_app):
        with TestClient(registry_app) as client:
            r = client.post(
                "/registry/register",
                json={
                    "agent_id": "ae1",
                    "url": "http://localhost:9002",
                    "card": {"skills": [{"id": "debate"}]},
                },
            )
            assert r.status_code == 200
            assert r.json() == {"status": "registered", "agent_id": "ae1"}

    def test_register_rejects_bad_payload(self, registry_app):
        with TestClient(registry_app) as client:
            r = client.post("/registry/register", json={"agent_id": "only"})
            assert r.status_code == 400
            assert r.json()["status"] == "error"

    def test_list_agents_after_register(self, registry_app):
        with TestClient(registry_app) as client:
            client.post(
                "/registry/register",
                json={"agent_id": "x", "url": "http://x", "card": {}},
            )
            r = client.get("/registry/agents")
            assert r.status_code == 200
            data = r.json()
            assert data["count"] == 1
            assert data["workers"][0]["agent_id"] == "x"

    def test_list_by_skill(self, registry_app):
        with TestClient(registry_app) as client:
            client.post(
                "/registry/register",
                json={
                    "agent_id": "ae1",
                    "url": "http://x",
                    "card": {"skills": [{"id": "debate"}]},
                },
            )
            r = client.get("/registry/by-skill/debate")
            assert r.status_code == 200
            body = r.json()
            assert body["count"] == 1
            assert body["workers"][0]["agent_id"] == "ae1"

            r = client.get("/registry/by-skill/missing")
            assert r.json()["count"] == 0

    def test_deregister(self, registry_app):
        with TestClient(registry_app) as client:
            client.post(
                "/registry/register",
                json={"agent_id": "x", "url": "http://x", "card": {}},
            )
            r = client.delete("/registry/x")
            assert r.status_code == 200
            r = client.delete("/registry/x")
            assert r.status_code == 404
