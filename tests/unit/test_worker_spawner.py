"""Unit tests for WorkerSpawner.

We mock `asyncio.create_subprocess_exec` so no real Python subprocess is ever
launched. The goal is to exercise port allocation, registry polling, timeout
behavior, and teardown bookkeeping.
"""

from __future__ import annotations

import asyncio

import pytest

from agents.orchestrator import worker_spawner as ws_module
from agents.orchestrator.worker_spawner import WorkerSpawner
from common.config import settings
from common.models import WorkerEntry


pytestmark = pytest.mark.asyncio


class FakeProcess:
    """Minimal asyncio.subprocess.Process stand-in."""

    def __init__(self, exits_with: int | None = None):
        self._returncode = exits_with
        self.terminate_called = False
        self.kill_called = False

    @property
    def returncode(self):
        return self._returncode

    def terminate(self):
        self.terminate_called = True
        self._returncode = 0

    def kill(self):
        self.kill_called = True
        self._returncode = 137

    async def wait(self):
        while self._returncode is None:
            await asyncio.sleep(0)
        return self._returncode


def patch_subprocess(monkeypatch, proc: FakeProcess):
    """Patch asyncio.create_subprocess_exec to return `proc`."""
    captured: dict = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return captured


# ── Port allocation ─────────────────────────────────────────────────────────


class TestPortAllocation:
    def test_first_port_is_pool_start(self, fresh_registry):
        spawner = WorkerSpawner(fresh_registry)
        assert spawner._allocate_port() == settings.worker_port_pool_start

    def test_sequential_allocation(self, fresh_registry):
        spawner = WorkerSpawner(fresh_registry)
        ports = [spawner._allocate_port() for _ in range(3)]
        assert ports == [
            settings.worker_port_pool_start,
            settings.worker_port_pool_start + 1,
            settings.worker_port_pool_start + 2,
        ]

    def test_exhaustion_raises(self, fresh_registry, monkeypatch):
        monkeypatch.setattr(settings, "worker_port_pool_size", 2)
        spawner = WorkerSpawner(fresh_registry)
        spawner._allocate_port()
        spawner._allocate_port()
        with pytest.raises(RuntimeError, match="exhausted"):
            spawner._allocate_port()


# ── spawn() happy/error paths ────────────────────────────────────────────────


class TestSpawn:
    async def test_spawn_registers_and_returns(self, fresh_registry, monkeypatch):
        """Worker appears in registry → spawn() returns the _SpawnedWorker."""
        proc = FakeProcess()
        patch_subprocess(monkeypatch, proc)

        async def register_after_delay():
            # Simulate the subprocess eventually self-registering.
            await asyncio.sleep(0.05)
            await fresh_registry.register(
                WorkerEntry(
                    agent_id="dyn_1",
                    url="http://localhost:9010",
                    card={"skills": [{"id": "debate"}]},
                )
            )

        spawner = WorkerSpawner(fresh_registry)
        registration_task = asyncio.create_task(register_after_delay())

        worker = await spawner.spawn(
            "dyn_1", wait_timeout_s=2.0, poll_interval_s=0.02
        )
        await registration_task

        assert worker.agent_id == "dyn_1"
        assert worker.port == settings.worker_port_pool_start
        assert "dyn_1" in spawner._spawned

    async def test_spawn_idempotent(self, fresh_registry, monkeypatch):
        """Spawning the same agent_id twice returns the same record."""
        proc = FakeProcess()
        patch_subprocess(monkeypatch, proc)

        await fresh_registry.register(
            WorkerEntry(agent_id="dyn_1", url="http://x", card={})
        )

        spawner = WorkerSpawner(fresh_registry)
        w1 = await spawner.spawn(
            "dyn_1", wait_timeout_s=1.0, poll_interval_s=0.01
        )
        w2 = await spawner.spawn(
            "dyn_1", wait_timeout_s=1.0, poll_interval_s=0.01
        )
        assert w1 is w2

    async def test_spawn_times_out_and_frees_port(
        self, fresh_registry, monkeypatch
    ):
        proc = FakeProcess()  # never exits, never registers
        patch_subprocess(monkeypatch, proc)

        spawner = WorkerSpawner(fresh_registry)
        with pytest.raises(TimeoutError):
            await spawner.spawn(
                "dyn_1", wait_timeout_s=0.1, poll_interval_s=0.02
            )
        # Port was released so the next allocation re-uses the pool start.
        assert spawner._allocate_port() == settings.worker_port_pool_start
        assert proc.terminate_called

    async def test_spawn_detects_dead_subprocess(
        self, fresh_registry, monkeypatch
    ):
        proc = FakeProcess(exits_with=1)  # already dead
        patch_subprocess(monkeypatch, proc)

        spawner = WorkerSpawner(fresh_registry)
        with pytest.raises(RuntimeError, match="died"):
            await spawner.spawn(
                "dyn_1", wait_timeout_s=1.0, poll_interval_s=0.01
            )


# ── teardown / teardown_all ──────────────────────────────────────────────────


class TestTeardown:
    async def test_teardown_releases_port(self, fresh_registry, monkeypatch):
        proc = FakeProcess()
        patch_subprocess(monkeypatch, proc)

        await fresh_registry.register(
            WorkerEntry(agent_id="dyn_1", url="http://x", card={})
        )
        spawner = WorkerSpawner(fresh_registry)
        await spawner.spawn("dyn_1", wait_timeout_s=1.0, poll_interval_s=0.01)

        port = list(spawner._in_use)[0]
        assert await spawner.teardown("dyn_1") is True
        assert port not in spawner._in_use
        assert proc.terminate_called

    async def test_teardown_unknown_returns_false(self, fresh_registry):
        spawner = WorkerSpawner(fresh_registry)
        assert await spawner.teardown("does_not_exist") is False

    async def test_teardown_all(self, fresh_registry, monkeypatch):
        await fresh_registry.register(
            WorkerEntry(agent_id="a", url="http://x", card={})
        )
        await fresh_registry.register(
            WorkerEntry(agent_id="b", url="http://y", card={})
        )
        patch_subprocess(monkeypatch, FakeProcess())

        spawner = WorkerSpawner(fresh_registry)
        await spawner.spawn("a", wait_timeout_s=1.0, poll_interval_s=0.01)
        patch_subprocess(monkeypatch, FakeProcess())
        await spawner.spawn("b", wait_timeout_s=1.0, poll_interval_s=0.01)

        await spawner.teardown_all()
        assert spawner._spawned == {}


# ── get_spawner singleton ────────────────────────────────────────────────────


class TestGetSpawner:
    def test_singleton_behavior(self, monkeypatch):
        # Reset the module-level singleton before the test
        monkeypatch.setattr(ws_module, "_spawner", None)
        s1 = ws_module.get_spawner()
        s2 = ws_module.get_spawner()
        assert s1 is s2
