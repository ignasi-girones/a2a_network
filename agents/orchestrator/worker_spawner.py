"""WorkerSpawner — launch additional specialized worker subprocesses on demand.

When a TaskPlan requires more concurrent workers of a given skill than are
currently registered, AgenticOrchestrator calls `WorkerSpawner.spawn()` to
bring up a new `agents.specialized` process on a free port from the pool.

The new worker self-registers with the orchestrator's AgentRegistry via the
same `register_self_with_orchestrator` lifespan hook as the static workers,
so we only need to wait for its agent_id to appear in the registry.

Lifecycle:
    spawn(agent_id)  → allocates port, launches subprocess, waits for
                       registry appearance (with timeout), returns port
    teardown(agent_id) → terminates subprocess, best-effort registry cleanup
    teardown_all()     → shutdown all spawned workers (called on exit)

Notes:
- Uses `sys.executable` so we stay inside the current venv on Windows.
- Ports are tracked in `_in_use` to avoid double-allocation; freed on teardown.
- We only spawn; reconfiguration (role/perspective) stays with the caller
  via the existing POST /internal/configure endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field

from agents.orchestrator.agent_registry import AgentRegistry
from common.config import settings

logger = logging.getLogger(__name__)


@dataclass
class _SpawnedWorker:
    agent_id: str
    port: int
    process: asyncio.subprocess.Process
    url: str = field(init=False)

    def __post_init__(self) -> None:
        self.url = f"http://localhost:{self.port}"


class WorkerSpawner:
    """Manages on-demand launch and teardown of specialized worker processes."""

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry
        self._in_use: set[int] = set()
        self._spawned: dict[str, _SpawnedWorker] = {}
        self._lock = asyncio.Lock()

    def _allocate_port(self) -> int:
        """Pick the first unused port from the configured pool."""
        start = settings.worker_port_pool_start
        size = settings.worker_port_pool_size
        for p in range(start, start + size):
            if p not in self._in_use:
                self._in_use.add(p)
                return p
        raise RuntimeError(
            f"Worker port pool exhausted (start={start}, size={size})"
        )

    async def spawn(
        self,
        agent_id: str,
        wait_timeout_s: float = 30.0,
        poll_interval_s: float = 0.5,
    ) -> _SpawnedWorker:
        """Launch a new specialized worker and wait for it to self-register.

        The subprocess inherits stdout/stderr so its logs appear alongside
        the orchestrator's — simpler for the demo than piping. If the
        subprocess dies before registering, the port is freed and we raise.
        """
        async with self._lock:
            if agent_id in self._spawned:
                return self._spawned[agent_id]

            port = self._allocate_port()

        logger.info("Spawning worker %s on port %d", agent_id, port)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "agents.specialized",
            "--port",
            str(port),
            "--agent-id",
            agent_id,
        )

        # Poll the registry until the new worker appears, or the process exits,
        # or we time out.
        deadline = asyncio.get_event_loop().time() + wait_timeout_s
        while True:
            if process.returncode is not None:
                self._in_use.discard(port)
                raise RuntimeError(
                    f"Spawned worker {agent_id} died with "
                    f"exit code {process.returncode}"
                )

            entry = await self.registry.find_by_agent_id(agent_id)
            if entry is not None:
                worker = _SpawnedWorker(
                    agent_id=agent_id, port=port, process=process
                )
                self._spawned[agent_id] = worker
                logger.info(
                    "Worker %s registered @ %s after %.1fs",
                    agent_id,
                    worker.url,
                    wait_timeout_s - (deadline - asyncio.get_event_loop().time()),
                )
                return worker

            if asyncio.get_event_loop().time() > deadline:
                process.terminate()
                self._in_use.discard(port)
                raise TimeoutError(
                    f"Worker {agent_id} did not register within "
                    f"{wait_timeout_s}s"
                )

            await asyncio.sleep(poll_interval_s)

    async def teardown(self, agent_id: str) -> bool:
        """Terminate a previously-spawned worker. Returns True if we had it."""
        async with self._lock:
            worker = self._spawned.pop(agent_id, None)
            if worker is None:
                return False
            self._in_use.discard(worker.port)

        logger.info("Tearing down worker %s (port %d)", agent_id, worker.port)
        try:
            worker.process.terminate()
            try:
                await asyncio.wait_for(worker.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker %s did not terminate in 5s, killing", agent_id
                )
                worker.process.kill()
                await worker.process.wait()
        except ProcessLookupError:
            pass  # already dead

        # Best-effort registry cleanup in case the lifespan hook didn't fire.
        await self.registry.deregister(agent_id)
        return True

    async def teardown_all(self) -> None:
        """Terminate every worker we spawned. Called on orchestrator shutdown."""
        async with self._lock:
            ids = list(self._spawned.keys())
        for agent_id in ids:
            try:
                await self.teardown(agent_id)
            except Exception as e:
                logger.warning("Error tearing down %s: %s", agent_id, e)


# Module-level singleton. Constructed lazily so importing this module from a
# test or a plan-preview context doesn't require the registry to be alive.
_spawner: WorkerSpawner | None = None


def get_spawner() -> WorkerSpawner:
    """Return the process-wide spawner instance, binding it to the registry."""
    global _spawner
    if _spawner is None:
        from agents.orchestrator.agent_registry import registry as _registry
        _spawner = WorkerSpawner(_registry)
    return _spawner
