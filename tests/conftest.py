"""Shared pytest fixtures for the a2a_network test suite.

The orchestrator modules hold module-level singletons (`registry`, `_spawner`)
that persist across test runs. Tests that touch global state MUST use the
`fresh_registry` / `isolated_spawner` fixtures to reset them.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio


# ── Environment setup ────────────────────────────────────────────────────────
# Prevent real LLM calls from leaking into unit tests; force dummy keys so
# litellm doesn't complain when modules import common.llm_provider at import time.
os.environ.setdefault("GROQ_API_KEY", "test-dummy")
os.environ.setdefault("GEMINI_API_KEY", "test-dummy")
os.environ.setdefault("MISTRAL_API_KEY", "test-dummy")
os.environ.setdefault("CEREBRAS_API_KEY", "test-dummy")


# ── Registry fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fresh_registry():
    """Return a brand-new AgentRegistry (not the module singleton)."""
    from agents.orchestrator.agent_registry import AgentRegistry

    return AgentRegistry()


@pytest_asyncio.fixture
async def registry_with_workers(fresh_registry) -> AsyncIterator:
    """Registry pre-populated with the Phase 3 dialectic quartet workers."""
    from common.models import CANONICAL_ROLES, WorkerEntry

    role_ports = {
        "analyst":         9002,
        "seeker":          9003,
        "devils_advocate": 9087,
        "empiricist":      9089,
        "pragmatist":      9090,
        "synthesizer":     9088,
    }
    entries = []
    for role in CANONICAL_ROLES:
        entries.append(
            WorkerEntry(
                agent_id=role,
                url=f"http://localhost:{role_ports[role]}",
                card={
                    "name": f"Worker ({role})",
                    "skills": [
                        {
                            "id": f"role_{role}",
                            "name": role.title(),
                            "tags": ["dialectic", role],
                        }
                    ],
                },
            )
        )
    for e in entries:
        await fresh_registry.register(e)
    yield fresh_registry


# ── Sample TaskPlan ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_debate_plan():
    """The canonical Phase 3 sextet TaskPlan.

    t1 (analyst) ──► t2 (seeker, deps=t1) ──► t4 (empiricist, deps=t1,t2) ──┐
                 └── t3 (DA, deps=t1)      └── t5 (pragmatist, deps=t1,t2) ─┴► t6 (synthesizer)
    """
    from agents.orchestrator.planner import _build_quartet_plan

    return _build_quartet_plan(
        goal="Should a startup pick microservices or monolith?",
        claim="Microservices outperform monoliths for early-stage startups",
    )


# ── Project root on sys.path ─────────────────────────────────────────────────
# Some environments invoke pytest from outside the project root. Adding the
# project root to sys.path at collection time keeps `from agents...` imports working.

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
