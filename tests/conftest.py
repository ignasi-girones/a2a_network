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
    """Registry pre-populated with the 4 base workers (normalizer, feedback, ae1, ae2)."""
    from common.models import WorkerEntry

    entries = [
        WorkerEntry(
            agent_id="normalizer",
            url="http://localhost:9001",
            card={
                "name": "Normalizer Agent",
                "skills": [
                    {"id": "normalize_input", "name": "Normalize Input", "tags": []}
                ],
            },
        ),
        WorkerEntry(
            agent_id="feedback",
            url="http://localhost:9004",
            card={
                "name": "Feedback Agent",
                "skills": [
                    {"id": "format_verdict", "name": "Format Verdict", "tags": []}
                ],
            },
        ),
        WorkerEntry(
            agent_id="ae1",
            url="http://localhost:9002",
            card={
                "name": "AE1",
                "skills": [{"id": "debate", "name": "Debate", "tags": ["debate"]}],
            },
        ),
        WorkerEntry(
            agent_id="ae2",
            url="http://localhost:9003",
            card={
                "name": "AE2",
                "skills": [{"id": "debate", "name": "Debate", "tags": ["debate"]}],
            },
        ),
    ]
    for e in entries:
        await fresh_registry.register(e)
    yield fresh_registry


# ── Sample TaskPlan ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_debate_plan():
    """The canonical 4-subtask plan produced for a yes/no debate prompt."""
    from common.models import SubTask, TaskPlan

    return TaskPlan(
        goal="Should a startup pick microservices or monolith?",
        subtasks=[
            SubTask(
                id="t1",
                description="Normalize input",
                required_skill="normalize_input",
            ),
            SubTask(
                id="t2",
                description="Argue pro",
                required_skill="debate",
                depends_on=["t1"],
                perspective="pro",
            ),
            SubTask(
                id="t3",
                description="Argue con",
                required_skill="debate",
                depends_on=["t1"],
                perspective="con",
            ),
            SubTask(
                id="t4",
                description="Synthesize verdict",
                required_skill="format_verdict",
                depends_on=["t2", "t3"],
            ),
        ],
        max_workers=3,
    )


# ── Project root on sys.path ─────────────────────────────────────────────────
# Some environments invoke pytest from outside the project root. Adding the
# project root to sys.path at collection time keeps `from agents...` imports working.

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
