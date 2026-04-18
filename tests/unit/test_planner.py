"""Unit tests for the Planner.

`llm_complete` is monkey-patched to return canned responses so no real LLM
call goes out. The focus is validation logic and the fallback/retry paths.
"""

from __future__ import annotations

import json

import pytest

from agents.orchestrator import planner as planner_module
from agents.orchestrator.planner import (
    Planner,
    _format_worker_catalog,
    _parse_plan,
)
from common.models import TaskPlan


pytestmark = pytest.mark.asyncio


# ── _format_worker_catalog ──────────────────────────────────────────────────


class TestFormatWorkerCatalog:
    def test_empty(self):
        assert "(no workers" in _format_worker_catalog([])

    def test_worker_without_skills(self):
        out = _format_worker_catalog(
            [{"agent_id": "bare", "url": "http://x", "skills": []}]
        )
        assert "bare" in out
        assert "no skills" in out

    def test_worker_with_skills(self):
        out = _format_worker_catalog(
            [
                {
                    "agent_id": "ae1",
                    "url": "http://x",
                    "skills": [
                        {"id": "debate", "name": "Debate", "tags": ["a", "b"]}
                    ],
                }
            ]
        )
        assert "ae1" in out
        assert "'debate'" in out
        assert "'Debate'" in out


# ── _parse_plan validation ───────────────────────────────────────────────────


class TestParsePlan:
    def test_valid(self):
        raw = json.dumps(
            {
                "goal": "x",
                "subtasks": [
                    {"id": "t1", "description": "a", "required_skill": "s1"},
                    {
                        "id": "t2",
                        "description": "b",
                        "required_skill": "s2",
                        "depends_on": ["t1"],
                    },
                ],
                "max_workers": 2,
            }
        )
        plan = _parse_plan(raw)
        assert isinstance(plan, TaskPlan)
        assert len(plan.subtasks) == 2

    def test_empty_subtasks_rejected(self):
        raw = json.dumps({"goal": "x", "subtasks": []})
        with pytest.raises(ValueError, match="no subtasks"):
            _parse_plan(raw)

    def test_duplicate_ids_rejected(self):
        raw = json.dumps(
            {
                "goal": "x",
                "subtasks": [
                    {"id": "t1", "description": "a", "required_skill": "s"},
                    {"id": "t1", "description": "b", "required_skill": "s"},
                ],
            }
        )
        with pytest.raises(ValueError, match="Duplicate"):
            _parse_plan(raw)

    def test_unknown_dependency_rejected(self):
        raw = json.dumps(
            {
                "goal": "x",
                "subtasks": [
                    {
                        "id": "t1",
                        "description": "a",
                        "required_skill": "s",
                        "depends_on": ["missing"],
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="unknown id"):
            _parse_plan(raw)

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_plan("not json")


# ── Planner.create_plan ──────────────────────────────────────────────────────


CATALOG_FULL = [
    {
        "agent_id": "normalizer",
        "url": "http://x",
        "skills": [{"id": "normalize_input", "name": "N", "tags": []}],
    },
    {
        "agent_id": "ae1",
        "url": "http://y",
        "skills": [{"id": "debate", "name": "D", "tags": []}],
    },
    {
        "agent_id": "feedback",
        "url": "http://z",
        "skills": [{"id": "format_verdict", "name": "F", "tags": []}],
    },
]


class TestPlannerCreate:
    async def test_happy_path(self, monkeypatch):
        """First LLM response is valid JSON — returned directly."""
        valid = json.dumps(
            {
                "goal": "test",
                "subtasks": [
                    {"id": "t1", "description": "a", "required_skill": "debate"}
                ],
            }
        )

        async def fake_complete(**kwargs):
            return valid

        monkeypatch.setattr(planner_module, "llm_complete", fake_complete)

        planner = Planner(model="fake/model")
        plan = await planner.create_plan("hello", CATALOG_FULL)
        assert plan.goal == "test"
        assert len(plan.subtasks) == 1

    async def test_retry_on_malformed_then_success(self, monkeypatch):
        """First response is bad JSON, second is good — planner retries once."""
        responses = iter(
            [
                "not json at all",
                json.dumps(
                    {
                        "goal": "x",
                        "subtasks": [
                            {"id": "t1", "description": "d", "required_skill": "debate"}
                        ],
                    }
                ),
            ]
        )

        async def fake_complete(**kwargs):
            return next(responses)

        monkeypatch.setattr(planner_module, "llm_complete", fake_complete)

        planner = Planner(model="fake/model")
        plan = await planner.create_plan("hi", CATALOG_FULL)
        assert plan.subtasks[0].id == "t1"

    async def test_fallback_plan_when_all_retries_fail(self, monkeypatch):
        """Both attempts fail → fallback to default debate plan (since catalog has all 3 skills)."""

        async def always_bad(**kwargs):
            return "garbage"

        monkeypatch.setattr(planner_module, "llm_complete", always_bad)

        planner = Planner(model="fake/model")
        plan = await planner.create_plan("hello", CATALOG_FULL)
        # Fallback plan: normalize → debate pro || debate con → format_verdict
        skills = {t.required_skill for t in plan.subtasks}
        assert skills == {"normalize_input", "debate", "format_verdict"}
        assert len(plan.subtasks) == 4

    async def test_fallback_fails_when_skills_missing(self, monkeypatch):
        """If the catalog lacks any of the fallback skills, planner raises."""

        async def always_bad(**kwargs):
            return "garbage"

        monkeypatch.setattr(planner_module, "llm_complete", always_bad)

        # Catalog has only 'debate' — missing normalize_input and format_verdict
        minimal_catalog = [
            {
                "agent_id": "ae1",
                "url": "http://y",
                "skills": [{"id": "debate", "name": "D", "tags": []}],
            }
        ]

        planner = Planner(model="fake/model")
        with pytest.raises(RuntimeError, match="failed to produce"):
            await planner.create_plan("hello", minimal_catalog)
