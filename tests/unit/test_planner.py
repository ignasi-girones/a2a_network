"""Unit tests for the Phase 3 Planner.

The Phase 3 planner produces a fixed canonical sextet:
  Analista → {Buscador, Abogado del Diablo}
           → {Empírico, Pragmático} (depend on Buscador output)
           → Sintetizador (depends on all)

The LLM is consulted only to extract `goal` and `claim` from the user input.
Skills are always of the form ``role_<role_id>``; the worker catalog is
informational, not constraining.

`llm_complete` is monkey-patched so no real LLM call goes out.
"""

from __future__ import annotations

import json

import pytest

from agents.orchestrator import planner as planner_module
from agents.orchestrator.planner import (
    Planner,
    _build_quartet_plan,  # backward-compat alias → same as _build_sextet_plan
    _format_worker_catalog,
    _parse_extraction,
    _validate_role_coverage,
)
from common.models import CANONICAL_ROLES

N_ROLES = len(CANONICAL_ROLES)  # 6


# ── _format_worker_catalog ──────────────────────────────────────────────────


class TestFormatWorkerCatalog:
    def test_empty(self):
        assert "(no workers" in _format_worker_catalog([])

    def test_single_worker(self):
        out = _format_worker_catalog(
            [
                {
                    "agent_id": "analyst",
                    "url": "http://x",
                    "skills": [{"id": "role_analyst", "name": "Analista"}],
                }
            ]
        )
        assert "analyst" in out
        assert "role_analyst" in out


# ── _parse_extraction ────────────────────────────────────────────────────────


class TestParseExtraction:
    def test_valid(self):
        raw = json.dumps({"goal": "Debatir IA", "claim": "IA es transformadora"})
        goal, claim = _parse_extraction(raw)
        assert goal == "Debatir IA"
        assert claim == "IA es transformadora"

    def test_missing_claim_rejected(self):
        with pytest.raises(ValueError, match="missing required fields"):
            _parse_extraction(json.dumps({"goal": "x"}))

    def test_missing_goal_rejected(self):
        with pytest.raises(ValueError, match="missing required fields"):
            _parse_extraction(json.dumps({"claim": "x"}))

    def test_empty_strings_rejected(self):
        with pytest.raises(ValueError, match="missing required fields"):
            _parse_extraction(json.dumps({"goal": "", "claim": "x"}))

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_extraction("not json")


# ── _build_quartet_plan (alias → sextet) ─────────────────────────────────────


class TestBuildQuartetPlan:
    def test_canonical_topology(self):
        plan = _build_quartet_plan("goal", "claim is true")
        assert plan.goal == "goal"
        assert plan.claim == "claim is true"
        assert len(plan.subtasks) == N_ROLES

        by_id = {t.id: t for t in plan.subtasks}
        # Phase 1
        assert by_id["t1"].role_id == "analyst"
        assert by_id["t1"].depends_on == []
        # Phase 2
        assert by_id["t2"].role_id == "seeker"
        assert by_id["t2"].depends_on == ["t1"]
        assert by_id["t3"].role_id == "devils_advocate"
        assert by_id["t3"].depends_on == ["t1"]
        # Phase 3
        assert by_id["t4"].role_id == "empiricist"
        assert set(by_id["t4"].depends_on) == {"t1", "t2"}
        assert by_id["t5"].role_id == "pragmatist"
        assert set(by_id["t5"].depends_on) == {"t1", "t2"}
        # Phase 4
        assert by_id["t6"].role_id == "synthesizer"
        assert set(by_id["t6"].depends_on) == {"t1", "t2", "t3", "t4", "t5"}

    def test_required_skills_are_role_skills(self):
        plan = _build_quartet_plan("g", "c")
        skills = {t.required_skill for t in plan.subtasks}
        assert skills == {f"role_{r}" for r in CANONICAL_ROLES}

    def test_descriptions_include_claim(self):
        plan = _build_quartet_plan("goal", "Microservices over monoliths")
        for t in plan.subtasks:
            assert "Microservices over monoliths" in t.description

    def test_max_workers_matches_sextet(self):
        plan = _build_quartet_plan("g", "c")
        assert plan.max_workers == N_ROLES


# ── _validate_role_coverage ─────────────────────────────────────────────────


class TestRoleCoverage:
    def test_full_coverage(self):
        catalog = [
            {
                "agent_id": r,
                "skills": [{"id": f"role_{r}"}],
            }
            for r in CANONICAL_ROLES
        ]
        covered, missing = _validate_role_coverage(catalog)
        assert covered == set(CANONICAL_ROLES)
        assert missing == set()

    def test_partial_coverage(self):
        catalog = [
            {"agent_id": "analyst", "skills": [{"id": "role_analyst"}]},
        ]
        covered, missing = _validate_role_coverage(catalog)
        assert covered == {"analyst"}
        assert missing == {
            "seeker", "devils_advocate", "empiricist", "pragmatist", "synthesizer"
        }

    def test_empty_catalog(self):
        covered, missing = _validate_role_coverage([])
        assert covered == set()
        assert missing == set(CANONICAL_ROLES)

    def test_irrelevant_skills_ignored(self):
        catalog = [
            {"agent_id": "x", "skills": [{"id": "debate"}, {"id": "search"}]},
        ]
        covered, missing = _validate_role_coverage(catalog)
        assert covered == set()
        assert missing == set(CANONICAL_ROLES)


# ── Planner.create_plan ─────────────────────────────────────────────────────


CATALOG_FULL = [
    {
        "agent_id": role,
        "url": f"http://{role}",
        "skills": [{"id": f"role_{role}", "name": role.title(), "tags": []}],
    }
    for role in CANONICAL_ROLES
]


@pytest.mark.asyncio
class TestPlannerCreate:
    async def test_happy_path(self, monkeypatch):
        """First LLM response is valid {goal, claim} → sextet returned."""
        valid = json.dumps(
            {"goal": "Decidir microservicios", "claim": "Microservicios > monolito"}
        )

        async def fake_complete(**kwargs):
            return valid

        monkeypatch.setattr(planner_module, "llm_complete", fake_complete)

        planner = Planner(model="fake/model")
        plan = await planner.create_plan("hello", CATALOG_FULL)
        assert plan.goal == "Decidir microservicios"
        assert plan.claim == "Microservicios > monolito"
        assert len(plan.subtasks) == N_ROLES
        assert {t.role_id for t in plan.subtasks} == set(CANONICAL_ROLES)

    async def test_retry_on_malformed_then_success(self, monkeypatch):
        """First response bad JSON, second response valid — retry once."""
        responses = iter(
            [
                "not json at all",
                json.dumps({"goal": "g", "claim": "c"}),
            ]
        )

        async def fake_complete(**kwargs):
            return next(responses)

        monkeypatch.setattr(planner_module, "llm_complete", fake_complete)

        planner = Planner(model="fake/model")
        plan = await planner.create_plan("hi", CATALOG_FULL)
        assert plan.goal == "g"
        assert plan.claim == "c"
        assert len(plan.subtasks) == N_ROLES

    async def test_retry_on_missing_claim(self, monkeypatch):
        responses = iter(
            [
                json.dumps({"goal": "only goal"}),
                json.dumps({"goal": "g", "claim": "c"}),
            ]
        )

        async def fake_complete(**kwargs):
            return next(responses)

        monkeypatch.setattr(planner_module, "llm_complete", fake_complete)

        planner = Planner(model="fake/model")
        plan = await planner.create_plan("hi", CATALOG_FULL)
        assert plan.claim == "c"

    async def test_fallback_to_user_input_on_repeated_failure(self, monkeypatch):
        """All retries fail → fallback uses user_input as both goal+claim.

        Phase 3 never raises here — the dialectic backbone matters more than
        clean extraction. The orchestrator should still see a valid sextet.
        """

        async def always_bad(**kwargs):
            return "garbage"

        monkeypatch.setattr(planner_module, "llm_complete", always_bad)

        planner = Planner(model="fake/model")
        plan = await planner.create_plan("microservicios o monolito", CATALOG_FULL)
        assert len(plan.subtasks) == N_ROLES
        assert "microservicios o monolito" in plan.goal
        assert "microservicios o monolito" in plan.claim

    async def test_fallback_with_empty_input(self, monkeypatch):
        """Even empty user_input still produces a sextet (sentinel goal)."""

        async def always_bad(**kwargs):
            return "garbage"

        monkeypatch.setattr(planner_module, "llm_complete", always_bad)

        planner = Planner(model="fake/model")
        plan = await planner.create_plan("   ", CATALOG_FULL)
        assert len(plan.subtasks) == N_ROLES
        assert plan.goal  # not empty

    async def test_works_with_empty_catalog(self, monkeypatch):
        """Catalog gaps are surfaced via warning, not a hard failure."""
        valid = json.dumps({"goal": "g", "claim": "c"})

        async def fake_complete(**kwargs):
            return valid

        monkeypatch.setattr(planner_module, "llm_complete", fake_complete)

        planner = Planner(model="fake/model")
        plan = await planner.create_plan("topic", [])
        assert len(plan.subtasks) == N_ROLES


@pytest.mark.asyncio
class TestPlannerReplan:
    async def test_replan_keeps_canonical_sextet(self, monkeypatch):
        """replan returns the same sextet, preserving goal+claim."""
        from common.models import SubTask

        async def fake_complete(**kwargs):
            return json.dumps({"goal": "ignored", "claim": "ignored"})

        monkeypatch.setattr(planner_module, "llm_complete", fake_complete)

        planner = Planner(model="fake/model")
        original = _build_quartet_plan("g", "c")
        failed = SubTask(
            id="t1",
            description="d",
            role_id="analyst",
            required_skill="role_analyst",
        )

        replanned = await planner.replan(original, failed, "boom", CATALOG_FULL)
        assert replanned.goal == "g"
        assert replanned.claim == "c"
        assert len(replanned.subtasks) == N_ROLES
