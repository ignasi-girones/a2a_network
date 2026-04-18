"""Unit tests for Pydantic models in common.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.models import (
    AgentRoleConfig,
    DebateRound,
    FlowResult,
    NormalizedPrompt,
    RoleDecision,
    SkillDefinition,
    SubTask,
    TaskPlan,
    WorkerEntry,
)


class TestWorkerEntry:
    def test_minimal_creation(self):
        entry = WorkerEntry(agent_id="ae1", url="http://localhost:9002")
        assert entry.agent_id == "ae1"
        assert entry.url == "http://localhost:9002"
        assert entry.card == {}
        # registered_at is an ISO timestamp auto-generated
        assert "T" in entry.registered_at

    def test_with_card(self):
        card = {"name": "AE1", "skills": [{"id": "debate"}]}
        entry = WorkerEntry(agent_id="ae1", url="http://x", card=card)
        assert entry.card == card

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            WorkerEntry(agent_id="ae1")  # missing url


class TestSubTask:
    def test_minimal(self):
        t = SubTask(id="t1", description="Do something", required_skill="debate")
        assert t.depends_on == []
        assert t.perspective is None

    def test_with_dependencies(self):
        t = SubTask(
            id="t2",
            description="Reply",
            required_skill="debate",
            depends_on=["t1"],
            perspective="pro",
        )
        assert t.depends_on == ["t1"]
        assert t.perspective == "pro"


class TestTaskPlan:
    def test_empty_subtasks_allowed(self):
        # Validation of plan structure (non-empty, unique ids) lives in the
        # Planner, not on the Pydantic model — so an empty list is accepted here.
        p = TaskPlan(goal="x")
        assert p.subtasks == []
        assert p.max_workers == 4  # default

    def test_roundtrip_json(self):
        p = TaskPlan(
            goal="test",
            subtasks=[
                SubTask(id="t1", description="a", required_skill="s1"),
                SubTask(
                    id="t2",
                    description="b",
                    required_skill="s2",
                    depends_on=["t1"],
                ),
            ],
            max_workers=2,
        )
        rehydrated = TaskPlan.model_validate_json(p.model_dump_json())
        assert rehydrated == p


class TestRoleDecision:
    def test_max_rounds_bounded(self):
        """RoleDecision.max_rounds must be between 2 and 5 inclusive."""
        ae1 = AgentRoleConfig(role="A", perspective="pro")
        ae2 = AgentRoleConfig(role="B", perspective="con")

        with pytest.raises(ValidationError):
            RoleDecision(ae1_config=ae1, ae2_config=ae2, max_rounds=1)

        with pytest.raises(ValidationError):
            RoleDecision(ae1_config=ae1, ae2_config=ae2, max_rounds=6)

        rd = RoleDecision(ae1_config=ae1, ae2_config=ae2, max_rounds=3)
        assert rd.max_rounds == 3


class TestSkillDefinition:
    def test_defaults(self):
        s = SkillDefinition(id="x", name="X")
        assert s.description == ""
        assert s.tags == []


class TestNormalizedPrompt:
    def test_minimal(self):
        n = NormalizedPrompt(
            topic="T", domain="tech", question_type="decision"
        )
        assert n.constraints == []
        assert n.suggested_perspectives == []


class TestFlowResult:
    def test_roundtrip(self):
        fr = FlowResult(
            topic="T",
            ae1_role="A",
            ae2_role="B",
            ae1_initial_opinion="pro",
            ae2_initial_opinion="con",
            debate_rounds=[
                DebateRound(round_number=1, ae1_argument="x", ae2_argument="y")
            ],
            consensus_reached=True,
            summary="s",
        )
        dump = fr.model_dump()
        assert dump["consensus_reached"] is True
        assert len(dump["debate_rounds"]) == 1
