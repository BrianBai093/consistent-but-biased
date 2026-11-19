import json
from pathlib import Path

import pytest

from agents.moderator_agent import ModeratorAgent
from exceptions import StructuredOutputError
from graph import (
    FINAL_POSITION_QUESTION,
    _normalize_text_specialist_output,
    create_complete_mdt_graph,
    opening_input,
    thread_config,
)
from schemas import (
    AgentRole,
    DefinitiveSurgeryStatus,
    ModeratorDecision,
    SurgeryFieldStatus,
    WorkflowLimits,
)
from services.checkpoint import open_sqlite_checkpointer
from services.llm_client import FakeModelClient
from services.plan_convergence import apply_position, project_plan_draft
from tests.dynamic_factories import ask, finalize, registry_with_client
from tests.factories import (
    neurology_output,
    position,
    radiology_report,
    spine_output,
)
from tests.test_opening_graph import case_input


ALLOWED_IDS = {"case-page-001", "test-page-001", "mri-1"}


def test_later_turn_without_position_is_rejected() -> None:
    output = neurology_output().model_copy(update={"position_on_plan": None})
    draft = project_plan_draft(spine_output(), source_sequence=1)

    with pytest.raises(StructuredOutputError, match="position_on_plan"):
        _normalize_text_specialist_output(
            output,
            role=AgentRole.NEUROLOGIST,
            allowed_ids=ALLOWED_IDS,
            working_plan=draft,
            require_position=True,
        )


def test_position_with_wrong_role_or_unknown_evidence_is_rejected() -> None:
    draft = project_plan_draft(spine_output(), source_sequence=1)
    wrong_role = neurology_output().model_copy(
        update={
            "position_on_plan": position(
                role=AgentRole.ANESTHESIOLOGIST, draft_version=1, message_sequence=2
            )
        }
    )
    unknown_evidence = neurology_output().model_copy(
        update={
            "position_on_plan": position(
                role=AgentRole.NEUROLOGIST,
                draft_version=1,
                message_sequence=2,
                evidence_ids=["invented-id"],
            )
        }
    )

    with pytest.raises(StructuredOutputError, match="own role"):
        _normalize_text_specialist_output(
            wrong_role,
            role=AgentRole.NEUROLOGIST,
            allowed_ids=ALLOWED_IDS,
            working_plan=draft,
        )
    with pytest.raises(StructuredOutputError, match="unknown evidence id"):
        _normalize_text_specialist_output(
            unknown_evidence,
            role=AgentRole.NEUROLOGIST,
            allowed_ids=ALLOWED_IDS,
            working_plan=draft,
        )


def test_position_without_working_plan_is_stripped_not_attached() -> None:
    output = neurology_output()

    normalized = _normalize_text_specialist_output(
        output,
        role=AgentRole.NEUROLOGIST,
        allowed_ids=ALLOWED_IDS,
        working_plan=None,
    )

    assert normalized.position_on_plan is None


def test_final_position_turn_forbids_new_questions() -> None:
    output = neurology_output().model_copy(
        update={
            "questions_for_other_agents": [
                __import__("schemas").AgentQuestion(
                    target=AgentRole.SPINE_SURGEON,
                    question="Another question.",
                    evidence_ids=["case-page-001"],
                )
            ]
        }
    )
    draft = project_plan_draft(spine_output(), source_sequence=1)

    with pytest.raises(StructuredOutputError, match="cannot raise new questions"):
        _normalize_text_specialist_output(
            output,
            role=AgentRole.NEUROLOGIST,
            allowed_ids=ALLOWED_IDS,
            working_plan=draft,
            forbid_new_questions=True,
        )


async def test_moderator_route_receives_working_plan_and_field_matrix() -> None:
    draft = project_plan_draft(spine_output(), source_sequence=1)
    draft = apply_position(
        draft,
        position(
            role=AgentRole.NEUROLOGIST,
            draft_version=1,
            message_sequence=3,
            action=__import__("schemas").PlanPositionAction.OPPOSE,
            target_fields=["/recommended_surgery/levels"],
            reservations=["Unlocalized levels."],
        ),
    )
    client = FakeModelClient({ModeratorDecision: finalize()})
    moderator = ModeratorAgent(client)
    await moderator.route(
        case_preop_md="",
        tests_preop_md="",
        radiology_preop_md="",
        radiology_report=radiology_report(),
        chat_log=[],
        speaker_counts={},
        eligible_speakers=[AgentRole.NEUROLOGIST],
        pending_requests=[],
        unresolved_issues=[],
        disagreements=[],
        last_specialist=None,
        maximum_turns_per_specialist=5,
        available_image_ids=["mri-1"],
        working_plan=draft,
    )

    payload = json.loads(client.calls[-1]["user_prompt"])
    assert payload["operative_working_plan"]["version"] == 1
    assert payload["operative_working_plan"]["preferred_option_id"] == "option-1"
    levels_row = next(
        row
        for row in payload["operative_field_matrix"]
        if row["field"] == "/recommended_surgery/levels"
    )
    assert levels_row["oppositions"][0]["role"] == "neurologist"
    assert "source_documents" not in payload
    assert "case_preop_md" not in payload


@pytest.mark.asyncio
async def test_stalled_discussion_triggers_final_position_pass_then_precise_plan(
    tmp_path: Path,
) -> None:
    """P4: when the discussion stalls (no new information), each specialist
    gets one final position turn, then the forced plan still publishes every
    adequately supported field."""

    limits = WorkflowLimits(
        max_turns_per_specialist=5,
        max_total_specialist_turns=20,
        max_absolute_specialist_turns=24,
        max_workflow_steps=100,
        max_no_progress_turns=2,
    )
    registry, _ = registry_with_client(
        moderator_responses=[
            ask(AgentRole.SPINE_SURGEON),
            ask(AgentRole.SPINE_SURGEON),
        ],
    )
    async with open_sqlite_checkpointer(tmp_path / "pass.sqlite3") as saver:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=saver,
            limits=limits,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-final-pass"),
        )

    assert result["force_finalize"] is True
    assert result["termination_reason"].value == "no_new_information"
    assert result["convergence_pass_done"] is True
    positions = result["working_plan"].positions
    positioned_roles = {item.role for item in positions}
    assert positioned_roles == {
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PERIOPERATIVE_PHYSICIAN,
    }
    final_position_messages = [
        message
        for message in result["chat_log"]
        if FINAL_POSITION_QUESTION in message.content
    ]
    # Roles that positioned during normal discussion are skipped; the pass only
    # fills the remaining stances, so between one and three final-position turns.
    assert 1 <= len(final_position_messages) <= 3
    plan = result["final_plan"]
    assert plan.force_finalized is True
    assert plan.definitive_surgery_status is DefinitiveSurgeryStatus.ESTABLISHED
    assert plan.recommended_surgery.procedure_status is SurgeryFieldStatus.ESTABLISHED
    assert plan.recommended_surgery.procedure_name == (
        "L4-L5 decompression and discectomy"
    )


@pytest.mark.asyncio
async def test_absolute_limit_exhaustion_skips_the_pass(tmp_path: Path) -> None:
    limits = WorkflowLimits(
        max_turns_per_specialist=1,
        max_total_specialist_turns=5,
        max_absolute_specialist_turns=5,
        max_workflow_steps=100,
        max_no_progress_turns=20,
    )
    registry, _ = registry_with_client(
        moderator_responses=[ask(AgentRole.SPINE_SURGEON)],
    )
    async with open_sqlite_checkpointer(tmp_path / "skip.sqlite3") as saver:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=saver,
            limits=limits,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-absolute-skip"),
        )

    assert result["force_finalize"] is True
    assert result["run_metadata"].specialist_message_count == 5
    assert result["working_plan"].positions == []
    assert result["final_plan"].force_finalized is True
