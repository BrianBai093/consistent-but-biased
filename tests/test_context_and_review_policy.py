import json
from pathlib import Path

import pytest

from agents import ModeratorAgent
from exceptions import StructuredOutputError
from graph import create_complete_mdt_graph, opening_input, thread_config
from schemas import (
    AgentRole,
    CaseImage,
    ChatMessage,
    ClaimEvidence,
    Disagreement,
    DocumentPage,
    FinalPlan,
    FinalPlanReview,
    FinalReviewIssue,
    ImageQuality,
    Modality,
    SpineSurgeonOutput,
    SpecialistContribution,
)
from services.checkpoint import open_sqlite_checkpointer
from services.context_policy import (
    issue_snapshot,
    source_documents_for_turn,
    validate_specialist_evidence,
)
from services.final_plan_review import validate_final_plan_evidence
from services.llm_client import FakeModelClient
from state import create_initial_state
from tests.dynamic_factories import ask, finalize, registry_with_client
from tests.factories import (
    all_fake_responses,
    case_input,
    final_plan,
    radiology_report,
    spine_output,
)


def local_case(tmp_path: Path):
    return case_input(
        case_id="case-context-policy",
        artifact_dir=str(tmp_path / "derived"),
        images=[
            CaseImage(
                image_id="mri-1",
                modality=Modality.MRI,
                path_or_url="https://example.test/mri-1.jpg",
                body_region="lumbar spine",
            )
        ],
    )


def test_chat_message_ids_cannot_be_used_as_clinical_evidence() -> None:
    chat_id = "case-context-policy:spine_surgeon:1"
    bad_specialist = spine_output().model_copy(
        update={
            "claim_evidence": [
                ClaimEvidence(
                    field_path="recommended_procedure",
                    claim="A procedure is supported.",
                    support_type="proposal",
                    evidence_ids=[chat_id],
                )
            ]
        }
    )
    with pytest.raises(StructuredOutputError, match="unknown evidence id"):
        validate_specialist_evidence(
            bad_specialist,
            allowed_ids={"case-page-001", "test-page-001", "mri-1"},
        )

    bad_plan = final_plan().model_copy(
        update={
            "claim_evidence": [
                claim.model_copy(update={"evidence_ids": [chat_id]})
                for claim in final_plan().claim_evidence
            ]
        }
    )
    with pytest.raises(StructuredOutputError, match="unknown evidence id"):
        validate_final_plan_evidence(
            bad_plan,
            allowed_ids={"case-page-001", "test-page-001", "mri-1"},
        )


def test_completed_assessment_requires_documented_field_evidence() -> None:
    unsupported = final_plan()
    unsupported = unsupported.model_copy(
        update={
            "perioperative_risks": unsupported.perioperative_risks.model_copy(
                update={
                    "anesthesia_plan": (
                        "Airway assessment has been completed; proceed with anesthesia."
                    )
                }
            )
        }
    )
    allowed_ids = {"case-page-001", "test-page-001", "mri-1"}

    with pytest.raises(StructuredOutputError, match="completed assessment"):
        validate_final_plan_evidence(unsupported, allowed_ids=allowed_ids)

    supported = unsupported.model_copy(
        update={
            "claim_evidence": [
                *unsupported.claim_evidence,
                ClaimEvidence(
                    field_path="/perioperative_risks/anesthesia_plan",
                    claim="The airway assessment is documented as completed.",
                    support_type="documented",
                    evidence_ids=["case-page-001"],
                ),
            ]
        }
    )
    assert validate_final_plan_evidence(supported, allowed_ids=allowed_ids) == supported


def test_followup_context_contains_only_selected_sources_and_recent_opinions(
    tmp_path: Path,
) -> None:
    state = create_initial_state(local_case(tmp_path))
    state["case_pages"] = [
        DocumentPage(
            page_id=f"case-page-{index:03d}",
            page_number=index,
            path_or_url=f"https://example.test/case-{index}.jpg",
            original_filename=f"private-{index}.jpg",
        )
        for index in range(1, 3)
    ]
    state["case_preop_md"] = (
        "<!-- source: case-page-001 -->\n\nRelevant history.\n\n"
        "<!-- source: case-page-002 -->\n\nUnrelated old page.\n\n"
        "<!-- source: addendum-safe-1 -->\n\nNew user addendum."
    )
    state["tests_preop_md"] = (
        "<!-- source: test-page-001 -->\n\nRelevant laboratory result.\n\n"
        "<!-- source: test-page-002 -->\n\nUnrelated laboratory page."
    )
    state["radiology_report"] = radiology_report()
    state["radiology_preop_md"] = "Full radiology narrative must not be reused."
    state["speaker_counts"] = {
        **state["speaker_counts"],
        AgentRole.SPINE_SURGEON.value: 1,
    }
    prior = spine_output()
    state["specialist_contributions"] = [
        SpecialistContribution(
            role=AgentRole.SPINE_SURGEON,
            turn_number=1,
            message_sequence=2,
            structured_output=prior,
        )
    ]
    state["chat_log"] = [
        ChatMessage(
            message_id="case-context-policy:spine_surgeon:1",
            sequence=2,
            speaker_id=AgentRole.SPINE_SURGEON.value,
            speaker_name="Spine Surgeon",
            speaker_role=AgentRole.SPINE_SURGEON,
            content="Prior structured opinion.",
            evidence_ids=["case-page-001", "mri-1"],
            turn_number=1,
        ),
        ChatMessage(
            message_id="case-context-policy:neurologist:1",
            sequence=4,
            speaker_id=AgentRole.NEUROLOGIST.value,
            speaker_name="Neurologist",
            speaker_role=AgentRole.NEUROLOGIST,
            content="Recent relevant opinion.",
            evidence_ids=["test-page-001"],
            turn_number=1,
        ),
    ]

    documents = source_documents_for_turn(state, AgentRole.SPINE_SURGEON)

    assert "Relevant history" in documents["case_preop_md"]
    assert "New user addendum" in documents["case_preop_md"]
    assert "Unrelated old page" not in documents["case_preop_md"]
    assert "Relevant laboratory result" in documents["tests_preop_md"]
    assert "Unrelated laboratory page" not in documents["tests_preop_md"]
    assert "Right L4-L5 disc extrusion" in documents["radiology_preop_md"]
    assert "Full radiology narrative must not be reused" not in documents[
        "radiology_preop_md"
    ]
    assert "Only MRI key images are available" not in documents["radiology_preop_md"]


def test_issue_snapshot_is_bounded_and_preserves_workflow_owned_issues() -> None:
    model_issues = [f"Model issue {index}" for index in range(20)]
    model_issues.append("  model   issue 0 ")
    workflow_issues = ["Core preparation failed", "Missing required records"]

    result = issue_snapshot(model_issues, workflow_issues)

    assert len(result) == 12
    assert result[:2] == workflow_issues
    assert result.count("Model issue 0") == 1


@pytest.mark.asyncio
async def test_moderator_route_never_serializes_source_documents_or_full_chat() -> None:
    client = FakeModelClient(all_fake_responses())
    agent = ModeratorAgent(client)
    disagreement = Disagreement(
        agents=[AgentRole.SPINE_SURGEON, AgentRole.NEUROLOGIST],
        issue="Level remains disputed.",
        positions=["L4-L5", "Uncertain"],
    )
    chat = [
        ChatMessage(
            message_id="old-spine",
            sequence=1,
            speaker_id=AgentRole.SPINE_SURGEON.value,
            speaker_name="Spine Surgeon",
            speaker_role=AgentRole.SPINE_SURGEON,
            content="OLD OPINION MUST BE OMITTED",
            turn_number=1,
        ),
        ChatMessage(
            message_id="new-spine",
            sequence=2,
            speaker_id=AgentRole.SPINE_SURGEON.value,
            speaker_name="Spine Surgeon",
            speaker_role=AgentRole.SPINE_SURGEON,
            content="Latest routing conclusion.",
            turn_number=2,
        ),
    ]

    await agent.route(
        case_preop_md="PRIVATE CASE DOCUMENT H:\\patient\\case.jpg",
        tests_preop_md="PRIVATE TEST DOCUMENT",
        radiology_preop_md="PRIVATE RADIOLOGY MARKDOWN",
        radiology_report=radiology_report(),
        chat_log=chat,
        speaker_counts={role.value: 1 for role in AgentRole if role is not AgentRole.MODERATOR},
        eligible_speakers=[AgentRole.SPINE_SURGEON],
        pending_requests=[],
        unresolved_issues=["Question remains open."],
        disagreements=[disagreement],
        last_specialist=AgentRole.SPINE_SURGEON,
        maximum_turns_per_specialist=2,
        available_image_ids=["mri-1"],
    )

    prompt = client.calls[-1]["user_prompt"]
    payload = json.loads(prompt)
    assert "PRIVATE" not in prompt
    assert "H:\\patient" not in prompt
    assert "OLD OPINION MUST BE OMITTED" not in prompt
    assert payload["latest_specialist_messages"][0]["content"] == (
        "Latest routing conclusion."
    )
    assert payload["current_disagreements"][0]["issue"] == disagreement.issue


@pytest.mark.asyncio
async def test_final_synthesis_uses_latest_contribution_and_review_is_blind(
    tmp_path: Path,
) -> None:
    first = spine_output()
    latest_option = first.operative_options[0].model_copy(
        update={"procedure_name": "Latest conditional procedure"}
    )
    latest = first.model_copy(
        update={
            "public_message": "Latest spine opinion.",
            "recommended_procedure": "Latest conditional procedure",
            "operative_options": [latest_option],
        }
    )
    final_option = latest.operative_options[0].model_copy(
        update={"procedure_name": "Final revised procedure"}
    )
    final_latest = latest.model_copy(
        update={
            "public_message": "Final spine opinion.",
            "recommended_procedure": "Final revised procedure",
            "operative_options": [final_option],
        }
    )
    route = [
        ask(AgentRole.SPINE_SURGEON),
        ask(AgentRole.NEUROLOGIST),
        ask(AgentRole.ANESTHESIOLOGIST),
        ask(AgentRole.PERIOPERATIVE_PHYSICIAN),
        ask(AgentRole.SPINE_SURGEON),
        finalize(),
    ]
    registry, client = registry_with_client(
        moderator_responses=route,
        overrides={SpineSurgeonOutput: [first, latest, final_latest]},
    )
    async with open_sqlite_checkpointer(tmp_path / "latest-and-blind.sqlite3") as saver:
        graph = create_complete_mdt_graph(agent_registry=registry, checkpointer=saver)
        result = await graph.ainvoke(
            opening_input(local_case(tmp_path)),
            config=thread_config("thread-latest-and-blind"),
        )

    synthesis_call = next(
        call for call in client.calls if call["output_model"] is FinalPlan
    )
    synthesis = json.loads(synthesis_call["user_prompt"])
    contributions = synthesis["latest_structured_contributions"]
    spine_contributions = [
        item for item in contributions if item["role"] == AgentRole.SPINE_SURGEON.value
    ]
    assert len(contributions) == 5
    assert len(spine_contributions) == 1
    assert spine_contributions[0]["turn_number"] == 3
    assert spine_contributions[0]["structured_output"]["recommended_procedure"] == (
        "Final revised procedure"
    )
    assert "public_chat" not in synthesis

    review_call = next(
        call for call in client.calls if call["output_model"] is FinalPlanReview
    )
    review = json.loads(review_call["user_prompt"])
    assert "latest_structured_contributions" not in review
    assert "public_chat" not in review
    assert "specialist_contributions" not in review
    # v5 assigns field-level statuses deterministically; a passing review is no
    # longer forced into a revision even when the discussion ended by limits.
    assert result["run_metadata"].final_review_outcome == "passed"
    assert result["final_plan"].recommended_surgery.procedure_name == (
        "Final revised procedure"
    )


@pytest.mark.asyncio
async def test_blind_review_can_trigger_exactly_one_revision(tmp_path: Path) -> None:
    review = FinalPlanReview(
        verdict="revise",
        issues=[
            FinalReviewIssue(
                field_path="/final_consensus/recommendation",
                problem="Make the uncertainty explicit.",
                evidence_ids=["case-page-001"],
            )
        ],
    )
    registry, client = registry_with_client(
        moderator_responses=[
            ask(AgentRole.SPINE_SURGEON),
            ask(AgentRole.NEUROLOGIST),
            ask(AgentRole.ANESTHESIOLOGIST),
            ask(AgentRole.PERIOPERATIVE_PHYSICIAN),
            finalize(),
        ],
        overrides={
            FinalPlan: [final_plan(), final_plan()],
            FinalPlanReview: review,
        },
    )
    async with open_sqlite_checkpointer(tmp_path / "one-revision.sqlite3") as saver:
        graph = create_complete_mdt_graph(agent_registry=registry, checkpointer=saver)
        result = await graph.ainvoke(
            opening_input(local_case(tmp_path)),
            config=thread_config("thread-one-revision"),
        )

    assert result["run_metadata"].final_review_outcome == "revised"
    assert sum(call["output_model"] is FinalPlan for call in client.calls) == 2
    assert sum(call["output_model"] is FinalPlanReview for call in client.calls) == 1


@pytest.mark.asyncio
async def test_limited_imaging_publishes_partial_field_level_plan(tmp_path: Path) -> None:
    limited_report = radiology_report().model_copy(
        update={"image_quality": ImageQuality.LIMITED}
    )
    registry, client = registry_with_client(
        moderator_responses=[
            ask(AgentRole.SPINE_SURGEON),
            ask(AgentRole.NEUROLOGIST),
            ask(AgentRole.ANESTHESIOLOGIST),
            ask(AgentRole.PERIOPERATIVE_PHYSICIAN),
            finalize(),
        ],
        overrides={
            type(limited_report): limited_report,
            FinalPlan: [final_plan(), final_plan()],
        },
    )
    async with open_sqlite_checkpointer(tmp_path / "abstention.sqlite3") as saver:
        graph = create_complete_mdt_graph(agent_registry=registry, checkpointer=saver)
        result = await graph.ainvoke(
            opening_input(local_case(tmp_path)),
            config=thread_config("thread-abstention"),
        )

    surgery = result["final_plan"].recommended_surgery
    assert result["run_metadata"].final_review_outcome == "passed"
    # v5: limited imaging withholds levels and fusion/fixation but keeps the
    # concrete procedure, approach, and decompression as established fields.
    assert result["final_plan"].definitive_surgery_status.value == "partial"
    assert surgery.procedure_name == "L4-L5 decompression and discectomy"
    assert surgery.procedure_status.value == "established"
    assert surgery.approach == "posterior"
    assert surgery.approach_status.value == "established"
    assert surgery.decompression_status.value == "established"
    assert surgery.levels == ["L4-L5"]
    assert surgery.levels_status.value == "conditional"
    assert surgery.fusion_fixation_status.value == "conditional"
    assert result["final_plan"].conditional_surgical_options == []
    assert sum(call["output_model"] is FinalPlan for call in client.calls) == 1
