from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from exceptions import InvalidStateUpdateError
from schemas import (
    AgentRole,
    CaseImage,
    CaseInput,
    ChatMessage,
    DocumentPage,
    FinalPlan,
    ImageQuality,
    Modality,
    RadiologyReport,
    SpineSurgeonOutput,
)
from state import append_unique_messages, create_checkpoint_serializer, create_initial_state


def sample_image() -> CaseImage:
    return CaseImage(
        image_id="mri-1",
        modality=Modality.MRI,
        path_or_url="H:/images/mri-1.jpg",
        body_region="lumbar spine",
        view_or_plane="sagittal",
        sequence=None,
        level=None,
        description=None,
    )


def sample_message(sequence: int = 1, content: str = "Opening") -> ChatMessage:
    return ChatMessage(
        message_id=f"message-{sequence}",
        sequence=sequence,
        speaker_id="moderator",
        speaker_name="MDT Moderator",
        speaker_role=AgentRole.MODERATOR,
        content=content,
        timestamp=datetime.now(UTC),
        turn_number=sequence,
    )


def sample_case(*, images=None) -> CaseInput:
    return CaseInput(
        case_id="case-1",
        case_pages=[
            DocumentPage(
                page_id="case-page-001",
                page_number=1,
                path_or_url="case.jpg",
                original_filename="case.jpg",
            )
        ],
        test_pages=[
            DocumentPage(
                page_id="test-page-001",
                page_number=1,
                path_or_url="tests.jpg",
                original_filename="tests.jpg",
            )
        ],
        images=images or [sample_image()],
        artifact_dir=".data/test-artifacts/case-1",
    )


def test_case_and_unknown_metadata_are_preserved() -> None:
    image = sample_image()
    case = sample_case(images=[image])
    assert case.images[0].sequence is None
    assert case.images[0].level is None


def test_invalid_modality_is_rejected() -> None:
    with pytest.raises(ValidationError, match="modality"):
        CaseImage(
            image_id="bad",
            modality="PET",
            path_or_url="bad.jpg",
            body_region="lumbar spine",
            view_or_plane="unknown",
        )


def test_missing_case_pages_are_rejected() -> None:
    with pytest.raises(ValidationError, match="case_pages"):
        CaseInput.model_validate({"case_id": "case-1", "images": []})


def test_duplicate_image_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="image_id values must be unique"):
        sample_case(images=[sample_image(), sample_image()])


def test_chat_reducer_is_append_only_and_idempotent() -> None:
    first = sample_message()
    second = sample_message(2, "Radiology report")
    assert append_unique_messages([first], [first, second]) == [first, second]


def test_chat_reducer_rejects_sequence_collision() -> None:
    first = sample_message()
    conflicting = sample_message(1, "Different content").model_copy(
        update={"message_id": "different-id"}
    )
    with pytest.raises(InvalidStateUpdateError, match="sequence collision"):
        append_unique_messages([first], [conflicting])


def test_chat_reducer_rejects_out_of_order_append() -> None:
    second = sample_message(2)
    first = sample_message(1)
    with pytest.raises(InvalidStateUpdateError, match="increasing sequence"):
        append_unique_messages([second], [first])


def test_specialist_output_confidence_is_bounded() -> None:
    payload = {
        "public_message": "Surgery is indicated.",
        "surgical_indication": True,
        "indication_rationale": ["Progressive deficit"],
        "functional_impact": "Walking is limited by radicular pain.",
        "conservative_treatment_review": ["Medication did not provide durable relief."],
        "preoperative_prerequisites": ["Confirm the neurologic baseline."],
        "recommended_procedure": "Decompression",
        "surgical_levels": ["L4-L5"],
        "approach": "Posterior",
        "decompression_required": True,
        "fusion_required": False,
        "fixation_required": False,
        "alternative_plans": [],
        "major_surgical_risks": ["Dural tear"],
        "questions_for_other_agents": [],
        "need_retrieval": False,
        "retrieval_query": None,
        "confidence": 0.8,
    }
    assert SpineSurgeonOutput.model_validate(payload).confidence == 0.8
    payload["confidence"] = 1.2
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        SpineSurgeonOutput.model_validate(payload)


def test_radiology_report_requires_fixed_fields() -> None:
    report = RadiologyReport(
        image_quality=ImageQuality.DIAGNOSTIC,
        available_modalities=[Modality.MRI],
        xray_findings=[],
        ct_findings=[],
        mri_findings=[],
        spine_levels=["L4-L5"],
        alignment="Preserved",
        bony_structures="No acute fracture",
        canal_and_foramina="Central stenosis",
        neural_compression="Right L5 root compression",
        cross_modality_synthesis="MRI finding only",
        possible_diagnoses=["L4-L5 disc herniation"],
        uncertain_findings=[],
        missing_views_or_sequences=["No CT"],
        public_summary="MRI shows an L4-L5 disc herniation.",
        confidence=0.85,
    )
    assert report.image_quality is ImageQuality.DIAGNOSTIC


def test_final_plan_missing_sections_is_rejected() -> None:
    with pytest.raises(ValidationError, match="radiology_summary"):
        FinalPlan.model_validate({"confidence": 0.5, "force_finalized": False})


def test_state_round_trip_uses_jsonplus_without_pickle() -> None:
    state = create_initial_state(sample_case())
    message = sample_message()
    state["chat_log"] = [message]
    serializer = create_checkpoint_serializer()
    serialized = serializer.dumps_typed(state)
    restored = serializer.loads_typed(serialized)
    assert restored["case_id"] == "case-1"
    assert restored["images"][0] == sample_image()
    assert restored["chat_log"][0] == message
    assert restored["run_metadata"].started_at.tzinfo is not None
