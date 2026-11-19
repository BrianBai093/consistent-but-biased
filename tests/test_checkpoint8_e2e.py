"""Offline end-to-end coverage for the CP8 synthetic clinical scenario matrix.

The scenarios are test fixtures only. They do not populate the disabled
knowledge base or claim to be clinical recommendations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from graph import create_complete_mdt_graph, opening_input, thread_config
from schemas import (
    AgentRole,
    AnesthesiologistOutput,
    CaseDocumentMarkdown,
    CaseImage,
    ConsistencyAssessment,
    Disagreement,
    ImageQuality,
    Modality,
    NeurologistOutput,
    PerioperativePhysicianOutput,
    RadiologyAddendum,
    RadiologyReport,
    RiskLevel,
    RunPhase,
    TestDocumentMarkdown as LabDocumentMarkdown,
    WorkflowLimits,
)
from services.checkpoint import open_sqlite_checkpointer
from tests.dynamic_factories import finalize, registry_with_client
from tests.factories import (
    anesthesia_output,
    case_input,
    neurology_output,
    perioperative_output,
    radiology_report,
)
from tests.test_dynamic_graph import STANDARD_ROUTE


@dataclass(frozen=True)
class Scenario:
    identifier: str
    case_text: str
    modalities: tuple[Modality, ...]
    overrides: dict[type, object] = field(default_factory=dict)
    moderator_responses: object = field(default_factory=lambda: list(STANDARD_ROUTE))
    limits: WorkflowLimits | None = None


def _image(case_id: str, index: int, modality: Modality) -> CaseImage:
    extension = ".png" if modality is Modality.MRI else ".jpg"
    return CaseImage(
        image_id=f"{modality.value.lower()}-1",
        modality=modality,
        path_or_url=f"/synthetic-e2e/{case_id}-{index}{extension}",
        body_region="spine",
        view_or_plane="sagittal" if modality is Modality.MRI else "axial",
        sequence="T2" if modality is Modality.MRI else None,
        level="L4-L5",
    )


def _case(scenario: Scenario):
    return case_input(
        case_id=scenario.identifier,
        images=[
            _image(scenario.identifier, index, modality)
            for index, modality in enumerate(scenario.modalities, start=1)
        ],
    )


_high_anesthesia = anesthesia_output().model_copy(
    update={
        "anesthesia_risk": RiskLevel.HIGH,
        "airway_risk": "High-risk airway features require formal assessment.",
        "recommend_delay": True,
        "delay_reasons": ["Airway evaluation is incomplete."],
    }
)
_medical_comorbidity = perioperative_output().model_copy(
    update={
        "major_comorbidities": ["Uncontrolled hypertension", "Chronic kidney disease"],
        "cardiopulmonary_risk": RiskLevel.HIGH,
        "hepatic_renal_risk": RiskLevel.HIGH,
        "preoperative_optimization": [
            "Complete blood-pressure optimization.",
            "Obtain renal function review.",
        ],
    }
)
_mismatch_neurology = neurology_output().model_copy(
    update={
        "symptom_imaging_consistency": ConsistencyAssessment.INCONSISTENT,
        "public_message": "Symptoms and the available imaging require reconciliation.",
    }
)
_limited_radiology = radiology_report().model_copy(
    update={
        "image_quality": ImageQuality.NONDIAGNOSTIC,
        "uncertain_findings": ["The supplied image does not support a diagnostic conclusion."],
        "public_summary": "The supplied image is nondiagnostic and requires repeat imaging.",
        "confidence": 0.2,
    }
)
_mismatch_disagreement = Disagreement(
    agents=[AgentRole.SPINE_SURGEON, AgentRole.NEUROLOGIST],
    issue="Symptom-imaging concordance remains unresolved.",
    positions=["Operative correlation is sufficient.", "Correlation remains uncertain."],
)
_mismatch_final = finalize().model_copy(
    update={
        "consensus_reached": False,
        "unresolved_issues": [_mismatch_disagreement.issue],
        "disagreements": [_mismatch_disagreement],
    }
)


SCENARIOS = [
    Scenario(
        "e2e-cervical",
        "Synthetic cervical radicular pain scenario for workflow testing.",
        (Modality.MRI,),
    ),
    Scenario(
        "e2e-lumbar-disc",
        "Synthetic lumbar disc-extrusion scenario for workflow testing.",
        (Modality.MRI,),
    ),
    Scenario(
        "e2e-lumbar-stenosis",
        "Synthetic lumbar canal-stenosis scenario for workflow testing.",
        (Modality.XRAY, Modality.MRI),
    ),
    Scenario(
        "e2e-spondylolisthesis",
        "Synthetic spondylolisthesis scenario for workflow testing.",
        (Modality.XRAY, Modality.CT, Modality.MRI),
    ),
    Scenario(
        "e2e-fracture",
        "Synthetic spinal-fracture scenario for workflow testing.",
        (Modality.CT, Modality.MRI),
    ),
    Scenario(
        "e2e-anesthesia-risk",
        "Synthetic case with a major anesthesia-risk flag for workflow testing.",
        (Modality.MRI,),
        {AnesthesiologistOutput: _high_anesthesia},
        [*STANDARD_ROUTE, finalize()],
    ),
    Scenario(
        "e2e-medical-risk",
        "Synthetic case with major medical comorbidities for workflow testing.",
        (Modality.MRI,),
        {PerioperativePhysicianOutput: _medical_comorbidity},
    ),
    Scenario(
        "e2e-mismatch",
        "Synthetic symptom-imaging mismatch scenario for workflow testing.",
        (Modality.MRI,),
        {NeurologistOutput: _mismatch_neurology},
        [*STANDARD_ROUTE[:-1], _mismatch_final],
    ),
    Scenario(
        "e2e-nondiagnostic",
        "Synthetic nondiagnostic-image scenario for workflow testing.",
        (Modality.MRI,),
        {RadiologyReport: _limited_radiology},
    ),
    Scenario(
        "e2e-forced-termination",
        "Synthetic forced-termination scenario for workflow testing.",
        (Modality.MRI,),
        moderator_responses={"public_message": "invalid"},
        limits=WorkflowLimits(max_no_progress_turns=20),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.identifier)
async def test_scenario_matrix_completes_with_traceable_safe_structure(
    tmp_path: Path,
    scenario: Scenario,
) -> None:
    registry, client = registry_with_client(
        moderator_responses=scenario.moderator_responses,
        overrides=scenario.overrides,
    )
    case = _case(scenario)
    async with open_sqlite_checkpointer(tmp_path / f"{scenario.identifier}.sqlite3") as saver:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=saver,
            limits=scenario.limits,
        )
        result = await graph.ainvoke(
            opening_input(case),
            config=thread_config(f"thread-{scenario.identifier}"),
        )

    assert result["phase"] is RunPhase.COMPLETED
    assert result["final_plan"] is not None
    assert all(count >= 1 for count in result["speaker_counts"].values())
    assert all(count <= 5 for count in result["speaker_counts"].values())
    assert [message.sequence for message in result["chat_log"]] == list(
        range(1, len(result["chat_log"]) + 1)
    )
    assert len({message.message_id for message in result["chat_log"]}) == len(
        result["chat_log"]
    )
    assert result["retrieved_evidence"] == []
    assert result["retrieval_count"] == 0
    assert all(
        not diagnosis.guideline_evidence_ids
        for diagnosis in result["final_plan"].diagnoses
    )
    if scenario.identifier == "e2e-anesthesia-risk":
        assert result["termination_reason"].value == "normal_consensus"
        assert result["speaker_counts"][AgentRole.SPINE_SURGEON.value] == 2

    # Case records are transcribed by deterministic OCR, so they make no model
    # call; only the test report still goes through a model.
    assert registry.case_document.engine.calls == [
        page.path_or_url for page in case.case_pages
    ]
    test_document_calls = [
        call for call in client.calls if call["output_model"] is LabDocumentMarkdown
    ]
    radiology_calls = [
        call for call in client.calls if call["output_model"] is RadiologyReport
    ]
    assert len(test_document_calls) == 1
    assert test_document_calls[0]["images"] == case.test_pages
    assert 1 <= len(radiology_calls) <= 5
    reviewed_images = [
        image
        for call in radiology_calls
        for image in call["images"]
    ]
    assert {image.image_id for image in reviewed_images} == {
        image.image_id for image in case.images
    }
    review_counts = {
        image.image_id: sum(
            reviewed.image_id == image.image_id for reviewed in reviewed_images
        )
        for image in case.images
    }
    assert len(set(review_counts.values())) == 1
    assert all(
        1 <= len(call["images"]) <= 4
        for call in radiology_calls
        if call["images"]
    )
    assert all(
        "cross_batch_synthesis" in call["user_prompt"]
        for call in radiology_calls
        if not call["images"]
    )
    for call in client.calls:
        if call["output_model"] in {
            CaseDocumentMarkdown,
            LabDocumentMarkdown,
            RadiologyAddendum,
            RadiologyReport,
        }:
            continue
        assert call["images"] == []
        assert all(image.path_or_url not in call["user_prompt"] for image in case.images)
        assert all(
            page.path_or_url not in call["user_prompt"]
            for page in [*case.case_pages, *case.test_pages]
        )


@pytest.mark.asyncio
async def test_repeated_case_runs_remain_structurally_valid_and_renderable(
    tmp_path: Path,
) -> None:
    scenario = SCENARIOS[1]
    case = _case(scenario)
    rendered_records: list[str] = []
    for run_number in (1, 2):
        registry, _ = registry_with_client(moderator_responses=STANDARD_ROUTE)
        async with open_sqlite_checkpointer(tmp_path / f"repeat-{run_number}.sqlite3") as saver:
            graph = create_complete_mdt_graph(agent_registry=registry, checkpointer=saver)
            result = await graph.ainvoke(
                opening_input(case),
                config=thread_config(f"thread-repeat-{run_number}"),
            )
        record = {
            "case_id": result["case_id"],
            "phase": result["phase"].value,
            "chat": [message.model_dump(mode="json") for message in result["chat_log"]],
            "final_plan": result["final_plan"].model_dump(mode="json"),
        }
        rendered_records.append(json.dumps(record, indent=2, sort_keys=True))

    assert all('"final_plan"' in record for record in rendered_records)
    assert all('"chat"' in record for record in rendered_records)
    assert all('"force_finalized"' in record for record in rendered_records)
