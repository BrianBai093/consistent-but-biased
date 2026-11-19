"""Offline fixtures for the department workflow.

One shared ``FakeModelClient`` across all seven roles, so a test can assert on
``client.calls`` -- including which system prompt reached which role, which is
how the style injection is verified end to end.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from department.agents import (
    AnesthesiaAgent,
    DepartmentRegistry,
    DirectorAgent,
    NeurologyAgent,
    PhysiatristAgent,
    RadiologyAgent,
    ResidentAgent,
    ViceDirectorAgent,
)
from department.schemas import (
    AnesthesiaAssessment,
    DepartmentBaselinePayload,
    FlagRelevance,
    RedFlag,
    AsaGrade,
    CompressionType,
    ConsensusOutcome,
    ConservativeStance,
    DirectorDecision,
    DirectorRuling,
    ImagingFindings,
    LevelStatus,
    LocalizationMatch,
    NeurologyAssessment,
    RehabAssessment,
    ResidentReport,
    RiskBand,
    SignalChange,
    StenosisGrade,
    SurgicalIndication,
    SymptomImagingCorrelation,
    ViceDirectorOpinion,
)
from schemas import CaseImage, DocumentPage, Modality
from services.llm_client import FakeModelClient


def resident_report(**overrides) -> ResidentReport:
    values = {
        "chief_complaint": "Neck pain with bilateral arm numbness for 20 days.",
        "history": "Symptoms began without a precipitating injury.",
        "physical_exam": "Grade 5 power throughout; brisk biceps reflexes.",
        "key_positive_signs": ["brisk reflexes"],
        "past_history": "No documented comorbidity.",
        "public_message": "Reporting the admission record.",
    }
    values.update(overrides)
    return ResidentReport(**values)


def imaging_findings(**overrides) -> ImagingFindings:
    values = {
        "levels": ["C4-5"],
        "level_status": LevelStatus.REPORTED,
        "compression_type": CompressionType.VENTRAL,
        "stenosis_grade": StenosisGrade.MODERATE,
        "signal_change": SignalChange.EQUIVOCAL,
        "correlation": SymptomImagingCorrelation.MODERATE,
        "public_message": "Ventral compression at C4-5.",
        "confidence": 0.6,
    }
    values.update(overrides)
    return ImagingFindings(**values)


def red_flag(**overrides) -> RedFlag:
    values = {
        "finding": "Absent ankle jerks with preserved knee jerks.",
        "relevance": FlagRelevance.COEXISTING,
        "rationale": "A length-dependent neuropathy beside the myelopathy.",
    }
    values.update(overrides)
    return RedFlag(**values)


def neurology_assessment(**overrides) -> NeurologyAssessment:
    values = {
        "localization": "Mid-cervical cord.",
        "match_level": LocalizationMatch.MEDIUM,
        "red_flags": [red_flag()],
        "public_message": "Findings fit a mid-cervical cord lesion, with caveats.",
    }
    values.update(overrides)
    return NeurologyAssessment(**values)


def anesthesia_assessment(**overrides) -> AnesthesiaAssessment:
    values = {
        "asa_grade": AsaGrade.ASA_II,
        "map_target_mmhg": 80,
        "blood_loss_risk": RiskBand.MODERATE,
        "extubation_difficulty": RiskBand.LOW,
        "prone_position_tolerance": RiskBand.MODERATE,
        "public_message": "ASA II; moderate blood loss risk.",
    }
    values.update(overrides)
    return AnesthesiaAssessment(**values)


def rehab_assessment(**overrides) -> RehabAssessment:
    values = {
        "conservative_success_rate": 65,
        "proposed_duration": "3-6 months",
        "preconditions_for_failure": ["progressive weakness"],
        "stance": ConservativeStance.HOLD,
        "public_message": "A strict conservative trial is warranted first.",
    }
    values.update(overrides)
    return RehabAssessment(**values)


def vice_opinion(**overrides) -> ViceDirectorOpinion:
    values = {
        "assessment": "Single-level compressive myelopathy.",
        "proposed_options": ["C4-5 anterior decompression and fusion"],
        "preferred_option": "C4-5 anterior decompression and fusion",
        "disagreement_points": [],
        "deferred_questions_to_director": [],
        "public_message": "I would favour single-level anterior surgery.",
    }
    values.update(overrides)
    return ViceDirectorOpinion(**values)


def ruling(**overrides) -> DirectorRuling:
    values = {
        "decision": DirectorDecision.OPERATE,
        "surgical_indication": SurgicalIndication.YES,
        "final_decision": "Proceed with single-level anterior surgery.",
        "surgical_plan": "C4-5 anterior cervical decompression and fusion.",
        "delegated_tasks": [],
        "if_delay": False,
        "has_contingency_plan": False,
        "consensus": ConsensusOutcome.REACHED,
        "addressed_roles": [],
        "ordered_tests": [],
        "family_requests": [],
        "public_message": "We operate.",
    }
    values.update(overrides)
    return DirectorRuling(**values)


def undecided(**overrides) -> DirectorRuling:
    values = {
        "decision": DirectorDecision.NOT_YET_DECIDED,
        "surgical_indication": SurgicalIndication.UNCERTAIN,
        "consensus": ConsensusOutcome.UNCLEAR,
        "public_message": "I want to hear anesthesia again before I rule.",
    }
    values.update(overrides)
    return ruling(
        **{
            **values,
            "final_decision": None,
            "surgical_plan": None,
        }
    )


def department_payload(**overrides) -> DepartmentBaselinePayload:
    values = {
        "frozen_at": datetime(2026, 9, 19, 9, 0, tzinfo=UTC),
        "resident": resident_report(),
        "imaging": imaging_findings(),
        "neurology": neurology_assessment(),
        "anesthesia": anesthesia_assessment(),
        "physiatrist": rehab_assessment(),
    }
    values.update(overrides)
    return DepartmentBaselinePayload(**values)


def department_responses(overrides: dict | None = None) -> dict:
    """Responses are keyed by output *type*, so they cannot be passed as kwargs."""

    responses = {
        ResidentReport: resident_report(),
        ImagingFindings: imaging_findings(),
        NeurologyAssessment: neurology_assessment(),
        AnesthesiaAssessment: anesthesia_assessment(),
        RehabAssessment: rehab_assessment(),
        ViceDirectorOpinion: vice_opinion(),
        DirectorRuling: ruling(),
    }
    responses.update(overrides or {})
    return responses


def department_with_client(
    overrides: dict | None = None,
) -> tuple[DepartmentRegistry, FakeModelClient]:
    client = FakeModelClient(department_responses(overrides))
    registry = DepartmentRegistry(
        resident=ResidentAgent(client),
        radiologist=RadiologyAgent(client),
        neurologist=NeurologyAgent(client),
        anesthesiologist=AnesthesiaAgent(client),
        physiatrist=PhysiatristAgent(client),
        vice_director=ViceDirectorAgent(client),
        director=DirectorAgent(client),
    )
    return registry, client


def case_sources(tmp_path: Path) -> dict:
    """Local files, so the baseline store can fingerprint them."""

    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, payload in (
        ("case-page-001", b"case bytes"),
        ("test-page-001", b"test bytes"),
        ("mri-1", b"image bytes"),
    ):
        path = source / f"{name}.bin"
        path.write_bytes(payload)
        written[name] = str(path)
    return {
        "case_pages": [
            DocumentPage(
                page_id="case-page-001",
                page_number=1,
                path_or_url=written["case-page-001"],
                original_filename="case.png",
            )
        ],
        "test_pages": [
            DocumentPage(
                page_id="test-page-001",
                page_number=1,
                path_or_url=written["test-page-001"],
                original_filename="tests.png",
            )
        ],
        "images": [
            CaseImage(
                image_id="mri-1",
                modality=Modality.MRI,
                path_or_url=written["mri-1"],
            )
        ],
    }
