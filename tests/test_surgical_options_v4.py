import pytest

from exceptions import StructuredOutputError
from schemas import (
    DefinitiveSurgeryStatus,
    DynamicStability,
    FinalPlan,
    FinalPlanReview,
    ImageQuality,
    MISSING_RECORDS,
    PlanningCertainty,
    SurgicalApproach,
    SurgeryFieldStatus,
)
from services.context_policy import (
    normalize_radiology_planning,
    normalize_spine_surgeon_output,
)
from services.final_plan_review import (
    apply_field_level_status,
    normalize_final_plan_for_evidence,
    validate_final_plan_evidence,
)
from services.plan_convergence import (
    CRITICAL_OPERATIVE_FIELDS,
    project_plan_draft,
)
from services.radiology_markdown import render_radiology_markdown
from tests.factories import final_plan, radiology_report, spine_output, surgical_option


ALLOWED_IDS = {"case-page-001", "test-page-001", "mri-1"}


def approving_review() -> FinalPlanReview:
    return FinalPlanReview(
        verdict="pass",
        definitive_surgery_allowed=True,
        established_field_paths=list(CRITICAL_OPERATIVE_FIELDS),
    )


def test_completed_pre_v5_plan_infers_definitive_status_when_read() -> None:
    payload = final_plan().model_dump(
        exclude={
            "definitive_surgery_status": True,
            "definitive_surgery_withheld_reasons": True,
            "conditional_surgical_options": True,
            "approach_comparison": True,
            "recommended_surgery": {
                "procedure_status": True,
                "levels_status": True,
                "approach_status": True,
                "decompression_status": True,
                "fusion_fixation_status": True,
            },
        }
    )

    restored = FinalPlan.model_validate(payload)

    assert restored.definitive_surgery_status is DefinitiveSurgeryStatus.ESTABLISHED
    assert restored.recommended_surgery.procedure_status is SurgeryFieldStatus.ESTABLISHED
    assert restored.recommended_surgery.levels_status is SurgeryFieldStatus.ESTABLISHED


def test_diagnostic_imaging_can_publish_definitive_option() -> None:
    plan = normalize_final_plan_for_evidence(
        final_plan(),
        allowed_ids=ALLOWED_IDS,
        report=radiology_report(),
        surgeon_output=spine_output(),
    )
    result = apply_field_level_status(
        plan,
        review=approving_review(),
        report=radiology_report(),
        working_plan=project_plan_draft(spine_output(), source_sequence=1),
    )

    assert result.definitive_surgery_status is DefinitiveSurgeryStatus.ESTABLISHED
    assert result.recommended_surgery.procedure_name == (
        "L4-L5 decompression and discectomy"
    )
    assert result.recommended_surgery.procedure_status is SurgeryFieldStatus.ESTABLISHED
    assert result.recommended_surgery.levels_status is SurgeryFieldStatus.ESTABLISHED
    assert result.recommended_surgery.approach_status is SurgeryFieldStatus.ESTABLISHED
    assert (
        result.recommended_surgery.decompression_status is SurgeryFieldStatus.ESTABLISHED
    )
    assert (
        result.recommended_surgery.fusion_fixation_status
        is SurgeryFieldStatus.ESTABLISHED
    )
    assert result.conditional_surgical_options == []
    assert result.approach_comparison is not None
    assert {item.approach for item in result.approach_comparison.assessments} == set(
        SurgicalApproach
    )


def test_limited_imaging_keeps_concrete_fields_but_withholds_levels_and_fixation() -> None:
    report = radiology_report().model_copy(
        update={"image_quality": ImageQuality.LIMITED}
    )
    plan = normalize_final_plan_for_evidence(
        final_plan(),
        allowed_ids=ALLOWED_IDS,
        report=report,
        surgeon_output=spine_output(),
    )
    result = apply_field_level_status(
        plan,
        review=approving_review(),
        report=report,
        working_plan=project_plan_draft(spine_output(), source_sequence=1),
    )

    assert result.definitive_surgery_status is DefinitiveSurgeryStatus.PARTIAL
    assert result.recommended_surgery.procedure_name == (
        "L4-L5 decompression and discectomy"
    )
    assert result.recommended_surgery.procedure_status is SurgeryFieldStatus.ESTABLISHED
    assert result.recommended_surgery.approach_status is SurgeryFieldStatus.ESTABLISHED
    assert (
        result.recommended_surgery.decompression_status is SurgeryFieldStatus.ESTABLISHED
    )
    assert result.recommended_surgery.levels_status is SurgeryFieldStatus.CONDITIONAL
    assert (
        result.recommended_surgery.fusion_fixation_status
        is SurgeryFieldStatus.CONDITIONAL
    )
    assert result.conditional_surgical_options == []
    assert result.recommended_surgery.levels == ["L4-L5"]


def test_review_disapproval_downgrades_fields_without_erasing_content() -> None:
    report = radiology_report()
    plan = normalize_final_plan_for_evidence(
        final_plan(),
        allowed_ids=ALLOWED_IDS,
        report=report,
        surgeon_output=spine_output(),
    )
    partial_review = approving_review().model_copy(
        update={
            "established_field_paths": [
                "/recommended_surgery/procedure_name",
                "/recommended_surgery/approach",
            ]
        }
    )
    result = apply_field_level_status(
        plan,
        review=partial_review,
        report=report,
        working_plan=project_plan_draft(spine_output(), source_sequence=1),
    )

    assert result.definitive_surgery_status is DefinitiveSurgeryStatus.PARTIAL
    assert result.recommended_surgery.procedure_status is SurgeryFieldStatus.ESTABLISHED
    assert result.recommended_surgery.approach_status is SurgeryFieldStatus.ESTABLISHED
    assert result.recommended_surgery.levels_status is SurgeryFieldStatus.CONDITIONAL
    assert result.recommended_surgery.levels == ["L4-L5"]


@pytest.mark.parametrize("quality", [ImageQuality.NONDIAGNOSTIC, ImageQuality.UNKNOWN])
def test_nondiagnostic_or_unknown_keeps_concrete_option_with_cap(
    quality: ImageQuality,
) -> None:
    report = radiology_report().model_copy(update={"image_quality": quality})
    plan = normalize_final_plan_for_evidence(
        final_plan(),
        allowed_ids=ALLOWED_IDS,
        report=report,
        surgeon_output=spine_output(),
    )
    result = apply_field_level_status(
        plan,
        review=approving_review(),
        report=report,
        working_plan=project_plan_draft(spine_output(), source_sequence=1),
    )

    surgery = result.recommended_surgery
    assert result.definitive_surgery_status is DefinitiveSurgeryStatus.WITHHELD
    assert surgery.procedure_name == "L4-L5 decompression and discectomy"
    assert surgery.procedure_status is SurgeryFieldStatus.CONDITIONAL
    assert surgery.approach == SurgicalApproach.POSTERIOR.value
    assert surgery.levels_status is SurgeryFieldStatus.CONDITIONAL
    assert surgery.fusion_fixation_status is SurgeryFieldStatus.CONDITIONAL
    assert result.conditional_surgical_options == []


def test_limited_c3_c6_open_door_candidate_is_not_erased() -> None:
    cervical_option = surgical_option().model_copy(
        update={
            "procedure_name": "后路 C3–6 单侧开门椎管扩大成形术",
            "levels": ["C3", "C4", "C5", "C6"],
            "decompression_extent": "C3–6 椎管扩大与脊髓减压",
            "fusion_or_fixation": "条件性植骨内固定，取决于稳定性与曲度复核",
            "fixation_certainty": PlanningCertainty.PROVISIONAL,
            "required_confirmation": ["复核曲度、动态稳定性及固定必要性"],
        }
    )
    surgeon = spine_output().model_copy(
        update={
            "operative_options": [cervical_option],
            "recommended_procedure": cervical_option.procedure_name,
            "surgical_levels": cervical_option.levels,
        }
    )
    report = radiology_report().model_copy(
        update={"image_quality": ImageQuality.LIMITED}
    )
    plan = normalize_final_plan_for_evidence(
        final_plan(),
        allowed_ids=ALLOWED_IDS,
        report=report,
        surgeon_output=surgeon,
    )
    result = apply_field_level_status(
        plan,
        review=approving_review(),
        report=report,
        working_plan=project_plan_draft(surgeon, source_sequence=1),
    )

    assert result.recommended_surgery.procedure_name == "后路 C3–6 单侧开门椎管扩大成形术"
    assert result.recommended_surgery.approach == SurgicalApproach.POSTERIOR.value
    assert result.recommended_surgery.levels == ["C3", "C4", "C5", "C6"]
    assert result.recommended_surgery.levels_status is SurgeryFieldStatus.CONDITIONAL
    assert (
        result.recommended_surgery.fusion_fixation_status
        is SurgeryFieldStatus.CONDITIONAL
    )
    assert result.definitive_surgery_status is DefinitiveSurgeryStatus.PARTIAL


def test_no_indication_or_missing_surgeon_does_not_create_option() -> None:
    no_indication = final_plan().model_copy(
        update={
            "surgical_indication": final_plan().surgical_indication.model_copy(
                update={"surgery_recommended": False}
            )
        }
    )
    without_indication = normalize_final_plan_for_evidence(
        no_indication,
        allowed_ids=ALLOWED_IDS,
        report=radiology_report(),
        surgeon_output=spine_output(),
    )
    without_surgeon = normalize_final_plan_for_evidence(
        final_plan(),
        allowed_ids=ALLOWED_IDS,
        report=radiology_report(),
        surgeon_output=None,
    )

    assert without_indication.conditional_surgical_options == []
    assert without_surgeon.conditional_surgical_options == []
    assert without_surgeon.approach_comparison is None
    assert without_surgeon.recommended_surgery.procedure_name == MISSING_RECORDS


def test_illegal_candidate_evidence_is_rejected_before_projection() -> None:
    invalid_option = surgical_option().model_copy(
        update={"evidence_ids": ["invented-image-id"]}
    )
    invalid = final_plan().model_copy(
        update={"conditional_surgical_options": [invalid_option]}
    )

    with pytest.raises(StructuredOutputError, match="unknown evidence id"):
        normalize_final_plan_for_evidence(
            invalid,
            allowed_ids=ALLOWED_IDS,
            report=radiology_report(),
            surgeon_output=spine_output(),
        )


def test_surgeon_indication_requires_structured_options_and_comparison() -> None:
    incomplete = spine_output().model_copy(
        update={
            "operative_options": [],
            "preferred_option_id": None,
            "approach_comparison": None,
        }
    )

    with pytest.raises(StructuredOutputError, match="one to three"):
        normalize_spine_surgeon_output(incomplete, allowed_ids=ALLOWED_IDS)


def test_preferred_option_and_approach_comparison_must_agree() -> None:
    surgeon = spine_output()
    conflicting = surgeon.model_copy(
        update={
            "approach_comparison": surgeon.approach_comparison.model_copy(
                update={"preferred_approach": SurgicalApproach.ANTERIOR}
            )
        }
    )

    with pytest.raises(StructuredOutputError, match="preferred approach"):
        normalize_spine_surgeon_output(conflicting, allowed_ids=ALLOWED_IDS)


def test_static_images_force_dynamic_stability_to_not_assessed() -> None:
    report = radiology_report().model_copy(
        update={
            "operative_planning": radiology_report().operative_planning.model_copy(
                update={"dynamic_stability": DynamicStability.STABLE}
            )
        }
    )
    normalized = normalize_radiology_planning(
        report,
        images=[],
        allowed_ids=ALLOWED_IDS,
    )

    assert normalized.operative_planning.dynamic_stability is DynamicStability.NOT_ASSESSED
    validate_final_plan_evidence(
        normalize_final_plan_for_evidence(
            final_plan(),
            allowed_ids=ALLOWED_IDS,
            report=normalized,
            surgeon_output=spine_output(),
        ),
        allowed_ids=ALLOWED_IDS,
    )


def test_radiology_markdown_deterministically_renders_planning_factors() -> None:
    report = radiology_report()

    first = render_radiology_markdown(report)
    second = render_radiology_markdown(report)

    assert first == second
    assert "## 手术规划相关影像因素" in first
    assert "- 矢状位曲度：lordotic" in first
    assert "- 动态稳定性：not_assessed" in first
    assert "- 固定相关影像支持：does_not_support" in first
