"""Deterministic integrity gates around model-generated final plans."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from exceptions import StructuredOutputError
from schemas import (
    AlternativePlan,
    ClaimEvidence,
    DefinitiveSurgeryStatus,
    FinalPlan,
    FinalPlanReview,
    ImageQuality,
    MISSING_RECORDS,
    PlanningCertainty,
    RadiologyReport,
    SpineSurgeonOutput,
    SurgeryFieldStatus,
    SurgicalOption,
    SurgicalOptionUncertainty,
    OperativePlanDraft,
)
from services.plan_convergence import (
    CRITICAL_OPERATIVE_FIELDS,
    OPERATIVE_FIELD_ATTRIBUTES,
    OPERATIVE_STATUS_ATTRIBUTES,
    derive_definitive_status,
    field_oppositions,
)
from termination import merge_unique_strings


CRITICAL_OPERATIVE_PATHS = frozenset(CRITICAL_OPERATIVE_FIELDS)

_IMAGING_GATED_FIELDS = {
    "/recommended_surgery/levels",
    "/recommended_surgery/fusion_or_fixation",
}

_COMPLETED_FACT_PATTERN = re.compile(
    r"(?:"
    r"\b(?:has|have|had|is|are|was|were)\s+(?:already\s+)?(?:been\s+)?"
    r"(?:completed|cleared|optimized|resolved|excluded)\b|"
    r"\bcleared\s+for\b|"
    r"\bmedically\s+optimized\b|"
    r"\bcompleted\s+(?:airway|cardiac|cardiopulmonary|medical|preoperative)\b|"
    r"(?:已完成|已经完成|已评估|已经评估|已优化|已经优化|已清除|已经清除|"
    r"已排除|已经排除|检查完成|评估完成)"
    r")",
    re.IGNORECASE,
)


def validate_final_plan_evidence(
    output: BaseModel,
    *,
    allowed_ids: set[str],
) -> BaseModel:
    plan = FinalPlan.model_validate(output)
    unknown = {
        evidence_id
        for claim in plan.claim_evidence
        for evidence_id in claim.evidence_ids
        if evidence_id not in allowed_ids
    }
    if unknown:
        raise StructuredOutputError("Final plan referenced an unknown evidence id.")
    direct_ids = {
        evidence_id
        for option in plan.conditional_surgical_options
        for evidence_id in option.evidence_ids
    }
    if plan.approach_comparison is not None:
        direct_ids.update(
            evidence_id
            for assessment in plan.approach_comparison.assessments
            for evidence_id in assessment.evidence_ids
        )
    if direct_ids - allowed_ids:
        raise StructuredOutputError(
            "A surgical option or approach assessment referenced an unknown evidence id."
        )
    supported_paths = {
        claim.field_path
        for claim in plan.claim_evidence
        if claim.evidence_ids and set(claim.evidence_ids) <= allowed_ids
    }
    required = _non_missing_operative_paths(plan)
    if required - supported_paths:
        raise StructuredOutputError(
            "Final plan omitted source support for a critical operative field."
        )
    documented_paths = {
        claim.field_path
        for claim in plan.claim_evidence
        if claim.support_type == "documented"
        and claim.evidence_ids
        and set(claim.evidence_ids) <= allowed_ids
    }
    unsupported_completed_facts = {
        path
        for path, value in _string_values(plan.model_dump(mode="json"))
        if _COMPLETED_FACT_PATTERN.search(value)
        and not any(_path_supports(source_path, path) for source_path in documented_paths)
    }
    if unsupported_completed_facts:
        raise StructuredOutputError(
            "Final plan asserted a completed assessment or optimization without "
            "documented source support."
        )
    return plan


def normalize_final_plan_for_evidence(
    output: BaseModel,
    *,
    allowed_ids: set[str],
    report: RadiologyReport,
    surgeon_output: SpineSurgeonOutput | None = None,
) -> FinalPlan:
    """Apply deterministic fail-closed defaults before strict validation."""

    plan = FinalPlan.model_validate(output)
    _validate_direct_planning_evidence(plan, allowed_ids=allowed_ids)
    claims = []
    for claim in plan.claim_evidence:
        evidence_ids = [
            evidence_id
            for evidence_id in claim.evidence_ids
            if evidence_id in allowed_ids
        ]
        if claim.support_type in {"documented", "inference"} and not evidence_ids:
            continue
        claims.append(claim.model_copy(update={"evidence_ids": evidence_ids}))
    plan = plan.model_copy(update={"claim_evidence": claims})
    plan = _apply_surgeon_owned_options(
        plan,
        report=report,
        surgeon_output=surgeon_output,
    )

    documented_paths = {
        claim.field_path
        for claim in plan.claim_evidence
        if claim.support_type == "documented" and claim.evidence_ids
    }
    unsupported_paths = {
        path
        for path, value in _string_values(plan.model_dump(mode="json"))
        if _COMPLETED_FACT_PATTERN.search(value)
        and not any(_path_supports(source_path, path) for source_path in documented_paths)
    }
    if unsupported_paths:
        payload = plan.model_dump(mode="json")
        for path in unsupported_paths:
            _replace_string_at_path(payload, path, MISSING_RECORDS)
        plan = FinalPlan.model_validate(payload)
    return plan


def validate_review_evidence(
    output: BaseModel,
    *,
    allowed_ids: set[str],
) -> BaseModel:
    review = FinalPlanReview.model_validate(output)
    unknown = {
        evidence_id
        for issue in review.issues
        for evidence_id in issue.evidence_ids
        if evidence_id not in allowed_ids
    }
    if unknown:
        raise StructuredOutputError("Final review referenced an unknown evidence id.")
    return review


def _preferred_draft_option(
    working_plan: OperativePlanDraft | None,
) -> SurgicalOption | None:
    if working_plan is None or not working_plan.surgical_indication:
        return None
    options = sorted(working_plan.options, key=lambda item: item.rank)
    if not options:
        return None
    preferred = working_plan.preferred_option_id
    if preferred is not None:
        match = next(
            (option for option in options if option.option_id == preferred),
            None,
        )
        if match is not None:
            return match
    return options[0]


def _field_present(value) -> bool:
    if isinstance(value, list):
        return bool(value)
    return bool(value) and value != MISSING_RECORDS


def _field_status(
    *,
    option: SurgicalOption | None,
    field: str,
    value,
    report: RadiologyReport,
    review: FinalPlanReview | None,
    working_plan: OperativePlanDraft | None,
) -> tuple[SurgeryFieldStatus, str]:
    """Decide one operative field's status. Never grants ESTABLISHED from
    positions or a forced run; evidence, imaging, and review gates all apply."""

    if option is None or not _field_present(value):
        return (
            SurgeryFieldStatus.WITHHELD,
            "No supported spine-surgeon operative option was available.",
        )
    quality = report.image_quality
    if quality in {ImageQuality.NONDIAGNOSTIC, ImageQuality.UNKNOWN}:
        return (
            SurgeryFieldStatus.CONDITIONAL,
            f"Radiology quality is {quality.value}; the field is kept conditional only.",
        )
    if field in _IMAGING_GATED_FIELDS and quality is not ImageQuality.DIAGNOSTIC:
        return (
            SurgeryFieldStatus.CONDITIONAL,
            "Diagnostic imaging is required before levels or fusion/fixation "
            "can be established.",
        )
    if review is None:
        return (
            SurgeryFieldStatus.CONDITIONAL,
            "Independent review has not yet approved this field.",
        )
    if field not in set(review.established_field_paths):
        return (
            SurgeryFieldStatus.CONDITIONAL,
            "The independent review did not approve this field as established.",
        )
    if field_oppositions(working_plan, option.option_id, field):
        return (
            SurgeryFieldStatus.CONDITIONAL,
            "A specialist opposed this field; the value is kept conditional with "
            "the opposition preserved.",
        )
    return SurgeryFieldStatus.ESTABLISHED, ""


def apply_field_level_status(
    plan: FinalPlan,
    *,
    review: FinalPlanReview | None,
    report: RadiologyReport,
    working_plan: OperativePlanDraft | None,
) -> FinalPlan:
    """Assign per-field statuses deterministically and project the preferred
    option into recommended_surgery. Withheld fields are blanked; established
    and conditional fields carry the surgeon's concrete values."""

    option = _preferred_draft_option(working_plan)
    if option is None or not plan.surgical_indication.surgery_recommended:
        reason = (
            "The final synthesis did not establish a surgical indication."
            if not plan.surgical_indication.surgery_recommended
            else "No supported spine-surgeon operative option is available for "
            "field-level publication."
        )
        return _apply_exact_surgery_abstention(plan, reason)
    options = sorted(working_plan.options, key=lambda item: item.rank)
    statuses: dict[str, SurgeryFieldStatus] = {}
    reasons: dict[str, str] = {}
    kept_paths: list[str] = []
    for field in CRITICAL_OPERATIVE_FIELDS:
        attribute = OPERATIVE_FIELD_ATTRIBUTES[field]
        status, reason = _field_status(
            option=option,
            field=field,
            value=getattr(option, attribute),
            report=report,
            review=review,
            working_plan=working_plan,
        )
        statuses[attribute] = status
        reasons[attribute] = reason
        if status in {
            SurgeryFieldStatus.ESTABLISHED,
            SurgeryFieldStatus.CONDITIONAL,
        }:
            kept_paths.append(field)

    existing = plan.recommended_surgery
    projected: dict[str, object] = {}
    for field in CRITICAL_OPERATIVE_FIELDS:
        attribute = OPERATIVE_FIELD_ATTRIBUTES[field]
        value = getattr(option, attribute)
        if statuses[attribute] is SurgeryFieldStatus.WITHHELD:
            projected[attribute] = _blank_field_value(attribute)
        elif attribute == "approach":
            projected[attribute] = value.value
        else:
            projected[attribute] = value
    recommended = existing.model_copy(
        update={
            **projected,
            **{
                status_attribute: statuses[attribute]
                for attribute, status_attribute in OPERATIVE_STATUS_ATTRIBUTES.items()
            },
        }
    )
    any_kept = bool(kept_paths)
    conditional = options[1:] if any_kept else options
    claims = [
        claim
        for claim in plan.claim_evidence
        if claim.field_path not in CRITICAL_OPERATIVE_PATHS
    ]
    claims.extend(
        _operative_projection_claim(path, recommended, option) for path in kept_paths
    )
    withheld_reasons = [
        reasons[attribute]
        for attribute, status in statuses.items()
        if status is SurgeryFieldStatus.WITHHELD and reasons[attribute]
    ]
    consensus = _project_consensus(
        plan.final_consensus,
        definitive=derive_definitive_status(recommended),
        statuses=statuses,
        withheld_reasons=withheld_reasons,
    )
    return plan.model_copy(
        update={
            "recommended_surgery": recommended,
            "definitive_surgery_status": derive_definitive_status(recommended),
            "definitive_surgery_withheld_reasons": withheld_reasons,
            "conditional_surgical_options": conditional,
            "approach_comparison": (
                working_plan.approach_comparison
                if working_plan is not None
                else None
            ),
            "alternatives": _project_legacy_alternatives(
                plan.alternatives,
                conditional,
            ),
            "final_consensus": consensus,
            "claim_evidence": claims,
        }
    )


def _project_consensus(
    existing,
    *,
    definitive: DefinitiveSurgeryStatus,
    statuses: dict[str, SurgeryFieldStatus],
    withheld_reasons: Sequence[str],
):
    conditional_fields = [
        attribute
        for attribute, status in statuses.items()
        if status is SurgeryFieldStatus.CONDITIONAL
    ]
    if definitive is DefinitiveSurgeryStatus.ESTABLISHED:
        recommendation = (
            "A definitive operative plan is published with all critical "
            "operative fields established."
        )
    elif definitive is DefinitiveSurgeryStatus.PARTIAL:
        recommendation = (
            "A partially established operative plan is published. Established "
            "fields are definitive; conditional fields ("
            + ", ".join(conditional_fields)
            + ") require the listed confirmations; withheld fields are blanked."
        )
    else:
        recommendation = (
            "No definitive operation is published. Review the conditional surgical "
            "options and their required confirmations."
        )
    return existing.model_copy(
        update={
            "recommendation": recommendation,
            "implementation_prerequisites": merge_unique_strings(
                existing.implementation_prerequisites,
                (
                    ["Complete the listed confirmations before finalizing "
                     "conditional fields."]
                    if conditional_fields
                    else []
                ),
            ),
            "remaining_concerns": merge_unique_strings(
                existing.remaining_concerns,
                list(withheld_reasons),
            ),
        }
    )


def _blank_field_value(attribute: str):
    if attribute == "levels":
        return []
    return MISSING_RECORDS


def _apply_exact_surgery_abstention(plan: FinalPlan, reason: str) -> FinalPlan:
    alternatives = _project_legacy_alternatives(
        plan.alternatives,
        plan.conditional_surgical_options,
    )
    recommended = plan.recommended_surgery.model_copy(
        update={
            "procedure_name": MISSING_RECORDS,
            "levels": [],
            "approach": MISSING_RECORDS,
            "decompression_extent": MISSING_RECORDS,
            "fusion_or_fixation": MISSING_RECORDS,
            "procedure_status": SurgeryFieldStatus.WITHHELD,
            "levels_status": SurgeryFieldStatus.WITHHELD,
            "approach_status": SurgeryFieldStatus.WITHHELD,
            "decompression_status": SurgeryFieldStatus.WITHHELD,
            "fusion_fixation_status": SurgeryFieldStatus.WITHHELD,
        }
    )
    consensus = plan.final_consensus.model_copy(
        update={
            "recommendation": (
                "No definitive operation is published. Review the conditional surgical "
                "options and their required confirmations."
            ),
            "implementation_prerequisites": merge_unique_strings(
                plan.final_consensus.implementation_prerequisites,
                ["Complete the listed confirmations before selecting a definitive operation."],
            ),
            "remaining_concerns": merge_unique_strings(
                plan.final_consensus.remaining_concerns,
                [reason],
            ),
        }
    )
    evidence = [
        item
        for item in plan.claim_evidence
        if item.field_path not in CRITICAL_OPERATIVE_PATHS
    ]
    return plan.model_copy(
        update={
            "recommended_surgery": recommended,
            "alternatives": alternatives,
            "definitive_surgery_status": DefinitiveSurgeryStatus.WITHHELD,
            "definitive_surgery_withheld_reasons": merge_unique_strings(
                plan.definitive_surgery_withheld_reasons,
                [reason],
            ),
            "final_consensus": consensus,
            "claim_evidence": evidence,
        }
    )


def _apply_surgeon_owned_options(
    plan: FinalPlan,
    *,
    report: RadiologyReport,
    surgeon_output: SpineSurgeonOutput | None,
) -> FinalPlan:
    """Use the surgeon contribution as the only source of concrete operations.

    All options are projected as conditional candidates here; the final
    field-level split between recommended_surgery and conditional options is
    assigned later by apply_field_level_status."""

    if surgeon_output is None:
        return _without_surgical_options(
            plan,
            "No structured spine-surgeon contribution is available.",
        )
    if not surgeon_output.surgical_indication:
        return _without_surgical_options(
            plan,
            "The Spine Surgeon did not identify a surgical indication.",
        )
    if not plan.surgical_indication.surgery_recommended:
        return _without_surgical_options(
            plan,
            "The final synthesis did not establish a surgical indication.",
        )

    options = [
        _apply_image_quality_to_option(option, report.image_quality)
        for option in sorted(surgeon_output.operative_options, key=lambda item: item.rank)
    ][:3]
    if not options:
        return _without_surgical_options(
            plan,
            "The Spine Surgeon did not submit a valid structured surgical option.",
        )
    return plan.model_copy(
        update={
            "conditional_surgical_options": options,
            "approach_comparison": surgeon_output.approach_comparison,
            "alternatives": _project_legacy_alternatives(
                plan.alternatives,
                options,
            ),
        }
    )


def _without_surgical_options(plan: FinalPlan, reason: str) -> FinalPlan:
    plan = plan.model_copy(
        update={
            "conditional_surgical_options": [],
            "approach_comparison": None,
            "alternatives": _project_legacy_alternatives(plan.alternatives, []),
        }
    )
    return _apply_exact_surgery_abstention(plan, reason)


def _apply_image_quality_to_option(
    option: SurgicalOption,
    quality: ImageQuality,
) -> SurgicalOption:
    updates = {}
    if quality is ImageQuality.LIMITED:
        if option.uncertainty is SurgicalOptionUncertainty.MODERATE:
            updates["uncertainty"] = SurgicalOptionUncertainty.HIGH
    elif quality in {ImageQuality.NONDIAGNOSTIC, ImageQuality.UNKNOWN}:
        updates.update(
            {
                "uncertainty": SurgicalOptionUncertainty.VERY_HIGH,
                "confidence": min(option.confidence, 0.25),
                "levels_certainty": (
                    PlanningCertainty.PROVISIONAL
                    if option.levels
                    else PlanningCertainty.NOT_ESTABLISHED
                ),
                "fixation_certainty": (
                    PlanningCertainty.PROVISIONAL
                    if option.fixation_certainty is PlanningCertainty.ESTABLISHED
                    else option.fixation_certainty
                ),
                "required_confirmation": merge_unique_strings(
                    option.required_confirmation,
                    ["Obtain diagnostic imaging before confirming levels and fixation."],
                ),
            }
        )
    return option.model_copy(update=updates)


def _project_legacy_alternatives(
    existing: AlternativePlan,
    options: Sequence[SurgicalOption],
) -> AlternativePlan:
    return existing.model_copy(
        update={
            "alternative_surgery": [_legacy_option_summary(option) for option in options],
            "applicability_conditions": [
                condition for option in options for condition in option.selection_conditions
            ],
            "key_differences": [
                reason for option in options for reason in option.reasons_against
            ],
        }
    )


def _legacy_option_summary(option: SurgicalOption) -> str:
    levels = ", ".join(option.levels) or "levels not established"
    return (
        f"#{option.rank} {option.procedure_name} | {option.approach.value} | {levels} "
        f"({option.levels_certainty.value}; uncertainty={option.uncertainty.value})"
    )


def _validate_direct_planning_evidence(
    plan: FinalPlan,
    *,
    allowed_ids: set[str],
) -> None:
    direct_ids = {
        evidence_id
        for option in plan.conditional_surgical_options
        for evidence_id in option.evidence_ids
    }
    if plan.approach_comparison is not None:
        direct_ids.update(
            evidence_id
            for assessment in plan.approach_comparison.assessments
            for evidence_id in assessment.evidence_ids
        )
    if direct_ids - allowed_ids:
        raise StructuredOutputError(
            "A surgical option or approach assessment referenced an unknown evidence id."
        )


def _operative_projection_claim(path, surgery, option):
    values = {
        "/recommended_surgery/procedure_name": surgery.procedure_name,
        "/recommended_surgery/levels": ", ".join(surgery.levels),
        "/recommended_surgery/approach": surgery.approach,
        "/recommended_surgery/decompression_extent": surgery.decompression_extent,
        "/recommended_surgery/fusion_or_fixation": surgery.fusion_or_fixation,
    }
    return ClaimEvidence(
        field_path=path,
        claim=values[path],
        support_type="proposal",
        evidence_ids=option.evidence_ids,
    )


def _non_missing_operative_paths(plan: FinalPlan) -> set[str]:
    surgery = plan.recommended_surgery
    values: dict[str, str | Sequence[str]] = {
        "/recommended_surgery/procedure_name": surgery.procedure_name,
        "/recommended_surgery/levels": surgery.levels,
        "/recommended_surgery/approach": surgery.approach,
        "/recommended_surgery/decompression_extent": surgery.decompression_extent,
        "/recommended_surgery/fusion_or_fixation": surgery.fusion_or_fixation,
    }
    return {
        path
        for path, value in values.items()
        if value and value != MISSING_RECORDS
    }


def _string_values(value, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _string_values(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _string_values(item, f"{path}/{index}")
    elif isinstance(value, str):
        yield path, value


def _path_supports(source_path: str, value_path: str) -> bool:
    return value_path == source_path or value_path.startswith(f"{source_path}/")


def _replace_string_at_path(payload: dict, path: str, replacement: str) -> None:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return
    current = payload
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    final = parts[-1]
    if isinstance(current, list):
        current[int(final)] = replacement
    else:
        current[final] = replacement
