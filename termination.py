"""Termination, progress, confidence, and final-plan integrity policies."""

import hashlib
import json
from typing import Sequence

from exceptions import FinalPlanIntegrityError, ModelFailureCategory
from schemas import (
    FinalPlan,
    ImageQuality,
    MISSING_RECORDS,
    SpecialistOutputPayload,
    SurgeryFieldStatus,
    TerminationReason,
    WorkflowLimits,
)
from services.plan_convergence import (
    OPERATIVE_STATUS_ATTRIBUTES,
    derive_definitive_status,
)
from state import MDTState, SPECIALIST_ROLES


_FINGERPRINT_EXCLUDED_FIELDS = {
    "public_message",
    "public_summary",
    "confidence",
    "questions_for_other_agents",
    "need_retrieval",
    "retrieval_query",
}


def specialist_fingerprint(output: SpecialistOutputPayload) -> str:
    payload = output.model_dump(mode="json")
    semantic_payload = {
        key: value
        for key, value in payload.items()
        if key not in _FINGERPRINT_EXCLUDED_FIELDS
    }
    serialized = json.dumps(
        semantic_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def update_progress(
    *,
    seen_fingerprints: Sequence[str],
    no_progress_streak: int,
    output: SpecialistOutputPayload,
) -> tuple[list[str], int]:
    fingerprint = specialist_fingerprint(output)
    if fingerprint in seen_fingerprints:
        return list(seen_fingerprints), no_progress_streak + 1
    return [*seen_fingerprints, fingerprint], 0


def normal_completion_ready(state: MDTState) -> bool:
    contribution_roles = {item.role for item in state["specialist_contributions"]}
    return (
        state["radiology_report"] is not None
        and all(role in contribution_roles for role in SPECIALIST_ROLES)
        and all(state["speaker_counts"].get(role.value, 0) >= 1 for role in SPECIALIST_ROLES)
        and not state["pending_requests"]
    )


def forced_termination_reason(
    state: MDTState,
    limits: WorkflowLimits,
    *,
    extend_total_limit: bool = False,
    allow_safety_response: bool = False,
) -> TerminationReason | None:
    if state["termination_reason"] is TerminationReason.SPECIALIST_API_ERROR:
        return TerminationReason.SPECIALIST_API_ERROR
    counts = [state["speaker_counts"].get(role.value, 0) for role in SPECIALIST_ROLES]
    if (
        counts
        and not allow_safety_response
        and all(count >= limits.max_turns_per_specialist for count in counts)
    ):
        return TerminationReason.ALL_SPEAKER_LIMITS
    total_limit = (
        limits.max_absolute_specialist_turns
        if extend_total_limit
        else limits.max_total_specialist_turns
    )
    if sum(counts) >= total_limit:
        return TerminationReason.TOTAL_SPEAKER_LIMIT
    if state["workflow_step_count"] >= limits.max_workflow_steps:
        return TerminationReason.GLOBAL_STEP_LIMIT
    coverage_complete = all(
        state["speaker_counts"].get(role.value, 0) >= 1
        for role in SPECIALIST_ROLES
    )
    if coverage_complete and state["no_progress_streak"] >= limits.max_no_progress_turns:
        return TerminationReason.NO_NEW_INFORMATION
    return None


def calculate_final_confidence(
    state: MDTState,
    plan: FinalPlan | None = None,
) -> float:
    covered_roles = {item.role for item in state["specialist_contributions"]}
    coverage_factor = len(covered_roles.intersection(SPECIALIST_ROLES)) / len(
        SPECIALIST_ROLES
    )
    quality = (
        state["radiology_report"].image_quality
        if state["radiology_report"] is not None
        else ImageQuality.UNKNOWN
    )
    image_factor = {
        ImageQuality.DIAGNOSTIC: 1.0,
        ImageQuality.LIMITED: 0.8,
        ImageQuality.UNKNOWN: 0.65,
        ImageQuality.NONDIAGNOSTIC: 0.55,
    }[quality]
    unresolved_factor = max(0.5, 1.0 - 0.05 * len(state["unresolved_issues"]))
    disagreement_factor = max(0.6, 1.0 - 0.1 * len(state["disagreements"]))
    evidence_factor = 1.0 if state["retrieved_evidence"] else 0.9
    field_factor = max(
        0.55,
        _established_field_ratio(plan if plan is not None else state["final_plan"]),
    )
    confidence = (
        coverage_factor
        * image_factor
        * unresolved_factor
        * disagreement_factor
        * evidence_factor
        * field_factor
    )
    return round(min(0.95, max(0.05, confidence)), 2)


def _established_field_ratio(plan: FinalPlan | None) -> float:
    if plan is None:
        return 0.55
    surgery = plan.recommended_surgery
    statuses = [
        getattr(surgery, attribute) for attribute in OPERATIVE_STATUS_ATTRIBUTES.values()
    ]
    if not statuses:
        return 0.55
    established = sum(
        1 for status in statuses if status is SurgeryFieldStatus.ESTABLISHED
    )
    return 0.55 + 0.45 * (established / len(statuses))


def validate_final_plan_integrity(plan: FinalPlan, state: MDTState) -> FinalPlan:
    if plan.force_finalized is not state["force_finalize"]:
        raise FinalPlanIntegrityError(
            ModelFailureCategory.FINAL_PLAN_STATE_MISMATCH
        )
    surgery = plan.recommended_surgery
    if plan.definitive_surgery_status is not derive_definitive_status(surgery):
        raise FinalPlanIntegrityError(
            ModelFailureCategory.FINAL_PLAN_FIELD_STATUS_MISMATCH
        )
    for attribute, status_attribute in OPERATIVE_STATUS_ATTRIBUTES.items():
        value = getattr(surgery, attribute)
        status = getattr(surgery, status_attribute)
        present = bool(value) and value != MISSING_RECORDS
        if status is SurgeryFieldStatus.WITHHELD and present:
            raise FinalPlanIntegrityError(
                ModelFailureCategory.FINAL_PLAN_FIELD_STATUS_MISMATCH
            )
        if status is not SurgeryFieldStatus.WITHHELD and not present:
            raise FinalPlanIntegrityError(
                ModelFailureCategory.FINAL_PLAN_FIELD_STATUS_MISMATCH
            )
    valid_evidence_ids = {item.evidence_id for item in state["retrieved_evidence"]}
    referenced_ids = {
        evidence_id
        for diagnosis in plan.diagnoses
        for evidence_id in diagnosis.guideline_evidence_ids
    }
    if not referenced_ids.issubset(valid_evidence_ids):
        raise FinalPlanIntegrityError(
            ModelFailureCategory.FINAL_PLAN_UNKNOWN_EVIDENCE
        )
    remaining_concerns = set(plan.final_consensus.remaining_concerns)
    missing_issues = [
        issue for issue in state["unresolved_issues"] if issue not in remaining_concerns
    ]
    if missing_issues:
        raise FinalPlanIntegrityError(
            ModelFailureCategory.FINAL_PLAN_OMITTED_ISSUES
        )
    plan_disagreements = {item.issue: item for item in plan.disagreements}
    for disagreement in state["disagreements"]:
        planned = plan_disagreements.get(disagreement.issue)
        if planned is None or planned.resolved:
            raise FinalPlanIntegrityError(
                ModelFailureCategory.FINAL_PLAN_OMITTED_DISAGREEMENT
            )
    if state["disagreements"] and not plan.final_consensus.has_unresolved_disagreement:
        raise FinalPlanIntegrityError(
            ModelFailureCategory.FINAL_PLAN_DISAGREEMENT_STATUS
        )
    return plan.model_copy(
        update={"confidence": calculate_final_confidence(state, plan=plan)}
    )


def merge_unique_strings(*groups: Sequence[str]) -> list[str]:
    result: list[str] = []
    for group in groups:
        for item in group:
            if item not in result:
                result.append(item)
    return result
