"""Deterministic workflow-owned fields for a model-generated final plan."""

from schemas import DisagreementResolution, FinalPlan
from state import MDTState, SPECIALIST_ROLES
from termination import merge_unique_strings


def apply_authoritative_workflow_fields(
    plan: FinalPlan,
    state: MDTState,
) -> FinalPlan:
    """Preserve workflow facts without overriding clinical synthesis fields."""

    valid_evidence_ids = {item.evidence_id for item in state["retrieved_evidence"]}
    diagnoses = [
        diagnosis.model_copy(
            update={
                "guideline_evidence_ids": [
                    evidence_id
                    for evidence_id in diagnosis.guideline_evidence_ids
                    if evidence_id in valid_evidence_ids
                ]
            }
        )
        for diagnosis in plan.diagnoses
    ]
    planned_disagreements = {item.issue: item for item in plan.disagreements}
    disagreements = []
    for disagreement in state["disagreements"]:
        planned = planned_disagreements.get(disagreement.issue)
        disagreements.append(
            DisagreementResolution(
                agents=disagreement.agents,
                issue=disagreement.issue,
                positions=disagreement.positions,
                moderator_resolution=(
                    planned.moderator_resolution
                    if planned
                    else "Not resolved before finalization."
                ),
                resolved=False,
                unresolved_reason=(
                    planned.unresolved_reason
                    if planned and planned.unresolved_reason
                    else "Insufficient agreement or evidence before termination."
                ),
            )
        )
    missing_roles = [
        role.value
        for role in SPECIALIST_ROLES
        if not any(item.role is role for item in state["specialist_contributions"])
    ]
    remaining_concerns = merge_unique_strings(
        plan.final_consensus.remaining_concerns,
        state["unresolved_issues"],
        (item.issue for item in state["disagreements"]),
        (f"Missing structured contribution: {role}" for role in missing_roles),
    )
    consensus = plan.final_consensus.model_copy(
        update={
            "remaining_concerns": remaining_concerns,
            "has_unresolved_disagreement": bool(disagreements),
        }
    )
    return plan.model_copy(
        update={
            "diagnoses": diagnoses,
            "disagreements": disagreements,
            "final_consensus": consensus,
            "force_finalized": state["force_finalize"],
        }
    )
