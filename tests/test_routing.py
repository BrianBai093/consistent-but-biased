import pytest
from pydantic import ValidationError

from routing import eligible_speakers, resolve_routing_decision
from schemas import (
    AgentRole,
    ModeratorAction,
    ModeratorDecision,
    PendingRequest,
)
from state import initial_speaker_counts
from tests.dynamic_factories import ask, finalize


def test_unspoken_specialists_are_the_only_initial_candidates() -> None:
    counts = initial_speaker_counts()
    counts[AgentRole.RADIOLOGIST.value] = 1
    assert eligible_speakers(counts, AgentRole.RADIOLOGIST) == [
        AgentRole.SPINE_SURGEON,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PERIOPERATIVE_PHYSICIAN,
    ]


def test_early_finalize_is_replaced_by_deterministic_fallback() -> None:
    counts = initial_speaker_counts()
    counts[AgentRole.RADIOLOGIST.value] = 1
    resolution = resolve_routing_decision(
        proposed=finalize(),
        speaker_counts=counts,
        pending_requests=[],
        unresolved_issues=[],
        last_specialist=AgentRole.RADIOLOGIST,
    )
    assert resolution.fallback_used is True
    assert resolution.decision.action is ModeratorAction.ASK_SPECIALIST
    assert resolution.decision.next_speaker is AgentRole.SPINE_SURGEON


def test_pending_request_overrides_other_valid_candidate() -> None:
    counts = initial_speaker_counts()
    counts[AgentRole.RADIOLOGIST.value] = 1
    request = PendingRequest(
        request_id="request-1",
        requested_by=AgentRole.RADIOLOGIST,
        target=AgentRole.NEUROLOGIST,
        question="Please assess symptom-imaging concordance.",
        source_message_sequence=2,
    )
    resolution = resolve_routing_decision(
        proposed=ask(AgentRole.SPINE_SURGEON),
        speaker_counts=counts,
        pending_requests=[request],
        unresolved_issues=[],
        last_specialist=AgentRole.RADIOLOGIST,
    )
    assert resolution.decision.next_speaker is AgentRole.NEUROLOGIST
    assert resolution.accepted_request_id == request.request_id


def test_pending_request_blocks_finalize_after_minimum_coverage() -> None:
    counts = {role.value: 1 for role in AgentRole if role is not AgentRole.MODERATOR}
    request = PendingRequest(
        request_id="request-followup",
        requested_by=AgentRole.SPINE_SURGEON,
        target=AgentRole.NEUROLOGIST,
        question="Please clarify the neurologic baseline.",
        source_message_sequence=8,
    )
    resolution = resolve_routing_decision(
        proposed=finalize(),
        speaker_counts=counts,
        pending_requests=[request],
        unresolved_issues=[],
        last_specialist=AgentRole.PERIOPERATIVE_PHYSICIAN,
    )
    assert resolution.decision.action is ModeratorAction.ASK_SPECIALIST
    assert resolution.decision.next_speaker is AgentRole.NEUROLOGIST
    assert resolution.accepted_request_id == request.request_id


def test_speaker_at_limit_is_rejected_and_expired_request_is_recorded() -> None:
    counts = {role.value: 1 for role in AgentRole if role is not AgentRole.MODERATOR}
    counts[AgentRole.RADIOLOGIST.value] = 5
    request = PendingRequest(
        request_id="request-radiology",
        requested_by=AgentRole.SPINE_SURGEON,
        target=AgentRole.RADIOLOGIST,
        question="Please review the level.",
        source_message_sequence=8,
    )
    resolution = resolve_routing_decision(
        proposed=ask(AgentRole.RADIOLOGIST),
        speaker_counts=counts,
        pending_requests=[request],
        unresolved_issues=[],
        last_specialist=AgentRole.SPINE_SURGEON,
    )
    assert resolution.decision.next_speaker is not AgentRole.RADIOLOGIST
    assert request.request_id in resolution.expired_request_ids
    assert any("speaking limit" in issue for issue in resolution.decision.unresolved_issues)


def test_no_candidates_forces_finalization() -> None:
    counts = {role.value: 5 for role in AgentRole if role is not AgentRole.MODERATOR}
    resolution = resolve_routing_decision(
        proposed=None,
        speaker_counts=counts,
        pending_requests=[],
        unresolved_issues=["An unresolved issue"],
        last_specialist=AgentRole.RADIOLOGIST,
    )
    assert resolution.force_finalize is True
    assert resolution.decision.action is ModeratorAction.FINALIZE
    assert resolution.decision.unresolved_issues == ["An unresolved issue"]


def test_unexplained_nonconsensus_finalize_is_rejected() -> None:
    counts = {role.value: 1 for role in AgentRole if role is not AgentRole.MODERATOR}
    proposed = finalize().model_copy(update={"consensus_reached": False})

    resolution = resolve_routing_decision(
        proposed=proposed,
        speaker_counts=counts,
        pending_requests=[],
        unresolved_issues=[],
        last_specialist=AgentRole.PERIOPERATIVE_PHYSICIAN,
    )

    assert resolution.fallback_used is True
    assert resolution.decision.action is ModeratorAction.ASK_SPECIALIST


def test_invalid_next_speaker_is_rejected_by_schema() -> None:
    with pytest.raises(ValidationError, match="specialist next_speaker"):
        ModeratorDecision(
            public_message="Invalid",
            action=ModeratorAction.ASK_SPECIALIST,
            next_speaker=AgentRole.MODERATOR,
            question_for_next_speaker="Answer.",
            unresolved_issues=[],
            disagreements=[],
            consensus_reached=False,
            routing_reason="Invalid test.",
            continue_discussion=True,
            enter_final_summary=False,
        )


def test_duplicate_target_image_ids_are_rejected_by_schema() -> None:
    payload = ask(AgentRole.RADIOLOGIST).model_dump()
    payload["target_image_ids"] = ["mri-1", "mri-1"]

    with pytest.raises(ValidationError, match="target_image_ids must be unique"):
        ModeratorDecision.model_validate(payload)
