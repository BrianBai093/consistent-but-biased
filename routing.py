"""Deterministic routing policy that constrains moderator model decisions."""

from dataclasses import dataclass
from typing import Sequence

from schemas import (
    AgentRole,
    ModeratorAction,
    ModeratorDecision,
    PendingRequest,
)
from state import SPECIALIST_ROLES


MAX_SPECIALIST_TURNS = 3
ROLE_PRIORITY: tuple[AgentRole, ...] = (
    AgentRole.SPINE_SURGEON,
    AgentRole.NEUROLOGIST,
    AgentRole.ANESTHESIOLOGIST,
    AgentRole.PERIOPERATIVE_PHYSICIAN,
    AgentRole.RADIOLOGIST,
)
ROLE_DISPLAY_NAMES = {
    AgentRole.RADIOLOGIST: "Radiologist",
    AgentRole.SPINE_SURGEON: "Attending Spine Surgeon",
    AgentRole.ANESTHESIOLOGIST: "Anesthesiologist",
    AgentRole.NEUROLOGIST: "Neurologic Function Specialist",
    AgentRole.PERIOPERATIVE_PHYSICIAN: "Perioperative Physician (Hospitalist)",
}
DEFAULT_QUESTIONS = {
    AgentRole.RADIOLOGIST: "Please reassess the imaging issue that remains unresolved.",
    AgentRole.SPINE_SURGEON: "Please assess the surgical indication and recommended operative strategy.",
    AgentRole.ANESTHESIOLOGIST: "Please assess anesthesia readiness, positioning, and blood-loss risk.",
    AgentRole.NEUROLOGIST: "Please localize the neurologic findings and assess imaging concordance.",
    AgentRole.PERIOPERATIVE_PHYSICIAN: "Please assess medical optimization and perioperative risk.",
}


@dataclass(frozen=True, slots=True)
class RoutingResolution:
    decision: ModeratorDecision
    accepted_request_id: str | None
    expired_request_ids: tuple[str, ...]
    force_finalize: bool
    fallback_used: bool


def _count(speaker_counts: dict[str, int], role: AgentRole) -> int:
    return int(speaker_counts.get(role.value, 0))


def coverage_complete(speaker_counts: dict[str, int]) -> bool:
    return all(_count(speaker_counts, role) >= 1 for role in SPECIALIST_ROLES)


def eligible_speakers(
    speaker_counts: dict[str, int],
    last_specialist: AgentRole | None,
    max_turns_per_specialist: int = MAX_SPECIALIST_TURNS,
) -> list[AgentRole]:
    available = [
        role
        for role in ROLE_PRIORITY
        if _count(speaker_counts, role) < max_turns_per_specialist
    ]
    unspoken = [role for role in available if _count(speaker_counts, role) == 0]
    candidates = unspoken or available
    if last_specialist in candidates and len(candidates) > 1:
        candidates = [role for role in candidates if role is not last_specialist]
    return candidates


def resolve_routing_decision(
    *,
    proposed: ModeratorDecision | None,
    speaker_counts: dict[str, int],
    pending_requests: Sequence[PendingRequest],
    unresolved_issues: Sequence[str],
    last_specialist: AgentRole | None,
    max_turns_per_specialist: int = MAX_SPECIALIST_TURNS,
    available_image_ids: Sequence[str] = (),
) -> RoutingResolution:
    candidates = eligible_speakers(
        speaker_counts,
        last_specialist,
        max_turns_per_specialist,
    )
    expired = tuple(
        request.request_id
        for request in pending_requests
        if _count(speaker_counts, request.target) >= max_turns_per_specialist
    )
    active_requests = [
        request for request in pending_requests if request.request_id not in expired
    ]
    priority_request = next(
        (request for request in active_requests if request.target in candidates),
        None,
    )
    issues = _merge_unique(
        unresolved_issues,
        [
            f"Request {request.request_id} could not be fulfilled because "
            f"{ROLE_DISPLAY_NAMES[request.target]} reached the speaking limit."
            for request in pending_requests
            if request.request_id in expired
        ],
    )

    if not candidates:
        decision = ModeratorDecision(
            public_message=(
                "All specialists have reached the speaking limit. The discussion is ready "
                "for final synthesis with any unresolved issues preserved."
            ),
            action=ModeratorAction.FINALIZE,
            next_speaker=None,
            question_for_next_speaker=None,
            unresolved_issues=issues,
            disagreements=proposed.disagreements if proposed else [],
            consensus_reached=proposed.consensus_reached if proposed else False,
            routing_reason="No eligible specialist remains under the speaking limit.",
            continue_discussion=False,
            enter_final_summary=True,
        )
        return RoutingResolution(decision, None, expired, True, proposed is not None)

    invalid = proposed is None
    if proposed is not None:
        if proposed.action is ModeratorAction.FINALIZE:
            unexplained_nonconsensus = (
                not proposed.consensus_reached
                and not proposed.unresolved_issues
                and not proposed.disagreements
            )
            invalid = (
                not coverage_complete(speaker_counts)
                or priority_request is not None
                or unexplained_nonconsensus
            )
        else:
            invalid = proposed.next_speaker not in candidates
            if proposed.next_speaker is AgentRole.RADIOLOGIST:
                invalid = invalid or not set(proposed.target_image_ids) <= set(
                    available_image_ids
                )
            if priority_request is not None and proposed.next_speaker is not priority_request.target:
                invalid = True

    if not invalid and proposed is not None:
        accepted = None
        if proposed.action is ModeratorAction.ASK_SPECIALIST:
            accepted_request = next(
                (
                    request
                    for request in active_requests
                    if request.target is proposed.next_speaker
                ),
                None,
            )
            accepted = accepted_request.request_id if accepted_request else None
        proposed_issues = _merge_unique(proposed.unresolved_issues, issues)
        decision = proposed.model_copy(update={"unresolved_issues": proposed_issues})
        return RoutingResolution(decision, accepted, expired, False, False)

    selected = priority_request.target if priority_request else candidates[0]
    question = priority_request.question if priority_request else DEFAULT_QUESTIONS[selected]
    reason = (
        "A pending specialist request has routing priority."
        if priority_request
        else "Deterministic fallback selected the highest-priority eligible specialist."
    )
    target_image_ids: list[str] = []
    if selected is AgentRole.RADIOLOGIST:
        allowed = set(available_image_ids)
        requested = (
            [
                evidence_id
                for evidence_id in priority_request.evidence_ids
                if evidence_id in allowed
            ]
            if priority_request
            else []
        )
        target_image_ids = (requested or list(available_image_ids)[:1])[:4]
    decision = ModeratorDecision(
        public_message=f"{ROLE_DISPLAY_NAMES[selected]}, {question}",
        action=ModeratorAction.ASK_SPECIALIST,
        next_speaker=selected,
        question_for_next_speaker=question,
        unresolved_issues=issues,
        disagreements=proposed.disagreements if proposed else [],
        consensus_reached=False,
        routing_reason=reason,
        continue_discussion=True,
        enter_final_summary=False,
        target_image_ids=target_image_ids,
    )
    return RoutingResolution(
        decision=decision,
        accepted_request_id=priority_request.request_id if priority_request else None,
        expired_request_ids=expired,
        force_finalize=False,
        fallback_used=True,
    )


def _merge_unique(*groups: Sequence[str]) -> list[str]:
    result: list[str] = []
    for group in groups:
        for item in group:
            if item not in result:
                result.append(item)
    return result
