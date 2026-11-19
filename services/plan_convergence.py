"""Deterministic projection and aggregation of the shared operative working draft.

The working draft is the only place where the Spine Surgeon's structured options
and other specialists' incremental positions meet. Everything here is
deterministic: no model call, no clinical rewriting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from schemas import (
    AgentRole,
    DefinitiveSurgeryStatus,
    MISSING_RECORDS,
    OperativePlanDraft,
    PlanPosition,
    PlanPositionAction,
    SpineSurgeonOutput,
    SurgeryFieldStatus,
    SurgicalOption,
)

CRITICAL_OPERATIVE_FIELDS: tuple[str, ...] = (
    "/recommended_surgery/procedure_name",
    "/recommended_surgery/levels",
    "/recommended_surgery/approach",
    "/recommended_surgery/decompression_extent",
    "/recommended_surgery/fusion_or_fixation",
)

FINAL_POSITION_ROLES: tuple[AgentRole, ...] = (
    AgentRole.NEUROLOGIST,
    AgentRole.ANESTHESIOLOGIST,
    AgentRole.PERIOPERATIVE_PHYSICIAN,
)

OPERATIVE_FIELD_ATTRIBUTES = {
    "/recommended_surgery/procedure_name": "procedure_name",
    "/recommended_surgery/levels": "levels",
    "/recommended_surgery/approach": "approach",
    "/recommended_surgery/decompression_extent": "decompression_extent",
    "/recommended_surgery/fusion_or_fixation": "fusion_or_fixation",
}

OPERATIVE_STATUS_ATTRIBUTES = {
    "procedure_name": "procedure_status",
    "levels": "levels_status",
    "approach": "approach_status",
    "decompression_extent": "decompression_status",
    "fusion_or_fixation": "fusion_fixation_status",
}


def derive_definitive_status(recommended_surgery) -> DefinitiveSurgeryStatus:
    """Derive the overall status from per-field statuses, deterministically."""

    statuses = [
        getattr(recommended_surgery, attribute)
        for attribute in OPERATIVE_STATUS_ATTRIBUTES.values()
    ]
    established = sum(
        1 for status in statuses if status is SurgeryFieldStatus.ESTABLISHED
    )
    if established == len(statuses):
        return DefinitiveSurgeryStatus.ESTABLISHED
    if established:
        return DefinitiveSurgeryStatus.PARTIAL
    return DefinitiveSurgeryStatus.WITHHELD


def _option_identity(option: SurgicalOption) -> str:
    payload = option.model_dump(mode="json")
    payload.pop("option_id", None)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def project_plan_draft(
    surgeon_output: SpineSurgeonOutput,
    *,
    source_sequence: int,
    previous: OperativePlanDraft | None = None,
) -> OperativePlanDraft | None:
    """Project a draft from the surgeon's normalized output.

    A draft exists only with a surgical indication. Positions from a previous
    draft version are carried forward only when the referenced option content
    is unchanged; a revision resets all other stances.
    """

    if not surgeon_output.surgical_indication:
        return None
    version = previous.version + 1 if previous is not None else 1
    positions: list[PlanPosition] = []
    if previous is not None and previous.surgical_indication:
        old_identities = {
            option.option_id: _option_identity(option) for option in previous.options
        }
        new_identities = {
            option.option_id: _option_identity(option) for option in surgeon_output.operative_options
        }
        for position in previous.positions:
            if position.draft_version != previous.version:
                continue
            if old_identities.get(position.option_id) != new_identities.get(position.option_id):
                continue
            if new_identities.get(position.option_id) is None:
                continue
            positions.append(position.model_copy(update={"draft_version": version}))
    return OperativePlanDraft(
        version=version,
        source_contribution_sequence=source_sequence,
        surgical_indication=True,
        options=list(surgeon_output.operative_options),
        preferred_option_id=surgeon_output.preferred_option_id,
        approach_comparison=surgeon_output.approach_comparison,
        positions=positions,
    )


def apply_position(
    draft: OperativePlanDraft | None,
    position: PlanPosition,
) -> OperativePlanDraft | None:
    """Attach one position to the current draft, replacing a prior stance of the
    same role on the same option. Mismatched versions or unknown options are
    ignored so stale stances never enter the shared object."""

    if draft is None or position.draft_version != draft.version:
        return draft
    option_ids = {option.option_id for option in draft.options}
    if position.option_id not in option_ids:
        return draft
    positions = [
        item
        for item in draft.positions
        if not (item.role is position.role and item.option_id == position.option_id)
    ]
    return draft.model_copy(update={"positions": [*positions, position]})


def current_positions(draft: OperativePlanDraft | None) -> list[PlanPosition]:
    if draft is None:
        return []
    return [
        position
        for position in draft.positions
        if position.draft_version == draft.version
    ]


def option_by_id(draft: OperativePlanDraft, option_id: str) -> SurgicalOption | None:
    return next(
        (option for option in draft.options if option.option_id == option_id),
        None,
    )


def positions_for_option(
    draft: OperativePlanDraft,
    option_id: str,
) -> list[PlanPosition]:
    return [
        position
        for position in current_positions(draft)
        if position.option_id == option_id
    ]


def position_fields(position: PlanPosition) -> frozenset[str]:
    """Fields a position addresses; an empty target list means every field."""

    if not position.target_fields:
        return frozenset(CRITICAL_OPERATIVE_FIELDS)
    return frozenset(
        field for field in position.target_fields if field in CRITICAL_OPERATIVE_FIELDS
    )


def field_oppositions(
    draft: OperativePlanDraft,
    option_id: str,
    field: str,
) -> list[PlanPosition]:
    """Positions that oppose the option on the given operative field."""

    if draft is None:
        return []
    return [
        position
        for position in positions_for_option(draft, option_id)
        if position.action is PlanPositionAction.OPPOSE
        and field in position_fields(position)
    ]


def field_reservations(
    draft: OperativePlanDraft,
    option_id: str,
    field: str,
) -> list[PlanPosition]:
    """Positions that support the option on the given field with reservations."""

    if draft is None:
        return []
    return [
        position
        for position in positions_for_option(draft, option_id)
        if position.action is PlanPositionAction.SUPPORT_WITH_RESERVATIONS
        and field in position_fields(position)
    ]


def compute_field_matrix(draft: OperativePlanDraft | None) -> list[dict]:
    """Field-level stance matrix for the moderator's routing context."""

    if draft is None:
        return []
    matrix: list[dict] = []
    for option in draft.options:
        for field in CRITICAL_OPERATIVE_FIELDS:
            positions = positions_for_option(draft, option.option_id)
            supports = [
                {
                    "role": position.role.value,
                    "reservations": position.reservations,
                    "evidence_ids": position.evidence_ids,
                }
                for position in positions
                if position.action is PlanPositionAction.SUPPORT
                and field in position_fields(position)
            ]
            reservations = [
                {
                    "role": position.role.value,
                    "reservations": position.reservations,
                    "evidence_ids": position.evidence_ids,
                }
                for position in positions
                if position.action is PlanPositionAction.SUPPORT_WITH_RESERVATIONS
                and field in position_fields(position)
            ]
            oppositions = [
                {
                    "role": position.role.value,
                    "reservations": position.reservations,
                    "evidence_ids": position.evidence_ids,
                }
                for position in positions
                if position.action is PlanPositionAction.OPPOSE
                and field in position_fields(position)
            ]
            matrix.append(
                {
                    "option_id": option.option_id,
                    "option_rank": option.rank,
                    "field": field,
                    "supports": supports,
                    "supports_with_reservations": reservations,
                    "oppositions": oppositions,
                }
            )
    return matrix


def has_positioned(role: AgentRole, draft: OperativePlanDraft | None) -> bool:
    return any(position.role is role for position in current_positions(draft))


def unpositioned_roles(
    draft: OperativePlanDraft | None,
    roles: Sequence[AgentRole],
) -> list[AgentRole]:
    return [role for role in roles if not has_positioned(role, draft)]


def render_plan_draft(draft: OperativePlanDraft | None) -> str:
    """Deterministic markdown rendering for text-agent prompts.

    Contains only structured schema data: evidence identifiers, never source
    text, image bytes, local paths, or patient directory names.
    """

    if draft is None:
        return "No shared operative working plan exists yet."
    lines = [
        f"## Shared operative working plan (version {draft.version})",
        f"- Surgical indication: {str(draft.surgical_indication).lower()}",
    ]
    if not draft.surgical_indication:
        return "\n".join(lines)
    preferred = draft.preferred_option_id
    lines.append(f"- Preferred option: {preferred or MISSING_RECORDS}")
    for option in draft.options:
        lines.extend(
            [
                "",
                f"### Option #{option.rank} ({option.option_id})"
                + (" [preferred]" if option.option_id == preferred else ""),
                f"- procedure_name: {option.procedure_name}",
                f"- levels: {', '.join(option.levels) or MISSING_RECORDS}"
                f" (certainty={option.levels_certainty.value})",
                f"- approach: {option.approach.value}",
                f"- decompression_extent: {option.decompression_extent}",
                f"- fusion_or_fixation: {option.fusion_or_fixation}"
                f" (certainty={option.fixation_certainty.value})",
                f"- uncertainty: {option.uncertainty.value}, confidence={option.confidence}",
                f"- evidence_ids: {', '.join(option.evidence_ids)}",
                f"- required_confirmation: {'; '.join(option.required_confirmation) or 'none'}",
            ]
        )
    positions = current_positions(draft)
    lines.append("")
    lines.append("### Specialist positions on the current draft")
    if not positions:
        lines.append("- No positions recorded yet.")
    for position in positions:
        fields = (
            ", ".join(position.target_fields)
            if position.target_fields
            else "all operative fields"
        )
        reservations = "; ".join(position.reservations) or "none"
        lines.append(
            f"- {position.role.value}: {position.action.value} on {position.option_id}"
            f" [{fields}] confidence={position.confidence};"
            f" reservations: {reservations};"
            f" evidence_ids: {', '.join(position.evidence_ids) or 'none'}"
        )
    return "\n".join(lines)
