"""Deterministic evidence and context policy for bounded MDT model calls."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from exceptions import StructuredOutputError
from schemas import (
    AgentRole,
    CaseImage,
    ChatMessage,
    ClaimEvidence,
    DynamicStability,
    MISSING_RECORDS,
    RadiologyReport,
    SpineSurgeonOutput,
    SpecialistContribution,
)
from services.radiology_markdown import render_radiology_markdown
from state import MDTState, SPECIALIST_ROLES


_SOURCE_BLOCK = re.compile(
    r"<!--\s*source\s*:\s*([^\s>]+)\s*-->\s*(.*?)(?=<!--\s*source\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def extract_source_blocks(markdown: str) -> dict[str, str]:
    """Return stable source-id blocks without interpreting their clinical content."""

    return {
        match.group(1): match.group(2).strip()
        for match in _SOURCE_BLOCK.finditer(markdown)
        if match.group(2).strip()
    }


def valid_source_ids(state: MDTState) -> set[str]:
    identifiers = {
        *(page.page_id for page in state["case_pages"]),
        *(page.page_id for page in state["test_pages"]),
        *(image.image_id for image in state["images"]),
    }
    identifiers.update(extract_source_blocks(state["case_preop_md"]))
    identifiers.update(extract_source_blocks(state["tests_preop_md"]))
    return identifiers


def evidence_ids_from_claims(claims: Sequence[ClaimEvidence]) -> list[str]:
    return _unique(
        evidence_id
        for claim in claims
        for evidence_id in claim.evidence_ids
    )


def evidence_ids_from_output(output: BaseModel) -> list[str]:
    claims = getattr(output, "claim_evidence", [])
    identifiers = evidence_ids_from_claims(claims)
    if isinstance(output, RadiologyReport):
        identifiers = _unique(
            identifiers,
            output.operative_planning.evidence_ids,
            *(
                finding.image_ids
                for finding in (
                    *output.xray_findings,
                    *output.ct_findings,
                    *output.mri_findings,
                )
            ),
        )
    if isinstance(output, SpineSurgeonOutput):
        identifiers = _unique(
            identifiers,
            *(option.evidence_ids for option in output.operative_options),
            *(
                assessment.evidence_ids
                for assessment in (
                    output.approach_comparison.assessments
                    if output.approach_comparison
                    else []
                )
            ),
        )
    return identifiers


def validate_specialist_evidence(
    output: BaseModel,
    *,
    allowed_ids: set[str],
) -> BaseModel:
    """Reject fabricated source identifiers before an opinion enters shared state."""

    claims = list(getattr(output, "claim_evidence", []))
    if not isinstance(output, RadiologyReport) and not claims:
        raise StructuredOutputError("A specialist output requires claim_evidence.")
    referenced = set(evidence_ids_from_output(output))
    questions = getattr(output, "questions_for_other_agents", [])
    referenced.update(
        evidence_id
        for question in questions
        for evidence_id in question.evidence_ids
    )
    unknown = referenced - allowed_ids
    if unknown:
        raise StructuredOutputError("A specialist output referenced an unknown evidence id.")
    return output


def normalize_radiology_planning(
    output: BaseModel,
    *,
    images: Sequence[CaseImage],
    allowed_ids: set[str],
) -> RadiologyReport:
    """Enforce imaging-only planning semantics and dynamic-view provenance."""

    report = RadiologyReport.model_validate(output).model_copy(update={"addenda": []})
    if not report.operative_planning.evidence_ids:
        raise StructuredOutputError(
            "Radiology operative planning requires at least one supplied image id."
        )
    has_dynamic_views = any(_is_dynamic_view(image) for image in images)
    if not has_dynamic_views:
        planning = report.operative_planning.model_copy(
            update={
                "dynamic_stability": DynamicStability.NOT_ASSESSED,
                "planning_uncertainties": _unique(
                    report.operative_planning.planning_uncertainties,
                    ["Dynamic stability was not assessed because no flexion-extension "
                     "or other dynamic view was supplied."],
                ),
            }
        )
        report = report.model_copy(update={"operative_planning": planning})
    return RadiologyReport.model_validate(
        validate_specialist_evidence(report, allowed_ids=allowed_ids)
    )


def normalize_spine_surgeon_output(
    output: BaseModel,
    *,
    allowed_ids: set[str],
) -> SpineSurgeonOutput:
    """Validate v4 operative options and deterministically project legacy fields."""

    surgeon = SpineSurgeonOutput.model_validate(
        validate_specialist_evidence(output, allowed_ids=allowed_ids)
    )
    options = sorted(surgeon.operative_options, key=lambda item: item.rank)
    if not surgeon.surgical_indication:
        if options:
            raise StructuredOutputError(
                "A spine-surgeon output without a surgical indication cannot publish "
                "operative options."
            )
        return surgeon.model_copy(
            update={
                "preferred_option_id": None,
                "approach_comparison": None,
                "recommended_procedure": MISSING_RECORDS,
                "surgical_levels": [],
                "approach": MISSING_RECORDS,
                "alternative_plans": [],
            }
        )

    if not 1 <= len(options) <= 3:
        raise StructuredOutputError(
            "A surgical indication requires one to three structured operative options."
        )
    option_ids = [option.option_id for option in options]
    if len(option_ids) != len(set(option_ids)):
        raise StructuredOutputError("Surgical option_id values must be unique.")
    if [option.rank for option in options] != list(range(1, len(options) + 1)):
        raise StructuredOutputError("Surgical option ranks must be contiguous from one.")
    preferred = options[0]
    if surgeon.preferred_option_id != preferred.option_id:
        raise StructuredOutputError("preferred_option_id must identify the rank-one option.")
    comparison = surgeon.approach_comparison
    if comparison is None:
        raise StructuredOutputError(
            "A surgical indication requires anterior/posterior/combined comparison."
        )
    if comparison.preferred_approach is not preferred.approach:
        raise StructuredOutputError(
            "The preferred approach must match the rank-one operative option."
        )
    return surgeon.model_copy(
        update={
            "operative_options": options,
            "recommended_procedure": preferred.procedure_name,
            "surgical_levels": preferred.levels,
            "approach": preferred.approach.value,
            "alternative_plans": [
                _legacy_option_summary(option) for option in options[1:]
            ],
        }
    )


def latest_contribution(
    state: MDTState,
    role: AgentRole,
) -> SpecialistContribution | None:
    return next(
        (
            contribution
            for contribution in reversed(state["specialist_contributions"])
            if contribution.role is role
        ),
        None,
    )


def latest_contributions(state: MDTState) -> list[SpecialistContribution]:
    return [
        contribution
        for role in SPECIALIST_ROLES
        if (contribution := latest_contribution(state, role)) is not None
    ]


def relevant_chat_for_role(state: MDTState, role: AgentRole) -> list[ChatMessage]:
    """Return at most four public messages that materially affect a follow-up."""

    if state["speaker_counts"].get(role.value, 0) == 0:
        return []

    sequences: set[int] = set()
    recent_specialists = [
        message
        for message in state["chat_log"]
        if message.speaker_role is not AgentRole.MODERATOR
    ][-2:]
    sequences.update(message.sequence for message in recent_specialists)
    prior = latest_contribution(state, role)
    if prior is not None:
        sequences.add(prior.message_sequence)
    for request in state["pending_requests"]:
        if request.target is role:
            sequences.add(request.source_message_sequence)
    if role is AgentRole.SPINE_SURGEON:
        for warning_role in (
            AgentRole.ANESTHESIOLOGIST,
            AgentRole.PERIOPERATIVE_PHYSICIAN,
        ):
            warning = latest_contribution(state, warning_role)
            if warning is not None:
                sequences.add(warning.message_sequence)
    selected = [
        message for message in state["chat_log"] if message.sequence in sequences
    ]
    return selected[-4:]


def source_documents_for_turn(
    state: MDTState,
    role: AgentRole,
) -> dict[str, str]:
    """Use full prepared documents once, then deterministic evidence blocks only."""

    if state["speaker_counts"].get(role.value, 0) == 0:
        return {
            "case_preop_md": state["case_preop_md"],
            "tests_preop_md": state["tests_preop_md"],
            "radiology_preop_md": state["radiology_preop_md"],
        }

    requested_ids = _requested_ids_for_turn(state, role)

    case_blocks = extract_source_blocks(state["case_preop_md"])
    test_blocks = extract_source_blocks(state["tests_preop_md"])
    requested_ids.update(
        identifier for identifier in case_blocks if identifier.startswith("addendum-")
    )
    return {
        "case_preop_md": _render_selected_blocks(case_blocks, requested_ids),
        "tests_preop_md": _render_selected_blocks(test_blocks, requested_ids),
        "radiology_preop_md": render_radiology_markdown(
            radiology_report_for_turn(state, role)
        ),
    }


def radiology_report_for_turn(
    state: MDTState,
    role: AgentRole,
) -> RadiologyReport:
    """Return the full first-turn report or a source-targeted follow-up view."""

    report = state["radiology_report"]
    if report is None:
        raise ValueError("A specialist turn requires a radiology report.")
    if state["speaker_counts"].get(role.value, 0) == 0:
        return report

    requested_ids = _requested_ids_for_turn(state, role)
    image_ids = {image.image_id for image in state["images"]}
    targeted_images = requested_ids & image_ids

    def selected(findings):
        return [
            finding
            for finding in findings
            if set(finding.image_ids) & targeted_images
        ]

    selected_xray = selected(report.xray_findings)
    selected_ct = selected(report.ct_findings)
    selected_mri = selected(report.mri_findings)
    selected_findings = [*selected_xray, *selected_ct, *selected_mri]
    targeted_summary = (
        "; ".join(finding.finding for finding in selected_findings)
        or "No image-linked finding was selected for this follow-up."
    )
    filtered_claims = [
        claim
        for claim in report.claim_evidence
        if set(claim.evidence_ids) & targeted_images
    ]

    return report.model_copy(
        update={
            "xray_findings": selected_xray,
            "ct_findings": selected_ct,
            "mri_findings": selected_mri,
            "spine_levels": [],
            "alignment": "Not repeated in targeted follow-up context.",
            "bony_structures": "Not repeated in targeted follow-up context.",
            "canal_and_foramina": "Use the selected image-linked findings.",
            "neural_compression": "Use the selected image-linked findings.",
            "cross_modality_synthesis": targeted_summary,
            "possible_diagnoses": [],
            "public_summary": targeted_summary,
            "questions_for_other_agents": [],
            "claim_evidence": filtered_claims,
            "addenda": [
                addendum
                for addendum in report.addenda
                if set(addendum.target_image_ids) & targeted_images
            ],
        }
    )


def compact_routing_chat(chat_log: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Keep only the latest public specialist conclusion per role for routing."""

    latest: dict[AgentRole, ChatMessage] = {}
    for message in chat_log:
        if message.speaker_role is not AgentRole.MODERATOR:
            latest[message.speaker_role] = message
    return sorted(latest.values(), key=lambda item: item.sequence)


def _requested_ids_for_turn(state: MDTState, role: AgentRole) -> set[str]:
    relevant_messages = relevant_chat_for_role(state, role)
    requested_ids = {
        evidence_id
        for message in relevant_messages
        for evidence_id in message.evidence_ids
    }
    prior = latest_contribution(state, role)
    if prior is not None:
        requested_ids.update(evidence_ids_from_output(prior.structured_output))
    for request in state["pending_requests"]:
        if request.target is role:
            requested_ids.update(request.evidence_ids)
    report = state["radiology_report"]
    if report is not None:
        requested_ids.update(
            image_id
            for addendum in report.addenda
            for image_id in addendum.target_image_ids
        )
    return requested_ids


def issue_snapshot(
    model_issues: Iterable[str],
    workflow_issues: Iterable[str],
    *,
    maximum: int = 12,
) -> list[str]:
    """Replace model issues each turn while always preserving workflow-owned issues."""

    workflow = _deduplicate_issues(workflow_issues)
    remaining = max(0, maximum - len(workflow))
    model = [
        issue
        for issue in _deduplicate_issues(model_issues)
        if _issue_key(issue) not in {_issue_key(item) for item in workflow}
    ][:remaining]
    return [*workflow, *model]


def _render_selected_blocks(
    blocks: dict[str, str],
    requested_ids: set[str],
) -> str:
    selected = [
        f"<!-- source: {identifier} -->\n\n{body}"
        for identifier, body in blocks.items()
        if identifier in requested_ids
    ]
    return "\n\n".join(selected) or "No additional source excerpt was selected."


def _deduplicate_issues(issues: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in issues:
        issue = " ".join(raw.split()).strip()
        key = _issue_key(issue)
        if issue and key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def _issue_key(issue: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", issue.casefold())


def _unique(*groups) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        values = [group] if isinstance(group, str) else group
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


def _legacy_option_summary(option) -> str:
    levels = ", ".join(option.levels) or "levels not established"
    return f"{option.procedure_name} | {option.approach.value} | {levels}"


def _is_dynamic_view(image: CaseImage) -> bool:
    metadata = " ".join(
        value
        for value in (
            image.view_or_plane,
            image.sequence,
            image.description,
        )
        if value
    ).casefold()
    return any(
        marker in metadata
        for marker in (
            "flex",
            "extension",
            "dynamic",
            "屈曲",
            "伸展",
            "屈伸",
            "过屈",
            "过伸",
            "动力位",
        )
    )
