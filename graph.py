"""Fixed opening and moderator-centered dynamic MDT workflows."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine, NotRequired, Sequence, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agents.registry import AgentRegistry
from exceptions import (
    DocumentPageError,
    InvalidStateUpdateError,
    ModelFailureCategory,
    ModelInvocationError,
    ModelRetryExhaustedError,
    ModelStageError,
    StructuredOutputError,
    model_failure_category,
)
from routing import (
    DEFAULT_QUESTIONS,
    ROLE_DISPLAY_NAMES,
    eligible_speakers,
    resolve_routing_decision,
)
from schemas import (
    DirectorStyle,
    AgentQuestion,
    AgentRole,
    AnesthesiologistOutput,
    CaseImage,
    CaseInput,
    ChatMessage,
    DocumentPage,
    Disagreement,
    FinalPlan,
    ImageQuality,
    ModeratorAction,
    ModeratorDecision,
    ModelFailureRecord,
    PendingRequest,
    PerioperativePhysicianOutput,
    RadiologyProgress,
    RunMetadata,
    RunPhase,
    SpecialistContribution,
    TerminationReason,
    WorkflowLimits,
)
from services.artifact_store import write_markdown
from services.context_policy import (
    evidence_ids_from_output,
    issue_snapshot,
    latest_contribution,
    latest_contributions,
    normalize_radiology_planning,
    normalize_spine_surgeon_output,
    radiology_report_for_turn,
    relevant_chat_for_role,
    source_documents_for_turn,
    valid_source_ids,
    validate_specialist_evidence,
)
from services.plan_convergence import (
    CRITICAL_OPERATIVE_FIELDS,
    FINAL_POSITION_ROLES,
    apply_position,
    has_positioned,
    project_plan_draft,
)
from services.final_plan_builder import apply_authoritative_workflow_fields
from services.final_plan_review import (
    apply_field_level_status,
    normalize_final_plan_for_evidence,
    validate_final_plan_evidence,
    validate_review_evidence,
)
from services.radiology_markdown import (
    merge_radiology_addendum,
    render_radiology_markdown,
)
from services.baseline_store import BaselineStore
from services.radiology_cache import RadiologyCache
from state import MDTState, create_initial_state
from termination import (
    forced_termination_reason,
    merge_unique_strings,
    normal_completion_ready,
    update_progress,
    validate_final_plan_integrity,
)


OPENING_QUESTION = (
    "Please provide an initial cross-modality review of all supplied X-ray, CT, and MRI "
    "key images, including image quality, principal findings, uncertainty, missing views, "
    "and imaging-only operative-planning factors."
)
CASE_DOCUMENT_STAGE = "CASE_DOCUMENT_TRANSCRIPTION"
TEST_DOCUMENT_STAGE = "TEST_DOCUMENT_TRANSCRIPTION"
CASE_SUMMARY_STAGE = "CASE_SUMMARIZATION"
RADIOLOGY_REVIEW_STAGE = "RADIOLOGY_REVIEW"
MODERATOR_ROUTING_STAGE = "moderator_routing"
SPECIALIST_CONTRIBUTION_STAGE = "specialist_contribution"
FINAL_PLAN_GENERATION_STAGE = "final_plan_generation"
FINAL_PLAN_REVIEW_STAGE = "final_plan_review"
FINAL_PLAN_REVISION_STAGE = "final_plan_revision"
INITIAL_TEXT_SPECIALIST_ROLES: tuple[AgentRole, ...] = (
    AgentRole.SPINE_SURGEON,
    AgentRole.NEUROLOGIST,
    AgentRole.ANESTHESIOLOGIST,
    AgentRole.PERIOPERATIVE_PHYSICIAN,
)


def _normalize_text_specialist_output(
    output,
    *,
    role: AgentRole,
    allowed_ids: set[str],
    working_plan=None,
    require_position: bool = False,
    forbid_new_questions: bool = False,
):
    if role is AgentRole.SPINE_SURGEON:
        return normalize_spine_surgeon_output(output, allowed_ids=allowed_ids)
    normalized = validate_specialist_evidence(output, allowed_ids=allowed_ids)
    position = getattr(normalized, "position_on_plan", None)
    if forbid_new_questions and normalized.questions_for_other_agents:
        raise StructuredOutputError(
            "A reserved final-position turn cannot raise new questions."
        )
    if position is None:
        if require_position:
            raise StructuredOutputError(
                f"{role.value} must state position_on_plan against the current "
                "operative working plan."
            )
        return normalized
    if working_plan is None:
        return normalized.model_copy(update={"position_on_plan": None})
    if position.role is not role:
        raise StructuredOutputError(
            "position_on_plan must use the speaking specialist's own role."
        )
    unknown = set(position.evidence_ids) - allowed_ids
    if unknown:
        raise StructuredOutputError(
            "position_on_plan referenced an unknown evidence id."
        )
    option_ids = {option.option_id for option in working_plan.options}
    if position.option_id not in option_ids:
        raise StructuredOutputError(
            "position_on_plan must reference an option in the current working plan."
        )
    return normalized


INDEPENDENT_ROUND_MESSAGE = (
    "The Radiologist's initial report is sealed. The remaining specialties will now "
    "provide independent first-round assessments from the same source evidence."
)

FINAL_POSITION_QUESTION = (
    "The discussion has reached its turn budget. State your final position on the "
    "current operative working plan: support, support_with_reservations, oppose, or "
    "abstain for the operative fields you can assess, with reservations and evidence "
    "identifiers. Do not raise new questions."
)


def _final_position_decision(role: AgentRole, state: MDTState) -> ModeratorDecision:
    return ModeratorDecision(
        public_message=(
            f"{ROLE_DISPLAY_NAMES[role]}, {FINAL_POSITION_QUESTION}"
        ),
        action=ModeratorAction.ASK_SPECIALIST,
        next_speaker=role,
        question_for_next_speaker=FINAL_POSITION_QUESTION,
        unresolved_issues=list(state["unresolved_issues"]),
        disagreements=state["disagreements"],
        consensus_reached=False,
        routing_reason=(
            "The turn budget was reached; collecting a final field-level position "
            "before synthesis."
        ),
        continue_discussion=True,
        enter_final_summary=False,
    )


class OpeningGraphInput(TypedDict):
    case_id: str
    director_style: NotRequired[DirectorStyle | str | None]
    case_pages: list[DocumentPage | dict[str, Any]]
    test_pages: list[DocumentPage | dict[str, Any]]
    images: list[CaseImage | dict[str, Any]]
    artifact_dir: str


NodeCallable = Callable[..., Coroutine[Any, Any, dict[str, Any]]]
RadiologyProgressSink = Callable[[str, RadiologyProgress], Awaitable[None]]


def opening_input(
    case: CaseInput,
    *,
    director_style: DirectorStyle | None = None,
) -> OpeningGraphInput:
    return OpeningGraphInput(
        case_id=case.case_id,
        director_style=director_style,
        case_pages=list(case.case_pages),
        test_pages=list(case.test_pages),
        images=list(case.images),
        artifact_dir=case.artifact_dir,
    )


def thread_config(
    thread_id: str,
    *,
    recursion_limit: int = 100,
) -> dict[str, Any]:
    normalized = thread_id.strip()
    if not normalized:
        raise ValueError("thread_id must not be empty")
    if recursion_limit < 1:
        raise ValueError("recursion_limit must be positive")
    return {
        "configurable": {"thread_id": normalized},
        "recursion_limit": recursion_limit,
    }


def _add_opening_nodes(
    builder: StateGraph,
    *,
    agent_registry: AgentRegistry,
    radiology_progress_sink: RadiologyProgressSink | None = None,
    radiology_cache: RadiologyCache | None = None,
    baseline_store: BaselineStore | None = None,
) -> None:
    async def normalize_input(state: OpeningGraphInput) -> MDTState:
        case = CaseInput.model_validate(
            {
                "case_id": state["case_id"],
                "case_pages": state["case_pages"],
                "test_pages": state["test_pages"],
                "images": state.get("images", []),
                "artifact_dir": state["artifact_dir"],
            }
        )
        raw_style = state.get("director_style")
        initial = create_initial_state(
            case,
            director_style=DirectorStyle(raw_style) if raw_style else None,
        )
        initial["phase"] = RunPhase.INPUT_NORMALIZED
        initial["workflow_step_count"] = 1
        # A stored baseline makes every later preparation node a no-op, so the
        # three discussion arms of one case argue over identical evidence
        # instead of re-running summarization and radiology review.
        baseline = (
            await baseline_store.load(
                case_pages=case.case_pages,
                test_pages=case.test_pages,
                images=case.images,
                case_id=case.case_id,
            )
            if baseline_store is not None
            else None
        )
        if baseline is not None:
            initial["case_preop_md"] = baseline.case_preop_md
            initial["tests_preop_md"] = baseline.tests_preop_md
            initial["case_summary_md"] = baseline.case_summary_md
            initial["radiology_preop_md"] = baseline.radiology_preop_md
            initial["radiology_report"] = baseline.radiology_report
            initial["preparation_warnings"] = list(baseline.preparation_warnings)
        return initial

    async def prepare_case_document(state: MDTState) -> dict[str, Any]:
        if state["case_preop_md"]:
            await write_markdown(
                state["artifact_dir"],
                "case_preop.md",
                state["case_preop_md"],
            )
            return {}
        try:
            result = await agent_registry.case_document.run(state["case_pages"])
            await write_markdown(
                state["artifact_dir"],
                "case_preop.md",
                result.output.markdown,
            )
            return {
                "case_preop_md": result.output.markdown,
                "preparation_warnings": result.output.warnings,
                "case_document_usages": list(result.all_usages),
            }
        except DocumentPageError as exc:
            raise ModelStageError(
                CASE_DOCUMENT_STAGE,
                exc.cause,
                item_id=exc.page_id,
            ) from exc
        except ModelInvocationError as exc:
            raise ModelStageError(CASE_DOCUMENT_STAGE, exc) from exc

    async def prepare_test_document(state: MDTState) -> dict[str, Any]:
        if state["tests_preop_md"]:
            await write_markdown(
                state["artifact_dir"],
                "tests_preop.md",
                state["tests_preop_md"],
            )
            return {}
        try:
            result = await agent_registry.test_document.run(state["test_pages"])
            await write_markdown(
                state["artifact_dir"],
                "tests_preop.md",
                result.output.markdown,
            )
            return {
                "tests_preop_md": result.output.markdown,
                "preparation_warnings": result.output.warnings,
                "test_document_usages": list(result.all_usages),
            }
        except DocumentPageError as exc:
            raise ModelStageError(
                TEST_DOCUMENT_STAGE,
                exc.cause,
                item_id=exc.page_id,
            ) from exc
        except ModelInvocationError as exc:
            raise ModelStageError(TEST_DOCUMENT_STAGE, exc) from exc

    async def prepare_documents(state: MDTState) -> dict[str, Any]:
        metadata = state["run_metadata"].model_copy(
            update={
                "model_calls": [
                    *state["run_metadata"].model_calls,
                    *state["case_document_usages"],
                    *state["test_document_usages"],
                ]
            }
        )
        return {
            "phase": RunPhase.DOCUMENT_PREPARATION,
            "run_metadata": metadata,
            "workflow_step_count": state["workflow_step_count"] + 1,
        }

    async def summarize_case(state: MDTState) -> dict[str, Any]:
        if state["case_summary_md"]:
            await write_markdown(
                state["artifact_dir"],
                "case_summary.md",
                state["case_summary_md"],
            )
            return {
                "phase": RunPhase.CASE_SUMMARIZATION,
                "workflow_step_count": state["workflow_step_count"] + 1,
            }
        try:
            result = await agent_registry.case_summary.run(
                case_preop_md=state["case_preop_md"],
                tests_preop_md=state["tests_preop_md"],
            )
        except ModelInvocationError as exc:
            raise ModelStageError(CASE_SUMMARY_STAGE, exc) from exc
        summary = result.output
        await write_markdown(state["artifact_dir"], "case_summary.md", summary.markdown)
        metadata = state["run_metadata"].model_copy(
            update={
                "model_calls": [
                    *state["run_metadata"].model_calls,
                    *result.all_usages,
                ]
            }
        )
        return {
            "phase": RunPhase.CASE_SUMMARIZATION,
            "case_summary_md": summary.markdown,
            "preparation_warnings": [
                *state["preparation_warnings"],
                *summary.warnings,
            ],
            "run_metadata": metadata,
            "workflow_step_count": state["workflow_step_count"] + 1,
        }

    async def moderator_opening(state: MDTState) -> dict[str, Any]:
        sequence = state["sequence_counter"]
        message = ChatMessage(
            message_id=f"{state['case_id']}:moderator-opening",
            sequence=sequence,
            speaker_id=AgentRole.MODERATOR.value,
            speaker_name="MDT Moderator",
            speaker_role=AgentRole.MODERATOR,
            content=OPENING_QUESTION,
            mentions=[AgentRole.RADIOLOGIST],
            turn_number=1,
        )
        return {
            "phase": RunPhase.MODERATOR_OPENING,
            "chat_log": [message],
            "sequence_counter": sequence + 1,
            "next_speaker": AgentRole.RADIOLOGIST,
            "current_question": OPENING_QUESTION,
            "unresolved_issues": ["Initial radiology review pending"],
            "workflow_step_count": state["workflow_step_count"] + 1,
        }

    async def radiologist_initial_report(
        state: MDTState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        allowed_ids = valid_source_ids(state)

        def validate_initial_report(output):
            return normalize_radiology_planning(
                output,
                images=state["images"],
                allowed_ids=allowed_ids,
            )

        cache_allowed = (
            radiology_cache is not None
            and "<!-- source: addendum-" not in state["case_preop_md"]
        )
        # A loaded baseline already carries the report, which is a stronger
        # guarantee than the cache: it is the very report the other arms read.
        baseline_report = state["radiology_report"]
        cached_report = baseline_report or (
            await radiology_cache.load(
                case_pages=state["case_pages"],
                test_pages=state["test_pages"],
                images=state["images"],
            )
            if cache_allowed and radiology_cache is not None
            else None
        )
        cache_hit = cached_report is not None and baseline_report is None
        usages = []
        if cached_report is not None:
            report = validate_initial_report(cached_report)
            progress_callback = _bind_radiology_progress(
                radiology_progress_sink,
                config,
            )
            if progress_callback is not None:
                await progress_callback(
                    RadiologyProgress(
                        stage="completed",
                        completed_images=len(state["images"]),
                        total_images=len(state["images"]),
                        total_batches=1,
                    )
                )
        else:
            try:
                result = await agent_registry.radiologist.run(
                    case_summary_md=state["case_summary_md"],
                    images=state["images"],
                    chat_log=state["chat_log"],
                    question=state["current_question"] or OPENING_QUESTION,
                    progress_callback=_bind_radiology_progress(
                        radiology_progress_sink,
                        config,
                    ),
                    output_processor=validate_initial_report,
                )
            except ModelInvocationError as exc:
                raise ModelStageError(RADIOLOGY_REVIEW_STAGE, exc) from exc
            report = result.output
            usages = list(result.all_usages)
            if cache_allowed and radiology_cache is not None:
                await radiology_cache.store(
                    case_pages=state["case_pages"],
                    test_pages=state["test_pages"],
                    images=state["images"],
                    report=report,
                )
        radiology_preop_md = render_radiology_markdown(report)
        await write_markdown(
            state["artifact_dir"], "radiology_preop.md", radiology_preop_md
        )
        counts = dict(state["speaker_counts"])
        turn_number = counts[AgentRole.RADIOLOGIST.value] + 1
        counts[AgentRole.RADIOLOGIST.value] = turn_number
        sequence = state["sequence_counter"]
        questions = list(report.questions_for_other_agents)
        requests = _new_pending_requests(
            case_id=state["case_id"],
            requested_by=AgentRole.RADIOLOGIST,
            turn_number=turn_number,
            source_sequence=sequence,
            questions=questions,
        )
        message = ChatMessage(
            message_id=f"{state['case_id']}:radiologist-initial",
            sequence=sequence,
            speaker_id=AgentRole.RADIOLOGIST.value,
            speaker_name="Radiologist",
            speaker_role=AgentRole.RADIOLOGIST,
            content=report.public_summary,
            mentions=_unique_roles(question.target for question in questions),
            confidence=report.confidence,
            evidence_ids=evidence_ids_from_output(report),
            turn_number=turn_number,
        )
        contribution = SpecialistContribution(
            role=AgentRole.RADIOLOGIST,
            turn_number=turn_number,
            message_sequence=sequence,
            structured_output=report,
        )
        metadata = state["run_metadata"].model_copy(
            update={
                "specialist_message_count": state["run_metadata"].specialist_message_count + 1,
                "model_calls": [
                    *state["run_metadata"].model_calls,
                    *usages,
                ],
                "radiology_cache_hit": cache_hit,
            }
        )
        seen_fingerprints, no_progress_streak = update_progress(
            seen_fingerprints=state["seen_conclusion_fingerprints"],
            no_progress_streak=state["no_progress_streak"],
            output=report,
        )
        return {
            "phase": RunPhase.RADIOLOGY_REVIEW,
            "radiology_report": report,
            "radiology_preop_md": radiology_preop_md,
            "radiology_ready": True,
            "chat_log": [message],
            "sequence_counter": sequence + 1,
            "speaker_counts": counts,
            "next_speaker": None,
            "current_question": None,
            "pending_requests": requests,
            "specialist_contributions": [contribution],
            "unresolved_issues": _merge_unique(question.question for question in questions),
            "run_metadata": metadata,
            "seen_conclusion_fingerprints": seen_fingerprints,
            "no_progress_streak": no_progress_streak,
            "workflow_step_count": state["workflow_step_count"] + 1,
        }

    async def moderator_review(state: MDTState) -> dict[str, Any]:
        if not state["radiology_ready"] or state["radiology_report"] is None:
            raise InvalidStateUpdateError(
                "Moderator review requires a completed structured radiology report."
            )
        # Preparation is complete here, so this is where a reusable baseline
        # becomes available to later runs of the same case.
        if baseline_store is not None:
            await baseline_store.store(
                case_pages=state["case_pages"],
                test_pages=state["test_pages"],
                images=state["images"],
                case_id=state["case_id"],
                case_preop_md=state["case_preop_md"],
                tests_preop_md=state["tests_preop_md"],
                case_summary_md=state["case_summary_md"],
                radiology_preop_md=state["radiology_preop_md"],
                preparation_warnings=state["preparation_warnings"],
                radiology_report=state["radiology_report"],
            )
        return {
            "phase": RunPhase.INDEPENDENT_SPECIALIST_ROUND,
            "next_speaker": None,
            "current_question": None,
            "workflow_step_count": state["workflow_step_count"] + 1,
        }

    builder.add_node("normalize_input", normalize_input)
    builder.add_node("prepare_case_document", prepare_case_document)
    builder.add_node("prepare_test_document", prepare_test_document)
    builder.add_node("prepare_documents", prepare_documents)
    builder.add_node("summarize_case", summarize_case)
    builder.add_node("moderator_opening", moderator_opening)
    builder.add_node("radiologist_initial_report", radiologist_initial_report)
    builder.add_node("moderator_review", moderator_review)
    builder.add_edge(START, "normalize_input")
    builder.add_edge("normalize_input", "prepare_case_document")
    builder.add_edge("normalize_input", "prepare_test_document")
    builder.add_edge(
        ["prepare_case_document", "prepare_test_document"],
        "prepare_documents",
    )
    builder.add_edge("prepare_documents", "summarize_case")
    builder.add_edge("summarize_case", "moderator_opening")
    builder.add_edge("moderator_opening", "radiologist_initial_report")


def create_opening_graph(
    *,
    agent_registry: AgentRegistry,
    checkpointer: BaseCheckpointSaver,
    interrupt_after: Sequence[str] | None = None,
    radiology_progress_sink: RadiologyProgressSink | None = None,
    radiology_cache: RadiologyCache | None = None,
    baseline_store: BaselineStore | None = None,
):
    """Compile the Checkpoint 3 opening graph without dynamic routing."""

    builder = StateGraph(MDTState, input_schema=OpeningGraphInput)
    _add_opening_nodes(
        builder,
        agent_registry=agent_registry,
        radiology_progress_sink=radiology_progress_sink,
        radiology_cache=radiology_cache,
        baseline_store=baseline_store,
    )
    builder.add_edge("radiologist_initial_report", "moderator_review")
    builder.add_edge("moderator_review", END)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=list(interrupt_after or []),
        name="spine-mdt-opening",
    )


def create_mdt_graph(
    *,
    agent_registry: AgentRegistry,
    checkpointer: BaseCheckpointSaver,
    limits: WorkflowLimits | None = None,
    interrupt_after: Sequence[str] | None = None,
    radiology_progress_sink: RadiologyProgressSink | None = None,
    radiology_cache: RadiologyCache | None = None,
    baseline_store: BaselineStore | None = None,
):
    """Compile the Checkpoint 4 graph that stops before final synthesis."""

    return _create_dynamic_graph(
        agent_registry=agent_registry,
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
        complete=False,
        limits=limits or WorkflowLimits(),
        radiology_progress_sink=radiology_progress_sink,
        radiology_cache=radiology_cache,
        baseline_store=baseline_store,
    )


def create_complete_mdt_graph(
    *,
    agent_registry: AgentRegistry,
    checkpointer: BaseCheckpointSaver,
    limits: WorkflowLimits | None = None,
    interrupt_after: Sequence[str] | None = None,
    radiology_progress_sink: RadiologyProgressSink | None = None,
    radiology_cache: RadiologyCache | None = None,
    baseline_store: BaselineStore | None = None,
):
    """Compile the Checkpoint 6 graph that always produces a FinalPlan."""

    return _create_dynamic_graph(
        agent_registry=agent_registry,
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
        complete=True,
        limits=limits or WorkflowLimits(),
        radiology_progress_sink=radiology_progress_sink,
        radiology_cache=radiology_cache,
        baseline_store=baseline_store,
    )


def _create_dynamic_graph(
    *,
    agent_registry: AgentRegistry,
    checkpointer: BaseCheckpointSaver,
    interrupt_after: Sequence[str] | None,
    complete: bool,
    limits: WorkflowLimits,
    radiology_progress_sink: RadiologyProgressSink | None,
    radiology_cache: RadiologyCache | None,
    baseline_store: BaselineStore | None,
):

    builder = StateGraph(MDTState, input_schema=OpeningGraphInput)
    _add_opening_nodes(
        builder,
        agent_registry=agent_registry,
        radiology_progress_sink=radiology_progress_sink,
        radiology_cache=radiology_cache,
        baseline_store=baseline_store,
    )

    async def moderator_route(state: MDTState) -> dict[str, Any]:
        if state["radiology_report"] is None:
            raise InvalidStateUpdateError("Dynamic routing requires a radiology report.")
        step_count = state["workflow_step_count"] + 1
        state_for_limits = dict(state)
        state_for_limits["workflow_step_count"] = step_count
        safety_issue = _unanswered_safety_warning(state) if complete else None
        safety_response_available = bool(
            complete and _safety_response_available(state, limits)
        )
        forced_reason = forced_termination_reason(
            state_for_limits,
            limits,
            extend_total_limit=_requires_extended_total_limit(state),
            allow_safety_response=safety_response_available,
        )
        if (
            safety_response_available
            and forced_reason is TerminationReason.NO_NEW_INFORMATION
        ):
            forced_reason = None
        mandatory_safety_response = (
            safety_response_available and forced_reason is None
        )
        last_specialist = _last_specialist(state["chat_log"])
        candidates = eligible_speakers(
            state["speaker_counts"],
            last_specialist,
            limits.max_turns_per_specialist,
        )
        if mandatory_safety_response:
            candidates = [AgentRole.SPINE_SURGEON]
        pass_triggered = bool(
            complete
            and not state["convergence_pass_done"]
            and forced_reason
            in {
                TerminationReason.ALL_SPEAKER_LIMITS,
                TerminationReason.TOTAL_SPEAKER_LIMIT,
                TerminationReason.NO_NEW_INFORMATION,
            }
        )
        pass_active = state["convergence_pass_active"] or (
            pass_triggered
            and state["working_plan"] is not None
            and state["working_plan"].surgical_indication
            and safety_issue is None
        )
        final_position_targets: list[AgentRole] = []
        if pass_active:
            total_turns = sum(state["speaker_counts"].values())
            final_position_targets = [
                role
                for role in FINAL_POSITION_ROLES
                if safety_issue is None
                and state["speaker_counts"].get(role.value, 0) > 0
                and not has_positioned(role, state["working_plan"])
                and total_turns < limits.max_absolute_specialist_turns
            ]
        if pass_active and not final_position_targets and forced_reason is None:
            forced_reason = TerminationReason.NO_NEW_INFORMATION
        proposed = None
        usage = None
        metadata = state["run_metadata"]
        routing_unresolved = list(state["unresolved_issues"])
        workflow_issues = list(state["workflow_issues"])
        if final_position_targets:
            forced_reason = None
            proposed = _final_position_decision(final_position_targets[0], state)
        if candidates and forced_reason is None and proposed is None:
            try:
                result = await agent_registry.moderator.route(
                    case_preop_md=state["case_preop_md"],
                    tests_preop_md=state["tests_preop_md"],
                    radiology_preop_md=state["radiology_preop_md"],
                    radiology_report=state["radiology_report"],
                    chat_log=state["chat_log"],
                    speaker_counts=state["speaker_counts"],
                    eligible_speakers=candidates,
                    pending_requests=state["pending_requests"],
                    unresolved_issues=state["unresolved_issues"],
                    disagreements=state["disagreements"],
                    last_specialist=last_specialist,
                    maximum_turns_per_specialist=(
                        limits.max_turns_per_specialist + 1
                        if mandatory_safety_response
                        else limits.max_turns_per_specialist
                    ),
                    available_image_ids=[image.image_id for image in state["images"]],
                    working_plan=state["working_plan"],
                )
                proposed = result.output
                usage = result.usage
                if result.attempt_failure_categories:
                    metadata, _ = _append_model_failure(
                        metadata,
                        case_id=state["case_id"],
                        role=AgentRole.MODERATOR,
                        stage=MODERATOR_ROUTING_STAGE,
                        outcome="recovered",
                        attempt_categories=result.attempt_failure_categories,
                        attempt_count=result.attempt_count,
                        max_attempts=result.max_attempts,
                    )
            except ModelInvocationError as exc:
                attempt_categories, attempt_count, max_attempts = _failure_trace(exc)
                metadata, issue = _append_model_failure(
                    metadata,
                    case_id=state["case_id"],
                    role=AgentRole.MODERATOR,
                    stage=MODERATOR_ROUTING_STAGE,
                    outcome="failed",
                    attempt_categories=attempt_categories,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                )
                routing_unresolved = merge_unique_strings(
                    routing_unresolved,
                    [issue],
                )
                workflow_issues = merge_unique_strings(workflow_issues, [issue])
                proposed = None
        if mandatory_safety_response:
            proposed = _safety_response_decision(proposed, safety_issue or "", state)
        if (
            complete
            and proposed is not None
            and proposed.action is ModeratorAction.FINALIZE
            and not normal_completion_ready(state)
        ):
            proposed = None

        if safety_issue is not None and not mandatory_safety_response:
            forced_reason = TerminationReason.UNANSWERED_SAFETY_WARNING

        convergence_pass_active = False
        convergence_pass_done = state["convergence_pass_done"]
        if mandatory_safety_response:
            if proposed is None:
                raise InvalidStateUpdateError("A reserved safety response requires routing.")
            decision = proposed
            accepted_request_id = None
            expired_request_ids = ()
            force_finalize = False
        elif final_position_targets:
            if proposed is None:
                raise InvalidStateUpdateError(
                    "A reserved final-position turn requires routing."
                )
            decision = proposed
            accepted_request_id = None
            expired_request_ids: Sequence[str] = ()
            force_finalize = False
            convergence_pass_active = True
        elif forced_reason is not None:
            decision = _forced_finalization_decision(
                forced_reason,
                state,
                additional_issues=[safety_issue] if safety_issue else [],
            )
            accepted_request_id = None
            expired_request_ids: Sequence[str] = ()
            force_finalize = True
            convergence_pass_done = True
        else:
            resolution = resolve_routing_decision(
                proposed=proposed,
                speaker_counts=state["speaker_counts"],
                pending_requests=state["pending_requests"],
                unresolved_issues=routing_unresolved,
                last_specialist=last_specialist,
                max_turns_per_specialist=limits.max_turns_per_specialist,
                available_image_ids=[image.image_id for image in state["images"]],
            )
            decision = resolution.decision
            accepted_request_id = resolution.accepted_request_id
            expired_request_ids = resolution.expired_request_ids
            force_finalize = resolution.force_finalize
        sequence = state["sequence_counter"]
        moderator_turn = _role_turn_number(state["chat_log"], AgentRole.MODERATOR) + 1
        message = ChatMessage(
            message_id=f"{state['case_id']}:moderator:{moderator_turn}",
            sequence=sequence,
            speaker_id=AgentRole.MODERATOR.value,
            speaker_name="MDT Moderator",
            speaker_role=AgentRole.MODERATOR,
            content=decision.public_message,
            mentions=[decision.next_speaker] if decision.next_speaker else [],
            turn_number=moderator_turn,
        )
        removed_request_ids = set(expired_request_ids)
        if accepted_request_id:
            removed_request_ids.add(accepted_request_id)
        pending = [
            request
            for request in state["pending_requests"]
            if request.request_id not in removed_request_ids
        ]
        if usage is not None:
            metadata = metadata.model_copy(
                update={"model_calls": [*metadata.model_calls, usage]}
            )
        finalizing = decision.action is ModeratorAction.FINALIZE
        if force_finalize:
            workflow_issues = merge_unique_strings(
                workflow_issues,
                [
                    f"Forced finalization triggered by "
                    f"{(forced_reason or TerminationReason.ALL_SPEAKER_LIMITS).value}."
                ],
            )
        unresolved_snapshot = issue_snapshot(
            decision.unresolved_issues,
            workflow_issues,
        )
        termination_reason = state["termination_reason"]
        if complete and finalizing:
            termination_reason = (
                forced_reason
                if force_finalize
                else TerminationReason.NORMAL_CONSENSUS
            )
        return {
            "phase": (
                RunPhase.AWAITING_FINALIZATION
                if finalizing
                else RunPhase.DYNAMIC_DISCUSSION
            ),
            "chat_log": [message],
            "sequence_counter": sequence + 1,
            "next_speaker": decision.next_speaker,
            "current_question": decision.question_for_next_speaker,
            "pending_requests": pending,
            "unresolved_issues": unresolved_snapshot,
            "workflow_issues": workflow_issues,
            "consensus_reached": decision.consensus_reached,
            "force_finalize": force_finalize,
            "moderator_decision": decision,
            "convergence_pass_active": convergence_pass_active,
            "convergence_pass_done": convergence_pass_done,
            "disagreements": _merge_disagreements(
                state["disagreements"],
                decision.disagreements,
            ),
            "termination_reason": termination_reason,
            "workflow_step_count": step_count,
            "run_metadata": metadata,
        }

    async def independent_specialist_round(state: MDTState) -> dict[str, Any]:
        if state["radiology_report"] is None:
            raise InvalidStateUpdateError(
                "Independent specialist sampling requires radiology evidence."
            )
        if any(
            state["speaker_counts"].get(role.value, 0) != 0
            for role in INITIAL_TEXT_SPECIALIST_ROLES
        ):
            raise InvalidStateUpdateError(
                "Independent specialist sampling requires unspoken text specialists."
            )

        allowed_ids = valid_source_ids(state)

        async def sample(role: AgentRole):
            agent = agent_registry.specialist(role)
            return await agent.run(
                case_preop_md=state["case_preop_md"],
                tests_preop_md=state["tests_preop_md"],
                radiology_preop_md=state["radiology_preop_md"],
                radiology_report=state["radiology_report"],
                chat_log=[],
                question=DEFAULT_QUESTIONS[role],
                prior_contribution=None,
                output_processor=lambda value: _normalize_text_specialist_output(
                    value,
                    role=role,
                    allowed_ids=allowed_ids,
                ),
            )

        tasks = [
            asyncio.create_task(
                sample(role),
                name=f"independent-initial-{role.value}",
            )
            for role in INITIAL_TEXT_SPECIALIST_ROLES
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        intro_sequence = state["sequence_counter"]
        moderator_turn = _role_turn_number(state["chat_log"], AgentRole.MODERATOR) + 1
        messages = [
            ChatMessage(
                message_id=f"{state['case_id']}:moderator-independent-round",
                sequence=intro_sequence,
                speaker_id=AgentRole.MODERATOR.value,
                speaker_name="MDT Moderator",
                speaker_role=AgentRole.MODERATOR,
                content=INDEPENDENT_ROUND_MESSAGE,
                mentions=list(INITIAL_TEXT_SPECIALIST_ROLES),
                turn_number=moderator_turn,
            )
        ]
        sequence = intro_sequence + 1
        counts = dict(state["speaker_counts"])
        pending = list(state["pending_requests"])
        contributions = list(state["specialist_contributions"])
        metadata = state["run_metadata"]
        workflow_issues = list(state["workflow_issues"])
        model_issues = list(state["unresolved_issues"])
        seen_fingerprints = list(state["seen_conclusion_fingerprints"])
        no_progress_streak = state["no_progress_streak"]
        any_failure = False
        working_plan = state["working_plan"]

        for role, result in zip(INITIAL_TEXT_SPECIALIST_ROLES, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, ModelInvocationError):
                    raise result
                any_failure = True
                attempt_categories, attempt_count, max_attempts = _failure_trace(result)
                metadata, issue = _append_model_failure(
                    metadata,
                    case_id=state["case_id"],
                    role=role,
                    stage=SPECIALIST_CONTRIBUTION_STAGE,
                    outcome="failed",
                    attempt_categories=attempt_categories,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                )
                workflow_issues = merge_unique_strings(workflow_issues, [issue])
                continue

            output = result.output
            agent = agent_registry.specialist(role)
            questions = list(output.questions_for_other_agents)
            turn_number = 1
            requests = _new_pending_requests(
                case_id=state["case_id"],
                requested_by=role,
                turn_number=turn_number,
                source_sequence=sequence,
                questions=questions,
            )
            pending = _append_unique_requests(pending, requests)
            messages.append(
                ChatMessage(
                    message_id=f"{state['case_id']}:{role.value}:{turn_number}",
                    sequence=sequence,
                    speaker_id=role.value,
                    speaker_name=agent.display_name,
                    speaker_role=role,
                    content=output.public_message,
                    mentions=_unique_roles(
                        question.target for question in questions
                    ),
                    reply_to=intro_sequence,
                    evidence_ids=evidence_ids_from_output(output),
                    confidence=output.confidence,
                    turn_number=turn_number,
                )
            )
            contributions.append(
                SpecialistContribution(
                    role=role,
                    turn_number=turn_number,
                    message_sequence=sequence,
                    structured_output=output,
                )
            )
            counts[role.value] = turn_number
            model_issues.extend(question.question for question in questions)
            if getattr(output, "need_retrieval", False):
                workflow_issues = merge_unique_strings(
                    workflow_issues,
                    [f"{role.value} requested medical retrieval, which is disabled."],
                )
            metadata = metadata.model_copy(
                update={
                    "specialist_message_count": metadata.specialist_message_count + 1,
                    "model_calls": [*metadata.model_calls, *result.all_usages],
                }
            )
            if result.attempt_failure_categories:
                metadata, _ = _append_model_failure(
                    metadata,
                    case_id=state["case_id"],
                    role=role,
                    stage=SPECIALIST_CONTRIBUTION_STAGE,
                    outcome="recovered",
                    attempt_categories=result.attempt_failure_categories,
                    attempt_count=result.attempt_count,
                    max_attempts=result.max_attempts,
                )
            seen_fingerprints, no_progress_streak = update_progress(
                seen_fingerprints=seen_fingerprints,
                no_progress_streak=no_progress_streak,
                output=output,
            )
            if role is AgentRole.SPINE_SURGEON:
                working_plan = project_plan_draft(
                    output,
                    source_sequence=sequence,
                    previous=state["working_plan"],
                )
            sequence += 1

        unresolved = issue_snapshot(model_issues, workflow_issues)
        update: dict[str, Any] = {
            "phase": (
                RunPhase.AWAITING_FINALIZATION
                if any_failure
                else RunPhase.AWAITING_DYNAMIC_ROUTING
            ),
            "chat_log": messages,
            "sequence_counter": sequence,
            "speaker_counts": counts,
            "pending_requests": pending,
            "specialist_contributions": contributions,
            "working_plan": working_plan,
            "unresolved_issues": unresolved,
            "workflow_issues": workflow_issues,
            "next_speaker": None,
            "current_question": None,
            "run_metadata": metadata,
            "seen_conclusion_fingerprints": seen_fingerprints,
            "no_progress_streak": no_progress_streak,
            "workflow_step_count": state["workflow_step_count"] + 1,
        }
        if any_failure:
            update.update(
                {
                    "force_finalize": True,
                    "termination_reason": TerminationReason.SPECIALIST_API_ERROR,
                }
            )
        return update

    def make_specialist_node(role: AgentRole) -> NodeCallable:
        async def specialist_node(
            state: MDTState,
            config: RunnableConfig,
        ) -> dict[str, Any]:
            current_count = state["speaker_counts"].get(role.value, 0)
            turn_limit = _specialist_turn_limit(state, role, limits)
            if current_count >= turn_limit:
                raise InvalidStateUpdateError(
                    f"{role.value} reached the configured speaking limit"
                )
            independent_first_turn = (
                role in INITIAL_TEXT_SPECIALIST_ROLES and current_count == 0
            )
            question = (
                DEFAULT_QUESTIONS[role]
                if independent_first_turn
                else state["current_question"]
            )
            if not question:
                raise InvalidStateUpdateError("A specialist turn requires a moderator question.")
            agent = agent_registry.specialist(role)
            allowed_ids = valid_source_ids(state)
            try:
                if role is AgentRole.RADIOLOGIST:
                    if state["radiology_report"] is None:
                        raise InvalidStateUpdateError(
                            "Targeted radiology review requires an existing report."
                        )
                    decision = state["moderator_decision"]
                    target_ids = list(decision.target_image_ids if decision else [])
                    by_id = {image.image_id: image for image in state["images"]}
                    if (
                        not 1 <= len(target_ids) <= 4
                        or len(target_ids) != len(set(target_ids))
                        or any(image_id not in by_id for image_id in target_ids)
                    ):
                        raise InvalidStateUpdateError(
                            "Radiologist follow-up requires 1-4 valid target_image_ids."
                        )
                    targeted_images = [by_id[image_id] for image_id in target_ids]
                    result = await agent_registry.radiologist.run_targeted(
                        case_summary_md=state["case_summary_md"],
                        images=targeted_images,
                        radiology_report=state["radiology_report"],
                        question=question,
                        target_image_ids=target_ids,
                        progress_callback=_bind_radiology_progress(
                            radiology_progress_sink,
                            config,
                        ),
                    )
                    addendum = result.output
                    output = merge_radiology_addendum(
                        state["radiology_report"],
                        addendum,
                    )
                    content = addendum.public_summary
                    confidence = addendum.confidence
                    evidence_ids = list(addendum.target_image_ids)
                    questions: list[AgentQuestion] = []
                else:
                    documents = source_documents_for_turn(state, role)
                    prior = (
                        None
                        if independent_first_turn
                        else latest_contribution(state, role)
                    )
                    turn_report = radiology_report_for_turn(state, role)
                    final_position_turn = bool(
                        state["convergence_pass_active"]
                        and role in FINAL_POSITION_ROLES
                    )
                    require_position = bool(
                        final_position_turn
                        or (
                            role
                            not in {
                                AgentRole.RADIOLOGIST,
                                AgentRole.SPINE_SURGEON,
                            }
                            and current_count > 0
                            and state["working_plan"] is not None
                        )
                    )
                    result = await agent.run(
                        case_preop_md=documents["case_preop_md"],
                        tests_preop_md=documents["tests_preop_md"],
                        radiology_preop_md=documents["radiology_preop_md"],
                        radiology_report=turn_report,
                        chat_log=(
                            []
                            if independent_first_turn
                            else relevant_chat_for_role(state, role)
                        ),
                        question=question,
                        prior_contribution=(
                            prior.structured_output if prior is not None else None
                        ),
                        working_plan=state["working_plan"],
                        output_processor=lambda value: _normalize_text_specialist_output(
                            value,
                            role=role,
                            allowed_ids=allowed_ids,
                            working_plan=state["working_plan"],
                            require_position=require_position,
                            forbid_new_questions=final_position_turn,
                        ),
                    )
                    output = result.output
                    content = output.public_message
                    confidence = output.confidence
                    evidence_ids = evidence_ids_from_output(output)
                    questions = list(output.questions_for_other_agents)
            except ModelInvocationError as exc:
                if not complete:
                    raise
                attempt_categories, attempt_count, max_attempts = _failure_trace(exc)
                metadata, issue = _append_model_failure(
                    state["run_metadata"],
                    case_id=state["case_id"],
                    role=role,
                    stage=SPECIALIST_CONTRIBUTION_STAGE,
                    outcome="failed",
                    attempt_categories=attempt_categories,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                )
                return {
                    "phase": RunPhase.AWAITING_FINALIZATION,
                    "force_finalize": True,
                    "termination_reason": TerminationReason.SPECIALIST_API_ERROR,
                    "unresolved_issues": merge_unique_strings(
                        state["unresolved_issues"],
                        [issue],
                    ),
                    "workflow_issues": merge_unique_strings(
                        state["workflow_issues"],
                        [issue],
                    ),
                    "next_speaker": None,
                    "current_question": None,
                    "workflow_step_count": state["workflow_step_count"] + 1,
                    "run_metadata": metadata,
                }
            turn_number = current_count + 1
            sequence = state["sequence_counter"]
            working_plan = state["working_plan"]
            if role is AgentRole.SPINE_SURGEON:
                working_plan = project_plan_draft(
                    output,
                    source_sequence=sequence,
                    previous=state["working_plan"],
                )
            elif (position := getattr(output, "position_on_plan", None)) is not None:
                draft = state["working_plan"]
                if draft is not None:
                    stamped = position.model_copy(
                        update={
                            "draft_version": draft.version,
                            "message_sequence": sequence,
                        }
                    )
                    working_plan = apply_position(draft, stamped)
            requests = _new_pending_requests(
                case_id=state["case_id"],
                requested_by=role,
                turn_number=turn_number,
                source_sequence=sequence,
                questions=questions,
            )
            message = ChatMessage(
                message_id=f"{state['case_id']}:{role.value}:{turn_number}",
                sequence=sequence,
                speaker_id=role.value,
                speaker_name=agent.display_name,
                speaker_role=role,
                content=content,
                mentions=_unique_roles(question.target for question in questions),
                reply_to=_last_moderator_sequence(state["chat_log"]),
                evidence_ids=evidence_ids,
                confidence=confidence,
                turn_number=turn_number,
            )
            contribution = SpecialistContribution(
                role=role,
                turn_number=turn_number,
                message_sequence=sequence,
                structured_output=output,
            )
            counts = dict(state["speaker_counts"])
            counts[role.value] = turn_number
            pending = _append_unique_requests(state["pending_requests"], requests)
            workflow_issues = list(state["workflow_issues"])
            unresolved = issue_snapshot(
                [
                    *state["unresolved_issues"],
                    *(question.question for question in questions),
                ],
                workflow_issues,
            )
            if getattr(output, "need_retrieval", False):
                retrieval_issue = (
                    f"{role.value} requested medical retrieval, which is disabled."
                )
                workflow_issues = merge_unique_strings(
                    workflow_issues,
                    [retrieval_issue],
                )
                unresolved = issue_snapshot(
                    unresolved,
                    workflow_issues,
                )
            metadata = state["run_metadata"].model_copy(
                update={
                    "specialist_message_count": (
                        state["run_metadata"].specialist_message_count + 1
                    ),
                    "model_calls": [
                        *state["run_metadata"].model_calls,
                        *result.all_usages,
                    ],
                }
            )
            if result.attempt_failure_categories:
                metadata, _ = _append_model_failure(
                    metadata,
                    case_id=state["case_id"],
                    role=role,
                    stage=SPECIALIST_CONTRIBUTION_STAGE,
                    outcome="recovered",
                    attempt_categories=result.attempt_failure_categories,
                    attempt_count=result.attempt_count,
                    max_attempts=result.max_attempts,
                )
            seen_fingerprints, no_progress_streak = update_progress(
                seen_fingerprints=state["seen_conclusion_fingerprints"],
                no_progress_streak=state["no_progress_streak"],
                output=output,
            )
            update: dict[str, Any] = {
                "phase": RunPhase.DYNAMIC_DISCUSSION,
                "chat_log": [message],
                "sequence_counter": sequence + 1,
                "speaker_counts": counts,
                "pending_requests": pending,
                "specialist_contributions": [
                    *state["specialist_contributions"],
                    contribution,
                ],
                "working_plan": working_plan,
                "unresolved_issues": unresolved,
                "workflow_issues": workflow_issues,
                "next_speaker": None,
                "current_question": None,
                "run_metadata": metadata,
                "seen_conclusion_fingerprints": seen_fingerprints,
                "no_progress_streak": no_progress_streak,
                "workflow_step_count": state["workflow_step_count"] + 1,
            }
            if role is AgentRole.RADIOLOGIST:
                radiology_preop_md = render_radiology_markdown(output)
                await write_markdown(
                    state["artifact_dir"],
                    "radiology_preop.md",
                    radiology_preop_md,
                )
                update["radiology_report"] = output
                update["radiology_preop_md"] = radiology_preop_md
                update["radiology_ready"] = True
            prospective = dict(state)
            prospective.update(update)
            safety_response_available = _safety_response_available(prospective, limits)
            reason = forced_termination_reason(
                prospective,
                limits,
                extend_total_limit=_requires_extended_total_limit(prospective),
                allow_safety_response=safety_response_available,
            )
            if reason is not None:
                defer_to_moderator = bool(
                    complete
                    and not prospective["convergence_pass_done"]
                    and reason
                    in {
                        TerminationReason.ALL_SPEAKER_LIMITS,
                        TerminationReason.TOTAL_SPEAKER_LIMIT,
                        TerminationReason.NO_NEW_INFORMATION,
                    }
                    and prospective["working_plan"] is not None
                    and prospective["working_plan"].surgical_indication
                )
                if defer_to_moderator:
                    update["phase"] = RunPhase.DYNAMIC_DISCUSSION
                    update["next_speaker"] = None
                    update["current_question"] = None
                else:
                    safety_issue = _unanswered_safety_warning(prospective)
                    forced_issue = f"Forced finalization triggered by {reason.value}."
                    update["workflow_issues"] = merge_unique_strings(
                        update["workflow_issues"],
                        [forced_issue],
                    )
                    if safety_issue is not None:
                        reason = TerminationReason.UNANSWERED_SAFETY_WARNING
                        update["workflow_issues"] = merge_unique_strings(
                            update["workflow_issues"],
                            [safety_issue],
                        )
                    update["unresolved_issues"] = issue_snapshot(
                        update["unresolved_issues"],
                        update["workflow_issues"],
                    )
                    update.update(
                        {
                            "phase": RunPhase.AWAITING_FINALIZATION,
                            "force_finalize": True,
                            "termination_reason": reason,
                            "next_speaker": None,
                            "current_question": None,
                        }
                    )
            return update

        return specialist_node

    def finalization_state(state: MDTState) -> tuple[TerminationReason, list[str], MDTState]:
        reason = state["termination_reason"] or TerminationReason.NORMAL_CONSENSUS
        workflow_issues = list(state["workflow_issues"])
        if state["force_finalize"]:
            workflow_issues = merge_unique_strings(
                workflow_issues,
                [f"Termination reason: {reason.value}"],
            )
        unresolved = issue_snapshot(state["unresolved_issues"], workflow_issues)
        state_for_plan = dict(state)
        state_for_plan["unresolved_issues"] = unresolved
        state_for_plan["workflow_issues"] = workflow_issues
        return reason, unresolved, state_for_plan

    async def draft_final_plan(state: MDTState) -> dict[str, Any]:
        if state["radiology_report"] is None:
            raise InvalidStateUpdateError("Final synthesis requires radiology evidence.")
        reason, unresolved, state_for_plan = finalization_state(state)
        allowed_ids = valid_source_ids(state)
        surgeon_contribution = latest_contribution(state, AgentRole.SPINE_SURGEON)
        surgeon_output = (
            surgeon_contribution.structured_output
            if surgeon_contribution is not None
            else None
        )

        def validate_draft(output):
            plan = normalize_final_plan_for_evidence(
                output,
                allowed_ids=allowed_ids,
                report=state["radiology_report"],
                surgeon_output=surgeon_output,
            )
            plan = apply_field_level_status(
                plan,
                review=None,
                report=state["radiology_report"],
                working_plan=state["working_plan"],
            )
            plan = validate_final_plan_evidence(plan, allowed_ids=allowed_ids)
            plan = apply_authoritative_workflow_fields(plan, state_for_plan)
            return validate_final_plan_integrity(plan, state_for_plan)

        try:
            result = await agent_registry.final_synthesis.draft(
                case_preop_md=state["case_preop_md"],
                tests_preop_md=state["tests_preop_md"],
                radiology_preop_md=state["radiology_preop_md"],
                radiology_report=state["radiology_report"],
                latest_contributions=latest_contributions(state),
                unresolved_issues=unresolved,
                disagreements=state["disagreements"],
                valid_evidence_ids=sorted(allowed_ids),
                force_finalize=state["force_finalize"],
                termination_reason=reason,
                output_processor=validate_draft,
            )
        except ModelInvocationError as exc:
            raise ModelStageError(FINAL_PLAN_GENERATION_STAGE, exc) from exc
        metadata = _metadata_with_result(
            state["run_metadata"],
            state["case_id"],
            result,
            FINAL_PLAN_GENERATION_STAGE,
        )
        return {
            "phase": RunPhase.AWAITING_FINALIZATION,
            "final_plan_draft": result.output,
            "unresolved_issues": unresolved,
            "termination_reason": reason,
            "workflow_step_count": state["workflow_step_count"] + 1,
            "run_metadata": metadata,
        }

    async def review_final_plan(state: MDTState) -> dict[str, Any]:
        if state["radiology_report"] is None or state["final_plan_draft"] is None:
            raise InvalidStateUpdateError("Final review requires evidence and a draft plan.")
        allowed_ids = valid_source_ids(state)
        try:
            result = await agent_registry.final_reviewer.review(
                case_preop_md=state["case_preop_md"],
                tests_preop_md=state["tests_preop_md"],
                radiology_preop_md=state["radiology_preop_md"],
                radiology_report=state["radiology_report"],
                draft=state["final_plan_draft"],
                valid_evidence_ids=sorted(allowed_ids),
                output_processor=lambda output: validate_review_evidence(
                    output,
                    allowed_ids=allowed_ids,
                ),
            )
        except ModelInvocationError as exc:
            raise ModelStageError(FINAL_PLAN_REVIEW_STAGE, exc) from exc
        review = result.output
        surgeon_contribution = latest_contribution(state, AgentRole.SPINE_SURGEON)
        surgeon_output = (
            surgeon_contribution.structured_output
            if surgeon_contribution is not None
            else None
        )
        approved = [
            field
            for field in review.established_field_paths
            if field in set(CRITICAL_OPERATIVE_FIELDS)
        ]
        definitive_eligible = bool(
            state["radiology_report"].image_quality is ImageQuality.DIAGNOSTIC
            and surgeon_output is not None
            and surgeon_output.surgical_indication
            and surgeon_output.operative_options
            and len(set(approved)) == len(CRITICAL_OPERATIVE_FIELDS)
            and review.definitive_surgery_allowed
            and not review.must_abstain_from_exact_surgery
        )
        review = review.model_copy(
            update={
                "established_field_paths": list(dict.fromkeys(approved)),
                "definitive_surgery_allowed": definitive_eligible,
                "must_abstain_from_exact_surgery": not definitive_eligible,
            }
        )
        draft = apply_field_level_status(
            state["final_plan_draft"],
            review=review,
            report=state["radiology_report"],
            working_plan=state["working_plan"],
        )
        metadata = _metadata_with_result(
            state["run_metadata"],
            state["case_id"],
            result,
            FINAL_PLAN_REVIEW_STAGE,
        )
        return {
            "final_plan_draft": draft,
            "final_plan_review": review,
            "workflow_step_count": state["workflow_step_count"] + 1,
            "run_metadata": metadata,
        }

    async def revise_final_plan(state: MDTState) -> dict[str, Any]:
        if (
            state["radiology_report"] is None
            or state["final_plan_draft"] is None
            or state["final_plan_review"] is None
        ):
            raise InvalidStateUpdateError("Final revision requires a draft and review.")
        _, _, state_for_plan = finalization_state(state)
        allowed_ids = valid_source_ids(state)
        surgeon_contribution = latest_contribution(state, AgentRole.SPINE_SURGEON)
        surgeon_output = (
            surgeon_contribution.structured_output
            if surgeon_contribution is not None
            else None
        )

        def validate_revision(output):
            plan = FinalPlan.model_validate(output)
            plan = normalize_final_plan_for_evidence(
                plan,
                allowed_ids=allowed_ids,
                report=state["radiology_report"],
                surgeon_output=surgeon_output,
            )
            plan = apply_field_level_status(
                plan,
                review=state["final_plan_review"],
                report=state["radiology_report"],
                working_plan=state["working_plan"],
            )
            plan = validate_final_plan_evidence(plan, allowed_ids=allowed_ids)
            plan = apply_authoritative_workflow_fields(plan, state_for_plan)
            return validate_final_plan_integrity(plan, state_for_plan)

        try:
            result = await agent_registry.final_synthesis.revise(
                case_preop_md=state["case_preop_md"],
                tests_preop_md=state["tests_preop_md"],
                radiology_preop_md=state["radiology_preop_md"],
                radiology_report=state["radiology_report"],
                draft=state["final_plan_draft"],
                review=state["final_plan_review"],
                output_processor=validate_revision,
            )
        except ModelInvocationError as exc:
            raise ModelStageError(FINAL_PLAN_REVISION_STAGE, exc) from exc
        metadata = _metadata_with_result(
            state["run_metadata"],
            state["case_id"],
            result,
            FINAL_PLAN_REVISION_STAGE,
        )
        return {
            "final_plan_draft": result.output,
            "workflow_step_count": state["workflow_step_count"] + 1,
            "run_metadata": metadata,
        }

    async def publish_final_plan(state: MDTState) -> dict[str, Any]:
        if state["final_plan_draft"] is None or state["final_plan_review"] is None:
            raise InvalidStateUpdateError("Publishing requires a reviewed final plan.")
        reason, unresolved, state_for_plan = finalization_state(state)
        plan = validate_final_plan_integrity(state["final_plan_draft"], state_for_plan)
        sequence = state["sequence_counter"]
        moderator_turn = _role_turn_number(state["chat_log"], AgentRole.MODERATOR) + 1
        message = ChatMessage(
            message_id=f"{state['case_id']}:moderator-final",
            sequence=sequence,
            speaker_id=AgentRole.MODERATOR.value,
            speaker_name="MDT Moderator",
            speaker_role=AgentRole.MODERATOR,
            content=plan.final_consensus.recommendation,
            mentions=[],
            evidence_ids=evidence_ids_from_output(plan),
            confidence=plan.confidence,
            turn_number=moderator_turn,
        )
        outcome = (
            "revised"
            if state["final_plan_review"].verdict == "revise"
            else "passed"
        )
        metadata = state["run_metadata"].model_copy(
            update={
                "completed_at": datetime.now(UTC),
                "forced_finalize_reason": reason.value if state["force_finalize"] else None,
                "finalization_fallback_used": False,
                "final_review_outcome": outcome,
            }
        )
        return {
            "phase": RunPhase.COMPLETED,
            "chat_log": [message],
            "sequence_counter": sequence + 1,
            "final_plan": plan,
            "unresolved_issues": unresolved,
            "termination_reason": reason,
            "next_speaker": None,
            "current_question": None,
            "workflow_step_count": state["workflow_step_count"] + 1,
            "run_metadata": metadata,
        }

    builder.add_node("independent_specialist_round", independent_specialist_round)
    builder.add_node("moderator_route", moderator_route)
    specialist_nodes = {
        AgentRole.RADIOLOGIST: "radiologist_dynamic",
        AgentRole.SPINE_SURGEON: "spine_surgeon",
        AgentRole.ANESTHESIOLOGIST: "anesthesiologist",
        AgentRole.NEUROLOGIST: "neurologist",
        AgentRole.PERIOPERATIVE_PHYSICIAN: "perioperative_physician",
    }
    for role, node_name in specialist_nodes.items():
        builder.add_node(node_name, make_specialist_node(role))
        if complete:
            builder.add_conditional_edges(
                node_name,
                lambda state: (
                    "finalize"
                    if state["phase"] is RunPhase.AWAITING_FINALIZATION
                    else "continue"
                ),
                {"continue": "moderator_route", "finalize": "draft_final_plan"},
            )
        else:
            builder.add_edge(node_name, "moderator_route")
    builder.add_edge("moderator_review", "independent_specialist_round")
    builder.add_conditional_edges(
        "independent_specialist_round",
        lambda state: (
            "finalize"
            if state["phase"] is RunPhase.AWAITING_FINALIZATION
            else "continue"
        ),
        {
            "continue": "moderator_route",
            "finalize": "draft_final_plan" if complete else END,
        },
    )
    if complete:
        builder.add_node("draft_final_plan", draft_final_plan)
        builder.add_node("review_final_plan", review_final_plan)
        builder.add_node("revise_final_plan", revise_final_plan)
        builder.add_node("publish_final_plan", publish_final_plan)
        builder.add_edge("draft_final_plan", "review_final_plan")
        builder.add_conditional_edges(
            "review_final_plan",
            lambda state: (
                "revise"
                if state["final_plan_review"] is not None
                and state["final_plan_review"].verdict == "revise"
                else "publish"
            ),
            {"revise": "revise_final_plan", "publish": "publish_final_plan"},
        )
        builder.add_edge("revise_final_plan", "publish_final_plan")
        builder.add_edge("publish_final_plan", END)
        builder.add_conditional_edges(
            "radiologist_initial_report",
            lambda state: (
                "finalize"
                if state["phase"] is RunPhase.AWAITING_FINALIZATION
                else "continue"
            ),
            {"continue": "moderator_review", "finalize": "draft_final_plan"},
        )
    else:
        builder.add_edge("radiologist_initial_report", "moderator_review")

    def route_after_moderator(state: MDTState) -> str:
        if state["phase"] is RunPhase.AWAITING_FINALIZATION:
            return "finalize"
        next_speaker = state["next_speaker"]
        if next_speaker not in specialist_nodes:
            raise InvalidStateUpdateError("Moderator selected an invalid next_speaker.")
        return next_speaker.value

    route_map = {role.value: node_name for role, node_name in specialist_nodes.items()}
    route_map["finalize"] = "draft_final_plan" if complete else END
    builder.add_conditional_edges("moderator_route", route_after_moderator, route_map)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=list(interrupt_after or []),
        name="spine-mdt-complete" if complete else "spine-mdt-dynamic",
    )


def _bind_radiology_progress(
    sink: RadiologyProgressSink | None,
    config: RunnableConfig,
) -> Callable[[RadiologyProgress], Awaitable[None]] | None:
    if sink is None:
        return None
    thread_id = str(config.get("configurable", {}).get("thread_id", "")).strip()
    if not thread_id:
        return None

    async def report(progress: RadiologyProgress) -> None:
        await sink(thread_id, progress)

    return report


def _new_pending_requests(
    *,
    case_id: str,
    requested_by: AgentRole,
    turn_number: int,
    source_sequence: int,
    questions: Sequence[AgentQuestion],
) -> list[PendingRequest]:
    return [
        PendingRequest(
            request_id=f"{case_id}:{requested_by.value}:{turn_number}:request:{index}",
            requested_by=requested_by,
            target=question.target,
            question=question.question,
            source_message_sequence=source_sequence,
            evidence_ids=question.evidence_ids,
        )
        for index, question in enumerate(questions, start=1)
    ]


def _append_unique_requests(
    existing: Sequence[PendingRequest],
    updates: Sequence[PendingRequest],
) -> list[PendingRequest]:
    result = list(existing)
    identifiers = {request.request_id for request in result}
    for request in updates:
        if request.request_id not in identifiers:
            result.append(request)
            identifiers.add(request.request_id)
    return result


def _last_specialist(chat_log: Sequence[ChatMessage]) -> AgentRole | None:
    for message in reversed(chat_log):
        if message.speaker_role is not AgentRole.MODERATOR:
            return message.speaker_role
    return None


def _last_moderator_sequence(chat_log: Sequence[ChatMessage]) -> int | None:
    for message in reversed(chat_log):
        if message.speaker_role is AgentRole.MODERATOR:
            return message.sequence
    return None


def _role_turn_number(chat_log: Sequence[ChatMessage], role: AgentRole) -> int:
    return sum(message.speaker_role is role for message in chat_log)


def _failure_trace(
    error: ModelInvocationError,
) -> tuple[tuple[ModelFailureCategory, ...], int, int]:
    if isinstance(error, ModelRetryExhaustedError):
        return (
            error.attempt_categories,
            error.attempt_count,
            error.max_attempts,
        )
    return ((model_failure_category(error),), 1, 1)


def _append_model_failure(
    metadata: RunMetadata,
    *,
    case_id: str,
    role: AgentRole,
    stage: str,
    outcome: str,
    attempt_categories: Sequence[ModelFailureCategory],
    attempt_count: int,
    max_attempts: int,
) -> tuple[RunMetadata, str]:
    category_values = [category.value for category in attempt_categories]
    final_category = category_values[-1]
    if outcome == "recovered":
        safe_message = (
            f"{role.value} model call at {stage} recovered on attempt "
            f"{attempt_count} of {max_attempts}; last failure category was "
            f"{final_category}."
        )
    else:
        safe_message = (
            f"{role.value} model call at {stage} failed after "
            f"{attempt_count} of {max_attempts} attempt(s); final category was "
            f"{final_category}."
        )
    record = ModelFailureRecord(
        failure_id=(
            f"{case_id}:model-failure:{len(metadata.model_failures) + 1:03d}"
        ),
        agent_role=role,
        stage=stage,
        outcome=outcome,
        category=final_category,
        attempt_categories=category_values,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        safe_message=safe_message,
    )
    return (
        metadata.model_copy(
            update={"model_failures": [*metadata.model_failures, record]}
        ),
        safe_message,
    )


def _metadata_with_result(
    metadata: RunMetadata,
    case_id: str,
    result,
    stage: str,
) -> RunMetadata:
    updated = metadata.model_copy(
        update={"model_calls": [*metadata.model_calls, *result.all_usages]}
    )
    if result.attempt_failure_categories:
        updated, _ = _append_model_failure(
            updated,
            case_id=case_id,
            role=AgentRole.MODERATOR,
            stage=stage,
            outcome="recovered",
            attempt_categories=result.attempt_failure_categories,
            attempt_count=result.attempt_count,
            max_attempts=result.max_attempts,
        )
    return updated


def _unique_roles(roles) -> list[AgentRole]:
    result: list[AgentRole] = []
    for role in roles:
        if role not in result:
            result.append(role)
    return result


def _merge_unique(*groups) -> list[str]:
    result: list[str] = []
    for group in groups:
        for item in group:
            if item not in result:
                result.append(item)
    return result


def _forced_finalization_decision(
    reason: TerminationReason,
    state: MDTState,
    *,
    additional_issues: Sequence[str] = (),
) -> ModeratorDecision:
    issue = f"Forced finalization triggered by {reason.value}."
    return ModeratorDecision(
        public_message=(
            "The discussion has reached a mandatory stopping condition. "
            "I will proceed to final synthesis while preserving unresolved limitations."
        ),
        action=ModeratorAction.FINALIZE,
        next_speaker=None,
        question_for_next_speaker=None,
        unresolved_issues=merge_unique_strings(
            state["unresolved_issues"],
            additional_issues,
            [issue],
        ),
        disagreements=state["disagreements"],
        consensus_reached=False,
        routing_reason=f"Mandatory termination condition: {reason.value}.",
        continue_discussion=False,
        enter_final_summary=True,
    )


def _safety_response_decision(
    proposed: ModeratorDecision | None,
    safety_issue: str,
    state: MDTState,
) -> ModeratorDecision:
    question = (
        "Please explicitly respond to the preoperative delay recommendation: state whether "
        "you accept it, the conditions required before proceeding, and any unresolved risk. "
        f"{safety_issue}"
    )
    return ModeratorDecision(
        public_message=f"Attending Spine Surgeon, {question}",
        action=ModeratorAction.ASK_SPECIALIST,
        next_speaker=AgentRole.SPINE_SURGEON,
        question_for_next_speaker=question,
        unresolved_issues=list(state["unresolved_issues"]),
        disagreements=(
            proposed.disagreements if proposed is not None else state["disagreements"]
        ),
        consensus_reached=False,
        routing_reason=(
            "A preoperative delay recommendation requires a later attending-surgeon response."
        ),
        continue_discussion=True,
        enter_final_summary=False,
    )


def _unanswered_safety_warning(state: MDTState) -> str | None:
    warnings: list[tuple[AgentRole, int, list[str]]] = []
    safety_roles = (
        (AgentRole.ANESTHESIOLOGIST, AnesthesiologistOutput),
        (AgentRole.PERIOPERATIVE_PHYSICIAN, PerioperativePhysicianOutput),
    )
    for role, output_type in safety_roles:
        contribution = _latest_role_contribution(state, role)
        if contribution is None or not isinstance(
            contribution.structured_output,
            output_type,
        ):
            continue
        output = contribution.structured_output
        if output.recommend_delay:
            warnings.append(
                (
                    role,
                    contribution.message_sequence,
                    output.delay_reasons or ["No reason was supplied."],
                )
            )
    if not warnings:
        return None

    latest_warning_sequence = max(sequence for _, sequence, _ in warnings)
    surgeon = _latest_role_contribution(state, AgentRole.SPINE_SURGEON)
    if surgeon is not None and surgeon.message_sequence > latest_warning_sequence:
        return None

    details = "; ".join(
        f"{role.value}: {', '.join(reasons)}" for role, _, reasons in warnings
    )
    return f"Unanswered preoperative delay recommendation: {details}"


def _requires_extended_total_limit(state: MDTState) -> bool:
    return bool(
        _unanswered_safety_warning(state)
        or state["disagreements"]
        or not state["consensus_reached"]
    )


def _safety_response_available(state: MDTState, limits: WorkflowLimits) -> bool:
    if _unanswered_safety_warning(state) is None:
        return False
    surgeon_turns = state["speaker_counts"].get(AgentRole.SPINE_SURGEON.value, 0)
    total_turns = sum(state["speaker_counts"].values())
    return bool(
        surgeon_turns < limits.max_turns_per_specialist + 1
        and total_turns < limits.max_absolute_specialist_turns
    )


def _specialist_turn_limit(
    state: MDTState,
    role: AgentRole,
    limits: WorkflowLimits,
) -> int:
    if role is AgentRole.SPINE_SURGEON and _safety_response_available(state, limits):
        return limits.max_turns_per_specialist + 1
    if state["convergence_pass_active"] and role in FINAL_POSITION_ROLES:
        return limits.max_turns_per_specialist + 1
    return limits.max_turns_per_specialist


def _latest_role_contribution(
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


def _merge_disagreements(
    existing: Sequence[Disagreement],
    updates: Sequence[Disagreement],
) -> list[Disagreement]:
    result = list(existing)
    keys = {
        (item.issue, tuple(sorted(role.value for role in item.agents))) for item in result
    }
    for item in updates:
        key = (item.issue, tuple(sorted(role.value for role in item.agents)))
        if key not in keys:
            result.append(item)
            keys.add(key)
    return result
