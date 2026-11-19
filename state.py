"""LangGraph shared state and deterministic reducers."""

from datetime import UTC, datetime
from typing import Annotated, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from exceptions import InvalidStateUpdateError
from schemas import (
    AgentRole,
    DirectorStyle,
    CaseImage,
    CaseInput,
    ChatMessage,
    DocumentPage,
    Disagreement,
    EvidenceRecord,
    FinalPlan,
    FinalPlanReview,
    ModeratorDecision,
    OperativePlanDraft,
    PendingRequest,
    ProviderUsage,
    RadiologyReport,
    RunMetadata,
    RunPhase,
    SpecialistContribution,
    TerminationReason,
)


CURRENT_WORKFLOW_VERSION = "v5"


SPECIALIST_ROLES: tuple[AgentRole, ...] = (
    AgentRole.RADIOLOGIST,
    AgentRole.SPINE_SURGEON,
    AgentRole.ANESTHESIOLOGIST,
    AgentRole.NEUROLOGIST,
    AgentRole.PERIOPERATIVE_PHYSICIAN,
)

CHECKPOINT_SCHEMA_ALLOWLIST: tuple[tuple[str, str], ...] = tuple(
    ("schemas", name)
    for name in (
        "Modality",
        "AgentRole",
        "DirectorStyle",
        "RunPhase",
        "ModeratorAction",
        "TerminationReason",
        "WorkflowLimits",
        "ImageQuality",
        "SagittalAlignment",
        "DynamicStability",
        "CompressionDistribution",
        "PosteriorCordDriftFeasibility",
        "FixationImagingSupport",
        "SurgicalApproach",
        "PlanningCertainty",
        "SurgicalOptionUncertainty",
        "ApproachSuitability",
        "DefinitiveSurgeryStatus",
        "SurgeryFieldStatus",
        "RiskLevel",
        "ConsistencyAssessment",
        "CaseImage",
        "DocumentPage",
        "CaseInput",
        "PreparedMarkdown",
        "CaseDocumentMarkdown",
        "TestDocumentMarkdown",
        "CaseSummaryMarkdown",
        "PreparedDocuments",
        "ChatMessage",
        "ImageLinkedFinding",
        "ClaimEvidence",
        "RadiologyAddendum",
        "OperativePlanningAssessment",
        "SurgicalOption",
        "ApproachAssessment",
        "ApproachComparison",
        "RadiologyReport",
        "AgentQuestion",
        "SpineSurgeonOutput",
        "AnesthesiologistOutput",
        "NeurologistOutput",
        "PerioperativePhysicianOutput",
        "PendingRequest",
        "SpecialistContribution",
        "Disagreement",
        "PlanPositionAction",
        "PlanPosition",
        "OperativePlanDraft",
        "ModeratorDecision",
        "EvidenceRecord",
        "RadiologySummary",
        "DiagnosisEvidence",
        "SurgicalIndicationPlan",
        "RecommendedSurgery",
        "AlternativePlan",
        "PerioperativeRiskSummary",
        "AdditionalTests",
        "DisagreementResolution",
        "FinalConsensus",
        "FinalPlan",
        "FinalReviewIssue",
        "FinalPlanReview",
        "ProviderUsage",
        "ModelFailureRecord",
        "RunMetadata",
        "CaseBaseline",
    )
) + tuple(
    # Department-hierarchy workflow. An unlisted type is not rejected on load --
    # it silently comes back as a plain dict, so an identity check against an
    # enum member would quietly start returning False after a resume.
    ("department.schemas", name)
    for name in (
        "LevelStatus",
        "CompressionType",
        "StenosisGrade",
        "SignalChange",
        "SymptomImagingCorrelation",
        "ImagingFindings",
        "ResidentReport",
        "LocalizationMatch",
        "FlagRelevance",
        "RedFlag",
        "NeurologyAssessment",
        "AsaGrade",
        "RiskBand",
        "AnesthesiaAssessment",
        "ConsensusOutcome",
        "ConservativeStance",
        "RehabAssessment",
        "ViceDirectorOpinion",
        "DirectorDecision",
        "SurgicalIndication",
        "FamilyRequest",
        "DirectorRuling",
        "RoleModelAssignment",
        "RoleUsage",
        "DepartmentBaselinePayload",
    )
) + (
    ("department.knowledge", "KnowledgeExcerpt"),
    ("department.observer", "ObserverMetrics"),
    ("department.state", "DepartmentPhase"),
    ("department.state", "TerminationCause"),
)


def create_checkpoint_serializer() -> JsonPlusSerializer:
    """Return a no-pickle serializer restricted to application schema types."""

    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_msgpack_modules=CHECKPOINT_SCHEMA_ALLOWLIST,
    )


def append_unique_messages(
    existing: list[ChatMessage] | None,
    updates: list[ChatMessage | dict] | None,
) -> list[ChatMessage]:
    """Append messages while making checkpoint replay idempotent."""

    result = [ChatMessage.model_validate(item) for item in (existing or [])]
    by_id = {message.message_id: message for message in result}
    by_sequence = {message.sequence: message for message in result}

    for raw_update in updates or []:
        update = ChatMessage.model_validate(raw_update)
        matching_id = by_id.get(update.message_id)
        matching_sequence = by_sequence.get(update.sequence)
        if matching_id is not None:
            if matching_id != update:
                raise InvalidStateUpdateError(
                    f"message_id collision for {update.message_id!r}"
                )
            continue
        if matching_sequence is not None:
            raise InvalidStateUpdateError(
                f"message sequence collision at {update.sequence}"
            )
        if result and update.sequence <= result[-1].sequence:
            raise InvalidStateUpdateError(
                "chat messages must be appended in increasing sequence order"
            )
        result.append(update)
        by_id[update.message_id] = update
        by_sequence[update.sequence] = update
    return result


def append_unique_strings(
    existing: list[str] | None,
    updates: list[str] | None,
) -> list[str]:
    result = list(existing or [])
    for update in updates or []:
        if update not in result:
            result.append(update)
    return result


class MDTState(TypedDict):
    workflow_version: str
    case_id: str
    director_style: DirectorStyle | None
    case_pages: list[DocumentPage]
    test_pages: list[DocumentPage]
    images: list[CaseImage]
    artifact_dir: str
    case_preop_md: str
    tests_preop_md: str
    case_summary_md: str
    radiology_preop_md: str
    preparation_warnings: Annotated[list[str], append_unique_strings]
    case_document_usages: list[ProviderUsage]
    test_document_usages: list[ProviderUsage]
    phase: RunPhase
    radiology_report: RadiologyReport | None
    radiology_ready: bool
    chat_log: Annotated[list[ChatMessage], append_unique_messages]
    sequence_counter: int
    speaker_counts: dict[str, int]
    next_speaker: AgentRole | None
    current_question: str | None
    pending_requests: list[PendingRequest]
    specialist_contributions: list[SpecialistContribution]
    working_plan: OperativePlanDraft | None
    convergence_pass_active: bool
    convergence_pass_done: bool
    unresolved_issues: list[str]
    workflow_issues: list[str]
    consensus_reached: bool
    force_finalize: bool
    workflow_step_count: int
    no_progress_streak: int
    seen_conclusion_fingerprints: list[str]
    termination_reason: TerminationReason | None
    disagreements: list[Disagreement]
    retrieved_evidence: list[EvidenceRecord]
    retrieval_count: int
    moderator_decision: ModeratorDecision | None
    final_plan: FinalPlan | None
    final_plan_draft: FinalPlan | None
    final_plan_review: FinalPlanReview | None
    run_metadata: RunMetadata


def initial_speaker_counts() -> dict[str, int]:
    return {role.value: 0 for role in SPECIALIST_ROLES}


def create_initial_state(
    case: CaseInput,
    *,
    director_style: DirectorStyle | None = None,
) -> MDTState:
    """Create a complete, serializable state from a validated case.

    ``director_style`` is the experiment arm. No role consumes it yet; it is
    carried through state, checkpoints, snapshots, and events so that a run can
    be attributed to its arm from the moment it starts.
    """

    return MDTState(
        workflow_version=CURRENT_WORKFLOW_VERSION,
        case_id=case.case_id,
        director_style=director_style,
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        artifact_dir=case.artifact_dir,
        case_preop_md="",
        tests_preop_md="",
        case_summary_md="",
        radiology_preop_md="",
        preparation_warnings=[],
        case_document_usages=[],
        test_document_usages=[],
        phase=RunPhase.CREATED,
        radiology_report=None,
        radiology_ready=False,
        chat_log=[],
        sequence_counter=1,
        speaker_counts=initial_speaker_counts(),
        next_speaker=None,
        current_question=None,
        pending_requests=[],
        specialist_contributions=[],
        working_plan=None,
        convergence_pass_active=False,
        convergence_pass_done=False,
        unresolved_issues=[],
        workflow_issues=[],
        consensus_reached=False,
        force_finalize=False,
        workflow_step_count=0,
        no_progress_streak=0,
        seen_conclusion_fingerprints=[],
        termination_reason=None,
        disagreements=[],
        retrieved_evidence=[],
        retrieval_count=0,
        moderator_decision=None,
        final_plan=None,
        final_plan_draft=None,
        final_plan_review=None,
        run_metadata=RunMetadata(
            started_at=datetime.now(UTC),
            completed_at=None,
            specialist_message_count=0,
            retrieval_count=0,
            model_calls=[],
            forced_finalize_reason=None,
            finalization_fallback_used=False,
        ),
    )
