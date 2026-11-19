"""Validated public contracts for the Spine MDT workflow."""

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
MISSING_RECORDS = "Not established from available records"


class StrictModel(BaseModel):
    """Base contract that rejects unexpected provider output fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Modality(str, Enum):
    XRAY = "XRAY"
    CT = "CT"
    MRI = "MRI"


class AgentRole(str, Enum):
    MODERATOR = "moderator"
    RADIOLOGIST = "radiologist"
    SPINE_SURGEON = "spine_surgeon"
    ANESTHESIOLOGIST = "anesthesiologist"
    NEUROLOGIST = "neurologist"
    PERIOPERATIVE_PHYSICIAN = "perioperative_physician"
    # Department-hierarchy roles. The radiologist, neurologist, and
    # anesthesiologist above are reused there with their own thin schemas.
    RESIDENT = "resident"
    PHYSIATRIST = "physiatrist"
    VICE_DIRECTOR = "vice_director"
    DIRECTOR = "director"
    # Preparation only: reads the unredacted record to find what states the
    # answer. Never a participant in any meeting.
    RECORD_REDACTOR = "record_redactor"
    # Analysis only: reads a finished run's rulings to code how they handled the
    # neurologist's red flags. Never a participant, never told the arm.
    OUTCOME_JUDGE = "outcome_judge"


class DirectorStyle(str, Enum):
    """Which department decision paradigm a run is simulating.

    This is the experiment's single manipulated variable: the same case is run
    once per style and any systematic difference in the outcome is attributed to
    the style rather than to the patient or to an individual clinician. It is a
    property of the run, never of the case, so it deliberately does not live on
    ``CaseInput`` and never enters a reusable baseline.

    ``NEUTRAL`` is the control arm (protocol amendment A2): the director and the
    deputy are told no style is prescribed and are given no rules of conduct.
    Without it, the three styles can only be compared with one another, never
    with what the department does when nobody tells it how to behave.
    """

    DEMOCRATIC = "democratic"
    AUTOCRATIC = "autocratic"
    AVOIDANT = "avoidant"
    NEUTRAL = "neutral"


class RunPhase(str, Enum):
    CREATED = "created"
    INPUT_NORMALIZED = "input_normalized"
    CASE_DOCUMENT_PREPARATION = "case_document_preparation"
    DOCUMENT_PREPARATION = "document_preparation"
    CASE_SUMMARIZATION = "case_summarization"
    MODERATOR_OPENING = "moderator_opening"
    RADIOLOGY_REVIEW = "radiology_review"
    INDEPENDENT_SPECIALIST_ROUND = "independent_specialist_round"
    AWAITING_DYNAMIC_ROUTING = "awaiting_dynamic_routing"
    DYNAMIC_DISCUSSION = "dynamic_discussion"
    AWAITING_FINALIZATION = "awaiting_finalization"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModeratorAction(str, Enum):
    ASK_SPECIALIST = "ask_specialist"
    FINALIZE = "finalize"


class TerminationReason(str, Enum):
    NORMAL_CONSENSUS = "normal_consensus"
    ALL_SPEAKER_LIMITS = "all_speaker_limits"
    TOTAL_SPEAKER_LIMIT = "total_speaker_limit"
    GLOBAL_STEP_LIMIT = "global_step_limit"
    NO_NEW_INFORMATION = "no_new_information"
    SPECIALIST_API_ERROR = "specialist_api_error"
    UNANSWERED_SAFETY_WARNING = "unanswered_safety_warning"


class ImageQuality(str, Enum):
    DIAGNOSTIC = "diagnostic"
    LIMITED = "limited"
    NONDIAGNOSTIC = "nondiagnostic"
    UNKNOWN = "unknown"


class SagittalAlignment(str, Enum):
    LORDOTIC = "lordotic"
    STRAIGHTENED = "straightened"
    NEUTRAL = "neutral"
    KYPHOTIC = "kyphotic"
    INDETERMINATE = "indeterminate"


class DynamicStability(str, Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    NOT_ASSESSED = "not_assessed"
    INDETERMINATE = "indeterminate"


class CompressionDistribution(str, Enum):
    VENTRAL = "ventral"
    DORSAL = "dorsal"
    CIRCUMFERENTIAL = "circumferential"
    MIXED = "mixed"
    INDETERMINATE = "indeterminate"


class PosteriorCordDriftFeasibility(str, Enum):
    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    INDETERMINATE = "indeterminate"


class FixationImagingSupport(str, Enum):
    SUPPORTS = "supports"
    DOES_NOT_SUPPORT = "does_not_support"
    INDETERMINATE = "indeterminate"
    NOT_ASSESSED = "not_assessed"


class SurgicalApproach(str, Enum):
    ANTERIOR = "anterior"
    POSTERIOR = "posterior"
    COMBINED = "combined"


class PlanningCertainty(str, Enum):
    ESTABLISHED = "established"
    PROVISIONAL = "provisional"
    NOT_ESTABLISHED = "not_established"


class SurgicalOptionUncertainty(str, Enum):
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ApproachSuitability(str, Enum):
    FAVORED = "favored"
    REASONABLE = "reasonable"
    DISFAVORED = "disfavored"
    INDETERMINATE = "indeterminate"


class DefinitiveSurgeryStatus(str, Enum):
    ESTABLISHED = "established"
    PARTIAL = "partial"
    WITHHELD = "withheld"


class SurgeryFieldStatus(str, Enum):
    """Per-field precision of one recommended-surgery operative field."""

    ESTABLISHED = "established"
    CONDITIONAL = "conditional"
    WITHHELD = "withheld"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class WorkflowLimits(StrictModel):
    max_turns_per_specialist: int = Field(default=5, ge=1, le=10)
    max_total_specialist_turns: int = Field(default=20, ge=1, le=100)
    max_absolute_specialist_turns: int = Field(default=24, ge=1, le=100)
    max_workflow_steps: int = Field(default=50, ge=1, le=1000)
    max_no_progress_turns: int = Field(default=2, ge=1, le=20)

    @model_validator(mode="after")
    def absolute_limit_must_cover_normal_limit(self) -> "WorkflowLimits":
        if self.max_absolute_specialist_turns < self.max_total_specialist_turns:
            raise ValueError(
                "max_absolute_specialist_turns must be greater than or equal to "
                "max_total_specialist_turns"
            )
        return self


class ConsistencyAssessment(str, Enum):
    CONSISTENT = "consistent"
    PARTIALLY_CONSISTENT = "partially_consistent"
    INCONSISTENT = "inconsistent"
    UNCERTAIN = "uncertain"


class DocumentPage(StrictModel):
    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    path_or_url: str = Field(min_length=1)
    original_filename: str = Field(min_length=1)


class CaseImage(StrictModel):
    image_id: str = Field(min_length=1)
    modality: Modality
    path_or_url: str = Field(min_length=1)
    original_filename: str = Field(default="uploaded-image", min_length=1)
    body_region: str | None = None
    view_or_plane: str | None = None
    sequence: str | None = None
    level: str | None = None
    description: str | None = None


VisualInput = CaseImage | DocumentPage


class CaseInput(StrictModel):
    case_id: str = Field(min_length=1)
    case_pages: list[DocumentPage] = Field(min_length=1)
    test_pages: list[DocumentPage] = Field(min_length=1)
    images: list[CaseImage] = Field(min_length=1)
    artifact_dir: str = Field(min_length=1)

    @field_validator("case_pages", "test_pages")
    @classmethod
    def ordered_unique_pages(cls, pages: list[DocumentPage]) -> list[DocumentPage]:
        identifiers = [page.page_id for page in pages]
        numbers = [page.page_number for page in pages]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("page_id values must be unique within a document")
        if numbers != list(range(1, len(pages) + 1)):
            raise ValueError("page_number values must be contiguous and ordered from 1")
        return pages

    @field_validator("images")
    @classmethod
    def unique_image_ids(cls, images: list[CaseImage]) -> list[CaseImage]:
        identifiers = [image.image_id for image in images]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("image_id values must be unique within a case")
        return images


class PreparedMarkdown(StrictModel):
    markdown: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class CaseDocumentMarkdown(PreparedMarkdown):
    pass


class TestDocumentMarkdown(PreparedMarkdown):
    pass


class CaseSummaryMarkdown(PreparedMarkdown):
    pass


class PreparedDocuments(StrictModel):
    case_preop_md: str = ""
    tests_preop_md: str = ""
    case_summary_md: str = ""
    radiology_preop_md: str = ""
    warnings: list[str] = Field(default_factory=list)


class ChatMessage(StrictModel):
    message_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    speaker_id: str = Field(min_length=1)
    speaker_name: str = Field(min_length=1)
    speaker_role: AgentRole
    content: str = Field(min_length=1)
    mentions: list[AgentRole] = Field(default_factory=list)
    reply_to: int | None = Field(default=None, ge=1)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Confidence | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    turn_number: int = Field(ge=1)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)


class ImageLinkedFinding(StrictModel):
    finding: str = Field(min_length=1)
    image_ids: list[str] = Field(min_length=1)


ClaimSupportType = Literal["documented", "inference", "proposal"]


class ClaimEvidence(StrictModel):
    """Auditable support for a structured or public clinical claim."""

    field_path: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    support_type: ClaimSupportType
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def observed_claims_require_sources(self) -> "ClaimEvidence":
        if self.support_type in {"documented", "inference"} and not self.evidence_ids:
            raise ValueError("documented and inference claims require evidence_ids")
        return self


class AgentQuestion(StrictModel):
    target: AgentRole
    question: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def target_must_be_specialist(self) -> "AgentQuestion":
        if self.target is AgentRole.MODERATOR:
            raise ValueError("a specialist question cannot target the moderator")
        return self


class RadiologyAddendum(StrictModel):
    target_image_ids: list[str] = Field(min_length=1, max_length=4)
    clarifications: list[ImageLinkedFinding] = Field(default_factory=list)
    uncertain_findings: list[str] = Field(default_factory=list)
    missing_views_or_sequences: list[str] = Field(default_factory=list)
    public_summary: str = Field(min_length=1)
    confidence: Confidence

    @model_validator(mode="after")
    def findings_must_reference_targeted_images(self) -> "RadiologyAddendum":
        targets = set(self.target_image_ids)
        if len(targets) != len(self.target_image_ids):
            raise ValueError("radiology addendum target_image_ids must be unique")
        if any(not set(item.image_ids) <= targets for item in self.clarifications):
            raise ValueError("radiology addendum findings must use targeted image_ids")
        return self


class OperativePlanningAssessment(StrictModel):
    """Imaging factors relevant to planning, without making the surgical decision."""

    sagittal_alignment: SagittalAlignment = SagittalAlignment.INDETERMINATE
    dynamic_stability: DynamicStability = DynamicStability.NOT_ASSESSED
    compression_distribution: CompressionDistribution = (
        CompressionDistribution.INDETERMINATE
    )
    posterior_cord_drift_feasibility: PosteriorCordDriftFeasibility = (
        PosteriorCordDriftFeasibility.INDETERMINATE
    )
    fixation_imaging_support: FixationImagingSupport = (
        FixationImagingSupport.NOT_ASSESSED
    )
    planning_uncertainties: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class SurgicalOption(StrictModel):
    option_id: str = Field(min_length=1)
    rank: int = Field(ge=1, le=3)
    procedure_name: str = Field(min_length=1)
    approach: SurgicalApproach
    levels: list[str] = Field(default_factory=list)
    levels_certainty: PlanningCertainty
    decompression_extent: str = Field(min_length=1)
    fusion_or_fixation: str = Field(min_length=1)
    fixation_certainty: PlanningCertainty
    rationale: list[str] = Field(min_length=1)
    selection_conditions: list[str] = Field(default_factory=list)
    reasons_against: list[str] = Field(default_factory=list)
    required_confirmation: list[str] = Field(default_factory=list)
    uncertainty: SurgicalOptionUncertainty
    confidence: Confidence
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def certainty_matches_levels(self) -> "SurgicalOption":
        if self.procedure_name == MISSING_RECORDS:
            raise ValueError("a surgical option requires a concrete procedure_name")
        if self.levels_certainty is PlanningCertainty.NOT_ESTABLISHED and self.levels:
            raise ValueError("not_established levels must be empty")
        if self.levels_certainty is not PlanningCertainty.NOT_ESTABLISHED and not self.levels:
            raise ValueError("established or provisional levels must be supplied")
        return self


class ApproachAssessment(StrictModel):
    approach: SurgicalApproach
    suitability: ApproachSuitability
    supporting_reasons: list[str] = Field(default_factory=list)
    limiting_factors: list[str] = Field(default_factory=list)
    required_confirmation: list[str] = Field(default_factory=list)
    confidence: Confidence
    evidence_ids: list[str] = Field(min_length=1)


class ApproachComparison(StrictModel):
    assessments: list[ApproachAssessment] = Field(min_length=3, max_length=3)
    preferred_approach: SurgicalApproach
    preference_rationale: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def includes_each_required_approach_once(self) -> "ApproachComparison":
        approaches = [item.approach for item in self.assessments]
        if len(approaches) != len(set(approaches)):
            raise ValueError("approach assessments must be unique")
        if set(approaches) != set(SurgicalApproach):
            raise ValueError("anterior, posterior, and combined assessments are required")
        return self


class PlanPositionAction(str, Enum):
    SUPPORT = "support"
    SUPPORT_WITH_RESERVATIONS = "support_with_reservations"
    OPPOSE = "oppose"
    ABSTAIN = "abstain"


class PlanPosition(StrictModel):
    """A specialist's incremental stance on the shared operative working draft.

    ``draft_version`` and ``message_sequence`` are workflow-owned: the model
    cannot know them, so they are stamped deterministically before the position
    is attached to the draft.
    """

    role: AgentRole
    option_id: str = Field(min_length=1)
    draft_version: int | None = Field(default=None, ge=1)
    action: PlanPositionAction
    target_fields: list[str] = Field(default_factory=list)
    reservations: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Confidence
    message_sequence: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def position_contract(self) -> "PlanPosition":
        if self.role in {AgentRole.MODERATOR, AgentRole.SPINE_SURGEON}:
            raise ValueError(
                "positions cannot be attributed to the moderator or the draft owner"
            )
        if self.action in {
            PlanPositionAction.SUPPORT_WITH_RESERVATIONS,
            PlanPositionAction.OPPOSE,
        } and not self.reservations:
            raise ValueError(
                "support_with_reservations and oppose positions require reservations"
            )
        if self.action is not PlanPositionAction.ABSTAIN and not self.evidence_ids:
            raise ValueError(
                "support and oppose positions require at least one evidence id"
            )
        return self


class RadiologyReport(StrictModel):
    image_quality: ImageQuality
    available_modalities: list[Modality]
    xray_findings: list[ImageLinkedFinding]
    ct_findings: list[ImageLinkedFinding]
    mri_findings: list[ImageLinkedFinding]
    spine_levels: list[str]
    alignment: str
    bony_structures: str
    canal_and_foramina: str
    neural_compression: str
    cross_modality_synthesis: str
    possible_diagnoses: list[str]
    uncertain_findings: list[str]
    missing_views_or_sequences: list[str]
    public_summary: str = Field(min_length=1)
    confidence: Confidence
    operative_planning: OperativePlanningAssessment = Field(
        default_factory=OperativePlanningAssessment
    )
    questions_for_other_agents: list[AgentQuestion] = Field(default_factory=list)
    claim_evidence: list[ClaimEvidence] = Field(default_factory=list)
    addenda: list[RadiologyAddendum] = Field(default_factory=list)


class CaseBaseline(StrictModel):
    """The frozen output of preparation for one case, reusable across runs.

    The three discussion arms of an experiment must argue over identical
    evidence, so preparation runs once and every arm loads this record instead
    of regenerating it. Reloading is refused unless the run presents the same
    source bytes and the same page and image identifiers, because the radiology
    report cites those identifiers directly.

    ``format_version`` records which payloads a stored record carries. It is
    deliberately *not* part of the storage key: the key answers "are these the
    same source files?", and that answer does not change when the record gains a
    field. Salting the key with it would orphan every existing baseline on each
    format change and silently trigger paid regeneration.
    """

    format_version: Literal["v1", "v2"] = "v2"
    baseline_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str = Field(min_length=1)
    case_page_ids: list[str] = Field(min_length=1)
    # Empty only for a text-only case -- a published case report, which has a
    # record and nothing else. The upload path still requires all three.
    test_page_ids: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)
    case_preop_md: str = Field(min_length=1)
    tests_preop_md: str = Field(min_length=1)
    case_summary_md: str = Field(min_length=1)
    radiology_preop_md: str = Field(min_length=1)
    preparation_warnings: list[str] = Field(default_factory=list)
    # Absent when only the department workflow has prepared these sources: its
    # imaging schema is a different, thinner shape, and inventing a v5 report to
    # fill the slot would put fabricated findings into the record.
    radiology_report: RadiologyReport | None = None
    # Stage I-II of the department workflow, absent in a record written before
    # that workflow existed. A department run regenerates and upgrades in place
    # rather than starting a second record for the same sources. It is held as
    # a plain mapping here so that this module does not import the department
    # package; the department side validates it into its own model.
    department: dict | None = None
    #: The signed redaction the case record was cleaned with. Without it there is
    #: no way to show afterwards which version of the record the arms read, or
    #: that a person looked at what was removed.
    redaction_sha: str | None = None
    created_at: datetime


class SpineSurgeonOutput(StrictModel):
    public_message: str = Field(min_length=1)
    surgical_indication: bool
    indication_rationale: list[str]
    functional_impact: str = Field(min_length=1)
    conservative_treatment_review: list[str]
    preoperative_prerequisites: list[str]
    recommended_procedure: str
    surgical_levels: list[str]
    approach: str
    decompression_required: bool
    fusion_required: bool
    fixation_required: bool
    alternative_plans: list[str]
    operative_options: list[SurgicalOption] = Field(default_factory=list, max_length=3)
    preferred_option_id: str | None = None
    approach_comparison: ApproachComparison | None = None
    major_surgical_risks: list[str]
    questions_for_other_agents: list[AgentQuestion]
    position_on_plan: PlanPosition | None = None
    need_retrieval: bool = False
    retrieval_query: str | None = None
    confidence: Confidence
    claim_evidence: list[ClaimEvidence] = Field(default_factory=list)


class AnesthesiologistOutput(StrictModel):
    public_message: str = Field(min_length=1)
    anesthesia_risk: RiskLevel
    asa_classification: str = Field(min_length=1)
    proposed_anesthesia_plan: str = Field(min_length=1)
    airway_risk: str
    positioning_risk: str
    cardiopulmonary_risk: str
    blood_loss_risk: str
    monitoring_recommendations: list[str]
    transfusion_preparation: list[str]
    preoperative_optimization: list[str]
    recommend_delay: bool
    delay_reasons: list[str]
    questions_for_other_agents: list[AgentQuestion]
    position_on_plan: PlanPosition | None = None
    need_retrieval: bool = False
    retrieval_query: str | None = None
    confidence: Confidence
    claim_evidence: list[ClaimEvidence] = Field(default_factory=list)


class NeurologistOutput(StrictModel):
    public_message: str = Field(min_length=1)
    neurologic_localization: str
    motor_function: str
    sensory_function: str
    reflexes: str
    sphincter_function: str
    symptom_imaging_consistency: ConsistencyAssessment
    neurologic_deterioration_risk: RiskLevel
    additional_tests: list[str]
    preoperative_neurologic_baseline: str
    questions_for_other_agents: list[AgentQuestion]
    position_on_plan: PlanPosition | None = None
    need_retrieval: bool = False
    retrieval_query: str | None = None
    confidence: Confidence
    claim_evidence: list[ClaimEvidence] = Field(default_factory=list)


class PerioperativePhysicianOutput(StrictModel):
    public_message: str = Field(min_length=1)
    major_comorbidities: list[str]
    cardiopulmonary_risk: RiskLevel
    infection_risk: RiskLevel
    thrombosis_risk: RiskLevel
    nutritional_risk: RiskLevel
    hepatic_renal_risk: RiskLevel
    medication_adjustments: list[str]
    preoperative_optimization: list[str]
    additional_tests: list[str]
    recommend_delay: bool
    delay_reasons: list[str]
    questions_for_other_agents: list[AgentQuestion]
    position_on_plan: PlanPosition | None = None
    need_retrieval: bool = False
    retrieval_query: str | None = None
    confidence: Confidence
    claim_evidence: list[ClaimEvidence] = Field(default_factory=list)


SpecialistOutputPayload = (
    RadiologyReport
    | SpineSurgeonOutput
    | AnesthesiologistOutput
    | NeurologistOutput
    | PerioperativePhysicianOutput
)


class PendingRequest(StrictModel):
    request_id: str = Field(min_length=1)
    requested_by: AgentRole
    target: AgentRole
    question: str = Field(min_length=1)
    source_message_sequence: int = Field(ge=1)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def roles_must_be_specialists(self) -> "PendingRequest":
        if self.requested_by is AgentRole.MODERATOR:
            raise ValueError("pending requests must originate from a specialist")
        if self.target is AgentRole.MODERATOR:
            raise ValueError("pending requests must target a specialist")
        return self


class SpecialistContribution(StrictModel):
    role: AgentRole
    turn_number: int = Field(ge=1, le=11)
    message_sequence: int = Field(ge=1)
    structured_output: SpecialistOutputPayload

    @model_validator(mode="after")
    def output_must_match_role(self) -> "SpecialistContribution":
        expected_types = {
            AgentRole.RADIOLOGIST: RadiologyReport,
            AgentRole.SPINE_SURGEON: SpineSurgeonOutput,
            AgentRole.ANESTHESIOLOGIST: AnesthesiologistOutput,
            AgentRole.NEUROLOGIST: NeurologistOutput,
            AgentRole.PERIOPERATIVE_PHYSICIAN: PerioperativePhysicianOutput,
        }
        expected = expected_types.get(self.role)
        if expected is None or not isinstance(self.structured_output, expected):
            raise ValueError("structured_output does not match the specialist role")
        return self


class Disagreement(StrictModel):
    agents: list[AgentRole] = Field(min_length=2)
    issue: str = Field(min_length=1)
    positions: list[str] = Field(min_length=2)


class OperativePlanDraft(StrictModel):
    """Shared working draft projected from the Spine Surgeon's structured options."""

    version: int = Field(ge=1)
    source_contribution_sequence: int = Field(ge=1)
    surgical_indication: bool
    options: list[SurgicalOption] = Field(default_factory=list, max_length=3)
    preferred_option_id: str | None = None
    approach_comparison: ApproachComparison | None = None
    positions: list[PlanPosition] = Field(default_factory=list)

    @model_validator(mode="after")
    def draft_contract(self) -> "OperativePlanDraft":
        if not self.surgical_indication:
            if (
                self.options
                or self.preferred_option_id is not None
                or self.approach_comparison is not None
            ):
                raise ValueError(
                    "a draft without a surgical indication cannot carry operative options"
                )
            if self.positions:
                raise ValueError("a draft without a surgical indication cannot hold positions")
            return self
        option_ids = {option.option_id for option in self.options}
        if self.preferred_option_id not in option_ids:
            raise ValueError("preferred_option_id must identify an option in the draft")
        if any(
            position.option_id not in option_ids
            or position.draft_version != self.version
            for position in self.positions
        ):
            raise ValueError(
                "positions must reference an existing option and the current draft version"
            )
        return self


class ModeratorDecision(StrictModel):
    public_message: str = Field(min_length=1)
    action: ModeratorAction
    next_speaker: AgentRole | None
    question_for_next_speaker: str | None
    unresolved_issues: list[str]
    disagreements: list[Disagreement]
    consensus_reached: bool
    routing_reason: str = Field(min_length=1)
    continue_discussion: bool
    enter_final_summary: bool
    target_image_ids: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_action_contract(self) -> "ModeratorDecision":
        if len(self.target_image_ids) != len(set(self.target_image_ids)):
            raise ValueError("target_image_ids must be unique")
        if self.action is ModeratorAction.ASK_SPECIALIST:
            if self.next_speaker in (None, AgentRole.MODERATOR):
                raise ValueError("ask_specialist requires a specialist next_speaker")
            if not self.question_for_next_speaker:
                raise ValueError("ask_specialist requires question_for_next_speaker")
            if not self.continue_discussion or self.enter_final_summary:
                raise ValueError("ask_specialist must continue discussion without finalizing")
            if self.next_speaker is AgentRole.RADIOLOGIST and not self.target_image_ids:
                raise ValueError("radiologist follow-up requires target_image_ids")
            if self.next_speaker is not AgentRole.RADIOLOGIST and self.target_image_ids:
                raise ValueError("target_image_ids are only valid for radiologist follow-up")
        if self.action is ModeratorAction.FINALIZE:
            if self.next_speaker is not None or self.question_for_next_speaker is not None:
                raise ValueError("finalize must not select a next speaker")
            if self.continue_discussion or not self.enter_final_summary:
                raise ValueError("finalize must stop discussion and enter final summary")
            if self.target_image_ids:
                raise ValueError("finalize must not include target_image_ids")
        return self


class EvidenceRecord(StrictModel):
    evidence_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    year: int = Field(ge=1900, le=2200)
    section: str = Field(min_length=1)
    content: str = Field(min_length=1)
    used_by: AgentRole
    used_in_message: int = Field(ge=1)


class RadiologySummary(StrictModel):
    xray_findings: list[str]
    ct_findings: list[str]
    mri_findings: list[str]
    cross_modality_conclusion: str
    uncertainties: list[str]


class DiagnosisEvidence(StrictModel):
    diagnosis: str
    case_evidence: list[str]
    neurologic_evidence: list[str]
    imaging_evidence: list[str]
    guideline_evidence_ids: list[str]


class SurgicalIndicationPlan(StrictModel):
    surgery_recommended: bool
    indications: list[str]
    functional_impact: str
    conservative_treatment_summary: list[str]
    reasons_not_to_operate: list[str]
    prerequisites: list[str]


class RecommendedSurgery(StrictModel):
    procedure_name: str
    levels: list[str]
    approach: str
    decompression_extent: str
    fusion_or_fixation: str
    procedure_status: SurgeryFieldStatus = SurgeryFieldStatus.WITHHELD
    levels_status: SurgeryFieldStatus = SurgeryFieldStatus.WITHHELD
    approach_status: SurgeryFieldStatus = SurgeryFieldStatus.WITHHELD
    decompression_status: SurgeryFieldStatus = SurgeryFieldStatus.WITHHELD
    fusion_fixation_status: SurgeryFieldStatus = SurgeryFieldStatus.WITHHELD
    intraoperative_monitoring: list[str]
    key_intraoperative_considerations: list[str]


class AlternativePlan(StrictModel):
    alternative_surgery: list[str]
    conservative_treatment: list[str]
    applicability_conditions: list[str]
    key_differences: list[str]


class PerioperativeRiskSummary(StrictModel):
    anesthesia_risk: str
    asa_classification: str
    anesthesia_plan: str
    cardiopulmonary_risk: str
    bleeding_risk: str
    positioning_risk: str
    infection_risk: str
    thrombosis_risk: str
    comorbidity_risk: str
    mitigation_measures: list[str]


class AdditionalTests(StrictModel):
    laboratory: list[str]
    imaging: list[str]
    cardiopulmonary: list[str]
    neurophysiology: list[str]
    specialty_consultations: list[str]
    other: list[str]


class DisagreementResolution(StrictModel):
    agents: list[AgentRole]
    issue: str
    positions: list[str]
    moderator_resolution: str
    resolved: bool
    unresolved_reason: str | None


class FinalConsensus(StrictModel):
    recommendation: str
    implementation_prerequisites: list[str]
    remaining_concerns: list[str]
    has_unresolved_disagreement: bool


class FinalPlan(StrictModel):
    radiology_summary: RadiologySummary
    diagnoses: list[DiagnosisEvidence]
    surgical_indication: SurgicalIndicationPlan
    recommended_surgery: RecommendedSurgery
    alternatives: AlternativePlan
    definitive_surgery_status: DefinitiveSurgeryStatus = (
        DefinitiveSurgeryStatus.WITHHELD
    )
    definitive_surgery_withheld_reasons: list[str] = Field(default_factory=list)
    conditional_surgical_options: list[SurgicalOption] = Field(
        default_factory=list,
        max_length=3,
    )
    approach_comparison: ApproachComparison | None = None
    perioperative_risks: PerioperativeRiskSummary
    additional_tests: AdditionalTests
    disagreements: list[DisagreementResolution]
    final_consensus: FinalConsensus
    confidence: Confidence
    force_finalized: bool
    claim_evidence: list[ClaimEvidence] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_definitive_status(cls, value: Any) -> Any:
        """Preserve the meaning of completed pre-v5 plans when they are read back."""

        if not isinstance(value, dict) or "definitive_surgery_status" in value:
            return value
        payload = dict(value)
        recommended = payload.get("recommended_surgery")
        if isinstance(recommended, BaseModel):
            procedure_name = getattr(recommended, "procedure_name", None)
        elif isinstance(recommended, dict):
            procedure_name = recommended.get("procedure_name")
        else:
            procedure_name = None
        payload["definitive_surgery_status"] = (
            DefinitiveSurgeryStatus.ESTABLISHED
            if procedure_name and procedure_name != MISSING_RECORDS
            else DefinitiveSurgeryStatus.WITHHELD
        )
        field_status = (
            SurgeryFieldStatus.ESTABLISHED
            if payload["definitive_surgery_status"] in {
                DefinitiveSurgeryStatus.ESTABLISHED,
                DefinitiveSurgeryStatus.ESTABLISHED.value,
            }
            else SurgeryFieldStatus.WITHHELD
        )
        if isinstance(recommended, dict):
            for field in (
                "procedure_status",
                "levels_status",
                "approach_status",
                "decompression_status",
                "fusion_fixation_status",
            ):
                recommended.setdefault(field, field_status.value)
        return payload


class FinalReviewIssue(StrictModel):
    field_path: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class FinalPlanReview(StrictModel):
    verdict: Literal["pass", "revise"]
    issues: list[FinalReviewIssue] = Field(default_factory=list)
    must_abstain_from_exact_surgery: bool = False
    definitive_surgery_allowed: bool = False
    established_field_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def revision_requires_an_issue(self) -> "FinalPlanReview":
        if self.verdict == "revise" and not self.issues:
            raise ValueError("a revision verdict requires at least one issue")
        return self


class ProviderUsage(StrictModel):
    provider: Literal["openai", "deepseek", "fake"]
    model: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


ModelFailureCode = Literal[
    "MODEL_TIMEOUT",
    "MODEL_RATE_LIMITED",
    "MODEL_CONNECTION_FAILED",
    "MODEL_SERVICE_FAILED",
    "MODEL_REQUEST_REJECTED",
    "MODEL_INVALID_OUTPUT",
    "FINAL_PLAN_STATE_MISMATCH",
    "FINAL_PLAN_UNKNOWN_EVIDENCE",
    "FINAL_PLAN_OMITTED_ISSUES",
    "FINAL_PLAN_OMITTED_DISAGREEMENT",
    "FINAL_PLAN_DISAGREEMENT_STATUS",
    "MODEL_INVOCATION_FAILED",
]
ModelFailureOutcome = Literal["recovered", "failed"]


RadiologyProgressStage = Literal[
    "reviewing",
    "batch_completed",
    "synthesizing",
    "completed",
]


class RadiologyProgress(StrictModel):
    stage: RadiologyProgressStage
    completed_images: int = Field(ge=0)
    total_images: int = Field(ge=1)
    batch_number: int | None = Field(default=None, ge=1)
    total_batches: int = Field(ge=1)
    current_image_start: int | None = Field(default=None, ge=1)
    current_image_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def counts_must_be_consistent(self) -> "RadiologyProgress":
        if self.completed_images > self.total_images:
            raise ValueError("completed_images cannot exceed total_images")
        if self.batch_number is not None and self.batch_number > self.total_batches:
            raise ValueError("batch_number cannot exceed total_batches")
        if (self.current_image_start is None) != (self.current_image_end is None):
            raise ValueError("current image range must be complete or absent")
        if self.current_image_start is not None:
            if self.current_image_start > self.current_image_end:
                raise ValueError("current image range is reversed")
            if self.current_image_end > self.total_images:
                raise ValueError("current image range exceeds total_images")
        return self


class ModelFailureRecord(StrictModel):
    failure_id: str = Field(min_length=1)
    agent_role: AgentRole
    stage: str = Field(min_length=1)
    outcome: ModelFailureOutcome
    category: ModelFailureCode
    attempt_categories: list[ModelFailureCode] = Field(min_length=1)
    attempt_count: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    safe_message: str = Field(min_length=1)

    @model_validator(mode="after")
    def attempts_must_be_consistent(self) -> "ModelFailureRecord":
        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count cannot exceed max_attempts")
        if len(self.attempt_categories) > self.attempt_count:
            raise ValueError("attempt categories cannot exceed attempt count")
        if self.category != self.attempt_categories[-1]:
            raise ValueError("category must match the final attempt category")
        return self


class RunMetadata(StrictModel):
    started_at: datetime
    completed_at: datetime | None
    specialist_message_count: int = Field(default=0, ge=0)
    retrieval_count: int = Field(default=0, ge=0)
    model_calls: list[ProviderUsage] = Field(default_factory=list)
    model_failures: list[ModelFailureRecord] = Field(default_factory=list)
    forced_finalize_reason: str | None = None
    finalization_fallback_used: bool = False
    context_policy_version: str = "v5"
    final_review_outcome: Literal["passed", "revised"] | None = None
    radiology_cache_hit: bool = False

    @field_validator("started_at", "completed_at")
    @classmethod
    def require_aware_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run metadata timestamps must be timezone-aware")
        return value.astimezone(UTC)
