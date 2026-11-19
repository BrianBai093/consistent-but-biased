"""Structured outputs for the department-hierarchy decision simulation.

These schemas are deliberately thin. The experiment measures how a department's
decision paradigm shifts treatment choice, so each role reports only what the
protocol asks of it; richer clinical structure would add variance that has
nothing to do with the manipulated variable.

The imaging fields follow one rule that the rest of this module exists to
enforce: **"could not be read" and "was read and is negative" are never the same
value.** Oncology has coded that distinction since TNM separated TX (primary
tumour cannot be assessed) from T0 (no evidence of primary tumour), and the
radiology term for an unreadable study is "nondiagnostic", not "normal".
Collapsing them would make every negative finding uninterpretable.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from schemas import AgentRole, Confidence, ProviderUsage, StrictModel

# ---------------------------------------------------------------- imaging ---


class LevelStatus(str, Enum):
    """Whether responsible spinal levels could be named."""

    REPORTED = "reported"
    NONE = "none"  # read, and no level is involved
    NONDIAGNOSTIC = "nondiagnostic"  # could not be read


class CompressionType(str, Enum):
    VENTRAL = "ventral"
    DORSAL = "dorsal"
    CIRCUMFERENTIAL = "circumferential"
    NONE = "none"
    NONDIAGNOSTIC = "nondiagnostic"


class StenosisGrade(str, Enum):
    """Graded rather than a percentage.

    The imaging supplied to this project is photographed film, and the reading
    model states outright that stenosis "cannot be measured or definitively
    graded" from it. Demanding a percentage would force a fabricated number.
    """

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    NONE = "none"
    NONDIAGNOSTIC = "nondiagnostic"


class SignalChange(str, Enum):
    """Cord T2 signal.

    ``EQUIVOCAL`` exists because published cervical cord T2 grading already
    separates faint, ill-defined signal from intense, well-defined signal.
    """

    PRESENT = "present"
    EQUIVOCAL = "equivocal"
    ABSENT = "absent"
    NONDIAGNOSTIC = "nondiagnostic"


class SymptomImagingCorrelation(str, Enum):
    """How well imaging explains the symptoms.

    ``INDETERMINATE`` is deliberately worded unlike the four fields above: it
    records genuine clinical ambiguity, which is the object of study here, not a
    defect in the source material. The protocol requires the radiologist to say
    so explicitly when the picture is unclear.
    """

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INDETERMINATE = "indeterminate"


UNREADABLE_VALUES = frozenset({"nondiagnostic"})
NEGATIVE_VALUES = frozenset({"none", "absent"})

_LEVEL_PATTERN = re.compile(
    r"^([CTLS])(\d{1,2})[-/]([CTLS])?(\d{1,2})$",
    re.IGNORECASE,
)


def normalize_level(value: str) -> str:
    """Return a spinal level in one canonical spelling.

    Source material mixes ``C3/4``, ``C3-C4``, and ``C3-4``; storing more than
    one spelling would make level counts across cases silently disagree.

    Within one region the lower region letter is dropped (``C3/4`` -> ``C3-4``),
    but a junctional level keeps both because the second letter is the only
    thing that identifies it: ``T12-L1`` must never collapse to ``T12-1``.
    """

    candidate = value.strip().upper().replace(" ", "")
    match = _LEVEL_PATTERN.match(candidate)
    if match is None:
        raise ValueError(f"unrecognized spinal level: {value!r}")
    region, upper, lower_region, lower = match.groups()
    if lower_region and lower_region != region:
        return f"{region}{int(upper)}-{lower_region}{int(lower)}"
    return f"{region}{int(upper)}-{int(lower)}"


class ImagingFindings(StrictModel):
    """Radiology report for the department workflow (protocol Agent 2)."""

    levels: list[str] = Field(default_factory=list)
    level_status: LevelStatus
    compression_type: CompressionType
    stenosis_grade: StenosisGrade
    signal_change: SignalChange
    correlation: SymptomImagingCorrelation
    public_message: str = Field(min_length=1)
    confidence: Confidence

    @field_validator("levels")
    @classmethod
    def canonical_levels(cls, values: list[str]) -> list[str]:
        return [normalize_level(item) for item in values]

    @model_validator(mode="after")
    def levels_match_their_status(self) -> "ImagingFindings":
        if self.level_status is LevelStatus.REPORTED and not self.levels:
            raise ValueError("reported level_status requires at least one level")
        if self.level_status is not LevelStatus.REPORTED and self.levels:
            raise ValueError(
                "levels must be empty unless level_status is 'reported'"
            )
        return self


# ----------------------------------------------------------- stage I & II ---


class ResidentReport(StrictModel):
    """Protocol Agent 1: reports facts, takes no position on treatment."""

    chief_complaint: str = Field(min_length=1)
    history: str = Field(min_length=1)
    physical_exam: str = Field(min_length=1)
    key_positive_signs: list[str] = Field(default_factory=list)
    past_history: str = Field(min_length=1)
    public_message: str = Field(min_length=1)


class LocalizationMatch(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FlagRelevance(str, Enum):
    """What a red flag does to the decision, which is not the same as being odd.

    Listing findings without grading them hands the meeting a set of
    equal-weight doubts, and the only answer to a set of equal-weight doubts is
    to order more tests. In four real cases the neurologist produced five or six
    ungraded flags every time, and the department ordered imaging against them
    in six rulings out of nine.
    """

    #: Offers a positive alternative explanation for the dominant syndrome, so
    #: the compression may not be the cause. The rarest, and the only one that
    #: puts the operation itself in question.
    CHANGES_INDICATION = "changes_indication"
    #: The compression is still the cause; which levels or which approach is not.
    CHANGES_PLAN = "changes_plan"
    #: Real, and changes neither: a coexisting neuropathy, an unexplained
    #: reflex, an incidental abnormality, a gap in the documentation.
    COEXISTING = "coexisting"


class RedFlag(StrictModel):
    finding: str = Field(min_length=1)
    relevance: FlagRelevance
    rationale: str = Field(min_length=1)


class NeurologyAssessment(StrictModel):
    """Protocol Agent 3: localizes findings and keeps a degree of ambiguity."""

    localization: str = Field(min_length=1)
    match_level: LocalizationMatch
    red_flags: list[RedFlag] = Field(default_factory=list)
    symptom_imaging_mismatch: str | None = None
    public_message: str = Field(min_length=1)


class AsaGrade(str, Enum):
    ASA_I = "I"
    ASA_II = "II"
    ASA_III = "III"
    ASA_IV = "IV"
    ASA_V = "V"
    ASA_VI = "VI"
    UNASSESSABLE = "unassessable"


class RiskBand(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNASSESSABLE = "unassessable"


class AnesthesiaAssessment(StrictModel):
    """Protocol Agent 4: states numbers and probabilities, stays neutral."""

    asa_grade: AsaGrade
    map_target_mmhg: int | None = Field(default=None, ge=40, le=140)
    blood_loss_risk: RiskBand
    extubation_difficulty: RiskBand
    prone_position_tolerance: RiskBand
    public_message: str = Field(min_length=1)


class ConservativeStance(str, Enum):
    """How firmly the physiatrist is holding the conservative line."""

    HOLD = "hold"  # still insists on conservative management
    CONDITIONAL = "conditional"  # accepts surgery only under stated conditions
    CONCEDE = "concede"  # persuaded that surgery is warranted


class RehabAssessment(StrictModel):
    """Protocol Agent 5: the strongest advocate for conservative care.

    ``stance`` and ``persuaded_by`` are what make "needs strong evidence before
    conceding" measurable. The physiatrist speaks in every negotiation round, so
    its last message carries its final position.
    """

    conservative_success_rate: int = Field(ge=0, le=100)
    proposed_duration: str = Field(min_length=1)
    preconditions_for_failure: list[str] = Field(min_length=1)
    stance: ConservativeStance
    persuaded_by: list[str] = Field(default_factory=list)
    public_message: str = Field(min_length=1)

    @field_validator("persuaded_by")
    @classmethod
    def reasons_are_substantive(cls, values: list[str]) -> list[str]:
        if any(not item.strip() for item in values):
            raise ValueError("a stated reason cannot be blank")
        return values

    @model_validator(mode="after")
    def conceding_requires_a_reason(self) -> "RehabAssessment":
        # Only full concession needs a cause. ``CONDITIONAL`` is the natural
        # opening position -- it is what preconditions_for_failure already
        # encodes -- and in the frozen baseline nothing has argued with it yet,
        # so demanding a cause there would be unsatisfiable.
        if self.stance is ConservativeStance.CONCEDE and not self.persuaded_by:
            raise ValueError(
                "conceding that surgery is warranted requires naming what persuaded you"
            )
        return self


# --------------------------------------------------------- stage III & IV ---


class ViceDirectorOpinion(StrictModel):
    """Protocol Agent 6: professional judgement plus managing upwards.

    The three style behaviours are readable straight off this schema rather than
    from prose: raising disagreement points (democratic), offering a single
    conservative option (autocratic), or declining to state a preference
    (avoidant, ``preferred_option`` left null).
    """

    assessment: str = Field(min_length=1)
    proposed_options: list[str] = Field(min_length=1)
    preferred_option: str | None = None
    disagreement_points: list[str] = Field(default_factory=list)
    deferred_questions_to_director: list[str] = Field(default_factory=list)
    public_message: str = Field(min_length=1)


class ConsensusOutcome(str, Enum):
    """Whether the director considers the department to have converged."""

    REACHED = "reached"
    NOT_REACHED = "not_reached"
    UNCLEAR = "unclear"


class SurgicalIndication(str, Enum):
    """Whether surgery is indicated, as a position rather than as an action.

    ``decision`` records what the director *did* with the meeting, and ordering
    more tests is a legitimate thing to do with it. But a corpus where most
    meetings end in further workup would leave the primary outcome -- operate or
    not -- undefined for most of the corpus, and the cheapest way to satisfy
    "produce a ruling" is precisely to order a test. This field is asked of
    every ruling, including one that decides nothing, so the position is on the
    record even when the action defers it.
    """

    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class DirectorDecision(str, Enum):
    NOT_YET_DECIDED = "not_yet_decided"
    OPERATE = "operate"
    CONSERVATIVE = "conservative"
    ADDITIONAL_WORKUP = "additional_workup"
    DEFER = "defer"


class FamilyRequest(str, Enum):
    """What the director asked of the patient or the family, stated as an act.

    Protocol dept-v1 asked the director whether it had "transferred liability to
    the family". It never said yes -- not once in nine clean rulings -- although
    the risk-averse director was plainly doing it ("present both options to the
    patient and family with detailed informed consent"), and one wrote that it
    did *not* transfer the decision in the same ruling that arranged for the
    family to make it. A question that names a fault gets denied. So the
    director reports what it asked for, in neutral terms, and whether that
    amounts to moving the decision is judged by the observer, not by the
    director.
    """

    #: The ordinary consent for the procedure being performed. Not a transfer.
    ROUTINE_CONSENT = "routine_consent"
    #: Consent beyond the routine, emphasising risk and the family's signature.
    DETAILED_CONSENT = "detailed_consent"
    #: The options are put to the patient or family to choose between.
    FAMILY_CHOOSES = "family_chooses"
    #: The decision waits for a meeting with the family.
    DECIDE_AFTER_FAMILY_MEETING = "decide_after_family_meeting"


#: The requests that move the decision, or the weight of it, onto the family.
DECISION_MOVING_REQUESTS = frozenset(
    {
        FamilyRequest.DETAILED_CONSENT,
        FamilyRequest.FAMILY_CHOOSES,
        FamilyRequest.DECIDE_AFTER_FAMILY_MEETING,
    }
)


class DirectorRuling(StrictModel):
    """Protocol Agent 7: the only decision maker, and the manipulated variable.

    The self-reported fields exist so the observer never has to infer an action
    from prose. They are collected alongside a text scan of the same behaviour
    and recorded separately, because asking a director about its own conduct may
    itself change the conduct -- which is why they ask about acts, not labels.
    """

    decision: DirectorDecision
    #: What was done with the meeting, and what is believed about the patient,
    #: are asked separately. A ruling that orders more tests still has to say
    #: whether surgery is indicated as things stand.
    surgical_indication: SurgicalIndication
    #: Where further workup would land. Required when the ruling is to seek more
    #: evidence: a director that cannot say what the test would change is not
    #: seeking evidence, and that becomes visible here instead of hidden.
    workup_would_change_decision_to: DirectorDecision | None = None
    final_decision: str | None = None
    surgical_plan: str | None = None
    # Every self-reported field is required rather than defaulted. The default
    # text provider is given the schema as prose in JSON-object mode, so an
    # omitted field is not rejected -- it silently becomes "I did not do that",
    # which would bias every declared metric downwards and differently per
    # provider. An explicit empty list is a statement; an omission is not.
    delegated_tasks: list[str]
    if_delay: bool
    reason_for_delay: str | None = None
    has_contingency_plan: bool
    consensus: ConsensusOutcome
    addressed_roles: list[AgentRole]
    ordered_tests: list[str]
    family_requests: list[FamilyRequest]
    public_message: str = Field(min_length=1)

    @field_validator("addressed_roles")
    @classmethod
    def addressed_roles_are_other_participants(
        cls,
        values: list[AgentRole],
    ) -> list[AgentRole]:
        """Naming oneself, or a role not in the room, is not addressing a junior."""

        allowed = {
            AgentRole.RESIDENT,
            AgentRole.RADIOLOGIST,
            AgentRole.NEUROLOGIST,
            AgentRole.ANESTHESIOLOGIST,
            AgentRole.PHYSIATRIST,
            AgentRole.VICE_DIRECTOR,
        }
        unknown = [role for role in values if role not in allowed]
        if unknown:
            raise ValueError(
                "addressed_roles may only name other participants in this meeting"
            )
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def ruling_is_self_consistent(self) -> "DirectorRuling":
        decided = self.decision is not DirectorDecision.NOT_YET_DECIDED
        if decided and not self.final_decision:
            raise ValueError("a ruling requires final_decision text")
        if not decided and self.final_decision:
            raise ValueError("final_decision must be empty until the director rules")
        if self.decision is DirectorDecision.OPERATE and not self.surgical_plan:
            raise ValueError("an operative ruling requires a surgical plan")
        if self.if_delay and self.decision not in {
            DirectorDecision.DEFER,
            DirectorDecision.ADDITIONAL_WORKUP,
            DirectorDecision.NOT_YET_DECIDED,
        }:
            raise ValueError("if_delay contradicts a decision that proceeds now")
        if self.if_delay and not self.reason_for_delay:
            raise ValueError("a delay requires a stated reason")
        if self.decision is DirectorDecision.DEFER and not self.reason_for_delay:
            # Otherwise pricing the workup exit simply moves the traffic here.
            raise ValueError("deferring requires a stated reason")
        if self.decision is DirectorDecision.ADDITIONAL_WORKUP:
            if not self.ordered_tests:
                raise ValueError("ruling for further workup requires naming the tests")
            if self.workup_would_change_decision_to not in {
                DirectorDecision.OPERATE,
                DirectorDecision.CONSERVATIVE,
            }:
                raise ValueError(
                    "ruling for further workup requires saying where it would land: "
                    "workup_would_change_decision_to must be 'operate' or 'conservative'"
                )
        elif self.workup_would_change_decision_to is not None:
            raise ValueError(
                "workup_would_change_decision_to belongs to an additional_workup ruling"
            )
        if (
            self.decision is DirectorDecision.OPERATE
            and self.surgical_indication is not SurgicalIndication.YES
        ):
            raise ValueError(
                "an operative ruling states that surgery is indicated"
            )
        if self.decision is DirectorDecision.CONSERVATIVE and self.surgical_plan:
            raise ValueError("a conservative ruling cannot carry a surgical plan")
        return self


# --------------------------------------------------------------- baseline ---


class DepartmentBaselinePayload(StrictModel):
    """Stage I and II, frozen once and read by all three arms.

    These five turns carry no style prefix and see no negotiation, so they are
    identical whichever director runs afterwards. Freezing them is what lets a
    difference in outcome be attributed to the style rather than to the evidence.
    """

    #: When these five statements were made. The arms that reuse them re-speak
    #: them under this time rather than under their own, so replaying the
    #: transcript produces the same messages every time it runs -- a message
    #: that differed only by its clock would be rejected as a collision on the
    #: second attempt at a thread.
    frozen_at: datetime
    resident: ResidentReport
    imaging: ImagingFindings
    #: What each batch of films reported before the readings were merged.
    imaging_batches: list[ImagingFindings] = Field(default_factory=list)
    neurology: NeurologyAssessment
    anesthesia: AnesthesiaAssessment
    physiatrist: RehabAssessment


# ------------------------------------------------------------- accounting ---


#: Which part of the meeting a model call belongs to. The arms that reuse the
#: frozen baseline never make its calls, so only the negotiation tokens measure
#: the same thing in all three arms.
UsageStage = Literal["baseline", "negotiation"]


class RoleUsage(StrictModel):
    """One model call attributed to the role that made it.

    ``ProviderUsage`` alone carries no role, and the observer reports token
    counts per role and per stage, so both have to be kept alongside it.
    """

    role: AgentRole
    usage: ProviderUsage
    stage: UsageStage
    #: Which attempt of this call the usage belongs to, counting from one. A
    #: value above one means the role's first answer was rejected by its own
    #: schema and had to be asked again.
    attempts: int = Field(default=1, ge=1)


class RoleModelAssignment(StrictModel):
    """Which provider and model actually served one role in one run.

    Roles may legitimately run on different models -- that stays constant across
    the three arms -- but changing a role's model between batches makes those
    batches incomparable. Without this record the drift is undetectable after
    the fact.
    """

    role: AgentRole
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
