"""The process observer: it counts, it does not judge.

Protocol Agent 8 is the one participant that is not an agent. The design
document is explicit that a language model must not be asked to do the counting,
so everything here is ordinary Python over the finished transcript, and two runs
over the same transcript always produce the same numbers.

Every behavioural metric is collected **twice**:

* *declared* -- the director's own structured self-report, and
* *detected* -- a lexicon scan of what the director actually said in public.

Both are recorded, and neither is corrected against the other. A director that
declares a family meeting but never mentions the family, or that talks about
the family deciding without declaring anything, is a finding about the
simulation rather than a defect to be smoothed away. The declared channel also
carries an observer effect -- being asked about one's own conduct may change the
conduct -- which is why it asks about acts, never about labels, and why the two
channels are compared.

Two limits are structural and must be stated wherever these numbers are used:

* **The lexicons below are not calibrated.** They were written from the style
  prefixes in the design document, not from a coded sample of real transcripts.
  They will miss paraphrases and will fire on negations ("no further imaging is
  needed" contains an imaging term). Until they are calibrated against human
  coding, the detected channel is a signal, not a measurement.
* **Token counts are not comparable between arms.** The arms that reuse the
  frozen baseline never make its model calls, so only ``negotiation_tokens``
  measures the same thing in all three. ``total_tokens`` is a cost figure.
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Iterable, Sequence

from pydantic import Field, model_validator

from department.schemas import (
    DECISION_MOVING_REQUESTS,
    ConsensusOutcome,
    ConservativeStance,
    DirectorDecision,
    DirectorRuling,
    FamilyRequest,
    FlagRelevance,
    NeurologyAssessment,
    RehabAssessment,
    SurgicalIndication,
    RoleUsage,
    UsageStage,
    ViceDirectorOpinion,
)
from department.state import DepartmentState, TerminationCause
from schemas import AgentRole, ChatMessage, DirectorStyle, StrictModel

# --------------------------------------------------------------- lexicons ---

#: How a director names a junior when addressing them.
#:
#: Whole words, never stems. A stem such as "radiolog" is by definition always
#: followed by another letter, so the guard against an adjacent Latin character
#: would stop it matching anything at all. Bare "影像"/"康复"/"麻醉" are absent
#: for the opposite reason: they occur in ordinary clinical prose ("影像学表现",
#: "康复训练"), where they describe a subject rather than summon a person.
ROLE_APPELLATIONS: tuple[tuple[AgentRole, tuple[str, ...]], ...] = (
    (AgentRole.RESIDENT, ("住院总", "管床医生", "一线医生", "resident")),
    (AgentRole.RADIOLOGIST, ("影像科", "放射科", "radiology", "radiologist")),
    (AgentRole.NEUROLOGIST, ("神经内科", "神内", "neurology", "neurologist")),
    (
        AgentRole.ANESTHESIOLOGIST,
        (
            "麻醉科",
            "anesthesia",
            "anaesthesia",
            "anesthesiology",
            "anesthesiologist",
            "anaesthetist",
            "anesthetist",
        ),
    ),
    (
        AgentRole.PHYSIATRIST,
        ("康复科", "康复医学科", "rehabilitation", "physiatrist", "physiatry"),
    ),
    (AgentRole.VICE_DIRECTOR, ("副主任", "带组", "deputy")),
)

#: Moving the decision onto the family, in words.
#:
#: Each entry is one concept and the ways it gets written, because a bilingual
#: transcript glosses itself and counting spellings would score one act twice.
#:
#: **Routine consent is not in here.** Every operation has a consent form, and
#: under dept-v1 an autocratic director telling the resident to "schedule the
#: surgery and obtain consent" was counted as handing the decision to the family.
#: What remains are the acts that do move it: a consent beyond the routine, the
#: family choosing between options, the decision waiting on a family meeting --
#: the same three the declared channel counts.
#:
#: Bare "家属"/"family" are absent: mentioning the family is not handing it the
#: decision. Bare "sign" is absent for a sharper reason -- "Hoffmann sign",
#: "Babinski sign" and "Lhermitte sign" are the core examination findings of
#: cervical myelopathy.
#:
#: **Still partly a manipulation check.** The risk-averse prefix asks for "a
#: detailed informed consent" in those very words and the other two prefixes
#: contain none of this vocabulary, so a difference between arms partly
#: confirms that the prefix was read. The paraphrases speak to behaviour the
#: prompt did not dictate.
LIABILITY_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "detailed-consent",
        (
            "详细知情同意",
            "详细的知情同意",
            "详细签字",
            "detailed informed consent",
            "detailed consent",
            "detailed written consent",
        ),
    ),
    (
        "family-chooses",
        (
            "由家属决定",
            "由患者及家属决定",
            "家属决定",
            "让家属选择",
            "家属选择",
            "family decides",
            "family to decide",
            "family to choose",
            "family will decide",
            "patient and family to decide",
            "options to the patient and family",
            "options to the family",
            "leave it to the family",
            "next of kin",
        ),
    ),
    (
        "family-meeting",
        (
            "家属谈话后",
            "与家属沟通后",
            "与家属商量",
            "family meeting",
            "reconvene with the patient and family",
            "reconvene with the family",
        ),
    ),
)

#: Named studies, and only named studies.
#:
#: Two acts that sit close to this construct are left out on purpose. A
#: consultation ("会诊") is what a consultative director convenes, and a
#: follow-up ("复查") is what a conservative ruling schedules; either would
#: score an arm for its defining behaviour rather than for seeking evidence.
#:
#: What this measures is *mentioning a study*, which is not the same as
#: ordering one. "The MRI already shows ventral compression" scores, and so
#: does "we will not order a CT". Because the evidence in a cervical case is
#: MRI and CT, every arm has a non-zero floor. The field is named for what it
#: counts, and it is not a second reading of ``tests_ordered_declared``.
STUDY_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "dynamic-mri",
        ("动态mri", "动态磁共振", "dynamic mri", "flexion-extension mri"),
    ),
    (
        "contrast-mri",
        ("增强mri", "增强磁共振", "contrast mri", "contrast-enhanced mri"),
    ),
    ("mri", ("mri", "磁共振")),
    ("ct", ("ct", "计算机断层")),
    (
        "radiograph",
        ("x线", "x-ray", "x-rays", "平片", "radiograph", "radiographs", "radiography"),
    ),
    ("emg", ("肌电图", "emg", "electromyography", "electromyogram")),
    (
        "evoked-potential",
        ("诱发电位", "evoked potential", "evoked potentials", "sep", "mep", "ssep"),
    ),
    ("nerve-conduction", ("神经传导", "nerve conduction", "nerve conduction study")),
    ("bone-density", ("骨密度", "bone density", "dexa")),
    ("myelogram", ("造影", "myelogram", "myelography")),
)


@lru_cache(maxsize=512)
def _needle(term: str) -> re.Pattern[str]:
    """Match a term, but never inside a longer Latin word.

    The lexicons mix scripts, and the two need different rules. Chinese runs
    without spaces, so a substring is the right test. A short Latin abbreviation
    is not: "ct" sits inside "laminectomy", "corpectomy", "discectomy" and
    "conduction", so a director that merely named the operation would be scored
    as having ordered a scan -- an error that pushes up exactly the arms which
    decide to operate. Python's word boundary cannot express this, because it
    counts a Chinese character as a word character and would then refuse to
    match "CT扫描"; the guard has to be against an adjacent *Latin* character.
    """

    escaped = re.escape(term.lower())
    before = r"(?<![a-z0-9])" if term[0].isascii() and term[0].isalnum() else ""
    after = r"(?![a-z0-9])" if term[-1].isascii() and term[-1].isalnum() else ""
    return re.compile(before + escaped + after)


def matched_concepts(
    text: str,
    concepts: Sequence[tuple[str, Sequence[str]]],
) -> list[str]:
    """Which concepts this text names, each counted once however it is spelled.

    Spellings are tried longest first and a match blanks the span it consumed,
    so "动态MRI" is the dynamic study rather than that study plus a plain MRI.
    """

    spellings = sorted(
        ((word, concept) for concept, words in concepts for word in words),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    haystack = text.lower()
    found: set[str] = set()
    for word, concept in spellings:
        pattern = _needle(word)
        if pattern.search(haystack):
            found.add(concept)
            haystack = pattern.sub(lambda match: " " * len(match.group()), haystack)
    return sorted(found)


def detected_roles(text: str) -> list[AgentRole]:
    """Which other participants this text names.

    A specialty's name is also its subject matter, in both languages, so a
    director discussing what the imaging showed reads here as addressing
    radiology. That inflates every arm rather than one, which attenuates the
    difference between them instead of manufacturing one -- the safer direction
    for an uncalibrated lexicon, and part of why the declared channel exists.
    """

    haystack = text.lower()
    return [
        role
        for role, appellations in ROLE_APPELLATIONS
        if any(_needle(word).search(haystack) for word in appellations)
    ]


# ---------------------------------------------------------------- metrics ---


class ObserverMetrics(StrictModel):
    """One row per run. Flat on purpose: these go straight into a table.

    Paired fields (``*_declared`` / ``*_detected``) are never reconciled here.
    """

    case_id: str
    director_style: DirectorStyle
    #: What the deputy read; ``None`` when it never spoke (no negotiation).
    deputy_style: DirectorStyle | None
    negotiation: bool = True
    workflow_version: str
    started_at: datetime
    completed_at: datetime | None = None
    # Which frozen stage I-II this arm read. Equal keys across the three arms
    # is the evidence that they argued over one baseline; it cannot be
    # reconstructed later, so it travels with the row.
    baseline_reused: bool
    baseline_key: str | None = None
    termination_cause: TerminationCause | None = None
    negotiation_rounds: int = Field(ge=0)

    # Outcome. "Not yet decided" is a result, not a missing value.
    #
    # Three readings, because the action and the position come apart. A corpus
    # where most meetings end in further workup would leave ``decision`` unable
    # to answer the question the study asks, so the director's stated position
    # and the branch it named for the workup are carried alongside it.
    decision: DirectorDecision
    surgical_indication: SurgicalIndication
    workup_would_change_decision_to: DirectorDecision | None = None
    surgery_recommended: bool
    surgery_intended: bool
    delay_declared: bool
    delay_terminated: bool

    # Discussion length. This replaces the design document's
    # "total_duration_minutes": a turn-based simulation has no minutes, so the
    # substitute is stated as its own measure and is not comparable with the
    # document's expected values.
    turn_count: int = Field(ge=0)
    total_characters: int = Field(ge=0)
    director_turns: int = Field(ge=0)
    director_characters: int = Field(ge=0)
    # Share is measured in characters only. The speaking order is a fixed edge,
    # so the director's share of *turns* is a function of the round count and
    # nothing else: it reads as 0.125 after one round and 0.182 after two, in
    # every arm. Reported as "director dominance" it would say that the arm
    # which ruled decisively dominated the meeting least.
    director_character_share: float = Field(ge=0, le=1)
    negotiation_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    # Juniors addressed.
    roles_addressed_declared: list[AgentRole] = Field(default_factory=list)
    roles_addressed_detected: list[AgentRole] = Field(default_factory=list)
    juniors_addressed_declared: int = Field(ge=0)
    juniors_addressed_detected: int = Field(ge=0)

    # Moving the decision onto the family, counted in director turns so both
    # channels share a denominator: at most one event per ruling, at most one per
    # public message. The declared count is derived here from what the director
    # says it asked of the family; routine consent is not a transfer.
    liability_transfer_declared: int = Field(ge=0)
    liability_transfer_detected: int = Field(ge=0)
    family_requests_declared: list[str] = Field(default_factory=list)
    liability_concepts_detected: list[str] = Field(default_factory=list)
    delegated_task_count: int = Field(ge=0)

    # Further workup. The two are not two readings of one construct: the first
    # counts what the director ordered, the second counts studies it named at
    # all, including the ones already in evidence and the ones it refused.
    tests_ordered_declared: int = Field(ge=0)
    tests_declared: list[str] = Field(default_factory=list)
    studies_mentioned_detected: int = Field(ge=0)
    studies_mentioned: list[str] = Field(default_factory=list)

    # The deputy, whose style behaviour is readable off its own schema: raising
    # contested points, stating a preference, or pushing the question upwards.
    deputy_rounds: int = Field(ge=0)
    deputy_preference_stated: int = Field(ge=0)
    deputy_disagreements: int = Field(ge=0)
    deputy_questions_to_director: int = Field(ge=0)
    contingency_plan_declared: bool = False

    # What the neurologist raised, graded. Frozen, so identical across arms:
    # a difference between arms can never be explained by it, and the split
    # between "changes the decision" and "coexisting" is the lever on how much
    # further workup a case attracts.
    flags_changing_indication: int = Field(ge=0)
    flags_changing_plan: int = Field(ge=0)
    flags_coexisting: int = Field(ge=0)

    # How hard the case is, read off the frozen baseline. It is identical in
    # all three arms by construction, which is what makes it usable as a
    # covariate: a difference in outcome between arms cannot be explained by it,
    # and a difference between cases can be adjusted for it.
    baseline_conservative_success_rate: int = Field(ge=0, le=100)

    # Consensus, from the director's own verdict and from the position the
    # conservative advocate was left holding.
    consensus_declared: ConsensusOutcome
    physiatrist_opening_stance: ConservativeStance
    physiatrist_final_stance: ConservativeStance
    physiatrist_moved: bool
    #: ``None`` when the ruling decides no treatment: there is then nothing for
    #: the advocate to agree or disagree with, and recording False would score
    #: the arms that order further workup as arms without consensus.
    advocate_agrees_with_ruling: bool | None = None
    persuaded_by_count: int = Field(ge=0)
    consensus_reached: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def _deputy_faced_the_director(cls, data):
        """A row written before amendment A4 has no ``deputy_style``.

        Every such run gave the deputy the director's style, so the row is read
        as saying that, rather than as one whose deputy read nothing.
        """

        if isinstance(data, dict) and "deputy_style" not in data:
            data = {**data, "deputy_style": data.get("director_style")}
        return data


def observe(state: DepartmentState) -> ObserverMetrics:
    """Summarise one finished run.

    Enum comparisons use ``==`` rather than ``is`` throughout: a state read back
    from a checkpoint or from stored JSON may carry the enum's string value, and
    these enums subclass ``str`` precisely so that both compare equal.
    """

    rulings = [DirectorRuling.model_validate(item) for item in state["director_rulings"]]
    if not rulings:
        raise ValueError("the observer needs at least one director ruling")
    final = rulings[-1]

    latest = state["physiatrist_latest"]
    opening = state["physiatrist_opening"]
    if latest is None or opening is None:
        raise ValueError("the observer needs the physiatrist's opening and final positions")
    latest = RehabAssessment.model_validate(latest)
    opening = RehabAssessment.model_validate(opening)
    final_stance = ConservativeStance(latest.stance)
    opening_stance = ConservativeStance(opening.stance)

    messages = [ChatMessage.model_validate(item) for item in state["chat_log"]]
    director_messages = [
        message for message in messages if message.speaker_role == AgentRole.DIRECTOR
    ]
    total_characters = sum(len(message.content) for message in messages)
    director_characters = sum(len(message.content) for message in director_messages)

    declared_roles = _ordered_roles(
        role for ruling in rulings for role in ruling.addressed_roles
    )
    found_roles = _ordered_roles(
        role for message in director_messages for role in detected_roles(message.content)
    )

    spoken = "\n".join(message.content for message in director_messages)
    liability_concepts = matched_concepts(spoken, LIABILITY_CONCEPTS)
    studies = matched_concepts(spoken, STUDY_CONCEPTS)
    declared_tests = _ordered_strings(
        test for ruling in rulings for test in ruling.ordered_tests
    )
    deputy = [ViceDirectorOpinion.model_validate(item) for item in state["vice_opinions"]]

    consensus = ConsensusOutcome(final.consensus)
    indication = SurgicalIndication(final.surgical_indication)
    agrees = advocate_agrees(DirectorDecision(final.decision), final_stance)
    return ObserverMetrics(
        case_id=state["case_id"],
        director_style=DirectorStyle(state["director_style"]),
        deputy_style=state.get("deputy_style", state["director_style"]),
        negotiation=state.get("negotiation", True),
        workflow_version=state["workflow_version"],
        started_at=state["started_at"],
        completed_at=state["completed_at"],
        baseline_reused=bool(state["baseline_reused"]),
        baseline_key=state["baseline_key"],
        termination_cause=state["termination_cause"],
        negotiation_rounds=state["round_number"],
        decision=DirectorDecision(final.decision),
        surgical_indication=indication,
        workup_would_change_decision_to=final.workup_would_change_decision_to,
        surgery_recommended=final.decision == DirectorDecision.OPERATE,
        # The meeting acted on surgery, or said it was indicated, or said the
        # workup it ordered would lead there.
        surgery_intended=(
            final.decision == DirectorDecision.OPERATE
            or indication == SurgicalIndication.YES
            or final.workup_would_change_decision_to == DirectorDecision.OPERATE
        ),
        delay_declared=any(ruling.if_delay for ruling in rulings),
        delay_terminated=state["termination_cause"] == TerminationCause.DECISION_DELAY,
        turn_count=len(messages),
        total_characters=total_characters,
        director_turns=len(director_messages),
        director_characters=director_characters,
        director_character_share=_share(director_characters, total_characters),
        negotiation_tokens=_tokens(state["role_usages"], stage="negotiation"),
        total_tokens=_tokens(state["role_usages"]),
        roles_addressed_declared=declared_roles,
        roles_addressed_detected=found_roles,
        juniors_addressed_declared=len(declared_roles),
        juniors_addressed_detected=len(found_roles),
        liability_transfer_declared=sum(
            1
            for ruling in rulings
            if any(request in DECISION_MOVING_REQUESTS for request in ruling.family_requests)
        ),
        family_requests_declared=list(
            dict.fromkeys(
                FamilyRequest(request).value
                for ruling in rulings
                for request in ruling.family_requests
            )
        ),
        liability_transfer_detected=sum(
            1
            for message in director_messages
            if matched_concepts(message.content, LIABILITY_CONCEPTS)
        ),
        liability_concepts_detected=liability_concepts,
        delegated_task_count=len(
            _ordered_strings(task for ruling in rulings for task in ruling.delegated_tasks)
        ),
        tests_ordered_declared=len(declared_tests),
        tests_declared=declared_tests,
        studies_mentioned_detected=len(studies),
        studies_mentioned=studies,
        deputy_rounds=len(deputy),
        deputy_preference_stated=sum(1 for item in deputy if item.preferred_option),
        deputy_disagreements=sum(len(item.disagreement_points) for item in deputy),
        deputy_questions_to_director=sum(
            len(item.deferred_questions_to_director) for item in deputy
        ),
        contingency_plan_declared=any(ruling.has_contingency_plan for ruling in rulings),
        flags_changing_indication=_flags(state, FlagRelevance.CHANGES_INDICATION),
        flags_changing_plan=_flags(state, FlagRelevance.CHANGES_PLAN),
        flags_coexisting=_flags(state, FlagRelevance.COEXISTING),
        baseline_conservative_success_rate=opening.conservative_success_rate,
        consensus_declared=consensus,
        physiatrist_opening_stance=opening_stance,
        physiatrist_final_stance=final_stance,
        physiatrist_moved=final_stance != opening_stance,
        persuaded_by_count=len(latest.persuaded_by),
        advocate_agrees_with_ruling=agrees,
        # The document asks for a single boolean, and it is the conjunction of
        # the two independent readings -- where there is a treatment to agree on.
        consensus_reached=(
            None if agrees is None else consensus == ConsensusOutcome.REACHED and agrees
        ),
    )


def _flags(state: DepartmentState, relevance: FlagRelevance) -> int:
    neurology = state["neurology"]
    if neurology is None:
        return 0
    assessment = NeurologyAssessment.model_validate(neurology)
    return sum(1 for flag in assessment.red_flags if flag.relevance == relevance)


def advocate_agrees(
    decision: DirectorDecision,
    stance: ConservativeStance,
) -> bool | None:
    """Does the conservative advocate's last position agree with the ruling?

    Agreement is not one fixed stance. Against an operative ruling it means
    conceding. Against a conservative ruling it means the opposite: the advocate
    that held its line got what it argued for, and conceding there would be the
    disagreement. Reading "conceded" as agreement in both directions would make
    consensus unreachable for the very ruling the risk-averse arm is expected to
    produce.

    A ruling that decides no treatment -- further workup, deferral, or no ruling
    at all -- leaves nothing to agree with, and the answer is "not applicable",
    not "no". Under dept-v1 it was "no", and six rulings of nine in the first
    clean batch were further workup: the arms that order the most tests were
    scored as the arms that reach consensus least, by definition.
    """

    if decision == DirectorDecision.OPERATE:
        return stance == ConservativeStance.CONCEDE
    if decision == DirectorDecision.CONSERVATIVE:
        return stance in {ConservativeStance.HOLD, ConservativeStance.CONDITIONAL}
    return None


def _ordered_roles(roles: Iterable[AgentRole]) -> list[AgentRole]:
    return list(dict.fromkeys(AgentRole(role) for role in roles))


def _ordered_strings(values: Iterable[str]) -> list[str]:
    """Distinct entries, compared without surrounding space or case."""

    seen: dict[str, str] = {}
    for value in values:
        key = " ".join(value.split()).casefold()
        if key and key not in seen:
            seen[key] = value.strip()
    return list(seen.values())


def _share(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _tokens(usages: Sequence[RoleUsage], *, stage: UsageStage | None = None) -> int:
    records = [RoleUsage.model_validate(item) for item in usages]
    return sum(
        record.usage.total_tokens
        for record in records
        if stage is None or record.stage == stage
    )
