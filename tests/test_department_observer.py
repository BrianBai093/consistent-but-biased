"""The observer counts what happened; it never decides what it meant."""

from pathlib import Path

from department.observer import (
    LIABILITY_CONCEPTS,
    STUDY_CONCEPTS,
    detected_roles,
    matched_concepts,
    observe,
)
from department.schemas import (
    ConsensusOutcome,
    ConservativeStance,
    DirectorDecision,
    FamilyRequest,
    DirectorRuling,
    RoleUsage,
)
from department.state import (
    DEPARTMENT_WORKFLOW_VERSION,
    DepartmentPhase,
    TerminationCause,
    create_initial_department_state,
)
from schemas import AgentRole, ChatMessage, DirectorStyle, ProviderUsage
from tests.department_factories import (
    rehab_assessment,
    ruling,
    undecided,
    vice_opinion,
)

BASELINE_ROLES = (
    AgentRole.RESIDENT,
    AgentRole.RADIOLOGIST,
    AgentRole.NEUROLOGIST,
    AgentRole.ANESTHESIOLOGIST,
    AgentRole.PHYSIATRIST,
)


def message(role: AgentRole, content: str, sequence: int) -> ChatMessage:
    return ChatMessage(
        message_id=f"case-1:{role.value}:{sequence}",
        sequence=sequence,
        speaker_id=role.value,
        speaker_name=role.value,
        speaker_role=role,
        content=content,
        turn_number=1,
    )


def observer_state(
    *,
    chat_log: list[ChatMessage],
    rulings: list[DirectorRuling],
    stance=ConservativeStance.HOLD,
    opening_stance=ConservativeStance.CONDITIONAL,
    usages: list[RoleUsage] | None = None,
    style: DirectorStyle = DirectorStyle.AVOIDANT,
    termination: TerminationCause | None = TerminationCause.DIRECTOR_RULED,
    rounds: int = 1,
):
    state = create_initial_department_state(
        case_id="case-1",
        director_style=style,
        artifact_dir="/derived",
        case_pages=[],
        test_pages=[],
        images=[],
        role_models=[],
    )
    state["chat_log"] = chat_log
    state["director_rulings"] = rulings
    state["physiatrist_opening"] = rehab_assessment(stance=opening_stance)
    state["physiatrist_latest"] = rehab_assessment(
        stance=stance,
        persuaded_by=["imaging progression"] if stance is ConservativeStance.CONCEDE else [],
    )
    state["role_usages"] = usages or []
    state["phase"] = DepartmentPhase.COMPLETED
    state["termination_cause"] = termination
    state["round_number"] = rounds
    return state


def transcript(director_says: str) -> list[ChatMessage]:
    """Five baseline statements of ten characters each, then one director turn."""

    log = [
        message(role, "0123456789", index)
        for index, role in enumerate(BASELINE_ROLES, start=1)
    ]
    log.append(message(AgentRole.DIRECTOR, director_says, len(log) + 1))
    return log


# ---------------------------------------------------------------- lexicons ---


def test_a_longer_spelling_wins_over_the_shorter_one_inside_it():
    """"动态MRI" is the dynamic study, not that study plus a plain MRI."""

    found = matched_concepts("先做动态MRI，再加一个CT", STUDY_CONCEPTS)

    assert found == ["ct", "dynamic-mri"]


def test_one_study_written_twice_is_still_one_study():
    """Chinese clinical prose glosses the Latin abbreviation as it goes."""

    assert matched_concepts("磁共振(MRI)已经完成", STUDY_CONCEPTS) == ["mri"]
    assert matched_concepts(
        "详细知情同意（detailed informed consent）", LIABILITY_CONCEPTS
    ) == ["detailed-consent"]


def test_naming_the_operation_is_not_ordering_a_scan():
    """"CT" sits inside every -ectomy, and an operative arm names its operation.

    A bare substring match would score a phantom scan in exactly the arms that
    decide to operate, which is the direction that would manufacture a result.
    """

    assert matched_concepts(
        "C4-5 anterior cervical discectomy and fusion", STUDY_CONCEPTS
    ) == []
    assert matched_concepts("laminectomy at C4-5", STUDY_CONCEPTS) == []
    assert matched_concepts("order a CT扫描", STUDY_CONCEPTS) == ["ct"]


def test_an_english_study_is_recognised_in_its_ordinary_forms():
    """The lexicon must survive the language the prompts are written in."""

    assert matched_concepts("order plain radiographs", STUDY_CONCEPTS) == ["radiograph"]
    assert matched_concepts("repeat the evoked potentials", STUDY_CONCEPTS) == [
        "evoked-potential"
    ]
    assert matched_concepts("get two x-rays", STUDY_CONCEPTS) == ["radiograph"]


def test_a_clinical_sign_is_not_a_signature():
    """Hoffmann, Babinski and Lhermitte are the examination, not a consent form."""

    assert matched_concepts("Hoffmann sign positive bilaterally", LIABILITY_CONCEPTS) == []
    assert matched_concepts("key positive signs are brisk reflexes", LIABILITY_CONCEPTS) == []
    assert matched_concepts(
        "have the family sign a detailed informed consent", LIABILITY_CONCEPTS
    ) == ["detailed-consent"]


def test_routine_consent_is_not_moving_the_decision():
    """Every operation has a consent form.

    Under dept-v1 an autocratic director telling the resident to "schedule the
    surgery and obtain consent" was counted as handing the decision to the family.
    """

    assert matched_concepts("Resident, schedule the surgery and obtain consent.", LIABILITY_CONCEPTS) == []
    assert matched_concepts("签署手术知情同意书后择期手术。", LIABILITY_CONCEPTS) == []
    assert matched_concepts(
        "We will reconvene with the patient and family to make a fully informed decision.",
        LIABILITY_CONCEPTS,
    ) == ["family-meeting"]
    assert matched_concepts(
        "present both options to the patient and family", LIABILITY_CONCEPTS
    ) == ["family-chooses"]


def test_discussing_a_specialty_is_not_addressing_it():
    """The lexicon names departments as people, not findings as subjects.

    "影像学表现" and "康复训练" are ordinary clinical prose. Counting them as
    the director summoning a junior would make every arm look consultative.
    """

    assert detected_roles("影像学表现符合脊髓受压，康复训练效果有限") == []
    assert detected_roles("影像科，你再看一下这个节段") == [AgentRole.RADIOLOGIST]


def test_an_english_director_addressing_juniors_is_detected():
    """The democratic arm is defined by naming juniors, and the prompts are English."""

    assert detected_roles("Radiology, please re-read C4-5.") == [AgentRole.RADIOLOGIST]
    assert detected_roles(
        "I want anesthesia and neurology to comment before I rule."
    ) == [AgentRole.NEUROLOGIST, AgentRole.ANESTHESIOLOGIST]
    assert detected_roles("Deputy, give me your preferred option.") == [
        AgentRole.VICE_DIRECTOR
    ]


# ------------------------------------------------------------ fixed sample ---


def test_a_fixed_transcript_produces_fixed_numbers():
    said = "住院总，先做动态MRI和肌电图，并让家属签字确认。"
    state = observer_state(
        chat_log=transcript(said),
        rulings=[
            undecided(
                ordered_tests=["dynamic MRI"],
                addressed_roles=[AgentRole.RESIDENT, AgentRole.VICE_DIRECTOR],
                family_requests=[FamilyRequest.DETAILED_CONSENT],
                if_delay=True,
                reason_for_delay="awaiting dynamic imaging",
            )
        ],
        termination=TerminationCause.DECISION_DELAY,
        rounds=2,
    )

    metrics = observe(state)

    assert metrics.turn_count == 6
    assert metrics.director_turns == 1
    assert metrics.total_characters == 50 + len(said)
    assert metrics.director_characters == len(said)
    assert metrics.director_character_share == round(len(said) / (50 + len(said)), 4)

    # Not yet decided is an outcome of its own: neither surgery nor conservative.
    assert metrics.decision is DirectorDecision.NOT_YET_DECIDED
    assert metrics.surgery_recommended is False
    assert metrics.delay_declared is True
    assert metrics.delay_terminated is True
    assert metrics.negotiation_rounds == 2


def test_the_two_channels_are_recorded_side_by_side_and_disagree():
    """Neither channel is corrected against the other; the gap is the finding."""

    state = observer_state(
        chat_log=transcript("住院总，先做动态MRI和肌电图，并让家属在知情同意书上详细签字确认。"),
        rulings=[
            undecided(
                ordered_tests=["dynamic MRI"],
                addressed_roles=[AgentRole.RESIDENT, AgentRole.VICE_DIRECTOR],
                family_requests=[FamilyRequest.DETAILED_CONSENT],
                if_delay=True,
                reason_for_delay="awaiting dynamic imaging",
            )
        ],
    )

    metrics = observe(state)

    # Declared two juniors, named one of them out loud.
    assert metrics.juniors_addressed_declared == 2
    assert metrics.juniors_addressed_detected == 1
    assert metrics.roles_addressed_detected == [AgentRole.RESIDENT]
    # Declared one test, named two studies.
    assert metrics.tests_ordered_declared == 1
    assert metrics.studies_mentioned_detected == 2
    assert metrics.studies_mentioned == ["dynamic-mri", "emg"]
    assert metrics.liability_transfer_declared == 1
    assert metrics.liability_transfer_detected == 1


def test_a_director_can_act_without_declaring_it():
    state = observer_state(
        chat_log=transcript("让家属在知情同意书上详细签字，再做个肌电图。"),
        rulings=[ruling()],
    )

    metrics = observe(state)

    assert metrics.liability_transfer_declared == 0
    assert metrics.liability_transfer_detected == 1
    assert metrics.tests_ordered_declared == 0
    assert metrics.studies_mentioned_detected == 1
    assert metrics.studies_mentioned == ["emg"]


def test_a_director_can_declare_what_it_never_said():
    state = observer_state(
        chat_log=transcript("按计划手术。"),
        rulings=[
            ruling(
                family_requests=[FamilyRequest.DETAILED_CONSENT],
                addressed_roles=[AgentRole.PHYSIATRIST],
                ordered_tests=[],
            )
        ],
    )

    metrics = observe(state)

    assert metrics.liability_transfer_declared == 1
    assert metrics.liability_transfer_detected == 0
    assert metrics.juniors_addressed_declared == 1
    assert metrics.juniors_addressed_detected == 0


def test_only_the_director_is_scanned():
    """Every detected metric is a claim about the director's own behaviour."""

    log = transcript("按计划手术。")
    log.append(
        message(
            AgentRole.PHYSIATRIST,
            "家属已经签字，但我仍建议复查动态MRI。",
            len(log) + 1,
        )
    )

    metrics = observe(observer_state(chat_log=log, rulings=[ruling()]))

    assert metrics.liability_transfer_detected == 0
    assert metrics.studies_mentioned_detected == 0
    assert metrics.turn_count == 7


# --------------------------------------------------------------- accounting ---


def test_declared_events_are_counted_across_every_round():
    """Keeping only the final ruling would erase what round one did."""

    state = observer_state(
        rounds=2,
        chat_log=transcript("先看看。"),
        rulings=[
            undecided(
                ordered_tests=["dynamic MRI"],
                family_requests=[FamilyRequest.DETAILED_CONSENT],
                if_delay=True,
                reason_for_delay="awaiting imaging",
            ),
            ruling(
                ordered_tests=["nerve conduction study"],
                family_requests=[FamilyRequest.DETAILED_CONSENT],
                addressed_roles=[AgentRole.VICE_DIRECTOR],
            ),
        ],
    )

    metrics = observe(state)

    assert metrics.tests_ordered_declared == 2
    assert metrics.liability_transfer_declared == 2
    # The outcome is still the ruling that ended the meeting.
    assert metrics.decision is DirectorDecision.OPERATE
    assert metrics.surgery_recommended is True


def test_only_negotiation_tokens_are_comparable_between_arms():
    usages = [
        RoleUsage(
            role=role,
            usage=ProviderUsage(provider="fake", model="m", total_tokens=100),
            stage=stage,
        )
        for role, stage in (
            (AgentRole.RESIDENT, "baseline"),
            (AgentRole.RADIOLOGIST, "baseline"),
            (AgentRole.DIRECTOR, "negotiation"),
        )
    ]

    metrics = observe(
        observer_state(
            chat_log=transcript("按计划手术。"),
            rulings=[ruling()],
            usages=usages,
        )
    )

    assert metrics.total_tokens == 300
    assert metrics.negotiation_tokens == 100


def test_consensus_needs_both_readings():
    """A director declaring agreement over an advocate that never moved is not consensus.

    "Conditional" is the advocate's stated opening position, so it cannot stand
    in for agreement: an advocate that opened there and stayed there agreed to
    nothing.
    """

    unmoved = observe(
        observer_state(
            chat_log=transcript("按计划手术。"),
            rulings=[ruling(consensus=ConsensusOutcome.REACHED)],
            opening_stance=ConservativeStance.CONDITIONAL,
            stance=ConservativeStance.CONDITIONAL,
        )
    )
    moved = observe(
        observer_state(
            chat_log=transcript("按计划手术。"),
            rulings=[ruling(consensus=ConsensusOutcome.REACHED)],
            opening_stance=ConservativeStance.CONDITIONAL,
            stance=ConservativeStance.CONCEDE,
        )
    )

    assert unmoved.consensus_declared is ConsensusOutcome.REACHED
    assert unmoved.physiatrist_moved is False
    assert unmoved.consensus_reached is False
    assert moved.physiatrist_moved is True
    assert moved.persuaded_by_count == 1
    assert moved.consensus_reached is True


def test_holding_the_line_is_agreement_with_a_conservative_ruling():
    """Agreement is not one fixed stance; it depends on what was ruled.

    The advocate that argued for conservative management and got it agreed with
    the ruling. Requiring it to concede would make consensus unreachable for
    precisely the outcome the risk-averse arm is expected to produce.
    """

    conservative = observe(
        observer_state(
            chat_log=transcript("保守治疗三个月后再评估。"),
            rulings=[
                ruling(
                    decision=DirectorDecision.CONSERVATIVE,
                    final_decision="A strict conservative trial first.",
                    surgical_plan=None,
                    consensus=ConsensusOutcome.REACHED,
                )
            ],
            stance=ConservativeStance.HOLD,
        )
    )
    operative = observe(
        observer_state(
            chat_log=transcript("按计划手术。"),
            rulings=[ruling(consensus=ConsensusOutcome.REACHED)],
            stance=ConservativeStance.HOLD,
        )
    )

    assert conservative.advocate_agrees_with_ruling is True
    assert conservative.consensus_reached is True
    # The same stance against an operative ruling is disagreement.
    assert operative.advocate_agrees_with_ruling is False
    assert operative.consensus_reached is False


def test_the_deputy_is_measured_on_its_own_schema():
    """Its three style behaviours are readable off its structured output."""

    state = observer_state(
        chat_log=transcript("按计划手术。"),
        rulings=[ruling(has_contingency_plan=True)],
    )
    state["vice_opinions"] = [
        vice_opinion(
            preferred_option=None,
            disagreement_points=["C4-5 as well?"],
            deferred_questions_to_director=["Which approach do you want?"],
        ),
        vice_opinion(disagreement_points=[]),
    ]

    metrics = observe(state)

    assert metrics.deputy_rounds == 2
    assert metrics.deputy_preference_stated == 1
    assert metrics.deputy_disagreements == 1
    assert metrics.deputy_questions_to_director == 1
    assert metrics.contingency_plan_declared is True


# -------------------------------------------------------------- end to end ---


async def test_the_three_arms_are_measured_on_the_same_scale(tmp_path: Path):
    """The denominators must be built identically, or the shares are not shares.

    Stage I-II is generated by whichever arm runs first and replayed for the
    others. If the replay were dropped, the first arm would carry five extra
    statements and its director would appear to say a smaller share of a longer
    meeting -- a difference produced by run order rather than by style.
    """

    from department.graph import create_department_graph, department_input, thread_config
    from services.baseline_store import BaselineStore
    from services.checkpoint import open_sqlite_checkpointer
    from tests.department_factories import case_sources, department_with_client

    store = BaselineStore(tmp_path / "baselines")
    rows = []
    for index, style in enumerate(DirectorStyle):
        registry, _ = department_with_client()
        async with open_sqlite_checkpointer(tmp_path / f"arm-{index}.sqlite3") as saver:
            graph = create_department_graph(
                registry=registry,
                checkpointer=saver,
                baseline_store=store,
            )
            result = await graph.ainvoke(
                department_input(
                    case_id="case-dept",
                    director_style=style,
                    artifact_dir=str(tmp_path / "derived"),
                    case_preop_md="History.",
                    tests_preop_md="Labs.",
                    case_summary_md="Cervical myelopathy.",
                    **case_sources(tmp_path),
                ),
                config=thread_config(f"arm-{index}"),
            )
        rows.append(observe(result))

    assert {row.director_style for row in rows} == set(DirectorStyle)
    assert len({row.turn_count for row in rows}) == 1
    assert len({row.total_characters for row in rows}) == 1
    assert len({row.director_character_share for row in rows}) == 1
    # One baseline, and every arm says which one it read.
    assert len({row.baseline_key for row in rows}) == 1
    assert [row.baseline_reused for row in rows] == [False] + [True] * (len(rows) - 1)


async def test_the_observer_reads_a_real_run(tmp_path: Path):
    from tests.test_department_graph import run_department

    result, _, _ = await run_department(
        tmp_path,
        responses={DirectorRuling: [undecided(), ruling()]},
    )

    metrics = observe(result)

    assert metrics.case_id == "case-dept"
    assert metrics.director_style is DirectorStyle.DEMOCRATIC
    assert metrics.baseline_reused is False
    assert metrics.workflow_version == DEPARTMENT_WORKFLOW_VERSION
    assert metrics.completed_at is not None
    # Five baseline statements, then two rounds of deputy, advocate, director.
    assert metrics.turn_count == 11
    assert metrics.director_turns == 2
    assert metrics.negotiation_rounds == 2
    assert metrics.decision is DirectorDecision.OPERATE


def test_a_workup_ruling_still_reports_where_it_was_heading():
    """Otherwise a corpus that mostly orders tests has no outcome variable."""

    from department.schemas import SurgicalIndication

    metrics = observe(
        observer_state(
            chat_log=transcript("先做动态MRI再定。"),
            rulings=[
                ruling(
                    decision=DirectorDecision.ADDITIONAL_WORKUP,
                    surgical_indication=SurgicalIndication.UNCERTAIN,
                    final_decision="Dynamic MRI first.",
                    surgical_plan=None,
                    ordered_tests=["dynamic cervical MRI"],
                    workup_would_change_decision_to=DirectorDecision.OPERATE,
                )
            ],
        )
    )

    # The action was not surgery; the position was that it is where this lands.
    assert metrics.surgery_recommended is False
    assert metrics.surgery_intended is True
    assert metrics.workup_would_change_decision_to is DirectorDecision.OPERATE


def test_a_workup_ruling_that_was_heading_to_conservative_is_not_surgery():
    from department.schemas import SurgicalIndication

    metrics = observe(
        observer_state(
            chat_log=transcript("先做动态MRI再定。"),
            rulings=[
                ruling(
                    decision=DirectorDecision.ADDITIONAL_WORKUP,
                    surgical_indication=SurgicalIndication.NO,
                    final_decision="Dynamic MRI first.",
                    surgical_plan=None,
                    ordered_tests=["dynamic cervical MRI"],
                    workup_would_change_decision_to=DirectorDecision.CONSERVATIVE,
                )
            ],
        )
    )

    assert metrics.surgery_recommended is False
    assert metrics.surgery_intended is False
    assert metrics.surgical_indication is SurgicalIndication.NO


def test_the_observer_decides_what_a_family_request_amounts_to():
    """Routine consent is not a transfer; the other three requests are."""

    routine = observe(
        observer_state(
            chat_log=transcript("按计划手术。"),
            rulings=[ruling(family_requests=[FamilyRequest.ROUTINE_CONSENT])],
        )
    )
    choosing = observe(
        observer_state(
            chat_log=transcript("先看看。"),
            rulings=[
                undecided(family_requests=[FamilyRequest.ROUTINE_CONSENT]),
                ruling(
                    family_requests=[
                        FamilyRequest.FAMILY_CHOOSES,
                        FamilyRequest.ROUTINE_CONSENT,
                    ]
                ),
            ],
            rounds=2,
        )
    )

    assert routine.liability_transfer_declared == 0
    assert routine.family_requests_declared == ["routine_consent"]
    assert choosing.liability_transfer_declared == 1
    assert choosing.family_requests_declared == ["routine_consent", "family_chooses"]


def test_a_ruling_that_decides_no_treatment_has_no_consensus_to_score():
    """Not "no consensus" -- not applicable.

    Scoring a workup ruling as a failure to agree would mark the arms that order
    the most tests as the arms that agree least, by definition rather than by
    behaviour.
    """

    workup = observe(
        observer_state(
            chat_log=transcript("先做动态MRI。"),
            rulings=[
                ruling(
                    decision=DirectorDecision.ADDITIONAL_WORKUP,
                    final_decision="Dynamic MRI first.",
                    surgical_plan=None,
                    ordered_tests=["dynamic cervical MRI"],
                    workup_would_change_decision_to=DirectorDecision.OPERATE,
                    consensus=ConsensusOutcome.REACHED,
                )
            ],
        )
    )

    assert workup.advocate_agrees_with_ruling is None
    assert workup.consensus_reached is None
