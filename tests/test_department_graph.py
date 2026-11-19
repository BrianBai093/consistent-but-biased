from pathlib import Path

import pytest

from department.graph import (
    baseline_transcript,
    create_department_graph,
    department_input,
    thread_config,
)
from department.schemas import (
    ConservativeStance,
    DirectorDecision,
    DirectorRuling,
    RehabAssessment,
    RoleModelAssignment,
    ViceDirectorOpinion,
)
from department.state import (
    MAX_NEGOTIATION_ROUNDS,
    DepartmentPhase,
    TerminationCause,
    department_thread_id,
    final_ruling,
    is_department_thread,
)
from schemas import AgentRole, DirectorStyle
from services.baseline_store import BaselineStore
from services.checkpoint import open_sqlite_checkpointer
from tests.department_factories import (
    case_sources,
    department_payload,
    department_with_client,
    rehab_assessment,
    ruling,
    undecided,
    vice_opinion,
)


def graph_input(tmp_path: Path, style: DirectorStyle, **overrides) -> dict:
    values = {
        "case_id": "case-dept",
        "director_style": style,
        "artifact_dir": str(tmp_path / "derived"),
        "case_preop_md": "<!-- source: case-page-001 -->\n\nHistory.",
        "tests_preop_md": "<!-- source: test-page-001 -->\n\nLabs.",
        "case_summary_md": "Cervical myelopathy.",
        **case_sources(tmp_path),
    }
    values.update(overrides)
    return department_input(**values)


async def run_department(tmp_path: Path, *, style=DirectorStyle.DEMOCRATIC, **kwargs):
    registry, client = department_with_client(kwargs.pop("responses", None))
    store = kwargs.pop("store", None)
    async with open_sqlite_checkpointer(tmp_path / "dept.sqlite3") as saver:
        graph = create_department_graph(
            registry=registry,
            checkpointer=saver,
            baseline_store=store,
        )
        result = await graph.ainvoke(
            graph_input(tmp_path, style, **kwargs),
            config=thread_config("one"),
        )
    return result, client, registry


# --------------------------------------------------------------- stage order ---


async def test_the_meeting_follows_the_protocol_order(tmp_path: Path):
    result, _, _ = await run_department(tmp_path)

    assert [message.speaker_role for message in result["chat_log"]] == [
        AgentRole.RESIDENT,
        AgentRole.RADIOLOGIST,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PHYSIATRIST,
        AgentRole.VICE_DIRECTOR,
        AgentRole.PHYSIATRIST,
        AgentRole.DIRECTOR,
    ]


async def test_the_conservative_advocate_speaks_before_the_director_rules(
    tmp_path: Path,
):
    """It needs something to concede to, and the director something to overrule."""

    result, _, _ = await run_department(tmp_path)
    order = [message.speaker_role for message in result["chat_log"]]
    negotiation = order[order.index(AgentRole.VICE_DIRECTOR) :]

    assert negotiation == [
        AgentRole.VICE_DIRECTOR,
        AgentRole.PHYSIATRIST,
        AgentRole.DIRECTOR,
    ]


async def test_a_ruling_ends_the_meeting_in_one_round(tmp_path: Path):
    result, _, _ = await run_department(tmp_path)

    assert result["round_number"] == 1
    assert result["termination_cause"] is TerminationCause.DIRECTOR_RULED
    assert result["phase"] is DepartmentPhase.COMPLETED
    assert result["completed_at"] is not None


# ---------------------------------------------------------- negotiation loop ---


async def test_an_undecided_director_gets_a_second_round(tmp_path: Path):
    result, _, _ = await run_department(
        tmp_path,
        responses={DirectorRuling: [undecided(), ruling()]},
    )

    assert result["round_number"] == 2
    assert result["termination_cause"] is TerminationCause.DIRECTOR_RULED
    roles = [message.speaker_role for message in result["chat_log"]]
    assert roles.count(AgentRole.VICE_DIRECTOR) == 2
    assert roles.count(AgentRole.DIRECTOR) == 2


async def test_two_rounds_without_a_ruling_is_recorded_as_decision_delay(
    tmp_path: Path,
):
    """The outcome the risk-averse arm is expected to produce, not a failure."""

    result, _, _ = await run_department(
        tmp_path,
        style=DirectorStyle.AVOIDANT,
        responses={DirectorRuling: [undecided(), undecided()]},
    )

    assert result["round_number"] == MAX_NEGOTIATION_ROUNDS
    assert result["termination_cause"] is TerminationCause.DECISION_DELAY
    assert result["phase"] is DepartmentPhase.COMPLETED
    assert final_ruling(result).decision is DirectorDecision.NOT_YET_DECIDED


async def test_the_director_is_told_when_it_is_the_final_round(tmp_path: Path):
    _, client, _ = await run_department(
        tmp_path,
        responses={DirectorRuling: [undecided(), undecided()]},
    )

    director_calls = [
        call for call in client.calls if call["output_model"] is DirectorRuling
    ]
    assert '"is_final_round": false' in director_calls[0]["user_prompt"]
    assert '"is_final_round": true' in director_calls[1]["user_prompt"]


async def test_the_physiatrist_can_move_across_rounds(tmp_path: Path):
    conceded = rehab_assessment(
        stance=ConservativeStance.CONCEDE,
        persuaded_by=["documented progressive weakness"],
    )
    result, _, _ = await run_department(
        tmp_path,
        responses={
            DirectorRuling: [undecided(), ruling()],
            RehabAssessment: [rehab_assessment(), rehab_assessment(), conceded],
        },
    )

    assert result["physiatrist_opening"].stance is ConservativeStance.HOLD
    assert result["physiatrist_latest"].stance is ConservativeStance.CONCEDE


# ------------------------------------------------------------- style routing ---


@pytest.mark.parametrize("style", list(DirectorStyle))
async def test_the_style_reaches_the_director_and_deputy_only(
    tmp_path: Path,
    style: DirectorStyle,
):
    _, client, registry = await run_department(tmp_path, style=style)
    # One style, two vantage points: the director holds it, the deputy faces it.
    prefixes = {
        registry.director.style_prompts[style],
        registry.vice_director.style_prompts[style],
    }
    assert len(prefixes) == 2

    styled = [
        call
        for call in client.calls
        if any(call["system_prompt"].startswith(prefix) for prefix in prefixes)
    ]
    assert len(styled) == 2  # one deputy turn, one director turn
    for call in client.calls:
        if call not in styled:
            for prefix in prefixes:
                assert prefix not in call["system_prompt"]


# ------------------------------------------------------------------ baseline ---


async def test_the_three_arms_share_one_frozen_stage_one_and_two(tmp_path: Path):
    """The manipulated variable must not reach the evidence."""

    store = BaselineStore(tmp_path / "baselines")
    results = {}
    calls_per_arm = []

    for index, style in enumerate(DirectorStyle):
        registry, client = department_with_client()
        async with open_sqlite_checkpointer(tmp_path / f"arm-{index}.sqlite3") as saver:
            graph = create_department_graph(
                registry=registry,
                checkpointer=saver,
                baseline_store=store,
            )
            results[style] = await graph.ainvoke(
                graph_input(tmp_path, style),
                config=thread_config(f"arm-{index}"),
            )
        calls_per_arm.append(len(client.calls))

    assert len(list((tmp_path / "baselines").glob("*.json"))) == 1
    # The first arm pays for stage I-II; the later arms only negotiate.
    assert calls_per_arm[1] == calls_per_arm[2] < calls_per_arm[0]

    first = results[DirectorStyle.DEMOCRATIC]
    for style, result in results.items():
        assert result["director_style"] is style
        assert result["imaging"] == first["imaging"]
        assert result["neurology"] == first["neurology"]
        assert result["anesthesia"] == first["anesthesia"]
        assert result["physiatrist_opening"] == first["physiatrist_opening"]


async def test_every_arm_negotiates_over_the_same_transcript(tmp_path: Path):
    """Identical evidence is not enough: the context must be identical too.

    The deputy and the director are given the discussion so far. If the arm that
    generated the baseline read five statements and the arms that reuse it read
    none, the style would be confounded with how much of the case its holder had
    been told -- and the difference in outcome could not be attributed to style.
    """

    import json

    store = BaselineStore(tmp_path / "baselines")
    seen = []

    for index, style in enumerate(DirectorStyle):
        registry, client = department_with_client()
        async with open_sqlite_checkpointer(tmp_path / f"arm-{index}.sqlite3") as saver:
            graph = create_department_graph(
                registry=registry,
                checkpointer=saver,
                baseline_store=store,
            )
            await graph.ainvoke(
                graph_input(tmp_path, style),
                config=thread_config(f"arm-{index}"),
            )
        opening = next(
            json.loads(call["user_prompt"])["discussion"]
            for call in client.calls
            if call["output_model"] is ViceDirectorOpinion
        )
        seen.append(opening)

    assert len(seen[0]) == 5
    assert seen[1] == seen[0]
    assert seen[2] == seen[0]


async def test_a_reused_baseline_skips_the_calls_but_not_the_transcript(
    tmp_path: Path,
):
    """Reuse must save the model calls without changing what anyone reads."""

    store = BaselineStore(tmp_path / "baselines")
    first, first_client, _ = await run_department(tmp_path, store=store)

    registry, client = department_with_client()
    async with open_sqlite_checkpointer(tmp_path / "second.sqlite3") as saver:
        graph = create_department_graph(
            registry=registry,
            checkpointer=saver,
            baseline_store=store,
        )
        result = await graph.ainvoke(
            graph_input(tmp_path, DirectorStyle.AUTOCRATIC),
            config=thread_config("second"),
        )

    assert result["baseline_reused"] is True
    assert result["imaging"] is not None
    assert len(client.calls) < len(first_client.calls)
    assert [message.speaker_role for message in result["chat_log"]] == [
        AgentRole.RESIDENT,
        AgentRole.RADIOLOGIST,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PHYSIATRIST,
        AgentRole.VICE_DIRECTOR,
        AgentRole.PHYSIATRIST,
        AgentRole.DIRECTOR,
    ]
    assert _spoken(result) == _spoken(first)


def _spoken(result) -> list[tuple]:
    """The transcript as every participant reads it, without its metadata.

    The identifier is excluded deliberately: it carries the arm, and no
    participant ever sees it.
    """

    return [
        (
            message.sequence,
            message.speaker_name,
            message.speaker_role,
            message.content,
            message.turn_number,
        )
        for message in result["chat_log"]
    ]


async def test_the_same_turn_in_two_arms_gets_two_identifiers(tmp_path: Path):
    """Three arms of one case produce the same speakers in the same order.

    Pooling their transcripts on the identifier alone would merge them into one
    run, so the arm is part of the identity.
    """

    store = BaselineStore(tmp_path / "baselines")
    first, _, _ = await run_department(tmp_path, store=store)

    registry, _ = department_with_client()
    async with open_sqlite_checkpointer(tmp_path / "second.sqlite3") as saver:
        graph = create_department_graph(
            registry=registry,
            checkpointer=saver,
            baseline_store=store,
        )
        second = await graph.ainvoke(
            graph_input(tmp_path, DirectorStyle.AVOIDANT),
            config=thread_config("second"),
        )

    first_ids = {message.message_id for message in first["chat_log"]}
    second_ids = {message.message_id for message in second["chat_log"]}
    assert first_ids & second_ids == set()
    assert "case-dept:democratic:resident:1" in first_ids
    assert "case-dept:avoidant:resident:1" in second_ids


def test_replaying_a_baseline_twice_gives_the_identical_statements(tmp_path: Path):
    """The replay is what an arm re-reads on every attempt, so it must not drift.

    The transcript reducer rejects a message that repeats an identifier with
    different content, and a clock reading is different content -- which is why
    the replayed statements are spoken under the hour the baseline was frozen
    rather than under the hour they were read.
    """

    registry, _ = department_with_client()
    payload = department_payload()

    first = baseline_transcript(
        case_id="case-dept",
        style=DirectorStyle.AVOIDANT,
        payload=payload,
        registry=registry,
    )
    second = baseline_transcript(
        case_id="case-dept",
        style=DirectorStyle.AVOIDANT,
        payload=payload,
        registry=registry,
    )

    assert first == second


async def test_a_finished_arm_cannot_be_quietly_run_again(tmp_path: Path):
    """Two discussions must not be blended into one transcript.

    Resuming an interrupted arm continues from its checkpoint and is covered
    elsewhere. Driving a *finished* arm again is a different act, and it fails
    loudly rather than appending a second negotiation to the first.
    """

    from exceptions import InvalidStateUpdateError

    store = BaselineStore(tmp_path / "baselines")
    await run_department(tmp_path, store=store)

    registry, _ = department_with_client()
    async with open_sqlite_checkpointer(tmp_path / "retry.sqlite3") as saver:
        graph = create_department_graph(
            registry=registry,
            checkpointer=saver,
            baseline_store=store,
        )
        config = thread_config("retry")
        await graph.ainvoke(graph_input(tmp_path, DirectorStyle.AVOIDANT), config)
        with pytest.raises(InvalidStateUpdateError):
            await graph.ainvoke(graph_input(tmp_path, DirectorStyle.AVOIDANT), config)


async def test_a_reusing_arm_still_knows_what_the_transcription_was_unsure_of(
    tmp_path: Path,
):
    """Warnings belong to the frozen evidence, not to the run that made it.

    An arm that reuses the baseline and reports no warnings would state that the
    record was read cleanly when it was not.
    """

    warnings = ["[OCR可疑] case-page-001: 第 3 行"]
    store = BaselineStore(tmp_path / "baselines")
    first, _, _ = await run_department(
        tmp_path, store=store, preparation_warnings=warnings
    )
    assert first["preparation_warnings"] == warnings

    registry, _ = department_with_client()
    async with open_sqlite_checkpointer(tmp_path / "second.sqlite3") as saver:
        graph = create_department_graph(
            registry=registry,
            checkpointer=saver,
            baseline_store=store,
        )
        second = await graph.ainvoke(
            graph_input(tmp_path, DirectorStyle.AVOIDANT),
            config=thread_config("second"),
        )

    assert second["baseline_reused"] is True
    assert second["preparation_warnings"] == warnings


async def test_the_arm_records_which_baseline_it_read(tmp_path: Path):
    """Sharing one baseline is the premise of the design; this is the evidence."""

    store = BaselineStore(tmp_path / "baselines")
    keys = []
    for index, style in enumerate(DirectorStyle):
        registry, _ = department_with_client()
        async with open_sqlite_checkpointer(tmp_path / f"arm-{index}.sqlite3") as saver:
            graph = create_department_graph(
                registry=registry,
                checkpointer=saver,
                baseline_store=store,
            )
            result = await graph.ainvoke(
                graph_input(tmp_path, style),
                config=thread_config(f"arm-{index}"),
            )
        keys.append(result["baseline_key"])

    assert all(key is not None for key in keys)
    assert len(set(keys)) == 1


async def test_the_model_each_role_used_is_carried_into_the_run(tmp_path: Path):
    """Without this record, a role whose model changed between batches is invisible."""

    assignments = [
        RoleModelAssignment(role=AgentRole.DIRECTOR, provider="fake", model="fake-model")
    ]
    result, _, _ = await run_department(tmp_path, role_models=assignments)

    assert result["role_models"] == assignments


async def test_the_department_payload_joins_the_existing_record(tmp_path: Path):
    """One record per set of source files, not one per workflow."""

    store = BaselineStore(tmp_path / "baselines")
    sources = case_sources(tmp_path)
    await store.store(
        case_id="case-dept",
        case_preop_md="History.",
        tests_preop_md="Labs.",
        case_summary_md="Summary.",
        radiology_preop_md="Radiology.",
        preparation_warnings=[],
        radiology_report=__import__(
            "tests.factories", fromlist=["radiology_report"]
        ).radiology_report(),
        **sources,
    )

    await run_department(tmp_path, store=store)

    records = list((tmp_path / "baselines").glob("*.json"))
    assert len(records) == 1
    loaded = await store.load(case_id="case-dept", **sources)
    assert loaded.format_version == "v2"
    assert loaded.department is not None
    assert loaded.radiology_report is not None  # the v5 payload survived


# ------------------------------------------------------------------- resume ---


async def test_the_run_resumes_mid_negotiation_without_repeating_turns(
    tmp_path: Path,
):
    registry, client = department_with_client({DirectorRuling: [undecided(), ruling()]})
    database = tmp_path / "resume.sqlite3"
    config = thread_config("resume")

    async with open_sqlite_checkpointer(database) as saver:
        graph = create_department_graph(
            registry=registry,
            checkpointer=saver,
            interrupt_after=["vice_director"],
        )
        interrupted = await graph.ainvoke(graph_input(tmp_path, DirectorStyle.DEMOCRATIC), config=config)
        assert interrupted["round_number"] == 1
        calls_at_pause = len(client.calls)

    async with open_sqlite_checkpointer(database) as saver:
        graph = create_department_graph(registry=registry, checkpointer=saver)
        resumed = await graph.ainvoke(None, config=config)

    assert resumed["phase"] is DepartmentPhase.COMPLETED
    assert len(client.calls) > calls_at_pause
    identifiers = [message.message_id for message in resumed["chat_log"]]
    assert len(identifiers) == len(set(identifiers))


async def test_enum_state_survives_a_checkpoint_round_trip(tmp_path: Path):
    """An unregistered type would come back as a plain string and break `is`."""

    database = tmp_path / "enums.sqlite3"
    config = thread_config("enums")
    registry, _ = department_with_client()

    async with open_sqlite_checkpointer(database) as saver:
        graph = create_department_graph(
            registry=registry,
            checkpointer=saver,
            interrupt_after=["freeze_baseline"],
        )
        await graph.ainvoke(graph_input(tmp_path, DirectorStyle.AVOIDANT), config=config)

    async with open_sqlite_checkpointer(database) as saver:
        graph = create_department_graph(registry=registry, checkpointer=saver)
        snapshot = await graph.aget_state(config)

    assert snapshot.values["phase"] is DepartmentPhase.NEGOTIATION
    assert snapshot.values["director_style"] is DirectorStyle.AVOIDANT
    assert snapshot.values["imaging"].compression_type.value == "ventral"


# -------------------------------------------------------------- thread names ---


def test_department_threads_are_namespaced():
    """The checkpoint tables key on the thread id alone; no graph name reaches it."""

    assert department_thread_id("abc") == "dept-abc"
    assert department_thread_id("dept-abc") == "dept-abc"
    assert is_department_thread(department_thread_id("abc"))
    assert not is_department_thread("abc")


def test_an_empty_thread_id_is_rejected():
    with pytest.raises(ValueError):
        department_thread_id("   ")


# -------------------------------------------------------------- bookkeeping ---


async def test_every_model_call_is_attributed_to_the_role_that_made_it(
    tmp_path: Path,
):
    """ProviderUsage carries no role, and the observer reports tokens per role."""

    result, client, _ = await run_department(tmp_path)

    assert len(result["role_usages"]) == len(client.calls)
    # In order, so that a usage attributed to the wrong role is caught rather
    # than absorbed into a set that happens to hold the same members.
    assert [usage.role for usage in result["role_usages"]] == [
        AgentRole.RESIDENT,
        AgentRole.RADIOLOGIST,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PHYSIATRIST,
        AgentRole.VICE_DIRECTOR,
        AgentRole.PHYSIATRIST,
        AgentRole.DIRECTOR,
    ]


async def test_the_transcript_never_repeats_a_sequence_number(tmp_path: Path):
    result, _, _ = await run_department(
        tmp_path,
        responses={DirectorRuling: [undecided(), ruling()]},
    )

    sequences = [message.sequence for message in result["chat_log"]]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))



async def test_every_ruling_is_kept_not_only_the_last(tmp_path: Path):
    """A director that orders tests in round one and rules in round two did both.

    Keeping only the final ruling would erase the round-one act, and ordering
    further workup is exactly the behaviour the risk-averse arm is measured on.
    """

    result, _, _ = await run_department(
        tmp_path,
        responses={
            DirectorRuling: [
                undecided(ordered_tests=["dynamic MRI"]),
                ruling(),
            ]
        },
    )

    rulings = result["director_rulings"]
    assert [item.decision for item in rulings] == [
        DirectorDecision.NOT_YET_DECIDED,
        DirectorDecision.OPERATE,
    ]
    assert rulings[0].ordered_tests == ["dynamic MRI"]


async def test_each_model_call_records_the_stage_it_was_spent_in(tmp_path: Path):
    """Only the negotiation tokens are comparable between arms.

    The arms that reuse the baseline never make its calls, so a total that mixes
    the two stages would make the first arm look like the longest discussion.
    """

    result, _, _ = await run_department(tmp_path)

    by_stage: dict[str, set] = {}
    for usage in result["role_usages"]:
        by_stage.setdefault(usage.stage, set()).add(usage.role)
    assert by_stage["baseline"] == {
        AgentRole.RESIDENT,
        AgentRole.RADIOLOGIST,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PHYSIATRIST,
    }
    assert by_stage["negotiation"] == {
        AgentRole.VICE_DIRECTOR,
        AgentRole.PHYSIATRIST,
        AgentRole.DIRECTOR,
    }


async def test_the_director_is_handed_what_the_room_left_unresolved(tmp_path: Path):
    """A question nobody received cannot be said to have been ignored.

    The deputy files its questions and objections as structured fields. Nothing
    carried them into the transcript, so in the first real run none of the four
    questions it put to the director reached anything the director could read.
    """

    import json

    _, client, _ = await run_department(
        tmp_path,
        responses={
            ViceDirectorOpinion: vice_opinion(
                disagreement_points=["Should C5-6 be included?"],
                deferred_questions_to_director=["Anterior or posterior?"],
            )
        },
    )

    director_call = next(
        call for call in client.calls if call["output_model"] is DirectorRuling
    )
    raised = json.loads(director_call["user_prompt"])["raised_and_unresolved"]

    assert raised["questions_the_deputy_put_to_you"] == ["Anterior or posterior?"]
    assert raised["objections_the_deputy_raised"] == ["Should C5-6 be included?"]
    assert raised["physiatrist_stance"] == ConservativeStance.HOLD.value


async def test_the_director_sees_which_flags_bear_on_the_decision(tmp_path: Path):
    """Graded, and only the ones that matter.

    The neurologist's structured output never reaches the director; only the
    transcript does. Passing the whole list would restore the flat set of
    equal-weight doubts that the grading exists to break up, so a coexisting
    finding stays in the record and out of this.
    """

    import json

    from department.schemas import FlagRelevance, NeurologyAssessment
    from tests.department_factories import neurology_assessment, red_flag

    _, client, _ = await run_department(
        tmp_path,
        responses={
            NeurologyAssessment: neurology_assessment(
                red_flags=[
                    red_flag(
                        finding="Perineal sensory loss.",
                        relevance=FlagRelevance.CHANGES_PLAN,
                        rationale="Image the conus before choosing levels.",
                    ),
                    red_flag(finding="Absent ankle jerks."),
                ]
            )
        },
    )

    director_call = next(
        call for call in client.calls if call["output_model"] is DirectorRuling
    )
    flags = json.loads(director_call["user_prompt"])["raised_and_unresolved"][
        "neurology_flags_that_bear_on_the_decision"
    ]

    assert [flag["finding"] for flag in flags] == ["Perineal sensory loss."]
    assert flags[0]["relevance"] == FlagRelevance.CHANGES_PLAN.value
