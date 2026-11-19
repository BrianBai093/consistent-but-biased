"""The four-stage department discussion.

Stage order is a fixed edge, not a model decision. There is no moderator
choosing who speaks: the protocol fixes objective reporting, then risk pressure,
then at most two negotiation rounds of deputy -> rehabilitation -> director. The
director is the only participant who decides anything. A run without
negotiation (amendment A4) goes from stage II straight to one ruling.

Stages I and II carry no style prefix and see no negotiation, so they are frozen
once per case and reused by all three arms. That is what lets a difference in
outcome be attributed to the director's style rather than to the evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from department.agents import DepartmentRegistry
from department.schemas import (
    DepartmentBaselinePayload,
    DirectorDecision,
    FlagRelevance,
    RoleModelAssignment,
    RoleUsage,
)
from department.state import (
    MAX_NEGOTIATION_ROUNDS,
    DepartmentPhase,
    DepartmentState,
    TerminationCause,
    arm_label,
    create_initial_department_state,
    department_thread_id,
    deputy_style_for,
)
from schemas import AgentRole, CaseImage, ChatMessage, DirectorStyle, DocumentPage
from services.baseline_store import BaselineStore
from services.llm_client import ModelResult

def department_input(
    *,
    case_id: str,
    director_style: DirectorStyle,
    artifact_dir: str,
    case_pages: Sequence[DocumentPage],
    test_pages: Sequence[DocumentPage],
    images: Sequence[CaseImage],
    case_preop_md: str,
    tests_preop_md: str,
    case_summary_md: str,
    preparation_warnings: Sequence[str] = (),
    role_models: Sequence[RoleModelAssignment] = (),
    deputy_style: DirectorStyle | None = None,
    negotiation: bool = True,
) -> dict:
    """The case, the arm, and the record of which model served each role.

    ``role_models`` is what makes two batches comparable after the fact: roles
    may legitimately run on different models, but a role whose model changed
    between batches is only detectable if each run says what it used.

    ``deputy_style`` and ``negotiation`` keep their defaults in every arm of
    the main design: the deputy faces the director's style, and the two
    negotiate.
    """

    return {
        "case_id": case_id,
        "director_style": director_style,
        "deputy_style": deputy_style_for(director_style, deputy_style, negotiation),
        "negotiation": negotiation,
        "artifact_dir": artifact_dir,
        "case_pages": list(case_pages),
        "test_pages": list(test_pages),
        "images": list(images),
        "case_preop_md": case_preop_md,
        "tests_preop_md": tests_preop_md,
        "case_summary_md": case_summary_md,
        "preparation_warnings": list(preparation_warnings),
        "role_models": list(role_models),
    }


def thread_config(thread_id: str, *, recursion_limit: int = 60) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": department_thread_id(thread_id)},
        "recursion_limit": recursion_limit,
    }


def _message_id(
    case_id: str,
    arm: str,
    role: AgentRole,
    turn_number: int,
) -> str:
    """Identify a message across the whole experiment, not just within one run.

    The arm is part of the identity because the three arms of one case produce
    the same speakers in the same order: pooling their transcripts on the
    identifier alone would silently merge them into one. It never reaches a
    model -- ``transcript()`` passes the sequence, speaker, role and content.
    """

    return f"{case_id}:{arm}:{role.value}:{turn_number}"


def _message(
    state: DepartmentState,
    *,
    role: AgentRole,
    name: str,
    content: str,
    turn_number: int,
) -> ChatMessage:
    return ChatMessage(
        message_id=_message_id(state["case_id"], arm_label(state), role, turn_number),
        sequence=state["sequence_counter"],
        speaker_id=role.value,
        speaker_name=name,
        speaker_role=role,
        content=content,
        turn_number=turn_number,
    )


def baseline_transcript(
    *,
    case_id: str,
    style: DirectorStyle,
    payload: DepartmentBaselinePayload,
    registry: DepartmentRegistry,
    arm: str | None = None,
) -> list[ChatMessage]:
    """Stage I-II re-spoken exactly as the arm that generated it recorded it.

    Loading the frozen findings without their public statements would leave the
    deputy and the director reading five statements in the arm that produced the
    baseline and an empty transcript in the two arms that reuse it: identical
    evidence, different context. That is the one confound this design exists to
    rule out, so the transcript is restored along with the findings.

    Identifiers, order and turn numbers follow the live path exactly, so a
    reused arm is indistinguishable from the arm that paid for the calls.
    """

    speakers = (
        (registry.resident, payload.resident),
        (registry.radiologist, payload.imaging),
        (registry.neurologist, payload.neurology),
        (registry.anesthesiologist, payload.anesthesia),
        (registry.physiatrist, payload.physiatrist),
    )
    return [
        ChatMessage(
            message_id=_message_id(case_id, arm or style.value, agent.role, 1),
            sequence=index,
            speaker_id=agent.role.value,
            speaker_name=agent.display_name,
            speaker_role=agent.role,
            content=output.public_message,
            # The hour the baseline was frozen, not the hour it was re-read, so
            # that replaying it twice produces the identical message.
            timestamp=payload.frozen_at,
            turn_number=1,
        )
        for index, (agent, output) in enumerate(speakers, start=1)
    ]


def _spoke(
    state: DepartmentState,
    result: ModelResult,
    *,
    role: AgentRole,
    name: str,
    turn_number: int = 1,
) -> dict[str, Any]:
    """The bookkeeping every speaking turn shares: transcript and token record."""

    output = result.output
    assert isinstance(output, BaseModel)
    # Read from the phase the node was entered in rather than passed in, so a
    # new node cannot mislabel its own calls. Everything before the baseline is
    # frozen is paid for once per case; everything after it, once per arm.
    stage = "negotiation" if state["phase"] == DepartmentPhase.NEGOTIATION else "baseline"
    return {
        "chat_log": [
            _message(
                state,
                role=role,
                name=name,
                content=output.public_message,
                turn_number=turn_number,
            )
        ],
        "sequence_counter": state["sequence_counter"] + 1,
        "role_usages": [
            *state["role_usages"],
            *(
                RoleUsage(
                    role=role,
                    usage=usage,
                    stage=stage,
                    attempts=result.attempt_count,
                )
                for usage in result.all_usages
            ),
        ],
    }


def create_department_graph(
    *,
    registry: DepartmentRegistry,
    checkpointer: BaseCheckpointSaver,
    baseline_store: BaselineStore | None = None,
    interrupt_after: Sequence[str] | None = None,
):
    builder = StateGraph(DepartmentState)

    async def load_case(state: DepartmentState) -> dict[str, Any]:
        """Start the run, reusing the frozen stage I-II if one exists."""

        initial = create_initial_department_state(
            case_id=state["case_id"],
            director_style=state["director_style"],
            artifact_dir=state["artifact_dir"],
            case_pages=list(state["case_pages"]),
            test_pages=list(state["test_pages"]),
            images=list(state["images"]),
            role_models=list(state.get("role_models") or []),
            deputy_style=state.get("deputy_style"),
            negotiation=state.get("negotiation", True),
        )
        initial["case_preop_md"] = state["case_preop_md"]
        initial["tests_preop_md"] = state["tests_preop_md"]
        initial["case_summary_md"] = state["case_summary_md"]
        initial["preparation_warnings"] = list(state.get("preparation_warnings") or [])
        initial["phase"] = DepartmentPhase.OBJECTIVE_REPORTING

        frozen = await _load_payload(state)
        if frozen is not None:
            payload, key, baseline_warnings = frozen
            replayed = baseline_transcript(
                case_id=state["case_id"],
                style=DirectorStyle(state["director_style"]),
                payload=payload,
                registry=registry,
                arm=arm_label(initial),
            )
            initial.update(
                preparation_warnings=list(baseline_warnings),
                resident_report=payload.resident,
                imaging=payload.imaging,
                imaging_batches=list(payload.imaging_batches),
                neurology=payload.neurology,
                anesthesia=payload.anesthesia,
                physiatrist_opening=payload.physiatrist,
                physiatrist_latest=payload.physiatrist,
                baseline_reused=True,
                baseline_key=key,
                chat_log=replayed,
                sequence_counter=len(replayed) + 1,
            )
        return dict(initial)

    async def _load_payload(
        state,
    ) -> tuple[DepartmentBaselinePayload, str, list[str]] | None:
        if baseline_store is None:
            return None
        baseline = await baseline_store.load(
            case_pages=list(state["case_pages"]),
            test_pages=list(state["test_pages"]),
            images=list(state["images"]),
            case_id=state["case_id"],
        )
        if baseline is None or baseline.department is None:
            return None
        try:
            payload = DepartmentBaselinePayload.model_validate(baseline.department)
        except ValueError:
            # A record written under an older payload shape is a miss, not an
            # error: regenerating is always correct.
            return None
        return payload, baseline.baseline_key, list(baseline.preparation_warnings)

    async def resident_turn(state: DepartmentState) -> dict[str, Any]:
        if state["resident_report"] is not None:
            return {}
        result = await registry.resident.run(
            case_preop_md=state["case_preop_md"],
            tests_preop_md=state["tests_preop_md"],
        )
        return {
            "resident_report": result.output,
            **_spoke(
                state,
                result,
                role=AgentRole.RESIDENT,
                name=registry.resident.display_name,
            ),
        }

    async def radiology_turn(state: DepartmentState) -> dict[str, Any]:
        if state["imaging"] is not None:
            return {}
        if state["images"]:
            result = await registry.radiologist.run(
                case_summary_md=state["case_summary_md"],
                images=list(state["images"]),
            )
        else:
            # A text-only case (a published case report). Its imaging exists
            # only as the record's prose; the upload path cannot reach here,
            # because it refuses a case without films.
            result = await registry.radiologist.read_record(
                case_preop_md=state["case_preop_md"],
            )
        return {
            "imaging": result.output,
            "imaging_batches": list(result.intermediates),
            **_spoke(
                state,
                result,
                role=AgentRole.RADIOLOGIST,
                name=registry.radiologist.display_name,
            ),
        }

    async def neurology_turn(state: DepartmentState) -> dict[str, Any]:
        if state["neurology"] is not None:
            return {}
        result = await registry.neurologist.run(
            resident_report=state["resident_report"],
            imaging=state["imaging"],
            case_preop_md=state["case_preop_md"],
        )
        return {
            "neurology": result.output,
            "phase": DepartmentPhase.RISK_PRESSURE,
            **_spoke(
                state,
                result,
                role=AgentRole.NEUROLOGIST,
                name=registry.neurologist.display_name,
            ),
        }

    async def anesthesia_turn(state: DepartmentState) -> dict[str, Any]:
        if state["anesthesia"] is not None:
            return {}
        result = await registry.anesthesiologist.run(
            resident_report=state["resident_report"],
            case_preop_md=state["case_preop_md"],
            tests_preop_md=state["tests_preop_md"],
        )
        return {
            "anesthesia": result.output,
            **_spoke(
                state,
                result,
                role=AgentRole.ANESTHESIOLOGIST,
                name=registry.anesthesiologist.display_name,
            ),
        }

    async def physiatrist_opening_turn(state: DepartmentState) -> dict[str, Any]:
        if state["physiatrist_opening"] is not None:
            return {}
        result = await registry.physiatrist.run(
            resident_report=state["resident_report"],
            imaging=state["imaging"],
            neurology=state["neurology"],
        )
        return {
            "physiatrist_opening": result.output,
            "physiatrist_latest": result.output,
            **_spoke(
                state,
                result,
                role=AgentRole.PHYSIATRIST,
                name=registry.physiatrist.display_name,
            ),
        }

    async def freeze_baseline(state: DepartmentState) -> dict[str, Any]:
        """Stage I-II is complete; store it for the other arms of this case."""

        update: dict[str, Any] = {"phase": DepartmentPhase.NEGOTIATION}
        if baseline_store is not None and not state["baseline_reused"]:
            update["baseline_key"] = await _store_payload(state)
        return update

    async def _store_payload(state: DepartmentState) -> str | None:
        payload = DepartmentBaselinePayload(
            frozen_at=datetime.now(UTC),
            resident=state["resident_report"],
            imaging=state["imaging"],
            imaging_batches=list(state["imaging_batches"]),
            neurology=state["neurology"],
            anesthesia=state["anesthesia"],
            physiatrist=state["physiatrist_opening"],
        )
        return await baseline_store.store_department(
            case_pages=list(state["case_pages"]),
            test_pages=list(state["test_pages"]),
            images=list(state["images"]),
            case_id=state["case_id"],
            case_preop_md=state["case_preop_md"],
            tests_preop_md=state["tests_preop_md"],
            case_summary_md=state["case_summary_md"],
            preparation_warnings=list(state["preparation_warnings"]),
            payload=payload,
        )

    async def vice_director_turn(state: DepartmentState) -> dict[str, Any]:
        round_number = state["round_number"] + 1
        result = await registry.vice_director.run(
            style=state["deputy_style"],
            chat_log=list(state["chat_log"]),
            round_number=round_number,
        )
        return {
            "round_number": round_number,
            "vice_opinions": [*state["vice_opinions"], result.output],
            **_spoke(
                state,
                result,
                role=AgentRole.VICE_DIRECTOR,
                name=registry.vice_director.display_name,
                turn_number=round_number,
            ),
        }

    async def physiatrist_response_turn(state: DepartmentState) -> dict[str, Any]:
        """The conservative advocate answers before the director rules.

        Speaking here rather than afterwards is what makes "needs strong
        evidence to concede" observable: it has something to concede to, and the
        director has something to overrule.
        """

        result = await registry.physiatrist.respond(
            prior=state["physiatrist_latest"],
            chat_log=list(state["chat_log"]),
            round_number=state["round_number"],
        )
        # Its opening turn already counted as turn 1 in stage II.
        turn_number = state["round_number"] + 1
        return {
            "physiatrist_latest": result.output,
            "physiatrist_responses": [*state["physiatrist_responses"], result.output],
            **_spoke(
                state,
                result,
                role=AgentRole.PHYSIATRIST,
                name=registry.physiatrist.display_name,
                turn_number=turn_number,
            ),
        }

    async def director_turn(state: DepartmentState) -> dict[str, Any]:
        if state["negotiation"]:
            round_number = state["round_number"]
            final_round = round_number >= MAX_NEGOTIATION_ROUNDS
        else:
            # One ruling on the stage I-II record, and nothing after it. The
            # director is told so, as it is told in the last negotiated round;
            # the state keeps zero negotiation rounds, because none took place.
            round_number, final_round = 1, True
        result = await registry.director.run(
            style=state["director_style"],
            chat_log=list(state["chat_log"]),
            round_number=round_number,
            final_round=final_round,
            open_items=_open_items(state),
        )
        ruling = result.output
        update: dict[str, Any] = {
            "director_rulings": [*state["director_rulings"], ruling],
            **_spoke(
                state,
                result,
                role=AgentRole.DIRECTOR,
                name=registry.director.display_name,
                turn_number=round_number,
            ),
        }
        if ruling.decision is not DirectorDecision.NOT_YET_DECIDED:
            update["termination_cause"] = TerminationCause.DIRECTOR_RULED
        elif final_round:
            # Two rounds spent without a ruling. This is a result in its own
            # right -- the outcome the risk-averse arm is expected to produce --
            # not a failure of the run.
            update["termination_cause"] = TerminationCause.DECISION_DELAY
        return update

    def _open_items(state: DepartmentState) -> dict:
        """What the room has formally put on the table this round.

        These live in structured fields that no one downstream could see. A
        question nobody received cannot be said to have been ignored.
        """

        advocate = state["physiatrist_latest"]
        neurology = state["neurology"]
        # Only the flags that bear on a decision. A coexisting finding is real
        # and recorded, but putting it here would restore the flat list of
        # equal-weight doubts that the grading exists to break up.
        flags = [
            {
                "finding": flag.finding,
                "relevance": flag.relevance.value,
                "why": flag.rationale,
            }
            for flag in (neurology.red_flags if neurology else [])
            if flag.relevance is not FlagRelevance.COEXISTING
        ]
        items: dict[str, Any] = {"neurology_flags_that_bear_on_the_decision": flags}
        if state["negotiation"]:
            # Without negotiation there is no deputy, and empty lists here
            # would tell the director that the deputy raised nothing.
            deputy = state["vice_opinions"][-1] if state["vice_opinions"] else None
            items.update(
                questions_the_deputy_put_to_you=list(
                    deputy.deferred_questions_to_director if deputy else []
                ),
                objections_the_deputy_raised=list(
                    deputy.disagreement_points if deputy else []
                ),
                deputy_preferred_option=deputy.preferred_option if deputy else None,
            )
        items.update(
            conservative_preconditions_for_failure=list(
                advocate.preconditions_for_failure if advocate else []
            ),
            physiatrist_stance=advocate.stance.value if advocate else None,
        )
        return items

    async def close_meeting(state: DepartmentState) -> dict[str, Any]:
        return {
            "phase": DepartmentPhase.COMPLETED,
            "completed_at": datetime.now(UTC),
        }

    def after_director(state: DepartmentState) -> str:
        return "close" if state["termination_cause"] is not None else "next_round"

    def after_baseline(state: DepartmentState) -> str:
        return "negotiate" if state["negotiation"] else "rule_alone"

    builder.add_node("load_case", load_case)
    builder.add_node("resident", resident_turn)
    builder.add_node("radiology", radiology_turn)
    builder.add_node("neurology", neurology_turn)
    builder.add_node("anesthesia", anesthesia_turn)
    builder.add_node("physiatrist_opening", physiatrist_opening_turn)
    builder.add_node("freeze_baseline", freeze_baseline)
    builder.add_node("vice_director", vice_director_turn)
    builder.add_node("physiatrist_response", physiatrist_response_turn)
    builder.add_node("director", director_turn)
    builder.add_node("close_meeting", close_meeting)

    builder.add_edge(START, "load_case")
    builder.add_edge("load_case", "resident")
    builder.add_edge("resident", "radiology")
    builder.add_edge("radiology", "neurology")
    builder.add_edge("neurology", "anesthesia")
    builder.add_edge("anesthesia", "physiatrist_opening")
    builder.add_edge("physiatrist_opening", "freeze_baseline")
    builder.add_conditional_edges(
        "freeze_baseline",
        after_baseline,
        {"negotiate": "vice_director", "rule_alone": "director"},
    )
    builder.add_edge("vice_director", "physiatrist_response")
    builder.add_edge("physiatrist_response", "director")
    builder.add_conditional_edges(
        "director",
        after_director,
        {"next_round": "vice_director", "close": "close_meeting"},
    )
    builder.add_edge("close_meeting", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=list(interrupt_after or []),
        name="department-decision",
    )
