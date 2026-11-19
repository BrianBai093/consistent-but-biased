"""Shared state for the department-hierarchy workflow.

Deliberately much smaller than the v5 MDT state. There is no moderator routing,
no shared operative draft, no blind review, and no per-speaker quota, because
the stage order is fixed and the director alone decides. What is left is the
evidence, the transcript, and where in the four stages the run currently is.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, TypedDict

from department.schemas import (
    AnesthesiaAssessment,
    DirectorRuling,
    ImagingFindings,
    NeurologyAssessment,
    RehabAssessment,
    ResidentReport,
    RoleModelAssignment,
    RoleUsage,
    ViceDirectorOpinion,
)
from schemas import CaseImage, ChatMessage, DirectorStyle, DocumentPage
from state import append_unique_messages

DEPARTMENT_WORKFLOW_VERSION = "dept-v2"

#: A thread belongs to exactly one workflow, but the checkpoint tables are keyed
#: on the thread id alone -- no graph name reaches the key. The prefix is what
#: keeps a department thread from being driven by the v5 graph and vice versa.
THREAD_PREFIX = "dept-"


class DepartmentPhase(str, Enum):
    CREATED = "created"
    OBJECTIVE_REPORTING = "objective_reporting"  # stage I
    RISK_PRESSURE = "risk_pressure"  # stage II
    NEGOTIATION = "negotiation"  # stages III-IV
    COMPLETED = "completed"


class TerminationCause(str, Enum):
    """Why the negotiation stopped.

    ``DECISION_DELAY`` is not only a stopping condition: it is the outcome the
    risk-averse arm is expected to produce, so it has to be recorded as a result
    rather than as an error.
    """

    DIRECTOR_RULED = "director_ruled"
    DECISION_DELAY = "decision_delay"


MAX_NEGOTIATION_ROUNDS = 2


class DepartmentState(TypedDict):
    workflow_version: str
    case_id: str
    director_style: DirectorStyle
    #: The prefix the deputy reads. It equals ``director_style`` in every arm of
    #: the main design; it differs only when one role carries the style and the
    #: other reads the neutral prefix (amendment A4), and it is ``None`` when the
    #: deputy never speaks.
    deputy_style: DirectorStyle | None
    #: False when the director rules once, alone, straight after stage II: no
    #: deputy turn, no advocate reply. The control for whether a style needs the
    #: discussion to act (amendment A4).
    negotiation: bool
    artifact_dir: str

    case_pages: list[DocumentPage]
    test_pages: list[DocumentPage]
    images: list[CaseImage]
    case_preop_md: str
    tests_preop_md: str
    case_summary_md: str
    #: What the transcription was unsure of. It belongs to the frozen evidence:
    #: an arm that reuses the baseline and reports no warnings would say the
    #: record was read cleanly when it was not.
    preparation_warnings: list[str]

    # Stage I and II, frozen and shared by every arm of this case.
    resident_report: ResidentReport | None
    imaging: ImagingFindings | None
    #: The per-batch readings the imaging report was consolidated from. The
    #: radiologist is the only role that sees the films, so this is the one
    #: place a merged report can be checked against what was actually read.
    imaging_batches: list[ImagingFindings]
    neurology: NeurologyAssessment | None
    anesthesia: AnesthesiaAssessment | None
    physiatrist_opening: RehabAssessment | None
    baseline_reused: bool
    #: Which frozen preparation this arm actually read. Without it there is no
    #: way to show after the fact that the three arms argued over one baseline.
    baseline_key: str | None

    # Stages III-IV, which differ by arm.
    round_number: int
    # One entry per round for all three negotiating roles, not one slot holding
    # the last. A director that orders tests in round one and rules in round two
    # performed both acts; a deputy that stated a preference in round one and
    # withdrew it in round two is the self-censorship the experiment exists to
    # observe. Keeping only the final value erases exactly what is measured.
    vice_opinions: list[ViceDirectorOpinion]
    physiatrist_responses: list[RehabAssessment]
    director_rulings: list[DirectorRuling]
    physiatrist_latest: RehabAssessment | None

    chat_log: Annotated[list[ChatMessage], append_unique_messages]
    sequence_counter: int

    phase: DepartmentPhase
    termination_cause: TerminationCause | None

    role_models: list[RoleModelAssignment]
    role_usages: list[RoleUsage]
    started_at: datetime
    completed_at: datetime | None


def create_initial_department_state(
    *,
    case_id: str,
    director_style: DirectorStyle,
    artifact_dir: str,
    case_pages: list[DocumentPage],
    test_pages: list[DocumentPage],
    images: list[CaseImage],
    role_models: list[RoleModelAssignment],
    deputy_style: DirectorStyle | None = None,
    negotiation: bool = True,
) -> DepartmentState:
    return DepartmentState(
        workflow_version=DEPARTMENT_WORKFLOW_VERSION,
        case_id=case_id,
        director_style=director_style,
        deputy_style=deputy_style_for(director_style, deputy_style, negotiation),
        negotiation=negotiation,
        artifact_dir=artifact_dir,
        case_pages=case_pages,
        test_pages=test_pages,
        images=images,
        case_preop_md="",
        tests_preop_md="",
        case_summary_md="",
        preparation_warnings=[],
        resident_report=None,
        imaging=None,
        imaging_batches=[],
        neurology=None,
        anesthesia=None,
        physiatrist_opening=None,
        baseline_reused=False,
        baseline_key=None,
        round_number=0,
        vice_opinions=[],
        physiatrist_responses=[],
        director_rulings=[],
        physiatrist_latest=None,
        chat_log=[],
        sequence_counter=1,
        phase=DepartmentPhase.CREATED,
        termination_cause=None,
        role_models=list(role_models),
        role_usages=[],
        started_at=datetime.now(UTC),
        completed_at=None,
    )


def deputy_style_for(
    director_style: DirectorStyle,
    deputy_style: DirectorStyle | None,
    negotiation: bool,
) -> DirectorStyle | None:
    """The deputy faces the director's style unless a run says otherwise.

    Without negotiation the deputy never speaks, so it reads nothing.
    """

    if not negotiation:
        return None
    return director_style if deputy_style is None else deputy_style


def arm_label(state: DepartmentState) -> str:
    """Name the arm the way the run's message identifiers carry it.

    An arm of the main design keeps the bare style, so its identifiers are the
    ones every earlier run recorded. The A4 arms share a director style with a
    main arm and must not share its identifiers.
    """

    style = DirectorStyle(state["director_style"])
    if not state.get("negotiation", True):
        return f"{style.value}-alone"
    deputy = state.get("deputy_style")
    if deputy is None or DirectorStyle(deputy) is style:
        return style.value
    return f"chair-{style.value}-deputy-{DirectorStyle(deputy).value}"


def final_ruling(state: DepartmentState) -> DirectorRuling | None:
    """The ruling that ended the meeting, or None if the director never spoke."""

    rulings = state["director_rulings"]
    return rulings[-1] if rulings else None


def department_thread_id(raw: str) -> str:
    """Namespace a thread id so the two workflows cannot drive each other."""

    identifier = raw.strip()
    if not identifier:
        raise ValueError("thread_id must not be empty")
    return identifier if identifier.startswith(THREAD_PREFIX) else THREAD_PREFIX + identifier


def is_department_thread(thread_id: str) -> bool:
    return thread_id.startswith(THREAD_PREFIX)
