"""Single-process background execution, controls, snapshots, and event emission."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Callable
from uuid import NAMESPACE_URL, uuid4, uuid5

from langgraph.checkpoint.base import BaseCheckpointSaver

from agents import AgentRegistry
from api_schemas import RunControlResponse, RunRecord, RunSnapshotResponse, RunStatus
from exceptions import (
    InvalidRunTransitionError,
    ModelStageError,
    ModelRetryExhaustedError,
    ModelTimeoutError,
    RunNotCompleteError,
    RunNotFoundError,
    ServiceError,
)
from graph import create_complete_mdt_graph, opening_input, thread_config
from schemas import (
    AgentRole,
    CaseInput,
    DirectorStyle,
    ChatMessage,
    FinalPlan,
    ModelFailureRecord,
    PreparedDocuments,
    RunMetadata,
    RunPhase,
)
from services.artifact_store import read_markdown, write_markdown
from services.baseline_store import BaselineStore
from services.radiology_cache import RadiologyCache
from services.run_store import RunStore
from state import CURRENT_WORKFLOW_VERSION


logger = logging.getLogger(__name__)
RegistryFactory = Callable[[], AgentRegistry]

_MODEL_STAGE_LABELS = {
    "CASE_DOCUMENT_TRANSCRIPTION": "Case-document transcription",
    "TEST_DOCUMENT_TRANSCRIPTION": "Test-report transcription",
    "CASE_SUMMARIZATION": "Case summarization",
    "RADIOLOGY_REVIEW": "Initial radiology review",
    "final_plan_generation": "Final plan synthesis",
    "final_plan_review": "Independent final plan review",
    "final_plan_revision": "Final plan revision",
}
_MODEL_FAILURE_REASONS = {
    "MODEL_TIMEOUT": "the model request timed out after retries",
    "MODEL_RATE_LIMITED": "the model service kept rate-limiting the request",
    "MODEL_CONNECTION_FAILED": "the model service could not be reached after retries",
    "MODEL_SERVICE_FAILED": "the model service kept returning a server error",
    "MODEL_REQUEST_REJECTED": "the model service rejected the request",
    "MODEL_INVALID_OUTPUT": "the model repeatedly returned invalid structured output",
    "MODEL_INVOCATION_FAILED": "the model request could not be completed",
}
_SINGLE_REQUEST_FAILURE_REASONS = {
    "MODEL_TIMEOUT": "the model request timed out",
    "MODEL_RATE_LIMITED": "the model service rate-limited the request",
    "MODEL_CONNECTION_FAILED": "the model service could not be reached",
    "MODEL_SERVICE_FAILED": "the model service returned a server error",
    "MODEL_REQUEST_REJECTED": "the model service rejected the request",
    "MODEL_INVALID_OUTPUT": "the model returned invalid structured output",
    "MODEL_INVOCATION_FAILED": "the model request could not be completed",
}

_RETRYABLE_FINALIZATION_NODES = {
    "final_plan_generation": "draft_final_plan",
    "final_plan_review": "review_final_plan",
    "final_plan_revision": "revise_final_plan",
}


def _retryable_finalization_node(error_code: str | None) -> str | None:
    if not error_code:
        return None
    for stage, node in _RETRYABLE_FINALIZATION_NODES.items():
        if error_code.startswith(f"{stage}_"):
            return node
    return None


def _external_input_update_node(values: dict) -> str:
    """Anchor an addendum update without relying on LangGraph's writer inference.

    Parallel document-preparation branches make the latest writer ambiguous.  The
    explicit anchor preserves the same downstream route while allowing paused
    checkpoints created before or after the join to accept an addendum.
    """

    phase = values.get("phase")
    if phase in {RunPhase.INPUT_NORMALIZED, RunPhase.INPUT_NORMALIZED.value}:
        return "prepare_case_document"
    if phase in {RunPhase.DOCUMENT_PREPARATION, RunPhase.DOCUMENT_PREPARATION.value}:
        return "prepare_documents"
    if phase in {RunPhase.CASE_SUMMARIZATION, RunPhase.CASE_SUMMARIZATION.value}:
        return "summarize_case"
    if phase in {RunPhase.MODERATOR_OPENING, RunPhase.MODERATOR_OPENING.value}:
        return "moderator_opening"
    if phase in {RunPhase.RADIOLOGY_REVIEW, RunPhase.RADIOLOGY_REVIEW.value}:
        return "radiologist_initial_report"
    if phase in {
        RunPhase.INDEPENDENT_SPECIALIST_ROUND,
        RunPhase.INDEPENDENT_SPECIALIST_ROUND.value,
    }:
        return "moderator_review"
    if phase in {
        RunPhase.AWAITING_DYNAMIC_ROUTING,
        RunPhase.AWAITING_DYNAMIC_ROUTING.value,
    }:
        speaker_counts = values.get("speaker_counts") or {}
        initial_text_roles = (
            AgentRole.SPINE_SURGEON,
            AgentRole.NEUROLOGIST,
            AgentRole.ANESTHESIOLOGIST,
            AgentRole.PERIOPERATIVE_PHYSICIAN,
        )
        if all(speaker_counts.get(role.value, 0) >= 1 for role in initial_text_roles):
            return "independent_specialist_round"
        return "moderator_review"
    if phase in {RunPhase.AWAITING_FINALIZATION, RunPhase.AWAITING_FINALIZATION.value}:
        if values.get("final_plan_review") is not None:
            return "review_final_plan"
        if values.get("final_plan_draft") is not None:
            return "draft_final_plan"
        return "moderator_route"
    if phase in {RunPhase.DYNAMIC_DISCUSSION, RunPhase.DYNAMIC_DISCUSSION.value}:
        last_message = (values.get("chat_log") or [])[-1:]
        if last_message:
            role = last_message[0].speaker_role
            if role is AgentRole.MODERATOR:
                return "moderator_route"
            if role is AgentRole.RADIOLOGIST:
                return "radiologist_dynamic"
            return role.value
        return "moderator_route"
    return "prepare_case_document"


def _safe_pause_boundary(values: dict) -> bool:
    """Do not pause between the parallel preparation branches and their join."""

    phase = values.get("phase")
    return phase not in {RunPhase.INPUT_NORMALIZED, RunPhase.INPUT_NORMALIZED.value}


def _terminal_model_failure(
    *,
    thread_id: str,
    error: ModelStageError,
) -> ModelFailureRecord:
    cause = error.cause
    if isinstance(cause, ModelRetryExhaustedError):
        categories = [item.value for item in cause.attempt_categories]
        attempt_count = cause.attempt_count
        max_attempts = cause.max_attempts
    else:
        categories = [error.category.value]
        attempt_count = 1
        max_attempts = 1
    role = (
        AgentRole.RADIOLOGIST
        if error.stage == "RADIOLOGY_REVIEW"
        else AgentRole.MODERATOR
    )
    return ModelFailureRecord(
        failure_id=str(
            uuid5(NAMESPACE_URL, f"{thread_id}:{error.stage}:terminal-model-failure")
        ),
        agent_role=role,
        stage=error.stage,
        outcome="failed",
        category=error.category.value,
        attempt_categories=categories,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        safe_message=_model_stage_error_message(error),
    )


def _model_stage_error_message(error: ModelStageError) -> str:
    stage = _MODEL_STAGE_LABELS.get(error.stage, "A core preparation stage")
    if error.stage in {
        "final_plan_generation",
        "final_plan_review",
        "final_plan_revision",
    }:
        reason = _MODEL_FAILURE_REASONS[error.category.value]
        return (
            f"{stage} failed because {reason}. No unreviewed final plan was published; "
            "prepared evidence and discussion remain available."
        )
    if error.item_id:
        reason = _SINGLE_REQUEST_FAILURE_REASONS[error.category.value]
        return (
            f"{stage} failed for {error.item_id} because {reason}. "
            "The run did not enter MDT discussion."
        )
    reason = _MODEL_FAILURE_REASONS[error.category.value]
    return f"{stage} failed because {reason}. The run did not enter MDT discussion."


class RunManager:
    def __init__(
        self,
        *,
        checkpointer: BaseCheckpointSaver,
        run_store: RunStore,
        registry_factory: RegistryFactory,
        max_additional_input_chars: int,
        radiology_cache: RadiologyCache | None = None,
        baseline_store: BaselineStore | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.run_store = run_store
        self.registry_factory = registry_factory
        self.max_additional_input_chars = max_additional_input_chars
        self.radiology_cache = radiology_cache
        self.baseline_store = baseline_store
        self._registry: AgentRegistry | None = None
        self._graph = None
        self._manager_lock = asyncio.Lock()
        self._run_locks: dict[str, asyncio.Lock] = {}
        self._control_locks: dict[str, asyncio.Lock] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._shutting_down = False

    async def ensure_ready(self) -> None:
        async with self._manager_lock:
            if self._graph is None:
                self._registry = self.registry_factory()
                self._graph = create_complete_mdt_graph(
                    agent_registry=self._registry,
                    checkpointer=self.checkpointer,
                    radiology_progress_sink=(
                        self.run_store.record_radiology_progress
                    ),
                    radiology_cache=self.radiology_cache,
                    baseline_store=self.baseline_store,
                )

    async def reconcile_startup(self) -> None:
        interrupted = await self.run_store.mark_active_runs_interrupted()
        for record in interrupted:
            await self.run_store.append_event(
                record.thread_id,
                "run.interrupted",
                _status_event_data(record),
            )

    async def shutdown(self) -> None:
        self._shutting_down = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        interrupted = await self.run_store.mark_nonterminal_runs_interrupted()
        for record in interrupted:
            await self.run_store.append_event(
                record.thread_id,
                "run.interrupted",
                _status_event_data(record),
            )
        await self._close_model_clients()

    async def start_case(
        self,
        *,
        thread_id: str,
        case: CaseInput,
        director_style: DirectorStyle | None = None,
    ) -> RunRecord:
        await self.ensure_ready()
        record = await self.run_store.create_run(
            thread_id,
            case.case_id,
            director_style=director_style,
        )
        async with self._manager_lock:
            self._schedule(
                thread_id,
                opening_input(case, director_style=director_style),
                resumed=False,
            )
        return record

    async def get_snapshot(
        self,
        thread_id: str,
        *,
        after_sequence: int = 0,
    ) -> RunSnapshotResponse:
        record = await self._require_run(thread_id)
        values = await self._checkpoint_values(thread_id)
        messages = _messages_after(values, after_sequence)
        phase = RunPhase(values["phase"]) if values and values.get("phase") else None
        final_plan = (
            FinalPlan.model_validate(values["final_plan"])
            if values and values.get("final_plan") is not None
            else None
        )
        metadata = (
            RunMetadata.model_validate(values["run_metadata"])
            if values and values.get("run_metadata") is not None
            else None
        )
        stored_failures = await self.run_store.list_model_failures(thread_id)
        if metadata is not None and stored_failures:
            failures_by_id = {
                failure.failure_id: failure
                for failure in [*metadata.model_failures, *stored_failures]
            }
            metadata = metadata.model_copy(
                update={"model_failures": list(failures_by_id.values())}
            )
        prepared_documents = None
        if values:
            artifact_dir = values.get("artifact_dir")
            artifact_values = ["", "", "", ""]
            if artifact_dir:
                artifact_values = await asyncio.gather(
                    read_markdown(artifact_dir, "case_preop.md"),
                    read_markdown(artifact_dir, "tests_preop.md"),
                    read_markdown(artifact_dir, "case_summary.md"),
                    read_markdown(artifact_dir, "radiology_preop.md"),
                )
            prepared_documents = PreparedDocuments(
                case_preop_md=values.get("case_preop_md") or artifact_values[0],
                tests_preop_md=values.get("tests_preop_md") or artifact_values[1],
                case_summary_md=values.get("case_summary_md") or artifact_values[2],
                radiology_preop_md=(
                    values.get("radiology_preop_md") or artifact_values[3]
                ),
                warnings=values.get("preparation_warnings", []),
            )
        radiology_progress = await self.run_store.get_radiology_progress(thread_id)
        return RunSnapshotResponse(
            case_id=record.case_id,
            thread_id=record.thread_id,
            status=record.status,
            director_style=record.director_style,
            phase=phase,
            new_messages=messages,
            is_complete=record.status is RunStatus.COMPLETED and final_plan is not None,
            final_plan=final_plan,
            run_metadata=metadata,
            prepared_documents=prepared_documents,
            radiology_progress=radiology_progress,
            error_code=record.error_code,
            error_message=record.error_message,
        )

    async def get_result(self, thread_id: str) -> RunSnapshotResponse:
        record = await self._require_run(thread_id)
        if record.status is not RunStatus.COMPLETED:
            raise RunNotCompleteError(thread_id)
        return await self.get_snapshot(thread_id)

    async def request_pause(self, thread_id: str) -> RunControlResponse:
        async with self._control_lock(thread_id):
            record = await self._require_run(thread_id)
            if record.status in {RunStatus.PAUSED, RunStatus.PAUSE_REQUESTED}:
                return _control_response(record)
            if record.status is RunStatus.INTERRUPTED:
                return _control_response(record)
            if record.status not in {RunStatus.RUNNING, RunStatus.QUEUED}:
                raise InvalidRunTransitionError(
                    thread_id,
                    f"A run in {record.status.value} state cannot be paused.",
                )
            updated = await self.run_store.update_status(
                thread_id,
                RunStatus.PAUSE_REQUESTED,
            )
            assert updated is not None
            await self.run_store.append_event(
                thread_id,
                "run.pause_requested",
                _status_event_data(updated),
            )
        return _control_response(updated)

    async def request_cancel(self, thread_id: str) -> RunControlResponse:
        async with self._control_lock(thread_id):
            record = await self._require_run(thread_id)
            if record.status in {RunStatus.CANCELLED, RunStatus.CANCEL_REQUESTED}:
                return _control_response(record)
            if record.status.is_terminal:
                raise InvalidRunTransitionError(
                    thread_id,
                    f"A run in {record.status.value} state cannot be cancelled.",
                )
            updated = await self.run_store.update_status(
                thread_id,
                RunStatus.CANCEL_REQUESTED,
            )
            assert updated is not None
            await self.run_store.append_event(
                thread_id,
                "run.cancel_requested",
                _status_event_data(updated),
            )
            if record.status in {RunStatus.PAUSED, RunStatus.INTERRUPTED}:
                updated = await self._mark_cancelled(thread_id)
        return _control_response(updated)

    async def resume(self, thread_id: str) -> RunControlResponse:
        await self._require_run(thread_id)
        await self.ensure_ready()
        async with self._manager_lock:
            async with self._control_lock(thread_id):
                record = await self._require_run(thread_id)
                retry_node = (
                    _retryable_finalization_node(record.error_code)
                    if record.status is RunStatus.FAILED
                    else None
                )
                retrying_failed_finalization = retry_node is not None
                if record.status.is_terminal and not retrying_failed_finalization:
                    raise InvalidRunTransitionError(
                        thread_id,
                        f"A run in {record.status.value} state cannot be resumed.",
                    )
                if record.status in {
                    RunStatus.PAUSE_REQUESTED,
                    RunStatus.CANCEL_REQUESTED,
                }:
                    raise InvalidRunTransitionError(
                        thread_id,
                        f"A run in {record.status.value} state cannot be resumed.",
                    )
                active_task = self._tasks.get(thread_id)
                if active_task is not None and not active_task.done():
                    return _control_response(record)
                if record.status is RunStatus.RUNNING:
                    return _control_response(record)
                if not retrying_failed_finalization and record.status not in {
                    RunStatus.PAUSED,
                    RunStatus.INTERRUPTED,
                }:
                    raise InvalidRunTransitionError(
                        thread_id,
                        f"A run in {record.status.value} state cannot be resumed.",
                    )
                snapshot = await self._graph.aget_state(thread_config(thread_id))
                values = dict(snapshot.values) if snapshot.values else None
                if (
                    values is not None
                    and values.get("workflow_version") != CURRENT_WORKFLOW_VERSION
                ):
                    raise ServiceError(
                        "WORKFLOW_VERSION_INCOMPATIBLE",
                        "This unfinished run uses an older workflow version and cannot "
                        "be resumed safely. Start a new run with the original inputs.",
                        status_code=409,
                        thread_id=thread_id,
                    )
                if retrying_failed_finalization and (
                    values is None
                    or values.get("final_plan") is not None
                    or tuple(snapshot.next) != (retry_node,)
                ):
                    raise InvalidRunTransitionError(
                        thread_id,
                        "The failed finalization checkpoint cannot be resumed safely.",
                    )
                updated = await self.run_store.update_status(thread_id, RunStatus.QUEUED)
                assert updated is not None
                self._schedule(thread_id, None, resumed=True)
                return _control_response(updated)

    async def add_input(self, thread_id: str, content: str) -> RunControlResponse:
        async with self._control_lock(thread_id):
            return await self._add_input_locked(thread_id, content)

    async def _add_input_locked(
        self,
        thread_id: str,
        content: str,
    ) -> RunControlResponse:
        record = await self._require_run(thread_id)
        if record.status not in {RunStatus.PAUSED, RunStatus.INTERRUPTED}:
            raise InvalidRunTransitionError(
                thread_id,
                "Additional input is accepted only while a run is paused or interrupted.",
            )
        normalized = content.strip()
        if not normalized:
            raise ServiceError(
                "INVALID_ADDITIONAL_INPUT",
                "Additional input must not be empty.",
                status_code=422,
                thread_id=thread_id,
            )
        if len(normalized) > self.max_additional_input_chars:
            raise ServiceError(
                "ADDITIONAL_INPUT_TOO_LARGE",
                "Additional input exceeds the configured length limit.",
                status_code=413,
                thread_id=thread_id,
            )
        await self.ensure_ready()
        values = await self._checkpoint_values(thread_id)
        if not values or not values.get("case_preop_md"):
            raise InvalidRunTransitionError(
                thread_id,
                "The run has no checkpointed case state to update.",
            )
        if values.get("workflow_version") != CURRENT_WORKFLOW_VERSION:
            raise ServiceError(
                "WORKFLOW_VERSION_INCOMPATIBLE",
                "This unfinished run uses an older workflow version. Start a new run "
                "before adding information.",
                status_code=409,
                thread_id=thread_id,
            )
        addendum_id = uuid4().hex
        created_at = datetime.now(UTC)
        label = (
            f"\n\n<!-- source: addendum-{addendum_id} -->\n\n"
            f"[Case addendum {addendum_id} submitted {created_at.isoformat()}]\n"
            f"{normalized}"
        )
        updated_case = f"{values['case_preop_md']}{label}"
        existing_summary = values.get("case_summary_md", "")
        updated_summary = (
            f"{existing_summary}{label}" if existing_summary else existing_summary
        )
        await self._graph.aupdate_state(
            thread_config(thread_id),
            {
                "case_preop_md": updated_case,
                "case_summary_md": updated_summary,
            },
            as_node=_external_input_update_node(values),
        )
        artifact_dir = values.get("artifact_dir")
        if artifact_dir:
            await write_markdown(artifact_dir, "case_preop.md", updated_case)
            if updated_summary:
                await write_markdown(artifact_dir, "case_summary.md", updated_summary)
        await self.run_store.add_addendum(
            addendum_id=addendum_id,
            thread_id=thread_id,
            content=normalized,
            created_at=created_at,
        )
        await self.run_store.append_event(
            thread_id,
            "input.accepted",
            {
                "thread_id": thread_id,
                "case_id": record.case_id,
                "addendum_id": addendum_id,
                "created_at": created_at.isoformat(),
            },
        )
        return _control_response(record)

    async def _drive(self, thread_id: str, graph_input, *, resumed: bool) -> None:
        lock = self._run_locks.setdefault(thread_id, asyncio.Lock())
        try:
            async with lock:
                async with self._control_lock(thread_id):
                    record = await self._require_run(thread_id)
                    if record.status is RunStatus.CANCEL_REQUESTED:
                        await self._mark_cancelled(thread_id)
                        return
                    if record.status is RunStatus.PAUSE_REQUESTED:
                        await self._mark_paused(thread_id)
                        return
                    running = await self.run_store.update_status(
                        thread_id,
                        RunStatus.RUNNING,
                    )
                    assert running is not None
                    await self.run_store.append_event(
                        thread_id,
                        "run.resumed" if resumed else "run.started",
                        _status_event_data(running),
                    )
                last_sequence = running.last_message_sequence
                async for part in self._graph.astream(
                    graph_input,
                    config=thread_config(thread_id),
                    stream_mode="values",
                    version="v2",
                ):
                    if part.get("type") != "values":
                        continue
                    values = part.get("data") or {}
                    for failure in _model_failures(values):
                        await self.run_store.record_model_failure(
                            thread_id,
                            failure,
                        )
                    messages = _messages_after(values, last_sequence)
                    for message in messages:
                        await self.run_store.append_event(
                            thread_id,
                            "chat.message",
                            message.model_dump(mode="json"),
                        )
                        last_sequence = message.sequence
                        await self.run_store.update_last_message_sequence(
                            thread_id,
                            last_sequence,
                        )
                    async with self._control_lock(thread_id):
                        phase = values.get("phase")
                        if phase == RunPhase.COMPLETED or phase == RunPhase.COMPLETED.value:
                            completed = await self.run_store.update_status(
                                thread_id,
                                RunStatus.COMPLETED,
                            )
                            assert completed is not None
                            await self.run_store.append_event(
                                thread_id,
                                "run.completed",
                                _status_event_data(completed),
                            )
                            return
                        control = await self._require_run(thread_id)
                        if control.status is RunStatus.CANCEL_REQUESTED:
                            await self._mark_cancelled(thread_id)
                            return
                        if (
                            control.status is RunStatus.PAUSE_REQUESTED
                            and _safe_pause_boundary(values)
                        ):
                            await self._mark_paused(thread_id)
                            return
                snapshot = await self._checkpoint_values(thread_id)
                if snapshot and snapshot.get("phase") in {
                    RunPhase.COMPLETED,
                    RunPhase.COMPLETED.value,
                }:
                    completed = await self.run_store.update_status(
                        thread_id,
                        RunStatus.COMPLETED,
                    )
                    assert completed is not None
                    await self.run_store.append_event(
                        thread_id,
                        "run.completed",
                        _status_event_data(completed),
                    )
                else:
                    interrupted = await self.run_store.update_status(
                        thread_id,
                        RunStatus.INTERRUPTED,
                    )
                    assert interrupted is not None
                    await self.run_store.append_event(
                        thread_id,
                        "run.interrupted",
                        _status_event_data(interrupted),
                    )
        except asyncio.CancelledError:
            if self._shutting_down:
                record = await self.run_store.update_status(
                    thread_id,
                    RunStatus.INTERRUPTED,
                )
                if record is not None:
                    await self.run_store.append_event(
                        thread_id,
                        "run.interrupted",
                        _status_event_data(record),
                    )
            raise
        except ModelStageError as exc:
            logger.exception(
                "Background model stage failed",
                extra={"thread_id": thread_id, "model_stage": exc.stage},
            )
            await self.run_store.record_model_failure(
                thread_id,
                _terminal_model_failure(thread_id=thread_id, error=exc),
            )
            failed = await self.run_store.update_status(
                thread_id,
                RunStatus.FAILED,
                error_code=f"{exc.stage}_{exc.category.value}",
                error_message=_model_stage_error_message(exc),
            )
            if failed is not None:
                await self.run_store.append_event(
                    thread_id,
                    "run.failed",
                    _status_event_data(failed),
                )
        except ModelTimeoutError:
            logger.exception("Background model request timed out", extra={"thread_id": thread_id})
            failed = await self.run_store.update_status(
                thread_id,
                RunStatus.FAILED,
                error_code="MODEL_TIMEOUT",
                error_message=(
                    "A model request timed out after retries. The run stopped before a "
                    "complete plan was generated."
                ),
            )
            if failed is not None:
                await self.run_store.append_event(
                    thread_id,
                    "run.failed",
                    _status_event_data(failed),
                )
        except Exception:
            logger.exception("Background run failed", extra={"thread_id": thread_id})
            failed = await self.run_store.update_status(
                thread_id,
                RunStatus.FAILED,
                error_code="RUN_EXECUTION_FAILED",
                error_message="The run failed due to an unexpected execution error.",
            )
            if failed is not None:
                await self.run_store.append_event(
                    thread_id,
                    "run.failed",
                    _status_event_data(failed),
                )
        finally:
            async with self._manager_lock:
                current = asyncio.current_task()
                if self._tasks.get(thread_id) is current:
                    self._tasks.pop(thread_id, None)

    async def _mark_cancelled(self, thread_id: str) -> RunRecord:
        cancelled = await self.run_store.update_status(thread_id, RunStatus.CANCELLED)
        assert cancelled is not None
        await self.run_store.append_event(
            thread_id,
            "run.cancelled",
            _status_event_data(cancelled),
        )
        return cancelled

    async def _mark_paused(self, thread_id: str) -> RunRecord:
        paused = await self.run_store.update_status(thread_id, RunStatus.PAUSED)
        assert paused is not None
        await self.run_store.append_event(
            thread_id,
            "run.paused",
            _status_event_data(paused),
        )
        return paused

    def _control_lock(self, thread_id: str) -> asyncio.Lock:
        return self._control_locks.setdefault(thread_id, asyncio.Lock())

    async def _close_model_clients(self) -> None:
        if self._registry is None:
            return
        agents = (
            self._registry.case_document,
            self._registry.test_document,
            self._registry.case_summary,
            self._registry.moderator,
            self._registry.final_synthesis,
            self._registry.final_reviewer,
            self._registry.radiologist,
            self._registry.spine_surgeon,
            self._registry.anesthesiologist,
            self._registry.neurologist,
            self._registry.perioperative_physician,
        )
        seen: set[int] = set()
        for agent in agents:
            # Deterministic transcription holds no provider client to close.
            client = getattr(agent, "client", None)
            if client is None:
                continue
            if id(client) in seen:
                continue
            seen.add(id(client))
            closer = getattr(client, "aclose", None)
            if closer is not None:
                try:
                    await closer()
                except Exception:
                    logger.exception("Model client shutdown failed")

    def _schedule(self, thread_id: str, graph_input, *, resumed: bool) -> None:
        if self._shutting_down:
            raise RuntimeError("RunManager is shutting down")
        task = asyncio.create_task(
            self._drive(thread_id, graph_input, resumed=resumed),
            name=f"spine-mdt-{thread_id}",
        )
        self._tasks[thread_id] = task

    async def _require_run(self, thread_id: str) -> RunRecord:
        record = await self.run_store.get_run(thread_id)
        if record is None:
            raise RunNotFoundError(thread_id)
        return record

    async def _checkpoint_values(self, thread_id: str) -> dict | None:
        checkpoint = await self.checkpointer.aget_tuple(thread_config(thread_id))
        if checkpoint is None:
            return None
        return checkpoint.checkpoint.get("channel_values", {})


def _status_event_data(record: RunRecord) -> dict:
    return {
        "thread_id": record.thread_id,
        "case_id": record.case_id,
        "status": record.status.value,
        "director_style": record.director_style.value if record.director_style else None,
        "error_code": record.error_code,
        "error_message": record.error_message,
        "updated_at": record.updated_at.isoformat(),
    }


def _messages_after(values: dict | None, sequence: int) -> list[ChatMessage]:
    messages = [
        ChatMessage.model_validate(item)
        for item in (values.get("chat_log", []) if values else [])
    ]
    return sorted(
        (message for message in messages if message.sequence > sequence),
        key=lambda message: message.sequence,
    )


def _model_failures(values: dict | None) -> list[ModelFailureRecord]:
    if not values or values.get("run_metadata") is None:
        return []
    metadata = RunMetadata.model_validate(values["run_metadata"])
    return list(metadata.model_failures)


def _control_response(record: RunRecord) -> RunControlResponse:
    return RunControlResponse(
        case_id=record.case_id,
        thread_id=record.thread_id,
        status=record.status,
    )
