"""Checkpoint 7 REST and SSE routes."""

from __future__ import annotations

import json
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from api_schemas import (
    AdditionalInputRequest,
    CaseCreateMetadata,
    ErrorResponse,
    RunControlResponse,
    RunSnapshotResponse,
)
from exceptions import ServiceError
from services.run_manager import RunManager
from services.upload_store import UploadStore


router = APIRouter(prefix="/v1")
ERROR_RESPONSES = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.post(
    "/cases",
    response_model=RunSnapshotResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ERROR_RESPONSES,
)
async def create_case(
    request: Request,
    metadata: Annotated[str, Form(description="JSON-encoded CaseCreateMetadata")],
    case_files: Annotated[
        list[UploadFile],
        File(description="Ordered preoperative case-document screenshots"),
    ],
    test_files: Annotated[
        list[UploadFile],
        File(description="Ordered preoperative test-report screenshots"),
    ],
    image_files: Annotated[
        list[UploadFile],
        File(description="Preoperative images in the same order as metadata.images"),
    ],
) -> RunSnapshotResponse:
    try:
        case_metadata = CaseCreateMetadata.model_validate_json(metadata)
    except ValidationError as exc:
        raise ServiceError(
            "INVALID_CASE_METADATA",
            "The multipart metadata field is not a valid case definition.",
            status_code=422,
        ) from exc
    manager = _manager(request)
    upload_store = _upload_store(request)
    await manager.ensure_ready()
    thread_id = uuid4().hex
    uploads = [*case_files, *test_files, *image_files]
    try:
        case = await upload_store.save_case_package(
            thread_id=thread_id,
            case_id=case_metadata.case_id,
            image_metadata=case_metadata.images,
            case_files=case_files,
            test_files=test_files,
            image_files=image_files,
        )
    finally:
        for upload in uploads:
            await upload.close()
    await manager.start_case(
        thread_id=thread_id,
        case=case,
        director_style=case_metadata.director_style,
    )
    return await manager.get_snapshot(thread_id)


@router.get(
    "/runs/{thread_id}",
    response_model=RunSnapshotResponse,
    responses=ERROR_RESPONSES,
)
async def get_run(
    request: Request,
    thread_id: str,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
) -> RunSnapshotResponse:
    return await _manager(request).get_snapshot(
        thread_id,
        after_sequence=after_sequence,
    )


@router.get(
    "/runs/{thread_id}/result",
    response_model=RunSnapshotResponse,
    responses=ERROR_RESPONSES,
)
async def get_result(request: Request, thread_id: str) -> RunSnapshotResponse:
    return await _manager(request).get_result(thread_id)


@router.post(
    "/runs/{thread_id}/pause",
    response_model=RunControlResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ERROR_RESPONSES,
)
async def pause_run(request: Request, thread_id: str) -> RunControlResponse:
    return await _manager(request).request_pause(thread_id)


@router.post(
    "/runs/{thread_id}/resume",
    response_model=RunControlResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ERROR_RESPONSES,
)
async def resume_run(request: Request, thread_id: str) -> RunControlResponse:
    return await _manager(request).resume(thread_id)


@router.post(
    "/runs/{thread_id}/cancel",
    response_model=RunControlResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ERROR_RESPONSES,
)
async def cancel_run(request: Request, thread_id: str) -> RunControlResponse:
    return await _manager(request).request_cancel(thread_id)


@router.post(
    "/runs/{thread_id}/input",
    response_model=RunControlResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ERROR_RESPONSES,
)
async def add_input(
    request: Request,
    thread_id: str,
    body: AdditionalInputRequest,
) -> RunControlResponse:
    return await _manager(request).add_input(thread_id, body.content)


@router.get(
    "/runs/{thread_id}/events",
    response_class=StreamingResponse,
    responses={
        **ERROR_RESPONSES,
        200: {
            "description": "Persistent server-sent run events",
            "content": {"text/event-stream": {}},
        },
    },
)
async def stream_events(
    request: Request,
    thread_id: str,
    after_event_id: Annotated[int | None, Query(ge=0)] = None,
) -> StreamingResponse:
    manager = _manager(request)
    await manager.get_snapshot(thread_id)
    if after_event_id is None:
        raw_last_event = request.headers.get("last-event-id", "0")
        try:
            cursor = int(raw_last_event)
            if cursor < 0:
                raise ValueError
        except ValueError as exc:
            raise ServiceError(
                "INVALID_LAST_EVENT_ID",
                "Last-Event-ID must be a non-negative integer.",
                status_code=422,
                thread_id=thread_id,
            ) from exc
    else:
        cursor = after_event_id
    heartbeat = request.app.state.settings.sse_heartbeat_seconds

    async def event_source():
        nonlocal cursor
        while True:
            if await request.is_disconnected():
                return
            events = await manager.run_store.list_events(
                thread_id,
                after_event_id=cursor,
            )
            if events:
                for event in events:
                    cursor = event.event_id
                    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
                    yield (
                        f"id: {event.event_id}\n"
                        f"event: {event.event_type}\n"
                        f"data: {data}\n\n"
                    )
                continue
            record = await manager.run_store.get_run(thread_id)
            if record is None or record.status.is_terminal:
                return
            events = await manager.run_store.wait_for_events(
                thread_id,
                after_event_id=cursor,
                timeout=heartbeat,
            )
            if not events:
                yield ": heartbeat\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _manager(request: Request) -> RunManager:
    manager = getattr(request.app.state, "run_manager", None)
    if manager is None:
        raise ServiceError(
            "SERVICE_NOT_READY",
            "The run service is not ready.",
            status_code=503,
        )
    return manager


def _upload_store(request: Request) -> UploadStore:
    store = getattr(request.app.state, "upload_store", None)
    if store is None:
        raise ServiceError(
            "SERVICE_NOT_READY",
            "The upload service is not ready.",
            status_code=503,
        )
    return store
