"""Pydantic request and response contracts for the Checkpoint 7 API."""

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator

from schemas import (
    ChatMessage,
    DirectorStyle,
    FinalPlan,
    Modality,
    PreparedDocuments,
    RadiologyProgress,
    RunMetadata,
    RunPhase,
    StrictModel,
)


class RunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    PAUSED = "PAUSED"
    INTERRUPTED = "INTERRUPTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in {self.CANCELLED, self.COMPLETED, self.FAILED}


class ImageUploadMetadata(StrictModel):
    image_id: str = Field(min_length=1, max_length=200)
    modality: Modality
    body_region: str | None = Field(default=None, min_length=1, max_length=200)
    view_or_plane: str | None = Field(default=None, min_length=1, max_length=200)
    sequence: str | None = Field(default=None, max_length=200)
    level: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)


class CaseCreateMetadata(StrictModel):
    case_id: str = Field(min_length=1, max_length=200)
    confirmed_preoperative_only: Literal[True]
    images: list[ImageUploadMetadata] = Field(min_length=1)
    # Which experiment arm this run belongs to. Optional because no role
    # consumes it yet; a run without one is simply unattributed.
    director_style: DirectorStyle | None = None

    @field_validator("images")
    @classmethod
    def unique_image_ids(
        cls,
        images: list[ImageUploadMetadata],
    ) -> list[ImageUploadMetadata]:
        identifiers = [image.image_id for image in images]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("image_id values must be unique within a case")
        return images


class AdditionalInputRequest(StrictModel):
    content: str = Field(min_length=1, max_length=1_000_000)


class RunRecord(StrictModel):
    thread_id: str
    case_id: str
    status: RunStatus
    director_style: DirectorStyle | None = None
    last_message_sequence: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StoredEvent(StrictModel):
    event_id: int = Field(ge=1)
    thread_id: str
    event_type: str
    data: dict
    created_at: datetime


class RunSnapshotResponse(StrictModel):
    case_id: str
    thread_id: str
    status: RunStatus
    director_style: DirectorStyle | None = None
    phase: RunPhase | None
    new_messages: list[ChatMessage]
    is_complete: bool
    final_plan: FinalPlan | None
    run_metadata: RunMetadata | None
    prepared_documents: PreparedDocuments | None = None
    radiology_progress: RadiologyProgress | None = None
    error_code: str | None = None
    error_message: str | None = None


class RunControlResponse(StrictModel):
    case_id: str
    thread_id: str
    status: RunStatus


class ErrorResponse(StrictModel):
    code: str
    message: str
    thread_id: str | None = None
