"""Application-specific exceptions with safe, user-facing messages."""

from enum import Enum


class SpineMDTError(Exception):
    """Base class for controlled application failures."""


class ConfigurationError(SpineMDTError):
    """Raised when a requested provider is missing required configuration."""


class ModelInvocationError(SpineMDTError):
    """Raised when a model request cannot be completed."""


class RetryableModelError(ModelInvocationError):
    """Base class for transient provider failures that may succeed on retry."""


class ModelTimeoutError(RetryableModelError):
    """Raised when a model request remains timed out after agent retries."""


class ModelRateLimitError(RetryableModelError):
    """Raised when a provider rate-limits a model request."""


class ModelConnectionError(RetryableModelError):
    """Raised when a model provider cannot be reached reliably."""


class ModelServiceError(RetryableModelError):
    """Raised for retryable provider-side 5xx failures."""


class ModelRequestError(ModelInvocationError):
    """Raised when a provider rejects a request that should not be retried."""


class StructuredOutputError(ModelInvocationError):
    """Raised when model output cannot be validated safely."""


class OcrError(ModelInvocationError):
    """Raised when deterministic OCR cannot transcribe a page.

    It inherits from ``ModelInvocationError`` so that an OCR failure travels the
    same stage-failure path as a provider failure, even though no model is
    involved: ``DocumentPageError`` requires that type, and the run must fail
    closed either way.
    """


class ModelFailureCategory(str, Enum):
    TIMEOUT = "MODEL_TIMEOUT"
    RATE_LIMITED = "MODEL_RATE_LIMITED"
    CONNECTION_FAILED = "MODEL_CONNECTION_FAILED"
    SERVICE_FAILED = "MODEL_SERVICE_FAILED"
    REQUEST_REJECTED = "MODEL_REQUEST_REJECTED"
    INVALID_OUTPUT = "MODEL_INVALID_OUTPUT"
    FINAL_PLAN_STATE_MISMATCH = "FINAL_PLAN_STATE_MISMATCH"
    FINAL_PLAN_UNKNOWN_EVIDENCE = "FINAL_PLAN_UNKNOWN_EVIDENCE"
    FINAL_PLAN_OMITTED_ISSUES = "FINAL_PLAN_OMITTED_ISSUES"
    FINAL_PLAN_OMITTED_DISAGREEMENT = "FINAL_PLAN_OMITTED_DISAGREEMENT"
    FINAL_PLAN_DISAGREEMENT_STATUS = "FINAL_PLAN_DISAGREEMENT_STATUS"
    FINAL_PLAN_FIELD_STATUS_MISMATCH = "FINAL_PLAN_FIELD_STATUS_MISMATCH"
    INVOCATION_FAILED = "MODEL_INVOCATION_FAILED"


class FinalPlanIntegrityError(StructuredOutputError):
    """Safe final-plan integrity failure with a stable diagnostic category."""

    def __init__(self, category: ModelFailureCategory) -> None:
        if category not in {
            ModelFailureCategory.FINAL_PLAN_STATE_MISMATCH,
            ModelFailureCategory.FINAL_PLAN_UNKNOWN_EVIDENCE,
            ModelFailureCategory.FINAL_PLAN_OMITTED_ISSUES,
            ModelFailureCategory.FINAL_PLAN_OMITTED_DISAGREEMENT,
            ModelFailureCategory.FINAL_PLAN_DISAGREEMENT_STATUS,
            ModelFailureCategory.FINAL_PLAN_FIELD_STATUS_MISMATCH,
        }:
            raise ValueError("category is not a final-plan integrity category")
        self.category = category
        super().__init__(f"Final plan integrity check failed with {category.value}.")


class ModelRetryExhaustedError(ModelInvocationError):
    """Safe retry summary that preserves categories without provider details."""

    def __init__(
        self,
        cause: ModelInvocationError,
        *,
        attempt_categories: tuple[ModelFailureCategory, ...],
        max_attempts: int,
    ) -> None:
        if not attempt_categories:
            raise ValueError("attempt_categories must not be empty")
        self.cause = cause
        self.attempt_categories = attempt_categories
        self.attempt_count = len(attempt_categories)
        self.max_attempts = max_attempts
        super().__init__(
            "Model request failed after "
            f"{self.attempt_count} attempt(s) with {attempt_categories[-1].value}."
        )


class ModelStageError(ModelInvocationError):
    """Safe model failure tagged with the workflow stage that invoked it."""

    def __init__(
        self,
        stage: str,
        cause: ModelInvocationError,
        *,
        item_id: str | None = None,
    ) -> None:
        self.stage = stage
        self.cause = cause
        self.category = model_failure_category(cause)
        self.item_id = item_id
        item_suffix = f" for {item_id}" if item_id else ""
        super().__init__(
            f"Model stage {stage}{item_suffix} failed with {self.category.value}."
        )


class DocumentPageError(ModelInvocationError):
    """Document transcription failure carrying only a stable synthetic page ID."""

    def __init__(self, page_id: str, cause: ModelInvocationError) -> None:
        self.page_id = page_id
        self.cause = cause
        super().__init__(f"Document transcription failed for {page_id}.")


def model_failure_category(error: ModelInvocationError) -> ModelFailureCategory:
    if isinstance(error, ModelRetryExhaustedError):
        return error.attempt_categories[-1]
    if isinstance(error, ModelTimeoutError):
        return ModelFailureCategory.TIMEOUT
    if isinstance(error, ModelRateLimitError):
        return ModelFailureCategory.RATE_LIMITED
    if isinstance(error, ModelConnectionError):
        return ModelFailureCategory.CONNECTION_FAILED
    if isinstance(error, ModelServiceError):
        return ModelFailureCategory.SERVICE_FAILED
    if isinstance(error, ModelRequestError):
        return ModelFailureCategory.REQUEST_REJECTED
    if isinstance(error, FinalPlanIntegrityError):
        return error.category
    if isinstance(error, StructuredOutputError):
        return ModelFailureCategory.INVALID_OUTPUT
    return ModelFailureCategory.INVOCATION_FAILED


class InvalidStateUpdateError(SpineMDTError):
    """Raised when a state reducer receives a conflicting update."""


class RetrievalDisabledError(SpineMDTError):
    """Raised if the intentionally disabled retrieval interface is called."""


class ServiceError(SpineMDTError):
    """Stable API error carrying a safe public code and HTTP status."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int,
        thread_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status_code = status_code
        self.thread_id = thread_id


class RunNotFoundError(ServiceError):
    def __init__(self, thread_id: str) -> None:
        super().__init__(
            "RUN_NOT_FOUND",
            "The requested run does not exist.",
            status_code=404,
            thread_id=thread_id,
        )


class InvalidRunTransitionError(ServiceError):
    def __init__(self, thread_id: str, message: str) -> None:
        super().__init__(
            "INVALID_RUN_TRANSITION",
            message,
            status_code=409,
            thread_id=thread_id,
        )


class RunNotCompleteError(ServiceError):
    def __init__(self, thread_id: str) -> None:
        super().__init__(
            "RUN_NOT_COMPLETE",
            "The run has not completed.",
            status_code=409,
            thread_id=thread_id,
        )


class UploadValidationError(ServiceError):
    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(code, message, status_code=status_code)
