"""Common model invocation infrastructure for all MDT agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Generic, Sequence, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
)

from exceptions import (
    ModelInvocationError,
    ModelRetryExhaustedError,
    RetryableModelError,
    StructuredOutputError,
    model_failure_category,
)
from schemas import (
    AgentRole,
    ChatMessage,
    OperativePlanDraft,
    RadiologyReport,
    VisualInput,
)
from services.llm_client import ModelResult, StructuredModelClient
from services.retry_policy import model_retry_wait


OutputT = TypeVar("OutputT", bound=BaseModel)
PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts"
_SAFE_VALIDATION_FEEDBACK = {
    "Final plan referenced an unknown evidence id.",
    "Final plan omitted source support for a critical operative field.",
    (
        "Final plan asserted a completed assessment or optimization without "
        "documented source support."
    ),
    "Final review referenced an unknown evidence id.",
    "A surgical option or approach assessment referenced an unknown evidence id.",
    "Radiology operative planning requires at least one supplied image id.",
}


def _validation_retry_feedback(error: ValidationError, schema: type) -> str:
    """Name the fields that failed and the rule they broke.

    Telling a model only that its answer "did not satisfy the required
    structured schema" leaves it to guess, so it re-sends the same shape until
    the retry budget is gone -- and a cross-field rule the prompt never stated
    is unguessable. A pydantic error carries the field path and the rule's own
    wording, both of which are ours and neither of which contains case material.

    The offending value is deliberately not included: that one *can* carry case
    material, and the model already knows what it wrote.
    """

    details = []
    for item in error.errors()[:8]:
        location = ".".join(str(part) for part in item.get("loc", ()))
        details.append(f"- {location or schema.__name__}: {item.get('msg', 'invalid')}")
    if not details:
        return "The previous response did not satisfy the required structured schema."
    return "These fields were rejected. Fix each one:\n" + "\n".join(details)


def _structured_retry_feedback(error: StructuredOutputError) -> str:
    message = str(error)
    category = getattr(error, "category", None)
    if category is not None:
        return f"Deterministic validation category: {category.value}."
    if message in _SAFE_VALIDATION_FEEDBACK:
        return message
    return "The previous response did not satisfy the required structured schema."


class BaseAgent(ABC, Generic[OutputT]):
    """Infrastructure-only base class; professional reasoning stays in subclasses."""

    role: AgentRole
    display_name: str
    prompt_filename: str
    output_model: type[OutputT]

    def __init__(self, client: StructuredModelClient) -> None:
        self.client = client
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        return self._load_prompt_file(self.prompt_filename)

    def _load_prompt_file(self, filename: str) -> str:
        path = PROMPT_ROOT / filename
        try:
            prompt = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ModelInvocationError(
                f"Prompt file for {self.role.value} could not be loaded."
            ) from exc
        if not prompt:
            raise ModelInvocationError(f"Prompt file for {self.role.value} is empty.")
        return prompt

    @abstractmethod
    def build_user_prompt(
        self,
        *,
        case_preop_md: str,
        tests_preop_md: str,
        radiology_preop_md: str,
        radiology_report: RadiologyReport | None,
        chat_log: Sequence[ChatMessage],
        question: str,
        prior_contribution: BaseModel | None = None,
        working_plan: OperativePlanDraft | None = None,
    ) -> str:
        """Build the role-specific, text-only request context."""

    async def run(
        self,
        *,
        case_preop_md: str,
        tests_preop_md: str,
        radiology_preop_md: str,
        radiology_report: RadiologyReport | None,
        chat_log: Sequence[ChatMessage],
        question: str,
        prior_contribution: BaseModel | None = None,
        working_plan: OperativePlanDraft | None = None,
        output_processor: Callable[[BaseModel], BaseModel] | None = None,
    ) -> ModelResult[OutputT]:
        user_prompt = self.build_user_prompt(
            case_preop_md=case_preop_md,
            tests_preop_md=tests_preop_md,
            radiology_preop_md=radiology_preop_md,
            radiology_report=radiology_report,
            chat_log=chat_log,
            question=question,
            prior_contribution=prior_contribution,
            working_plan=working_plan,
        )
        return await self._invoke(
            user_prompt=user_prompt,
            images=(),
            output_processor=output_processor,
        )

    async def _invoke(
        self,
        *,
        user_prompt: str,
        images: Sequence[VisualInput],
        output_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
        output_processor: Callable[[BaseModel], BaseModel] | None = None,
    ) -> ModelResult:
        schema = output_model or self.output_model
        max_attempts = self.client.max_retries + 1
        attempt_count = 0
        attempt_failure_categories = []
        attempt_usages = []
        retry_feedback: str | None = None

        def remember_failed_attempt(retry_state) -> None:
            error = retry_state.outcome.exception()
            if isinstance(error, ModelInvocationError):
                attempt_failure_categories.append(model_failure_category(error))

        retryer = AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=model_retry_wait(self.client.provider),
            retry=retry_if_exception_type(
                (RetryableModelError, StructuredOutputError)
            ),
            before_sleep=remember_failed_attempt,
            reraise=True,
        )
        try:
            async for attempt in retryer:
                attempt_count = attempt.retry_state.attempt_number
                with attempt:
                    try:
                        effective_user_prompt = user_prompt
                        if retry_feedback:
                            effective_user_prompt = (
                                f"{user_prompt}\n\nRETRY CORRECTION:\n{retry_feedback}\n"
                                "Regenerate the entire structured response; do not merely "
                                "describe the correction."
                            )
                        result = await self.client.generate_structured(
                            system_prompt=system_prompt or self.system_prompt,
                            user_prompt=effective_user_prompt,
                            output_model=schema,
                            images=images,
                        )
                        attempt_usages.extend(result.all_usages)
                        output = schema.model_validate(result.output)
                        if output_processor is not None:
                            output = schema.model_validate(
                                output_processor(output)
                            )
                    except ValidationError as exc:
                        retry_feedback = _validation_retry_feedback(exc, schema)
                        raise StructuredOutputError(
                            f"{self.role.value} returned invalid structured output."
                        ) from exc
                    except StructuredOutputError as exc:
                        retry_feedback = _structured_retry_feedback(exc)
                        raise
                    return ModelResult(
                        output=output,
                        usage=result.usage,
                        usages=tuple(attempt_usages),
                        attempt_failure_categories=tuple(
                            attempt_failure_categories
                        ),
                        attempt_count=attempt_count,
                        max_attempts=max_attempts,
                    )
        except ModelInvocationError as exc:
            attempt_failure_categories.append(model_failure_category(exc))
            raise ModelRetryExhaustedError(
                exc,
                attempt_categories=tuple(attempt_failure_categories),
                max_attempts=max_attempts,
            ) from exc
        raise StructuredOutputError(f"{self.role.value} produced no structured result.")


def chat_context(chat_log: Sequence[ChatMessage]) -> list[dict]:
    """Return public fields only for text-agent context construction."""

    return [
        {
            "sequence": message.sequence,
            "speaker_role": message.speaker_role.value,
            "content": message.content,
            "evidence_ids": message.evidence_ids,
        }
        for message in chat_log
    ]


def report_context(report: RadiologyReport | None) -> dict | None:
    return report.model_dump(mode="json") if report is not None else None


def structured_context(value: BaseModel | None) -> dict | None:
    return value.model_dump(mode="json") if value is not None else None


def document_context(
    *,
    case_preop_md: str,
    tests_preop_md: str,
    radiology_preop_md: str,
) -> dict[str, str]:
    return {
        "case_preop_md": case_preop_md,
        "tests_preop_md": tests_preop_md,
        "radiology_preop_md": radiology_preop_md,
    }
