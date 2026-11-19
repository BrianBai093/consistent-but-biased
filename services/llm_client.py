"""Unified model-client contract and DeepSeek text adapter."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Generic, Protocol, Sequence, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from config import ProviderConfig
from exceptions import (
    ModelConnectionError,
    ModelFailureCategory,
    ModelInvocationError,
    ModelRateLimitError,
    ModelRequestError,
    ModelServiceError,
    ModelTimeoutError,
    StructuredOutputError,
)
from schemas import ProviderUsage, VisualInput
from services.image_loader import resolve_image_url


OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ModelResult(Generic[OutputT]):
    output: OutputT | dict[str, Any]
    usage: ProviderUsage
    usages: tuple[ProviderUsage, ...] = ()
    #: Readings this result was consolidated from, when it was built out of
    #: several calls. The radiologist reads the films in batches and merges
    #: them; without the batches, a merged report that drops or softens a
    #: finding is indistinguishable from one that never saw it.
    intermediates: tuple[Any, ...] = ()
    attempt_failure_categories: tuple[ModelFailureCategory, ...] = ()
    attempt_count: int = 1
    max_attempts: int = 1

    @property
    def all_usages(self) -> tuple[ProviderUsage, ...]:
        """Return every provider call represented by this logical agent result."""

        return self.usages or (self.usage,)


def aggregate_usage(usages: Sequence[ProviderUsage]) -> ProviderUsage:
    """Combine provider accounting while retaining individual calls separately."""

    if not usages:
        raise ValueError("At least one provider usage record is required")
    first = usages[0]
    known_costs = [item.estimated_cost_usd for item in usages]
    estimated_cost = (
        sum(cost for cost in known_costs if cost is not None)
        if all(cost is not None for cost in known_costs)
        else None
    )
    return ProviderUsage(
        provider=first.provider,
        model=first.model,
        input_tokens=sum(item.input_tokens for item in usages),
        output_tokens=sum(item.output_tokens for item in usages),
        total_tokens=sum(item.total_tokens for item in usages),
        latency_ms=sum(item.latency_ms for item in usages),
        estimated_cost_usd=estimated_cost,
    )


class StructuredModelClient(Protocol):
    provider: str
    model: str
    max_retries: int

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[OutputT],
        images: Sequence[VisualInput] = (),
    ) -> ModelResult[OutputT]: ...


class DeepSeekTextClient:
    """OpenAI-compatible Chat Completions adapter for DeepSeek text models."""

    provider = "deepseek"

    def __init__(
        self,
        config: ProviderConfig,
        *,
        thinking: str = "off",
        supports_images: bool = False,
        image_detail: str = "original",
    ) -> None:
        if not config.base_url:
            raise ModelInvocationError("DeepSeek base URL is required.")
        self.model = config.model
        self.max_retries = config.max_retries
        self.timeout_seconds = config.timeout_seconds
        self.thinking = thinking
        self.supports_images = supports_images
        self.image_detail = image_detail
        self._client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[OutputT],
        images: Sequence[VisualInput] = (),
    ) -> ModelResult[OutputT]:
        if images and not self.supports_images:
            raise ModelRequestError("The DeepSeek text adapter does not accept images.")
        # Serialised as JSON, not interpolated as a Python object: a bare repr
        # writes single quotes and ``False``/``None``, which is not JSON and not
        # what the sentence above it claims to be showing.
        schema_instruction = (
            "Return one JSON object only. It must satisfy this JSON Schema:\n"
            + json.dumps(output_model.model_json_schema(), ensure_ascii=False)
        )
        user_content = f"{user_prompt}\n\n{schema_instruction}"
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_content}]
            for image in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": await resolve_image_url(image.path_or_url),
                            "detail": self.image_detail,
                        },
                    }
                )
        else:
            content = user_content
        started = time.perf_counter()
        try:
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "response_format": {"type": "json_object"},
                "timeout": self.timeout_seconds,
            }
            if self.thinking != "off":
                create_kwargs["reasoning_effort"] = self.thinking
            response = await self._client.chat.completions.create(**create_kwargs)
            message_content = response.choices[0].message.content
            if not message_content:
                raise StructuredOutputError("DeepSeek returned an empty structured response.")
            output = output_model.model_validate_json(message_content)
        except StructuredOutputError:
            raise
        except RateLimitError as exc:
            raise ModelRateLimitError("The DeepSeek model request was rate limited.") from exc
        except APITimeoutError as exc:
            raise ModelTimeoutError("The DeepSeek model request timed out.") from exc
        except APIConnectionError as exc:
            raise ModelConnectionError("The DeepSeek model connection failed.") from exc
        except ValidationError as exc:
            raise StructuredOutputError("DeepSeek returned invalid structured output.") from exc
        except APIStatusError as exc:
            if exc.status_code >= 500:
                raise ModelServiceError("The DeepSeek model service failed.") from exc
            raise ModelRequestError("The DeepSeek model request was rejected.") from exc
        except OpenAIError as exc:
            raise ModelRequestError("The DeepSeek model request was rejected.") from exc
        except (IndexError, AttributeError) as exc:
            raise StructuredOutputError("DeepSeek returned invalid structured output.") from exc

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
        return ModelResult(
            output=output,
            usage=ProviderUsage(
                provider="deepseek",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
            ),
        )


class FakeModelClient:
    """Deterministic, offline client used by unit and graph tests."""

    provider = "fake"

    def __init__(
        self,
        responses: dict[
            type[BaseModel],
            BaseModel
            | dict[str, Any]
            | ModelInvocationError
            | list[BaseModel | dict[str, Any] | ModelInvocationError],
        ],
        *,
        model: str = "fake-model",
        max_retries: int = 0,
    ) -> None:
        self.responses = responses
        self.model = model
        self.max_retries = max_retries
        self.calls: list[dict[str, Any]] = []
        self._response_indices: dict[type[BaseModel], int] = {}

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[OutputT],
        images: Sequence[VisualInput] = (),
    ) -> ModelResult[OutputT]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "output_model": output_model,
                "images": list(images),
            }
        )
        if output_model not in self.responses:
            raise ModelInvocationError(f"No fake response configured for {output_model.__name__}.")
        configured = self.responses[output_model]
        if isinstance(configured, list):
            index = self._response_indices.get(output_model, 0)
            if index >= len(configured):
                raise ModelInvocationError(
                    f"No scripted fake response remains for {output_model.__name__}."
                )
            response = configured[index]
            self._response_indices[output_model] = index + 1
        else:
            response = configured
        if isinstance(response, ModelInvocationError):
            raise response
        return ModelResult(
            output=response,
            usage=ProviderUsage(provider="fake", model=self.model),
        )
