"""OpenAI Responses API adapter for multimodal structured output."""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from config import ProviderConfig
from exceptions import (
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestError,
    ModelServiceError,
    ModelTimeoutError,
    StructuredOutputError,
)
from schemas import ProviderUsage, VisualInput
from services.image_loader import resolve_image_url
from services.llm_client import ModelResult, OutputT


class OpenAIResponsesClient:
    provider = "openai"

    def __init__(
        self,
        config: ProviderConfig,
        *,
        image_detail: Literal["original", "high"] = "original",
    ) -> None:
        self.model = config.model
        self.max_retries = config.max_retries
        self.timeout_seconds = config.timeout_seconds
        self.image_detail = image_detail
        kwargs: dict[str, Any] = {
            "api_key": config.api_key.get_secret_value(),
            "timeout": config.timeout_seconds,
            "max_retries": 0,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = AsyncOpenAI(**kwargs)

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
        started = time.perf_counter()
        try:
            response = await self._build_and_parse_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_model=output_model,
                images=images,
            )
            output = response.output_parsed
            if output is None:
                raise StructuredOutputError("OpenAI returned no parsed structured output.")
        except StructuredOutputError:
            raise
        except RateLimitError as exc:
            raise ModelRateLimitError("The OpenAI model request was rate limited.") from exc
        except APITimeoutError as exc:
            raise ModelTimeoutError("The OpenAI model request timed out.") from exc
        except APIConnectionError as exc:
            raise ModelConnectionError("The OpenAI model connection failed.") from exc
        except APIStatusError as exc:
            if exc.status_code >= 500:
                raise ModelServiceError("The OpenAI model service failed.") from exc
            raise ModelRequestError("The OpenAI model request was rejected.") from exc
        except OpenAIError as exc:
            raise ModelRequestError("The OpenAI model request was rejected.") from exc

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
        return ModelResult(
            output=output,
            usage=ProviderUsage(
                provider="openai",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
            ),
        )

    async def _build_and_parse_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[OutputT],
        images: Sequence[VisualInput],
    ):
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        for image in images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": await resolve_image_url(image.path_or_url),
                    "detail": self.image_detail,
                }
            )
        return await self._parse_response(
            system_prompt=system_prompt,
            content=content,
            output_model=output_model,
        )

    async def _parse_response(
        self,
        *,
        system_prompt: str,
        content: list[dict[str, Any]],
        output_model: type[OutputT],
    ):
        return await self._client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=[{"role": "user", "content": content}],
            text_format=output_model,
            timeout=self.timeout_seconds,
        )
