from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import Request, Response
from openai import APITimeoutError, AuthenticationError, InternalServerError, RateLimitError
from pydantic import SecretStr

from agents import (
    AnesthesiologistAgent,
    ModeratorAgent,
    NeurologistAgent,
    PerioperativePhysicianAgent,
    RadiologistAgent,
    SpineSurgeonAgent,
)
from config import ProviderConfig
from exceptions import (
    ModelConnectionError,
    ModelInvocationError,
    ModelRateLimitError,
    ModelRequestError,
    ModelRetryExhaustedError,
    ModelServiceError,
    ModelTimeoutError,
)
from schemas import CaseImage, Modality, RadiologyReport, SpineSurgeonOutput
from services.llm_client import DeepSeekTextClient, FakeModelClient
from services.multimodal_client import OpenAIResponsesClient
from tests.factories import all_fake_responses, radiology_report


def image(image_id: str, url: str = "https://example.test/image.jpg") -> CaseImage:
    return CaseImage(
        image_id=image_id,
        modality=Modality.MRI,
        path_or_url=url,
        body_region="lumbar spine",
        view_or_plane="sagittal",
    )


def provider_config(*, base_url: str | None = None, max_retries: int = 0) -> ProviderConfig:
    return ProviderConfig(
        api_key=SecretStr("unit-test-key"),
        model="unit-test-model",
        base_url=base_url,
        timeout_seconds=5,
        max_retries=max_retries,
    )


def test_six_agents_are_independent_and_load_role_prompts() -> None:
    client = FakeModelClient(all_fake_responses())
    agents = [
        ModeratorAgent(client),
        RadiologistAgent(client),
        SpineSurgeonAgent(client),
        AnesthesiologistAgent(client),
        NeurologistAgent(client),
        PerioperativePhysicianAgent(client),
    ]
    assert len({type(agent) for agent in agents}) == 6
    assert len({agent.prompt_filename for agent in agents}) == 6
    for agent in agents:
        assert "## Core responsibilities" in agent.system_prompt
        assert "## Output contract" in agent.system_prompt
        assert "## Edge cases" in agent.system_prompt


@pytest.mark.asyncio
async def test_text_agents_never_receive_raw_images() -> None:
    client = FakeModelClient(all_fake_responses())
    agents = [
        ModeratorAgent(client),
        SpineSurgeonAgent(client),
        AnesthesiologistAgent(client),
        NeurologistAgent(client),
        PerioperativePhysicianAgent(client),
    ]
    for agent in agents:
        result = await agent.run(
            case_preop_md="Right L5 radiculopathy.",
            tests_preop_md="Hemoglobin 128 g/L.",
            radiology_preop_md="# Radiology\nL4-L5 disc extrusion.",
            radiology_report=radiology_report(),
            chat_log=[],
            question="Provide your assessment.",
        )
        assert isinstance(result.output, agent.output_model)
    assert len(client.calls) == 5
    assert all(call["images"] == [] for call in client.calls)
    assert all("example.test" not in call["user_prompt"] for call in client.calls)


@pytest.mark.asyncio
async def test_radiologist_receives_multiple_images_without_paths_in_prompt() -> None:
    client = FakeModelClient(all_fake_responses())
    agent = RadiologistAgent(client)
    images = [image("mri-1"), image("mri-2", "https://example.test/image2.jpg")]
    result = await agent.run(
        case_summary_md="Right leg pain.",
        images=images,
        chat_log=[],
        question="Provide the initial report.",
    )
    assert isinstance(result.output, RadiologyReport)
    assert client.calls[0]["images"] == images
    assert "https://example.test" not in client.calls[0]["user_prompt"]
    assert "mri-1" in client.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_radiologist_batches_twenty_images_and_synthesizes_without_images() -> None:
    client = FakeModelClient(all_fake_responses())
    agent = RadiologistAgent(
        client,
        batch_max_images=6,
        batch_max_bytes=100 * 1024 * 1024,
        batch_concurrency=2,
    )
    images = [
        image(f"mri-{index:02d}", f"https://example.test/image-{index:02d}.jpg")
        for index in range(1, 21)
    ]

    result = await agent.run(
        case_summary_md="Cervical myelopathy.",
        images=images,
        chat_log=[],
        question="Provide the initial report.",
    )

    assert isinstance(result.output, RadiologyReport)
    assert [len(call["images"]) for call in client.calls] == [6, 6, 6, 2, 0]
    assert [
        reviewed.image_id
        for call in client.calls[:-1]
        for reviewed in call["images"]
    ] == [item.image_id for item in images]
    assert "cross_batch_synthesis" in client.calls[-1]["user_prompt"]
    assert all(
        item.path_or_url not in call["user_prompt"]
        for call in client.calls
        for item in images
    )
    assert len(result.all_usages) == 5


@pytest.mark.asyncio
async def test_radiologist_reports_ordered_four_image_batch_progress() -> None:
    client = FakeModelClient(all_fake_responses())
    agent = RadiologistAgent(client)
    images = [
        image(f"mri-{index}", f"https://example.test/image-{index}.jpg")
        for index in range(1, 6)
    ]
    progress_updates = []

    async def record_progress(progress) -> None:
        progress_updates.append(progress)

    await agent.run(
        case_summary_md="Cervical myelopathy.",
        images=images,
        chat_log=[],
        question="Provide the initial report.",
        progress_callback=record_progress,
    )

    assert [len(call["images"]) for call in client.calls] == [4, 1, 0]
    assert [item.stage for item in progress_updates] == [
        "reviewing",
        "batch_completed",
        "reviewing",
        "batch_completed",
        "synthesizing",
        "completed",
    ]
    assert [
        (item.current_image_start, item.current_image_end)
        for item in progress_updates
        if item.stage == "reviewing"
    ] == [(1, 4), (5, 5)]
    assert [
        item.completed_images
        for item in progress_updates
        if item.stage == "batch_completed"
    ] == [4, 5]


@pytest.mark.asyncio
async def test_radiologist_batching_also_limits_local_payload_bytes(tmp_path) -> None:
    client = FakeModelClient(all_fake_responses())
    agent = RadiologistAgent(
        client,
        batch_max_images=10,
        batch_max_bytes=10,
        batch_concurrency=1,
    )
    images = []
    for index in range(1, 4):
        path = tmp_path / f"image-{index}.jpg"
        path.write_bytes(b"12345678")
        images.append(image(f"mri-{index}", str(path)))

    await agent.run(
        case_summary_md="Cervical myelopathy.",
        images=images,
        chat_log=[],
        question="Provide the initial report.",
    )

    assert [len(call["images"]) for call in client.calls] == [1, 1, 1, 0]
    assert all(str(tmp_path) not in call["user_prompt"] for call in client.calls)


@pytest.mark.asyncio
async def test_invalid_agent_output_is_retried_and_classified() -> None:
    client = FakeModelClient(
        {SpineSurgeonOutput: {"public_message": "incomplete"}},
        max_retries=1,
    )
    agent = SpineSurgeonAgent(client)
    with pytest.raises(ModelRetryExhaustedError) as raised:
        await agent.run(
            case_preop_md="Case text",
            tests_preop_md="Tests",
            radiology_preop_md="Radiology",
            radiology_report=None,
            chat_log=[],
            question="Assess surgery.",
        )
    assert len(client.calls) == 2
    assert raised.value.attempt_count == 2
    assert [item.value for item in raised.value.attempt_categories] == [
        "MODEL_INVALID_OUTPUT",
        "MODEL_INVALID_OUTPUT",
    ]


class RecoveringSpineClient(FakeModelClient):
    def __init__(self, failures) -> None:
        super().__init__(all_fake_responses(), max_retries=3)
        self.failures = list(failures)

    async def generate_structured(self, **kwargs):
        if kwargs["output_model"] is SpineSurgeonOutput and self.failures:
            self.calls.append(kwargs)
            raise self.failures.pop(0)
        return await super().generate_structured(**kwargs)


@pytest.mark.asyncio
async def test_text_agent_recovers_after_three_retryable_failures() -> None:
    client = RecoveringSpineClient(
        [
            ModelTimeoutError("private timeout detail"),
            ModelRateLimitError("private rate-limit detail"),
            ModelConnectionError("private connection detail"),
        ]
    )
    result = await SpineSurgeonAgent(client).run(
        case_preop_md="Case text",
        tests_preop_md="Tests",
        radiology_preop_md="Radiology",
        radiology_report=None,
        chat_log=[],
        question="Assess surgery.",
    )

    assert len(client.calls) == 4
    assert result.attempt_count == 4
    assert result.max_attempts == 4
    assert [item.value for item in result.attempt_failure_categories] == [
        "MODEL_TIMEOUT",
        "MODEL_RATE_LIMITED",
        "MODEL_CONNECTION_FAILED",
    ]


@pytest.mark.asyncio
async def test_text_agent_retries_service_failure() -> None:
    client = RecoveringSpineClient(
        [ModelServiceError("private provider 5xx detail")]
    )
    result = await SpineSurgeonAgent(client).run(
        case_preop_md="Case text",
        tests_preop_md="Tests",
        radiology_preop_md="Radiology",
        radiology_report=None,
        chat_log=[],
        question="Assess surgery.",
    )

    assert len(client.calls) == 2
    assert result.attempt_count == 2
    assert [item.value for item in result.attempt_failure_categories] == [
        "MODEL_SERVICE_FAILED"
    ]


@pytest.mark.asyncio
async def test_rejected_text_request_is_not_retried() -> None:
    client = RecoveringSpineClient([ModelRequestError("private request detail")])
    with pytest.raises(ModelRetryExhaustedError) as raised:
        await SpineSurgeonAgent(client).run(
            case_preop_md="Case text",
            tests_preop_md="Tests",
            radiology_preop_md="Radiology",
            radiology_report=None,
            chat_log=[],
            question="Assess surgery.",
        )

    assert len(client.calls) == 1
    assert raised.value.attempt_count == 1
    assert raised.value.attempt_categories[-1].value == "MODEL_REQUEST_REJECTED"


@pytest.mark.asyncio
async def test_deepseek_adapter_validates_json_and_rejects_images() -> None:
    client = DeepSeekTextClient(provider_config(base_url="https://deepseek.example/v1"))
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=radiology_report().model_dump_json()))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )
    create = AsyncMock(return_value=completion)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = await client.generate_structured(
        system_prompt="System",
        user_prompt="User",
        output_model=RadiologyReport,
    )
    assert result.output == radiology_report()
    assert result.usage.total_tokens == 30
    with pytest.raises(ModelInvocationError, match="does not accept images"):
        await client.generate_structured(
            system_prompt="System",
            user_prompt="User",
            output_model=RadiologyReport,
            images=[image("mri-1")],
        )


@pytest.mark.asyncio
async def test_deepseek_adapter_sends_max_reasoning_effort_when_configured() -> None:
    client = DeepSeekTextClient(
        provider_config(base_url="https://deepseek.example/v1"),
        thinking="max",
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=radiology_report().model_dump_json()))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    create = AsyncMock(return_value=completion)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    await client.generate_structured(
        system_prompt="System",
        user_prompt="User",
        output_model=RadiologyReport,
    )

    assert create.await_args.kwargs["reasoning_effort"] == "max"

    off_client = DeepSeekTextClient(
        provider_config(base_url="https://deepseek.example/v1"),
        thinking="off",
    )
    off_create = AsyncMock(return_value=completion)
    off_client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=off_create))
    )
    await off_client.generate_structured(
        system_prompt="System",
        user_prompt="User",
        output_model=RadiologyReport,
    )
    assert "reasoning_effort" not in off_create.await_args.kwargs


@pytest.mark.asyncio
async def test_deepseek_adapter_accepts_images_when_enabled() -> None:
    client = DeepSeekTextClient(
        provider_config(base_url="https://deepseek.example/v1"),
        supports_images=True,
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=radiology_report().model_dump_json()))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    create = AsyncMock(return_value=completion)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    await client.generate_structured(
        system_prompt="System",
        user_prompt="User",
        output_model=RadiologyReport,
        images=[image("mri-1"), image("mri-2")],
    )

    messages = create.await_args.kwargs["messages"]
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert [item["type"] for item in user_content] == ["text", "image_url", "image_url"]
    assert user_content[1]["image_url"]["url"] == "https://example.test/image.jpg"
    assert user_content[1]["image_url"]["detail"] == "original"


@pytest.mark.asyncio
async def test_openai_responses_adapter_builds_multi_image_request() -> None:
    client = OpenAIResponsesClient(provider_config())
    parse = AsyncMock(
        return_value=SimpleNamespace(
            output_parsed=radiology_report(),
            usage=SimpleNamespace(input_tokens=15, output_tokens=25, total_tokens=40),
        )
    )
    client._client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    result = await client.generate_structured(
        system_prompt="System",
        user_prompt="User",
        output_model=RadiologyReport,
        images=[image("mri-1"), image("mri-2")],
    )
    assert result.output == radiology_report()
    request = parse.await_args.kwargs
    content = request["input"][0]["content"]
    assert [item["type"] for item in content] == ["input_text", "input_image", "input_image"]
    assert [item["detail"] for item in content[1:]] == ["original", "original"]
    assert request["text_format"] is RadiologyReport


@pytest.mark.asyncio
async def test_openai_timeout_is_preserved_as_a_controlled_timeout_error() -> None:
    client = OpenAIResponsesClient(provider_config())
    parse = AsyncMock(
        side_effect=APITimeoutError(request=Request("POST", "https://api.openai.test/v1"))
    )
    client._client = SimpleNamespace(responses=SimpleNamespace(parse=parse))

    with pytest.raises(ModelTimeoutError, match="timed out"):
        await client.generate_structured(
            system_prompt="System",
            user_prompt="User",
            output_model=RadiologyReport,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_error", "controlled_error"),
    [
        (RateLimitError, ModelRateLimitError),
        (InternalServerError, ModelServiceError),
        (AuthenticationError, ModelRequestError),
    ],
)
async def test_openai_status_errors_are_safely_classified(
    provider_error,
    controlled_error,
) -> None:
    request = Request("POST", "https://api.openai.test/v1")
    status = 429 if provider_error is RateLimitError else 500
    if provider_error is AuthenticationError:
        status = 401
    error = provider_error(
        "provider detail that must not escape",
        response=Response(status, request=request),
        body=None,
    )
    client = OpenAIResponsesClient(provider_config())
    client._client = SimpleNamespace(
        responses=SimpleNamespace(parse=AsyncMock(side_effect=error))
    )

    with pytest.raises(controlled_error):
        await client.generate_structured(
            system_prompt="System",
            user_prompt="User",
            output_model=RadiologyReport,
        )
