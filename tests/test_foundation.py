import importlib
import logging

import pytest
from httpx import ASGITransport, AsyncClient

from app import app
from config import Settings
from exceptions import ConfigurationError
from schemas import RunMetadata, WorkflowLimits
from services.agent_factory import create_live_agent_registry
from services.run_logger import SecretRedactionFilter
from state import CURRENT_WORKFLOW_VERSION


@pytest.mark.parametrize(
    "module_name",
    ["app", "graph", "state", "schemas", "config", "exceptions", "agents", "services", "tools"],
)
def test_required_modules_import(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "spine-mdt"}


def test_missing_provider_configuration_is_controlled() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key=None,
    )
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        settings.require_openai()


def test_a_blank_credential_is_a_missing_credential() -> None:
    """``OPENAI_API_KEY=`` in a .env file reads as "", not as absent.

    Letting it through produces a rejected request from the provider after the
    case has already been ingested and transcribed, which reads like a provider
    outage rather than like an unset variable.
    """

    settings = Settings(_env_file=None, openai_api_key="", deepseek_api_key="   ")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        settings.require_openai()
    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        settings.require_deepseek()


def test_document_transcription_stability_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.model_max_retries == 3
    assert settings.document_timeout_seconds == 180
    assert settings.radiology_batch_max_images == 4
    assert settings.radiology_batch_max_bytes == 12 * 1024 * 1024
    assert settings.radiology_batch_concurrency == 2
    assert settings.document_previous_page_tail_chars == 1200
    assert settings.openai_image_detail == "original"
    assert settings.openai_critical_model == "gpt-5.6-sol"
    assert settings.critical_max_retries == 1
    assert settings.deepseek_model == "deepseek-v4-pro"
    assert settings.deepseek_thinking == "max"
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_timeout_seconds == 300
    assert settings.deepseek_max_retries == 3
    assert settings.text_provider == "deepseek"
    assert settings.image_provider == "openai"
    assert not hasattr(settings, "document_max_retries")
    assert not hasattr(settings, "document_batch_max_pages")
    assert not hasattr(settings, "document_batch_max_bytes")
    assert not hasattr(settings, "document_request_concurrency")


def test_v5_workflow_and_extended_discussion_defaults() -> None:
    limits = WorkflowLimits()

    assert CURRENT_WORKFLOW_VERSION == "v5"
    assert RunMetadata.model_fields["context_policy_version"].default == "v5"
    assert limits.max_turns_per_specialist == 5
    assert limits.max_total_specialist_turns == 20
    assert limits.max_absolute_specialist_turns == 24
    assert limits.max_workflow_steps == 50


def test_absolute_speaker_limit_cannot_be_below_normal_limit() -> None:
    with pytest.raises(ValueError, match="max_absolute_specialist_turns"):
        WorkflowLimits(
            max_total_specialist_turns=20,
            max_absolute_specialist_turns=19,
        )


@pytest.mark.asyncio
async def test_live_registry_routes_terra_text_roles_to_deepseek() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="openai-test-key",
        deepseek_api_key="deepseek-test-key",
    )
    registry = create_live_agent_registry(settings)
    # Case records are transcribed by pinned local OCR, not by a provider.
    assert registry.case_document.engine.name == "rapidocr-ppocrv6"
    document_client = registry.test_document.client
    text_client = registry.case_summary.client
    try:
        assert document_client.provider == "openai"
        assert document_client.model == "gpt-5.6-terra"
        assert document_client.timeout_seconds == 180
        assert document_client.max_retries == 0
        assert registry.radiologist.batch_max_images == 4
        assert registry.radiologist.batch_max_bytes == 12 * 1024 * 1024
        assert registry.radiologist.batch_concurrency == 2
        assert document_client.image_detail == "original"
        assert registry.radiologist.client.image_detail == "original"
        assert registry.radiologist.client.model == "gpt-5.6-terra"
        assert registry.radiologist.client.provider == "openai"
        assert registry.final_synthesis.client.model == "gpt-5.6-sol"
        assert registry.final_reviewer.client.model == "gpt-5.6-sol"
        assert registry.final_synthesis.client is not registry.final_reviewer.client
        # Terra text roles use DeepSeek-V4-Pro with max thinking by default.
        assert text_client.provider == "deepseek"
        assert text_client.model == "deepseek-v4-pro"
        assert text_client.thinking == "max"
        assert text_client.timeout_seconds == 300
        assert text_client.max_retries == 3
        assert registry.moderator.client is text_client
        assert registry.spine_surgeon.client is text_client
        assert registry.anesthesiologist.client is text_client
        assert registry.neurologist.client is text_client
        assert registry.perioperative_physician.client is text_client
        assert text_client is not document_client
        assert text_client is not registry.radiologist.client
    finally:
        await document_client.aclose()
        await registry.radiologist.client.aclose()
        await registry.final_synthesis.client.aclose()
        await registry.final_reviewer.client.aclose()
        await text_client.aclose()


@pytest.mark.asyncio
async def test_provider_routing_is_configurable() -> None:
    openai_only = Settings(
        _env_file=None,
        openai_api_key="openai-test-key",
        text_provider="openai",
    )
    registry = create_live_agent_registry(openai_only)
    text_client = registry.case_summary.client
    try:
        assert text_client.provider == "openai"
        assert text_client.model == "gpt-5.6-terra"
    finally:
        await text_client.aclose()

    deepseek_images = Settings(
        _env_file=None,
        openai_api_key="openai-test-key",
        deepseek_api_key="deepseek-test-key",
        image_provider="deepseek",
    )
    registry = create_live_agent_registry(deepseek_images)
    assert registry.case_document.engine.name == "rapidocr-ppocrv6"
    document_client = registry.test_document.client
    try:
        assert document_client.provider == "deepseek"
        assert document_client.model == "deepseek-v4-pro"
        assert document_client.supports_images is True
        assert registry.radiologist.client.provider == "deepseek"
    finally:
        await document_client.aclose()
        await registry.radiologist.client.aclose()
        await registry.final_synthesis.client.aclose()
        await registry.final_reviewer.client.aclose()


def test_missing_deepseek_key_fails_loud() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="openai-test-key",
        text_provider="deepseek",
    )

    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        settings.require_deepseek()


def test_log_filter_redacts_secret() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "token=%s", ("top-secret",), None)
    assert SecretRedactionFilter(["top-secret"]).filter(record)
    assert "top-secret" not in record.getMessage()
    assert "[REDACTED]" in record.getMessage()


def test_the_experiment_runner_retries_a_timed_out_page() -> None:
    """v5 transcribes without retries; the experiment runner does not follow it.

    Transcription is sequential and all-or-nothing. On a real four-case batch
    one page timed out on page 12 of 22 and discarded the eleven pages already
    paid for. The retry is narrow -- only a timeout, a rate limit or a dropped
    connection is retried, never a response that arrived -- so nothing already
    transcribed is transcribed again and made to disagree with itself.
    """

    from tools.run_department_experiment import DOCUMENT_RETRIES

    assert DOCUMENT_RETRIES == 2
    assert not hasattr(Settings(_env_file=None), "document_max_retries")
