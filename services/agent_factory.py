"""Lazy construction of the live provider-backed MDT agent registry."""

from agents import (
    AgentRegistry,
    AnesthesiologistAgent,
    CaseDocumentTranscriber,
    CaseSummaryAgent,
    FinalReviewerAgent,
    FinalSynthesisAgent,
    ModeratorAgent,
    NeurologistAgent,
    PerioperativePhysicianAgent,
    RadiologistAgent,
    SpineSurgeonAgent,
    TestDocumentAgent,
)
from config import ProviderConfig, Settings
from services.llm_client import DeepSeekTextClient
from services.multimodal_client import OpenAIResponsesClient
from services.ocr_engine import RapidOcr


def text_client(
    settings: Settings,
    openai_config: ProviderConfig,
) -> DeepSeekTextClient | OpenAIResponsesClient:
    if settings.text_provider == "deepseek":
        return DeepSeekTextClient(
            settings.require_deepseek(),
            thinking=settings.deepseek_thinking,
        )
    return OpenAIResponsesClient(
        openai_config,
        image_detail=settings.openai_image_detail,
    )


def image_client(
    settings: Settings,
    openai_config: ProviderConfig,
    *,
    timeout_seconds: float,
    max_retries: int,
) -> DeepSeekTextClient | OpenAIResponsesClient:
    if settings.image_provider == "deepseek":
        return DeepSeekTextClient(
            settings.require_deepseek().model_copy(
                update={
                    "timeout_seconds": timeout_seconds,
                    "max_retries": max_retries,
                }
            ),
            thinking=settings.deepseek_thinking,
            supports_images=True,
            image_detail=settings.openai_image_detail,
        )
    return OpenAIResponsesClient(
        openai_config.model_copy(
            update={
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries,
            }
        ),
        image_detail=settings.openai_image_detail,
    )


def create_live_agent_registry(settings: Settings) -> AgentRegistry:
    openai_config = settings.require_openai()
    shared_text_client = text_client(settings, openai_config)
    document_client = image_client(
        settings,
        openai_config,
        timeout_seconds=settings.document_timeout_seconds,
        max_retries=0,
    )
    radiology_client = image_client(
        settings,
        openai_config,
        timeout_seconds=settings.radiology_timeout_seconds,
        max_retries=settings.radiology_max_retries,
    )
    critical_synthesis_client = OpenAIResponsesClient(
        settings.require_openai_critical(),
        image_detail=settings.openai_image_detail,
    )
    critical_review_client = OpenAIResponsesClient(
        settings.require_openai_critical(),
        image_detail=settings.openai_image_detail,
    )
    return AgentRegistry(
        case_document=CaseDocumentTranscriber(
            RapidOcr(settings.ocr_model_dir),
            low_confidence_threshold=settings.ocr_low_confidence_threshold,
        ),
        test_document=TestDocumentAgent(
            document_client,
            previous_page_tail_chars=settings.document_previous_page_tail_chars,
        ),
        case_summary=CaseSummaryAgent(shared_text_client),
        moderator=ModeratorAgent(shared_text_client),
        final_synthesis=FinalSynthesisAgent(critical_synthesis_client),
        final_reviewer=FinalReviewerAgent(critical_review_client),
        radiologist=RadiologistAgent(
            radiology_client,
            batch_max_images=settings.radiology_batch_max_images,
            batch_max_bytes=settings.radiology_batch_max_bytes,
            batch_concurrency=settings.radiology_batch_concurrency,
        ),
        spine_surgeon=SpineSurgeonAgent(shared_text_client),
        anesthesiologist=AnesthesiologistAgent(shared_text_client),
        neurologist=NeurologistAgent(shared_text_client),
        perioperative_physician=PerioperativePhysicianAgent(shared_text_client),
    )
