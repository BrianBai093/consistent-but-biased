import json

import pytest

from agents import CaseDocumentAgent, TestDocumentAgent as LabDocumentAgent
from exceptions import (
    DocumentPageError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestError,
    ModelServiceError,
    ModelTimeoutError,
)
from schemas import (
    CaseDocumentMarkdown,
    DocumentPage,
    ProviderUsage,
    TestDocumentMarkdown as LabDocumentOutput,
)
from services.llm_client import ModelResult


def pages(prefix: str, count: int) -> list[DocumentPage]:
    return [
        DocumentPage(
            page_id=f"{prefix}-page-{index:03d}",
            page_number=index,
            path_or_url=f"https://private.invalid/{prefix}/{index}.jpg",
            original_filename=f"patient-{prefix}-{index}.jpg",
        )
        for index in range(1, count + 1)
    ]


class OrderedDocumentClient:
    provider = "fake"
    model = "ordered-document-fake"
    max_retries = 3

    def __init__(self) -> None:
        self.calls = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        output_model = kwargs["output_model"]
        page = kwargs["images"][0]
        return ModelResult(
            output=output_model(
                markdown=(
                    f"<!-- source: model-must-not-own-markers -->\n\n"
                    f"transcription for page {page.page_number}"
                ),
                warnings=["[无法辨认]"],
            ),
            usage=ProviderUsage(provider="fake", model=self.model),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_type", "prefix", "count", "output_model"),
    [
        (CaseDocumentAgent, "case", 6, CaseDocumentMarkdown),
        (LabDocumentAgent, "test", 10, LabDocumentOutput),
    ],
)
async def test_document_pages_use_one_image_and_only_previous_page_tail(
    agent_type,
    prefix: str,
    count: int,
    output_model,
) -> None:
    client = OrderedDocumentClient()
    source_pages = pages(prefix, count)

    result = await agent_type(client).run(source_pages)

    assert isinstance(result.output, output_model)
    assert len(client.calls) == count
    assert all(len(call["images"]) == 1 for call in client.calls)
    assert [call["images"][0].page_id for call in client.calls] == [
        page.page_id for page in source_pages
    ]
    assert result.output.warnings == ["[无法辨认]"]
    assert len(result.all_usages) == count

    for index, call in enumerate(client.calls):
        prompt = json.loads(call["user_prompt"])
        assert prompt["current_page"]["page_id"] == source_pages[index].page_id
        previous = prompt["previous_transcription_md"]
        if index == 0:
            assert previous is None
        else:
            assert previous == f"transcription for page {index}"
            assert "<!-- source:" not in previous
        assert "private.invalid" not in call["user_prompt"]
        assert "patient-" not in call["user_prompt"]

    assert result.output.markdown == _expected_markdown(source_pages)
    assert "model-must-not-own-markers" not in result.output.markdown


@pytest.mark.asyncio
async def test_new_page_output_cannot_rewrite_existing_markdown() -> None:
    client = OrderedDocumentClient()
    source_pages = pages("case", 3)

    first_two = await CaseDocumentAgent(client).run(source_pages[:2])
    first_two_markdown = first_two.output.markdown
    client.calls.clear()
    complete = await CaseDocumentAgent(client).run(source_pages)

    assert complete.output.markdown.startswith(first_two_markdown)
    assert complete.output.markdown.count("transcription for page 1") == 1
    assert complete.output.markdown.count("transcription for page 2") == 1


class FailingDocumentClient(OrderedDocumentClient):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ModelTimeoutError("timeout"),
        ModelRateLimitError("rate limited"),
        ModelConnectionError("connection"),
        ModelServiceError("service"),
        ModelRequestError("rejected"),
    ],
)
async def test_document_page_failures_are_never_retried(error) -> None:
    client = FailingDocumentClient(error)

    with pytest.raises(DocumentPageError) as caught:
        await CaseDocumentAgent(client).run(pages("case", 2))

    assert caught.value.page_id == "case-page-001"
    assert caught.value.cause is error
    assert len(client.calls) == 1
    assert client.calls[0]["images"][0].page_id == "case-page-001"


def _expected_markdown(source_pages: list[DocumentPage]) -> str:
    return "\n\n".join(
        f"<!-- source: {page.page_id} -->\n\ntranscription for page {page.page_number}"
        for page in source_pages
    )
