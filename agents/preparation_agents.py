"""Non-discussion agents for preparing preoperative source documents."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
)

from exceptions import (
    DocumentPageError,
    ModelInvocationError,
    OcrError,
    RetryableModelError,
    StructuredOutputError,
)
from schemas import (
    CaseDocumentMarkdown,
    CaseSummaryMarkdown,
    DocumentPage,
    PreparedMarkdown,
    TestDocumentMarkdown,
)
from services.llm_client import ModelResult, StructuredModelClient, aggregate_usage
from services.ocr_engine import OcrEngine, OcrLine
from services.retry_policy import model_retry_wait


PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts"


class _PreparationAgent:
    prompt_filename: str
    output_model = PreparedMarkdown

    def __init__(
        self,
        client: StructuredModelClient,
        *,
        previous_page_tail_chars: int = 1200,
    ) -> None:
        self.client = client
        self.previous_page_tail_chars = previous_page_tail_chars
        self.system_prompt = (PROMPT_ROOT / self.prompt_filename).read_text(
            encoding="utf-8"
        ).strip()

    async def _invoke_once(
        self,
        *,
        user_prompt: str,
        pages: Sequence[DocumentPage] = (),
    ) -> ModelResult[PreparedMarkdown]:
        result = await self.client.generate_structured(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            output_model=self.output_model,
            images=pages,
        )
        try:
            output = self.output_model.model_validate(result.output)
        except ValidationError as exc:
            raise StructuredOutputError(
                "Document preparation returned invalid structured output."
            ) from exc
        return ModelResult(
            output=output,
            usage=result.usage,
            usages=result.all_usages,
        )

    async def _invoke(
        self,
        *,
        user_prompt: str,
        pages: Sequence[DocumentPage] = (),
    ) -> ModelResult[PreparedMarkdown]:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.client.max_retries + 1),
            wait=model_retry_wait(self.client.provider),
            retry=retry_if_exception_type(RetryableModelError),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                return await self._invoke_once(
                    user_prompt=user_prompt,
                    pages=pages,
                )
        raise StructuredOutputError("Document preparation produced no result.")

    async def _transcribe_pages(
        self,
        *,
        task: str,
        pages: Sequence[DocumentPage],
    ) -> ModelResult[PreparedMarkdown]:
        if not pages:
            raise ValueError("Document transcription requires at least one page")
        sections: list[str] = []
        previous_page_body = ""
        warnings: list[str] = []
        usages = []
        for input_position, page in enumerate(pages, start=1):
            previous_markdown = (
                previous_page_body[-self.previous_page_tail_chars :]
                if self.previous_page_tail_chars
                else ""
            )
            try:
                result = await self._invoke_once(
                    user_prompt=json.dumps(
                        {
                            "task": task,
                            "instructions": (
                                "The previous transcription is read-only context. Transcribe "
                                "only the current image and return only the current page body. "
                                "Do not repeat or revise previous pages, and do not emit a "
                                "source marker."
                            ),
                            "current_page": {
                                "input_position": input_position,
                                "page_id": page.page_id,
                                "page_number": page.page_number,
                            },
                            "previous_transcription_md": previous_markdown or None,
                        },
                        ensure_ascii=False,
                    ),
                    pages=[page],
                )
                body = _page_body(result.output.markdown)
                if not body:
                    raise StructuredOutputError(
                        "Document transcription returned an empty current page."
                    )
            except ModelInvocationError as exc:
                raise DocumentPageError(page.page_id, exc) from exc

            sections.append(f"<!-- source: {page.page_id} -->\n\n{body}")
            previous_page_body = body
            warnings.extend(result.output.warnings)
            usages.extend(result.all_usages)

        document = self.output_model(
            markdown="\n\n".join(sections),
            warnings=_merge_unique(warnings),
        )
        return ModelResult(
            output=document,
            usage=aggregate_usage(usages),
            usages=tuple(usages),
        )


class CaseDocumentAgent(_PreparationAgent):
    prompt_filename = "case_document.md"
    output_model = CaseDocumentMarkdown

    async def run(self, pages: Sequence[DocumentPage]) -> ModelResult[PreparedMarkdown]:
        return await self._transcribe_pages(
            task="Transcribe the ordered preoperative case-document pages to Markdown.",
            pages=pages,
        )


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """The shape a graph node needs from a transcriber, without a model call.

    OCR is not a provider request, so it reports no ``ProviderUsage``: run
    metadata must keep counting model calls only.
    """

    output: PreparedMarkdown

    @property
    def all_usages(self) -> tuple[()]:
        return ()


class CaseDocumentTranscriber:
    """Deterministic OCR transcription of ordered case-record pages.

    This replaces a language model for case records so that the same page bytes
    always produce the same Markdown, which is what lets one frozen baseline be
    shared across the three discussion arms. It also structurally cannot
    summarise or invent text for an illegible passage.

    Page provenance stays with the program: the ``<!-- source: ... -->`` markers
    and the page order are assembled here, exactly as in the model-backed path.
    """

    output_model = CaseDocumentMarkdown

    def __init__(
        self,
        engine: OcrEngine,
        *,
        low_confidence_threshold: float = 0.95,
    ) -> None:
        self.engine = engine
        self.low_confidence_threshold = low_confidence_threshold

    async def run(self, pages: Sequence[DocumentPage]) -> TranscriptionResult:
        if not pages:
            raise ValueError("Document transcription requires at least one page")

        sections: list[str] = []
        warnings: list[str] = []
        for page in pages:
            lines = await self._recognize(page)
            body = "\n".join(line.text for line in lines).strip()
            if not body:
                raise DocumentPageError(
                    page.page_id,
                    OcrError("OCR found no text on a case-record page."),
                )
            sections.append(f"<!-- source: {page.page_id} -->\n\n{body}")
            warning = self._low_confidence_warning(page, lines)
            if warning:
                warnings.append(warning)

        document = self.output_model(
            markdown="\n\n".join(sections),
            warnings=_merge_unique(warnings),
        )
        return TranscriptionResult(output=document)

    async def _recognize(self, page: DocumentPage) -> list[OcrLine]:
        try:
            # OCR is blocking CPU work; running it inline would stall the event
            # loop and with it every in-flight SSE stream.
            return await asyncio.to_thread(self.engine.recognize, page.path_or_url)
        except OcrError as exc:
            raise DocumentPageError(page.page_id, exc) from exc

    def _low_confidence_warning(
        self,
        page: DocumentPage,
        lines: Sequence[OcrLine],
    ) -> str | None:
        suspect = [
            position
            for position, item in enumerate(lines, start=1)
            if item.confidence < self.low_confidence_threshold
        ]
        if not suspect:
            return None
        positions = ", ".join(str(position) for position in suspect)
        return (
            f"[OCR可疑] {page.page_id}: 第 {positions} 行 "
            f"（共 {len(suspect)}/{len(lines)} 行低于置信度阈值 "
            f"{self.low_confidence_threshold}）"
        )


class TestDocumentAgent(_PreparationAgent):
    prompt_filename = "test_document.md"
    output_model = TestDocumentMarkdown

    async def run(self, pages: Sequence[DocumentPage]) -> ModelResult[PreparedMarkdown]:
        return await self._transcribe_pages(
            task="Transcribe the ordered preoperative test-report pages to Markdown.",
            pages=pages,
        )


class CaseSummaryAgent(_PreparationAgent):
    prompt_filename = "case_summary.md"
    output_model = CaseSummaryMarkdown

    async def run(
        self,
        *,
        case_preop_md: str,
        tests_preop_md: str,
    ) -> ModelResult[PreparedMarkdown]:
        return await self._invoke(
            user_prompt=json.dumps(
                {
                    "task": "Prepare a concise clinical summary for independent radiology review.",
                    "case_preop_md": case_preop_md,
                    "tests_preop_md": tests_preop_md,
                },
                ensure_ascii=False,
            )
        )


_SOURCE_MARKER = re.compile(r"<!--\s*source\s*:[\s\S]*?-->", re.IGNORECASE)


def _page_body(markdown: str) -> str:
    """Remove model-generated reserved markers before deterministic assembly."""

    return _SOURCE_MARKER.sub("", markdown).strip()


def _merge_unique(values) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            merged.append(value)
    return merged
