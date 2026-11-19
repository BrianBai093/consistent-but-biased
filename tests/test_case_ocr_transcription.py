import pytest

from agents import CaseDocumentTranscriber
from exceptions import DocumentPageError, OcrError
from schemas import CaseDocumentMarkdown, DocumentPage
from services.ocr_engine import FakeOcrEngine, OcrLine


def pages(count: int) -> list[DocumentPage]:
    return [
        DocumentPage(
            page_id=f"case-page-{index:03d}",
            page_number=index,
            path_or_url=f"https://private.invalid/case/{index}.png",
            original_filename=f"patient-case-{index}.png",
        )
        for index in range(1, count + 1)
    ]


def line(text: str, *, top: float, confidence: float = 1.0) -> OcrLine:
    return OcrLine(
        text=text,
        confidence=confidence,
        top=top,
        left=0.1,
        width=0.7,
        height=0.02,
    )


def engine_for(
    document_pages: list[DocumentPage],
    *,
    confidence: float = 1.0,
) -> FakeOcrEngine:
    return FakeOcrEngine(
        {
            page.path_or_url: [
                line(f"line one of page {page.page_number}", top=0.1),
                line(f"line two of page {page.page_number}", top=0.2, confidence=confidence),
            ]
            for page in document_pages
        }
    )


async def test_pages_are_assembled_in_order_behind_program_owned_markers():
    document_pages = pages(3)
    transcriber = CaseDocumentTranscriber(engine_for(document_pages))

    result = await transcriber.run(document_pages)

    assert isinstance(result.output, CaseDocumentMarkdown)
    assert result.output.markdown == "\n\n".join(
        f"<!-- source: case-page-{index:03d} -->\n\n"
        f"line one of page {index}\nline two of page {index}"
        for index in (1, 2, 3)
    )


async def test_transcription_reports_no_model_usage():
    """OCR is not a provider request, so run metadata must not count one."""

    document_pages = pages(2)
    transcriber = CaseDocumentTranscriber(engine_for(document_pages))

    result = await transcriber.run(document_pages)

    assert result.all_usages == ()


async def test_each_page_is_read_exactly_once_and_in_order():
    document_pages = pages(4)
    engine = engine_for(document_pages)

    await CaseDocumentTranscriber(engine).run(document_pages)

    assert engine.calls == [page.path_or_url for page in document_pages]


async def test_identical_input_produces_byte_identical_output():
    document_pages = pages(3)

    first = await CaseDocumentTranscriber(engine_for(document_pages)).run(document_pages)
    second = await CaseDocumentTranscriber(engine_for(document_pages)).run(document_pages)

    assert first.output.markdown == second.output.markdown
    assert first.output.warnings == second.output.warnings


async def test_adding_a_page_appends_without_rewriting_earlier_pages():
    document_pages = pages(3)
    engine = engine_for(document_pages)

    partial = await CaseDocumentTranscriber(engine).run(document_pages[:2])
    complete = await CaseDocumentTranscriber(engine).run(document_pages)

    assert complete.output.markdown.startswith(partial.output.markdown)
    assert complete.output.markdown.count("line one of page 1") == 1


async def test_low_confidence_lines_are_flagged_without_blocking_the_run():
    document_pages = pages(2)
    transcriber = CaseDocumentTranscriber(
        engine_for(document_pages, confidence=0.42),
        low_confidence_threshold=0.95,
    )

    result = await transcriber.run(document_pages)

    assert result.output.warnings == [
        "[OCR可疑] case-page-001: 第 2 行 （共 1/2 行低于置信度阈值 0.95）",
        "[OCR可疑] case-page-002: 第 2 行 （共 1/2 行低于置信度阈值 0.95）",
    ]
    assert "line two of page 1" in result.output.markdown


async def test_confident_pages_produce_no_warnings():
    document_pages = pages(2)
    transcriber = CaseDocumentTranscriber(engine_for(document_pages))

    result = await transcriber.run(document_pages)

    assert result.output.warnings == []


async def test_an_engine_failure_names_the_page_and_stops_the_document():
    document_pages = pages(3)
    engine = FakeOcrEngine(
        {
            document_pages[0].path_or_url: [line("readable", top=0.1)],
            document_pages[1].path_or_url: OcrError("backend unavailable"),
        }
    )

    with pytest.raises(DocumentPageError) as caught:
        await CaseDocumentTranscriber(engine).run(document_pages)

    assert caught.value.page_id == "case-page-002"
    assert isinstance(caught.value.cause, OcrError)
    assert engine.calls == [page.path_or_url for page in document_pages[:2]]


async def test_a_page_without_recognizable_text_fails_closed():
    document_pages = pages(1)
    engine = FakeOcrEngine({document_pages[0].path_or_url: []})

    with pytest.raises(DocumentPageError) as caught:
        await CaseDocumentTranscriber(engine).run(document_pages)

    assert caught.value.page_id == "case-page-001"


async def test_an_empty_page_list_is_rejected():
    transcriber = CaseDocumentTranscriber(FakeOcrEngine({}))

    with pytest.raises(ValueError):
        await transcriber.run([])


async def test_local_paths_and_filenames_never_reach_the_transcript():
    document_pages = pages(2)
    transcriber = CaseDocumentTranscriber(engine_for(document_pages))

    result = await transcriber.run(document_pages)

    assert "private.invalid" not in result.output.markdown
    assert "patient-" not in result.output.markdown
