import pytest

from exceptions import ModelInvocationError, OcrError
from services.ocr_engine import FakeOcrEngine, OcrLine, RapidOcr


def line(text: str, *, top: float, left: float = 0.1, confidence: float = 1.0) -> OcrLine:
    return OcrLine(
        text=text,
        confidence=confidence,
        top=top,
        left=left,
        width=0.5,
        height=0.02,
    )


def test_ocr_failures_travel_the_model_stage_path():
    """``DocumentPageError`` only accepts a ``ModelInvocationError`` cause."""

    assert issubclass(OcrError, ModelInvocationError)


def test_lines_are_returned_in_reading_order():
    engine = FakeOcrEngine(
        {
            "page.png": [
                line("second line", top=0.40),
                line("right column", top=0.10, left=0.60),
                line("left column", top=0.10, left=0.05),
            ]
        }
    )

    lines = engine.recognize("page.png")

    assert [item.text for item in lines] == [
        "left column",
        "right column",
        "second line",
    ]


def test_each_recognized_page_is_recorded_for_assertions():
    engine = FakeOcrEngine({}, default=[line("body", top=0.1)])

    engine.recognize("a.png")
    engine.recognize("b.png")

    assert engine.calls == ["a.png", "b.png"]


def test_a_scripted_failure_is_raised_for_that_page_only():
    engine = FakeOcrEngine(
        {"broken.png": OcrError("scanner exploded")},
        default=[line("body", top=0.1)],
    )

    with pytest.raises(OcrError):
        engine.recognize("broken.png")
    assert engine.recognize("fine.png")


def test_an_unscripted_page_without_a_default_fails_loudly():
    engine = FakeOcrEngine({})

    with pytest.raises(OcrError):
        engine.recognize("unexpected.png")


def test_geometry_stays_within_the_normalized_range():
    with pytest.raises(ValueError):
        OcrLine(text="x", confidence=1.0, top=1.4, left=0.0, width=0.1, height=0.1)
    with pytest.raises(ValueError):
        OcrLine(text="x", confidence=1.7, top=0.1, left=0.0, width=0.1, height=0.1)


def test_empty_text_is_rejected_so_a_blank_line_never_enters_a_document():
    with pytest.raises(ValueError):
        OcrLine(text="", confidence=1.0, top=0.1, left=0.0, width=0.1, height=0.1)


def test_missing_pinned_weights_fail_loudly_instead_of_downloading(tmp_path):
    """A silent download would make transcription depend on the network."""

    with pytest.raises(OcrError) as caught:
        RapidOcr(tmp_path)

    message = str(caught.value)
    assert "PP-OCRv6_det_small.onnx" in message
    assert "PP-OCRv6_rec_small.onnx" in message
    assert "ch_ppocr_mobile_v2.0_cls_mobile.onnx" in message
