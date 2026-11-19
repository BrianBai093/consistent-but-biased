"""Deterministic optical character recognition for clinical-record pages.

Case-record transcription runs on OCR rather than on a language model because
the same bytes must always produce the same text: a run of the experiment
compares three discussion arms against one frozen baseline, so any drift in the
baseline would be indistinguishable from an arm effect. OCR also cannot
summarise or fill in an illegible passage, which a language model will
cheerfully do.

The engine is a Protocol so offline tests never import a real OCR backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from pydantic import Field
from PIL import Image, UnidentifiedImageError

from exceptions import OcrError
from schemas import StrictModel

DET_MODEL_FILENAME = "PP-OCRv6_det_small.onnx"
REC_MODEL_FILENAME = "PP-OCRv6_rec_small.onnx"
CLS_MODEL_FILENAME = "ch_ppocr_mobile_v2.0_cls_mobile.onnx"


class OcrLine(StrictModel):
    """One recognised line of text and where it sits on the page.

    Geometry is normalised to the 0-1 range so that a consumer never needs the
    pixel dimensions, and so that slicing a page does not change how a line is
    described relative to its own image.
    """

    text: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    left: float = Field(ge=0, le=1)
    width: float = Field(ge=0, le=1)
    height: float = Field(ge=0, le=1)


class OcrEngine(Protocol):
    """Reads a page image and returns its lines in natural reading order."""

    name: str

    def recognize(self, image_path: str) -> list[OcrLine]: ...


def _reading_order(lines: Sequence[OcrLine]) -> list[OcrLine]:
    return sorted(lines, key=lambda line: (line.top, line.left))


class RapidOcr:
    """RapidOCR (ONNX Runtime) with explicitly pinned local model weights.

    Model paths are always passed explicitly. RapidOCR would otherwise download
    weights on first use, which would make transcription depend on the network
    and on whatever the upstream project publishes that day -- the opposite of
    the reproducibility this module exists to provide.
    """

    name = "rapidocr-ppocrv6"

    def __init__(self, model_dir: Path | str) -> None:
        directory = Path(model_dir)
        self._paths = {
            "Det.model_path": directory / DET_MODEL_FILENAME,
            "Rec.model_path": directory / REC_MODEL_FILENAME,
            "Cls.model_path": directory / CLS_MODEL_FILENAME,
        }
        missing = sorted(path.name for path in self._paths.values() if not path.is_file())
        if missing:
            raise OcrError(
                "Pinned OCR model weights are missing: " + ", ".join(missing)
            )
        self._backend = None

    def _load(self):
        """Build the ONNX session on first use.

        Weight paths are checked eagerly so a misconfigured deployment still
        fails at startup, but loading 30 MB of models is deferred so that merely
        assembling the agent registry stays cheap.
        """

        if self._backend is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:  # pragma: no cover - optional at import time
                raise OcrError("The RapidOCR backend is not installed.") from exc
            try:
                self._backend = RapidOCR(
                    params={key: str(path) for key, path in self._paths.items()}
                )
            except Exception as exc:  # pragma: no cover - backend build failure
                raise OcrError("The RapidOCR backend could not be initialised.") from exc
        return self._backend

    def recognize(self, image_path: str) -> list[OcrLine]:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            raise OcrError("A page image could not be decoded for OCR.") from exc
        if not width or not height:
            raise OcrError("A page image has no usable dimensions.")

        engine = self._load()
        try:
            result = engine(image_path)
        except Exception as exc:
            raise OcrError("The OCR backend failed to read a page image.") from exc

        texts = getattr(result, "txts", None) or ()
        scores = getattr(result, "scores", None) or ()
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(texts) == 0:
            return []

        lines: list[OcrLine] = []
        for index, text in enumerate(texts):
            stripped = text.strip()
            if not stripped:
                continue
            corners = boxes[index]
            xs = [float(corner[0]) for corner in corners]
            ys = [float(corner[1]) for corner in corners]
            left = min(xs) / width
            top = min(ys) / height
            lines.append(
                OcrLine(
                    text=stripped,
                    confidence=_clamp(float(scores[index]) if index < len(scores) else 0.0),
                    top=_clamp(top),
                    left=_clamp(left),
                    width=_clamp(max(xs) / width - left),
                    height=_clamp(max(ys) / height - top),
                )
            )
        return _reading_order(lines)


class FakeOcrEngine:
    """Scripted engine for offline tests; never touches a real OCR backend."""

    name = "fake-ocr"

    def __init__(
        self,
        responses: dict[str, list[OcrLine] | OcrError],
        *,
        default: list[OcrLine] | OcrError | None = None,
    ) -> None:
        self.responses = responses
        self.default = default
        self.calls: list[str] = []

    def recognize(self, image_path: str) -> list[OcrLine]:
        self.calls.append(image_path)
        response = self.responses.get(image_path, self.default)
        if response is None:
            raise OcrError(f"No fake OCR response configured for {image_path}.")
        if isinstance(response, OcrError):
            raise response
        return _reading_order(response)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
