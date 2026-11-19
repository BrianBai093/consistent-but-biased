# Pinned OCR model weights

Case-record transcription must be reproducible: the same page bytes have to
produce the same Markdown on any machine and at any later date, because the
three discussion arms of the experiment are compared against one frozen
baseline. RapidOCR would otherwise fetch weights on first use, which would make
transcription depend on the network and on whatever upstream publishes that day.

These files are therefore committed and passed to RapidOCR explicitly via
`Det.model_path` / `Rec.model_path` / `Cls.model_path`. A missing or renamed
file fails loudly at construction (`OcrError`) instead of silently falling back
to a download.

| File | SHA-256 | Size |
|---|---|---|
| `PP-OCRv6_det_small.onnx` | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` | 9.5 MB |
| `PP-OCRv6_rec_small.onnx` | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` | 20.3 MB |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` | 0.6 MB |

## Provenance

Distributed with `rapidocr==3.9.2` (Apache-2.0), which converts the PaddleOCR
models (Apache-2.0) to ONNX. Detection and recognition are PP-OCRv6 `small`;
angle classification is PP-OCRv4 `mobile`. The copies here were taken verbatim
from that package's `rapidocr/models/` directory; they are not retrained,
quantised, or otherwise modified.

## Verifying

```bash
shasum -a 256 models/ocr/*.onnx
```

Upstream: <https://github.com/RapidAI/RapidOCR> · <https://github.com/PaddlePaddle/PaddleOCR>
