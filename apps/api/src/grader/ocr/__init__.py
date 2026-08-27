"""Transcription engines and the logic that picks one.

The adapter interface exists because no benchmark tells us which engine reads a
given student's handwriting best. OmniDocBench, where the leading models score
in the mid-nineties, has no handwriting category at all, and IAM is a corpus of
adult writing. So engines are interchangeable and the golden set decides, per
document rather than once and for all.
"""

from __future__ import annotations

import os

from vedaai_contracts import DocumentKind, OcrEngine, SourceFile

from .base import EngineUnavailable, PageInput, TranscribedLine, TranscriptionEngine
from .native_pdf import PdfTextLayerEngine
from .paddle import PaddleOcrEngine
from .textract import TextractEngine

__all__ = [
    "EngineUnavailable",
    "PageInput",
    "TextractEngine",
    "TranscribedLine",
    "TranscriptionEngine",
    "PaddleOcrEngine",
    "PdfTextLayerEngine",
    "handwriting_engines",
    "select_engine",
    "trusts_own_order",
]

#: Which engine reads handwriting, from the environment.
#:
#: Named rather than inferred from whichever credentials happen to be present. A
#: deployment that silently fell back to a different recognizer would report
#: accuracy figures for an engine nobody chose, and the fallback is exactly the
#: case where the numbers differ.
OCR_ENGINE = os.getenv("OCR_ENGINE", "").strip().lower()


def handwriting_engines() -> list[TranscriptionEngine]:
    """Handwriting engines in the order they should be tried.

    Preference follows ``OCR_ENGINE`` when it names one. With nothing set, a
    hosted recognizer is preferred over the local model — the local one needs
    600 MB of weights and takes about fourteen seconds a page, which is the right
    default for a laptop and the wrong one for a service.
    """
    textract, paddle = TextractEngine(), PaddleOcrEngine()
    if OCR_ENGINE == "paddle":
        return [paddle, textract]
    if OCR_ENGINE == "textract":
        return [textract, paddle]
    return [textract, paddle]


def select_engine(source: SourceFile) -> TranscriptionEngine:
    """Choose an engine for a document.

    A printed question paper carrying a real text layer is read from that layer:
    the text is exact rather than recognized, the word boxes are exact rather
    than estimated, and it consumes no quota. Recognizing text that the document
    already states would trade accuracy for nothing.

    Anything else — every answer sheet, and any scanned paper — goes to the local
    handwriting recognizer.
    """
    if source.kind is DocumentKind.QUESTION_PAPER and source.has_text_layer:
        return PdfTextLayerEngine()

    for engine in handwriting_engines():
        if engine.available():
            return engine

    raise EngineUnavailable(
        f"no transcription engine available for {source.filename!r} "
        f"(kind={source.kind.value}, has_text_layer={source.has_text_layer}). "
        "Handwriting needs either AWS credentials for Textract — the 'aws' extra, "
        "with AWS_REGION and a role or key pair — or the local model, via "
        "`uv sync --extra ocr-local` in apps/api."
    )


def trusts_own_order(engine: OcrEngine) -> bool:
    """Whether an engine's output order should be preserved as reading order.

    True only for the PDF text layer, where the producing application's block
    and line numbering encodes the document's real structure — including
    columns — more reliably than a geometric heuristic can rebuild it. OCR
    output arrives in detection order and must be sorted.
    """
    return engine is OcrEngine.PDF_TEXT_LAYER
