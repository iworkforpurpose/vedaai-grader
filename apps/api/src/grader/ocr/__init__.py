"""Transcription engines and the logic that picks one.

The adapter interface exists because no benchmark tells us which engine reads a
given student's handwriting best. OmniDocBench, where the leading models score
in the mid-nineties, has no handwriting category at all, and IAM is a corpus of
adult writing. So engines are interchangeable and the golden set decides, per
document rather than once and for all.
"""

from __future__ import annotations

from vedaai_contracts import DocumentKind, OcrEngine, SourceFile

from .base import EngineUnavailable, PageInput, TranscribedLine, TranscriptionEngine
from .native_pdf import PdfTextLayerEngine

__all__ = [
    "EngineUnavailable",
    "PageInput",
    "TranscribedLine",
    "TranscriptionEngine",
    "PdfTextLayerEngine",
    "select_engine",
    "trusts_own_order",
]


def select_engine(source: SourceFile) -> TranscriptionEngine:
    """Choose an engine for a document.

    A printed question paper carrying a real text layer is read from that layer:
    the text is exact rather than recognized, the word boxes are exact rather
    than estimated, and it consumes no quota. Recognizing text that the document
    already states would trade accuracy for nothing.

    Anything else — every answer sheet, and any scanned paper — needs OCR.
    """
    if source.kind is DocumentKind.QUESTION_PAPER and source.has_text_layer:
        return PdfTextLayerEngine()

    # OCR engines land in Phase 1's remaining work; until one is configured this
    # raises with a message naming the fix rather than failing obscurely inside
    # a pipeline stage.
    raise EngineUnavailable(
        f"no transcription engine available for {source.filename!r} "
        f"(kind={source.kind.value}, has_text_layer={source.has_text_layer}). "
        "A handwriting OCR engine is required for answer sheets and scanned papers."
    )


def trusts_own_order(engine: OcrEngine) -> bool:
    """Whether an engine's output order should be preserved as reading order.

    True only for the PDF text layer, where the producing application's block
    and line numbering encodes the document's real structure — including
    columns — more reliably than a geometric heuristic can rebuild it. OCR
    output arrives in detection order and must be sorted.
    """
    return engine is OcrEngine.PDF_TEXT_LAYER
