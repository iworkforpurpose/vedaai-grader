"""OCR output: the line index that all geometry is derived from.

This module defines the single most important data structure in the system.
The model is never shown page images and asked for coordinates; it is shown
*numbered lines of text* and asked which line IDs belong to which question.
Geometry is then looked up here. That indirection is what keeps highlight
accuracy independent of the model's spatial reasoning, which benchmarks put
at an IoU often below 0.2 for fine-grained text localization.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .documents import DocumentKind
from .geometry import BBox


class OcrEngine(StrEnum):
    """Which engine produced a transcription.

    Recorded per line, not per document, because the pipeline re-reads
    low-confidence regions with a different engine (crop-rezoom) and merges
    the results. Knowing the provenance of each line is what makes engine
    disagreement usable as a confidence signal.
    """

    PDF_TEXT_LAYER = "pdf_text"
    """Not OCR at all — geometry and text read straight from a PDF's embedded
    text layer. Exact and free, so it is preferred for printed question papers
    that carry one. Never used for the answer sheet, which is handwritten, and
    never trusted without a raster cross-check when the source is untrusted:
    a text layer can hold content that is invisible on the rendered page."""

    GOOGLE_CLOUD_VISION = "gcv"
    PADDLE_OCR_VL = "paddle"
    SYNTHETIC = "synthetic"  # ground truth from the synthetic generator


class Word(BaseModel):
    """A single recognized word with its own box.

    Optional because not every engine returns word-level geometry. Where it is
    available it buys two things: tighter highlights (trailing whitespace
    excluded) and sentence-level rubric citations.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    box: BBox
    confidence: float = Field(ge=0.0, le=1.0)


class Line(BaseModel):
    """One transcribed line of text with its geometry.

    ``line_id`` is unique across the whole submission and carries its document
    prefix (``qp:0042``, ``as:0107``). The prefix is not decoration: these IDs
    are pasted into prompts, and an unprefixed integer would let the model
    confuse a question-paper line with an answer-sheet line in exactly the
    situation where that mistake is hardest to notice.
    """

    model_config = ConfigDict(frozen=True)

    line_id: str = Field(pattern=r"^(qp|as):\d{4,}$")
    kind: DocumentKind
    page: int = Field(ge=0)
    box: BBox
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    engine: OcrEngine
    words: list[Word] = Field(default_factory=list)

    @computed_field
    @property
    def is_low_confidence(self) -> bool:
        """Whether this line should be considered for a crop-rezoom re-read."""
        return self.confidence < 0.60


class LineIndex(BaseModel):
    """All transcribed lines for one document, in reading order.

    Reading order is a property of this index, not of the individual lines,
    because establishing it requires page-level layout analysis (column
    detection) that no single line can perform. The list order *is* the reading
    order; ``Line.page`` and ``Line.box`` describe position, not sequence.
    """

    kind: DocumentKind
    lines: list[Line] = Field(default_factory=list)
    engine: OcrEngine
    reading_order_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agreement between geometric ordering and the model's proposed "
        "ordering. Low values flag multi-column papers where sequence is uncertain.",
    )

    def by_id(self) -> dict[str, Line]:
        return {ln.line_id: ln for ln in self.lines}

    def resolve_span(self, start_line_id: str, end_line_id: str) -> list[Line]:
        """Return the lines from ``start`` to ``end`` inclusive, in reading order.

        Raises on unknown IDs rather than skipping them. A model that invents a
        line ID has produced an unusable answer, and failing loudly here is what
        turns a hallucination into a retry instead of a silently empty highlight.
        """
        ids = [ln.line_id for ln in self.lines]
        try:
            i = ids.index(start_line_id)
            j = ids.index(end_line_id)
        except ValueError as exc:
            raise KeyError(
                f"span references a line_id not present in this index: {exc}"
            ) from exc
        if j < i:
            i, j = j, i
        return self.lines[i : j + 1]

    def span_geometry(self, start_line_id: str, end_line_id: str) -> list[tuple[int, BBox]]:
        """Geometry for a span, as one union box per page it touches.

        Per-page rather than one overall box: an answer running from the bottom
        of page 2 to the top of page 3 has no meaningful single rectangle, and
        unioning across pages would produce a box covering both entire pages.
        """
        lines = self.resolve_span(start_line_id, end_line_id)
        per_page: dict[int, list[BBox]] = {}
        for ln in lines:
            per_page.setdefault(ln.page, []).append(ln.box)
        return [(page, BBox.union_all(boxes)) for page, boxes in sorted(per_page.items())]
