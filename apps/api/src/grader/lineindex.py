"""Builds the global line index from per-page engine output.

Line IDs are allocated here rather than by the engines, for one reason: they must
run in reading order across the whole document. An engine that numbered its own
output would restart at each page, and the index's ordering is the thing that
later lets a span cross a page boundary by naming two IDs.
"""

from __future__ import annotations

from vedaai_contracts import DocumentKind, Line, LineIndex, OcrEngine

from .ocr.base import TranscribedLine
from .reading_order import order_lines

#: Prefix per document, so an ID carries which document it came from. These IDs
#: are pasted into prompts alongside both documents' lines, and an unprefixed
#: integer would let a model conflate a question with an answer.
_PREFIX: dict[DocumentKind, str] = {
    DocumentKind.QUESTION_PAPER: "qp",
    DocumentKind.ANSWER_SHEET: "as",
}


def sort_reading_order(lines: list[TranscribedLine]) -> list[TranscribedLine]:
    """Order lines within one page: banded top-to-bottom, then left-to-right.

    A per-page pre-order only, and column-blind: on a two-column page it
    interleaves the columns, because the first line of the right column sits
    level with the first line of the left.

    That is not the ordering the system relies on. Full reading order — full-width
    headings splitting the page into bands, gutters found by projection — lives in
    ``grader.reading_order`` and runs over the assembled index, which is where
    line IDs are allocated and therefore where order has to be correct. This
    function exists to give each page a stable starting order before that pass.

    Lines are bucketed into bands before sorting horizontally, because raw ``y0``
    ordering scrambles words on a shared baseline whose boxes differ by a pixel.
    """
    if not lines:
        return []

    band = _median_line_height(lines) * 0.6
    return sorted(lines, key=lambda ln: (round(ln.box.y0 / band) if band > 0 else 0, ln.box.x0))


def _median_line_height(lines: list[TranscribedLine]) -> float:
    heights = sorted(ln.box.y1 - ln.box.y0 for ln in lines)
    if not heights:
        return 0.0
    mid = len(heights) // 2
    return heights[mid]


def build_index(
    kind: DocumentKind,
    per_page: list[list[TranscribedLine]],
    engine: OcrEngine,
    *,
    trust_engine_order: bool = False,
    reading_order_confidence: float = 1.0,
) -> LineIndex:
    """Assemble a ``LineIndex``, assigning IDs in reading order.

    ``trust_engine_order`` preserves the order the engine produced. That is right
    for a PDF text layer, where the producing application's own block and line
    numbering reflects the document's structure — including column structure —
    more reliably than any geometric heuristic could reconstruct. It is wrong for
    OCR output, which typically arrives in detection order.
    """
    # Build lines first, order them across the whole document, and only then
    # allocate IDs. Numbering before ordering would produce IDs that do not run
    # in reading order, and ``resolve_span`` depends on them doing so — a span is
    # named by two IDs and means everything between them.
    prefix = _PREFIX[kind]
    staged: list[Line] = []
    for page_index, page_lines in enumerate(per_page):
        ordered = page_lines if trust_engine_order else sort_reading_order(page_lines)
        for transcribed in ordered:
            staged.append(
                Line(
                    line_id=f"{prefix}:0000",  # placeholder, replaced below
                    kind=kind,
                    page=page_index,
                    box=transcribed.box,
                    text=transcribed.text,
                    confidence=transcribed.confidence,
                    engine=engine,
                    words=transcribed.words,
                )
            )

    if trust_engine_order:
        # A PDF's own block numbering already encodes structure, columns
        # included, more reliably than geometry can reconstruct it.
        final, confidence = staged, reading_order_confidence
    else:
        final, confidence = order_lines(staged)

    lines = [
        line.model_copy(update={"line_id": f"{prefix}:{i:04d}"})
        for i, line in enumerate(final, start=1)
    ]

    return LineIndex(
        kind=kind,
        lines=lines,
        engine=engine,
        reading_order_confidence=confidence,
    )


def numbered_text(index: LineIndex, *, max_chars: int | None = None) -> str:
    """Render the index as numbered lines, the form given to a model.

    This is the whole indirection in one function. The model sees IDs and text;
    it never sees pixels or coordinates, so it cannot emit a wrong coordinate.
    It can only name lines, and a named line either resolves to real geometry or
    fails validation.
    """
    out: list[str] = []
    used = 0
    for line in index.lines:
        entry = f"[{line.line_id}] {line.text}"
        if max_chars is not None and used + len(entry) > max_chars:
            out.append("[truncated]")
            break
        out.append(entry)
        used += len(entry) + 1
    return "\n".join(out)
