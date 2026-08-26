"""Builds the global line index from per-page engine output.

Line IDs are allocated here rather than by the engines, for one reason: they must
run in reading order across the whole document. An engine that numbered its own
output would restart at each page, and the index's ordering is the thing that
later lets a span cross a page boundary by naming two IDs.
"""

from __future__ import annotations

from vedaai_contracts import DocumentKind, Line, LineIndex, OcrEngine

from .ocr.base import TranscribedLine

#: Prefix per document, so an ID carries which document it came from. These IDs
#: are pasted into prompts alongside both documents' lines, and an unprefixed
#: integer would let a model conflate a question with an answer.
_PREFIX: dict[DocumentKind, str] = {
    DocumentKind.QUESTION_PAPER: "qp",
    DocumentKind.ANSWER_SHEET: "as",
}


def sort_reading_order(lines: list[TranscribedLine]) -> list[TranscribedLine]:
    """Order lines within one page geometrically: top to bottom, then left to right.

    A deliberately naive baseline. It is correct for single-column pages and
    wrong for multi-column ones, where it interleaves columns — which is exactly
    the failure that column detection addresses later. Keeping the naive version
    behind a named function means the debug overlay shows the interleaving
    plainly instead of it hiding inside a larger routine.

    Lines are bucketed into bands before sorting horizontally, because raw
    ``y0`` ordering scrambles words on a shared baseline whose boxes differ by a
    pixel or two.
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
    lines: list[Line] = []
    counter = 0
    prefix = _PREFIX[kind]

    for page_index, page_lines in enumerate(per_page):
        ordered = page_lines if trust_engine_order else sort_reading_order(page_lines)
        for transcribed in ordered:
            counter += 1
            lines.append(
                Line(
                    line_id=f"{prefix}:{counter:04d}",
                    kind=kind,
                    page=page_index,
                    box=transcribed.box,
                    text=transcribed.text,
                    confidence=transcribed.confidence,
                    engine=engine,
                    words=transcribed.words,
                )
            )

    return LineIndex(
        kind=kind,
        lines=lines,
        engine=engine,
        reading_order_confidence=reading_order_confidence,
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
