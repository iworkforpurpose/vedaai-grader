"""Transcription from a PDF's embedded text layer.

For a printed, digitally-generated question paper this is strictly better than
OCR: the text is exact rather than recognized, word boxes are exact rather than
estimated, it costs nothing, consumes no quota, and cannot misread a character.
Given that question papers are typically typed, this is the default engine for
them and OCR is the fallback for scans.

It is never used for the answer sheet. Handwriting has no text layer, and a
scanned sheet's layer is either absent or spurious.

One caveat worth being explicit about. A PDF's text layer can contain content
that is invisible on the rendered page — white text, zero-size fonts, glyphs
positioned off-page. That is the mechanism behind hidden-prompt injection, and
it is also a plain correctness hazard, since invisible text would be extracted
as though it were a question. Two mitigations are applied here: glyphs outside
the page rectangle are dropped, and empty or whitespace-only lines are skipped.
A stronger check — comparing this extraction against OCR of the rendered raster,
which is what PhantomLint does — belongs at the pipeline level and is worth
adding for untrusted input.
"""

from __future__ import annotations

import fitz  # PyMuPDF
from vedaai_contracts import BBox, OcrEngine, Word

from .base import PageInput, TranscribedLine

#: Indices into PyMuPDF's ``get_text("words")`` tuples.
_X0, _Y0, _X1, _Y1, _TEXT, _BLOCK, _LINE, _WORD = range(8)


class PdfTextLayerEngine:
    """Reads text and geometry directly from the PDF."""

    @property
    def engine(self) -> OcrEngine:
        return OcrEngine.PDF_TEXT_LAYER

    def available(self) -> bool:
        return True

    def transcribe(self, page: PageInput) -> list[TranscribedLine]:
        if page.document is None:
            return []

        doc = fitz.open(stream=page.document, filetype="pdf")
        try:
            if page.index >= doc.page_count:
                return []
            pdf_page = doc[page.index]

            # Normalize against the page rectangle in points rather than the
            # rendered pixel size. The two are proportional, so the result is
            # identical, but this keeps DPI out of the conversion entirely —
            # one less place for a units mistake to hide.
            rect = pdf_page.rect
            page_w, page_h = rect.width, rect.height
            if page_w <= 0 or page_h <= 0:
                return []

            words = pdf_page.get_text("words")
            hidden = _invisible_regions(pdf_page)
            return _group_into_lines(words, page_w, page_h, hidden)
        finally:
            doc.close()


#: A font size at or below this is not text anybody is meant to read.
_INVISIBLE_SIZE = 1.0

#: How close to white a fill has to be before it is treated as invisible on a
#: white page. Papers do print in colour, and only invisibility is suspicious, so
#: this is deliberately tight: a pale grey watermark is still readable and still
#: read.
_WHITE = 0.98


def _invisible_regions(pdf_page) -> list[tuple[float, float, float, float]]:
    """Boxes holding text a person looking at this page would not see.

    The question paper is fully attacker-controlled and its text reaches a marking
    model. Pages are rasterised before recognition precisely so a text layer never
    gets there — but a paper that *has* a text layer takes the direct path, which
    is exact and free and what every typed paper uses. On that path the only thing
    dropped was a glyph lying outside the page rectangle.

    So white-on-white text and text set at a hundredth of a point were extracted as
    ordinary question text and pasted into the prompt, with nothing about the
    resulting question looking wrong to anyone reading the paper.

    Reported by box rather than by string so the caller can drop the words that
    fall inside one, without this function needing to know how words are grouped.
    """
    regions: list[tuple[float, float, float, float]] = []
    try:
        blocks = pdf_page.get_text("dict").get("blocks", [])
    except Exception:  # noqa: BLE001 - a page that will not describe itself is read as-is
        return regions

    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if not str(span.get("text", "")).strip():
                    continue
                size = float(span.get("size", 0.0) or 0.0)
                colour = int(span.get("color", 0) or 0)
                channels = ((colour >> 16) & 255, (colour >> 8) & 255, colour & 255)
                white = all(c / 255.0 >= _WHITE for c in channels)
                if size <= _INVISIBLE_SIZE or white:
                    regions.append(tuple(span["bbox"]))  # type: ignore[arg-type]
    return regions


def _inside_any(
    box: tuple[float, float, float, float],
    regions: list[tuple[float, float, float, float]],
) -> bool:
    """Whether a word's box sits within one of the invisible spans.

    Centre-in-box rather than full containment: the span box and the word box come
    from two different extractions of the same glyphs and agree to within rounding,
    not exactly.
    """
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in regions)


def _group_into_lines(
    words: list[tuple],
    page_w: float,
    page_h: float,
    hidden: list[tuple[float, float, float, float]] | None = None,
) -> list[TranscribedLine]:
    """Group word tuples into lines using PyMuPDF's block and line numbering.

    Grouping by the reported ``(block, line)`` pair rather than by y-coordinate
    proximity matters on multi-column papers: two columns sit at the same
    vertical position, and a y-based grouping would splice a line from the left
    column onto one from the right.
    """
    buckets: dict[tuple[int, int], list[tuple]] = {}
    for w in words:
        text = str(w[_TEXT])
        if not text.strip():
            continue

        # Drop glyphs outside the page. This is where off-page hidden text is
        # discarded, and it also removes bleed from malformed documents.
        if w[_X1] <= 0 or w[_Y1] <= 0 or w[_X0] >= page_w or w[_Y0] >= page_h:
            continue

        # Text a reader cannot see is not this paper's content, whatever it says.
        if hidden and _inside_any((w[_X0], w[_Y0], w[_X1], w[_Y1]), hidden):
            continue

        buckets.setdefault((int(w[_BLOCK]), int(w[_LINE])), []).append(w)

    lines: list[TranscribedLine] = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda w: w[_WORD])

        word_models: list[Word] = []
        for w in group:
            box = _clamped_box(w[_X0], w[_Y0], w[_X1], w[_Y1], page_w, page_h)
            if box is None:
                continue
            word_models.append(Word(text=str(w[_TEXT]), box=box, confidence=1.0))

        if not word_models:
            continue

        line_box = BBox.union_all([wm.box for wm in word_models])
        lines.append(
            TranscribedLine(
                text=" ".join(wm.text for wm in word_models),
                box=line_box,
                # The text layer is not a recognition result; there is no
                # uncertainty to report. This is precisely why it is preferred
                # over OCR when available.
                confidence=1.0,
                words=word_models,
            )
        )
    return lines


def _clamped_box(
    x0: float, y0: float, x1: float, y1: float, page_w: float, page_h: float
) -> BBox | None:
    """Normalize and clamp a word box, or None if it degenerates.

    Glyphs can extend a hair past the page edge through rounding or a bad
    bounding box. Clamping keeps them inside the unit square the contract
    requires; a box that collapses to zero area once clamped is discarded rather
    than forced through, since it would render as an invisible highlight.
    """
    nx0 = max(0.0, min(1.0, x0 / page_w))
    ny0 = max(0.0, min(1.0, y0 / page_h))
    nx1 = max(0.0, min(1.0, x1 / page_w))
    ny1 = max(0.0, min(1.0, y1 / page_h))

    if nx1 <= nx0 or ny1 <= ny0:
        return None
    return BBox(x0=nx0, y0=ny0, x1=nx1, y1=ny1)
