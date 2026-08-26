"""Tests for the line index — the indirection that keeps models away from geometry.

The pipeline's central bet is that a model can reliably say *which lines* an
answer occupies even though it cannot reliably say *where* the answer is. These
tests cover the two places that bet could quietly fail: a model naming a line
that does not exist, and a span crossing a page boundary.
"""

from __future__ import annotations

import pytest

from vedaai_contracts import BBox, DocumentKind, Line, LineIndex, OcrEngine


def line(line_id: str, page: int, y0: float, y1: float, text: str = "x") -> Line:
    return Line(
        line_id=line_id,
        kind=DocumentKind.ANSWER_SHEET,
        page=page,
        box=BBox(x0=0.1, y0=y0, x1=0.9, y1=y1),
        text=text,
        confidence=0.9,
        engine=OcrEngine.SYNTHETIC,
    )


@pytest.fixture
def index() -> LineIndex:
    """Six lines: four on page 0, two on page 1 — so spans can cross pages."""
    return LineIndex(
        kind=DocumentKind.ANSWER_SHEET,
        engine=OcrEngine.SYNTHETIC,
        lines=[
            line("as:0001", 0, 0.10, 0.14, "Answer to Q1"),
            line("as:0002", 0, 0.16, 0.20, "continues here"),
            line("as:0003", 0, 0.30, 0.34, "Answer to Q2"),
            line("as:0004", 0, 0.80, 0.86, "runs to the page foot"),
            line("as:0005", 1, 0.08, 0.12, "and onto the next page"),
            line("as:0006", 1, 0.20, 0.24, "Answer to Q3"),
        ],
    )


class TestLineIdDiscipline:
    @pytest.mark.parametrize("bad_id", ["0001", "x:0001", "as:1", "as-0001", "AS:0001"])
    def test_rejects_ids_without_a_document_prefix(self, bad_id: str) -> None:
        # These IDs get pasted into prompts alongside both documents' lines. An
        # unprefixed integer would let a model confuse a question-paper line
        # with an answer-sheet line, in precisely the case that is hardest to
        # spot afterwards.
        with pytest.raises(ValueError):
            line(bad_id, 0, 0.1, 0.2)

    @pytest.mark.parametrize("good_id", ["qp:0001", "as:9999", "as:000123"])
    def test_accepts_prefixed_ids(self, good_id: str) -> None:
        assert line(good_id, 0, 0.1, 0.2).line_id == good_id


class TestSpanResolution:
    def test_resolves_an_inclusive_span(self, index: LineIndex) -> None:
        got = index.resolve_span("as:0001", "as:0003")
        assert [ln.line_id for ln in got] == ["as:0001", "as:0002", "as:0003"]

    def test_resolves_a_single_line_span(self, index: LineIndex) -> None:
        got = index.resolve_span("as:0003", "as:0003")
        assert [ln.line_id for ln in got] == ["as:0003"]

    def test_tolerates_a_reversed_span(self, index: LineIndex) -> None:
        # Cheap to normalize, and a model emitting end-before-start is a
        # transposition rather than a hallucination — no reason to fail.
        got = index.resolve_span("as:0003", "as:0001")
        assert [ln.line_id for ln in got] == ["as:0001", "as:0002", "as:0003"]

    def test_raises_on_an_invented_line_id(self, index: LineIndex) -> None:
        # The hallucination guard. Failing loudly turns a fabricated span into a
        # retry; skipping unknown IDs would produce a silently empty highlight
        # that looks like a mapping miss.
        with pytest.raises(KeyError, match="not present in this index"):
            index.resolve_span("as:0001", "as:4242")


class TestSpanGeometry:
    def test_unions_lines_within_a_single_page(self, index: LineIndex) -> None:
        geom = index.span_geometry("as:0001", "as:0002")
        assert len(geom) == 1
        page, box = geom[0]
        assert page == 0
        assert box.y0 == pytest.approx(0.10)
        assert box.y1 == pytest.approx(0.20)

    def test_splits_a_multi_page_span_per_page(self, index: LineIndex) -> None:
        # The property that makes multi-page answers work without special
        # casing. One union box across pages would cover both entire pages,
        # which is why geometry is grouped rather than merged.
        geom = index.span_geometry("as:0004", "as:0006")
        assert [page for page, _ in geom] == [0, 1]

        page0_box = geom[0][1]
        assert page0_box.y0 == pytest.approx(0.80)
        assert page0_box.y1 == pytest.approx(0.86)

        page1_box = geom[1][1]
        assert page1_box.y0 == pytest.approx(0.08)
        assert page1_box.y1 == pytest.approx(0.24)

    def test_returns_pages_in_ascending_order(self, index: LineIndex) -> None:
        geom = index.span_geometry("as:0006", "as:0004")
        assert [page for page, _ in geom] == [0, 1]


class TestLowConfidenceFlagging:
    def test_flags_lines_that_warrant_a_crop_rezoom(self) -> None:
        weak = Line(
            line_id="as:0007",
            kind=DocumentKind.ANSWER_SHEET,
            page=0,
            box=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
            text="illegible",
            confidence=0.31,
            engine=OcrEngine.GOOGLE_CLOUD_VISION,
        )
        assert weak.is_low_confidence

    def test_does_not_flag_confident_lines(self, index: LineIndex) -> None:
        assert not index.lines[0].is_low_confidence

    def test_by_id_covers_every_line(self, index: LineIndex) -> None:
        assert len(index.by_id()) == len(index.lines)
