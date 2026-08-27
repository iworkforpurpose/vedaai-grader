"""Tests for the local handwriting engine adapter.

Scope note: these verify the *adapter*, not PaddleOCR's accuracy. What matters
here is that whatever the model returns is converted into geometry that obeys the
coordinate contract, because a box that escapes the unit square or inverts an
axis would render as a misplaced highlight rather than an error.

The model is slow to initialize — about 46 seconds — so it is built once for the
whole module and every case shares it.

Recognition quality on real handwriting is measured separately against real
scripts, which are not committed here: student work does not belong in a
repository. Point ``GRADER_SAMPLE_DIR`` at a directory of real pages to run that
check locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from vedaai_contracts import DocumentKind

from grader import render
from grader.lineindex import build_index, sort_reading_order
from grader.ocr import PaddleOcrEngine
from grader.ocr.base import PageInput
from grader.storage import PageStore

from .fixtures import answer_sheet_with_text

#: Every case here loads the OCR model, so the whole module is slow by nature.
pytestmark = pytest.mark.slow

paddle = PaddleOcrEngine()
requires_paddle = pytest.mark.skipif(
    not paddle.available(),
    reason="local OCR extra not installed (uv sync --extra ocr-local)",
)


@pytest.fixture(scope="module")
def rendered_page(tmp_path_factory) -> PageInput:
    """One rasterized page, ready to transcribe."""
    store = PageStore(root=tmp_path_factory.mktemp("pages"))
    data, _ = answer_sheet_with_text()
    source = render.inspect(data, "sheet.pdf", DocumentKind.ANSWER_SHEET)
    first = next(iter(render.render_pages(data, source, store)))
    return PageInput(
        index=0,
        width=first.page.width,
        height=first.page.height,
        png=first.png,
        document=data,
        filename="sheet.pdf",
    )


@pytest.fixture(scope="module")
def transcribed(rendered_page: PageInput):
    if not paddle.available():
        pytest.skip("local OCR extra not installed")
    return paddle.transcribe(rendered_page)


class TestAdapterContract:
    @requires_paddle
    def test_finds_text_on_the_page(self, transcribed) -> None:
        assert transcribed, "expected the recognizer to find text"

    @requires_paddle
    def test_every_box_obeys_the_coordinate_contract(self, transcribed) -> None:
        # The conversion from the model's pixel polygons into normalized space
        # happens in the adapter. A mistake there produces uniformly shifted or
        # scaled highlights, which look like a mapping bug.
        for line in transcribed:
            assert 0.0 <= line.box.x0 < line.box.x1 <= 1.0
            assert 0.0 <= line.box.y0 < line.box.y1 <= 1.0

    @requires_paddle
    def test_confidences_are_in_range(self, transcribed) -> None:
        for line in transcribed:
            assert 0.0 <= line.confidence <= 1.0

    @requires_paddle
    def test_no_empty_text_survives(self, transcribed) -> None:
        # An empty region carries no information but would occupy an index slot
        # and could be cited by a model as though it held an answer.
        for line in transcribed:
            assert line.text.strip()

    @requires_paddle
    def test_boxes_are_ordered_after_the_reading_order_sort(self, transcribed) -> None:
        # Detection order is arbitrary, so the index must sort. The invariant is
        # band-major ordering, not strictly ascending y: two regions sharing a
        # baseline ("Name: Suyash" and "Class: 6C") differ in y by a pixel or
        # two, and are correctly ordered left-to-right instead. Asserting
        # ascending y would demand the scrambling the banding exists to prevent.
        ordered = sort_reading_order(list(transcribed))
        heights = sorted(ln.box.y1 - ln.box.y0 for ln in ordered)
        band = heights[len(heights) // 2]

        for previous, current in zip(ordered, ordered[1:], strict=False):
            backward_jump = previous.box.y0 - current.box.y0
            assert backward_jump <= band, (
                f"reading order jumps back up the page by {backward_jump:.4f}, "
                f"more than one line height ({band:.4f})"
            )

    @requires_paddle
    def test_builds_a_valid_line_index(self, transcribed) -> None:
        index = build_index(
            DocumentKind.ANSWER_SHEET,
            [list(transcribed)],
            paddle.engine,
            trust_engine_order=False,
        )
        assert all(ln.line_id.startswith("as:") for ln in index.lines)
        # Geometry must resolve, which is what a highlight ultimately depends on.
        first, last = index.lines[0].line_id, index.lines[-1].line_id
        assert index.span_geometry(first, last)


class TestAvailability:
    def test_reports_availability_without_raising(self) -> None:
        # Called during engine selection, so it must never raise even when the
        # optional extra is absent.
        assert isinstance(PaddleOcrEngine().available(), bool)

    @requires_paddle
    def test_rejects_a_page_with_no_pixels(self, rendered_page: PageInput) -> None:
        from grader.ocr.base import EngineUnavailable

        pixel_free = PageInput(
            index=0,
            width=rendered_page.width,
            height=rendered_page.height,
            png=None,
            document=rendered_page.document,
        )
        with pytest.raises(EngineUnavailable, match="rendered page pixels"):
            paddle.transcribe(pixel_free)


@pytest.mark.skipif(
    not os.getenv("GRADER_SAMPLE_DIR"),
    reason="set GRADER_SAMPLE_DIR to a directory of real handwritten pages",
)
@requires_paddle
def test_detection_recall_on_real_handwriting() -> None:
    """Opt-in check against real handwritten scripts.

    Reports detection counts and confidence distribution rather than asserting a
    recall threshold, because the useful output here is a measurement to compare
    engines against, not a pass/fail. On the sample used during development,
    recall was around 90% — good enough for highlighting to work, and short of
    complete, which is why ink geometry exists as an independent source.
    """
    sample_dir = Path(os.environ["GRADER_SAMPLE_DIR"])
    images = sorted(
        p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        pytest.skip(f"no images in {sample_dir}")

    for image_path in images[:3]:
        data = image_path.read_bytes()
        source = render.inspect(data, image_path.name, DocumentKind.ANSWER_SHEET)
        store = PageStore(root=sample_dir / ".pagestore")
        first = next(iter(render.render_pages(data, source, store)))
        png = first.png or store.read(first.page.image_key)

        lines = paddle.transcribe(
            PageInput(
                index=0,
                width=first.page.width,
                height=first.page.height,
                png=png,
                filename=image_path.name,
            )
        )
        low = sum(1 for ln in lines if ln.confidence < 0.7)
        print(f"{image_path.name}: {len(lines)} regions, {low} below 0.7 confidence")
        assert lines, f"no text detected in {image_path.name}"
        for line in lines:
            assert 0.0 <= line.box.x0 < line.box.x1 <= 1.0
