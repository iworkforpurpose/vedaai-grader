"""Tests for ink extraction and region classification.

Synthetic pages are drawn here rather than using real scans, because these tests
need to know the ground truth: which region is faint, which is crossed out, which
is speckle. Real pages verify that the thresholds hold on actual handwriting, and
that check lives in the opt-in test at the bottom of this file.

Writing is drawn as separated strokes rather than solid bars. That matters: a
solid bar contains long horizontal runs and would satisfy the strike test
trivially, so a test built on bars would report success while the detector was
firing on every line of text.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from vedaai_contracts import BBox, DocumentKind, InkRegionKind, Line, OcrEngine

from grader import ink, regions, render
from grader.storage import PageStore

PAGE_W, PAGE_H = 900, 1200


def blank_page() -> np.ndarray:
    return np.full((PAGE_H, PAGE_W, 3), 255, dtype=np.uint8)


def draw_writing(
    page: np.ndarray,
    y: int,
    *,
    x0: int = 100,
    x1: int = 500,
    darkness: int = 30,
    stroke_gap: int = 9,
) -> None:
    """Draw a line of text-like strokes: short verticals with gaps between them.

    The gaps are the point. Real handwriting has no long uninterrupted horizontal
    run, so a strike detector must not find one here.
    """
    for x in range(x0, x1, stroke_gap):
        cv2.line(page, (x, y), (x + 2, y + 16), (darkness, darkness, darkness), 2)


def draw_strike(page: np.ndarray, y: int, *, x0: int = 100, x1: int = 500) -> None:
    """Draw a single horizontal line through a line of text — a tidy deletion."""
    cv2.line(page, (x0, y + 8), (x1, y + 8), (25, 25, 25), 3)


def draw_scribble(page: np.ndarray, y: int, *, x0: int = 100, x1: int = 500) -> None:
    """Obliterate a line with loops, as students actually do.

    Leaves no long horizontal run, so this is the case the strike test misses and
    ink density has to catch.
    """
    for x in range(x0, x1, 12):
        cv2.ellipse(page, (x, y + 8), (8, 10), 0, 0, 360, (25, 25, 25), 3)


def add_shadow(page: np.ndarray) -> None:
    """Darken one corner, as a hand or phone body does on a photographed page."""
    gradient = np.linspace(1.0, 0.55, PAGE_W // 2)
    for x, factor in enumerate(gradient):
        page[:, x] = (page[:, x] * factor).astype(np.uint8)


class TestIlluminationFlattening:
    def test_a_shadow_is_not_mistaken_for_ink(self) -> None:
        # A global threshold on the raw image classifies the shadow as ink. The
        # test script has a hand shadow across one corner, so this is the normal
        # case rather than a contrived one.
        page = blank_page()
        add_shadow(page)
        draw_writing(page, 600, x0=600, x1=850)

        found = ink.find_regions(page, page=0)
        substantive = [r for r in found if r.is_substantive]

        # The writing sits on the unshadowed right side. If flattening failed,
        # the entire shadowed left half becomes one enormous region.
        for region in substantive:
            assert region.box.area < 0.25, (
                f"region covering {region.box.area:.1%} of the page — the shadow was "
                "probably thresholded as ink"
            )

    def test_flattening_preserves_the_writing(self) -> None:
        page = blank_page()
        add_shadow(page)
        draw_writing(page, 300, x0=100, x1=400)  # inside the shadowed half

        masks = ink.build_masks(page)
        # Writing under a shadow must still register as dark after flattening.
        band = masks.strict[295:320, 100:400]
        assert band.any(), "writing inside the shadow was lost by flattening"


class TestRegionClassification:
    def test_dark_writing_is_writing(self) -> None:
        page = blank_page()
        draw_writing(page, 200)
        found = [r for r in ink.find_regions(page, page=0) if r.is_substantive]

        assert found, "expected to find the writing"
        assert all(r.kind is InkRegionKind.WRITING for r in found), [r.kind for r in found]

    def test_faint_marking_is_bleed_through(self) -> None:
        # Writing showing through from the reverse side. Density cannot separate
        # this from real ink — both are "marks on the page" — but darkness can,
        # and treating the back of the sheet as answers on the front would be a
        # substantive error, not a cosmetic one.
        page = blank_page()
        draw_writing(page, 200, darkness=205)

        found = [r for r in ink.find_regions(page, page=0) if r.pixel_count >= 40]
        assert found, "faint marking should still be detected, just classified apart"
        assert any(r.kind is InkRegionKind.BLEED_THROUGH for r in found), [
            (r.kind, round(r.mean_darkness, 2)) for r in found
        ]

    def test_dark_and_faint_writing_are_told_apart_on_one_page(self) -> None:
        page = blank_page()
        draw_writing(page, 200, darkness=30)  # pen on this side
        draw_writing(page, 500, darkness=205)  # showing through from the other

        found = [r for r in ink.find_regions(page, page=0) if r.pixel_count >= 40]
        kinds = {r.kind for r in found}
        assert InkRegionKind.WRITING in kinds
        assert InkRegionKind.BLEED_THROUGH in kinds

    def test_a_single_line_deletion_is_detected(self) -> None:
        page = blank_page()
        draw_writing(page, 200)
        draw_strike(page, 200)

        found = [r for r in ink.find_regions(page, page=0) if r.is_substantive]
        assert any(r.kind is InkRegionKind.STRUCK_THROUGH for r in found), [
            (r.kind, r.has_horizontal_strike) for r in found
        ]

    def test_plain_writing_is_not_reported_as_struck_through(self) -> None:
        # The false positive that matters. If ordinary text triggers the strike
        # test, every answer is excluded from grading and every student scores
        # zero — a failure far worse than missing a deletion.
        page = blank_page()
        for y in (150, 250, 350, 450):
            draw_writing(page, y)

        found = [r for r in ink.find_regions(page, page=0) if r.is_substantive]
        assert found
        assert not any(r.has_horizontal_strike for r in found), (
            "text-like strokes triggered the strike detector"
        )
        assert all(r.kind is InkRegionKind.WRITING for r in found)

    def test_speckle_is_noise(self) -> None:
        page = blank_page()
        cv2.circle(page, (700, 900), 1, (40, 40, 40), -1)

        found = ink.find_regions(page, page=0)
        assert all(
            r.kind is InkRegionKind.NOISE or not r.is_substantive for r in found
        ), "a single speck should not read as deliberate marking"

    def test_a_blank_page_yields_no_substantive_ink(self) -> None:
        # The basis for reporting a question unanswered, so it has to be solid.
        found = [r for r in ink.find_regions(blank_page(), page=0) if r.is_substantive]
        assert found == []


class TestGeometry:
    def test_boxes_obey_the_coordinate_contract(self) -> None:
        page = blank_page()
        draw_writing(page, 200)
        draw_writing(page, 400)
        for region in ink.find_regions(page, page=0):
            assert 0.0 <= region.box.x0 < region.box.x1 <= 1.0
            assert 0.0 <= region.box.y0 < region.box.y1 <= 1.0

    def test_the_box_lands_on_the_writing(self) -> None:
        page = blank_page()
        draw_writing(page, 400, x0=200, x1=600)

        found = [r for r in ink.find_regions(page, page=0) if r.is_substantive]
        assert len(found) == 1
        box = found[0].box

        # Expected in normalized terms, allowing for the merge dilation.
        assert 0.18 < box.x0 < 0.25
        assert 0.63 < box.x1 < 0.72
        assert 0.31 < box.y0 < 0.35
        assert 0.33 < box.y1 < 0.37

    def test_page_index_is_carried_through(self) -> None:
        page = blank_page()
        draw_writing(page, 200)
        for region in ink.find_regions(page, page=3):
            assert region.page == 3
            assert region.region_id.startswith("ink:3:")

    def test_separate_lines_stay_separate(self) -> None:
        # The vertical dilation must not bridge line gaps, or a whole answer
        # collapses into one region and per-line geometry is lost.
        page = blank_page()
        draw_writing(page, 200)
        draw_writing(page, 400)
        draw_writing(page, 600)

        found = [r for r in ink.find_regions(page, page=0) if r.is_substantive]
        assert len(found) == 3, f"expected 3 separate lines, got {len(found)}"


class TestReconciliation:
    @staticmethod
    def line(line_id: str, box: BBox, confidence: float) -> Line:
        return Line(
            line_id=line_id,
            kind=DocumentKind.ANSWER_SHEET,
            page=0,
            box=box,
            text="something",
            confidence=confidence,
            engine=OcrEngine.PADDLE_OCR_VL,
        )

    def test_ink_with_no_transcription_becomes_orphan_ink(self) -> None:
        # This is what makes ~90% detection recall survivable: a line the
        # recognizer missed still has geometry, so it can still be highlighted.
        page = blank_page()
        draw_writing(page, 200)
        draw_writing(page, 500)
        found = ink.find_regions(page, page=0)

        transcribed = [r for r in found if r.is_substantive][0]
        reconciled = regions.reconcile(found, [self.line("as:0001", transcribed.box, 0.95)])

        orphans = regions.orphan_ink(reconciled)
        assert len(orphans) == 1, [(r.region_id, r.kind, r.covered_by_ocr) for r in reconciled]
        assert not orphans[0].covered_by_ocr

    def test_transcribed_ink_is_not_an_orphan(self) -> None:
        page = blank_page()
        draw_writing(page, 200)
        found = ink.find_regions(page, page=0)
        substantive = [r for r in found if r.is_substantive]

        reconciled = regions.reconcile(
            found, [self.line(f"as:{i:04d}", r.box, 0.95) for i, r in enumerate(substantive)]
        )
        assert regions.orphan_ink(reconciled) == []

    def test_a_scribbled_out_line_is_caught_by_density(self) -> None:
        # The case a strike test cannot see. Students obliterate with loops, not
        # a single stroke, which leaves no long horizontal run — but it does leave
        # far more ink than ordinary writing, alongside unreadable text.
        page = blank_page()
        for y in (150, 250, 350):
            draw_writing(page, y)  # normal writing sets the density baseline
        draw_writing(page, 500)
        draw_scribble(page, 500)  # same line, obliterated

        found = ink.find_regions(page, page=0)
        substantive = [r for r in found if r.is_substantive]
        scribbled = max(substantive, key=lambda r: r.ink_ratio)

        lines = []
        for i, region in enumerate(substantive):
            # The recognizer reads normal writing confidently and the scribble not at all.
            confidence = 0.25 if region.region_id == scribbled.region_id else 0.95
            lines.append(self.line(f"as:{i:04d}", region.box, confidence))

        reconciled = regions.reconcile(found, lines)
        struck = regions.struck_through(reconciled)
        assert scribbled.region_id in {r.region_id for r in struck}, (
            f"densest region ({scribbled.ink_ratio:.3f}) was not flagged; "
            f"kinds were {[(r.region_id, r.kind) for r in reconciled if r.is_substantive]}"
        )

    def test_low_confidence_alone_does_not_mean_struck_through(self) -> None:
        # Uniformly poor handwriting is illegible, not deleted. Treating it as
        # deleted would silently drop a real answer from grading.
        page = blank_page()
        for y in (150, 250, 350, 450):
            draw_writing(page, y)
        found = ink.find_regions(page, page=0)
        substantive = [r for r in found if r.is_substantive]

        lines = [self.line(f"as:{i:04d}", r.box, 0.30) for i, r in enumerate(substantive)]
        reconciled = regions.reconcile(found, lines)
        assert regions.struck_through(reconciled) == []

    def test_struck_through_and_bleed_through_lines_are_excluded_from_grading(self) -> None:
        # The grading guard. Without it a student who crossed out a wrong answer
        # and rewrote it correctly can be marked on the abandoned version.
        page = blank_page()
        draw_writing(page, 200)
        draw_strike(page, 200)
        draw_writing(page, 500, darkness=205)
        draw_writing(page, 800)

        found = ink.find_regions(page, page=0)
        interesting = [r for r in found if r.pixel_count >= 40]
        lines = [self.line(f"as:{i:04d}", r.box, 0.9) for i, r in enumerate(interesting)]
        reconciled = regions.reconcile(found, lines)

        excluded = regions.lines_excluded_from_grading(reconciled, lines)
        kept = {ln.line_id for ln in lines} - excluded

        assert excluded, "expected the struck-through and bleed-through lines to be excluded"
        assert kept, "the clean line must survive"

    def test_bleed_through_is_excluded_from_page_ink(self) -> None:
        # Bleed-through appears on every double-sided script. Counting it as
        # unexplained ink would suppress every legitimate absence claim.
        page = blank_page()
        draw_writing(page, 500, darkness=205)
        found = ink.find_regions(page, page=0)

        assert any(r.kind is InkRegionKind.BLEED_THROUGH for r in found)
        assert ink.page_ink_ratio(found) == pytest.approx(0.0, abs=1e-6)

    def test_struck_through_still_counts_as_page_ink(self) -> None:
        # The student did write there, so its presence argues against the page
        # being blank even though it must not be graded.
        page = blank_page()
        draw_writing(page, 200)
        draw_strike(page, 200)
        found = ink.find_regions(page, page=0)

        assert any(r.kind is InkRegionKind.STRUCK_THROUGH for r in found)
        assert ink.page_ink_ratio(found) > 0.0


@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("GRADER_SAMPLE_DIR"),
    reason="set GRADER_SAMPLE_DIR to a directory of real handwritten pages",
)
def test_ink_on_real_handwriting(tmp_path) -> None:
    """Reports the classification breakdown on real scripts.

    A measurement rather than an assertion on exact counts, which would be
    brittle. What is asserted is what must hold on any real page: substantive ink
    exists, geometry is valid, and the page is not swallowed by one giant region
    from a shadow.
    """
    sample_dir = Path(os.environ["GRADER_SAMPLE_DIR"])
    images = sorted(
        p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        pytest.skip(f"no images in {sample_dir}")

    store = PageStore(root=tmp_path / "pages")
    for image_path in images[:3]:
        data = image_path.read_bytes()
        source = render.inspect(data, image_path.name, DocumentKind.ANSWER_SHEET)
        first = next(iter(render.render_pages(data, source, store)))
        png = first.png or store.read(first.page.image_key)

        import io

        from PIL import Image

        array = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        found = ink.find_regions(array, page=0)

        counts: dict[str, int] = {}
        for region in found:
            counts[region.kind.value] = counts.get(region.kind.value, 0) + 1
        substantive = [r for r in found if r.is_substantive]
        print(f"{image_path.name}: {len(found)} regions {counts}, {len(substantive)} substantive")

        assert substantive, f"no substantive ink found in {image_path.name}"
        for region in found:
            assert 0.0 <= region.box.x0 < region.box.x1 <= 1.0
            assert 0.0 <= region.box.y0 < region.box.y1 <= 1.0
        biggest = max(r.box.area for r in substantive)
        assert biggest < 0.6, (
            f"one region covers {biggest:.0%} of the page — illumination flattening "
            "probably failed on this scan"
        )


class TestOverlapDirection:
    """Regression tests for which side an overlap fraction is measured against.

    Ink components and OCR lines segment a page differently, so the two questions
    need opposite directions. Using one for both meant a small ink fragment
    sitting wholly inside a long transcribed line was reported as untranscribed,
    inflating the orphan count and making recognition look worse than it was.
    """

    @staticmethod
    def line(line_id: str, box: BBox, confidence: float = 0.95) -> Line:
        return Line(
            line_id=line_id,
            kind=DocumentKind.ANSWER_SHEET,
            page=0,
            box=box,
            text="something",
            confidence=confidence,
            engine=OcrEngine.PADDLE_OCR_VL,
        )

    def test_a_fragment_inside_a_long_line_counts_as_transcribed(self) -> None:
        # The exact geometry that exposed the bug: a narrow ink region wholly
        # within a wide line. Measured against the line's area it scores ~0.21
        # and fails; measured against the region's, it is plainly covered.
        page = blank_page()
        draw_writing(page, 200, x0=100, x1=700)
        found = ink.find_regions(page, page=0)
        substantive = [r for r in found if r.is_substantive]
        assert substantive

        wide_line = self.line("as:0001", BBox(x0=0.08, y0=0.16, x1=0.80, y1=0.18))
        reconciled = regions.reconcile(found, [wide_line])

        covered = [r for r in reconciled if r.is_substantive and r.covered_by_ocr]
        assert covered, "a region inside the line should be marked transcribed"

    def test_a_tiny_struck_fragment_does_not_disqualify_a_whole_line(self) -> None:
        # The other direction. Exclusion is line-centric, so a speck of
        # struck-through ink must not remove an entire legible answer from
        # grading — that would silently zero a question the student answered.
        page = blank_page()
        draw_writing(page, 200, x0=100, x1=700)
        found = ink.find_regions(page, page=0)

        tiny_struck = [
            r.model_copy(update={"kind": InkRegionKind.STRUCK_THROUGH})
            for r in found
            if r.is_substantive
        ][:1]
        # A long line that the small region barely overlaps.
        long_line = self.line("as:0001", BBox(x0=0.02, y0=0.10, x1=0.98, y1=0.90))

        excluded = regions.lines_excluded_from_grading(tiny_struck, [long_line])
        assert excluded == set(), "a small struck region should not exclude a whole line"
