"""Tests for page correction.

The asymmetry that shapes these tests: leaving a page slightly crooked costs a few
percent of recognition accuracy, and warping the wrong region costs the whole page.
So roughly half of what follows asserts that a correction *declines* — on a clean
render, on a blank page, on a photograph with no findable paper edge.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from grader import preprocess


def page_with_text(
    *, width: int = 800, height: int = 1100, rotation: float = 0.0, lines: int = 18
) -> np.ndarray:
    """A white page carrying rows of dark text-like marks."""
    image = np.full((height, width), 255, dtype=np.uint8)
    for row in range(lines):
        y = 80 + row * 55
        # Ragged runs rather than a solid rule, so the skew estimator is measuring
        # the same kind of thing it will see on a real page.
        x = 70
        while x < width - 120:
            run = 18 + (row * 7 + x) % 40
            cv2.rectangle(image, (x, y), (x + run, y + 18), 40, -1)
            x += run + 14
    if rotation:
        image = preprocess._rotate(image, rotation)
    return image


def keystoned(image: np.ndarray, *, shift: int = 90) -> np.ndarray:
    """The page as if photographed from an angle, inset on a dark desk."""
    height, width = image.shape[:2]
    frame = np.full((height + 200, width + 200, 3), 70, dtype=np.uint8)

    colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    target = np.array(
        [
            [100 + shift, 100],
            [width + 99, 100 + shift // 2],
            [width + 99 - shift, height + 99],
            [100, height + 99 - shift // 2],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(
        colour, matrix, (width + 200, height + 200), dst=frame, borderMode=cv2.BORDER_TRANSPARENT
    )


class TestSkew:
    @pytest.mark.parametrize("rotation", [-4.0, -2.0, 2.0, 5.0])
    def test_finds_the_rotation_of_a_tilted_page(self, rotation) -> None:
        measured = preprocess._skew_angle(page_with_text(rotation=rotation))
        # Recovering the angle to within half a degree is enough: below that the
        # resampling blur costs more than the remaining tilt.
        assert measured == pytest.approx(-rotation, abs=0.6)

    def test_leaves_a_straight_page_alone(self) -> None:
        result = preprocess.correct(page_with_text(), is_photograph=False)
        assert result.skew_degrees == 0.0
        assert result.dewarped is False

    def test_declines_on_a_blank_page(self) -> None:
        # There is no text to be crooked, so any angle would be fitting noise.
        blank = np.full((600, 400), 255, dtype=np.uint8)
        assert preprocess._skew_angle(blank) == 0.0

    def test_rotation_grows_the_canvas_rather_than_cropping(self) -> None:
        # Losing a corner of the page to rotation would lose whatever was written
        # there, and it would be invisible in the result.
        page = page_with_text(width=400, height=500)
        rotated = preprocess._rotate(page, 6.0)
        assert rotated.shape[0] > page.shape[0]
        assert rotated.shape[1] > page.shape[1]


class TestPerspective:
    def test_finds_the_page_in_a_photograph_taken_at_an_angle(self) -> None:
        photo = keystoned(page_with_text())
        quad = preprocess._page_quadrilateral(photo)
        assert quad is not None, "the page corners were not found"

        corrected = preprocess.correct(photo, is_photograph=True)
        assert corrected.dewarped is True
        # The corrected page should be close to the aspect ratio of the original,
        # which is the observable sign that the warp mapped the paper rather than
        # some other quadrilateral.
        height, width = corrected.image.shape[:2]
        assert 0.5 < (width / height) / (800 / 1100) < 2.0

    def test_declines_when_the_page_fills_the_frame(self) -> None:
        # With no visible border, a corner detector is fitting the frame rather
        # than the paper.
        assert preprocess._page_quadrilateral(page_with_text()) is None

    def test_a_rendered_pdf_page_is_never_dewarped(self) -> None:
        # It is already square and evenly lit, so any quadrilateral found would be
        # a false positive by construction.
        photo = keystoned(page_with_text())
        assert preprocess.correct(photo, is_photograph=False).dewarped is False

    def test_rejects_a_shape_that_is_not_roughly_rectangular(self) -> None:
        frame = np.full((900, 900, 3), 60, dtype=np.uint8)
        # A wedge covering plenty of the frame, but with nothing like right angles.
        cv2.fillConvexPoly(
            frame,
            np.array([[80, 80], [820, 200], [700, 860], [120, 300]], dtype=np.int32),
            (250, 250, 250),
        )
        quad = preprocess._page_quadrilateral(frame)
        if quad is not None:
            assert preprocess._sharpest_corner_cosine(quad) <= preprocess.MAX_CORNER_COSINE

    def test_corners_are_ordered_geometrically(self) -> None:
        # Not by the order the contour tracer produced them, or the warp mirrors
        # or rotates the page.
        quad = np.array([[300, 20], [10, 400], [320, 420], [20, 10]], dtype=np.float32)
        ordered = preprocess._order_corners(quad)
        assert list(ordered[0]) == [20, 10]
        assert list(ordered[2]) == [320, 420]


class TestIllumination:
    def test_removes_a_shadow_gradient(self) -> None:
        page = page_with_text()
        gradient = np.linspace(1.0, 0.45, page.shape[1], dtype=np.float32)
        shadowed = (page * gradient[None, :]).astype(np.uint8)

        # Before: the page's own brightness varies hugely across the width, so no
        # single threshold can mean the same thing on both sides.
        left = float(shadowed[:, :100].max())
        right = float(shadowed[:, -100:].max())
        assert left - right > 60

        flattened = preprocess._flatten_illumination(shadowed)
        assert float(flattened[:, :100].max()) - float(flattened[:, -100:].max()) < 40

    def test_keeps_grey_levels_rather_than_binarizing(self) -> None:
        # Recognition uses stroke thickness and antialiasing; a cleaner-looking
        # two-tone page costs accuracy. Blurred so the fixture has the soft edges
        # a real photograph has — the assertion is meaningless on a two-tone one.
        page = cv2.GaussianBlur(page_with_text(), (5, 5), 0)
        flattened = preprocess._flatten_illumination(page)
        # Well clear of two-tone. The bound is low because this fixture only has
        # as many levels as one Gaussian blur creates; a photograph has hundreds.
        assert len(np.unique(flattened)) > 8

    def test_one_dark_speck_does_not_swallow_the_stretch(self) -> None:
        # A finger at the edge of frame, a punch hole, the desk showing past the
        # paper — every real photograph has one, and under a min-max stretch it
        # sets the black point and the writing gains almost nothing.
        page = cv2.GaussianBlur(page_with_text(), (5, 5), 0)
        faint = (page.astype(np.float32) * 0.35 + 160).clip(0, 255).astype(np.uint8)

        without = preprocess._flatten_illumination(faint)
        speckled = faint.copy()
        speckled[5:15, 5:15] = 0
        with_speck = preprocess._flatten_illumination(speckled)

        def spread(img: np.ndarray) -> float:
            body = img[40:-40, 40:-40]
            return float(np.percentile(body, 98) - np.percentile(body, 2))

        # The page's own contrast must survive the intruding dark patch.
        assert spread(with_speck) > spread(without) * 0.8


class TestReporting:
    def test_describes_what_it_did(self) -> None:
        # A teacher comparing the screen with the paper on their desk should not
        # have to wonder why it looks different.
        result = preprocess.correct(keystoned(page_with_text(rotation=3.0)), is_photograph=True)
        described = result.describe()
        assert "camera angle" in described or "rotated" in described
        assert result.changed is True

    def test_says_nothing_when_it_did_nothing(self) -> None:
        result = preprocess.correct(page_with_text(), is_photograph=False)
        assert result.describe() == ""
        assert result.changed is False

    def test_an_empty_image_is_returned_untouched(self) -> None:
        empty = np.zeros((0, 0), dtype=np.uint8)
        assert preprocess.correct(empty).changed is False
