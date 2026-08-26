"""Tests for the coordinate contract.

These are the highest-value tests in the repo. A geometry bug does not crash —
it draws a rectangle in the wrong place, which looks like a mapping failure and
sends you debugging the wrong module. Every invariant that would otherwise fail
silently is asserted here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vedaai_contracts import BBox, PageBox
from vedaai_contracts.geometry import HGBENCH_SCALE


class TestBBoxInvariants:
    def test_accepts_a_well_formed_box(self) -> None:
        box = BBox(x0=0.1, y0=0.2, x1=0.5, y1=0.6)
        assert box.area == pytest.approx(0.4 * 0.4)

    @pytest.mark.parametrize(
        ("x0", "y0", "x1", "y1"),
        [
            (0.5, 0.1, 0.2, 0.6),  # x inverted
            (0.1, 0.6, 0.5, 0.2),  # y inverted
            (0.3, 0.1, 0.3, 0.6),  # zero width
            (0.1, 0.4, 0.5, 0.4),  # zero height
        ],
    )
    def test_rejects_inverted_or_degenerate_edges(self, x0, y0, x1, y1) -> None:
        # A collapsed box is nearly always an empty OCR token or a dropped ink
        # component upstream. Rejecting it here turns an invisible highlight
        # into a traceable error.
        with pytest.raises(ValidationError):
            BBox(x0=x0, y0=y0, x1=x1, y1=y1)

    @pytest.mark.parametrize(
        ("x0", "y0", "x1", "y1"),
        [
            (-0.1, 0.2, 0.5, 0.6),
            (0.1, 0.2, 1.5, 0.6),
            (0.1, -2.0, 0.5, 0.6),
        ],
    )
    def test_rejects_coordinates_outside_the_unit_square(self, x0, y0, x1, y1) -> None:
        # Catches the classic mistake of handing raw pixels to the constructor.
        with pytest.raises(ValidationError):
            BBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def test_is_immutable(self) -> None:
        box = BBox(x0=0.1, y0=0.1, x1=0.2, y1=0.2)
        with pytest.raises(ValidationError):
            box.x0 = 0.9  # type: ignore[misc]


class TestConstructionFromForeignSystems:
    def test_from_pixels_normalizes_against_page_size(self) -> None:
        box = BBox.from_pixels(100, 200, 300, 400, width=1000, height=2000)
        assert (box.x0, box.y0, box.x1, box.y1) == (0.1, 0.1, 0.3, 0.2)

    def test_from_pixels_rejects_a_nonpositive_page(self) -> None:
        with pytest.raises(ValueError, match="page size must be positive"):
            BBox.from_pixels(1, 2, 3, 4, width=0, height=100)

    def test_from_polygon_takes_the_axis_aligned_hull(self) -> None:
        # Engines return quadrilaterals for skewed text. We collapse to a hull
        # because highlights render as rectangles; deskew happens upstream.
        skewed = [(100.0, 210.0), (300.0, 200.0), (305.0, 400.0), (95.0, 410.0)]
        box = BBox.from_polygon(skewed, width=1000, height=1000)
        assert box.x0 == pytest.approx(0.095)
        assert box.y0 == pytest.approx(0.200)
        assert box.x1 == pytest.approx(0.305)
        assert box.y1 == pytest.approx(0.410)

    def test_from_polygon_rejects_an_empty_polygon(self) -> None:
        with pytest.raises(ValueError, match="at least one point"):
            BBox.from_polygon([], width=100, height=100)

    def test_pixel_round_trip_is_stable(self) -> None:
        original = BBox.from_pixels(150, 300, 450, 600, width=1200, height=1600)
        assert original.to_pixels(width=1200, height=1600) == (150, 300, 450, 600)


class TestHgBenchInterop:
    def test_round_trips_through_the_benchmark_grid(self) -> None:
        # We store [0,1] and convert only at the eval boundary, so HG-Bench's
        # 0-1000 convention never leaks into the pipeline.
        box = BBox(x0=0.125, y0=0.25, x1=0.5, y1=0.75)
        assert box.to_hgbench() == [125, 250, 500, 750]
        assert BBox.from_hgbench(box.to_hgbench()) == box

    def test_rejects_a_malformed_benchmark_box(self) -> None:
        with pytest.raises(ValueError, match="4 values"):
            BBox.from_hgbench([1, 2, 3])

    def test_uses_the_documented_scale(self) -> None:
        full = BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
        assert full.to_hgbench() == [0, 0, HGBENCH_SCALE, HGBENCH_SCALE]


class TestGeometryOperations:
    def test_union_covers_both_boxes(self) -> None:
        # This is how a highlight is built from the lines a model selected.
        a = BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2)
        b = BBox(x0=0.2, y0=0.5, x1=0.6, y1=0.7)
        assert a.union(b) == BBox(x0=0.1, y0=0.1, x1=0.6, y1=0.7)

    def test_union_all_requires_at_least_one_box(self) -> None:
        with pytest.raises(ValueError, match="zero boxes"):
            BBox.union_all([])

    def test_iou_is_one_for_identical_boxes(self) -> None:
        box = BBox(x0=0.2, y0=0.2, x1=0.4, y1=0.4)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_is_zero_for_disjoint_boxes(self) -> None:
        a = BBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)
        b = BBox(x0=0.5, y0=0.5, x1=0.7, y1=0.7)
        assert a.iou(b) == 0.0

    def test_iou_matches_a_hand_computed_overlap(self) -> None:
        a = BBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)  # area 0.04
        b = BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.3)  # area 0.04
        # overlap 0.1 x 0.1 = 0.01; union = 0.04 + 0.04 - 0.01 = 0.07
        assert a.iou(b) == pytest.approx(0.01 / 0.07)

    def test_contains_enforces_the_hierarchical_constraint(self) -> None:
        # HG-Bench requires every step box to sit inside its question box.
        parent = BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9)
        inside = BBox(x0=0.2, y0=0.2, x1=0.5, y1=0.5)
        straddling = BBox(x0=0.05, y0=0.2, x1=0.5, y1=0.5)
        assert parent.contains(inside)
        assert not parent.contains(straddling)

    def test_contains_is_reflexive_within_tolerance(self) -> None:
        box = BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9)
        assert box.contains(box)

    def test_expand_clamps_to_the_page(self) -> None:
        # Crop-rezoom pads a region before re-reading it; padding must not run
        # off the page or the crop will fail.
        box = BBox(x0=0.02, y0=0.02, x1=0.98, y1=0.98)
        grown = box.expand(0.05)
        assert (grown.x0, grown.y0, grown.x1, grown.y1) == (0.0, 0.0, 1.0, 1.0)

    def test_center_is_the_midpoint(self) -> None:
        box = BBox(x0=0.2, y0=0.4, x1=0.6, y1=0.8)
        assert box.center == (pytest.approx(0.4), pytest.approx(0.6))


class TestPageBox:
    def test_rejects_a_negative_page_index(self) -> None:
        # Pages are 0-indexed everywhere. A -1 usually means a 1-indexed value
        # was decremented twice.
        with pytest.raises(ValidationError):
            PageBox(page=-1, box=BBox(x0=0.1, y0=0.1, x1=0.2, y1=0.2))

    def test_carries_page_into_benchmark_output(self) -> None:
        pb = PageBox(page=2, box=BBox(x0=0.1, y0=0.2, x1=0.3, y1=0.4))
        assert pb.to_hgbench() == {"page": 2, "box": [100, 200, 300, 400]}
