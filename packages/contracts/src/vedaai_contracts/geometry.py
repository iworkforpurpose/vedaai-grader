"""The coordinate contract.

Every box in this system obeys these rules, without exception:

  * normalized to ``[0, 1]`` floats — never pixels, never points
  * relative to the page **as rendered at RENDER_DPI**
  * origin top-left, ``y`` increasing downward
  * ``page`` is 0-indexed

Normalized coordinates are not an aesthetic choice. They let the browser lay
highlights out as percentages, so the overlay stays correct at any rendered
size without a resize listener, and they make Python and TypeScript agree
without either side knowing the other's render width.

These invariants are enforced by validators below rather than left to
convention, because coordinate-convention drift across the Python/TypeScript
boundary is the single most likely source of silent wrongness in this codebase:
a flipped axis or an off-by-one page still type-checks, still renders, and is
only visible as a highlight sitting in the wrong place.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: DPI at which pages are rasterized. Every normalized box is relative to a
#: page rendered at this density. Changing it invalidates cached geometry.
RENDER_DPI: int = 200

#: HG-Bench (arXiv 2606.25491) publishes boxes on a 0-1000 integer grid. We
#: store [0,1] internally and convert only at the eval boundary, so the
#: benchmark's convention never leaks into the pipeline.
HGBENCH_SCALE: int = 1000


class BBox(BaseModel):
    """An axis-aligned box in normalized page space.

    Immutable, so a box handed to a scorer or a renderer cannot be mutated
    underneath its owner.
    """

    model_config = ConfigDict(frozen=True)

    x0: float = Field(ge=0.0, le=1.0, description="Left edge, normalized.")
    y0: float = Field(ge=0.0, le=1.0, description="Top edge, normalized.")
    x1: float = Field(ge=0.0, le=1.0, description="Right edge, normalized.")
    y1: float = Field(ge=0.0, le=1.0, description="Bottom edge, normalized.")

    @model_validator(mode="after")
    def _edges_ordered(self) -> BBox:
        # A zero-area box is almost always a bug upstream (an empty OCR token,
        # a collapsed ink component) and is far cheaper to reject here than to
        # debug as an invisible highlight later.
        if self.x1 <= self.x0:
            raise ValueError(f"x1 must exceed x0, got x0={self.x0} x1={self.x1}")
        if self.y1 <= self.y0:
            raise ValueError(f"y1 must exceed y0, got y0={self.y0} y1={self.y1}")
        return self

    # -- construction from foreign coordinate systems ----------------------

    @classmethod
    def from_pixels(cls, x0: float, y0: float, x1: float, y1: float, *, width: int, height: int) -> BBox:
        """Build from pixel coordinates on a page of the given rendered size.

        This is the *only* sanctioned entry point for OCR engine output. Every
        adapter in ``grader.ocr`` funnels through here, which is what keeps
        engine-specific pixel conventions from reaching the rest of the system.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"page size must be positive, got {width}x{height}")
        return cls(x0=x0 / width, y0=y0 / height, x1=x1 / width, y1=y1 / height)

    @classmethod
    def from_polygon(cls, points: list[tuple[float, float]], *, width: int, height: int) -> BBox:
        """Build the tight axis-aligned hull of a polygon.

        Several engines (Azure, Google Cloud Vision) return quadrilaterals
        rather than rectangles, which matters for skewed scans. We deliberately
        collapse to an axis-aligned hull: highlights are drawn as rectangles,
        so keeping the rotation would imply a precision the UI cannot render.
        Deskewing happens upstream in preprocessing, where it belongs.
        """
        if not points:
            raise ValueError("polygon must have at least one point")
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls.from_pixels(min(xs), min(ys), max(xs), max(ys), width=width, height=height)

    @classmethod
    def from_hgbench(cls, box: list[int]) -> BBox:
        """Build from HG-Bench's ``[xmin, ymin, xmax, ymax]`` on a 0-1000 grid."""
        if len(box) != 4:
            raise ValueError(f"HG-Bench box must have 4 values, got {len(box)}")
        xmin, ymin, xmax, ymax = box
        s = HGBENCH_SCALE
        return cls(x0=xmin / s, y0=ymin / s, x1=xmax / s, y1=ymax / s)

    def to_hgbench(self) -> list[int]:
        """Emit ``[xmin, ymin, xmax, ymax]`` on HG-Bench's 0-1000 grid."""
        s = HGBENCH_SCALE
        return [round(self.x0 * s), round(self.y0 * s), round(self.x1 * s), round(self.y1 * s)]

    def to_pixels(self, *, width: int, height: int) -> tuple[int, int, int, int]:
        """Project back to pixels, for cropping a region out of a page bitmap."""
        return (
            round(self.x0 * width),
            round(self.y0 * height),
            round(self.x1 * width),
            round(self.y1 * height),
        )

    # -- geometry ----------------------------------------------------------

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def union(self, other: BBox) -> BBox:
        """Smallest box containing both.

        This is how a highlight is built: the union of the line boxes the model
        selected. Geometry comes from OCR, never from the model.
        """
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def intersection_area(self, other: BBox) -> float:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return w * h if w > 0 and h > 0 else 0.0

    def iou(self, other: BBox) -> float:
        """Intersection over union — the primary highlight-accuracy metric."""
        inter = self.intersection_area(other)
        denom = self.area + other.area - inter
        return inter / denom if denom > 0 else 0.0

    def contains(self, other: BBox, *, tolerance: float = 1e-6) -> bool:
        """Whether ``other`` sits inside this box.

        Used to enforce HG-Bench's hierarchical constraint that every step box
        lies within its parent question box.
        """
        return (
            self.x0 - tolerance <= other.x0
            and self.y0 - tolerance <= other.y0
            and self.x1 + tolerance >= other.x1
            and self.y1 + tolerance >= other.y1
        )

    def expand(self, pad: float) -> BBox:
        """Pad on all sides, clamped to the page.

        Highlights read better with a little breathing room around the ink, and
        crop-rezoom re-reads need context beyond the tight box to give the
        recognizer something to work with.
        """
        return BBox(
            x0=max(0.0, self.x0 - pad),
            y0=max(0.0, self.y0 - pad),
            x1=min(1.0, self.x1 + pad),
            y1=min(1.0, self.y1 + pad),
        )

    @staticmethod
    def union_all(boxes: list[BBox]) -> BBox:
        """Union of a non-empty list."""
        if not boxes:
            raise ValueError("cannot take the union of zero boxes")
        acc = boxes[0]
        for b in boxes[1:]:
            acc = acc.union(b)
        return acc


class PageBox(BaseModel):
    """A box on a specific page.

    Highlights are lists of these. Keeping the page index inside the geometry
    is what lets an answer span pages without any special-case handling: a
    multi-page highlight is just a list whose entries carry different pages.
    """

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=0, description="0-indexed page number.")
    box: BBox

    def to_hgbench(self) -> dict[str, object]:
        return {"page": self.page, "box": self.box.to_hgbench()}
