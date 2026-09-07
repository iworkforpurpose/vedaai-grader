"""Turning a run of transcription lines into the region a teacher sees.

Geometry is computed from line boxes and never asked of a model, which is the
invariant the whole product rests on. The shape follows how the answer was
written: lines sharing a row are joined across, and a gap wide enough to be a
margin ends the band rather than swallowing whitespace.
"""

from __future__ import annotations

from vedaai_contracts import (
    AnswerBlock,
    BBox,
    Highlight,
    PageBox,
)

from .tuning import _MERGE_GAP_SHARE, _MERGE_OVERLAP_SHARE, _ROW_GAP_MAX, _SAME_ROW_OVERLAP


def _highlight(blocks: list[AnswerBlock]) -> Highlight | None:
    """Where the writing is: one band per region of it.

    Two shapes were tried before this one and both were wrong in opposite
    directions.

    One box per page reads cleanly and covers far too much. On a multi-line answer
    60 to 74 per cent of the rectangle was blank paper, and on a page of
    handwritten code it painted 0.77 of the sheet to mark 0.28 of writing —
    including, on a page where a student answers one question at the top and
    another at the bottom, the answer in between.

    One box per line is tight and unreadable. Question 16 of a real script came out
    as ten separate rectangles down a page of ruled paper, each cut to the ragged
    end of its own line, and a teacher said so. Worse, each was drawn round the
    recognised text, so the first started after the "I" of "In" and clipped the
    letter it was meant to mark.

    So lines that sit under one another and overlap horizontally become one band,
    and writing elsewhere on the page gets a band of its own. On ruled prose that is
    a single rectangle. On the code page it stays several, because there the
    separation is real.
    """
    boxes: list[PageBox] = [pb for block in blocks for pb in block.geometry]
    if not boxes:
        return None

    per_page: dict[int, list[BBox]] = {}
    for pb in boxes:
        per_page.setdefault(pb.page, []).append(pb.box)

    merged: list[PageBox] = []
    for page, page_boxes in sorted(per_page.items()):
        # Rows first, then bands. A recognizer returns a margin number as its own
        # box beside the sentence it labels, and those two are one row of writing
        # however little horizontal extent they share — so they have to be joined
        # before anything asks whether one run continues another, which is a
        # question about lines and not about fragments of a line.
        for run in _runs(_rows(page_boxes)):
            merged.extend(PageBox(page=page, box=box) for box in _band_shape(run))

    derived = "ink_regions" if all(b.is_text_free for b in blocks) else "ocr_lines"
    return Highlight(boxes=merged, derived_from=derived)


def _band_shape(run: list[BBox]) -> list[BBox]:
    """One band of writing, drawn as the shape a text selection has.

    A run of lines was previously drawn as a single rectangle around all of them,
    and that rectangle is mostly paper. Ruled writing leaves a gap of about
    four-fifths of a line between one line and the next, and the last line of a
    paragraph ends wherever the sentence ended, so a box around five lines of
    prose covers roughly twice the area of the writing inside it. Measured
    against the lines a teacher can see, the single rectangle scores 0.535 and
    not one answer in the golden set reaches 0.75.

    The opposite shape — one rectangle per line — was tried and rejected for a
    reason that has nothing to do with area. A teacher reported ten stripes down
    a page of ruled paper, because consecutive line boxes have visible gaps
    between them and read as ten separate marks rather than as one answer.

    Both complaints are satisfied at once by the shape every document viewer
    already uses for selected text: one box per row, each extended vertically to
    meet its neighbour so the run renders as a single connected shape with no
    gaps, while each row keeps its own left and right edge so a short final line
    does not drag a rectangle of blank paper across the page.

    The two published highlight numbers move in opposite directions here, as they
    always do — this covers the writing far better and the enclosing region
    slightly less well. That is the trade the region metric exists to make
    visible, and it is made deliberately: a teacher looks at the writing.
    """
    rows = sorted(run, key=lambda b: b.y0)
    if len(rows) <= 1:
        return rows

    # Each row grows down to meet the next, and the boundary is the midpoint of
    # the gap so neither row claims more of it than the other. Rows that already
    # overlap keep the same rule, which simply moves their shared edge to the
    # middle of the overlap rather than letting them double-count it.
    # Horizontally, the same shape again: a selection runs to the right margin on
    # every row but the last, and starts at the left margin on every row but the
    # first. Only the two ends are ragged, because only the two ends are actually
    # ragged in the writing. Keeping every row's own edges was measured and reads
    # worse in both senses — it saws a notch out of the middle of a paragraph
    # wherever a recognizer returned a short box, and it gives up a fifth of the
    # enclosing region for a gain in writing coverage that the ends alone already
    # buy.
    left = min(row.x0 for row in rows)
    right = max(row.x1 for row in rows)

    shaped: list[BBox] = []
    for index, row in enumerate(rows):
        first, last = index == 0, index == len(rows) - 1
        top = row.y0 if first else (rows[index - 1].y1 + row.y0) / 2
        bottom = row.y1 if last else (row.y1 + rows[index + 1].y0) / 2
        shaped.append(
            BBox(
                x0=row.x0 if first else left,
                y0=min(top, row.y0),
                x1=row.x1 if last else right,
                y1=max(bottom, row.y1),
            )
        )
    return shaped


def _rows(boxes: list[BBox]) -> list[BBox]:
    """One box per row of writing on a page, top to bottom.

    Every open row is offered the box rather than only the most recent one, for
    the same reason ``_runs`` does it: two columns interleave when sorted by
    height, and comparing against the last row only would put the left column's
    second line on the right column's first.
    """
    rows: list[BBox] = []
    for box in sorted(boxes, key=lambda b: (b.y0, b.x0)):
        for index, row in enumerate(rows):
            if _same_row(row, box):
                rows[index] = BBox.union_all([row, box])
                break
        else:
            rows.append(box)
    return sorted(rows, key=lambda b: (b.y0, b.x0))


def _same_row(a: BBox, b: BBox) -> bool:
    """Whether two boxes are fragments of one line of writing.

    Vertical overlap says they sit at the same height; the horizontal gap says
    whether they are one line or two things that happen to be level.
    """
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap <= 0:
        return False
    shorter = min(a.y1 - a.y0, b.y1 - b.y0)
    if shorter <= 0 or overlap / shorter < _SAME_ROW_OVERLAP:
        return False
    gap = max(a.x0, b.x0) - min(a.x1, b.x1)
    return gap <= _ROW_GAP_MAX


def _runs(boxes: list[BBox]) -> list[list[BBox]]:
    """Group boxes on one page into bands of writing.

    Every open run is offered the line, not merely the most recent one. Two columns
    of writing interleave when sorted by height — left, right, left, right — so
    comparing against the last run only meant each line started a run of its own and
    a two-column page came out as one rectangle per line.
    """
    runs: list[list[BBox]] = []

    for box in sorted(boxes, key=lambda b: (b.y0, b.x0)):
        for run in runs:
            if _joins(box, run):
                run.append(box)
                break
        else:
            runs.append([box])

    return runs


def _joins(box: BBox, run: list[BBox]) -> bool:
    """Whether this line continues that band."""
    bottom = max(b.y1 for b in run)
    # Measured against the taller of the two lines, so a short line does not make
    # an ordinary gap look enormous.
    height = max(box.y1 - box.y0, max(b.y1 - b.y0 for b in run))
    if box.y0 - bottom > height * _MERGE_GAP_SHARE:
        return False
    return _shares_width(box, run)


def _shares_width(box: BBox, run: list[BBox]) -> bool:
    """Whether a line lines up with the run above it well enough to join it."""
    left, right = min(b.x0 for b in run), max(b.x1 for b in run)
    overlap = min(right, box.x1) - max(left, box.x0)
    if overlap <= 0:
        return False
    # Against the narrower of the two, so a short final line still joins the
    # paragraph it belongs to, while an indented line under a long one does not.
    narrower = min(right - left, box.x1 - box.x0)
    return narrower > 0 and overlap / narrower >= _MERGE_OVERLAP_SHARE
