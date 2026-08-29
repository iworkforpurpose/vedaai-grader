"""Putting transcribed lines into the order a person would read them.

Phase 1 shipped a deliberately naive top-to-bottom sort with a test asserting
that it interleaves two columns. This replaces it, because "extract every question
in the correct printed order" is a graded requirement and a two-column paper
defeats sorting by height alone: the first line of the right column sits level
with the first line of the left, so a y-sort alternates between them and produces
a question list that is complete and useless.

Three ideas, in the order they apply.

**Full-width lines split the page into bands.** A title or a section header spans
both columns, so it is neither left nor right — it is a boundary. Content above
and below it is ordered independently. Without this, a mid-page section header
gets sorted into whichever column its left edge happens to fall in.

**Gutters are found by projection, not by clustering left edges.** Indented
sub-parts share a column while having quite different left edges, so clustering on
``x0`` splits a single column into several. What actually separates columns is a
vertical strip of page that no text crosses.

**Single-column pages take the simple path.** Most papers are single-column, and
inventing columns where there are none is a way to break the common case in
service of the rare one.
"""

from __future__ import annotations

from dataclasses import dataclass

from vedaai_contracts import Line

#: A line at least this wide, relative to the text area, spans the page and acts
#: as a band boundary rather than belonging to a column.
_FULL_WIDTH_SHARE = 0.62

#: A gutter must be at least this wide, relative to the text area, to be a real
#: column separator rather than the space between two words.
_MIN_GUTTER_SHARE = 0.035

#: Resolution of the horizontal projection.
_PROJECTION_BINS = 200

#: A column must hold at least this many lines to be believed. Two or three
#: strays at one side of a page are far more likely to be marginal annotations
#: than a column of their own.
_MIN_COLUMN_LINES = 3

#: Two boxes sharing at least this much of the shorter one's height are on the
#: same written row.
_SAME_ROW_OVERLAP = 0.5

#: A line no wider than this share of the page's text width is a token — a
#: question number in the margin, not a line of writing.
_TOKEN_MAX_WIDTH_SHARE = 0.10

#: Characters. A margin label is "7", "11.", "(iii)" — never a sentence.
_TOKEN_MAX_CHARS = 6


@dataclass
class OrderedPage:
    lines: list[Line]
    column_count: int
    confidence: float


def order_lines(lines: list[Line]) -> tuple[list[Line], float]:
    """Order every line across all pages, returning them with a confidence.

    Confidence is the share of lines that sit in an unambiguous column. It is
    surfaced rather than kept internal because a low value is a real warning: on a
    paper the model cannot resolve into columns, question order is the thing most
    likely to be wrong, and that is worth telling a teacher.
    """
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        by_page.setdefault(line.page, []).append(line)

    ordered: list[Line] = []
    confidences: list[float] = []
    for page in sorted(by_page):
        result = order_page(by_page[page])
        ordered.extend(result.lines)
        confidences.append(result.confidence)

    overall = sum(confidences) / len(confidences) if confidences else 1.0
    return ordered, overall


def order_page(lines: list[Line]) -> OrderedPage:
    """Order the lines of a single page."""
    if len(lines) <= 1:
        return OrderedPage(lines=list(lines), column_count=1, confidence=1.0)

    left = min(line.box.x0 for line in lines)
    right = max(line.box.x1 for line in lines)
    span = right - left
    if span <= 0:
        return OrderedPage(lines=_by_position(lines), column_count=1, confidence=1.0)

    # Band splitting exists to stop a full-width heading being swallowed by one
    # column of a two-column paper. On a single-column page it has nothing to do,
    # and doing it anyway is how this went wrong: on an answer sheet the writing
    # *is* most of the page width, so every ordinary line measured as "full width",
    # became a band of its own, and a question number in the margin could never
    # sort ahead of the line it labels. Measured on a real script: 0 of 12 numbers
    # bound to their own answer, so every answer began with its neighbour's words.
    #
    # Asking whether the page has columns at all, before cutting it into bands,
    # keeps the two-column behaviour and leaves the common case alone.
    if not _has_columns(lines, left=left, span=span):
        return OrderedPage(lines=_by_position(lines), column_count=1, confidence=1.0)

    bands = _split_into_bands(lines, left=left, span=span)

    ordered: list[Line] = []
    column_counts: list[int] = []
    for band in bands:
        columns = _detect_columns(band, left=left, span=span)
        column_counts.append(len(columns))
        for column in columns:
            ordered.extend(_by_position(column))

    column_count = max(column_counts) if column_counts else 1
    # A single column is the unambiguous case. Multiple columns are believed but
    # reported with less confidence, since that is where ordering goes wrong.
    confidence = 1.0 if column_count == 1 else 0.75
    return OrderedPage(lines=ordered, column_count=column_count, confidence=confidence)


def _is_token(line: Line, *, span: float) -> bool:
    """Whether a line is a margin token rather than a line of writing.

    A question number written beside an answer. Short in both senses — few
    characters and little width — because either alone is ambiguous: "OK" is a
    short word inside a sentence, and a narrow box can be one long word.
    """
    width_share = (line.box.x1 - line.box.x0) / span if span > 0 else 1.0
    return (
        width_share <= _TOKEN_MAX_WIDTH_SHARE
        and len(line.text.strip()) <= _TOKEN_MAX_CHARS
    )


def _shares_a_row(a: Line, b: Line) -> bool:
    top, bottom = max(a.box.y0, b.box.y0), min(a.box.y1, b.box.y1)
    if bottom <= top:
        return False
    shorter = min(a.box.y1 - a.box.y0, b.box.y1 - b.box.y0)
    return shorter > 0 and (bottom - top) / shorter >= _SAME_ROW_OVERLAP


def _has_columns(lines: list[Line], *, left: float, span: float) -> bool:
    """Whether this page really is laid out in columns.

    The gutter test alone says yes far too readily on an answer sheet: numbers
    down the left margin leave a genuine vertical strip of untouched page, wide
    enough for the projection to find, and calling that a column would order every
    number before every answer — worse than the fault being fixed.

    What separates the two is what is *in* the candidate column. A column of a
    question paper holds lines of text. A margin holds tokens, and each token sits
    on a row with writing beside it. So a candidate whose members are nearly all
    tokens with a row-mate elsewhere is a margin, not a column.
    """
    columns = _detect_columns(lines, left=left, span=span)
    if len(columns) < 2:
        return False

    real = 0
    for column in columns:
        others = [line for line in lines if line not in column]
        tokens_with_mates = sum(
            1
            for line in column
            if _is_token(line, span=span)
            and any(_shares_a_row(line, other) for other in others)
        )
        if tokens_with_mates < len(column) * 0.6:
            real += 1

    return real >= 2


def _by_position(lines: list[Line]) -> list[Line]:
    """Sort within one column: by row, then left to right inside each row.

    Rows are built by asking which boxes actually overlap vertically, not by
    rounding ``y0`` onto a grid. Rounding was the earlier approach and it is
    unreliable in exactly the case that matters: a question number and its line
    differ by a fraction of a line height, so whether they land in the same bucket
    depends on where that fraction falls relative to a rounding boundary. The
    first number on a page would bind and the second would not.

    Overlap has no boundary to fall the wrong side of. Two boxes either share
    vertical extent or they do not.
    """
    if not lines:
        return []

    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda ln: ln.box.y0):
        # Compared against the row's lowest member rather than its first, so a
        # tall row does not keep absorbing lines below it.
        placed = False
        for row in rows:
            if any(_shares_a_row(line, member) for member in row):
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])

    ordered: list[Line] = []
    for row in sorted(rows, key=lambda r: min(member.box.y0 for member in r)):
        ordered.extend(sorted(row, key=lambda ln: ln.box.x0))
    return ordered


def _split_into_bands(lines: list[Line], *, left: float, span: float) -> list[list[Line]]:
    """Cut the page at full-width lines.

    Each full-width line becomes a band of its own, so it keeps its position in
    the reading order instead of being absorbed into a column.
    """
    ordered = sorted(lines, key=lambda line: line.box.y0)
    bands: list[list[Line]] = []
    current: list[Line] = []

    for line in ordered:
        width_share = (line.box.x1 - line.box.x0) / span
        if width_share >= _FULL_WIDTH_SHARE:
            if current:
                bands.append(current)
                current = []
            bands.append([line])
        else:
            current.append(line)

    if current:
        bands.append(current)
    return bands


def _detect_columns(lines: list[Line], *, left: float, span: float) -> list[list[Line]]:
    """Split a band into columns by finding vertical strips no text crosses."""
    if len(lines) < _MIN_COLUMN_LINES * 2:
        # Too few lines to support more than one column. Splitting here would
        # invent structure from noise.
        return [lines]

    occupied = [False] * _PROJECTION_BINS
    for line in lines:
        start = int((line.box.x0 - left) / span * (_PROJECTION_BINS - 1))
        end = int((line.box.x1 - left) / span * (_PROJECTION_BINS - 1))
        for i in range(max(0, start), min(_PROJECTION_BINS, end + 1)):
            occupied[i] = True

    min_gutter_bins = max(1, int(_MIN_GUTTER_SHARE * _PROJECTION_BINS))
    boundaries: list[float] = []
    run_start: int | None = None

    for i, filled in enumerate(occupied):
        if not filled:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_gutter_bins:
                # Cut through the middle of the gutter.
                midpoint = (run_start + i) / 2 / (_PROJECTION_BINS - 1)
                boundaries.append(left + midpoint * span)
            run_start = None

    if not boundaries:
        return [lines]

    columns: list[list[Line]] = [[] for _ in range(len(boundaries) + 1)]
    for line in lines:
        centre = (line.box.x0 + line.box.x1) / 2
        index = sum(1 for boundary in boundaries if centre > boundary)
        columns[index].append(line)

    populated = [column for column in columns if len(column) >= _MIN_COLUMN_LINES]
    if len(populated) < 2:
        # The split produced one real column and some strays; the strays are
        # marginal annotations, not a column, and reordering the page around them
        # would be worse than leaving it alone.
        return [lines]

    # Any lines in rejected columns still belong somewhere. Attach each to the
    # nearest surviving column rather than dropping it — a lost line is a lost
    # question.
    rejected = [line for column in columns if len(column) < _MIN_COLUMN_LINES for line in column]
    for line in rejected:
        centre = (line.box.x0 + line.box.x1) / 2
        nearest = min(
            populated,
            key=lambda column: abs(
                centre - sum((c.box.x0 + c.box.x1) / 2 for c in column) / len(column)
            ),
        )
        nearest.append(line)

    return populated
