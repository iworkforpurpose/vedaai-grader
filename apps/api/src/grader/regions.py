"""Reconciling ink regions against transcribed lines.

``ink`` classifies from pixels alone and knows nothing about the recognizer.
This module supplies the other half: which ink was transcribed, which was not,
and which was written and then abandoned.

Two things come out of it.

**Orphan ink** — substantive marking with no transcribed line over it. Either a
diagram, which has no text by nature, or a line the recognizer missed, which
measurement puts at roughly one in ten. Both need to remain highlightable, and
neither can be found without comparing the two sources.

**Scribbled-out work** — the case a horizontal-strike test misses. Real students
often obliterate a line with loops rather than crossing it out once, which leaves
no long horizontal run. What it does leave is a region carrying far more ink than
ordinary writing, because the original text and the scribble are both still
there, paired with a recognizer that cannot read the result. Neither signal alone
is sufficient: dense-but-legible is just bold handwriting, and
faint-but-unreadable is bleed-through.
"""

from __future__ import annotations

from statistics import median

from vedaai_contracts import InkRegion, InkRegionKind, Line

#: Ink density above this multiple of the page's own normal-writing baseline,
#: combined with unreadable text, reads as writing that has been scribbled over.
#: Relative rather than absolute because pen width and handwriting size vary far
#: more between students than a fixed threshold could accommodate.
_SCRIBBLE_DENSITY_MULTIPLE = 1.8

#: Recognition confidence below which the text in a region is treated as
#: unreadable. Deliberately generous: the cost of missing a deletion is a wrong
#: mark, while the cost of a false positive is one region excluded from grading
#: that the teacher can still see highlighted.
_UNREADABLE_CONFIDENCE = 0.70

#: Minimum overlap before an ink region and a transcribed line are taken to
#: describe the same marking.
#:
#: Which side the fraction is measured against is not interchangeable, and
#: getting it wrong was a real bug. Ink components and OCR lines segment the page
#: differently: ink may split one line into fragments, or merge two into one. So
#: the two questions need opposite directions.
#:
#: *Was this region transcribed?* Region-centric — is most of the **region**
#: inside some line? Measuring against the line's area instead meant a small
#: fragment sitting wholly inside a long line scored 0.21 and was reported as
#: untranscribed, inflating the orphan count and making recognition look worse
#: than it is.
#:
#: *Should this line be excluded from grading?* Line-centric — is most of the
#: **line** inside a struck-through or bleed-through region? A tiny struck
#: fragment should not disqualify an entire legible line.
_OVERLAP_FRACTION = 0.30


def _share_of_line_inside_region(region: InkRegion, line: Line) -> float:
    """How much of the line falls within the region."""
    if line.box.area <= 0:
        return 0.0
    return region.box.intersection_area(line.box) / line.box.area


def _share_of_region_inside_line(region: InkRegion, line: Line) -> float:
    """How much of the region falls within the line."""
    if region.box.area <= 0:
        return 0.0
    return region.box.intersection_area(line.box) / region.box.area


def _describes_same_marking(region: InkRegion, line: Line) -> bool:
    """Whether a region and a line are the same ink, seen two ways.

    Either direction suffices. Ink and OCR segment a page differently — ink may
    fragment one line or merge two — so requiring containment in a particular
    direction would reject genuine matches on whichever side happened to be
    larger.
    """
    return (
        _share_of_region_inside_line(region, line) >= _OVERLAP_FRACTION
        or _share_of_line_inside_region(region, line) >= _OVERLAP_FRACTION
    )


def reconcile(regions: list[InkRegion], lines: list[Line]) -> list[InkRegion]:
    """Annotate ink regions with what the recognizer found in them.

    Returns new regions; ``InkRegion`` is immutable so that geometry handed to a
    renderer cannot be altered behind its back.
    """
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        by_page.setdefault(line.page, []).append(line)

    # Baseline for "normal" ink density on this page, taken from regions that
    # were transcribed confidently. Using the page's own writing as the
    # reference is what makes the scribble test robust across handwriting styles.
    baseline = _density_baseline(regions, by_page)

    out: list[InkRegion] = []
    for region in regions:
        page_lines = by_page.get(region.page, [])
        overlapping = [line for line in page_lines if _describes_same_marking(region, line)]

        covered = bool(overlapping)
        kind = region.kind

        if kind is InkRegionKind.WRITING and covered and baseline > 0:
            worst = min(line.confidence for line in overlapping)
            dense = region.ink_ratio >= baseline * _SCRIBBLE_DENSITY_MULTIPLE
            if dense and worst < _UNREADABLE_CONFIDENCE:
                kind = InkRegionKind.STRUCK_THROUGH

        out.append(region.model_copy(update={"covered_by_ocr": covered, "kind": kind}))

    return out


def _density_baseline(regions: list[InkRegion], by_page: dict[int, list[Line]]) -> float:
    """Median ink density of confidently-transcribed writing.

    Falls back to the median across all writing when nothing was read
    confidently — a page of uniformly poor handwriting should not make every
    region look like a deletion.
    """
    confident: list[float] = []
    all_writing: list[float] = []

    for region in regions:
        if region.kind is not InkRegionKind.WRITING or not region.is_substantive:
            continue
        all_writing.append(region.ink_ratio)

        page_lines = by_page.get(region.page, [])
        overlapping = [line for line in page_lines if _describes_same_marking(region, line)]
        if overlapping and min(line.confidence for line in overlapping) >= _UNREADABLE_CONFIDENCE:
            confident.append(region.ink_ratio)

    if confident:
        return median(confident)
    if all_writing:
        return median(all_writing)
    return 0.0


def orphan_ink(regions: list[InkRegion]) -> list[InkRegion]:
    """Substantive student marking that transcription did not account for.

    This is the list that makes incomplete recognition survivable. Each entry
    carries geometry for content the recognizer never reported, so a question
    answered by a diagram — or by a line that was simply missed — can still be
    highlighted.
    """
    return [r for r in regions if r.is_orphan_ink]


def struck_through(regions: list[InkRegion]) -> list[InkRegion]:
    """Regions the student crossed out.

    Excluded from grading input and from competing for a question. Retained in
    geometry, because the marking still occupies that space on the page.
    """
    return [r for r in regions if r.kind is InkRegionKind.STRUCK_THROUGH]


#: Share of the page above which a region may no longer disqualify lines.
#:
#: The second half of the fault documented on ``ink._STRIKE_MAX_STROKE_HEIGHTS``,
#: and kept as a separate guard on purpose. That one stops a page-sized region
#: being *called* struck-through; this one stops any single region — however it
#: came to be classified — from removing a whole page's writing from marking.
#:
#: The asymmetry is what justifies belt and braces here. Excluding a line the
#: student did not cross out marks their answer as if they had never written it,
#: which is a confident zero on work that was read at high confidence and is
#: exactly the error a teacher will be challenged on and cannot explain. Failing
#: to exclude an abandoned line lets a struck-out attempt reach the grader, which
#: is worse marking but visible: the rubric citations show a teacher which lines
#: were credited.
#:
#: A tenth of a page is far larger than any real crossing-out and far smaller than
#: the 0.65-of-a-page blobs that caused the damage.
_MAX_EXCLUDING_REGION_AREA = 0.10


def lines_excluded_from_grading(regions: list[InkRegion], lines: list[Line]) -> set[str]:
    """Line IDs sitting inside struck-through or bleed-through ink.

    The grading guard. Without it, a student who wrote a wrong answer, crossed it
    out and wrote the correct one below can be marked on the version they
    explicitly abandoned — and the score gives the teacher no hint that happened.

    Regions larger than a crossing-out could plausibly be are ignored here. See
    ``_MAX_EXCLUDING_REGION_AREA``: on a real script two of them removed 63 of 119
    lines, at a mean recognition confidence of 0.877.
    """
    excluded: set[str] = set()
    suspect = [
        r
        for r in regions
        if r.kind in {InkRegionKind.STRUCK_THROUGH, InkRegionKind.BLEED_THROUGH}
        and r.box.area <= _MAX_EXCLUDING_REGION_AREA
    ]
    for line in lines:
        for region in suspect:
            if (
                region.page == line.page
                and _share_of_line_inside_region(region, line) >= _OVERLAP_FRACTION
            ):
                excluded.add(line.line_id)
                break
    return excluded
