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

#: Minimum share of a line's box that must fall inside an ink region before the
#: two are considered to describe the same marking.
_OVERLAP_FRACTION = 0.30


def _overlap_fraction(region: InkRegion, line: Line) -> float:
    """Share of the line's box lying inside the region's box."""
    if line.box.area <= 0:
        return 0.0
    return region.box.intersection_area(line.box) / line.box.area


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
        overlapping = [
            line for line in page_lines if _overlap_fraction(region, line) >= _OVERLAP_FRACTION
        ]

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
        overlapping = [
            line for line in page_lines if _overlap_fraction(region, line) >= _OVERLAP_FRACTION
        ]
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


def lines_excluded_from_grading(regions: list[InkRegion], lines: list[Line]) -> set[str]:
    """Line IDs sitting inside struck-through or bleed-through ink.

    The grading guard. Without it, a student who wrote a wrong answer, crossed it
    out and wrote the correct one below can be marked on the version they
    explicitly abandoned — and the score gives the teacher no hint that happened.
    """
    excluded: set[str] = set()
    suspect = [
        r
        for r in regions
        if r.kind in {InkRegionKind.STRUCK_THROUGH, InkRegionKind.BLEED_THROUGH}
    ]
    for line in lines:
        for region in suspect:
            if region.page == line.page and _overlap_fraction(region, line) >= _OVERLAP_FRACTION:
                excluded.add(line.line_id)
                break
    return excluded
