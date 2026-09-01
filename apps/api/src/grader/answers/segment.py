"""Dividing an answer sheet into candidate answer blocks.

Blocks are the units the alignment works on, and they come from layout rather
than from a model — so a segmentation mistake is inspectable and fixable in code
rather than being an opaque property of a prompt.

The rule that matters most here is negative. **A gap in the transcribed text is
not a block boundary.** Detection recall on real handwriting is about 90%, so a
missed line leaves a text-shaped hole in the middle of an answer, and splitting
there breaks one answer into two — which then presents to the mapper as a
phantom orphan and to the teacher as a wrong highlight.

The ink mask settles it. Ink is found by thresholding, independently of
recognition, so ink sitting in a gap means writing is there and the gap is an OCR
miss rather than a margin. That check is the whole reason the two geometry sources
exist as peers.

Blocks may also contain no text at all. A hand-drawn diagram produces ink and no
lines, and it is still an answer that has to be highlightable.
"""

from __future__ import annotations

from vedaai_contracts import (
    AnswerBlock,
    InkRegion,
    InkRegionKind,
    Line,
    PageBox,
    Question,
)

from ..questions.numbering import detect_section_prefixes, parse_label
from . import furniture

#: A vertical gap this many times the normal line spacing suggests a new block.
#: Generous, because over-splitting is the more damaging error: two blocks where
#: there should be one produces a spurious orphan answer, whereas one block where
#: there should be two is repairable by the sub-part split that runs later.
_GAP_MULTIPLE = 2.2

#: Ink covering at least this share of a gap means writing is there and the gap
#: is a recognition failure, not a boundary.
_GAP_INK_SHARE = 0.25

#: Phrases a student writes when an answer continues elsewhere.
_CONTINUATION_MARKERS = (
    "cont.",
    "contd",
    "continued",
    "cont on",
    "cont. on",
    "see next page",
    "on next page",
    "overleaf",
    "p.t.o",
    "pto",
)


def segment_blocks(
    lines: list[Line],
    ink_regions: list[InkRegion],
    questions: list[Question] | None = None,
) -> list[AnswerBlock]:
    """Group answer-sheet lines into blocks, using ink to avoid false splits.

    ``questions`` is what the paper actually asks, and it is what makes a written
    label trustworthy enough to cut an answer in two. Passing None keeps the old
    behaviour of believing any label-shaped line, which is only safe when the
    paper is unknown.
    """
    # Script details — name, class, roll number, "Set 3" — are removed before
    # anything is grouped. They are not answers, and left in they become candidate
    # blocks: on the golden set the line "Name: Test Student  Class: 6C" was
    # assigned to a question the student had left blank.
    usable, _details = furniture.strip([line for line in lines if line.text.strip()])
    writing_ink = [
        region
        for region in ink_regions
        if region.kind.counts_as_page_ink and region.is_substantive
    ]

    # What the paper asks, as the segmenter needs it: the set of token paths that
    # name a real question, and the prefix styles it numbers them with. Both come
    # from the paper rather than from a grammar written in advance.
    known_paths: set[tuple[str, ...]] | None = None
    prefixes: frozenset[str] = frozenset()
    if questions:
        known_paths = {tuple(question.path) for question in questions}
        prefixes = detect_section_prefixes([q.label_raw for q in questions])

    blocks: list[AnswerBlock] = []
    current: list[Line] = []
    spacing = _normal_spacing(usable)
    # Where the writing starts, so a label can be told from an enumeration.
    body_left = _body_left_edge(usable)
    margin_labels = _labels_in_the_margin(usable, body_left)

    for index, line in enumerate(usable):
        starts_new = False

        if not current:
            starts_new = False
        elif _label_boundary(line, known_paths, prefixes, body_left, margin_labels):
            # An explicit label is the strongest boundary a student ever gives.
            starts_new = True
        else:
            previous = usable[index - 1]
            if _is_gap(previous, line, spacing) and not _ink_bridges(
                previous, line, writing_ink
            ):
                starts_new = True

        if starts_new:
            blocks.append(_build(len(blocks), current))
            current = []

        current.append(line)

    if current:
        blocks.append(_build(len(blocks), current))

    blocks = _attach_ink(blocks, writing_ink)
    return blocks


#: Percentile of observed gaps taken as normal line spacing.
#:
#: The median is the obvious choice and is wrong here. On a sheet where most
#: answers are a couple of lines long, close to half of all gaps are *between*
#: answers, so the median sits between intra-answer and inter-answer spacing and
#: the boundary threshold never fires. On the unlabelled golden case that
#: collapsed an entire sheet into a single block.
#:
#: A low percentile describes within-answer spacing, which is what "normal" has
#: to mean for a gap to stand out against it.
_SPACING_PERCENTILE = 0.30


def _normal_spacing(lines: list[Line]) -> float:
    """Typical vertical distance between consecutive lines *within* an answer.

    Measured from the page itself rather than assumed, because handwriting size
    varies far more between students than any fixed threshold could accommodate.
    """
    gaps: list[float] = []
    for previous, current in zip(lines, lines[1:], strict=False):
        if previous.page == current.page:
            gap = current.box.y0 - previous.box.y1
            if gap >= 0:
                gaps.append(gap)
    if not gaps:
        return 0.02

    gaps.sort()
    index = min(len(gaps) - 1, int(len(gaps) * _SPACING_PERCENTILE))
    value = gaps[index]
    # Guard against a degenerate value on tightly-packed writing, which would
    # make every gap look enormous.
    #
    # Note the inherent limit: with only one or two gaps on the page, whichever
    # gap exists *is* the baseline, so no gap can stand out against it and the
    # sheet stays a single block. That is not a fixable threshold — an outlier
    # cannot be identified from one sample — and it is the right failure
    # direction anyway, since merging is repairable downstream and splitting is
    # not.
    return max(value, 0.004)


def _is_gap(previous: Line, current: Line, spacing: float) -> bool:
    if previous.page != current.page:
        # A page break is a boundary unless the student said otherwise.
        #
        # This was the opposite way round, on the reasoning that answers routinely
        # continue over a page and the decision could be deferred to the aligner.
        # It cannot be: the aligner assigns whole blocks and has no means of
        # dividing one, so a page break that never splits fuses two answers
        # permanently. On a real two-page script that is exactly what happened —
        # both programs became a single block, and the second question read as
        # unanswered while the first claimed writing from a page it had nothing to
        # do with.
        #
        # The asymmetry runs the other way here than it does within a page. A
        # wrongly split page-spanning answer is repairable, because the aligner's
        # ``continue`` move can rejoin consecutive blocks under one question; a
        # wrongly merged pair of answers is not repairable by anything. So the
        # default is to split, and continuation has to be evidenced rather than
        # assumed.
        #
        # Geometry deliberately plays no part. "Ran out of room mid-answer" and
        # "filled the page, then started the next answer" both put the last line
        # at the bottom and the next at the top, so position cannot tell them
        # apart and using it would just restore the old assumption with extra
        # steps.
        return not _mentions_continuation(previous.text)
    return (current.box.y0 - previous.box.y1) > spacing * _GAP_MULTIPLE


def _ink_bridges(previous: Line, current: Line, ink: list[InkRegion]) -> bool:
    """Whether substantive ink sits in the gap between two lines.

    This is what stops an OCR miss from being read as a block boundary. The
    horizontal extent is deliberately ignored: a missed line may sit anywhere
    across the writing area, and requiring it to align with its neighbours would
    defeat the check on indented or centred text.
    """
    top = previous.box.y1
    bottom = current.box.y0
    height = bottom - top
    if height <= 0:
        return False

    covered = 0.0
    for region in ink:
        if region.page != previous.page:
            continue
        overlap = min(bottom, region.box.y1) - max(top, region.box.y0)
        if overlap > 0:
            covered += overlap

    return (covered / height) >= _GAP_INK_SHARE


#: How far left of the body text a label must sit to count as a margin label,
#: as a share of page width. About a centimetre on A4.
#:
#: This exists because a student's own enumeration of the parts of one answer is
#: indistinguishable from a question label by *text*. A physics script answered
#: "State three properties of magnetic field lines" as:
#:
#:     1. Magnetic field lines go from north pole to south pole outside
#:     the magnet.
#:     2. The lines are closer together where the field is stronger.
#:     3. The lines cross each other at the poles.
#:
#: Every one of "1.", "2." and "3." parses as a label, and the paper has
#: questions 1, 2 and 3, so all three passed the known-paths check that was
#: supposed to make labels safe. The answer was cut into three blocks, each was
#: *confirmed* as an anchor and pinned — the highest authority the aligner has —
#: and the result was that question 1 held the tail of its own working plus the
#: first property, question 2 held only the second property, and question 3 held
#: the third property plus the whole of the next answer.
#:
#: What separates them is where they sit. A question number is written in the
#: margin, left of the writing; an enumeration sits at the body indent, because
#: it *is* the body. Measured on that script: the margin labels were at x0≈0.035
#: and the pointers at x0≈0.099, against a body edge of 0.100.
_MARGIN_LEFT_OF_BODY = 0.030

#: Characters after the label below which the line is the label and nothing else.
#: A bare "Q4." on its own line is a question label wherever it sits.
_BARE_LABEL_MAX_REMAINDER = 3


#: A margin token is short in both senses — few characters and little width.
#: The same reasoning ``reading_order._is_token`` uses.
_MARGIN_TOKEN_MAX_CHARS = 6


def _labels_in_the_margin(lines: list[Line], body_left: float) -> bool:
    """Whether this script writes its question numbers in the margin.

    Asked of the whole script rather than of one line, because it is a fact about
    the student's habit and a single line cannot reveal it.

    This is what makes the enumeration rule safe. Position alone is too blunt: a
    student who writes "Q1." and "Q2." inline with no margin at all is labelling,
    and refusing to split there fuses two answers — the one damage segmentation
    cannot undo. But if the script *does* carry margin numbers, then a number at
    the body indent is the student's own enumeration inside an answer, because
    their question numbers are demonstrably somewhere else.

    Deliberately not ``parse_label``. A margin label is often a bare digit, and a
    bare digit does not parse as a label at all — it has no corroborating bracket
    or ``Q`` — which is exactly why the physics script's own "1", "3" and "5" were
    invisible to every label rule in the pipeline.
    """
    return any(
        len(line.text.strip()) <= _MARGIN_TOKEN_MAX_CHARS
        and line.box.x0 <= body_left - _MARGIN_LEFT_OF_BODY
        for line in lines
    )


def _body_left_edge(lines: list[Line]) -> float:
    """Where the answer text starts, measured from the script itself.

    The median over lines long enough to be prose rather than a label or a
    fragment, so a margin number cannot drag the estimate left and make itself
    look like body text.
    """
    edges = sorted(line.box.x0 for line in lines if len(line.text.strip()) > 12)
    if not edges:
        return 0.0
    return edges[len(edges) // 2]


def _label_boundary(
    line: Line,
    known_paths: set[tuple[str, ...]] | None,
    prefixes: frozenset[str],
    body_left: float = 0.0,
    margin_labels: bool = False,
) -> bool:
    """Whether this line opens a new answer by naming a question.

    A label is the strongest boundary a student ever gives, and it is the least
    reversible decision this module makes: the aligner assigns whole blocks and
    cannot divide one, so a boundary invented here can never be undone.

    Which is why the label has to name a question the paper contains. Handwritten
    working is full of text a label grammar accepts. Measured on one mathematics
    script, 8 lines parsed as labels and 5 were false: ``5(n) + 3(P) = 190`` as
    question 5 part n, ``5(26) + 3P = 190`` as question 5 part 26, and ``(i)`` and
    ``(ii)`` -- the student's own numbering of two halves of a proof -- both read
    by the recognizer as ``(9)``. Each one cut an answer in two, and the fragments
    then competed for questions separately and landed on questions the student had
    never attempted.

    This is the rule the two downstream consumers of labels already apply:
    ``anchors._resolve`` refuses a label naming no question, and ``_label_hints``
    requires the claimed qid to exist before a label may even carry a weighted
    hint. Segmentation was the one stage trusting a label unchecked while making
    the most permanent decision of the three.

    The cost, and it is real: extraction recall now gates segmentation. A question
    missed on the paper means a student's correct label for it no longer splits
    their answer. Accepted because the failure being removed is both worse and
    commoner, and because merging is what the aligner's ``continue`` move exists
    to repair while splitting is repairable by nothing.
    """
    parsed = parse_label(line.text, prefixes=prefixes)
    if parsed is None:
        return False
    if known_paths is None:
        return True
    if parsed.tokens not in known_paths:
        return False

    # Named a real question, and now: is it in the margin, or is it the student's
    # own numbering inside their answer? See ``_MARGIN_LEFT_OF_BODY``.
    #
    # A label alone on its line is exempt: a student who writes "Q4." and then
    # starts the answer underneath has labelled it, wherever the number sits.
    if len(parsed.remainder.strip()) <= _BARE_LABEL_MAX_REMAINDER:
        return True
    if not margin_labels:
        # This script has no margin numbers, so an inline label is how it labels.
        # Refusing here would fuse two answers, and that is the one damage
        # segmentation cannot undo.
        return True
    return line.box.x0 <= body_left - _MARGIN_LEFT_OF_BODY


def _build(index: int, lines: list[Line]) -> AnswerBlock:
    text = " ".join(line.text.strip() for line in lines if line.text.strip())
    pages = sorted({line.page for line in lines})
    return AnswerBlock(
        block_id=f"blk:{index:03d}",
        line_ids=[line.line_id for line in lines],
        text=text,
        # Per line, not one box per page. Collapsing here threw the shape of the
        # answer away before anything downstream could use it, and the highlight
        # inherited a rectangle that was mostly blank paper. What to merge is a
        # question about how the answer looks on the page, so it belongs where the
        # highlight is drawn, not here.
        geometry=[PageBox(page=ln.page, box=ln.box) for ln in lines],
        pages_spanned=pages,
        has_continuation_marker=_mentions_continuation(text),
    )


def _mentions_continuation(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _CONTINUATION_MARKERS)


def _attach_ink(blocks: list[AnswerBlock], ink: list[InkRegion]) -> list[AnswerBlock]:
    """Assign ink regions to blocks, and promote unclaimed ink to blocks of its own.

    Ink that overlaps a block belongs to it. Ink that overlaps nothing is content
    the recognizer never reported — a diagram, or a line it missed — and becomes a
    text-free block so that it remains highlightable. Discarding it would mean a
    question answered by a drawing has no answer at all.
    """
    claimed: dict[str, list[str]] = {block.block_id: [] for block in blocks}
    unclaimed: list[InkRegion] = []

    for region in ink:
        owner = _best_block(region, blocks)
        if owner is None:
            unclaimed.append(region)
        else:
            claimed[owner].append(region.region_id)

    out = [
        block.model_copy(update={"ink_region_ids": claimed[block.block_id]})
        for block in blocks
    ]

    for offset, group in enumerate(_group_adjacent(unclaimed)):
        out.append(
            AnswerBlock(
                block_id=f"blk:ink{offset:03d}",
                line_ids=[],
                ink_region_ids=[region.region_id for region in group],
                text="",
                geometry=[PageBox(page=region.page, box=region.box) for region in group],
                pages_spanned=sorted({region.page for region in group}),
            )
        )

    return out


def _best_block(region: InkRegion, blocks: list[AnswerBlock]) -> str | None:
    """The block a region sits in, by area of overlap.

    Summed across the block's boxes, not maximised over them. A block carries a
    box per line rather than one per page — so that a highlight marks the writing
    instead of the rectangle around it — and a connected component of handwriting
    spans the whole answer. Against tight per-line boxes no single one holds much
    of such a region: on ruled spacing the largest share is 0.26 at three lines
    and 0.09 at eight, while the region genuinely sits about 0.75 inside the
    block. Maximising therefore refused the ink of every answer longer than two
    lines and promoted it to a text-free block, which alignment then declines for
    good reasons of its own. The writing was read, transcribed and mapped, and
    still counted as ink belonging to nothing: unassigned ink reached 0.97 on the
    mathematics paper and 1.00 on the reassignment case, and past the 0.18
    threshold that downgrades every absence claim, neither could report an
    unanswered question at all.

    Lines within a block do not overlap one another, so the sum is the area of the
    region actually covered rather than a figure inflated by double counting.
    """
    best_id: str | None = None
    best_overlap = 0.0
    for block in blocks:
        overlap = sum(
            pb.box.intersection_area(region.box)
            for pb in block.geometry
            if pb.page == region.page
        )
        if overlap > best_overlap:
            best_overlap, best_id = overlap, block.block_id
    if best_id is None or region.box.area <= 0:
        return None
    # Require a real share of the region to be inside, so a block does not claim
    # a diagram that merely brushes its edge.
    return best_id if (best_overlap / region.box.area) >= 0.30 else None


def _group_adjacent(regions: list[InkRegion]) -> list[list[InkRegion]]:
    """Cluster unclaimed ink into blocks by vertical adjacency.

    A diagram decomposes into many components — axes, labels, a curve — and each
    on its own is not an answer. Grouping what sits together makes one
    highlightable region out of one drawing.
    """
    if not regions:
        return []

    ordered = sorted(regions, key=lambda r: (r.page, r.box.y0))
    heights = sorted(r.box.y1 - r.box.y0 for r in ordered)
    tolerance = max(heights[len(heights) // 2] * 2.0, 0.02)

    groups: list[list[InkRegion]] = [[ordered[0]]]
    for region in ordered[1:]:
        last = groups[-1][-1]
        same_page = region.page == last.page
        close = (region.box.y0 - last.box.y1) <= tolerance
        if same_page and close:
            groups[-1].append(region)
        else:
            groups.append([region])
    return groups


def substantive_writing_ink(regions: list[InkRegion]) -> list[InkRegion]:
    """Ink that represents the student's own writing on this page."""
    return [
        region
        for region in regions
        if region.kind is InkRegionKind.WRITING and region.is_substantive
    ]
