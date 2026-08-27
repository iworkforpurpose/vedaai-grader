"""Tests for answer segmentation and anchor confirmation.

Two failure modes drive most of what is asserted here.

Splitting one answer into two produces a phantom orphan and a truncated
highlight, and at ~90% detection recall the trigger is common: a missed line
leaves a text-shaped hole mid-answer. Several tests below check that ink in the
hole prevents the split.

Believing a wrong label maps an answer to the wrong question *while reporting
high confidence*. So anchors are tested both for confirming correct labels and
for declining to condemn labels that are merely unusual — a correct answer that
reuses none of its question's vocabulary must not be disputed.
"""

from __future__ import annotations

import pytest
from vedaai_contracts import (
    AnchorStatus,
    BBox,
    DocumentKind,
    InkRegion,
    InkRegionKind,
    Line,
    OcrEngine,
    Question,
)

from grader.answers import anchors
from grader.answers.segment import segment_blocks
from grader.answers.similarity import LexicalOverlap


def line(index: int, text: str, *, y0: float, page: int = 0, x0: float = 0.10) -> Line:
    return Line(
        line_id=f"as:{index:04d}",
        kind=DocumentKind.ANSWER_SHEET,
        page=page,
        box=BBox(x0=x0, y0=y0, x1=x0 + 0.60, y1=y0 + 0.020),
        text=text,
        confidence=0.9,
        engine=OcrEngine.PADDLE_OCR_VL,
    )


def ink(index: int, *, y0: float, y1: float, page: int = 0, x0: float = 0.10) -> InkRegion:
    return InkRegion(
        region_id=f"ink:{index:03d}",
        page=page,
        box=BBox(x0=x0, y0=y0, x1=x0 + 0.60, y1=y1),
        kind=InkRegionKind.WRITING,
        ink_ratio=0.20,
        mean_darkness=0.25,
        pixel_count=2000,
    )


def question(qid: str, label: str, text: str, order: int, path: list[str]) -> Question:
    return Question(
        qid=qid, label_raw=label, text=text, path=path, print_order=order
    )


class TestSegmentation:
    def test_a_large_gap_starts_a_new_block(self) -> None:
        lines = [
            line(1, "First answer, line one.", y0=0.10),
            line(2, "First answer, line two.", y0=0.13),
            # A gap several times the line spacing.
            line(3, "Second answer entirely.", y0=0.30),
        ]
        blocks = segment_blocks(lines, [])
        assert len(blocks) == 2
        assert blocks[0].line_ids == ["as:0001", "as:0002"]

    def test_a_written_question_label_starts_a_new_block(self) -> None:
        # The strongest boundary a student ever gives, and it holds even without
        # a gap to support it.
        lines = [
            line(1, "1. First answer.", y0=0.10),
            line(2, "2. Second answer.", y0=0.13),
        ]
        blocks = segment_blocks(lines, [])
        assert len(blocks) == 2

    @staticmethod
    def _sheet_with_a_hole() -> list[Line]:
        """Three tight lines to set a baseline, then a gap, then two more.

        A baseline of several close gaps is required before any gap can look
        abnormal — with one gap on the page, that gap *is* normal spacing.
        """
        return [
            line(1, "Answer one, first line.", y0=0.10),
            line(2, "Answer one, second line.", y0=0.13),
            line(3, "Answer one, third line.", y0=0.16),
            # A hole four times the established spacing.
            line(4, "continues after the hole.", y0=0.30),
            line(5, "and ends here.", y0=0.33),
        ]

    def test_ink_in_the_gap_prevents_a_false_split(self) -> None:
        # The Finding A case. A line the recognizer missed leaves a text-shaped
        # hole inside one answer; splitting there invents an orphan and truncates
        # the highlight. Ink is found independently of recognition, so ink in the
        # hole says writing is there.
        lines = self._sheet_with_a_hole()

        without = segment_blocks(lines, [])
        with_ink = segment_blocks(lines, [ink(1, y0=0.20, y1=0.27)])

        text_blocks = [b for b in with_ink if b.line_ids]
        assert len(without) == 2, "a bare hole should split"
        assert len(text_blocks) == 1, "ink in the hole should prevent the split"

    def test_bleed_through_in_a_gap_does_not_prevent_a_split(self) -> None:
        # Bleed-through is not the student's writing on this page, so it is not
        # evidence that an answer continues. Counting it would merge unrelated
        # answers on every double-sided script.
        lines = self._sheet_with_a_hole()
        faint = [
            ink(1, y0=0.20, y1=0.27).model_copy(
                update={"kind": InkRegionKind.BLEED_THROUGH}
            )
        ]
        text_blocks = [b for b in segment_blocks(lines, faint) if b.line_ids]
        assert len(text_blocks) == 2

    def test_a_page_break_splits_unless_continuation_is_stated(self) -> None:
        # Deferring this to the aligner does not work, because the aligner assigns
        # whole blocks and cannot divide one — so a page break that never splits
        # fuses two answers permanently. Observed on a real two-page script: both
        # programs became one block, the second question read as unanswered, and
        # the first claimed writing from a page it had nothing to do with.
        #
        # The asymmetry runs the other way than it does within a page. A wrongly
        # split page-spanning answer is repairable by the ``continue`` move; a
        # wrongly merged pair of answers is repairable by nothing.
        lines = [
            line(1, "The first answer ends here.", y0=0.90, page=0),
            line(2, "A different answer starts here.", y0=0.08, page=1),
        ]
        blocks = segment_blocks(lines, [])
        assert len(blocks) == 2
        assert [b.pages_spanned for b in blocks] == [[0], [1]]

    def test_a_stated_continuation_carries_an_answer_across_the_break(self) -> None:
        # The requirement that answers may span pages, honoured on evidence rather
        # than on assumption.
        lines = [
            line(1, "Answer continues, cont. on next page", y0=0.90, page=0),
            line(2, "onto the next page.", y0=0.08, page=1),
        ]
        blocks = segment_blocks(lines, [])
        assert len(blocks) == 1
        assert blocks[0].spans_pages
        assert blocks[0].pages_spanned == [0, 1]

    @pytest.mark.parametrize(
        "marker",
        ["cont. on next page", "continued overleaf", "P.T.O.", "see next page"],
    )
    def test_detects_continuation_markers(self, marker: str) -> None:
        lines = [line(1, f"The answer so far, {marker}", y0=0.10)]
        blocks = segment_blocks(lines, [])
        assert blocks[0].has_continuation_marker

    def test_unclaimed_ink_becomes_a_text_free_block(self) -> None:
        # A hand-drawn diagram produces ink and no text, and is still an answer
        # that has to be highlightable. Discarding it would leave the question it
        # answers with nothing at all.
        lines = [line(1, "1. See the diagram below.", y0=0.10)]
        diagram = [ink(i, y0=0.40 + i * 0.03, y1=0.42 + i * 0.03) for i in range(3)]

        blocks = segment_blocks(lines, diagram)
        text_free = [b for b in blocks if b.is_text_free]

        assert text_free, "the diagram should become its own block"
        assert text_free[0].geometry, "and must carry geometry to be highlightable"

    def test_ink_overlapping_a_block_is_attached_not_duplicated(self) -> None:
        lines = [line(1, "1. An answer.", y0=0.10)]
        overlapping = [ink(1, y0=0.10, y1=0.12)]

        blocks = segment_blocks(lines, overlapping)
        assert len(blocks) == 1
        assert blocks[0].ink_region_ids == ["ink:001"]
        assert not blocks[0].is_text_free

    def test_normal_spacing_is_not_the_median_of_all_gaps(self) -> None:
        # On a sheet of short answers, close to half of all gaps are *between*
        # answers, so a median sits between the two populations and no gap ever
        # looks large. That collapsed a whole sheet into one block.
        lines: list[Line] = []
        y = 0.05
        for pair in range(5):
            lines.append(line(pair * 2 + 1, f"Answer {pair} line one.", y0=y))
            lines.append(line(pair * 2 + 2, f"Answer {pair} line two.", y0=y + 0.024))
            y += 0.024 + 0.055  # a clear gap before the next answer

        blocks = segment_blocks(lines, [])
        assert len(blocks) == 5, f"expected one block per answer, got {len(blocks)}"

    def test_an_empty_sheet_yields_no_blocks(self) -> None:
        assert segment_blocks([], []) == []

    def test_blank_lines_are_ignored(self) -> None:
        lines = [line(1, "   ", y0=0.10), line(2, "Real text.", y0=0.13)]
        blocks = segment_blocks(lines, [])
        assert len(blocks) == 1
        assert blocks[0].line_ids == ["as:0002"]


PAPER = [
    question("A/1", "1.", "Define refraction of light.", 0, ["1"]),
    question("A/2/a", "2 (a)", "State the laws of reflection.", 1, ["2", "a"]),
    question("A/2/b", "(b)", "Give the SI unit of power.", 2, ["2", "b"]),
    question("A/3", "3.", "Explain the working of an electric motor.", 3, ["3"]),
]


def block_with(text: str, block_id: str = "blk:000"):
    from vedaai_contracts import AnswerBlock, PageBox

    return AnswerBlock(
        block_id=block_id,
        line_ids=["as:0001"],
        text=text,
        geometry=[PageBox(page=0, box=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.2))],
        pages_spanned=[0],
    )


class TestAnchorDetection:
    def test_confirms_a_label_whose_answer_matches(self) -> None:
        lines = [line(1, "1. Refraction is the bending of light between media.", y0=0.10)]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, PAPER)

        assert len(found) == 1
        assert found[0].claimed_qid == "A/1"
        assert found[0].status is AnchorStatus.CONFIRMED

    def test_matches_a_label_despite_different_punctuation(self) -> None:
        # A student writes "2 a)" where the paper printed "2 (a)". The structure
        # is the same; only the punctuation differs.
        lines = [line(1, "2 a) The angle of incidence equals the angle of reflection.", y0=0.10)]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, PAPER)
        assert found[0].claimed_qid == "A/2/a"

    def test_resolves_a_lone_subpart_against_the_recent_parent(self) -> None:
        # Having written "2 (a)" above, a student sees no need to repeat the 2.
        lines = [
            line(1, "2 (a) The angle of incidence equals the angle of reflection.", y0=0.10),
            line(2, "(b) The watt is the SI unit of power measurement.", y0=0.30),
        ]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, PAPER)

        assert [a.claimed_qid for a in found] == ["A/2/a", "A/2/b"]

    def test_disputes_a_label_naming_no_question_on_the_paper(self) -> None:
        # Strong evidence of mislabelling, and nothing to pin an alignment to
        # regardless.
        lines = [line(1, "8. Refraction is the bending of light between media.", y0=0.10)]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, PAPER)

        assert found[0].claimed_qid is None
        assert found[0].status is AnchorStatus.DISPUTED

    def test_does_not_dispute_a_correct_label_with_no_shared_vocabulary(self) -> None:
        # The false positive that cost a good anchor. "Explain the working of an
        # electric motor" answered by "a current-carrying coil in a magnetic
        # field" is correct and shares no content word, so an absolute similarity
        # floor condemned it.
        lines = [
            line(
                1,
                "3. A current-carrying coil placed in a magnetic field turns.",
                y0=0.10,
            )
        ]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, PAPER)

        assert found[0].claimed_qid == "A/3"
        assert found[0].status is not AnchorStatus.DISPUTED

    def test_out_of_order_answers_are_not_disputed(self) -> None:
        # Answering out of order is explicitly permitted, so falling outside the
        # increasing run is not evidence of anything.
        lines = [
            line(1, "3. A current-carrying coil placed in a magnetic field turns.", y0=0.10),
            line(2, "1. Refraction is the bending of light between two media.", y0=0.35),
        ]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, PAPER)

        assert {a.claimed_qid for a in found} == {"A/3", "A/1"}
        assert all(a.status is not AnchorStatus.DISPUTED for a in found)

    def test_a_short_answer_is_not_disputed_for_being_short(self) -> None:
        # "The watt." is a correct answer and carries too little text for
        # similarity to mean anything. Absence of evidence is not disagreement.
        # The label resolves via the paper's own printed "(b)", so a lone
        # sub-part with no preceding parent is still placeable.
        lines = [line(1, "(b) The watt.", y0=0.10)]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, PAPER)

        assert found[0].semantic_agreement is None
        assert found[0].status is not AnchorStatus.DISPUTED

    def test_order_consistency_confirms_a_forward_run(self) -> None:
        lines = [
            line(1, "1. Refraction is the bending of light between two media.", y0=0.10),
            line(2, "2 (a) Angles of incidence and reflection are equal here.", y0=0.35),
            line(3, "3. A coil in a magnetic field experiences a turning force.", y0=0.60),
        ]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, PAPER)

        assert all(a.order_consistent for a in found)
        assert all(a.status is AnchorStatus.CONFIRMED for a in found)

    def test_only_confirmed_anchors_may_pin(self) -> None:
        lines = [line(1, "8. Some writing that names no real question at all.", y0=0.10)]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, PAPER)

        assert anchors.confirmed(found) == []
        assert not found[0].may_pin

    def test_a_block_with_no_label_produces_no_anchor(self) -> None:
        lines = [line(1, "Refraction is the bending of light between two media.", y0=0.10)]
        blocks = segment_blocks(lines, [])
        assert anchors.detect(blocks, lines, PAPER) == []

    def test_a_text_free_block_produces_no_anchor(self) -> None:
        diagram = [ink(i, y0=0.40 + i * 0.03, y1=0.42 + i * 0.03) for i in range(3)]
        blocks = segment_blocks([], diagram)
        assert anchors.detect(blocks, [], PAPER) == []


class TestLexicalOverlap:
    def test_identical_text_scores_high(self) -> None:
        overlap = LexicalOverlap()
        assert overlap.score("bending of light", "bending of light") == pytest.approx(1.0)

    def test_unrelated_text_scores_zero(self) -> None:
        overlap = LexicalOverlap()
        assert overlap.score("bending of light", "photosynthesis in plants") == 0.0

    def test_stopwords_alone_do_not_create_similarity(self) -> None:
        # Otherwise every pair of English sentences looks related, and the
        # measure becomes a constant.
        overlap = LexicalOverlap()
        assert overlap.score("the and of in", "a to for with") == 0.0

    def test_repetition_counts(self) -> None:
        # Term frequency is kept deliberately: an answer mentioning reflection
        # three times is more likely about reflection than one mentioning it once.
        overlap = LexicalOverlap()
        focused = overlap.score("reflection", "reflection reflection reflection")
        diluted = overlap.score("reflection", "reflection alpha beta gamma delta")
        assert focused > diluted

    def test_empty_text_scores_zero(self) -> None:
        overlap = LexicalOverlap()
        assert overlap.score("", "anything") == 0.0
        assert overlap.score("anything", "") == 0.0
