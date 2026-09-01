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
from grader.answers import furniture as answer_furniture
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

    def test_algebra_that_looks_like_a_label_does_not_split_an_answer(self) -> None:
        # Handwritten working is full of text that a label grammar accepts. On a
        # real script "5(n) + 3(P) = 190" parsed as question 5 part n, and
        # "5(26) + 3P = 190" as question 5 part 26 -- and a label is the strongest
        # boundary the segmenter has, so each cut the answer in two. The fragments
        # then competed for questions on their own and landed on questions the
        # student had never attempted.
        #
        # The paper says which numbers are real. Nothing here is question 5.
        paper = [
            question("1", "Q1", "A man buys five notebooks and three pens.", 0, ["1"]),
            question("2", "Q2", "A father and his son have some coins.", 1, ["2"]),
        ]
        lines = [
            line(1, "Q1. Given 5 notebooks and 3 pens = 190", y0=0.10),
            line(2, "5(n) + 3(P) = 190", y0=0.13),
            line(3, "5(26) + 3P = 190", y0=0.16),
            line(4, "So n = 26 and P = 20", y0=0.19),
        ]
        blocks = segment_blocks(lines, [], paper)

        assert len(blocks) == 1, "one answer, not three"
        assert "5(26)" in blocks[0].text

    def test_a_misread_subpart_marker_does_not_split_an_answer(self) -> None:
        # The student wrote "(i)" and "(ii)" to number the parts of one proof; the
        # recognizer read both as "(9)". Parsed as question 9 -- a question this
        # paper does not have -- each split the proof, and the fragment carrying
        # its conclusion drifted onto a different question entirely.
        paper = [
            question("T/1", "T1", "AB is a line segment and P is its mid-point.", 0, ["1"]),
            question("T/2", "T2", "Two isosceles triangles on the same base.", 1, ["2"]),
        ]
        lines = [
            line(1, "T1 P is the mid point of AB, so AP = BP.", y0=0.10),
            line(2, "(9) Now, AP = BP and angle APD = angle EPB.", y0=0.13),
            line(3, "(9) Then, AD = BE by CPCT.", y0=0.16),
        ]
        blocks = segment_blocks(lines, [], paper)

        assert len(blocks) == 1, "one proof, not three"
        assert "CPCT" in blocks[0].text

    def test_a_label_naming_a_real_question_still_splits(self) -> None:
        # The guard must not cost the boundary it exists to find. Two answers, each
        # opening with a label the paper actually printed.
        paper = [
            question("1", "Q1", "A man buys five notebooks and three pens.", 0, ["1"]),
            question("2", "Q2", "A father and his son have some coins.", 1, ["2"]),
        ]
        lines = [
            line(1, "Q1. The notebook costs 26 rupees.", y0=0.10),
            line(2, "Q2. The father has 42 coins.", y0=0.13),
        ]
        blocks = segment_blocks(lines, [], paper)
        assert len(blocks) == 2

    def test_ink_spanning_several_lines_of_a_block_is_attached_to_it(self) -> None:
        # A connected component of handwriting covers the whole answer, not one
        # line of it, and block geometry is a box *per line*. Asking whether any
        # single line box contains a third of the region answers no as soon as the
        # answer is three lines long, and the region is then promoted to a
        # text-free block that alignment is right to refuse — so the writing that
        # was read, transcribed and mapped is counted as ink belonging to nothing,
        # and the unassigned-ink total that qualifies every absence claim on the
        # page goes to one.
        lines = [
            line(1, "1. The first line of the answer.", y0=0.10),
            line(2, "The second line of the answer.", y0=0.13),
            line(3, "The third line of the answer.", y0=0.16),
            line(4, "The fourth line of the answer.", y0=0.19),
        ]
        spanning = [ink(1, y0=0.10, y1=0.21)]

        blocks = segment_blocks(lines, spanning)

        assert len(blocks) == 1, "the answer is one block and the ink is its own"
        assert blocks[0].ink_region_ids == ["ink:001"]
        assert not any(b.is_text_free for b in blocks)

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

    def test_a_written_label_resolves_within_its_own_section(self) -> None:
        # A paper whose numbering restarts per section: Q1-Q2 carry no section,
        # T1-T2 sit under "T". Both sections therefore contain a question whose
        # path is ("2",), so a path-keyed index has two entries for one key and
        # the later section silently wins.
        #
        # On the real mathematics paper that made every one of Q1-Q4 unreachable:
        # the student wrote "Q4." above their own Q4 working and the anchor
        # resolved to T/4 — a question they never attempted — then confirmed it
        # and pinned the alignment to it. Their correct label was what moved the
        # answer to the wrong question.
        paper = [
            question("1", "Q1", "A man buys five notebooks and three pens.", 0, ["1"]),
            question("2", "Q2", "A father and his son have some coins each.", 1, ["2"]),
            question("T/1", "T1", "AB is a line segment and P is its mid-point.", 2, ["1"]),
            question("T/2", "T2", "Two isosceles triangles stand on the same base.", 3, ["2"]),
        ]
        lines = [line(1, "Q2. The father has 42 coins and the son has 30 coins.", y0=0.10)]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, paper)

        assert found[0].claimed_qid == "2", "the paper printed this question as Q2"

    def test_a_label_matching_two_sections_pins_nothing(self) -> None:
        # The same paper, and a student who wrote a bare "2." with no section
        # letter. That is genuinely ambiguous between Q2 and T2, and there is no
        # evidence here to settle it. Picking one is fabricated certainty, and
        # because a resolved anchor may pin the alignment, the fabrication would
        # outrank both semantics and continuation.
        paper = [
            question("1", "Q1", "A man buys five notebooks and three pens.", 0, ["1"]),
            question("2", "Q2", "A father and his son have some coins each.", 1, ["2"]),
            question("T/1", "T1", "AB is a line segment and P is its mid-point.", 2, ["1"]),
            question("T/2", "T2", "Two isosceles triangles stand on the same base.", 3, ["2"]),
        ]
        lines = [line(1, "2. 12A + 18B = 324, which is the second given equation.", y0=0.10)]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, paper)

        assert found[0].claimed_qid is None, "ambiguous between Q2 and T2"
        assert not found[0].may_pin

    def test_reads_a_written_label_in_the_papers_own_section_style(self) -> None:
        # The paper numbers a section with a letter prefix -- "T1".."T5" beside
        # "Q1".."Q4" -- and the student wrote "T2" above their answer. `parse_label`
        # cannot see that as a label without being told the paper's prefixes, and
        # `detect` already holds the questions those prefixes come from.
        #
        # Left unpassed, the strongest signal on the page is discarded: the block
        # gets no anchor at all and is placed on whatever the wording happens to
        # favour. On the real script that was a question the student never touched.
        # Three numbers under each prefix, because that is what makes a prefix a
        # scheme rather than a coincidence -- see `detect_section_prefixes`.
        paper = [
            question("1", "Q1", "A man buys five notebooks and three pens.", 0, ["1"]),
            question("2", "Q2", "A father and his son have some coins each.", 1, ["2"]),
            question("3", "Q3", "A fruit seller sells apples and oranges.", 2, ["3"]),
            question("T/1", "T1", "AB is a line segment and P is its mid-point.", 3, ["1"]),
            question("T/2", "T2", "Two isosceles triangles stand on the same base.", 4, ["2"]),
            question("T/3", "T3", "Bisectors of two angles meet at a point O.", 5, ["3"]),
        ]
        lines = [
            line(1, "T2", y0=0.10),
            line(2, "Proof of (i): triangle ABD is congruent to triangle ACD.", y0=0.13),
        ]
        blocks = segment_blocks(lines, [])
        found = anchors.detect(blocks, lines, paper)

        assert found, "the student's own T2 must be read as a label"
        assert found[0].claimed_qid == "T/2"

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


class TestSemanticSimilarity:
    """Understanding, rather than counting shared characters.

    The gap this closes, measured on the surface scorers:

        Define refraction of light.            -> The bending of light...   0.236
        Name the process by which plants...    -> It is called transpiration 0.000
        State the SI unit of pressure.         -> The pascal.               0.081
        Describe the causes of the revolution  -> Heavy taxes, a bankrupt... 0.169

    Four of five correct answers score at or near zero against their own question,
    because an answer restates the idea rather than the wording — which is what a
    good answer does. With semantics silent the aligner falls back to position, and
    a script whose answers are not in order is then placed by habit.

    Stubbed rather than calling a live service: what belongs to this module is the
    caching, the batching and the fallback, and none of that needs a network.
    """

    def test_meaning_beats_shared_words(self) -> None:
        from grader.answers.similarity import SemanticSimilarity

        # Vectors chosen so the surface scorers would get this exactly backwards:
        # the wrong answer repeats the question's own words, the right one does not.
        vectors = {
            "Name the process by which plants lose water as water vapour.": [1.0, 0.0],
            "It is called transpiration.": [0.96, 0.28],
            "The process of water and plants is a process.": [0.0, 1.0],
        }
        similarity = SemanticSimilarity(embed=lambda texts: [vectors[t] for t in texts])

        question = "Name the process by which plants lose water as water vapour."
        right = similarity.score(question, "It is called transpiration.")
        wrong = similarity.score(question, "The process of water and plants is a process.")
        assert right > wrong

    def test_texts_are_embedded_once_however_often_they_are_compared(self) -> None:
        """Every question is scored against every block, so caching is not an
        optimisation — it is the difference between a few calls and a few hundred.
        """
        from grader.answers.similarity import SemanticSimilarity

        calls: list[list[str]] = []

        def embed(texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

        similarity = SemanticSimilarity(embed=embed)
        for _ in range(5):
            similarity.score("a question", "an answer")

        assert len(calls) == 1, f"embedded {len(calls)} times, expected once"
        assert sorted(calls[0]) == ["a question", "an answer"]

    def test_falls_back_when_the_service_is_unavailable(self) -> None:
        """A failure here must not fail a submission.

        Marking already degrades to a rubric when no key is set, and mapping has
        to degrade the same way: the surface scorers are weaker, not useless, and
        an answer sheet placed by trigrams and position is far better than a
        submission that errored.
        """
        from grader.answers.similarity import SemanticSimilarity

        def broken(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("no network")

        similarity = SemanticSimilarity(embed=broken)
        score = similarity.score(
            "Define refraction of light.",
            "Refraction is the bending of light.",
        )
        assert score > 0.0, "must fall back to the surface measure, not return zero"

    def test_one_outage_does_not_follow_every_later_submission(self) -> None:
        """A provider blip must not silently downgrade the rest of the process.

        The scorer is built once, at import, and shared by every submission the
        task handles. The flag it set on a failed call had no expiry, so a single
        timeout meant every script uploaded afterwards — for hours, until the task
        recycled — was placed by word overlap instead of by meaning, with nothing
        said to anybody. It is the most likely explanation for the same script
        mapping differently on two runs.
        """
        from grader.answers.similarity import SemanticSimilarity

        clock = [0.0]
        attempts: list[int] = []

        def flaky(texts: list[str]) -> list[list[float]]:
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise RuntimeError("no network")
            return [[1.0, 0.0] for _ in texts]

        similarity = SemanticSimilarity(embed=flaky, now=lambda: clock[0])
        similarity.score("Define refraction.", "Refraction bends light.")
        assert similarity.degraded is True

        clock[0] += 3600.0
        similarity.score("Name the process.", "It is transpiration.")
        assert len(attempts) == 2, "the provider must be tried again once the wait is over"
        assert similarity.degraded is False

    def test_a_failure_is_reported_rather_than_absorbed(self) -> None:
        # Falling back is right; falling back quietly is not. A mapping placed by
        # spelling is a materially different product from one placed by meaning,
        # and the teacher looking at it is the person who needs to know.
        from grader.answers.similarity import SemanticSimilarity

        def broken(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("no network")

        similarity = SemanticSimilarity(embed=broken)
        assert similarity.degraded is False
        similarity.score("Define refraction.", "Refraction bends light.")
        assert similarity.degraded is True

    def test_an_empty_text_scores_zero_without_calling_out(self) -> None:
        from grader.answers.similarity import SemanticSimilarity

        calls: list[list[str]] = []
        similarity = SemanticSimilarity(embed=lambda t: (calls.append(t), [[1.0]] * len(t))[1])
        assert similarity.score("", "anything") == 0.0
        assert calls == []


class TestScriptDetails:
    """Separating a student's answers from the details they wrote at the top.

    Question papers have had their headers stripped from the beginning; answer
    sheets had nothing, so a name badge was a candidate answer. On the golden set
    "Name: Test Student  Class: 6C" was assigned to "Describe an experiment to show
    that air has mass" — a question the student left blank.

    The asymmetry is the usual one: keeping a stray line costs a teacher one glance,
    discarding a real answer loses a mark. So these tests are weighted toward
    refusing to strip.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Name: Test Student        Class: 6C",
            "Name - Anjana S Kamath",
            "Roll No: 41",
            "Roll number: 22CS041",
            "Register No : 6282350749",
            "Class: 6C",
            "Semester: 8",
            "Branch - CSE",
            "Subject: Physics",
            "Date: 12/03/2026",
        ],
    )
    def test_recognizes_the_fields_a_student_fills_in(self, text: str) -> None:
        assert answer_furniture.is_furniture(line(1, text, y0=0.02)) is True

    @pytest.mark.parametrize(
        "text",
        [
            # The words themselves appear inside real answers, which is why the
            # pattern requires them to be used as labelled fields.
            "Name the type of reaction shown above.",
            "The class of the compound is an alkane.",
            "Date the sample was collected is not given, so assume today.",
            "Refraction is the bending of light when it changes medium.",
            "R = V / I = 10 / 2 = 5 ohm",
        ],
    )
    def test_never_strips_an_answer(self, text: str) -> None:
        assert answer_furniture.is_furniture(line(1, text, y0=0.02)) is False

    def test_a_paper_set_marker_is_stripped_only_in_the_header(self) -> None:
        assert answer_furniture.is_furniture(line(1, "Set 3", y0=0.02)) is True
        # Further down the page it could be part of an answer — "Set 3 is empty",
        # for instance — and position is the only thing separating the two.
        assert answer_furniture.is_furniture(line(2, "Set 3", y0=0.55)) is False

    def test_a_long_number_is_stripped_only_in_the_header(self) -> None:
        assert answer_furniture.is_furniture(line(1, "6282350749", y0=0.01)) is True
        assert answer_furniture.is_furniture(line(2, "6282350749", y0=0.60)) is False

    def test_a_bare_page_number_is_stripped_at_the_page_edge(self) -> None:
        assert answer_furniture.is_furniture(line(1, "2", y0=0.97)) is True
        assert answer_furniture.is_furniture(line(2, "Page 2 of 4", y0=0.98)) is True
        # And at the top, which is where this script's own page number sits.
        assert answer_furniture.is_furniture(line(3, "Page 2", y0=0.02)) is True

    def test_a_bare_number_in_the_body_is_arithmetic_not_a_page_number(self) -> None:
        # The page-number pattern was tested *before* the header guard, so a bare
        # one-to-three digit number counted as furniture wherever it appeared. On a
        # real script that deleted 190, 25, 15, 2 and 3 from the working -- pieces
        # of 5(26)+3P=190, 25*theta=230 and 198/15 -- so the grader was shown an
        # answer with its arithmetic removed and read the gaps as the student's.
        assert answer_furniture.is_furniture(line(1, "190", y0=0.45)) is False
        assert answer_furniture.is_furniture(line(2, "25", y0=0.57)) is False
        assert answer_furniture.is_furniture(line(3, "15", y0=0.74)) is False

    def test_strips_a_header_that_repeats_in_place_across_pages(self) -> None:
        # The roll-number box exactly as the recogniser rendered it on four pages:
        # every page different, none of them matching any identity pattern, because
        # furniture is the text *least* likely to be read correctly -- small print,
        # boxed, at the very edge of a scan.
        #
        # What is stable is where it sits: y0 within 0.004 of the same height on
        # every page. Position identifies it and its words never will.
        header = ["Rdi No: 37", "RdiNo: 37", "Rdino: 3", "Rdl No: 37"]
        lines = []
        for page, text in enumerate(header):
            lines.append(
                line(page * 2 + 1, text, y0=0.088 + page * 0.001, page=page, x0=0.38)
            )
            lines.append(
                line(
                    page * 2 + 2,
                    f"The answer to question {page + 1} carries on down the page.",
                    y0=0.30,
                    page=page,
                )
            )
        answers, details = answer_furniture.strip(lines)
        assert [ln.text for ln in details] == header
        assert len(answers) == 4, "the four answer lines must all survive"

    def test_strip_returns_both_halves(self) -> None:
        # Neither half is discarded, so a caller can report what was set aside
        # rather than have it disappear.
        lines = [
            line(1, "Name: Test Student   Class: 6C", y0=0.02),
            line(2, "1. Refraction is the bending of light.", y0=0.12),
        ]
        answers, details = answer_furniture.strip(lines)
        assert [ln.line_id for ln in answers] == ["as:0002"]
        assert [ln.line_id for ln in details] == ["as:0001"]

    def test_the_header_does_not_become_an_answer_block(self) -> None:
        # The end-to-end shape of the fix.
        lines = [
            line(1, "Name: Test Student   Class: 6C", y0=0.02),
            line(2, "1. Refraction is the bending of light.", y0=0.12),
        ]
        blocks = segment_blocks(lines, [])
        assert len(blocks) == 1
        assert "Name" not in blocks[0].text
