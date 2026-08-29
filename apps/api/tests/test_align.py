"""Tests for answer-to-question alignment.

This is the module the project is graded on, and the tests are weighted toward
the one error that matters most: telling a teacher a question was left blank when
it was answered. A teacher acts on that without re-reading the script, so several
cases below assert that the system prefers admitting uncertainty over claiming
absence — even where that costs precision elsewhere.
"""

from __future__ import annotations

import pytest
from vedaai_contracts import (
    Anchor,
    AnchorStatus,
    AnswerBlock,
    AnswerStatus,
    BBox,
    InkRegion,
    InkRegionKind,
    PageBox,
    Question,
    QuestionPaper,
    Requirement,
    Section,
)

from grader.align import resolve


def q(qid: str, label: str, text: str, order: int, path: list[str], *, marks: int | None = 2,
      section: str | None = None) -> Question:
    return Question(
        qid=qid,
        label_raw=label,
        text=text,
        path=path,
        print_order=order,
        marks=marks,
        section_id=section,
    )


def block(
    block_id: str,
    text: str,
    *,
    y0: float,
    page: int = 0,
    line_ids: list[str] | None = None,
    ink_ids: list[str] | None = None,
    continuation: bool = False,
    pages: list[int] | None = None,
) -> AnswerBlock:
    return AnswerBlock(
        block_id=block_id,
        line_ids=line_ids if line_ids is not None else [f"as:{block_id[-3:]}"],
        ink_region_ids=ink_ids or [],
        text=text,
        geometry=[PageBox(page=page, box=BBox(x0=0.1, y0=y0, x1=0.9, y1=y0 + 0.08))],
        pages_spanned=pages or [page],
        has_continuation_marker=continuation,
    )


def anchor(anchor_id: str, label: str, qid: str | None, line_id: str, *,
           status: AnchorStatus = AnchorStatus.CONFIRMED) -> Anchor:
    return Anchor(
        anchor_id=anchor_id,
        claimed_label=label,
        claimed_qid=qid,
        line_id=line_id,
        page=0,
        box=BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.13),
        status=status,
    )


def paper(questions: list[Question], sections: list[Section] | None = None) -> QuestionPaper:
    return QuestionPaper(questions=questions, sections=sections or [])


REFRACTION = q("A/1", "1.", "Define refraction of light.", 0, ["1"])
REFLECTION = q("A/2", "2.", "State the laws of reflection.", 1, ["2"])
MOTOR = q("A/3", "3.", "Explain the working of an electric motor.", 2, ["3"], marks=5)


class TestConfirmedAnchors:
    def test_a_confirmed_anchor_places_its_answer(self) -> None:
        questions = [REFRACTION, REFLECTION]
        blocks = [
            block("blk:000", "Refraction is the bending of light.", y0=0.10, line_ids=["as:0001"]),
            block("blk:001", "Angles of incidence and reflection are equal.", y0=0.30,
                  line_ids=["as:0002"]),
        ]
        anchors = [
            anchor("anc:000", "1.", "A/1", "as:0001"),
            anchor("anc:001", "2.", "A/2", "as:0002"),
        ]
        result = resolve(paper(questions), blocks, anchors, [])
        by_qid = result.by_qid()

        assert by_qid["A/1"].block_ids == ["blk:000"]
        assert by_qid["A/2"].block_ids == ["blk:001"]

    def test_confirmed_anchors_are_honoured_in_reverse_order(self) -> None:
        # The design error this corrects. Treating anchors as pins for a monotone
        # DP meant only a monotone subset survived, so a fully reversed sheet lost
        # nearly every correctly-labelled answer to the orphan pile.
        questions = [REFRACTION, REFLECTION, MOTOR]
        blocks = [
            block("blk:000", "A coil in a magnetic field turns.", y0=0.10, line_ids=["as:0001"]),
            block("blk:001", "Angles of incidence and reflection are equal.", y0=0.35,
                  line_ids=["as:0002"]),
            block("blk:002", "Refraction is the bending of light.", y0=0.60,
                  line_ids=["as:0003"]),
        ]
        anchors = [
            anchor("anc:000", "3.", "A/3", "as:0001"),
            anchor("anc:001", "2.", "A/2", "as:0002"),
            anchor("anc:002", "1.", "A/1", "as:0003"),
        ]
        result = resolve(paper(questions), blocks, anchors, [])
        by_qid = result.by_qid()

        assert by_qid["A/3"].block_ids == ["blk:000"]
        assert by_qid["A/2"].block_ids == ["blk:001"]
        assert by_qid["A/1"].block_ids == ["blk:002"]
        assert result.orphans == []

    def test_a_disputed_anchor_does_not_place_its_answer(self) -> None:
        # It still influences the score, but it may not fix a pairing the aligner
        # cannot revisit — that is the whole point of confirmation.
        questions = [REFRACTION]
        blocks = [block("blk:000", "Refraction is the bending of light.", y0=0.10,
                        line_ids=["as:0001"])]
        anchors = [
            anchor("anc:000", "8.", None, "as:0001", status=AnchorStatus.DISPUTED)
        ]
        result = resolve(paper(questions), blocks, anchors, [])
        mapping = result.by_qid()["A/1"]
        # The DP may still place it on semantic grounds; what matters is that the
        # disputed label did not do the placing.
        assert mapping.evidence.label_agreement < 3.0


class TestAnAnswerMustFitWhatTheQuestionAsked:
    """A one-mark question is not answered in five lines, and a diagram is not prose.

    From four pages of genuine handwriting. The student's answer to question 1 —
    "Explain how pandas in China are similar to koalas in Australia, and how both
    are different from the python", worth 5 — runs five lines and thirty-one words:

        As described in the article, pandas eat an abundance of bamboo and
        therefore are specialist to China, while the koala eats eucalyptus leaves
        almost exclusively. Both these animals are specialists to one habitat
        while the python, a generalist is able to live in a wider range of
        habitats virtually anywhere.

    Because it names a python and a generalist, it scored higher against 3(ii) —
    "State whether a python is a specialist or a generalist", worth ONE mark —
    than against the question it plainly answers. Clicking question 3(ii) lit up
    the whole pandas paragraph.

    Meaning alone cannot separate these: 3(ii) really is a topic of the paragraph.
    What separates them is scale. One mark buys a phrase, and a question asking
    for a labelled diagram is not answered in thirty-one words of prose at all.
    """

    ANSWER = (
        "As described in the article, pandas eat an abundance of bamboo and "
        "therefore are specialist to China, while the koala eats eucalyptus "
        "leaves almost exclusively. Both these animals are specialists to one "
        "habitat while the python, a generalist is able to live in a wider "
        "range of habitats virtually anywhere."
    )

    COMPARE = q(
        "A/1", "1.",
        "Explain how pandas in China are similar to koalas in Australia, and how "
        "both are different from the python.",
        0, ["1"], marks=5,
    )
    STATE = q(
        "A/3/ii", "(ii)",
        "State whether a python is a specialist or a generalist.",
        1, ["3", "ii"], marks=1,
    )
    DRAW = q(
        "A/4", "4.",
        "Draw a labelled diagram contrasting the range of a specialist with that "
        "of a generalist.",
        2, ["4"], marks=3,
    )

    def _where_it_landed(self) -> str | None:
        result = resolve(
            paper([self.COMPARE, self.STATE, self.DRAW]),
            [block("blk:000", self.ANSWER, y0=0.10, line_ids=["as:0001"])],
            [],
            [],
        )
        for mapping in result.mappings:
            if mapping.status is AnswerStatus.ANSWERED:
                return mapping.qid
        return None

    def test_the_paragraph_goes_to_the_question_it_answers(self) -> None:
        assert self._where_it_landed() == "A/1"

    def test_a_one_mark_question_does_not_claim_five_lines(self) -> None:
        assert self._where_it_landed() != "A/3/ii"

    def test_a_drawing_question_is_not_answered_in_prose(self) -> None:
        assert self._where_it_landed() != "A/4"

    def test_a_terse_answer_to_a_big_question_is_still_its_answer(self) -> None:
        # The mirror case, and the reason under-length costs nothing. Students are
        # not paid by the word: a correct one-line answer to a five-mark question
        # loses marks, not its place on the paper.
        result = resolve(
            paper([MOTOR, REFRACTION]),
            [block("blk:000", "The motor spins because the current turns the coil.",
                   y0=0.10, line_ids=["as:0001"])],
            [],
            [],
        )
        landed = {m.qid: m.status for m in result.mappings}
        assert landed["A/3"] is AnswerStatus.ANSWERED


class TestGapsAndOrphans:
    def test_an_unanswered_question_is_a_gap_on_the_question_axis(self) -> None:
        questions = [REFRACTION, MOTOR]
        blocks = [block("blk:000", "Refraction is the bending of light.", y0=0.10,
                        line_ids=["as:0001"])]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, [])

        assert result.by_qid()["A/3"].status is not AnswerStatus.ANSWERED
        assert result.by_qid()["A/3"].highlight is None

    def test_an_orphan_answer_is_a_gap_on_the_block_axis(self) -> None:
        # Required by the brief, and worth surfacing: an orphan often means our
        # own extraction missed a question rather than the student writing extra.
        questions = [REFRACTION]
        blocks = [
            block("blk:000", "Refraction is the bending of light.", y0=0.10,
                  line_ids=["as:0001"]),
            block("blk:001", "Rough work: 12 x 4 = 48 divided by 6 gives 8.", y0=0.60,
                  line_ids=["as:0002"]),
        ]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, [])

        assert [o.block_id for o in result.orphans] == ["blk:001"]
        assert result.orphans[0].highlight.boxes


class TestMultiBlockAnswers:
    def test_a_page_spanning_block_keeps_one_box_per_page(self) -> None:
        questions = [REFRACTION]
        spanning = AnswerBlock(
            block_id="blk:000",
            line_ids=["as:0001", "as:0002"],
            text="Refraction is the bending of light as it passes between media.",
            geometry=[
                PageBox(page=0, box=BBox(x0=0.1, y0=0.85, x1=0.9, y1=0.95)),
                PageBox(page=1, box=BBox(x0=0.1, y0=0.05, x1=0.9, y1=0.20)),
            ],
            pages_spanned=[0, 1],
        )
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), [spanning], anchors, [])
        highlight = result.by_qid()["A/1"].highlight

        assert highlight is not None
        assert highlight.spans_pages
        assert highlight.pages == [0, 1]
        assert len(highlight.boxes) == 2, "one box per page, not one across both"

    def test_a_highlight_covers_the_writing_and_not_the_gaps(self) -> None:
        """Four lines of an answer, spread down a page with space between them.

        The bounding box of those four lines is mostly paper. Measured on real
        submissions, 60 to 74 per cent of a multi-line highlight was blank page,
        which is what "it highlights more than it should" looks like as a number.

        A highlight is allowed several boxes on a page, so there is no reason to
        pay for the gaps.
        """
        lines = [
            PageBox(page=0, box=BBox(x0=0.11, y0=0.10 + i * 0.08, x1=0.55, y1=0.12 + i * 0.08))
            for i in range(4)
        ]
        spread = AnswerBlock(
            block_id="blk:000",
            line_ids=[f"as:{i:04d}" for i in range(1, 5)],
            text="Refraction is the bending of light as it passes between media.",
            geometry=lines,
            pages_spanned=[0],
        )
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper([REFRACTION]), [spread], anchors, [])
        highlight = result.by_qid()["A/1"].highlight

        assert highlight is not None
        painted = sum(
            (b.box.x1 - b.box.x0) * (b.box.y1 - b.box.y0) for b in highlight.boxes
        )
        ink = sum((b.box.x1 - b.box.x0) * (b.box.y1 - b.box.y0) for b in lines)
        assert painted == pytest.approx(ink, rel=0.02), (
            f"highlight paints {painted:.4f} to cover {ink:.4f} of writing"
        )


class TestATailCarriesOnFromTheAnswerAboveIt:
    """A block that continues a placed answer joins it, rather than finding a home.

    Page four of the same real script, in full. One answer to question 2 — "The
    author uses the word 'invasive' throughout the passage. Explain why this word
    is significant" — written down the page with a paragraph break in the middle,
    which the segmenter read as the end of one block and the start of another:

        The word "invasive", is very important to rest of the article.

        ecosystems, the author writes, "Biologists, however, say that invasive
        species unchecked by natural predators are a major threat to biodiversity."

    The second block opens mid-sentence and quotes the passage, and that quoting
    is what undid it: question 6 asks the student to describe a tone "quoting one
    phrase in support", so the tail landed there instead.

    Question 2 had already been claimed by the block above, and a claimed question
    leaves the alignment entirely — so the move that exists for exactly this,
    carrying an answer across a block boundary, had nothing left to carry it to.
    A tail whose own preference is no better than settling therefore joins the
    answer directly above it before the rest of the alignment runs.
    """

    QUESTIONS = [
        q("A/1", "1.", "Explain how pandas in China are similar to koalas in "
          "Australia, and how both are different from the python.", 0, ["1"], marks=5),
        q("A/2", "2.", "The author uses the word 'invasive' throughout the passage. "
          "Explain why this word is significant.", 1, ["2"], marks=4),
        q("A/3/i", "(i)", "Name the single food source each specialist depends on.",
          2, ["3", "i"], marks=2),
        q("A/3/ii", "(ii)", "State whether a python is a specialist or a generalist.",
          3, ["3", "ii"], marks=1),
        q("A/4", "4.", "Draw a labelled diagram contrasting the range of a "
          "specialist with that of a generalist.", 4, ["4"], marks=3),
        q("B/5", "5.", "Suggest two measures a government could take to limit the "
          "spread of an invasive species.", 5, ["5"], marks=5),
        q("B/6", "6.", "Describe the tone the author adopts towards the pet trade, "
          "quoting one phrase in support.", 6, ["6"], marks=5),
    ]
    QID_BY_TEXT = {question.text: question.qid for question in QUESTIONS}

    HEAD = 'The word "invasive", is very important to rest of the article.'
    TAIL = (
        'ecosystems, the author writes, "Biologists, however, say that invasive '
        'species unchecked by natural predators are a major threat to biodiversity."'
    )

    #: Measured, not invented. These are what `text-embedding-3-small` returns for
    #: this page against this paper — the scorer the deployed service uses. Word
    #: overlap does not reproduce the fault at all, and a test that cannot fail
    #: before the fix demonstrates nothing, so the real numbers are pinned here.
    MEASURED = {
        ("A/1", "head"): 0.083, ("A/1", "tail"): 0.115,
        ("A/2", "head"): 0.369, ("A/2", "tail"): 0.176,
        ("A/3/i", "head"): 0.043, ("A/3/i", "tail"): 0.064,
        ("A/3/ii", "head"): 0.081, ("A/3/ii", "tail"): 0.122,
        ("A/4", "head"): 0.048, ("A/4", "tail"): 0.107,
        ("B/5", "head"): 0.186, ("B/5", "tail"): 0.215,
        ("B/6", "head"): 0.143, ("B/6", "tail"): 0.115,
    }

    def _similarity(self):
        outer = self

        class Measured:
            unrelated_below = 0.30

            def score(self, question_text: str, block_text: str) -> float:
                which = "head" if block_text == outer.HEAD else "tail"
                for (qid, part), value in outer.MEASURED.items():
                    if part == which and qid in outer.QID_BY_TEXT.get(question_text, ""):
                        return value
                return 0.0

        return Measured()

    def _mapping(self):
        result = resolve(
            paper(self.QUESTIONS),
            [
                block("blk:004", self.HEAD, y0=0.22, page=3, line_ids=["as:0004"]),
                block("blk:005", self.TAIL, y0=0.29, page=3, line_ids=["as:0005"]),
            ],
            [],
            [],
            similarity=self._similarity(),
        )
        return {m.qid: m for m in result.mappings}

    def test_the_tail_joins_the_answer_it_continues(self) -> None:
        assert self._mapping()["A/2"].block_ids == ["blk:004", "blk:005"]

    def test_the_tail_does_not_answer_a_question_of_its_own(self) -> None:
        assert self._mapping()["B/6"].status is not AnswerStatus.ANSWERED

    def test_an_unreadable_region_below_an_answer_is_not_swallowed_by_it(self) -> None:
        """A region with no readable text has nothing to be the rest of.

        Caught by the golden set rather than by reasoning: the first version of
        the tail rule reached around the guard that says an unreadable region
        needs a reason — a drawing question, or a label the student wrote — and
        attached one to the answer above purely because it sat underneath. Which
        is how a highlight grows past the writing it is meant to mark, and how
        the unassigned-ink total that qualifies every absence claim on the page
        quietly goes to zero.
        """
        ink = AnswerBlock(
            block_id="blk:ink000",
            line_ids=[],
            ink_region_ids=["ink:000"],
            text="",
            geometry=[PageBox(page=0, box=BBox(x0=0.1, y0=0.22, x1=0.5, y1=0.30))],
            pages_spanned=[0],
            is_text_free=True,
        )
        result = resolve(
            paper([REFRACTION]),
            [
                block("blk:000", "Refraction is the bending of light as it passes "
                                 "from one medium into another.", y0=0.10,
                      line_ids=["as:0001"]),
                ink,
            ],
            [anchor("anc:000", "1.", "A/1", "as:0001")],
            [],
        )
        placed = {m.qid: m.block_ids for m in result.mappings}
        assert placed["A/1"] == ["blk:000"]
        assert [o.block_id for o in result.orphans] == ["blk:ink000"]

    def test_a_block_that_clearly_answers_its_own_question_is_left_alone(self) -> None:
        # The guard. Two answers written one after the other must not be merged
        # just because they are adjacent — the tail rule only applies where the
        # block has no better home of its own than settling for its neighbour's.
        result = resolve(
            paper([REFRACTION, REFLECTION]),
            [
                block("blk:000", "Refraction is the bending of light as it passes "
                                 "from one medium into another.", y0=0.10,
                      line_ids=["as:0001"]),
                block("blk:001", "The angle of incidence equals the angle of "
                                 "reflection, and both rays lie in one plane.",
                      y0=0.40, line_ids=["as:0002"]),
            ],
            [],
            [],
        )
        placed = {m.qid: m.block_ids for m in result.mappings}
        assert placed["A/1"] == ["blk:000"]
        assert placed["A/2"] == ["blk:001"]


class TestWritingThatAnswersNothing:
    """Refusing to place writing that does not answer the question it is offered.

    Both cases below were found by looking at review pages, not by a metric, and
    both come from the same omission: the aligner scored on each block's deviation
    from its own mean and never asked how similar the pair actually was. A block
    always has a best question, even when the answer is about something else
    entirely, so there was always something to place.
    """

    def test_a_sheet_that_answers_a_different_paper_places_nothing(self) -> None:
        """A comprehension paper against a script of handwritten C.

        Measured on the deployed service: every block's best question scored 0.148
        to 0.154, where a genuine match scores 0.54 to 0.78. It reported five of
        seven questions answered and highlighted C code as an essay about pandas.
        Nothing on that sheet answers anything on that paper, and saying so is the
        only useful thing the system can do with it.
        """

        class Unrelated:
            unrelated_below = 0.30

            def score(self, a: str, b: str) -> float:
                return 0.15

        blocks = [
            block("blk:000", "for (i = 0; i < n; i++) { scanf(\"%d\", &a[i]); }", y0=0.10,
                  line_ids=["as:0001"]),
            block("blk:001", "printf(\"Enter the sorted array\"); temp = diff[i];", y0=0.40,
                  line_ids=["as:0002"]),
        ]
        result = resolve(paper([REFRACTION, REFLECTION]), blocks, [], [], similarity=Unrelated())

        for mapping in result.mappings:
            assert mapping.status is not AnswerStatus.ANSWERED, (
                f"{mapping.qid} was reported answered by writing that answers nothing"
            )
            assert not mapping.highlight or not mapping.highlight.boxes, (
                f"{mapping.qid} highlighted writing that does not answer it"
            )

    def test_a_block_is_not_settled_onto_a_much_worse_question(self) -> None:
        """The second answer to a question already taken.

        A real script answered question 2 across two pages. The first page took
        question 2; the second scored 0.689 for question 2 and 0.439 for question 5,
        and was placed on question 5 — where it was highlighted under a question it
        plainly does not answer. Half as good is not good enough to accept.
        """

        class Fixed:
            unrelated_below = 0.30

            def __init__(self, table: dict) -> None:
                self._table = table

            def score(self, a: str, b: str) -> float:
                return self._table.get((a[:20], b[:20]), 0.10)

        first = "Invasive is a significant word in the article because of the damage."
        second = "The word invasive is very important to the rest of the article too."
        table = {
            (REFLECTION.text[:20], first[:20]): 0.693,
            (REFLECTION.text[:20], second[:20]): 0.689,
            (MOTOR.text[:20], first[:20]): 0.413,
            (MOTOR.text[:20], second[:20]): 0.439,
        }
        blocks = [
            block("blk:000", first, y0=0.10, line_ids=["as:0001"]),
            block("blk:001", second, y0=0.50, line_ids=["as:0002"]),
        ]
        result = resolve(
            paper([REFLECTION, MOTOR]), blocks, [], [], similarity=Fixed(table)
        )
        motor = result.by_qid()["A/3"]
        assert motor.status is not AnswerStatus.ANSWERED, (
            "a block scoring 0.439 was accepted by a question whose own best is 0.689"
        )


class TestSubPartsAnsweredInOneRun:
    """A block answering two sub-parts must not be cut off from one of them.

    Question 11 of the science script, answered as a single 44-word run:

        11. Atomic number is the number of protons in the nucleus and mass number
        is the total of protons and neutrons. The atom has atomic number 11 and
        mass number 23.

    11(a) is worth 2 and 11(b) worth 1, so measured against 11(b) alone the run is
    nearly four times too long — and the rule that stops a block settling for a
    question when its own is taken read that as settling, and cut 11(b) before the
    move that exists for exactly this could be considered. 11(b) came back
    uncertain on a question the student had plainly answered.

    A sibling of the block's best question is not somewhere to settle for. It is
    the other half of the same answer, and whether the block really covers both is
    what `share` and its length support are there to decide.
    """

    PARENT = q("B/11", "11.", "Answer the following:", 0, ["11"], marks=None)
    PART_A = q("B/11/a", "11 (a)", "Define atomic number and mass number.",
               1, ["11", "a"], marks=2)
    PART_B = q("B/11/b", "11 (b)",
               "An atom has 11 protons and 12 neutrons. Give its atomic number "
               "and mass number.", 2, ["11", "b"], marks=1)

    #: The block verbatim, forty-four words of it. The length matters to the test:
    #: measured against 11(b)'s single mark the run is nearly four times too long,
    #: and it is that ratio, not the wording, that used to lose it.
    BOTH = (
        "11. Atomic number is the number of protons in the nucleus and mass "
        "number is the total of protons and neutrons. For the given atom the "
        "atomic number is 11 and the mass number is 11 + 12 = 23. So it is sodium."
    )

    #: What `text-embedding-3-small` returns for this run against these two, which
    #: is what the deployed service scores with. Word overlap does not reproduce
    #: the fault, and a test that passes before the fix demonstrates nothing.
    MEASURED = {"B/11/a": 0.659, "B/11/b": 0.752, "C/15": 0.244, "C/16": 0.207}

    def _similarity(self):
        by_text = {
            question.text: question.qid
            for question in (self.PART_A, self.PART_B, self.DISTRACTOR)
        }
        measured = self.MEASURED

        class Measured:
            unrelated_below = 0.30

            def score(self, question_text: str, _block_text: str) -> float:
                return measured.get(by_text.get(question_text, ""), 0.0)

        return Measured()

    DISTRACTOR = q("C/15", "15.",
                   "Explain the process of photosynthesis, naming the raw "
                   "materials and the products.", 3, ["15"], marks=5)

    def test_both_sub_parts_are_reported_answered(self) -> None:
        result = resolve(
            paper([self.PART_A, self.PART_B, self.DISTRACTOR]),
            [block("blk:008", self.BOTH, y0=0.10,
                   line_ids=["as:0008", "as:0009", "as:0010"])],
            [],
            [],
            similarity=self._similarity(),
        )
        status = {m.qid: m.status for m in result.mappings}
        assert status["B/11/a"] is AnswerStatus.ANSWERED
        assert status["B/11/b"] is AnswerStatus.ANSWERED, (
            "the student answered it in the same breath as (a)"
        )

    def test_two_whole_questions_are_not_parts_of_one(self) -> None:
        """The exemption is for parts of a question, not for questions.

        The first version of it asked `_may_share`, which calls two questions
        siblings when their labels share a parent — and the parent of every
        top-level question is the same empty root. So question 1, question 4 and
        question 6 were all each other's siblings, and one block about refraction
        was let through the settle rule onto "the chemical formula of washing
        soda" and "the tissue that transports water in a plant".
        """
        refraction = block(
            "blk:000",
            "Refraction is the bending of light as it passes from one medium "
            "into another of different density.",
            y0=0.10, line_ids=["as:0001"],
        )
        result = resolve(
            paper([
                REFRACTION,
                q("A/4", "4.", "Write the chemical formula of washing soda.",
                  1, ["4"], marks=1),
                q("A/6", "6.", "Name the tissue that transports water in a plant.",
                  2, ["6"], marks=1),
            ]),
            [refraction],
            [anchor("anc:000", "1.", "A/1", "as:0001")],
            [],
        )
        placed = {m.qid: m.block_ids for m in result.mappings}
        assert placed["A/1"] == ["blk:000"]
        assert placed["A/4"] == []
        assert placed["A/6"] == []


class TestFourStateStatus:
    def test_a_genuinely_blank_question_is_unanswered(self) -> None:
        # The only status that asserts absence, and the only one a teacher acts on
        # without checking. Everything else must be preferred where it fits.
        questions = [REFRACTION, MOTOR]
        blocks = [block("blk:000", "Refraction is the bending of light.", y0=0.10,
                        line_ids=["as:0001"])]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, [])

        assert result.by_qid()["A/3"].status is AnswerStatus.UNANSWERED

    @staticmethod
    def _ink_block() -> AnswerBlock:
        return AnswerBlock(
            block_id="blk:ink000",
            line_ids=[],
            ink_region_ids=["ink:001"],
            text="",
            geometry=[PageBox(page=0, box=BBox(x0=0.1, y0=0.3, x1=0.6, y1=0.6))],
            pages_spanned=[0],
        )

    @staticmethod
    def _ink_region() -> InkRegion:
        return InkRegion(
            region_id="ink:001",
            page=0,
            box=BBox(x0=0.1, y0=0.3, x1=0.6, y1=0.6),
            kind=InkRegionKind.WRITING,
            ink_ratio=0.2,
            pixel_count=40_000,
        )

    def test_a_drawing_question_accepts_a_region_with_no_text(self) -> None:
        # The case the ink pipeline exists for. A diagram has no text by nature,
        # so refusing it would leave every drawn answer unfindable.
        drawing = q("A/6", "6.", "Draw a labelled diagram of the eye.", 0, ["6"], marks=5)
        result = resolve(paper([drawing]), [self._ink_block()], [], [self._ink_region()])
        mapping = result.by_qid()["A/6"]

        assert mapping.status is AnswerStatus.OCR_FAILED
        assert mapping.highlight is not None
        assert mapping.highlight.derived_from == "ink_regions"

    def test_an_unreadable_region_is_not_handed_to_a_prose_question(self) -> None:
        # A region with no readable text says nothing about *which* question it
        # answers. Attached to one anyway it would vanish from the orphan list and
        # stop counting towards unassigned ink — and that total is what downgrades
        # absence claims elsewhere on the page. Reporting it honestly keeps both.
        result = resolve(paper([REFRACTION]), [self._ink_block()], [], [self._ink_region()])
        mapping = result.by_qid()["A/1"]

        assert [o.block_id for o in result.orphans] == ["blk:ink000"]
        assert mapping.status is not AnswerStatus.ANSWERED

    def test_and_it_is_never_reported_as_blank(self) -> None:
        # The claim that must not be made. There is writing on the page; we simply
        # could not read it or place it.
        result = resolve(paper([REFRACTION]), [self._ink_block()], [], [self._ink_region()])
        assert result.by_qid()["A/1"].status is not AnswerStatus.UNANSWERED

    def test_an_optional_question_may_be_skipped(self) -> None:
        # "Attempt any one" satisfied by one answer means the other is not an
        # omission. Reporting it as one is a product error a teacher would spot.
        optional = Section(
            section_id="B", label_raw="SECTION B", requirement=Requirement(answer_any=1)
        )
        first = q("B/4", "4.", "Describe an experiment about air pressure.", 0, ["4"],
                  section="B")
        second = q("B/5", "5.", "Discuss the behaviour of gases when heated.", 1, ["5"],
                   section="B")
        blocks = [block("blk:000", "An experiment about air pressure was performed.",
                        y0=0.10, line_ids=["as:0001"])]
        anchors = [anchor("anc:000", "4.", "B/4", "as:0001")]

        result = resolve(paper([first, second], [optional]), blocks, anchors, [])
        assert result.by_qid()["B/5"].status is AnswerStatus.NOT_REQUIRED

    def test_an_unmet_requirement_still_reports_omissions(self) -> None:
        # Below the quota the student genuinely owes answers, and calling them
        # optional would hide a real gap.
        optional = Section(
            section_id="B", label_raw="SECTION B", requirement=Requirement(answer_any=2)
        )
        first = q("B/4", "4.", "Describe an experiment about air pressure.", 0, ["4"],
                  section="B")
        second = q("B/5", "5.", "Discuss the behaviour of gases when heated.", 1, ["5"],
                   section="B")
        blocks = [block("blk:000", "An experiment about air pressure was performed.",
                        y0=0.10, line_ids=["as:0001"])]
        anchors = [anchor("anc:000", "4.", "B/4", "as:0001")]

        result = resolve(paper([first, second], [optional]), blocks, anchors, [])
        assert result.by_qid()["B/5"].status is not AnswerStatus.NOT_REQUIRED

    def test_a_continuation_marker_on_the_last_page_reports_pages_missing(self) -> None:
        questions = [REFRACTION, MOTOR]
        blocks = [
            block("blk:000", "Refraction is the bending of light, cont. on next page",
                  y0=0.85, line_ids=["as:0001"], continuation=True),
        ]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, [], pages_uploaded=1)

        assert result.by_qid()["A/3"].status is AnswerStatus.PAGES_MISSING


class TestAbsenceGuards:
    def test_unassigned_ink_suppresses_every_absence_claim(self) -> None:
        # Substantial writing belonging to no block means some answer went
        # unmapped, and the system is in no position to call anything blank.
        questions = [REFRACTION, MOTOR]
        blocks = [
            block("blk:000", "Refraction is the bending of light.", y0=0.10,
                  line_ids=["as:0001"]),
            block("blk:ink000", "", y0=0.50, line_ids=[], ink_ids=["ink:002"]),
        ]
        ink = [
            InkRegion(
                region_id="ink:002",
                page=0,
                box=BBox(x0=0.1, y0=0.5, x1=0.9, y1=0.9),
                kind=InkRegionKind.WRITING,
                ink_ratio=0.30,
                pixel_count=9000,
            )
        ]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, ink)

        if result.absence_claims_suppressed:
            assert all(
                m.status is not AnswerStatus.UNANSWERED
                for m in result.mappings
                if m.status is not AnswerStatus.ANSWERED
            )

    def test_bleed_through_does_not_suppress_absence_claims(self) -> None:
        # It appears on most double-sided scripts, so counting it would suppress
        # every legitimate unanswered report the product exists to make.
        questions = [REFRACTION, MOTOR]
        blocks = [block("blk:000", "Refraction is the bending of light.", y0=0.10,
                        line_ids=["as:0001"])]
        faint = [
            InkRegion(
                region_id="ink:009",
                page=0,
                box=BBox(x0=0.1, y0=0.5, x1=0.9, y1=0.9),
                kind=InkRegionKind.BLEED_THROUGH,
                ink_ratio=0.30,
                pixel_count=9000,
            )
        ]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, faint)

        assert result.unassigned_ink_ratio == 0.0
        assert not result.absence_claims_suppressed

    def test_a_plausible_answer_prevents_a_blank_claim(self) -> None:
        # Writing that looks like an answer to this question is on the sheet, even
        # though the aligner placed it elsewhere. "Found but unplaced" is honest;
        # "blank" is not.
        questions = [REFRACTION, REFLECTION]
        blocks = [
            block(
                "blk:000",
                "Refraction of light is the bending of light. "
                "Reflection of light obeys the laws of reflection.",
                y0=0.10,
                line_ids=["as:0001"],
            )
        ]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, [])

        assert result.by_qid()["A/2"].status is not AnswerStatus.UNANSWERED

    def test_an_answered_sibling_prevents_a_blank_claim(self) -> None:
        # Structural evidence, and it reaches where semantics cannot. A maths
        # answer like "R = V / I = 10 / 2 = 5 ohm" shares almost no word with its
        # question, so no similarity threshold rescues it — but a student who
        # answered 5 (a) very rarely leaves 5 (b) silently blank.
        first = q("B/5/a", "5 (a)", "State Ohm's law.", 0, ["5", "a"])
        second = q("B/5/b", "5 (b)", "A resistor carries 2 A at 10 V. Find its resistance.",
                   1, ["5", "b"])
        blocks = [
            block("blk:000", "Current is proportional to potential difference.", y0=0.10,
                  line_ids=["as:0001"])
        ]
        anchors = [anchor("anc:000", "5 (a)", "B/5/a", "as:0001")]
        result = resolve(paper([first, second]), blocks, anchors, [])

        assert result.by_qid()["B/5/b"].status is not AnswerStatus.UNANSWERED

    def test_evidence_of_an_answer_outranks_the_optional_rule(self) -> None:
        # Precedence, and getting it backwards filed an answered sub-part as
        # "not required" because its section's quota was already met. What the
        # student appears to have written outranks what the paper permitted.
        optional = Section(
            section_id="B", label_raw="SECTION B", requirement=Requirement(answer_any=1)
        )
        first = q("B/5/a", "5 (a)", "State Ohm's law.", 0, ["5", "a"], section="B")
        second = q("B/5/b", "5 (b)", "Find the resistance of the circuit.", 1, ["5", "b"],
                   section="B")
        blocks = [
            block("blk:000", "Current is proportional to potential difference.", y0=0.10,
                  line_ids=["as:0001"])
        ]
        anchors = [anchor("anc:000", "5 (a)", "B/5/a", "as:0001")]
        result = resolve(paper([first, second], [optional]), blocks, anchors, [])

        assert result.by_qid()["B/5/b"].status is AnswerStatus.UNCERTAIN


class TestReportShape:
    def test_every_question_gets_a_mapping(self) -> None:
        # The teacher's list must be complete, or a question silently vanishes.
        questions = [REFRACTION, REFLECTION, MOTOR]
        result = resolve(paper(questions), [], [], [])
        assert {m.qid for m in result.mappings} == {qq.qid for qq in questions}

    def test_status_counts_are_reported(self) -> None:
        questions = [REFRACTION, MOTOR]
        blocks = [block("blk:000", "Refraction is the bending of light.", y0=0.10,
                        line_ids=["as:0001"])]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        counts = resolve(paper(questions), blocks, anchors, []).counts_by_status()
        assert sum(counts.values()) == 2

    def test_an_empty_sheet_produces_no_answers_and_no_crash(self) -> None:
        result = resolve(paper([REFRACTION]), [], [], [])
        assert result.by_qid()["A/1"].status is not AnswerStatus.ANSWERED
        assert result.orphans == []


class TestTeacherCorrections:
    """``reassign`` — the path a teacher's correction takes.

    Worth testing directly because it is the one place where a human overrules
    the aligner, and because its failure mode is silent: a correction that drops
    writing leaves the teacher with a script they can no longer fully see.
    """

    PAPER = paper([REFRACTION, REFLECTION, MOTOR])

    def _resolved(self, blocks):
        return blocks, resolve(self.PAPER, blocks, [], [])

    def test_a_moved_block_lands_on_the_chosen_question(self) -> None:
        from grader.align import reassign

        blocks = [
            block("blk:000", "Light bends when it changes medium.", y0=0.10),
            block("blk:001", "Angle of incidence equals angle of reflection.", y0=0.30),
        ]
        result = resolve(self.PAPER, blocks, [], [])
        moved = reassign(
            self.PAPER, blocks, result, block_id="blk:001", to_qid="A/3"
        )

        motor = next(m for m in moved.mappings if m.qid == "A/3")
        assert motor.block_ids == ["blk:001"]
        assert motor.status is AnswerStatus.ANSWERED
        assert motor.teacher_override is True
        assert motor.highlight is not None

    def test_the_question_losing_a_block_is_never_called_blank(self) -> None:
        # A teacher moving an answer says nothing about whether the original
        # question was attempted. Asserting a blank on the strength of a
        # correction elsewhere is the unfounded absence claim to avoid.
        from grader.align import reassign

        blocks = [block("blk:000", "Light bends when it changes medium.", y0=0.10)]
        result = resolve(self.PAPER, blocks, [], [])
        moved = reassign(
            self.PAPER, blocks, result, block_id="blk:000", to_qid="A/3"
        )

        loser = next(m for m in moved.mappings if m.qid == "A/1")
        assert loser.block_ids == []
        assert loser.status is AnswerStatus.UNCERTAIN
        assert loser.highlight is None

    def test_moving_onto_an_answered_question_keeps_both_blocks(self) -> None:
        # The correction that replace-semantics made impossible: an answer split
        # across two blocks, one of which the aligner gave to the neighbour.
        # Moving it back must leave the question holding both, or restoring the
        # first block would displace the second and the split could never be
        # repaired.
        from grader.align import reassign

        blocks = [
            block("blk:000", "A coil carrying current sits in a magnetic field.", y0=0.10),
            block("blk:001", "The force on it makes the coil rotate.", y0=0.30),
        ]
        result = resolve(self.PAPER, blocks, [], [])
        moved = reassign(self.PAPER, blocks, result, block_id="blk:000", to_qid="A/3")
        moved = reassign(self.PAPER, blocks, moved, block_id="blk:001", to_qid="A/3")

        motor = next(m for m in moved.mappings if m.qid == "A/3")
        assert motor.block_ids == ["blk:000", "blk:001"]
        assert motor.highlight is not None
        assert len(motor.highlight.boxes) >= 1

    def test_merged_blocks_stay_in_document_order(self) -> None:
        # start_line_id and end_line_id name a span, so the order the teacher
        # happened to click in must not decide which end is which.
        from grader.align import reassign

        blocks = [
            block("blk:000", "First part of the answer.", y0=0.10, line_ids=["as:0001"]),
            block("blk:001", "Second part of the answer.", y0=0.30, line_ids=["as:0002"]),
        ]
        result = resolve(self.PAPER, blocks, [], [])
        moved = reassign(self.PAPER, blocks, result, block_id="blk:001", to_qid="A/3")
        moved = reassign(self.PAPER, blocks, moved, block_id="blk:000", to_qid="A/3")

        motor = next(m for m in moved.mappings if m.qid == "A/3")
        assert motor.block_ids == ["blk:000", "blk:001"]
        assert motor.start_line_id == "as:0001"
        assert motor.end_line_id == "as:0002"

    def test_displaced_writing_is_never_lost(self) -> None:
        # Every block must remain reachable after a correction: owned by a
        # question, or listed as an orphan. Writing that is neither is invisible
        # to the teacher.
        from grader.align import reassign

        blocks = [
            block("blk:000", "Light bends when it changes medium.", y0=0.10),
            block("blk:001", "Angle of incidence equals angle of reflection.", y0=0.30),
            block("blk:002", "A coil rotates in a magnetic field.", y0=0.50),
        ]
        result = resolve(self.PAPER, blocks, [], [])
        moved = reassign(self.PAPER, blocks, result, block_id="blk:002", to_qid="A/1")

        reachable = {bid for m in moved.mappings for bid in m.block_ids}
        reachable |= {o.block_id for o in moved.orphans}
        assert reachable == {"blk:000", "blk:001", "blk:002"}

    def test_an_unknown_block_or_question_changes_nothing(self) -> None:
        from grader.align import reassign

        blocks = [block("blk:000", "Light bends when it changes medium.", y0=0.10)]
        result = resolve(self.PAPER, blocks, [], [])

        assert reassign(self.PAPER, blocks, result, block_id="blk:999", to_qid="A/1") is result
        assert reassign(self.PAPER, blocks, result, block_id="blk:000", to_qid="Z/9") is result
