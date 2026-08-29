"""Tests for question extraction.

Extraction accuracy is the first graded criterion, and the failure modes are
asymmetric: a missed question does not exist and cannot be answered, while a
spurious one is visible and dismissible. The tests below lean accordingly —
several assert that plausible-looking non-questions are rejected, and several
assert that structures which look like noise are kept.
"""

from __future__ import annotations

import pytest
from vedaai_contracts import BBox, DocumentKind, Line, LineRole, OcrEngine, Question

from grader.questions import furniture, numbering, optionality, validate
from grader.questions.extract import extract, mark_stems, reads_as_a_heading
from grader.reading_order import order_lines


def line(
    index: int,
    text: str,
    *,
    x0: float = 0.09,
    y0: float,
    page: int = 0,
    width: float = 0.55,
) -> Line:
    return Line(
        line_id=f"qp:{index:04d}",
        kind=DocumentKind.QUESTION_PAPER,
        page=page,
        box=BBox(x0=x0, y0=y0, x1=min(1.0, x0 + width), y1=y0 + 0.018),
        text=text,
        confidence=1.0,
        engine=OcrEngine.PDF_TEXT_LAYER,
    )


def index_of(*texts_and_indents: tuple[str, float]):
    """Build a single-page LineIndex from (text, indent) pairs."""
    from vedaai_contracts import LineIndex

    lines = [
        line(i + 1, text, x0=indent, y0=0.05 + i * 0.03)
        for i, (text, indent) in enumerate(texts_and_indents)
    ]
    return LineIndex(
        kind=DocumentKind.QUESTION_PAPER,
        lines=lines,
        engine=OcrEngine.PDF_TEXT_LAYER,
    )


class TestLabelParsing:
    @pytest.mark.parametrize(
        ("text", "tokens", "raw"),
        [
            ("1. Define refraction.", ("1",), "1."),
            ("2) State the law.", ("2",), "2)"),
            ("11 (a) Draw a diagram.", ("11", "a"), "11 (a)"),
            ("11(a) Draw a diagram.", ("11", "a"), "11(a)"),
            ("2 (i) (a) Draw it.", ("2", "i", "a"), "2 (i) (a)"),
            ("Q.5 Explain the working.", ("5",), "Q.5"),
            ("Q5 Explain the working.", ("5",), "Q5"),
            ("(i) State the laws.", ("i",), "(i)"),
            ("(iii) Heat and temperature.", ("iii",), "(iii)"),
            ("(II.) Roman, upper case.", ("II",), "(II.)"),
            ("a) A sub-part without a bracket.", ("a",), "a)"),
        ],
    )
    def test_parses_the_notations_real_papers_use(self, text, tokens, raw) -> None:
        parsed = numbering.parse_label(text)
        assert parsed is not None, f"failed to parse {text!r}"
        assert parsed.tokens == tokens
        assert parsed.raw == raw

    @pytest.mark.parametrize(
        "text",
        [
            "In 1947 India became independent.",
            "1.5 kg of copper was used in the experiment.",
            "Quote the formula for kinetic energy.",
            "The value was 3 times larger.",
            "Water boils at 100 degrees.",
        ],
    )
    def test_rejects_text_that_merely_resembles_a_label(self, text) -> None:
        # Each of these broke an earlier version. A sentence opening with a year
        # became question 1947; a decimal became question 1 answered by "5 kg".
        assert numbering.parse_label(text) is None, f"{text!r} was wrongly parsed"

    def test_preserves_the_printed_notation_verbatim(self) -> None:
        # The requirement is to preserve original numbering, so the label is kept
        # as printed rather than canonicalized.
        parsed = numbering.parse_label("11  (a)   Draw a diagram.")
        assert parsed is not None
        assert parsed.raw == "11 (a)"

    def test_does_not_interpret_roman_versus_letter(self) -> None:
        # The ambiguity the parser deliberately refuses to resolve: "(i)" is both
        # roman one and the letter i, and only context distinguishes them.
        # Nothing downstream needs the answer, because ordering comes from
        # position on the page.
        assert numbering.parse_label("(i) x").tokens == ("i",)
        assert numbering.parse_label("(h) x").tokens == ("h",)


class TestLabelPrefixes:
    """Notations beyond a bare number, taken from real papers."""

    @pytest.mark.parametrize(
        ("text", "tokens"),
        [
            ("Question 4", ("4",)),
            ("Question 12. Describe the carbon cycle.", ("12",)),
            ("Q.8 Explain osmosis.", ("8",)),
            ("13 [4 marks]", ("13",)),
        ],
    )
    def test_parses_prefixes_real_papers_use(self, text, tokens) -> None:
        parsed = numbering.parse_label(text)
        assert parsed is not None, f"{text!r} should parse"
        assert parsed.tokens == tokens

    def test_a_word_beginning_with_q_is_not_a_question_label(self) -> None:
        # "Quote the formula" must not become question "uote", and the guard has
        # to survive spelling out "Question".
        for text in ("Quote the source.", "Questions like this are common."):
            assert numbering.parse_label(text) is None

    def test_learns_a_section_letter_the_paper_uses_consistently(self) -> None:
        """`T1`..`T5` alongside `Q1`..`Q4`, from a real mathematics paper.

        Hardcoding a list of letters would be guessing at what a school prints.
        What can be observed instead is repetition: a paper that labels with `T`
        does it several times and counts upward, and one stray "A4 paper" in a
        sentence does not.
        """
        lines = [
            "Q1 (5 Marks)", "Q2 (5 Marks)",
            "T1 (5 Marks)", "T2 (5 Marks)", "T3 (5 Marks)",
        ]
        prefixes = numbering.detect_section_prefixes(lines)
        assert "T" in prefixes

        parsed = numbering.parse_label("T2 (5 Marks)", prefixes=prefixes)
        assert parsed is not None
        assert parsed.tokens == ("2",)

    def test_does_not_invent_a_prefix_from_one_mention(self) -> None:
        lines = [
            "1. Fold the A4 sheet in half.",
            "2. Describe the reaction.",
            "3. State the law.",
        ]
        assert numbering.detect_section_prefixes(lines) == frozenset()


class TestMarks:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Define refraction.  [2]", 2),
            ("Define refraction. (5)", 5),
            ("Define refraction. [5 marks]", 5),
            ("Define refraction. [12 Marks]", 12),
        ],
    )
    def test_captures_printed_marks(self, text, expected) -> None:
        body, marks = numbering.extract_marks(text)
        assert marks == expected
        assert "[" not in body and "(" not in body

    def test_does_not_treat_a_mid_sentence_bracket_as_marks(self) -> None:
        # A chemistry paper writes [Fe(H2O)6]3+ and a maths paper writes f(3).
        # Anchoring to the end of the line is what keeps those intact.
        body, marks = numbering.extract_marks("Name the complex [Fe(H2O)6]3+")
        assert marks is None
        assert body == "Name the complex [Fe(H2O)6]3+"


class TestSectionLevelMarks:
    """Marks stated once for a whole section belong to every question in it.

    A real Class 9 science paper prints "(Each question carries 1 mark)" under
    SECTION A and nothing beside questions 1 to 6, which is how most papers are
    laid out: repeating "[1]" six times is noise a teacher does not write. The
    marks reached no question, so six answered questions were graded out of
    nothing and displayed to the teacher as "0/0".

    Two separate faults produced that. The line was classified as furniture and
    discarded — SECTION C's directive survived only because it happens to also
    say "attempt any three", which matches the instruction phrase list — and even
    the surviving one was read for its choice rule and not for its marks.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("(Each question carries 1 mark)", 1),
            ("(Each question carries 3 marks)", 3),
            ("(Each question carries 5 marks. Attempt any three questions.)", 5),
            ("Each question carries 2 marks.", 2),
            ("Each carries 4 marks", 4),
            ("Every question is of 3 marks", 3),
            ("Each question carry 2 marks", 2),  # printed on real papers
            ("Answer any five questions of 6 marks each.", 6),
            ("Questions 1 to 10 carry 1 mark each.", 1),
        ],
    )
    def test_reads_a_per_question_allocation(self, text, expected) -> None:
        assert numbering.per_question_marks(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Maximum Marks: 70",
            "Each section carries 20 marks",  # the section total, not per question
            "This paper carries 80 marks",
            "All questions carry equal marks",  # true, and says nothing about how many
            "Draw neat diagrams wherever necessary.",
            "Define refraction of light. [2]",
        ],
    )
    def test_does_not_invent_one(self, text) -> None:
        assert numbering.per_question_marks(text) is None

    def test_the_directive_is_kept_rather_than_dismissed_as_an_aside(self) -> None:
        # Bracketed asides are furniture by default, which is why this line was
        # being thrown away before it could be read.
        role = furniture.classify(
            line(1, "(Each question carries 1 mark)", y0=0.1),
            repeated=set(),
            previous_role=LineRole.SECTION_HEADER,
        )
        assert role is LineRole.INSTRUCTION

    def test_every_question_in_the_section_gets_them(self) -> None:
        paper = extract(
            index_of(
                ("SECTION A", 0.09),
                ("(Each question carries 1 mark)", 0.09),
                ("1. Define refraction of light.", 0.09),
                ("2. State the SI unit of pressure.", 0.09),
                ("SECTION B", 0.09),
                ("(Each question carries 3 marks)", 0.09),
                ("7. Explain why a pencil appears bent in water.", 0.09),
            )
        )
        assert {q.qid: q.marks for q in paper.questions} == {
            "A/1": 1,
            "A/2": 1,
            "B/7": 3,
        }

    def test_a_printed_mark_beside_the_question_still_wins(self) -> None:
        # The section states a default; the question states a fact.
        paper = extract(
            index_of(
                ("SECTION A", 0.09),
                ("(Each question carries 1 mark)", 0.09),
                ("1. Define refraction of light.", 0.09),
                ("2. State the SI unit of pressure. [4]", 0.09),
            )
        )
        assert {q.qid: q.marks for q in paper.questions} == {"A/1": 1, "A/2": 4}

    def test_sub_parts_divide_the_question_rather_than_each_taking_it_whole(self) -> None:
        # "Each question carries 4 marks" is about question 11, not about 11(a)
        # and 11(b) separately; giving both 4 would double the paper's total.
        paper = extract(
            index_of(
                ("SECTION B", 0.09),
                ("(Each question carries 4 marks)", 0.09),
                ("11 (a) Define atomic number.", 0.09),
                ("11 (b) Define mass number.", 0.09),
            )
        )
        assert {q.qid: q.marks for q in paper.questions} == {"B/11/a": 2, "B/11/b": 2}

    def test_a_sub_part_with_printed_marks_leaves_the_remainder_to_its_sibling(self) -> None:
        paper = extract(
            index_of(
                ("SECTION C", 0.09),
                ("(Each question carries 5 marks)", 0.09),
                ("14. (i) State the law of conservation of mass. [2]", 0.09),
                ("14. (ii) Find the mass of carbon dioxide formed.", 0.09),
            )
        )
        assert {q.qid: q.marks for q in paper.questions} == {"C/14/i": 2, "C/14/ii": 3}

    def test_an_uneven_split_is_declined_rather_than_guessed(self) -> None:
        # Three sub-parts sharing five marks could be 2/2/1 or 1/2/2. Inventing
        # one produces a denominator a teacher cannot check, so nothing is set.
        paper = extract(
            index_of(
                ("SECTION C", 0.09),
                ("(Each question carries 5 marks)", 0.09),
                ("13. (i) Name the process.", 0.09),
                ("13. (ii) Describe the process.", 0.09),
                ("13. (iii) Explain one use of it.", 0.09),
            )
        )
        assert all(q.marks is None for q in paper.questions)

    def test_a_paper_with_no_sections_takes_the_rule_from_its_cover(self) -> None:
        paper = extract(
            index_of(
                ("Annual Examination 2026", 0.09),
                ("Each question carries 2 marks.", 0.09),
                ("1. Define refraction of light.", 0.09),
                ("2. State the SI unit of pressure.", 0.09),
            )
        )
        assert [q.marks for q in paper.questions] == [2, 2]

    def test_a_section_rule_beats_the_cover_page(self) -> None:
        paper = extract(
            index_of(
                ("Annual Examination 2026", 0.09),
                ("Each question carries 2 marks.", 0.09),
                ("SECTION B", 0.09),
                ("(Each question carries 3 marks)", 0.09),
                ("7. Explain why a pencil appears bent in water.", 0.09),
            )
        )
        assert [q.marks for q in paper.questions] == [3]


class TestFurniture:
    def test_recognizes_section_headers(self) -> None:
        for text in ("SECTION A", "Section B", "PART C", "MODULE 2"):
            assert furniture.section_label(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "Attempt all questions from Section A and any four from Section B",
            "Answer any FIVE full questions",
            "The marks for questions are shown in brackets.",
            "Time allowed: 1 hour     Maximum Marks: 40",
            "Use black ink or black ball-point pen. Do not use pencil.",
            "You may ask for graph paper, tracing paper and more answer paper.",
            "There is no overall choice. However, an internal choice has been provided.",
        ],
    )
    def test_classifies_rubric_as_instruction(self, text) -> None:
        # Quoted from official papers. Swallowing any of these into a question
        # would corrupt the text sent for grading.
        role = furniture.classify(
            line(1, text, y0=0.1), repeated=set(), previous_role=None
        )
        assert role is LineRole.INSTRUCTION

    def test_a_label_alone_on_its_line_still_opens_a_question(self) -> None:
        """The heading layout, which is most of a real paper's ways of numbering.

        `Q1 (5 Marks)` on its own line with the question below it is how the paper
        that extracted one question out of nine was set. Eight of the eleven
        real-world label styles this extractor dropped were dropped for this one
        reason: the body is empty once the mark allocation is taken off it, and an
        empty body was read as "a stray number, not a question".
        """
        lines = [
            line(1, "Q1 (5 Marks)", y0=0.05),
            line(2, "Two taps A and B fill a tank. Find the time each takes alone.", y0=0.09),
            line(3, "Q2 (5 Marks)", y0=0.15),
            line(4, "A father and son have x and y coins. Find x and y.", y0=0.19),
        ]
        roles = furniture.classify_all(lines)
        assert roles["qp:0001"] is LineRole.QUESTION_START
        assert roles["qp:0002"] is LineRole.QUESTION_CONTINUATION
        assert roles["qp:0003"] is LineRole.QUESTION_START

    def test_a_bare_label_with_nothing_after_it_is_still_not_a_question(self) -> None:
        """The rule the heading layout must not undo.

        A number alone at the foot of a page, with nothing following it, is a page
        number or a stray. Only a label followed by actual question text counts.
        """
        lines = [
            line(1, "1. Define refraction of light.", y0=0.05),
            line(2, "7", y0=0.95),
        ]
        roles = furniture.classify_all(lines)
        assert roles["qp:0002"] is not LineRole.QUESTION_START

    def test_an_enumerated_instruction_block_is_not_a_list_of_questions(self) -> None:
        """Lettered instructions under a heading, before any question exists.

        Taken verbatim from a paper that produced two phantom questions: the
        teacher's review screen opened with "(a) All questions are compulsory" and
        "(c) Draw neat diagrams", both shown as *answered*, because a lettered line
        parses as a label and nothing said we were still in the preamble. Only
        "(b)" escaped, and only because it happened to contain the words "attempt
        any" — which is matching on vocabulary and hoping.

        Position is the reliable signal: nothing before the first question or
        section header can be a question.
        """
        lines = [
            line(1, "Greenfield Public School", y0=0.04),
            line(2, "General Instructions:", y0=0.09),
            line(3, "(a) All questions are compulsory except where stated otherwise.", y0=0.13),
            line(4, "(b) In Section C, attempt any three of the four questions.", y0=0.17),
            line(5, "(c) Draw neat diagrams wherever necessary.", y0=0.21),
            line(6, "SECTION A", y0=0.27),
            line(7, "1. Define refraction of light.", y0=0.32),
            line(8, "2. State the SI unit of pressure.", y0=0.37),
        ]
        roles = furniture.classify_all(lines)

        for preamble in ("qp:0003", "qp:0004", "qp:0005"):
            assert roles[preamble] is not LineRole.QUESTION_START, (
                f"{preamble} is an instruction, not a question"
            )
        assert roles["qp:0007"] is LineRole.QUESTION_START
        assert roles["qp:0008"] is LineRole.QUESTION_START

    def test_a_lettered_sub_part_after_a_question_is_still_a_question(self) -> None:
        """The rule must not swallow real sub-parts.

        "(a)" and "(b)" under question 11 look identical to the instruction case
        character for character. What separates them is that a question has already
        been seen by the time they appear.
        """
        lines = [
            line(1, "SECTION B", y0=0.05),
            line(2, "11. Answer both parts.", y0=0.10),
            line(3, "(a) Define atomic number and mass number.", y0=0.15),
            line(4, "(b) An atom has 11 protons and 12 neutrons. Give its mass number.", y0=0.20),
        ]
        roles = furniture.classify_all(lines)
        assert roles["qp:0003"] is LineRole.QUESTION_START
        assert roles["qp:0004"] is LineRole.QUESTION_START

    def test_classifies_a_competency_tag_as_furniture(self) -> None:
        # ICSE prints these against questions; they are metadata, not content.
        role = furniture.classify(
            line(1, "[Analysis & Evaluation]", y0=0.1), repeated=set(), previous_role=None
        )
        assert role is LineRole.FURNITURE

    def test_classifies_a_lone_mark_allocation(self) -> None:
        role = furniture.classify(line(1, "[5]", y0=0.1), repeated=set(), previous_role=None)
        assert role is LineRole.MARKS

    def test_classifies_a_page_number_as_furniture(self) -> None:
        for text in ("3", "Page 3 of 12", "12"):
            role = furniture.classify(
                line(1, text, y0=0.95), repeated=set(), previous_role=None
            )
            assert role is LineRole.FURNITURE, text

    def test_an_unlabelled_line_continues_the_previous_question(self) -> None:
        role = furniture.classify(
            line(1, "and state the units of each quantity.", y0=0.2),
            repeated=set(),
            previous_role=LineRole.QUESTION_START,
        )
        assert role is LineRole.QUESTION_CONTINUATION

    def test_an_unlabelled_line_after_nothing_is_furniture(self) -> None:
        role = furniture.classify(
            line(1, "Candidate name .....................", y0=0.05),
            repeated=set(),
            previous_role=None,
        )
        assert role is LineRole.FURNITURE

    def test_a_two_page_paper_header_is_detected_at_the_page_edge(self) -> None:
        # Two pages cannot show three-page repetition, and a real paper proved
        # the cost of waiting for it: "SCIENCE - UNIT TEST (page 2)" was absorbed
        # into the last question on page 1. Position supplies the evidence
        # repetition alone cannot at this length.
        two_pages = [
            line(1, "SCIENCE UNIT TEST", y0=0.02, page=0),
            line(2, "SCIENCE UNIT TEST", y0=0.02, page=1),
        ]
        assert len(furniture.find_repeated_lines(two_pages)) == 2

    def test_a_two_page_repeat_in_the_body_is_not_a_header(self) -> None:
        # The other half of the same rule. Mid-page text repeated on both pages
        # of a two-page paper is as likely a genuine repeat as furniture, and
        # discarding a real question line is the more expensive mistake.
        two_pages = [
            line(1, "Draw a labelled diagram.", y0=0.45, page=0),
            line(2, "Draw a labelled diagram.", y0=0.45, page=1),
        ]
        assert furniture.find_repeated_lines(two_pages) == set()

    def test_a_single_page_paper_has_no_headers_to_find(self) -> None:
        one_page = [line(1, "SCIENCE UNIT TEST", y0=0.02, page=0)]
        assert furniture.find_repeated_lines(one_page) == set()

    def test_a_header_carrying_its_own_page_number_is_furniture(self) -> None:
        # The header shape repetition can never catch: it differs on every page,
        # so bucketing by text never groups it. Observed on a real paper, where it
        # was appended to the text of the last question on the preceding page.
        role = furniture.classify(
            line(1, "SCIENCE — UNIT TEST (page 2)", y0=0.03, page=1),
            repeated=set(),
            previous_role=LineRole.QUESTION_START,
        )
        assert role is LineRole.FURNITURE

    @pytest.mark.parametrize(
        "text",
        [
            "Page 2 of 12",
            "Page 2   Physics Higher Tier",
            "SCIENCE UNIT TEST — Page 2",
            "Physics Higher Tier (page 3 of 16)",
        ],
    )
    def test_recognizes_the_page_marker_shapes_papers_print(self, text) -> None:
        role = furniture.classify(
            line(1, text, y0=0.03, page=1), repeated=set(), previous_role=None
        )
        assert role is LineRole.FURNITURE, text

    def test_a_mid_sentence_reference_to_another_page_is_not_a_header(self) -> None:
        # "Refer to the graph on page 2" is question text. The marker rule is
        # anchored to a delimiter at one end of the line for exactly this case.
        role = furniture.classify(
            line(1, "Refer to the graph on page 2 and answer the following.", y0=0.4),
            repeated=set(),
            previous_role=LineRole.QUESTION_START,
        )
        assert role is LineRole.QUESTION_CONTINUATION

    def test_a_numbered_question_opening_a_page_survives_the_marker_rule(self) -> None:
        # Ordering guard: the label test runs first, so a question is never
        # discarded for sitting where a header would.
        role = furniture.classify(
            line(1, "8. How many pages 2 sheets make?", y0=0.03, page=1),
            repeated=set(),
            previous_role=None,
        )
        assert role is LineRole.QUESTION_START

    def test_a_running_header_across_four_pages_is_furniture(self) -> None:
        lines = [
            line(i + 1, "SCIENCE UNIT TEST", y0=0.02, page=i) for i in range(4)
        ]
        assert len(furniture.find_repeated_lines(lines)) == 4

    def test_repeated_text_at_different_heights_is_not_a_header(self) -> None:
        # "Answer all questions" may legitimately appear once per section in the
        # body. Position is what separates a header from a genuine repeat.
        lines = [
            line(1, "Answer all questions", y0=0.10, page=0),
            line(2, "Answer all questions", y0=0.55, page=1),
            line(3, "Answer all questions", y0=0.80, page=2),
        ]
        assert furniture.find_repeated_lines(lines) == set()


class TestOptionality:
    @pytest.mark.parametrize(
        ("text", "count"),
        [
            ("Attempt any two questions from this Section", 2),
            ("Answer any FIVE full questions", 5),
            ("any 4 questions", 4),
            ("Answer any three of the following", 3),
        ],
    )
    def test_reads_a_choice_count(self, text, count) -> None:
        assert optionality.parse_count(text) == count

    def test_reads_a_two_part_instruction(self) -> None:
        # ICSE: "Attempt all questions from Section A and any four questions from
        # Section B". Without splitting on "and", Section B silently becomes
        # compulsory and a legitimately skipped question reads as an omission.
        rules = optionality.parse_all(
            ["Attempt all questions from Section A and any four questions from Section B"]
        )
        assert rules["A"].answer_any is None
        assert rules["B"].answer_any == 4

    def test_all_required_when_no_choice_is_offered(self) -> None:
        rules = optionality.parse_all(["There is no overall choice."])
        assert rules[None].answer_any is None

    def test_a_line_with_no_rule_yields_nothing(self) -> None:
        assert optionality.parse_all(["Use black ink or ball-point pen."]) == {}

    def test_satisfaction(self) -> None:
        from vedaai_contracts import Requirement

        assert optionality.satisfied(Requirement(answer_any=2), 2)
        assert not optionality.satisfied(Requirement(answer_any=4), 3)
        assert optionality.satisfied(Requirement(), 0), "compulsory has nothing to satisfy early"


class TestReadingOrder:
    def test_a_single_column_page_reads_top_to_bottom(self) -> None:
        lines = [line(i + 1, f"line {i}", y0=0.1 + i * 0.05) for i in range(6)]
        ordered, confidence = order_lines(lines)
        assert [ln.text for ln in ordered] == [f"line {i}" for i in range(6)]
        assert confidence == 1.0

    def test_two_columns_are_read_one_after_the_other(self) -> None:
        # The Phase 1 limitation, now fixed. A y-sort alternates between columns
        # and produces a question list that is complete and useless.
        lines: list[Line] = []
        index = 1
        for k in range(5):
            y = 0.10 + k * 0.06
            lines.append(line(index, f"L{k}", x0=0.05, y0=y, width=0.38))
            index += 1
            lines.append(line(index, f"R{k}", x0=0.55, y0=y, width=0.38))
            index += 1

        ordered, confidence = order_lines(lines)
        texts = [ln.text for ln in ordered]
        assert texts == ["L0", "L1", "L2", "L3", "L4", "R0", "R1", "R2", "R3", "R4"]
        assert confidence < 1.0, "a multi-column page should report less certainty"

    def test_a_full_width_heading_keeps_its_place(self) -> None:
        # A title spans both columns, so it is a boundary rather than a member of
        # either. Without banding it sorts into whichever column its left edge
        # falls in, and a mid-page section header lands in the wrong place.
        title = line(1, "SECTION B" + " " * 40, x0=0.05, y0=0.05)
        title = title.model_copy(
            update={"box": BBox(x0=0.05, y0=0.05, x1=0.95, y1=0.07)}
        )
        lines = [title]
        index = 2
        for k in range(4):
            y = 0.15 + k * 0.06
            lines.append(line(index, f"L{k}", x0=0.05, y0=y, width=0.38))
            index += 1
            lines.append(line(index, f"R{k}", x0=0.55, y0=y, width=0.38))
            index += 1

        ordered, _confidence = order_lines(lines)
        assert ordered[0].text.strip() == "SECTION B"

    def test_a_few_strays_do_not_become_a_column(self) -> None:
        # Marginal annotations sit at one side of the page. Treating two of them
        # as a column would reorder the whole page around them.
        lines = [line(i + 1, f"body {i}", x0=0.09, y0=0.1 + i * 0.05) for i in range(8)]
        lines.append(line(99, "note", x0=0.88, y0=0.2))
        ordered, _confidence = order_lines(lines)
        assert len(ordered) == 9

    def test_a_margin_question_number_reads_before_its_own_line(self) -> None:
        """The answer-sheet case, and the reason every answer picked up its
        neighbour's words.

        Geometry copied from a real script: the number sits a fraction *below* the
        top of the line it labels, because a hand does not write a number on the
        same baseline as the text beside it. Sorting by top edge therefore emits
        the text first, segmentation starts a block at the number, and the first
        line of every answer is left attached to the answer above.

        The number and its line share a row. That is the fact the ordering has to
        respect, and it holds whichever of the two happens to sit a pixel higher.
        """
        lines: list[Line] = []
        index = 1
        for k in range(4):
            y = 0.09 + k * 0.12
            # Text first in input order, and marginally higher — exactly the
            # arrangement that produced 0 of 12 correct bindings.
            lines.append(line(index, f"answer {k} first line", x0=0.115, y0=y, width=0.44))
            index += 1
            lines.append(line(index, f"{k + 1}.", x0=0.049, y0=y + 0.0016, width=0.017))
            index += 1
            lines.append(line(index, f"answer {k} second line", x0=0.115, y0=y + 0.03, width=0.44))
            index += 1

        ordered, _confidence = order_lines(lines)
        texts = [ln.text for ln in ordered]

        assert texts == [
            "1.", "answer 0 first line", "answer 0 second line",
            "2.", "answer 1 first line", "answer 1 second line",
            "3.", "answer 2 first line", "answer 2 second line",
            "4.", "answer 3 first line", "answer 3 second line",
        ], "each margin number must lead the answer it labels"

    def test_margin_numbers_are_not_mistaken_for_a_column(self) -> None:
        """A strip of numbers down the left edge is not a column of text.

        There is a real gutter between them and the writing, wide enough that
        projection finds it, so a column detector reading the page structurally
        would put every number first and every answer after — which is a worse
        ordering than the one being fixed.
        """
        lines: list[Line] = []
        index = 1
        for k in range(6):
            y = 0.08 + k * 0.09
            lines.append(line(index, f"{k + 1}.", x0=0.049, y0=y + 0.001, width=0.017))
            index += 1
            lines.append(line(index, f"the answer to question {k + 1}", x0=0.115, y0=y, width=0.46))
            index += 1

        ordered, _confidence = order_lines(lines)
        texts = [ln.text for ln in ordered]
        assert texts[:4] == ["1.", "the answer to question 1", "2.", "the answer to question 2"]


class TestExtraction:
    def test_extracts_a_flat_paper(self) -> None:
        index = index_of(
            ("SECTION A", 0.09),
            ("(Attempt all questions from this Section)", 0.09),
            ("1. Define refraction of light.  [2]", 0.09),
            ("2. State the laws of reflection.  [3]", 0.09),
        )
        paper = extract(index)

        assert [q.label_raw for q in paper.in_print_order()] == ["1.", "2."]
        assert [q.marks for q in paper.in_print_order()] == [2, 3]
        assert paper.questions[0].section_id == "A"
        assert paper.total_marks == 5

    def test_strips_marks_and_furniture_from_question_text(self) -> None:
        index = index_of(
            ("1. Explain the working of an electric motor.  [5]", 0.09),
            ("[Analysis & Evaluation]", 0.12),
        )
        paper = extract(index)
        assert len(paper.questions) == 1
        assert paper.questions[0].text == "Explain the working of an electric motor."
        assert paper.questions[0].marks == 5

    def test_a_continuation_line_joins_its_question(self) -> None:
        index = index_of(
            ("1. State the laws of reflection and", 0.09),
            ("give the units of each quantity.  [3]", 0.09),
        )
        paper = extract(index)
        assert len(paper.questions) == 1
        assert "units of each quantity" in paper.questions[0].text

    def test_builds_a_nested_hierarchy_from_indentation(self) -> None:
        index = index_of(
            ("2. Answer the following:", 0.09),
            ("(i) State the laws of reflection.  [2]", 0.13),
            ("(a) Draw a labelled ray diagram.  [3]", 0.17),
            ("(ii) What is the SI unit of power?  [1]", 0.13),
        )
        paper = extract(index)
        paths = [tuple(q.path) for q in paper.in_print_order()]

        assert paths == [("2",), ("2", "i"), ("2", "i", "a"), ("2", "ii")]

    def test_sub_parts_are_separate_questions(self) -> None:
        # An explicit requirement: "11 (a)" and "11 (b)" are two entries.
        index = index_of(
            ("11 (a) What is meant by an echo?  [2]", 0.09),
            ("(b) State two conditions for hearing one.  [3]", 0.13),
        )
        paper = extract(index)
        assert len(paper.questions) == 2
        assert [q.label_raw for q in paper.in_print_order()] == ["11 (a)", "(b)"]
        assert [tuple(q.path) for q in paper.in_print_order()] == [("11", "a"), ("11", "b")]

    def test_section_namespacing_prevents_a_collision(self) -> None:
        # Some boards restart numbering per section, so an unnamespaced "1" would
        # collide and two different questions would share an identity.
        index = index_of(
            ("SECTION A", 0.09),
            ("1. First question of A.", 0.09),
            ("SECTION B", 0.09),
            ("1. First question of B.", 0.09),
        )
        paper = extract(index)
        qids = [q.qid for q in paper.in_print_order()]
        assert qids == ["A/1", "B/1"]
        assert len(set(qids)) == 2

    def test_print_order_is_the_ordering_authority(self) -> None:
        # Labels cannot order anything reliably: they restart, they mix romans
        # with letters, and "(ii)" sorts before "(i)" as a string.
        index = index_of(
            ("SECTION A", 0.09),
            ("1. One.", 0.09),
            ("SECTION B", 0.09),
            ("1. Also one.", 0.09),
            ("2. Two.", 0.09),
        )
        paper = extract(index)
        orders = [q.print_order for q in paper.questions]
        assert orders == sorted(orders)
        assert len(set(orders)) == len(orders)

    def test_attaches_choice_rules_to_their_sections(self) -> None:
        index = index_of(
            ("Attempt all questions from Section A and any two questions from Section B", 0.09),
            ("SECTION A", 0.09),
            ("1. Compulsory question.", 0.09),
            ("SECTION B", 0.09),
            ("2. Optional question.", 0.09),
        )
        paper = extract(index)
        by_id = {s.section_id: s for s in paper.sections}

        assert by_id["A"].requirement.answer_any is None
        assert by_id["B"].requirement.answer_any == 2
        assert by_id["B"].requirement.is_optional

    def test_a_year_in_running_text_does_not_become_a_question(self) -> None:
        index = index_of(
            ("1. Describe the events of 1947.", 0.09),
            ("In 1947 India gained independence from Britain.", 0.09),
        )
        paper = extract(index)
        assert len(paper.questions) == 1
        assert paper.questions[0].label_raw == "1."


class TestGapValidation:
    def test_reports_a_genuinely_missing_number(self) -> None:
        index = index_of(
            ("1. First.", 0.09),
            ("2. Second.", 0.09),
            ("4. Fourth.", 0.09),
        )
        paper = extract(index)
        assert [g.expected_label for g in paper.gaps] == ["3."]

    def test_a_parent_existing_only_through_children_is_not_a_gap(self) -> None:
        # The bug the eval harness caught. A paper printing "2 (i)" and "2 (ii)"
        # has no standalone entry for 2, so collecting only final tokens saw the
        # top level as 1, 3 and reported 2 missing on every paper.
        index = index_of(
            ("1. First.", 0.09),
            ("2 (i) Part one.", 0.09),
            ("(ii) Part two.", 0.13),
            ("3. Third.", 0.09),
        )
        paper = extract(index)
        assert paper.gaps == [], [g.expected_label for g in paper.gaps]

    def test_numbering_restarting_per_section_is_not_a_gap(self) -> None:
        index_ = index_of(
            ("SECTION A", 0.09),
            ("1. One.", 0.09),
            ("2. Two.", 0.09),
            ("SECTION B", 0.09),
            ("1. One again.", 0.09),
        )
        paper = extract(index_)
        assert paper.gaps == []

    def test_roman_sequences_are_not_gap_checked(self) -> None:
        # Deciding whether "(i)" follows "(h)" or starts a roman sequence is the
        # ambiguity the parser refuses to resolve; guessing here would invent
        # gaps that do not exist.
        index = index_of(
            ("1. Answer the following:", 0.09),
            ("(i) One.", 0.13),
            ("(iii) Three.", 0.13),
        )
        paper = extract(index)
        assert paper.gaps == []

    def test_suspicious_flags_an_empty_extraction(self) -> None:
        assert validate.suspicious([]) == ["no questions were extracted at all"]

    def test_suspicious_flags_duplicate_identities(self) -> None:
        from vedaai_contracts import Question

        q = Question(qid="A/1", label_raw="1.", text="One", path=["1"], print_order=0)
        problems = validate.suspicious([q, q])
        assert any("duplicate" in p for p in problems)


class TestStems:
    """Telling a heading apart from a task whose marks are itemised below it.

    Structurally identical — a parent with no marks of its own — and the
    consequence of confusing them is asymmetric. Calling a heading a question
    reports a blank the paper never invited and lets it absorb its own sub-part's
    answer. Calling a task a heading removes the question the student answered
    from the candidate list entirely, which is worse.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Answer the following:",
            "Answer the following about the program you wrote for question 1:",
            "Answer both parts:",
            "Distinguish between the following pairs:",
            "Attempt all parts of the question below:",
            # A question in isolation, but a heading when sub-parts follow: the
            # equations are the answerable parts, and this line is meaningless
            # without them. The structural test decides whether it matters.
            "Balance the following equations:",
        ],
    )
    def test_recognizes_a_heading(self, text) -> None:
        assert reads_as_a_heading(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            # The case that forced the wording check. No marks of its own — they
            # sit on its (a) and (b) — but it is the task the student answered.
            "Write a program that reads an array of 0s and 1s and prints the length "
            "of the longest run of 1s in that array.",
            "Explain the working of an electric motor.",
            # A colon, but nothing pointing at parts below it.
            "State Ohm's law and write its formula:",
        ],
    )
    def test_does_not_mistake_a_question_for_a_heading(self, text) -> None:
        assert reads_as_a_heading(text) is False

    def test_a_parent_that_prints_marks_is_answerable(self) -> None:
        # An allocation says something is expected here, whatever the wording.
        marked = Question(
            qid="A/2",
            label_raw="2.",
            text="Answer the following:",
            path=["2"],
            print_order=0,
            marks=4,
        )
        child = Question(
            qid="A/2/i", label_raw="(i)", text="One.", path=["2", "i"], print_order=1, marks=2
        )
        flagged = mark_stems([marked, child])
        assert flagged[0].is_stem is False

    def test_a_leaf_is_never_a_stem(self) -> None:
        # Nothing extends its path, so there are no parts for it to introduce —
        # even though its wording would otherwise qualify.
        leaf = Question(
            qid="A/2",
            label_raw="2.",
            text="Answer the following:",
            path=["2"],
            print_order=0,
            marks=None,
        )
        assert mark_stems([leaf])[0].is_stem is False

    def test_flags_a_heading_with_children_and_no_marks(self) -> None:
        parent = Question(
            qid="A/2",
            label_raw="2.",
            text="Answer the following:",
            path=["2"],
            print_order=0,
            marks=None,
        )
        child = Question(
            qid="A/2/i", label_raw="(i)", text="One.", path=["2", "i"], print_order=1, marks=2
        )
        flagged = mark_stems([parent, child])
        assert flagged[0].is_stem is True
        assert flagged[1].is_stem is False


class TestTheRealTwoPagePaper:
    """Extraction over the whole chain, on the fixture paper that broke it.

    The unit tests above exercise ``classify`` on a single line. This runs the
    real path — transcribe, order, extract — because the header bug lived in the
    interaction between three modules: the classifier saw an unlabelled line, the
    page boundary carried the previous role across it, and the extractor appended
    the result to the question it was still building.
    """

    @staticmethod
    def _paper():
        from grader import render
        from grader.lineindex import build_index
        from grader.ocr.base import PageInput
        from grader.ocr.native_pdf import PdfTextLayerEngine

        from .fixtures import question_paper

        data, _ = question_paper()
        engine = PdfTextLayerEngine()
        source = render.inspect(data, "paper.pdf", DocumentKind.QUESTION_PAPER)
        per_page = []
        for page_index in range(source.page_count):
            width, height = render.page_size(data, "paper.pdf", page_index)
            per_page.append(
                engine.transcribe(
                    PageInput(
                        index=page_index,
                        width=width,
                        height=height,
                        document=data,
                        filename="paper.pdf",
                    )
                )
            )

        index = build_index(
            DocumentKind.QUESTION_PAPER, per_page, engine.engine, trust_engine_order=True
        )
        return extract(index)

    def test_the_page_two_header_is_not_absorbed_into_a_question(self) -> None:
        paper = self._paper()
        for question in paper.questions:
            assert "UNIT TEST" not in question.text, f"{question.label_raw}: {question.text}"
            assert "page 2" not in question.text.lower(), question.label_raw

    def test_the_last_question_on_page_one_keeps_its_own_text(self) -> None:
        # The other half: the fix must not have removed the question the header
        # was attached to, nor truncated it.
        paper = self._paper()
        seven = next(q for q in paper.questions if q.label_raw.startswith("7"))
        assert seven.text == "Calculate the resistance of the circuit shown."

    def test_questions_on_both_pages_are_extracted(self) -> None:
        paper = self._paper()
        labels = [q.label_raw for q in paper.questions]
        assert any(label.startswith("1.") for label in labels)
        assert any(label.startswith("11") for label in labels), "page 2 questions missing"
