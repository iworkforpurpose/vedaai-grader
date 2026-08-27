"""Tests for question extraction.

Extraction accuracy is the first graded criterion, and the failure modes are
asymmetric: a missed question does not exist and cannot be answered, while a
spurious one is visible and dismissible. The tests below lean accordingly —
several assert that plausible-looking non-questions are rejected, and several
assert that structures which look like noise are kept.
"""

from __future__ import annotations

import pytest
from vedaai_contracts import BBox, DocumentKind, Line, LineRole, OcrEngine

from grader.questions import furniture, numbering, optionality, validate
from grader.questions.extract import extract
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

    def test_running_headers_need_three_pages_to_be_detected(self) -> None:
        # On two pages, text appearing on both is as likely a genuine repeat as
        # furniture, so repetition alone is not enough evidence.
        two_pages = [
            line(1, "SCIENCE UNIT TEST", y0=0.02, page=0),
            line(2, "SCIENCE UNIT TEST", y0=0.02, page=1),
        ]
        assert furniture.find_repeated_lines(two_pages) == set()

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
