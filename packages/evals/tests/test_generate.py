"""Tests for the synthetic generator.

A golden set is a measuring instrument. If a case labelled "out_of_order" quietly
writes answers in order, every run scores well against it and the capability is
never actually tested — the harness reports success for work that was never done.
So each case is checked to produce the structure it claims.
"""

from __future__ import annotations

import fitz
import pytest
from vedaai_contracts import AnswerStatus

from vedaai_evals import generate
from vedaai_evals.runner import validate_truth
from vedaai_evals.schema import load_sample


def case_by_id(case_id: str) -> generate.CaseConfig:
    for config in generate.CASES:
        if config.case_id == case_id:
            return config
    raise KeyError(case_id)


def pdf_text(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return " ".join(doc[i].get_text("text") for i in range(doc.page_count))
    finally:
        doc.close()


def page_count(data: bytes) -> int:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()


class TestQuestionPaper:
    def test_every_question_appears_with_its_label(self) -> None:
        data, truth = generate.build_question_paper()
        text = pdf_text(data)

        assert len(truth) == len(generate.PAPER)
        for q in generate.PAPER:
            assert q.text in text, f"{q.qid} text missing from the rendered paper"

    def test_print_order_is_unique_and_sequential(self) -> None:
        _data, truth = generate.build_question_paper()
        orders = [q.print_order for q in truth]
        assert orders == list(range(len(orders)))

    def test_labels_include_nested_subparts(self) -> None:
        # Three-level nesting, as ICSE papers actually use.
        _data, truth = generate.build_question_paper()
        labels = {q.label_raw for q in truth}
        assert "2 (i)" in labels
        assert "2 (i) (a)" in labels
        assert "2 (ii)" in labels

    def test_marks_are_printed_and_recorded(self) -> None:
        data, truth = generate.build_question_paper()
        text = pdf_text(data)
        assert all(q.marks is not None for q in truth)
        assert "[5]" in text


class TestCaseStructures:
    @pytest.mark.parametrize("config", generate.CASES, ids=lambda c: c.case_id)
    def test_every_case_produces_valid_truth(self, config, tmp_path) -> None:
        sample = generate.generate_case(config, tmp_path)
        problems = validate_truth(sample)
        assert problems == [], f"{config.case_id}: {problems}"

    @pytest.mark.parametrize("config", generate.CASES, ids=lambda c: c.case_id)
    def test_every_box_obeys_the_coordinate_contract(self, config, tmp_path) -> None:
        sample = generate.generate_case(config, tmp_path)
        for answer in sample.answers:
            for pb in answer.complete_answer_box:
                assert pb.page >= 0
                assert 0.0 <= pb.box.x0 < pb.box.x1 <= 1.0
                assert 0.0 <= pb.box.y0 < pb.box.y1 <= 1.0

    @pytest.mark.parametrize("config", generate.CASES, ids=lambda c: c.case_id)
    def test_every_case_round_trips_through_disk(self, config, tmp_path) -> None:
        generated = generate.generate_case(config, tmp_path)
        loaded = load_sample(tmp_path / config.case_id)
        assert loaded == generated

    def test_unanswered_case_marks_those_questions_blank(self) -> None:
        config = case_by_id("unanswered")
        _data, answers = generate.build_answer_sheet(config)
        by_qid = {a.qid: a for a in answers}

        for qid in config.omit:
            assert by_qid[qid].status is AnswerStatus.UNANSWERED
            assert by_qid[qid].complete_answer_box == []

        answered = [a for a in answers if a.is_answered]
        assert answered, "the rest should still be answered"

    def test_optional_skipped_is_not_an_omission(self) -> None:
        # The distinction the product exists to get right: skipping an optional
        # question is correct behaviour, not a missing answer.
        config = case_by_id("optional_skipped")
        _data, answers = generate.build_answer_sheet(config)
        by_qid = {a.qid: a for a in answers}

        for qid in config.skip_optional:
            assert by_qid[qid].status is AnswerStatus.NOT_REQUIRED

    def test_orphans_add_writing_that_answers_nothing(self) -> None:
        config = case_by_id("orphans")
        data, answers = generate.build_answer_sheet(config)
        text = pdf_text(data)

        assert any(block[:20] in text for block in generate.ORPHAN_BLOCKS)
        # Orphan blocks belong to no question, so they must not appear as answers.
        assert all(a.qid in {q.qid for q in generate.PAPER} for a in answers)

    def test_page_spanning_answers_cross_a_boundary(self) -> None:
        config = case_by_id("page_spanning")
        data, answers = generate.build_answer_sheet(config)
        by_qid = {a.qid: a for a in answers}

        assert page_count(data) > 1
        for qid in config.span_pages:
            pages = {pb.page for pb in by_qid[qid].complete_answer_box}
            assert len(pages) > 1, f"{qid} was supposed to span pages but sits on {pages}"

    def test_page_spanning_truth_is_one_box_per_page(self) -> None:
        # Not one box across both pages, which would cover everything between.
        config = case_by_id("page_spanning")
        _data, answers = generate.build_answer_sheet(config)
        by_qid = {a.qid: a for a in answers}
        boxes = by_qid[config.span_pages[0]].complete_answer_box
        assert len({pb.page for pb in boxes}) == len(boxes)

    def test_merged_subparts_share_one_block(self) -> None:
        # The hardest supported case: several sub-parts written as one
        # undivided block, with no structural cue for where each ends.
        config = case_by_id("merged_subparts")
        _data, answers = generate.build_answer_sheet(config)
        by_qid = {a.qid: a for a in answers}

        for group in config.merge_subparts:
            boxes = [by_qid[qid].complete_answer_box for qid in group]
            assert all(b == boxes[0] for b in boxes), (
                f"group {group} should share one region"
            )

    def test_mislabelled_answers_carry_the_wrong_number(self) -> None:
        # Attacks label anchors directly. A design that trusts a written label
        # will map these to the wrong question.
        config = case_by_id("mislabelled")
        data, _answers = generate.build_answer_sheet(config)
        text = pdf_text(data)

        for _qid, wrong_label in config.mislabel:
            assert wrong_label in text, f"expected the wrong label {wrong_label!r} on the sheet"

    def test_unlabelled_answers_carry_no_numbers(self) -> None:
        # Leaves only order and content to map by, which is what the semantic
        # signal exists for.
        config = case_by_id("unlabelled")
        data, _answers = generate.build_answer_sheet(config)
        text = pdf_text(data)

        # No answer should begin with a question label.
        for q in generate.PAPER:
            assert f"{q.label} {q.answer[:16]}" not in text

    def test_reversed_order_is_actually_reversed(self) -> None:
        config = case_by_id("reversed")
        data, _answers = generate.build_answer_sheet(config)
        text = pdf_text(data)

        first = generate.PAPER[0].answer[:24]
        last = generate.PAPER[-1].answer[:24]
        assert text.index(last) < text.index(first), "answers were not reversed"

    def test_shuffled_order_differs_from_the_paper(self) -> None:
        config = case_by_id("out_of_order")
        data, _answers = generate.build_answer_sheet(config)
        text = pdf_text(data)

        positions = [text.index(q.answer[:24]) for q in generate.PAPER if q.answer[:24] in text]
        assert positions != sorted(positions), "the shuffle produced the printed order"

    def test_baseline_is_in_order_and_complete(self) -> None:
        config = case_by_id("baseline")
        data, answers = generate.build_answer_sheet(config)
        text = pdf_text(data)

        assert all(a.is_answered for a in answers)
        positions = [text.index(q.answer[:24]) for q in generate.PAPER if q.answer[:24] in text]
        assert positions == sorted(positions)

    def test_the_combined_case_exercises_several_structures_at_once(self) -> None:
        # Real scripts do not arrive one problem at a time.
        config = case_by_id("everything")
        _data, answers = generate.build_answer_sheet(config)
        statuses = {a.status for a in answers}

        assert AnswerStatus.UNANSWERED in statuses
        assert AnswerStatus.ANSWERED in statuses
        spanning = [a for a in answers if len({pb.page for pb in a.complete_answer_box}) > 1]
        assert spanning, "expected a page-spanning answer"


class TestDeterminism:
    def test_the_same_case_generates_identical_truth(self) -> None:
        # A shuffled case must be reproducible, or a score cannot be compared
        # against a previous run and every change looks like a regression.
        config = case_by_id("out_of_order")
        _d1, first = generate.build_answer_sheet(config)
        _d2, second = generate.build_answer_sheet(config)
        assert first == second


class TestFontFallback:
    def test_font_discovery_returns_a_path_or_none(self) -> None:
        # Handwriting faces are macOS system fonts and absent on the Linux
        # worker. Tolerable because realism is not what synthetic pages measure,
        # but it must degrade rather than crash.
        font = generate.available_handwriting_font()
        assert font is None or font.endswith((".ttf", ".ttc", ".otf"))

    def test_generation_works_without_a_handwriting_font(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(generate, "_HANDWRITING_FONTS", [])
        sample = generate.generate_case(case_by_id("baseline"), tmp_path)
        assert sample.answers
        assert validate_truth(sample) == []
