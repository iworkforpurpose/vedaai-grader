"""Tests for the metric definitions.

These matter more than most tests in the project. A metric that is subtly wrong
does not fail — it produces an authoritative-looking number that sends work in
the wrong direction, and it will be trusted precisely because it is quantitative.
"""

from __future__ import annotations

import pytest
from vedaai_contracts import AnswerStatus, BBox, PageBox

from vedaai_evals import metrics


def box(x0: float, y0: float, x1: float, y1: float) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def page_box(page: int, x0: float, y0: float, x1: float, y1: float) -> PageBox:
    return PageBox(page=page, box=box(x0, y0, x1, y1))


class TestLineRecall:
    def test_perfect_detection(self) -> None:
        truth = [(0, box(0.1, 0.1, 0.9, 0.15), "one"), (0, box(0.1, 0.2, 0.9, 0.25), "two")]
        predicted = [(0, box(0.1, 0.1, 0.9, 0.15)), (0, box(0.1, 0.2, 0.9, 0.25))]
        report = metrics.line_recall(truth, predicted)

        assert report.recall == 1.0
        assert report.precision == 1.0
        assert report.missed_text == []

    def test_a_missed_line_is_named(self) -> None:
        # After a low recall the useful question is always *which* lines were
        # lost, which a percentage cannot answer.
        truth = [
            (0, box(0.1, 0.1, 0.9, 0.15), "found me"),
            (0, box(0.1, 0.5, 0.9, 0.55), "long declaration line"),
        ]
        predicted = [(0, box(0.1, 0.1, 0.9, 0.15))]
        report = metrics.line_recall(truth, predicted)

        assert report.recall == 0.5
        assert report.missed_text == ["long declaration line"]

    def test_one_prediction_cannot_satisfy_two_truth_lines(self) -> None:
        # Without greedy consumption, a single sprawling box covering a whole
        # page would score perfect recall — the exact failure mode of a merge
        # that bridged unrelated content.
        truth = [
            (0, box(0.1, 0.10, 0.9, 0.20), "one"),
            (0, box(0.1, 0.22, 0.9, 0.32), "two"),
        ]
        sprawling = [(0, box(0.05, 0.05, 0.95, 0.40))]
        report = metrics.line_recall(truth, sprawling)

        assert report.matched <= 1, "a single box claimed credit for multiple lines"

    def test_a_line_on_another_page_does_not_match(self) -> None:
        truth = [(1, box(0.1, 0.1, 0.9, 0.15), "page two")]
        predicted = [(0, box(0.1, 0.1, 0.9, 0.15))]
        assert metrics.line_recall(truth, predicted).recall == 0.0

    def test_spurious_predictions_lower_precision_only(self) -> None:
        truth = [(0, box(0.1, 0.1, 0.9, 0.15), "one")]
        predicted = [(0, box(0.1, 0.1, 0.9, 0.15)), (0, box(0.1, 0.6, 0.9, 0.65))]
        report = metrics.line_recall(truth, predicted)

        assert report.recall == 1.0
        assert report.precision == 0.5

    def test_no_truth_means_full_recall(self) -> None:
        assert metrics.line_recall([], []).recall == 1.0


class TestCharacterErrorRate:
    def test_identical_text_scores_zero(self) -> None:
        assert metrics.character_error_rate("hello", "hello") == 0.0

    def test_substitution_counts_once(self) -> None:
        assert metrics.character_error_rate("cat", "cot") == pytest.approx(1 / 3)

    def test_matches_a_real_recognition_error(self) -> None:
        # From the actual test script: "#include <stdio.h>" came back as
        # "Hinclude (stdio.h7".
        rate = metrics.character_error_rate("#include <stdio.h>", "Hinclude (stdio.h7")
        assert 0.1 < rate < 0.35

    def test_empty_prediction_scores_one(self) -> None:
        assert metrics.character_error_rate("hello", "") == 1.0

    def test_hallucinated_text_may_exceed_one(self) -> None:
        # Not a bug. A recognizer that invents text is worse than one returning
        # nothing, and the metric should be able to say so.
        assert metrics.character_error_rate("hi", "a very long invention") > 1.0

    def test_levenshtein_basics(self) -> None:
        assert metrics.levenshtein("", "abc") == 3
        assert metrics.levenshtein("abc", "") == 3
        assert metrics.levenshtein("kitten", "sitting") == 3


class TestExtractionScores:
    def test_all_labels_found_in_order(self) -> None:
        truth = ["1.", "2 (i)", "2 (ii)", "3."]
        report = metrics.extraction_scores(truth, list(truth))

        assert report.f1 == 1.0
        assert report.order_tau == pytest.approx(1.0)

    def test_whitespace_differences_still_match(self) -> None:
        # "11 (a)" and "11  (a)" are the same printed label.
        report = metrics.extraction_scores(["11 (a)"], ["11  (a)"])
        assert report.matched == 1

    def test_dropping_brackets_does_not_match(self) -> None:
        # The requirement is to preserve the original numbering, so the brackets
        # the paper printed are part of the label rather than noise.
        report = metrics.extraction_scores(["11 (a)"], ["11a"])
        assert report.matched == 0
        assert report.missed == ["11 (a)"]

    def test_merging_two_subparts_is_penalized(self) -> None:
        # The failure a text-similarity metric would forgive: treating "11(a)"
        # and "11(b)" as one question. Sub-parts being separate entries is an
        # explicit requirement.
        truth = ["11 (a)", "11 (b)"]
        report = metrics.extraction_scores(truth, ["11"])
        assert report.recall == 0.0
        assert report.spurious == ["11"]

    def test_scrambled_order_is_caught_by_tau(self) -> None:
        # A run can find every question and still present them in the wrong
        # order, which is what a naive reading order does on a two-column paper.
        # Precision and recall are blind to it.
        truth = ["1.", "2.", "3.", "4."]
        scrambled = ["4.", "3.", "2.", "1."]
        report = metrics.extraction_scores(truth, scrambled)

        assert report.f1 == 1.0, "every label was found"
        assert report.order_tau == pytest.approx(-1.0), "yet the order is fully reversed"

    def test_tau_is_none_for_a_single_match(self) -> None:
        report = metrics.extraction_scores(["1."], ["1."])
        assert report.order_tau is None


class TestKendallTau:
    def test_identical_orders(self) -> None:
        assert metrics.kendall_tau([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

    def test_reversed_orders(self) -> None:
        assert metrics.kendall_tau([0, 1, 2], [2, 1, 0]) == pytest.approx(-1.0)

    def test_one_swap_in_four(self) -> None:
        # Six pairs, one discordant: (6-1-1)/6... concretely 4/6.
        assert metrics.kendall_tau([0, 1, 2, 3], [1, 0, 2, 3]) == pytest.approx(4 / 6)

    def test_too_short_to_correlate(self) -> None:
        assert metrics.kendall_tau([0], [0]) is None
        assert metrics.kendall_tau([], []) is None


class TestMultiBoxIou:
    def test_identical_regions(self) -> None:
        boxes = [page_box(0, 0.1, 0.1, 0.5, 0.5)]
        assert metrics.multi_box_iou(boxes, boxes) == pytest.approx(1.0)

    def test_disjoint_regions(self) -> None:
        a = [page_box(0, 0.0, 0.0, 0.2, 0.2)]
        b = [page_box(0, 0.6, 0.6, 0.8, 0.8)]
        assert metrics.multi_box_iou(a, b) == 0.0

    def test_boxes_on_different_pages_never_intersect(self) -> None:
        a = [page_box(0, 0.1, 0.1, 0.5, 0.5)]
        b = [page_box(1, 0.1, 0.1, 0.5, 0.5)]
        assert metrics.multi_box_iou(a, b) == 0.0

    def test_a_page_spanning_answer_is_scoreable(self) -> None:
        # The property that makes multi-page answers measurable at all: per-page
        # intersection and union, summed. One box across two pages has no
        # meaningful rectangle to compare.
        truth = [page_box(0, 0.1, 0.8, 0.9, 0.95), page_box(1, 0.1, 0.05, 0.9, 0.3)]
        assert metrics.multi_box_iou(truth, truth) == pytest.approx(1.0)

    def test_matching_only_one_page_of_two_scores_partially(self) -> None:
        truth = [page_box(0, 0.1, 0.8, 0.9, 0.95), page_box(1, 0.1, 0.05, 0.9, 0.3)]
        predicted = [page_box(0, 0.1, 0.8, 0.9, 0.95)]
        iou = metrics.multi_box_iou(truth, predicted)
        assert 0.0 < iou < 1.0

    def test_empty_on_both_sides_is_perfect(self) -> None:
        assert metrics.multi_box_iou([], []) == 1.0

    def test_empty_on_one_side_is_zero(self) -> None:
        assert metrics.multi_box_iou([page_box(0, 0.1, 0.1, 0.2, 0.2)], []) == 0.0


def case(
    qid: str,
    truth_status: AnswerStatus,
    predicted_status: AnswerStatus,
    *,
    truth_boxes: list[PageBox] | None = None,
    predicted_boxes: list[PageBox] | None = None,
) -> metrics.MappingCase:
    return metrics.MappingCase(
        qid=qid,
        truth_status=truth_status,
        truth_boxes=truth_boxes or [],
        predicted_status=predicted_status,
        predicted_boxes=predicted_boxes or [],
    )


class TestMappingScores:
    def test_a_correct_mapping(self) -> None:
        boxes = [page_box(0, 0.1, 0.1, 0.9, 0.2)]
        report, ious = metrics.mapping_scores(
            [
                case(
                    "A/1",
                    AnswerStatus.ANSWERED,
                    AnswerStatus.ANSWERED,
                    truth_boxes=boxes,
                    predicted_boxes=boxes,
                )
            ]
        )
        assert report.correct == 1
        assert report.accuracy == 1.0
        assert ious == [pytest.approx(1.0)]

    def test_right_question_wrong_region(self) -> None:
        # An alignment success with a geometry failure. Counted apart from a
        # wrong question, because the two have different fixes.
        report, _ = metrics.mapping_scores(
            [
                case(
                    "A/1",
                    AnswerStatus.ANSWERED,
                    AnswerStatus.ANSWERED,
                    truth_boxes=[page_box(0, 0.1, 0.1, 0.9, 0.2)],
                    predicted_boxes=[page_box(0, 0.1, 0.7, 0.9, 0.8)],
                )
            ]
        )
        assert report.correct == 0
        assert report.wrong_region == ["A/1"]

    def test_a_false_unanswered_is_reported_separately(self) -> None:
        # The headline safety number. A teacher acts on "unanswered" without
        # re-reading the script, so this error reaches a grade uncorrected.
        report, _ = metrics.mapping_scores(
            [
                case(
                    "A/1",
                    AnswerStatus.ANSWERED,
                    AnswerStatus.UNANSWERED,
                    truth_boxes=[page_box(0, 0.1, 0.1, 0.9, 0.2)],
                )
            ]
        )
        assert report.missed == ["A/1"]
        assert report.false_unanswered_rate == 1.0

    def test_needs_review_is_not_counted_as_absence(self) -> None:
        # Hedging is not the same as claiming a question was skipped, and
        # scoring it as absence would credit the system for uncertainty.
        boxes = [page_box(0, 0.1, 0.1, 0.9, 0.2)]
        report, _ = metrics.mapping_scores(
            [
                case(
                    "A/1",
                    AnswerStatus.ANSWERED,
                    AnswerStatus.OCR_FAILED,
                    truth_boxes=boxes,
                    predicted_boxes=boxes,
                )
            ]
        )
        assert report.missed == []
        assert report.correct == 1

    def test_claiming_an_answer_where_there_is_none(self) -> None:
        report, _ = metrics.mapping_scores(
            [
                case(
                    "A/1",
                    AnswerStatus.UNANSWERED,
                    AnswerStatus.ANSWERED,
                    predicted_boxes=[page_box(0, 0.1, 0.1, 0.9, 0.2)],
                )
            ]
        )
        assert report.false_answer == ["A/1"]

    def test_correctly_identifying_a_blank(self) -> None:
        report, _ = metrics.mapping_scores(
            [case("A/1", AnswerStatus.UNANSWERED, AnswerStatus.UNANSWERED)]
        )
        assert report.correctly_unanswered == 1
        assert report.accuracy == 1.0
        assert report.false_unanswered_rate == 0.0

    def test_an_optional_question_may_be_skipped(self) -> None:
        # "Attempt any four of seven" — skipping is correct, and reporting it as
        # an omission is a product error rather than a mapping one.
        report, _ = metrics.mapping_scores(
            [case("B/6", AnswerStatus.NOT_REQUIRED, AnswerStatus.NOT_REQUIRED)]
        )
        assert report.not_required_respected == 1
        assert report.not_required_violated == []

    def test_reporting_an_optional_question_as_answered_is_a_violation(self) -> None:
        report, _ = metrics.mapping_scores(
            [
                case(
                    "B/6",
                    AnswerStatus.NOT_REQUIRED,
                    AnswerStatus.ANSWERED,
                    predicted_boxes=[page_box(0, 0.1, 0.1, 0.9, 0.2)],
                )
            ]
        )
        assert report.not_required_violated == ["B/6"]

    def test_false_unanswered_rate_ignores_genuinely_blank_questions(self) -> None:
        # Denominator is answered questions only. Including blanks would dilute
        # the rate on a mostly-empty script, which is when it matters most.
        boxes = [page_box(0, 0.1, 0.1, 0.9, 0.2)]
        report, _ = metrics.mapping_scores(
            [
                case("A/1", AnswerStatus.ANSWERED, AnswerStatus.UNANSWERED, truth_boxes=boxes),
                case(
                    "A/2",
                    AnswerStatus.ANSWERED,
                    AnswerStatus.ANSWERED,
                    truth_boxes=boxes,
                    predicted_boxes=boxes,
                ),
                case("A/3", AnswerStatus.UNANSWERED, AnswerStatus.UNANSWERED),
                case("A/4", AnswerStatus.UNANSWERED, AnswerStatus.UNANSWERED),
            ]
        )
        assert report.false_unanswered_rate == pytest.approx(0.5)


class TestIouSummary:
    def test_summarizes_thresholds(self) -> None:
        summary = metrics.summarize_iou([0.4, 0.6, 0.8, 0.9])
        assert summary["mean"] == pytest.approx(0.675)
        assert summary["at_50"] == pytest.approx(0.75)
        assert summary["at_75"] == pytest.approx(0.5)

    def test_empty_is_zero_not_an_error(self) -> None:
        assert metrics.summarize_iou([])["n"] == 0.0
