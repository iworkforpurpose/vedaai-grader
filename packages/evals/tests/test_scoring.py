"""Tests for the scoring metrics.

The metrics are the instrument, so a fault in them is worse than a fault in the
thing they measure: a wrong number here is believed and acted on. These tests are
therefore written against hand-built truth where the right answer is obvious by
inspection, not against pipeline output.

The cases that matter most are the ones where a plausible implementation gets it
wrong: a false zero cancelling a false credit in the total, an excluded question
dragging the denominator down, and mark error being averaged across placements
that were never comparable.
"""

from __future__ import annotations

import pytest
from vedaai_contracts import AnswerStatus
from vedaai_evals import metrics
from vedaai_evals.marks import MarkSet, MarkTruth


def truth(**kwargs) -> MarkTruth:
    base = dict(qid="q", marks_available=4.0, status=AnswerStatus.ANSWERED)
    return MarkTruth(**{**base, **kwargs})


def fact(qid: str, awarded: float, *, status: str = "answered", **kwargs) -> metrics.GradedFact:
    base = dict(
        qid=qid,
        status=status,
        marks_available=4.0,
        marks_awarded=awarded,
        judged=True,
    )
    return metrics.GradedFact(**{**base, **kwargs})


def markset(*questions: MarkTruth, **kwargs) -> MarkSet:
    return MarkSet(doc_id="t", source="authored", questions=list(questions), **kwargs)


# -- the schema refuses truth that cannot be checked -----------------------


def test_unmarked_question_must_say_why():
    """An exclusion has to be a decision, never an omission."""
    with pytest.raises(ValueError, match="excluded_reason"):
        truth(qid="1", marks_awarded=None)


def test_award_above_the_denominator_is_refused():
    with pytest.raises(ValueError, match="of 4.0 available"):
        truth(qid="1", marks_awarded=5.0)


def test_award_outside_its_own_band_is_refused():
    """The band is a claim about the mark; the two cannot disagree."""
    with pytest.raises(ValueError, match="outside its own band"):
        truth(qid="1", marks_awarded=4.0, band_low=1.0, band_high=2.0)


def test_unanswered_cannot_carry_marks():
    with pytest.raises(ValueError, match="unanswered but awarded"):
        truth(qid="1", status=AnswerStatus.UNANSWERED, marks_awarded=2.0)


def test_band_defaults_to_the_exact_mark():
    """Not to a guessed tolerance. A band is a claim and has to be stated."""
    assert truth(qid="1", marks_awarded=3.0).band == (3.0, 3.0)


# -- error, and what it is measured against -------------------------------


def test_error_is_measured_against_the_band_midpoint():
    """A mark inside the band still contributes error, but to the midpoint.

    Otherwise a band would zero out the error of everything inside it and the
    mean would stop moving as the marker drifted within the tolerance.
    """
    report = metrics.scoring_scores(
        markset(truth(qid="1", marks_awarded=3.0, band_low=2.0, band_high=4.0)),
        {"1": fact("1", 4.0)},
    )
    assert report.errors["1"] == pytest.approx(1.0)
    assert report.within_band == ["1"]
    assert report.within_band_rate == 1.0


def test_a_perfect_run_scores_zero_error():
    report = metrics.scoring_scores(
        markset(
            truth(qid="1", marks_awarded=4.0),
            truth(qid="2", marks_awarded=2.0, marks_available=2.0),
        ),
        {"1": fact("1", 4.0), "2": fact("2", 2.0, marks_available=2.0)},
    )
    assert report.mae == 0.0
    assert report.total_error == 0.0
    assert report.false_zeros == []
    assert report.total_within_band


# -- the headline: false zeros do not average away ------------------------


def test_false_zero_is_reported_even_when_the_total_is_right():
    """The case a total-only figure cannot see.

    One earned answer scored zero, one unattempted question credited the same
    amount. The script total is exactly right and two questions are badly wrong.
    This is the economics mislabel, reduced to its arithmetic.
    """
    report = metrics.scoring_scores(
        markset(
            truth(qid="3", marks_awarded=3.0, marks_available=3.0),
            truth(qid="4", marks_awarded=0.0, marks_available=3.0,
                  status=AnswerStatus.UNANSWERED),
        ),
        {
            "3": fact("3", 0.0, marks_available=3.0),
            "4": fact("4", 3.0, marks_available=3.0),
        },
    )
    assert report.total_error == 0.0, "the total hides it, which is the point"
    assert report.false_zeros == ["3"]
    assert report.false_credit == ["4"]
    assert report.mae == pytest.approx(3.0)


def test_false_zero_denominator_excludes_the_genuine_zeros():
    """A script with many unattempted questions must not dilute the rate.

    Four questions, one earned marks and was scored zero, three earned nothing
    and were correctly scored zero. The rate is 1/1, not 1/4 -- otherwise a
    mostly-blank script makes any false zero look rare.
    """
    report = metrics.scoring_scores(
        markset(
            truth(qid="1", marks_awarded=4.0),
            *[
                truth(qid=str(n), marks_awarded=0.0, status=AnswerStatus.UNANSWERED)
                for n in (2, 3, 4)
            ],
        ),
        {
            "1": fact("1", 0.0),
            **{str(n): fact(str(n), 0.0, status="unanswered") for n in (2, 3, 4)},
        },
    )
    assert report.false_zero_rate == pytest.approx(1.0)


def test_no_earned_marks_gives_no_false_zero_rate():
    """None, not zero. A rate over an empty denominator is not 'perfect'."""
    report = metrics.scoring_scores(
        markset(truth(qid="1", marks_awarded=0.0, status=AnswerStatus.UNANSWERED)),
        {"1": fact("1", 0.0, status="unanswered")},
    )
    assert report.false_zero_rate is None


# -- exclusions leave both sides of the fraction --------------------------


def test_an_excluded_question_leaves_the_denominator_too():
    """Otherwise an unmarkable question silently penalises the script."""
    report = metrics.scoring_scores(
        markset(
            truth(qid="1", marks_awarded=4.0),
            truth(qid="2", marks_awarded=None, excluded_reason="figure is indeterminate"),
        ),
        {"1": fact("1", 4.0), "2": fact("2", 0.0)},
    )
    assert report.scored == ["1"]
    assert report.excluded == ["2"]
    assert report.available_total == 4.0
    assert report.mae == 0.0


# -- attribution: marker error vs aligner error ---------------------------


def test_hedging_on_a_blank_is_not_a_placement_error():
    """`uncertain` where truth says `unanswered` is the system declining to
    assert a blank -- the conservative direction this product prefers.

    The first version compared statuses for equality and called this misplaced.
    On the real mathematics script that inverted the attribution outright: five
    untouched questions came back `uncertain`, so the report claimed placement
    failed five times when placement was in fact perfect and every mark of the
    error belonged to the marker.
    """
    report = metrics.scoring_scores(
        markset(truth(qid="1", marks_awarded=0.0, status=AnswerStatus.UNANSWERED)),
        {"1": fact("1", 0.0, status="uncertain")},
    )
    assert report.placed_right == ["1"]
    assert report.placed_wrong == []
    assert report.hedged == ["1"]


def test_placing_writing_on_a_question_nobody_answered_is_a_placement_error():
    report = metrics.scoring_scores(
        markset(truth(qid="1", marks_awarded=0.0, status=AnswerStatus.UNANSWERED)),
        {"1": fact("1", 2.0, status="answered")},
    )
    assert report.placed_wrong == ["1"]
    assert report.hedged == []
    assert report.false_credit == ["1"]


def test_error_is_partitioned_by_whether_placement_was_right():
    """Without this, a marking fix is scored against an aligner fault.

    Same reasoning the runner already applies when it reads synthetic sheets
    from their text layer so a mapping regression cannot be mistaken for a
    recognition one.
    """
    report = metrics.scoring_scores(
        markset(
            truth(qid="1", marks_awarded=4.0),
            truth(qid="2", marks_awarded=4.0),
        ),
        {
            "1": fact("1", 3.0),                        # placed right, one mark off
            "2": fact("2", 0.0, status="uncertain"),    # never placed at all
        },
    )
    assert report.placed_right == ["1"]
    assert report.placed_wrong == ["2"]
    assert report.mae_placed_right == pytest.approx(1.0)
    assert report.mae_placed_wrong == pytest.approx(4.0)
    assert report.marks_lost_to_placement == pytest.approx(4.0)


# -- hygiene that needs no labels at all ----------------------------------


def test_a_rubric_split_that_changes_the_denominator_is_caught():
    """A wrong denominator makes every mark on that question wrong.

    The economics paper prints [4] against a question in a section that says 3,
    so this is the check that says which one the rubric took.
    """
    report = metrics.scoring_scores(
        markset(truth(qid="1", marks_awarded=4.0)),
        {"1": fact("1", 4.0, criteria=3, rubric_marks_sum=3.0)},
    )
    assert report.denominator_mismatch == [("1", 3.0, 4.0)]
    assert not report.denominator_ok


def test_matching_denominator_is_not_flagged():
    report = metrics.scoring_scores(
        markset(truth(qid="1", marks_awarded=4.0)),
        {"1": fact("1", 4.0, criteria=2, rubric_marks_sum=4.0)},
    )
    assert report.denominator_ok


def test_unmarked_questions_are_counted_by_reason():
    """A showcase run could be half unmarked and a mark-error figure would not say so."""
    report = metrics.scoring_scores(
        markset(
            truth(qid="1", marks_awarded=0.0, status=AnswerStatus.UNANSWERED),
            truth(qid="2", marks_awarded=None, excluded_reason="not required"),
        ),
        {
            "1": fact("1", 0.0, status="unanswered", unmarked_reason="Nothing was written."),
            "2": fact("2", 0.0, status="not_required", unmarked_reason="Not required."),
        },
    )
    assert report.unmarked_reasons == {"Nothing was written.": 1, "Not required.": 1}


def test_an_answered_question_nobody_judged_is_named():
    """An unjudged zero and a judged zero are the same number and different facts."""
    report = metrics.scoring_scores(
        markset(truth(qid="1", marks_awarded=4.0)),
        {"1": fact("1", 0.0, judged=False)},
    )
    assert report.unjudged == ["1"]


def test_citation_rate_is_over_mark_bearing_points_only():
    report = metrics.scoring_scores(
        markset(truth(qid="1", marks_awarded=4.0)),
        {"1": fact("1", 4.0, awarded_points=4, cited_points=3)},
    )
    assert report.citation_rate == pytest.approx(0.75)


# -- truth that names a question the run never produced -------------------


def test_a_missing_question_is_reported_not_skipped():
    """Scoring the intersection would let a paper that lost half its questions
    report a flawless mark error."""
    report = metrics.scoring_scores(
        markset(truth(qid="1", marks_awarded=4.0), truth(qid="2", marks_awarded=4.0)),
        {"1": fact("1", 4.0)},
    )
    assert report.missing_from_run == ["2"]
    assert report.scored == ["1"]


# -- stability ------------------------------------------------------------


def test_stability_reports_the_range_not_the_variance():
    """A teacher meets the range. Variance understates a rare large disagreement."""
    spread = metrics.mark_stability([{"1": 3.0}, {"1": 3.0}, {"1": 5.0}, {"1": 3.0}])
    assert spread == {"1": 2.0}


def test_stability_of_a_reproducible_run_is_zero():
    assert metrics.mark_stability([{"1": 3.0}, {"1": 3.0}]) == {"1": 0.0}


def test_stability_tolerates_a_question_missing_from_one_run():
    """A run that failed on one question must not erase the others' stability."""
    assert metrics.mark_stability([{"1": 2.0, "2": 1.0}, {"1": 4.0}]) == {"1": 2.0, "2": 0.0}
