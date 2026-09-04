"""Answering binary checks by entailment.

The decision logic is tested without a model on purpose: thresholds, the deferral
band, the windows and the citation choice are where this can be wrong in ways
that cost a student marks, and none of them need 400MB of weights to exercise.
"""

from __future__ import annotations

import pytest
from vedaai_contracts import BBox, DocumentKind, Line, OcrEngine

from grader.grading import verifier
from grader.grading.scheme import Check


def line(n: int, text: str) -> Line:
    return Line(
        line_id=f"as:{n:04d}",
        kind=DocumentKind.ANSWER_SHEET,
        page=0,
        box=BBox(x0=0.1, y0=0.1 + n * 0.03, x1=0.7, y1=0.12 + n * 0.03),
        text=text,
        confidence=0.9,
        engine=OcrEngine.AWS_TEXTRACT,
    )


class Fixed:
    """An entailment model that returns whatever the test says."""

    name = "fixed"

    def __init__(self, by_hypothesis: dict[str, float], default: float = 0.0):
        self.by_hypothesis = by_hypothesis
        self.default = default
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs):
        self.pairs.extend(pairs)
        out = []
        for premise, _hypothesis in pairs:
            best = self.default
            for needle, value in self.by_hypothesis.items():
                if needle in premise:
                    best = max(best, value)
            out.append(best)
        return out


class TestTheCheckBecomesAClaim:
    """NLI wants a statement. The bank writes questions."""

    @pytest.mark.parametrize(
        ("ask", "expected"),
        [
            ("Does the answer give the value 15 m/s?", "The answer gives the value 15 m/s"),
            ("Does the answer state the SI unit as the ohm?",
             "The answer states the SI unit as the ohm"),
            ("Does it mention a closed circuit?", "The answer mentions a closed circuit"),
            ("Does the response identify the process?", "The answer identifies the process"),
        ],
    )
    def test_a_question_becomes_the_claim_it_tests(self, ask, expected) -> None:
        assert verifier.as_statement(ask) == expected

    def test_third_person_agreement_handles_the_awkward_verbs(self) -> None:
        assert verifier.as_statement("Does the answer discuss both sides?").startswith(
            "The answer discusses"
        )
        assert verifier.as_statement("Does the answer identify two causes?").startswith(
            "The answer identifies"
        )

    def test_anything_unrecognised_keeps_its_words(self) -> None:
        """A slightly awkward hypothesis is a far smaller error than a mangled one."""
        assert verifier.as_statement("Two distinct valid reasons are given?") == (
            "Two distinct valid reasons are given"
        )


class TestAnsweringACheck:
    CHECKS = [Check(ask="Does the answer give the value 15 m/s?", marks=1.0)]

    def test_a_clearly_entailed_check_is_met_and_cites_its_line(self) -> None:
        """The citation is the line that entailed it, so it cannot be invented.

        Asking a model for line ids produced fabricated ones, and `citations.check`
        then refused the whole question rather than one mark.
        """
        lines = [line(1, "I used speed = distance over time"), line(2, "so the speed is 15 m/s")]
        model = Fixed({"15 m/s": 0.95})

        [verdict] = verifier.verify(self.CHECKS, lines, model)

        assert verdict.met is True
        assert "as:0002" in verdict.cited_line_ids

    def test_a_check_nothing_supports_is_refused_with_a_reason(self) -> None:
        """A refusal with no named fault is treated as a shrug downstream."""
        lines = [line(1, "The bird is safe because feathers are insulators")]
        model = Fixed({}, default=0.05)

        [verdict] = verifier.verify(self.CHECKS, lines, model)

        assert verdict.met is False
        assert verdict.error
        assert verdict.cited_line_ids == []

    def test_the_middle_is_deferred_rather_than_guessed(self) -> None:
        """The band between the thresholds replaces a generative marker's 'unsure'.

        A false yes is a mark awarded for something the student did not write,
        which is the error a teacher gets challenged on. One named check to settle
        costs them far less than that.
        """
        lines = [line(1, "the speed works out to about fifteen")]
        model = Fixed({"fifteen": 0.45})

        [verdict] = verifier.verify(self.CHECKS, lines, model)

        assert verdict.met is None
        assert verdict.cited_line_ids, "a deferred check still says where to look"

    def test_a_claim_split_across_lines_is_still_found(self) -> None:
        """A value on one line and its unit on the next is one claim.

        Scoring lines only in isolation refuses those, which is a false zero on an
        answer a teacher would mark correct.
        """
        lines = [line(1, "speed = 150 / 10 = 15"), line(2, "m/s")]
        model = Fixed({"15 m/s": 0.9})  # only the joined window contains it

        [verdict] = verifier.verify(self.CHECKS, lines, model)

        assert verdict.met is True
        assert set(verdict.cited_line_ids) == {"as:0001", "as:0002"}

    def test_an_answer_with_no_readable_text_defers_every_check(self) -> None:
        """Emphatically not a zero. The writing exists; we could not read it."""
        verdicts = verifier.verify(self.CHECKS, [line(1, "   ")], Fixed({}))

        assert [v.met for v in verdicts] == [None]
        assert verdicts[0].error

    def test_each_check_is_answered_independently(self) -> None:
        """Two checks, one satisfied. A question that asks two things must be able
        to earn one mark — that is the whole reason for binary checks."""
        checks = [
            Check(ask="Does the answer define resistance?", marks=1.0),
            Check(ask="Does the answer state the SI unit as the ohm?", marks=1.0),
        ]
        lines = [line(1, "Resistance is the opposition to current. Its unit is the ohm.")]
        model = Fixed({"opposition to current": 0.2, "ohm": 0.9})

        met = [v.met for v in verifier.verify(checks, lines, model)]

        assert met == [True, True]


class TestTheJudgementShape:
    def test_it_is_the_shape_the_generative_marker_returns(self) -> None:
        """Deliberately identical, so both markers become a grade the same way.

        `assemble_checks` owns the marks, the deferral handling, the unverifiable
        credit and the citation validation. Two markers that assembled grades
        differently would be two products.
        """
        verdicts = [
            verifier.Verdict(met=True, cited_line_ids=["as:0001"], score=0.9),
            verifier.Verdict(met=False, cited_line_ids=[], score=0.1, error="not stated"),
            verifier.Verdict(met=None, cited_line_ids=["as:0002"], score=0.5, error="close"),
        ]

        judgement = verifier.as_judgement(verdicts)

        assert [c["index"] for c in judgement["checks"]] == [1, 2, 3]
        assert [c["met"] for c in judgement["checks"]] == [True, False, None]
        assert judgement["checks"][0]["cited_line_ids"] == ["as:0001"]
        assert judgement["uncertain"] is False

    def test_a_wholly_undecided_question_is_reported_uncertain(self) -> None:
        verdicts = [verifier.Verdict(met=None, cited_line_ids=[], score=0.4)]

        assert verifier.as_judgement(verdicts)["uncertain"] is True


class TestItIsDeterministic:
    def test_the_same_answer_scores_the_same_twice(self) -> None:
        """The property the five-sample panel existed to approximate.

        An encoder has no decode noise, so the panel and its five-fold cost go
        together, and "a mark a teacher cannot reproduce is not a mark" becomes
        exactly true rather than true 42 times in 45.
        """
        lines = [line(1, "the speed is 15 m/s")]
        checks = [Check(ask="Does the answer give the value 15 m/s?", marks=1.0)]

        first = verifier.verify(checks, lines, Fixed({"15 m/s": 0.9}))
        second = verifier.verify(checks, lines, Fixed({"15 m/s": 0.9}))

        assert first == second
