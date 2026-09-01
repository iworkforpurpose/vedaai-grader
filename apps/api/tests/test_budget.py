"""Tests for what one upload is allowed to cost.

The rate limit counts submissions. Marking counts questions, one paid call each,
and nothing connected the two: a document is accepted up to 60 pages, and a page
holds around 40 lines, so a paper crafted so that every line parses as a question
turns a single upload into thousands of model calls. Thirty of those an hour is
inside the rate limit and outside any budget anybody chose.

This is OWASP's "unbounded consumption" — the denial-of-wallet case — and the
mitigation it asks for is a limit applied before the work is done rather than a
bill noticed after it.

The numbers here are not the interesting part. What matters is that a cap exists,
that exceeding it is reported rather than hidden, and that the questions beyond it
are still extracted and still located, because those cost nothing and are most of
what the product does.
"""

from __future__ import annotations

import asyncio

from vedaai_contracts import (
    AnswerStatus,
    DocumentKind,
    LineIndex,
    Mapping,
    MappingResult,
    OcrEngine,
    Question,
    QuestionPaper,
)

from grader.grading import run
from grader.grading.run import MAX_MARKED_QUESTIONS


class _CountingGrader:
    """Stands in for the paid API, and counts what it would have cost."""

    def __init__(self) -> None:
        self.calls = 0

    # `scheme` is part of the Grader protocol: the correct answer, derived from the
    # question before the script is read. Accepted and ignored here — this double
    # counts calls, it does not mark.
    async def grade(self, *, question, rubric, index, line_ids, scheme=None):  # noqa: ANN001
        self.calls += 1
        from vedaai_contracts import QuestionGrade

        return QuestionGrade(
            qid=question.qid,
            marks_available=rubric.marks_available,
            marks_awarded=0.0,
            rubric_points=[],
            judged=True,
        )


def _paper(count: int) -> QuestionPaper:
    return QuestionPaper(
        questions=[
            Question(
                qid=f"A/{i}",
                label_raw=f"{i}.",
                text=f"Define term number {i} in one sentence.",
                path=[str(i)],
                print_order=i,
                marks=2,
            )
            for i in range(count)
        ],
        sections=[],
    )


def _answered(paper: QuestionPaper, index: LineIndex) -> MappingResult:
    line = index.lines[0].line_id
    return MappingResult(
        mappings=[
            Mapping(
                qid=q.qid,
                status=AnswerStatus.ANSWERED,
                start_line_id=line,
                end_line_id=line,
            )
            for q in paper.questions
        ],
        orphans=[],
        unassigned_ink_ratio=0.0,
    )


def _index() -> LineIndex:
    from vedaai_contracts import BBox, Line

    return LineIndex(
        kind=DocumentKind.ANSWER_SHEET,
        lines=[
            Line(
                line_id="as:0001",
                kind=DocumentKind.ANSWER_SHEET,
                page=0,
                box=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.13),
                text="An answer to everything at once.",
                confidence=0.9,
                engine=OcrEngine.PADDLE_OCR_VL,
            )
        ],
        engine=OcrEngine.PADDLE_OCR_VL,
    )


def _mark(paper: QuestionPaper, grader: _CountingGrader):
    """The grades. `grade_submission` also returns the failures, unused here."""
    index = _index()
    result, _failures = asyncio.run(
        run.grade_submission(
            paper=paper,
            mapping=_answered(paper, index),
            index=index,
            grader=grader,
            excluded_line_ids=set(),
        )
    )
    return result


class TestWhatOneUploadMayCost:
    def test_an_ordinary_paper_is_marked_in_full(self) -> None:
        grader = _CountingGrader()
        paper = _paper(18)
        result = _mark(paper, grader)
        assert grader.calls == 18
        assert len(result.grades) == 18

    def test_a_paper_past_the_cap_stops_paying_at_it(self) -> None:
        grader = _CountingGrader()
        result = _mark(_paper(MAX_MARKED_QUESTIONS + 40), grader)
        assert grader.calls == MAX_MARKED_QUESTIONS
        # Every question still comes back. Extraction and location cost nothing
        # and are most of what the product does; only the marking is capped.
        assert len(result.grades) == MAX_MARKED_QUESTIONS + 40

    def test_and_says_so_rather_than_looking_marked(self) -> None:
        result = _mark(_paper(MAX_MARKED_QUESTIONS + 5), _CountingGrader())
        beyond = result.grades[-1]
        assert beyond.judged is False
        assert beyond.rubric_points
        assert "more questions than" in (beyond.rubric_points[0].comment or "")

    def test_the_ones_within_the_cap_are_still_marked(self) -> None:
        result = _mark(_paper(MAX_MARKED_QUESTIONS + 5), _CountingGrader())
        assert result.grades[0].judged is True
