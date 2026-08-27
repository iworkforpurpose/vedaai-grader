"""Grading a whole submission.

The order here is the design. Rubrics come from the paper, the lines come from
the mapping with abandoned work already removed, and only then is anything
judged — so the two failure modes that would be invisible in a score are
structurally impossible rather than guarded against downstream:

  * A question with no answer is never marked. It is reported by the status the
    mapping gave it, so "not answered" and "we could not read it" do not both
    arrive as zero.
  * Struck-through and bleed-through writing is filtered before the prompt is
    built, so a student's abandoned attempt cannot be the thing that gets marked.
"""

from __future__ import annotations

import asyncio

from vedaai_contracts import (
    AnswerStatus,
    GradeResult,
    LineIndex,
    MappingResult,
    QuestionGrade,
    QuestionPaper,
    RubricPoint,
)

from . import rubric as rubric_mod
from .citations import gradable_lines
from .engine import Grader

#: How many answers to judge at once. Grading is IO-bound on the model, and a
#: script is small, so a modest fan-out keeps a full script well inside the
#: patience of someone watching a progress bar.
CONCURRENCY = 4


def _ungraded(
    qid: str,
    marks: float,
    criteria: list[rubric_mod.Criterion],
    *,
    reason: str,
) -> QuestionGrade:
    return QuestionGrade(
        qid=qid,
        marks_available=marks,
        marks_awarded=0.0,
        rubric_points=[
            RubricPoint(
                point_id=f"{qid}#{i + 1}",
                criterion=c.criterion,
                marks_available=c.marks,
                marks_awarded=0.0,
                satisfied=False,
                cited_line_ids=[],
                comment=reason,
            )
            for i, c in enumerate(criteria)
        ],
        confidence=0.0,
        graded_on_partial_text=False,
    )


#: Why a question was left unmarked, phrased for a teacher. Keyed by the status
#: the mapping assigned, so grading never re-derives a conclusion the aligner
#: already reached with better evidence.
_SKIP_REASON = {
    AnswerStatus.UNANSWERED: "Nothing was written for this question.",
    AnswerStatus.OCR_FAILED: "There is writing here, but it could not be read. Check the page.",
    AnswerStatus.NOT_REQUIRED: "Not required — the student answered the alternatives.",
    AnswerStatus.PAGES_MISSING: "A page appears to be missing.",
    AnswerStatus.UNCERTAIN: "The answer to this could not be located on the sheet.",
}


async def grade_submission(
    *,
    paper: QuestionPaper,
    mapping: MappingResult,
    index: LineIndex,
    grader: Grader,
    excluded_line_ids: set[str],
) -> GradeResult:
    """Grade every answered question, and say why the rest were not marked."""
    by_qid = {m.qid: m for m in mapping.mappings}
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def one(question) -> QuestionGrade:
        spec = rubric_mod.derive(question)
        entry = by_qid.get(question.qid)

        if question.is_stem:
            # A heading. Reporting it as unmarked would be as misleading as
            # reporting it unanswered: nothing was asked here.
            return _ungraded(
                question.qid,
                spec.marks_available,
                spec.criteria,
                reason="This introduces the parts below and carries no marks of its own.",
            )

        if entry is None or entry.status is not AnswerStatus.ANSWERED:
            status = entry.status if entry else AnswerStatus.UNANSWERED
            return _ungraded(
                question.qid,
                spec.marks_available,
                spec.criteria,
                reason=_SKIP_REASON.get(status, "Not marked."),
            )

        line_ids = gradable_lines(
            index,
            answer_line_ids=_answer_lines(entry, index),
            excluded=excluded_line_ids,
        )

        if not line_ids:
            # Every line was struck through or unreadable. The writing exists, so
            # this is emphatically not a blank answer, and marking it zero would
            # assert exactly that.
            return _ungraded(
                question.qid,
                spec.marks_available,
                spec.criteria,
                reason="The writing here was crossed out or could not be read. "
                "Check the page before marking.",
            )

        async with semaphore:
            try:
                return await grader.grade(
                    question=question, rubric=spec, index=index, line_ids=line_ids
                )
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                # A model that cannot be reached is not a failed submission. The
                # rubric is already derived and the answer already located, which
                # is most of the value; the mark is what is missing.
                #
                # Caught per question rather than around the whole batch because
                # these run concurrently, and one refused request would otherwise
                # discard the grades that succeeded beside it.
                failures.append(_describe(exc))
                return _ungraded(
                    question.qid,
                    spec.marks_available,
                    spec.criteria,
                    reason="Could not be marked automatically. The rubric is ready "
                    "for you.",
                )

    failures: list[str] = []
    ordered = sorted(paper.questions, key=lambda q: q.print_order)
    grades = await asyncio.gather(*(one(q) for q in ordered))
    result = GradeResult(grades=list(grades), weak_topics=_weak_topics(list(grades), paper))

    # Deduplicated, because a bad key fails identically on every question and a
    # teacher does not need to read the same sentence eight times.
    return result, list(dict.fromkeys(failures))


#: Provider errors translated into the thing to change. The raw messages arrive in
#: a warning beside a student's script, where "401" is not actionable.
_MARKING_EXPLANATIONS = (
    ("authenticationerror", "the API key was rejected. Check OPENAI_API_KEY or "
     "ANTHROPIC_API_KEY in .env."),
    ("permissiondenied", "the key is valid but not permitted to use this model. "
     "Try another via GRADER_MODEL."),
    ("notfounderror", "the model name was not recognised. Set GRADER_MODEL to one "
     "the account can use."),
    ("ratelimit", "the provider is rate limiting or the account is out of credit."),
    ("badrequest", "the provider rejected the request. If GRADER_MODEL was changed, "
     "it may not support structured output."),
    ("apiconnection", "the provider could not be reached. Check network access."),
    ("apitimeout", "the provider timed out."),
)


def _describe(error: Exception) -> str:
    """A marking failure phrased as something the operator can act on."""
    name = type(error).__name__.lower()
    for needle, explanation in _MARKING_EXPLANATIONS:
        if needle in name:
            return f"Answers were not marked: {explanation}"
    return f"Answers were not marked: {type(error).__name__}: {error}"


def _answer_lines(entry, index: LineIndex) -> list[str]:
    """The lines of one answer, in document order.

    Taken from the mapping's span rather than from its blocks, so a
    teacher-reassigned answer and an aligner-produced one are read the same way.
    """
    if entry.start_line_id and entry.end_line_id:
        try:
            return [line.line_id for line in index.resolve_span(
                entry.start_line_id, entry.end_line_id
            )]
        except KeyError:
            pass
    return []


def _weak_topics(grades: list[QuestionGrade], paper: QuestionPaper) -> list[str]:
    """Questions the student lost most of the marks on.

    Only ever computed from questions that were actually marked. Including
    unmarked ones would name a topic as weak on the strength of a page we could
    not read.
    """
    text = {q.qid: q.text for q in paper.questions}
    weak = [
        text.get(g.qid, g.qid)
        for g in grades
        if g.fraction is not None and g.fraction < 0.5 and g.rubric_points
        and any(p.cited_line_ids for p in g.rubric_points)
    ]
    return weak[:5]
