"""Grading engines behind one interface.

Two, for the same reason the transcription layer has two: the interesting
decisions should be measurable against each other rather than assumed. Here the
split also does something else — it keeps grading useful when no model is
configured at all.

``RubricOnly`` proposes no marks. It derives the rubric from the paper, locates
the answer, and hands the teacher both with every point unjudged. That is a
genuine product: a marking aid that structures the work without inventing a
score. It is also the honest fallback, because a grade nobody can justify is
worth less than no grade.

``Claude`` judges the answer and cites the lines behind each mark. Its output is
validated before it is shown; a fabricated or out-of-scope citation invalidates
the grade rather than being quietly dropped.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from vedaai_contracts import LineIndex, Question, QuestionGrade, RubricPoint

from . import citations, prompt
from .rubric import Rubric

#: The model used when none is named. Overridable so a deployment can pin one.
DEFAULT_MODEL = os.getenv("GRADER_MODEL", "claude-sonnet-5")

#: Cap on the answer text handed to the model, in characters. A whole script is
#: small; this exists to bound a pathological OCR result rather than to save cost.
MAX_ANSWER_CHARS = 20_000


class Grader(Protocol):
    """Grades one answer, having been given the rubric and the lines to use."""

    name: str

    async def grade(
        self,
        *,
        question: Question,
        rubric: Rubric,
        index: LineIndex,
        line_ids: list[str],
    ) -> QuestionGrade: ...


def _unjudged_points(rubric: Rubric, *, comment: str) -> list[RubricPoint]:
    return [
        RubricPoint(
            point_id=f"{rubric.qid}#{i + 1}",
            criterion=criterion.criterion,
            marks_available=criterion.marks,
            marks_awarded=0.0,
            satisfied=False,
            cited_line_ids=[],
            comment=comment,
        )
        for i, criterion in enumerate(rubric.criteria)
    ]


class RubricOnly:
    """Produces the rubric and locates the answer, awarding nothing.

    Deliberately not a scoring heuristic. Keyword overlap would produce numbers
    that look like marks and are not, and a teacher shown a plausible wrong score
    is worse off than one shown none — the first visible mistake costs more than
    the help was worth.
    """

    name = "rubric_only"

    async def grade(
        self,
        *,
        question: Question,
        rubric: Rubric,
        index: LineIndex,
        line_ids: list[str],
    ) -> QuestionGrade:
        return QuestionGrade(
            qid=question.qid,
            marks_available=rubric.marks_available,
            marks_awarded=0.0,
            rubric_points=_unjudged_points(
                rubric, comment="Not marked automatically — awaiting the teacher."
            ),
            feedback=None,
            confidence=0.0,
            graded_on_partial_text=False,
        )


#: The shape the model must return. Written by hand rather than generated from the
#: contract so that the model is asked for exactly the fields it should decide —
#: it never sets identity, availability, or anything downstream code computes.
JUDGEMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based position of the rubric point being judged.",
                    },
                    "marks_awarded": {"type": "number", "minimum": 0},
                    "satisfied": {"type": "boolean"},
                    "cited_line_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Line IDs from inside the answer fence. Required "
                        "whenever marks are awarded.",
                    },
                    "comment": {"type": "string"},
                },
                "required": ["index", "marks_awarded", "satisfied", "cited_line_ids"],
            },
        },
        "feedback": {
            "type": "string",
            "description": "One or two sentences addressed to the student.",
        },
        "uncertain": {
            "type": "boolean",
            "description": "True when the transcription was too damaged to judge fairly.",
        },
    },
    "required": ["points", "uncertain"],
}


class ClaudeUnavailable(RuntimeError):
    """Raised when the Claude grader is asked for but cannot be built."""


class Claude:
    """Judges an answer with Claude, citing the lines behind each mark."""

    name = "claude"

    def __init__(self, *, model: str | None = None, client=None) -> None:
        self.model = model or DEFAULT_MODEL
        if client is not None:
            self._client = client
            return

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ClaudeUnavailable(
                "ANTHROPIC_API_KEY is not set, so answers cannot be marked automatically. "
                "The rubric and the located answer are still produced."
            )
        try:
            from anthropic import AsyncAnthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise ClaudeUnavailable(
                "the anthropic package is not installed; install the 'grading' extra"
            ) from exc
        self._client = AsyncAnthropic()

    async def grade(
        self,
        *,
        question: Question,
        rubric: Rubric,
        index: LineIndex,
        line_ids: list[str],
    ) -> QuestionGrade:
        # A drawing is not gradable from a transcription it does not have. Asking
        # anyway produces a confident zero for a correct answer.
        if not rubric.gradable_from_text:
            return QuestionGrade(
                qid=question.qid,
                marks_available=rubric.marks_available,
                marks_awarded=0.0,
                rubric_points=_unjudged_points(
                    rubric,
                    comment="Answered by a drawing. Needs a person to look at the page.",
                ),
                confidence=0.0,
                graded_on_partial_text=True,
            )

        message = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=prompt.SYSTEM,
            tools=[
                {
                    "name": "record_judgement",
                    "description": "Record the judgement for each rubric point.",
                    "input_schema": JUDGEMENT_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "record_judgement"},
            messages=[
                {
                    "role": "user",
                    "content": prompt.build(
                        question=question, rubric=rubric, index=index, line_ids=line_ids
                    ),
                }
            ],
        )

        judgement = _tool_input(message)
        return assemble(
            question=question,
            rubric=rubric,
            index=index,
            line_ids=line_ids,
            judgement=judgement,
        )


def _tool_input(message) -> dict:
    """The judgement from a tool-use response, whatever else the message holds."""
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            data = getattr(block, "input", None)
            if isinstance(data, str):
                return json.loads(data)
            if isinstance(data, dict):
                return data
    raise ValueError("the model returned no judgement")


def assemble(
    *,
    question: Question,
    rubric: Rubric,
    index: LineIndex,
    line_ids: list[str],
    judgement: dict,
) -> QuestionGrade:
    """Turn a model judgement into a grade, or refuse it.

    Separated from the transport so the validation can be tested without a
    network call — it is the part that decides whether a grade is trustworthy,
    which makes it the part most worth testing.

    A judgement whose citations do not hold is not repaired. Every point reverts
    to unjudged and the reason is recorded, because a grade assembled from
    partially fabricated evidence is not a smaller grade, it is an unfounded one.
    """
    allowed = set(line_ids)
    points: list[RubricPoint] = []

    for i, criterion in enumerate(rubric.criteria):
        raw = _point_for(judgement, i + 1)
        if raw is None:
            points.append(
                RubricPoint(
                    point_id=f"{rubric.qid}#{i + 1}",
                    criterion=criterion.criterion,
                    marks_available=criterion.marks,
                    marks_awarded=0.0,
                    satisfied=False,
                    cited_line_ids=[],
                    comment="The model did not judge this point.",
                )
            )
            continue

        awarded = max(0.0, min(float(raw.get("marks_awarded", 0.0)), criterion.marks))
        points.append(
            RubricPoint(
                point_id=f"{rubric.qid}#{i + 1}",
                criterion=criterion.criterion,
                marks_available=criterion.marks,
                marks_awarded=awarded,
                satisfied=bool(raw.get("satisfied", False)),
                cited_line_ids=[str(lid) for lid in raw.get("cited_line_ids", [])],
                comment=raw.get("comment"),
            )
        )

    problems = citations.check(points, index, allowed_line_ids=allowed)
    if problems:
        reason = "; ".join(str(p) for p in problems[:3])
        return QuestionGrade(
            qid=question.qid,
            marks_available=rubric.marks_available,
            marks_awarded=0.0,
            rubric_points=_unjudged_points(
                rubric, comment=f"Marking was refused — evidence did not check out ({reason})."
            ),
            feedback=None,
            confidence=0.0,
            graded_on_partial_text=True,
        )

    uncertain = bool(judgement.get("uncertain", False))
    return QuestionGrade(
        qid=question.qid,
        marks_available=rubric.marks_available,
        marks_awarded=sum(p.marks_awarded for p in points),
        rubric_points=points,
        feedback=judgement.get("feedback"),
        # Confidence is not asked of the model — a self-reported number is not
        # evidence. It is derived from whether the model flagged the transcription
        # as unreadable and from how much of the rubric it managed to cite.
        confidence=0.35 if uncertain else _cited_share(points),
        graded_on_partial_text=uncertain,
    )


def _point_for(judgement: dict, index: int) -> dict | None:
    for raw in judgement.get("points", []) or []:
        if isinstance(raw, dict) and raw.get("index") == index:
            return raw
    return None


def _cited_share(points: list[RubricPoint]) -> float:
    """Share of awarded points that pointed at specific lines.

    A grade whose marks are all traceable is one a teacher can check in seconds;
    one whose marks rest on unstated evidence needs re-reading. That difference is
    what the number is for.
    """
    awarded = [p for p in points if p.marks_awarded > 0]
    if not awarded:
        # Nothing awarded is a definite judgement, not an uncertain one: a zero
        # needs no citation to be checkable, since the whole answer is the
        # evidence.
        return 0.8
    cited = sum(1 for p in awarded if p.cited_line_ids)
    return round(0.5 + 0.5 * (cited / len(awarded)), 2)
