"""Grading engines behind one interface.

Three, for the same reason the transcription layer has more than one: the
interesting decisions should be measurable against each other rather than assumed.
Here the split also does something else — it keeps grading useful when no model is
configured at all.

``RubricOnly`` proposes no marks. It derives the rubric from the paper, locates
the answer, and hands the teacher both with every point unjudged. That is a
genuine product: a marking aid that structures the work without inventing a
score. It is also the honest fallback, because a grade nobody can justify is
worth less than no grade.

``Claude`` and ``OpenAIGrader`` judge the answer and cite the lines behind each
mark. Their output is validated before it is shown; a fabricated or out-of-scope
citation invalidates the grade rather than being quietly dropped.

Which one runs is configuration, and the answer is recorded on every grade. That
matters more with two providers than it did with one: a mark is only checkable if
you know what made it, and a small model and a large one are not interchangeable
evidence.

**On using a small model here.** It is a reasonable thing to try, and the reason is
structural rather than optimistic. This task is not a reasoning showcase — it is
"read a rubric, read numbered lines, decide, and cite the line IDs" — and the two
ways a weak model fails it are both already contained. Malformed output is
prevented by asking for a schema rather than for prose. Invented citations are
caught by validation, which refuses the grade instead of displaying it, so the
failure mode is *no mark* rather than a wrong one.

What is not contained is judgement: deciding whether a student's own words satisfy
a criterion, in text a recognizer has already damaged. Nothing in the architecture
can rescue that, and it is the part worth measuring before trusting the numbers.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from vedaai_contracts import LineIndex, Question, QuestionGrade, RubricPoint

from . import citations, prompt
from .rubric import Rubric

#: Default model per provider, used when GRADER_MODEL names none.
#:
#: OpenAI's default is the small one deliberately. Marking is a short, highly
#: constrained call — a rubric, a few lines, a schema to fill — and the expensive
#: part of a large model is capability this task mostly does not use. If it proves
#: not good enough, that shows up in the grades rather than in the bill.
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
}

#: Explicit provider choice. With nothing set, whichever key is present is used —
#: convenient, and safe only because the provider is recorded on every grade, so
#: "which engine judged this" is never a guess.
GRADER_PROVIDER = os.getenv("GRADER_PROVIDER", "").strip().lower()

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
            graded_by="rubric_only",
            graded_on_partial_text=False,
        )


#: Fixed sampling seed, so two runs over one script agree.
#:
#: A constant rather than a config knob: there is no version of this product where
#: a teacher wants marking to vary run to run, and a configurable seed would be a
#: setting whose only effect is to make results harder to compare.
GRADER_SEED = 20240817


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


class GraderUnavailable(RuntimeError):
    """Raised when a model-backed grader is asked for but cannot be built.

    Not fatal anywhere it is raised: the caller falls back to the rubric-only
    grader, which is a working product rather than an error path.
    """


class Claude:
    """Judges an answer with Claude, citing the lines behind each mark."""

    name = "claude"

    def __init__(self, *, model: str | None = None, client=None) -> None:
        self.model = model or os.getenv("GRADER_MODEL") or DEFAULT_MODELS["anthropic"]
        if client is not None:
            self._client = client
            return

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise GraderUnavailable(
                "ANTHROPIC_API_KEY is not set, so answers cannot be marked automatically. "
                "The rubric and the located answer are still produced."
            )
        try:
            from anthropic import AsyncAnthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise GraderUnavailable(
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
            return _needs_a_person(rubric, question, graded_by=self.provenance)

        message = await self._client.messages.create(
            model=self.model,
            # Same reason as the OpenAI path: a mark that changes between identical
            # runs is not checkable. This provider exposes no seed.
            temperature=0.0,
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
            graded_by=self.provenance,
        )

    @property
    def provenance(self) -> str:
        return f"anthropic:{self.model}"


#: The same judgement, expressed for OpenAI's structured-output mode.
#:
#: A separate schema rather than a shared one, because strict mode is genuinely
#: stricter and the differences are not cosmetic: every property must appear in
#: ``required``, ``additionalProperties`` must be false, and numeric bounds like
#: ``minimum`` are not supported. An optional field is therefore expressed as a
#: nullable type instead of an absent key.
#:
#: Worth the duplication. Strict mode guarantees the response parses and matches,
#: which removes the failure a small model is most likely to produce.
STRICT_JUDGEMENT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["points", "feedback", "uncertain"],
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "marks_awarded", "satisfied", "cited_line_ids", "comment"],
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based position of the rubric point being judged.",
                    },
                    "marks_awarded": {"type": "number"},
                    "satisfied": {"type": "boolean"},
                    "cited_line_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Line IDs copied from inside the answer fence. "
                        "Required whenever marks are awarded.",
                    },
                    "comment": {"type": ["string", "null"]},
                },
            },
        },
        "feedback": {
            "type": ["string", "null"],
            "description": "One or two sentences addressed to the student.",
        },
        "uncertain": {
            "type": "boolean",
            "description": "True when the transcription was too damaged to judge fairly.",
        },
    },
}


class OpenAIGrader:
    """Judges an answer with an OpenAI model, citing the lines behind each mark."""

    name = "openai"

    def __init__(self, *, model: str | None = None, client=None) -> None:
        self.model = model or os.getenv("GRADER_MODEL") or DEFAULT_MODELS["openai"]
        if client is not None:
            self._client = client
            return

        if not os.getenv("OPENAI_API_KEY"):
            raise GraderUnavailable(
                "OPENAI_API_KEY is not set, so answers cannot be marked automatically. "
                "The rubric and the located answer are still produced."
            )
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise GraderUnavailable(
                "the openai package is not installed; install the 'grading' extra"
            ) from exc
        self._client = AsyncOpenAI()

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
            return _needs_a_person(rubric, question, graded_by=self.provenance)

        completion = await self._client.chat.completions.create(
            model=self.model,
            # Marking the same script twice must give the same marks.
            #
            # This was unset, and the API default is 1.0 — so re-marking one
            # submission five times produced totals of 3, 3, 2, 2 and 0. A teacher
            # cannot check a mark that changes when they look again, and it makes
            # every accuracy figure measured through this path noise.
            #
            # A seed as well, which asks the provider for reproducible sampling. It
            # is best-effort rather than guaranteed, so temperature is the load
            # bearing part; the seed narrows what is left.
            temperature=0.0,
            seed=GRADER_SEED,
            messages=[
                {"role": "system", "content": prompt.SYSTEM},
                {
                    "role": "user",
                    "content": prompt.build(
                        question=question, rubric=rubric, index=index, line_ids=line_ids
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judgement",
                    "strict": True,
                    "schema": STRICT_JUDGEMENT_SCHEMA,
                },
            },
        )

        content = completion.choices[0].message.content
        if not content:
            raise ValueError("the model returned no judgement")

        return assemble(
            question=question,
            rubric=rubric,
            index=index,
            line_ids=line_ids,
            judgement=json.loads(content),
            graded_by=self.provenance,
        )

    @property
    def provenance(self) -> str:
        return f"openai:{self.model}"


def select_grader() -> Grader:
    """The grader this deployment should use.

    An explicit ``GRADER_PROVIDER`` wins. Otherwise whichever key is present is
    used, and if neither is, the rubric-only grader — which is a working product,
    not an error path: it structures the marking without inventing a score.
    """
    providers: list[tuple[str, type]] = [("anthropic", Claude), ("openai", OpenAIGrader)]
    if GRADER_PROVIDER == "openai":
        providers.reverse()
    elif GRADER_PROVIDER == "none":
        return RubricOnly()

    reasons: list[str] = []
    for _name, engine in providers:
        try:
            return engine()
        except GraderUnavailable as unavailable:
            reasons.append(str(unavailable))

    raise GraderUnavailable(" ".join(reasons))


def _needs_a_person(rubric: Rubric, question: Question, *, graded_by: str) -> QuestionGrade:
    """A question whose answer is a drawing, left for someone who can see it."""
    return QuestionGrade(
        qid=question.qid,
        marks_available=rubric.marks_available,
        marks_awarded=0.0,
        rubric_points=_unjudged_points(
            rubric, comment="Answered by a drawing. Needs a person to look at the page."
        ),
        confidence=0.0,
        graded_by=graded_by,
        graded_on_partial_text=True,
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
    graded_by: str | None = None,
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
            graded_by=graded_by,
            graded_on_partial_text=True,
        )

    uncertain = bool(judgement.get("uncertain", False))
    return QuestionGrade(
        qid=question.qid,
        marks_available=rubric.marks_available,
        marks_awarded=sum(p.marks_awarded for p in points),
        rubric_points=points,
        # The only place this is true. Every other constructor in this module
        # returns a grade nobody judged, and the difference is not otherwise
        # recoverable from the payload — see the field's own note.
        judged=True,
        feedback=judgement.get("feedback"),
        graded_by=graded_by,
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
