"""Turning a printed question into credit-bearing criteria.

Deterministic on purpose. The marks a question carries and the number of things
it asks for are printed on the paper; deriving them by rule is exact, and asking
a model to restate them would introduce error into the one part of grading that
has a ground truth.

What a rule cannot do is judge whether an answer satisfies a criterion. That is
the model's job, and it is the only part delegated to one.

The command verb is extracted for a reason beyond phrasing. "Draw a labelled
diagram" is answered in ink, not words, so its transcription is empty or noise —
a model grading that text would mark a correct diagram wrong, and confidently.
Questions whose evidence is not textual are routed to a human instead of graded,
which is the same asymmetry the rest of the pipeline observes: admit uncertainty
rather than assert a wrong finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vedaai_contracts import Question

from ..questions.expects import EvidenceKind, evidence_kind

#: Number words papers use when asking for a fixed count of items.
_COUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}

#: "State two conditions", "Give any three examples", "List 2 uses".
_COUNTED_REQUEST = re.compile(
    r"\b(?:any\s+)?(one|two|three|four|five|six|[1-6])\s+"
    r"(?!marks?\b)([a-z][a-z-]{2,})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Criterion:
    """One thing the student had to do, and what it is worth."""

    criterion: str
    marks: float
    evidence: EvidenceKind

    @property
    def gradable_from_text(self) -> bool:
        """Whether a transcription can evidence this at all.

        A diagram cannot. Grading one from its (absent) text would produce a
        confident zero for a correct answer, which is worse than declining.
        """
        return self.evidence is not EvidenceKind.DRAWING


@dataclass(frozen=True)
class Rubric:
    """The criteria for one question."""

    qid: str
    criteria: list[Criterion]
    marks_available: float
    #: True when per-criterion marks were divided evenly because the paper did
    #: not state a split. The total is exact; the division is inferred.
    marks_split_inferred: bool

    @property
    def gradable_from_text(self) -> bool:
        return any(c.gradable_from_text for c in self.criteria)


def requested_count(text: str) -> int | None:
    """How many distinct items the question asks for, if it says.

    ``None`` means the question did not state a count, not that it wants one
    thing — the difference matters, because inventing a count would split marks
    against a structure the paper never claimed.
    """
    match = _COUNTED_REQUEST.search(text)
    if match is None:
        return None

    word = match.group(1).lower()
    count = _COUNTS.get(word) or int(word)
    return count if count > 1 else None


def derive(question: Question) -> Rubric:
    """Criteria for one question, from what the paper printed.

    Marks come from the paper. Where a question states how many items it wants,
    the marks divide across them; otherwise the question is one criterion, which
    is honest about the fact that the paper did not say more.
    """
    text = question.text.strip()
    kind = evidence_kind(text)
    available = float(question.marks or 0.0)
    count = requested_count(text)

    if count is None or count < 2:
        return Rubric(
            qid=question.qid,
            criteria=[Criterion(criterion=text, marks=available, evidence=kind)],
            marks_available=available,
            marks_split_inferred=False,
        )

    # An even split rounded to the half mark, since boards award those. Any
    # remainder lands on the first criterion rather than vanishing, so the
    # criteria always sum to the printed total.
    each = round((available / count) * 2) / 2
    marks = [each] * count
    marks[0] = round(available - each * (count - 1), 2)

    return Rubric(
        qid=question.qid,
        criteria=[
            Criterion(
                criterion=f"{text} ({ordinal(i + 1)} of {count})",
                marks=marks[i],
                evidence=kind,
            )
            for i in range(count)
        ],
        marks_available=available,
        marks_split_inferred=True,
    )


def ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}.get(
        n, f"{n}th"
    )
