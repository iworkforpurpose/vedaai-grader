"""Marking answers so that every mark points at ink on the page.

Grading is the one place in this pipeline where a model's judgement is the
product rather than an input, so it is also the place where an unfounded output
is hardest to notice — a wrong score looks exactly like a right one. The package
is built around making that visible instead of trusting it.

Order:

1. ``rubric`` — derive credit-bearing criteria from what the paper printed. Rules,
   not a model: the marks and the number of items asked for have a ground truth.
2. ``citations`` — filter abandoned work out of the answer, and validate that
   every awarded mark cites a real line inside that answer.
3. ``prompt`` — assemble what the model sees: numbered lines, and student writing
   fenced as untrusted data.
4. ``engine`` — the graders. ``RubricOnly`` proposes no marks; ``Claude`` judges
   and cites. Both return the same type.
5. ``run`` — grade a submission, and record why anything unmarked was skipped.

The invariant worth stating once: a mark with no resolvable citation is refused,
not reduced. A grade assembled from partly invented evidence is not a smaller
grade — it is an unfounded one.
"""

from .citations import CitationProblem, check, gradable_lines
from .engine import Claude, ClaudeUnavailable, Grader, RubricOnly, assemble
from .rubric import Criterion, EvidenceKind, Rubric, derive
from .run import grade_submission

__all__ = [
    "CitationProblem",
    "Claude",
    "ClaudeUnavailable",
    "Criterion",
    "EvidenceKind",
    "Grader",
    "Rubric",
    "RubricOnly",
    "assemble",
    "check",
    "derive",
    "gradable_lines",
    "grade_submission",
]
