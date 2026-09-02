"""Validating that a grade points at real ink.

Every rubric point that awards marks must cite the lines that earned them, and
those IDs are checked here before a grade is shown to anyone. The check is the
same indirection the highlights rely on: the model names lines, never geometry,
so a citation either resolves to real writing or it does not exist.

Three properties follow, and they are the reason grading is safe to offer at all:

  * A teacher can click a rubric point and read the sentence behind it. A grade
    becomes checkable rather than merely plausible.
  * A fabricated justification fails validation instead of being displayed.
  * A citation outside the answer's own lines is rejected. Without that, a model
    could credit question 4 with writing that belongs to question 9 — including
    writing the student never intended as an answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from vedaai_contracts import LineIndex, RubricPoint


@dataclass(frozen=True)
class CitationProblem:
    """Why a grade was refused."""

    point_id: str
    line_id: str
    reason: str

    def __str__(self) -> str:
        return f"{self.point_id} cites {self.line_id}: {self.reason}"


def resolve(line_id: str, index: LineIndex) -> str:
    """The line a citation means, canonicalised where that is unambiguous.

    Exact matches pass through, and so does anything unresolvable — a citation
    that names no line must still reach :func:`check` and be refused, because
    that refusal is what stops a fabricated justification being displayed.

    Between those two sits one repair, and it exists because of four marks lost
    on a correct answer. Line IDs are zero-padded to a fixed width, and the model
    wrote ``as:00010`` for a line the index calls ``as:0010`` — the same number,
    padded to five digits instead of four. Nothing was invented and nothing was
    misread; the answer named a delta and named it correctly. But the ID did not
    match a string, so the citation check refused the whole question and the
    teacher was shown "the citation was invented", which was both a lost mark and
    a false accusation.

    So a numeric suffix is compared as a number rather than as text, and only
    where exactly one known line answers to it. Two candidates, a different
    prefix, or a non-numeric tail are all left alone for the check to reject: a
    repair that guesses is worse than a refusal, because it puts a mark behind
    the wrong sentence.
    """
    known = {line.line_id for line in index.lines}
    if line_id in known:
        return line_id

    prefix, separator, tail = line_id.rpartition(":")
    if not separator or not tail.isdigit():
        return line_id

    wanted = int(tail)
    matches = {
        candidate
        for candidate in known
        if (parts := candidate.rpartition(":"))[0] == prefix
        and parts[2].isdigit()
        and int(parts[2]) == wanted
    }
    if len(matches) == 1:
        return matches.pop()
    return line_id


def resolve_all(line_ids: list[str], index: LineIndex) -> list[str]:
    """:func:`resolve` over a citation list, order and duplicates preserved."""
    return [resolve(str(line_id), index) for line_id in line_ids]


def check(
    points: list[RubricPoint],
    index: LineIndex,
    *,
    allowed_line_ids: set[str],
) -> list[CitationProblem]:
    """Problems with a graded answer's citations. Empty means it is acceptable.

    ``allowed_line_ids`` is the answer's own lines, minus anything excluded from
    grading. Passing the whole index would let a grade cite the paper it was
    marked against, or another question's answer.
    """
    known = {line.line_id for line in index.lines}
    problems: list[CitationProblem] = []

    for point in points:
        for line_id in point.cited_line_ids:
            if line_id not in known:
                problems.append(
                    CitationProblem(
                        point_id=point.point_id,
                        line_id=line_id,
                        reason="no such line — the citation was invented",
                    )
                )
            elif line_id not in allowed_line_ids:
                problems.append(
                    CitationProblem(
                        point_id=point.point_id,
                        line_id=line_id,
                        reason="outside this answer, so it cannot evidence this mark",
                    )
                )

        # A mark with nothing behind it is the failure mode this whole mechanism
        # exists to prevent, so it is refused rather than merely flagged.
        if point.marks_awarded > 0 and not point.cited_line_ids:
            problems.append(
                CitationProblem(
                    point_id=point.point_id,
                    line_id="(none)",
                    reason="marks awarded with no line cited",
                )
            )

        if point.marks_awarded > point.marks_available:
            problems.append(
                CitationProblem(
                    point_id=point.point_id,
                    line_id="(none)",
                    reason=f"awarded {point.marks_awarded} of {point.marks_available} available",
                )
            )

    return problems


def gradable_lines(
    index: LineIndex,
    *,
    answer_line_ids: list[str],
    excluded: set[str],
) -> list[str]:
    """The answer's lines with struck-through and bleed-through writing removed.

    The guard from Finding B, applied at the point where text is chosen rather
    than left to the grader's discretion. A student who wrote a wrong answer,
    crossed it out and wrote the right one below must be marked on the version
    they kept. Filtering here also means the excluded lines are not merely
    ignored in scoring — they never reach the model, so they cannot influence it.
    """
    known = {line.line_id for line in index.lines}
    return [lid for lid in answer_line_ids if lid in known and lid not in excluded]
