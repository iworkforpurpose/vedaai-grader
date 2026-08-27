"""Checking an extracted paper against its own numbering.

A paper numbers its questions, which means the paper carries a checksum for the
extraction. If a run produces 1, 2, 4 then either question 3 exists and was
missed, or the paper genuinely skips it. Both deserve a teacher's attention and
only one is our bug — but silently continuing serves neither.

This is the cheapest accuracy safeguard available. It needs no ground truth, no
model and no configuration: the evidence is already printed on the page.
"""

from __future__ import annotations

from vedaai_contracts import NumberingGap, Question


def _numeric(token: str) -> int | None:
    return int(token) if token.isdigit() else None


def find_gaps(questions: list[Question]) -> list[NumberingGap]:
    """Missing numbers in each numeric sequence on the paper.

    Sequences are compared per level and per section, so a paper restarting its
    numbering in each section is not reported as one enormous gap, and sub-parts
    of question 4 are not compared against sub-parts of question 5.

    The subtlety that broke the first version: a question can exist only through
    its children. A paper printing ``2 (i)``, ``2 (i) (a)`` and ``2 (ii)`` has no
    standalone entry for 2 at all, so collecting only the *final* token of each
    path saw the top level as 1, 3, 4, 6 and reported 2 and 5 as missing on every
    paper. Tokens are therefore gathered by position within the path, which counts
    a parent that appears solely as a prefix.

    Only numeric tokens are checked. Alphabetic and roman sequences are skipped
    deliberately: deciding whether ``(i)`` follows ``(h)`` or begins a new roman
    sequence is the same ambiguity the parser refuses to rule on, and guessing
    would manufacture gaps that do not exist.
    """
    # (section, prefix) -> token at the next position -> an example question
    levels: dict[tuple[str | None, tuple[str, ...]], dict[str, Question]] = {}
    for question in questions:
        path = tuple(question.path)
        for depth in range(len(path)):
            key = (question.section_id, path[:depth])
            levels.setdefault(key, {}).setdefault(path[depth], question)

    gaps: list[NumberingGap] = []
    for (_section, _prefix), tokens in sorted(levels.items(), key=lambda kv: str(kv[0])):
        numbered = sorted(
            (int(token), example)
            for token, example in tokens.items()
            if token.isdigit()
        )
        if len(numbered) < 2:
            continue

        for (previous_value, previous_q), (value, question) in zip(
            numbered, numbered[1:], strict=False
        ):
            for missing in range(previous_value + 1, value):
                gaps.append(
                    NumberingGap(
                        expected_label=_expected_label(previous_q, previous_value, missing),
                        after_qid=previous_q.qid,
                        before_qid=question.qid,
                    )
                )

    return gaps


def _expected_label(neighbour: Question, present_value: int, missing: int) -> str:
    """Describe the missing question the way the paper would have printed it.

    Built from a neighbour's own label so the description matches the paper's
    notation, which is what makes it useful to a teacher checking whether the
    question really exists.
    """
    raw = neighbour.label_raw
    token = str(present_value)
    if token in raw:
        return raw.replace(token, str(missing), 1)
    return str(missing)


def suspicious(paper_questions: list[Question]) -> list[str]:
    """Extraction results worth a second look.

    Reported rather than corrected. Each of these is a pattern that shows up both
    when extraction went wrong and when a paper is simply unusual, so the useful
    action is to surface them, not to guess.
    """
    problems: list[str] = []
    if not paper_questions:
        return ["no questions were extracted at all"]

    seen: dict[str, int] = {}
    for question in paper_questions:
        seen[question.qid] = seen.get(question.qid, 0) + 1
    duplicates = [qid for qid, count in seen.items() if count > 1]
    if duplicates:
        problems.append(
            "duplicate question ids, so two questions share an identity: "
            + ", ".join(sorted(duplicates)[:6])
        )

    empty = [q.label_raw for q in paper_questions if not q.text.strip()]
    if empty:
        problems.append(
            "questions with no text, which usually means a label was matched in "
            "running text: " + ", ".join(empty[:6])
        )

    very_long = [q.label_raw for q in paper_questions if len(q.text) > 1200]
    if very_long:
        problems.append(
            "questions with implausibly long text, which usually means a following "
            "question's label was missed and its lines were absorbed: "
            + ", ".join(very_long[:6])
        )

    return problems
