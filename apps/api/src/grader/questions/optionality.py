"""Reading a paper's own rules about how many questions must be answered.

This is what makes the difference between "the student skipped question 6" and
"the student was entitled to skip question 6", and the two must never be conflated
— reporting a legitimately-exercised choice as an omission is a straightforward
product error, visible to any teacher who knows their own paper.

The phrasings below are quoted from official papers rather than guessed:

    ICSE:  "Attempt all questions from Section A and any four questions
            from Section B"
    VTU:   "Answer any FIVE full questions, choosing ONE full question
            from each module"
    CBSE:  "There is no overall choice. However, an internal choice has been
            provided in two questions"
    CISCE: "Attempt any two questions from this Section"

Numbers appear as digits and as words, in upper and mixed case, which is why the
word list exists.
"""

from __future__ import annotations

import re

from vedaai_contracts import Requirement

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_ANY_COUNT = re.compile(
    r"\bany\s+(?P<count>\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\b",
    re.IGNORECASE,
)

_ALL_REQUIRED = re.compile(
    r"\b(?:attempt|answer)\s+all\b|\bthere\s+is\s+no\s+overall\s+choice\b",
    re.IGNORECASE,
)

#: Section named in the same clause as a count, so "any four questions from
#: Section B" attaches its choice to B rather than to whichever section the
#: instruction happened to be printed under. On papers where the rubric sits on
#: the cover page, that distinction is the whole meaning of the sentence.
_NAMED_SECTION = re.compile(
    r"\bfrom\s+(?:SECTION|PART|GROUP)\s+([A-Z0-9]{1,3})\b", re.IGNORECASE
)


def parse_count(text: str) -> int | None:
    """The number in an "any N" clause, digits or words."""
    match = _ANY_COUNT.search(text)
    if match is None:
        return None
    raw = match.group("count").lower()
    if raw.isdigit():
        return int(raw)
    return _NUMBER_WORDS.get(raw)


def parse_requirement(text: str) -> tuple[str | None, Requirement] | None:
    """Read one instruction line.

    Returns the section it applies to — None meaning "the section this was
    printed under" — and the requirement. Returns None when the line states no
    rule about how many questions to answer.
    """
    count = parse_count(text)
    if count is not None:
        section_match = _NAMED_SECTION.search(text)
        section = section_match.group(1).upper() if section_match else None
        return section, Requirement(answer_any=count, source_text=text.strip())

    if _ALL_REQUIRED.search(text):
        section_match = _NAMED_SECTION.search(text)
        section = section_match.group(1).upper() if section_match else None
        return section, Requirement(answer_any=None, source_text=text.strip())

    return None


def parse_all(instruction_lines: list[str]) -> dict[str | None, Requirement]:
    """Collect requirements from every instruction on the paper.

    A single sentence can carry two rules — "attempt all questions from Section A
    and any four questions from Section B" — so clauses are split on "and" before
    parsing. Without that split the sentence yields only its first rule, and
    Section B silently becomes compulsory.

    Later statements win, on the assumption that a rubric printed beside a section
    is more specific than one on the cover page.
    """
    out: dict[str | None, Requirement] = {}
    for text in instruction_lines:
        for clause in re.split(r"\band\b", text, flags=re.IGNORECASE):
            parsed = parse_requirement(clause)
            if parsed is not None:
                section, requirement = parsed
                out[section] = requirement
    return out


def satisfied(requirement: Requirement, answered_count: int) -> bool:
    """Whether enough questions have been answered to meet a requirement."""
    if requirement.answer_any is None:
        return True  # every question is compulsory; nothing to satisfy early
    return answered_count >= requirement.answer_any
