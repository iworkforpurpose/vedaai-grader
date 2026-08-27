"""Telling a student's answers apart from the details they wrote at the top.

Every script starts with something that is not an answer: a name, a class, a roll
number, a phone number, "Set 3". A question paper has had its headers and rubric
stripped since the beginning; an answer sheet had nothing, so those lines became
candidate answers.

That is not a cosmetic problem. On the golden set the line "Name: Test Student
Class: 6C" was assigned to "Describe an experiment to show that air has mass" —
a question the student had left blank. The mapping was scored as a false answer,
and a teacher would have seen a name badge highlighted as though it were an
experiment. On the real scripts the same thing happens with "Abhinami Anil / S8
CSE A1" and "Anjana S Kamath / BTECH-CSE A-S8 / 6282350749".

The asymmetry is the same one this codebase observes everywhere: discarding a real
answer is far worse than keeping a stray line, so a line is only rejected on
positive evidence — a recognisable identity field, or a bare page number — and
never for merely looking short or odd. Position is required as well for the weaker
patterns, since a phone number could conceivably be an answer further down a page
and cannot be one in the header.
"""

from __future__ import annotations

import re

from vedaai_contracts import Line

#: How far down a page the header band extends. Identity details are written at
#: the very top, above the first answer.
_HEADER_BAND = 0.12

#: Fields a student fills in about themselves. Matched as labelled fields — the
#: word followed by a separator — because "name" and "class" are also ordinary
#: words that appear inside real answers ("name the type of reaction", "the class
#: of the compound").
_IDENTITY_FIELD = re.compile(
    r"\b(?:"
    r"name|student\s*name|candidate|"
    r"class|section|semester|sem|branch|course|department|dept|"
    r"roll\s*(?:no|number)?|reg(?:ister|istration)?\s*(?:no|number)?|"
    r"enroll?ment(?:\s*no)?|admission\s*no|seat\s*no|"
    r"subject|date|batch|div(?:ision)?"
    r")\s*[:\-–]\s*\S",
    re.IGNORECASE,
)

#: "Set 1", "Set B" — which variant of the paper this script answers.
_PAPER_SET = re.compile(r"^\s*set\s*[-:]?\s*[A-Z0-9]{1,3}\s*$", re.IGNORECASE)

#: A bare page number, alone on its line.
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s*)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?\s*$", re.IGNORECASE)

#: A run of digits long enough to be a phone or enrolment number and too long to
#: be an answer's arithmetic.
_LONG_NUMBER = re.compile(r"\b\d{8,}\b")

#: A line that is only a person's name: two or three capitalised words and
#: nothing else. Deliberately narrow, and only trusted in the header band, since
#: "Ohms Law" would otherwise match.
_BARE_NAME = re.compile(r"^\s*(?:[A-Z][a-z.]{1,15}\s+){1,3}[A-Z][a-z.]{1,15}\s*$")


def is_furniture(line: Line) -> bool:
    """Whether this line is a detail about the script rather than an answer."""
    text = line.text.strip()
    if not text:
        return True

    # A labelled identity field is unambiguous wherever it appears: no answer is
    # phrased "Roll No: 41".
    if _IDENTITY_FIELD.search(text):
        return True

    if _PAGE_NUMBER.match(text):
        return True

    in_header = line.box.y0 <= _HEADER_BAND
    if not in_header:
        return False

    # The weaker patterns, trusted only at the top of a page.
    return bool(
        _PAPER_SET.match(text) or _LONG_NUMBER.search(text) or _BARE_NAME.match(text)
    )


def strip(lines: list[Line]) -> tuple[list[Line], list[Line]]:
    """Split answer-sheet lines into answer text and script details.

    Both halves are returned rather than one discarded, so the caller can report
    what was set aside instead of it vanishing.
    """
    answers: list[Line] = []
    details: list[Line] = []
    for line in lines:
        (details if is_furniture(line) else answers).append(line)
    return answers, details
