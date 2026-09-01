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


#: How close to the top or bottom edge a line must sit before a bare number reads
#: as a page number. Page numbers sit at either edge; identity details only at the
#: top, which is why the two bands are kept separate below.
_EDGE_BAND = 0.12


def _at_page_edge(line: Line) -> bool:
    return line.box.y0 <= _EDGE_BAND or line.box.y0 >= 1.0 - _EDGE_BAND


def is_furniture(line: Line) -> bool:
    """Whether this line is a detail about the script rather than an answer."""
    text = line.text.strip()
    if not text:
        return True

    # A labelled identity field is unambiguous wherever it appears: no answer is
    # phrased "Roll No: 41".
    if _IDENTITY_FIELD.search(text):
        return True

    in_header = line.box.y0 <= _HEADER_BAND

    # A bare number is a page number at the edge of a page and arithmetic
    # anywhere else. This test used to run before any position check, which made
    # every short number on the sheet furniture: on a mathematics script it
    # deleted 190, 25, 15, 2 and 3 -- pieces of 5(26)+3P=190, 25x=230 and 198/15
    # -- out of the working, and the grader then read an answer with holes in its
    # arithmetic as the student's own mistake.
    if (in_header or _at_page_edge(line)) and _PAGE_NUMBER.match(text):
        return True

    if not in_header:
        return False

    # The weaker patterns, trusted only at the top of a page.
    return bool(
        _PAPER_SET.match(text) or _LONG_NUMBER.search(text) or _BARE_NAME.match(text)
    )


#: How near two lines on different pages must be to count as the same place. The
#: vertical figure is tight because a printed or ruled box holds its height: a
#: roll-number box measured 0.087, 0.091, 0.091 and 0.087 down four scanned
#: pages. The horizontal one is loose because deskewing shifts a line sideways —
#: the same box started at 0.555, 0.565, 0.600 and 0.617.
_REPEAT_Y = 0.03
_REPEAT_X = 0.12

#: Longer than this and a repeated line is prose, not a label on a box. A guard
#: against deleting an answer, not a description of furniture.
_REPEAT_MAX_CHARS = 24


def _repeated_at_the_edge(lines: list[Line]) -> set[str]:
    """Line ids of short text that recurs in the same spot at the page edge.

    The patterns above all decide by reading words, and furniture is the text
    least likely to be read correctly: it is small, often boxed, and sits at the
    very edge of a scan where a camera loses most. One script's roll-number box
    came back as ``Rdi No: 37``, ``RdiNo: 37``, ``Rdino: 3`` and ``Rdl No: 37``
    across four pages, and ``Pagu:D``, ``Pag0回2``, ``Pago:回3``, ``Pagu:4``
    beneath it. Not one of the eight matches any identity field, so all eight
    survived into the answers.

    They cost more than untidiness. Anchors are read from a block's *first* line,
    so a header sitting above every answer means the student's own question
    number is the second line and is never read at all -- on that script their
    ``T2`` was invisible and the block drifted onto a question they had not
    attempted.

    What the words will not tell you, the geometry will: it is in the same place
    on every page. So this looks for position rather than content, and is fenced
    in three ways because the cost of deleting a real answer is a lost mark:
    only at the page edge, only short lines, and only where the repetition spans
    at least half the pages. An answer in the body of a page cannot be caught by
    it.
    """
    pages = {line.page for line in lines}
    if len(pages) < 2:
        # Repetition cannot be observed on one page, and guessing from a single
        # sample is how a real answer gets thrown away.
        return set()

    buckets: dict[tuple[int, int], list[Line]] = {}
    for line in lines:
        text = line.text.strip()
        if not text or len(text) > _REPEAT_MAX_CHARS or not _at_page_edge(line):
            continue
        key = (round(line.box.y0 / _REPEAT_Y), round(line.box.x0 / _REPEAT_X))
        buckets.setdefault(key, []).append(line)

    needed = max(2, len(pages) // 2)
    repeated: set[str] = set()
    for group in buckets.values():
        if len({line.page for line in group}) >= needed:
            repeated.update(line.line_id for line in group)
    return repeated


def strip(lines: list[Line]) -> tuple[list[Line], list[Line]]:
    """Split answer-sheet lines into answer text and script details.

    Both halves are returned rather than one discarded, so the caller can report
    what was set aside instead of it vanishing.
    """
    repeated = _repeated_at_the_edge(lines)
    answers: list[Line] = []
    details: list[Line] = []
    for line in lines:
        detail = line.line_id in repeated or is_furniture(line)
        (details if detail else answers).append(line)
    return answers, details
