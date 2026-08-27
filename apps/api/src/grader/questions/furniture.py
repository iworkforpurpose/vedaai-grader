"""Telling questions apart from everything else printed on a question paper.

A paper carries a great deal of text that is not a question: running headers,
page numbers, rubric instructions, mark allocations, competency tags such as
``[Analysis & Evaluation]``, and notes like ``(for V.I. candidates)`` — all taken
from real CISCE and CBSE papers.

Getting this wrong is costly in both directions, and asymmetrically so. Swallowing
an instruction into a question corrupts the text that goes to grading. Discarding
a real question line loses a question outright, which is the graded criterion. So
the rules below are deliberately conservative about discarding: a line is only
called furniture on positive evidence, never merely for failing to look like a
question.
"""

from __future__ import annotations

import re
from collections import defaultdict

from vedaai_contracts import Line, LineRole

from .numbering import extract_marks, looks_like_a_question, parse_label

#: A section or part heading. Papers divide themselves this way and questions
#: often renumber across the boundary, so these are structural rather than noise.
_SECTION_HEADER = re.compile(
    r"^\s*(?:SECTION|PART|GROUP|MODULE)\s+([A-Z0-9]{1,3})\s*[-:.]?\s*$",
    re.IGNORECASE,
)

#: Rubric text. Matched by phrase because these are highly conventional; the
#: wording below is quoted from official papers rather than invented.
_INSTRUCTION_PHRASES = (
    "answer all",
    "answer any",
    "attempt all",
    "attempt any",
    "the marks for questions",
    "the intended marks",
    "maximum marks",
    "max. marks",
    "time allowed",
    "the time given at the head of this paper",
    "you are not allowed to write",
    "write the answers",
    "use black ink",
    "use blue or black",
    "do not use pencil",
    "draw diagrams in pencil",
    "you may ask for",
    "you are expected to use a calculator",
    "you are reminded of the need",
    "all working must be",
    "there is no overall choice",
    "an internal choice has been provided",
    "answers to this paper must be written",
    "read the following instructions",
    "candidates are advised",
    "turn over",
    "end of paper",
    "this question paper",
)

#: A bracketed aside whose content is words rather than a number. Competency tags
#: and candidate notes look like this; a mark allocation does not, and is handled
#: separately so its value can be kept.
_BRACKETED_ASIDE = re.compile(r"^\s*[\[(]\s*[^\d\])]{3,}\s*[\])]\s*$")

#: A line consisting only of a mark allocation.
_ONLY_MARKS = re.compile(r"^\s*[\[(]\s*\d{1,3}\s*(?:marks?|m)?\s*[\])]\s*$", re.IGNORECASE)

#: A bare page number, or "Page 3 of 12".
_PAGE_NUMBER = re.compile(
    r"^\s*(?:page\s+)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?\s*$", re.IGNORECASE
)

#: A page marker embedded in a longer line, as a running header carries it:
#: ``SCIENCE - UNIT TEST (page 2)`` or ``Page 2 of 12  Physics``.
#:
#: Deliberately anchored to a bracket or dash at one end of the line rather than
#: matching the words anywhere, because "Refer to the graph on page 2" is a
#: genuine question and its marker sits mid-sentence with no delimiter.
_PAGE_MARKER = re.compile(
    r"^\s*[\[(\-\u2013\u2014]?\s*p(?:age|g)\.?\s*\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?"
    r"|[\[(\-\u2013\u2014]\s*p(?:age|g)\.?\s*\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?\s*[\])]?\s*$",
    re.IGNORECASE,
)

#: How close two lines must be vertically to count as the same running header.
_HEADER_BAND = 0.02

#: A line must appear on at least this share of pages to be a running header.
_REPEAT_SHARE = 0.6

#: Pages needed before repetition alone is evidence. Three, because on a two-page
#: paper text appearing on both is as likely a genuine repeat as furniture.
_REPEAT_MIN_PAGES = 3

#: Distance from the top or bottom edge within which a *repeated* line is a
#: header or footer even on a two-page paper.
#:
#: Requiring three pages was too strict, and a real two-page paper showed why: its
#: page-2 header "SCIENCE - UNIT TEST (page 2)" was absorbed into the last
#: question on page 1, corrupting that question's text.
#:
#: Position supplies the missing evidence. Repetition alone is weak on two pages,
#: but repetition *at the page edge* is not — a rubric line such as "Answer all
#: questions" may legitimately recur once per section, and it recurs in the body,
#: not pinned to the margin.
_EDGE_BAND = 0.10


def find_repeated_lines(lines: list[Line]) -> set[str]:
    """Line IDs that look like running headers or footers.

    Detected by repetition at a consistent vertical position rather than by
    matching known header text, because a header is whatever a particular school
    chose to print at the top of its own paper.

    Position matters as much as repetition: "Answer all questions" may legitimately
    appear once per section in the body of a paper. Only text that recurs at the
    *same height* across most pages is furniture.
    """
    pages = {line.page for line in lines}
    if len(pages) < 2:
        return set()

    buckets: dict[tuple[str, int], list[Line]] = defaultdict(list)
    for line in lines:
        key = (line.text.strip().lower(), round(line.box.y0 / _HEADER_BAND))
        buckets[key].append(line)

    repeated: set[str] = set()
    threshold = max(_REPEAT_MIN_PAGES, int(len(pages) * _REPEAT_SHARE))

    for group in buckets.values():
        page_count = len({line.page for line in group})
        if page_count >= threshold:
            repeated.update(line.line_id for line in group)
        elif page_count >= 2 and all(_at_page_edge(line) for line in group):
            # Repeated at the page margin. Enough on a two-page paper, where
            # three-page repetition can never be observed.
            repeated.update(line.line_id for line in group)

    return repeated


def _at_page_edge(line: Line) -> bool:
    return line.box.y0 <= _EDGE_BAND or line.box.y1 >= (1.0 - _EDGE_BAND)


def classify(
    line: Line,
    *,
    repeated: set[str],
    previous_role: LineRole | None,
) -> LineRole:
    """Decide what one line is.

    ``previous_role`` resolves the case no single line can: an unlabelled line is
    a continuation when it follows a question and furniture when it follows
    nothing.
    """
    text = line.text.strip()
    if not text:
        return LineRole.FURNITURE

    if line.line_id in repeated:
        return LineRole.FURNITURE

    if _SECTION_HEADER.match(text):
        return LineRole.SECTION_HEADER

    if _ONLY_MARKS.match(text):
        return LineRole.MARKS

    if _PAGE_NUMBER.match(text):
        return LineRole.FURNITURE

    lowered = text.lower()
    if any(phrase in lowered for phrase in _INSTRUCTION_PHRASES):
        return LineRole.INSTRUCTION

    # Checked after instructions, so "(Attempt any two questions)" is recognised
    # as rubric rather than dismissed as a bracketed aside.
    if _BRACKETED_ASIDE.match(text):
        return LineRole.FURNITURE

    label = parse_label(text)
    if label is not None:
        body, _marks = extract_marks(label.remainder)
        if looks_like_a_question(label, body):
            return LineRole.QUESTION_START
        # A label with nothing usable after it — a stray number, or a
        # continuation marker. Not a question, and not worth discarding either.
        return LineRole.FURNITURE

    # A running header carrying its own page number. Checked here, after the label
    # test, so a genuine numbered question opening a page is never caught by it.
    #
    # This is the one header shape repetition cannot find: a header that prints
    # the page number differs on every page, so bucketing by text never groups it.
    # A real two-page paper showed the cost — "SCIENCE - UNIT TEST (page 2)" read
    # as a continuation of the last question on page 1 and was appended to its
    # text. Both conditions are needed: the marker rules out ordinary prose, and
    # the page edge rules out a mid-page reference to another page.
    if _at_page_edge(line) and _PAGE_MARKER.search(text):
        return LineRole.FURNITURE

    if previous_role in {LineRole.QUESTION_START, LineRole.QUESTION_CONTINUATION, LineRole.STEM}:
        return LineRole.QUESTION_CONTINUATION

    return LineRole.FURNITURE


def classify_all(lines: list[Line]) -> dict[str, LineRole]:
    """Classify every line in reading order."""
    repeated = find_repeated_lines(lines)
    roles: dict[str, LineRole] = {}
    previous: LineRole | None = None

    for line in lines:
        role = classify(line, repeated=repeated, previous_role=previous)
        roles[line.line_id] = role
        # Instructions and section headers interrupt a question's text, so they
        # must not leave a following unlabelled line looking like a continuation
        # of something several lines back.
        if role in {LineRole.SECTION_HEADER, LineRole.INSTRUCTION, LineRole.FURNITURE}:
            previous = None
        else:
            previous = role

    return roles


def section_label(text: str) -> str | None:
    """The section identifier from a header line, e.g. ``B`` from ``SECTION B``."""
    match = _SECTION_HEADER.match(text.strip())
    return match.group(1).upper() if match else None
