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

from .numbering import (
    detect_section_prefixes,
    extract_marks,
    looks_like_a_question,
    parse_label,
)

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


#: How far below a heading the question text may sit.
#:
#: A geometry paper puts its figure between the heading and the question, and the
#: figure's labels transcribe as single letters and digits — "E", "D", "A", "B",
#: "P". Looking only at the very next line found "E", concluded the heading was a
#: stray, and left five questions of a real paper invisible.
#:
#: Twelve rather than eight because one figure on that paper carried exactly eight
#: labels — "D C 3 2 4 1 A B" — and a limit of eight stopped one line short of the
#: question, losing T4 while finding T3 and T5 either side of it. Reaching too far
#: is safe: the scan stops at the next label or section header regardless, so it
#: cannot walk into another question's text.
_HEADING_LOOKAHEAD = 12

#: Below this many characters a line is a fragment — a figure label, a stray mark —
#: rather than the question.
_FRAGMENT_MAX_CHARS = 4


def _body_from_following(
    body: str, following: tuple[str, ...], *, prefixes: frozenset[str]
) -> str:
    """The question text for a label that has none on its own line.

    Skips fragments on the way down, because a diagram sits between the heading
    and the question more often than not. Stops at anything that starts something
    else — another label, a section header — so a bare number at the foot of a
    page cannot reach forward and adopt the next question's text as its own.
    """
    for text in following[:_HEADING_LOOKAHEAD]:
        candidate = text.strip()
        if not candidate or len(candidate) <= _FRAGMENT_MAX_CHARS:
            continue
        if parse_label(candidate, prefixes=prefixes) is not None:
            break
        if _SECTION_HEADER.match(candidate):
            break
        if len(candidate) >= 12:
            return candidate
        break
    return body


def classify(
    line: Line,
    *,
    repeated: set[str],
    previous_role: LineRole | None,
    started: bool = True,
    following: tuple[str, ...] = (),
    prefixes: frozenset[str] = frozenset(),
) -> LineRole:
    """Decide what one line is.

    ``previous_role`` resolves the case no single line can: an unlabelled line is
    a continuation when it follows a question and furniture when it follows
    nothing.

    ``started`` says whether the paper has begun — whether any question or section
    header has appeared yet. Everything above the first one is the preamble, and a
    lettered line in the preamble is an instruction however much it looks like a
    question. A real paper opened a teacher's review screen with "(a) All questions
    are compulsory" and "(c) Draw neat diagrams" listed as questions and marked
    answered; "(b)" escaped only because it happened to contain the words "attempt
    any". Matching on vocabulary catches the instructions somebody thought of.
    Position catches the rest.

    Defaults to ``True`` so a caller classifying one line in isolation — which the
    tests do, and which is a reasonable thing to want — gets the judgement of the
    line itself rather than a silent assumption about where it sits.
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

    label = parse_label(text, prefixes=prefixes)
    if label is not None:
        body, _marks = extract_marks(label.remainder)

        # The heading layout: the label on its own line, the question below it.
        #
        # `Q1 (5 Marks)` leaves an empty body once the allocation is taken off,
        # and an empty body used to mean "a stray number". Eight of the eleven
        # real-world label styles this extractor dropped were dropped for that one
        # reason, including every question of the paper that started this work.
        #
        # What makes it a heading rather than a stray is the line underneath: real
        # question text, and not itself a label. A number alone at the foot of a
        # page has nothing following it and stays furniture.
        if not looks_like_a_question(label, body):
            body = _body_from_following(body, following, prefixes=prefixes)

        if looks_like_a_question(label, body):
            # Nothing has opened the paper yet, so this line either opens it or is
            # still preamble. A paper opens at a number — 1, Q1, 11 (a). An
            # enumerated instruction block opens at a letter, which is exactly the
            # shape that produced "(a) All questions are compulsory" as question
            # one. So a lettered label with no question above it is preamble, and
            # a numeric one is the paper starting.
            if not started and not label.tokens[0].isdigit():
                return LineRole.INSTRUCTION
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
    # Learned once from the whole document, because a single line cannot tell a
    # section letter from a coincidence — `T1` is only a label because `T2` and
    # `T3` are there too.
    prefixes = detect_section_prefixes([line.text for line in lines])
    roles: dict[str, LineRole] = {}
    previous: LineRole | None = None

    # Whether the paper proper has begun. A section header opens it as surely as a
    # question does — "SECTION A" means everything above it was preamble — and
    # papers exist that head straight into question 1 with no section at all.
    started = False

    for position, line in enumerate(lines):
        upcoming = tuple(
            ln.text for ln in lines[position + 1 : position + 1 + _HEADING_LOOKAHEAD]
        )
        role = classify(
            line,
            repeated=repeated,
            previous_role=previous,
            started=started,
            following=upcoming,
            prefixes=prefixes,
        )
        if role in {LineRole.QUESTION_START, LineRole.SECTION_HEADER}:
            started = True
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
