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
    per_question_marks,
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

#: A heading's text points forward at the parts beneath it. Matched together with
#: a trailing colon, because that pairing is what separates a heading from a task
#: whose marks happen to be itemised below it.
#:
#: Structure alone is not enough, and a real paper showed why. "3. Write a program
#: that reads an array of 0s and 1s and prints the length of the longest run of
#: 1s." carries no marks — they sit on its (a) and (b) — but it is the task, and
#: calling it a heading would leave the question the student actually answered out
#: of the candidate list. "2. Answer the following about the program you wrote for
#: question 1:" is a heading, and the difference between them is not structural: it
#: is that one is self-contained and the other is meaningless without its parts.
#:
#: ``follows?`` and not ``follow``, and the missing ``s`` cost eight marks. A
#: history paper prints "Read the source below and answer the question that
#: follows." — singular — and the trailing ``\b`` on the alternation refused it,
#: because the boundary after "follow" falls between two word characters in
#: "follows". The plural form a geography paper happened to use matched, so the
#: rule looked complete. Both numbers are now accepted on both verbs.
_POINTS_AT_ITS_PARTS = re.compile(
    r"\b(?:the following"
    r"|both parts?|all parts?|each part"
    r"|the (?:parts?|questions?)\s+(?:below|that follows?|which follows?|given below)"
    r"|these questions?"
    r")\b",
    re.IGNORECASE,
)

#: A heading tells the student to answer what comes next. A task merely mentions
#: it. This is what stands in for the colon when a paper does not use one.
_INVITES_ANSWERS = re.compile(r"\b(?:answer|attempt|respond to)\b", re.IGNORECASE)


def reads_as_a_heading(text: str) -> bool:
    """Whether a line introduces other questions rather than asking one.

    Pointing at the parts is necessary but never sufficient. "Balance the following
    equation" points at something and is still a question, so a second signal has
    to say that the parts are what gets answered.

    A colon is one such signal. Requiring it was the whole rule, and a geography
    paper ended the sentence instead: "Study the sketch of the river below and
    answer the parts that follow." stayed an answerable question, sat in the
    candidate list beside its own (i) and (ii), and took the answer to (ii) — which
    was then reported uncertain on a question the student had answered in full.

    The invitation is the other signal, and the more direct one: a heading asks the
    student to *answer* what follows. "Balance the following equation." does not,
    and stays a question whichever mark ends it.

    Two callers now, and they use it for different things. ``mark_stems`` asks it of
    a *labelled* question, to decide whether the label heads its parts rather than
    asking anything. ``classify`` asks it of an *unlabelled* line, to decide whether
    the line interrupts the question above it. See the note at that call site.
    """
    stripped = text.strip()
    if _POINTS_AT_ITS_PARTS.search(stripped) is None:
        return False
    return stripped.endswith(":") or _INVITES_ANSWERS.search(stripped) is not None


#: An instruction that points at material printed beneath it: a source, a passage,
#: a table, a figure. What follows is content the questions refer to, not furniture.
#:
#: Matched separately from ``_INSTRUCTION_PHRASES`` because the consequence is
#: different. An ordinary rubric line is discarded and nothing is lost; these open a
#: scope, and everything inside it has to survive. A history paper's source extract
#: was thrown away as furniture and the question that asks "what does the source
#: suggest?" was then marked without the source.
_POINTS_AT_MATERIAL = re.compile(
    r"\b(?:"
    # "the following" needs a noun. "Read the following carefully." is a cover-page
    # rubric line and it opened a material scope that then swallowed "Each question
    # carries 4 marks" — so the paper's own denominators went missing and every
    # question on it was graded out of nothing.
    r"read the (?:source|passage|extract|text)"
    r"|read the following (?:source|passage|extract|text|table|figure|carefully and)"
    r"|study the (?:source|passage|extract|sketch|figure|diagram|map|table|graph)"
    r"|(?:the )?(?:table|figure|diagram|sketch|graph|map|source|passage|extract)\s+"
    r"(?:below|above|given below)"
    r"|refer to the (?:table|figure|diagram|sketch|graph|map|source|passage)"
    r"|based on the (?:source|passage|extract|table|figure)"
    r")\b",
    re.IGNORECASE,
)

#: A line that is itself printed material rather than prose about it: a quotation,
#: or a short cell from a table.
_QUOTED = re.compile("^\\s*[\"\u201c\u2018']")
_TABLE_CELL = re.compile("^\\s*[\\d.,%$\u20b9-]{1,12}\\s*$")


def points_at_material(text: str) -> bool:
    """Whether this line introduces material printed beneath it."""
    return _POINTS_AT_MATERIAL.search(text.strip()) is not None


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


def _body_offset(following: tuple[str, ...], *, prefixes: frozenset[str]) -> int | None:
    """How far below a bare label its question text starts, if it does.

    Skips fragments on the way down, because a diagram sits between the heading
    and the question more often than not. Stops at anything that starts something
    else — another label, a section header — so a bare number at the foot of a
    page cannot reach forward and adopt the next question's text as its own.

    Returned as a position rather than as the text itself, because the lines
    skipped over are the figure and the caller needs to know which ones they were.
    """
    for offset, text in enumerate(following[:_HEADING_LOOKAHEAD]):
        candidate = text.strip()
        if not candidate or len(candidate) <= _FRAGMENT_MAX_CHARS:
            continue
        if parse_label(candidate, prefixes=prefixes) is not None:
            break
        if _SECTION_HEADER.match(candidate):
            break
        if len(candidate) >= 12:
            return offset
        break
    return None


def _body_from_following(
    body: str, following: tuple[str, ...], *, prefixes: frozenset[str]
) -> str:
    """The question text for a label that has none on its own line."""
    offset = _body_offset(following, prefixes=prefixes)
    return following[offset].strip() if offset is not None else body


def classify(
    line: Line,
    *,
    repeated: set[str],
    previous_role: LineRole | None,
    started: bool = True,
    following: tuple[str, ...] = (),
    prefixes: frozenset[str] = frozenset(),
    material_open: bool = False,
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

    lowered_early = text.lower()

    # Inside an open material scope, anything that is not plainly a new question, a
    # section boundary or a piece of rubric is the material itself. Checked before
    # the *discarding* rules, because that is where the content was being lost — a
    # quoted source line is not a labelled question, so every rule below it
    # eventually called it furniture, and a table's numbers matched the
    # bare-page-number pattern exactly.
    #
    # But after the rubric rules, and that ordering is load-bearing. A scope opened
    # on the cover page otherwise swallows "Answer all questions. Each question
    # carries 4 marks." — which is where the marks come from, so every question on
    # the paper ends up graded out of nothing.
    if material_open and not _SECTION_HEADER.match(text):
        rubric = any(phrase in lowered_early for phrase in _INSTRUCTION_PHRASES) or (
            per_question_marks(text) is not None
        )
        if not rubric and parse_label(text, prefixes=prefixes) is None:
            return LineRole.MATERIAL

    if _QUOTED.match(text):
        # A quotation is printed material wherever it sits. Nothing a paper asks is
        # phrased inside quote marks from the first character.
        return LineRole.MATERIAL

    if _SECTION_HEADER.match(text):
        return LineRole.SECTION_HEADER

    if _ONLY_MARKS.match(text):
        return LineRole.MARKS

    if _PAGE_NUMBER.match(text):
        return LineRole.FURNITURE

    lowered = text.lower()
    if any(phrase in lowered for phrase in _INSTRUCTION_PHRASES):
        return LineRole.INSTRUCTION

    # A marks allocation stated for a whole section is rubric, and the section's
    # questions are graded out of nothing without it. It needs naming here rather
    # than in the phrase list above because the phrase that identifies it is a
    # shape — "each ... N marks" — not a fixed string.
    #
    # Both rules below would otherwise discard it. On the science paper,
    # "(Each question carries 1 mark)" and "(Each question carries 3 marks)" were
    # dismissed as bracketed asides, and only SECTION C's directive survived,
    # because it happens to also say "attempt any three questions".
    if per_question_marks(text) is not None:
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

    # An unlabelled line that introduces what comes after it, rather than
    # continuing what came before.
    #
    # This is the fault that cost the most measured marks of anything in the
    # pipeline. A history paper prints a source extract between Q.3(b) and Q.4:
    #
    #     Q.3 (b)  Give one example of it breaking down before 1914.
    #     Read the source below and answer the question that follows.
    #     "We were told the war would be over by Christmas. ..."
    #     Q.4  What does the source suggest about how opinion at home changed?
    #
    # Nothing interrupted the chain. The instruction carries no label, matches no
    # fixed rubric phrase, and followed a QUESTION_START — so it and all three
    # quoted lines became continuations of Q.3(b), whose text grew to include the
    # entire source. That polluted text then out-competed the real Q.4 for the
    # answer about the source *and* absorbed Q.3's own answer from a later page.
    # Three questions reported wrong, eight of the paper's twenty earned marks
    # gone, and it presented as a marking failure rather than an extraction one.
    #
    # Checked after the label test, so a labelled heading — "2. Answer the
    # following:" — is still a QUESTION_START and still becomes a stem later.
    # Only a line with no label of its own reaches here, and such a line cannot be
    # a question, so calling it rubric costs nothing even when the guess is wrong:
    # the worst case is one instruction line not appended to the question above it.
    if reads_as_a_heading(text):
        return LineRole.INSTRUCTION

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

    # Lines a heading has already accounted for as its figure — see below.
    inside_a_figure: set[int] = set()

    # Whether a line that points at material has been seen and its content has not
    # yet been closed off by a question or a section boundary.
    material_open = False

    for position, line in enumerate(lines):
        upcoming = tuple(
            ln.text for ln in lines[position + 1 : position + 1 + _HEADING_LOOKAHEAD]
        )
        if position in inside_a_figure:
            # The figure a heading stepped over. Material rather than furniture: it
            # is content the question refers to, and a geometry paper's "A", "B" and
            # "N" are the only readable trace of the sketch its parts ask about.
            # It still does not interrupt anything.
            roles[line.line_id] = LineRole.MATERIAL
            continue

        role = classify(
            line,
            repeated=repeated,
            previous_role=previous,
            started=started,
            following=upcoming,
            prefixes=prefixes,
            material_open=material_open,
        )
        if role in {LineRole.QUESTION_START, LineRole.SECTION_HEADER}:
            started = True
        roles[line.line_id] = role

        # A line pointing at material opens the scope; a question or a section
        # closes it. Question text itself may open one — "The table below shows the
        # price and quantity demanded of wheat" is a stem whose rows follow it — so
        # the check runs after classification rather than instead of it.
        if points_at_material(line.text):
            material_open = True
        elif role in {LineRole.QUESTION_START, LineRole.SECTION_HEADER}:
            material_open = False

        # A label whose own line carries no question text is a heading, and what
        # sits between it and its text is a diagram. Classification already worked
        # that out — it is why the heading is a question at all — and the lines it
        # stepped over are named here so extraction steps over them too.
        #
        # Page 3 of a real mathematics paper: "T4 (5 Marks)", a quadrilateral with
        # its vertices labelled D, C, 3, 2, A, B, then the question. T4 came out
        # with the text "D C" — the first row of vertex labels, read as the start
        # of its text — and the sentence that was actually the question came out
        # as nothing at all, because the labels below broke the chain and left
        # every line under them, the question included, looking like furniture.
        if role is LineRole.QUESTION_START:
            parsed = parse_label(line.text, prefixes=prefixes)
            body, _marks = extract_marks(parsed.remainder) if parsed else ("x", None)
            if parsed is not None and not looks_like_a_question(parsed, body):
                offset = _body_offset(upcoming, prefixes=prefixes)
                if offset:
                    inside_a_figure.update(
                        range(position + 1, position + 1 + offset)
                    )

        # Instructions and section headers interrupt a question's text, so they
        # must not leave a following unlabelled line looking like a continuation
        # of something several lines back.
        if role in {
            LineRole.SECTION_HEADER,
            LineRole.INSTRUCTION,
            LineRole.FURNITURE,
            LineRole.MATERIAL,
        }:
            previous = None
        else:
            previous = role

    return roles


def section_label(text: str) -> str | None:
    """The section identifier from a header line, e.g. ``B`` from ``SECTION B``."""
    match = _SECTION_HEADER.match(text.strip())
    return match.group(1).upper() if match else None
