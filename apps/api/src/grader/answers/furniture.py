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
    r"subject|date|batch|div(?:ision)?|"
    # A labelled page field, and it belongs here rather than with the positional
    # rules below. On a real mathematics script "Page : 02" and "Page : 03" sat at
    # y=0.129 and y=0.130 against a header band of 0.12 — outside it by one
    # hundredth of the page — so every position-gated rule declined to look, and
    # `_PAGE_NUMBER` would have refused them anyway because it requires the digits
    # to follow "page" directly and these carry " : " between.
    #
    # Both became the *first line* of a block. That is worse than one stray line
    # highlighted: anchors are read from a block's first line, so a header sitting
    # there makes the student's own question number the second line and it is
    # never read at all — which is the failure `_repeated_at_the_edge` was written
    # to prevent, arriving by a different route.
    #
    # As a labelled field it needs no position at all, which is the point of this
    # pattern: "Page : 3" is a header wherever it appears, and an answer that
    # mentions a page does not punctuate it with a colon.
    r"page\s*(?:no|number)?|p\.?\s*no"
    r")\s*[:\-–]\s*\S",
    re.IGNORECASE,
)

#: A field label whose value recognition put on a different line.
#:
#: ``_IDENTITY_FIELD`` requires something after the colon, which is right for
#: telling "Date: 4 March" from a sentence containing the word date. But a scan
#: splits a boxed header down the middle: one real script gave ``Page :`` on its
#: own line with the ``02`` beside it as a separate line, so the field matched
#: nothing and led a block. A label with *no* value is not ambiguous — no answer
#: consists of the word "Page" and a colon — so it needs no value to be trusted.
_BARE_FIELD_LABEL = re.compile(
    r"^\s*(?:name|candidate|class|section|semester|sem|branch|course|department"
    r"|dept|roll\s*(?:no|number)?|reg(?:ister|istration)?\s*(?:no|number)?"
    r"|enroll?ment(?:\s*no)?|admission\s*no|seat\s*no|subject|date|batch"
    r"|div(?:ision)?|page\s*(?:no|number)?|p\.?\s*no)"
    r"\s*[:\-–]?\s*$",
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


def is_furniture(line: Line, *, has_writing_beside_it: bool = False) -> bool:
    """Whether this line is a detail about the script rather than an answer.

    ``has_writing_beside_it`` says another line shares this one's row further
    right. It is what separates a page number from a question number written in
    the margin — see the note on the page-number test below.
    """
    text = line.text.strip()
    if not text:
        return True

    # A labelled identity field is unambiguous wherever it appears: no answer is
    # phrased "Roll No: 41".
    if _IDENTITY_FIELD.search(text) or _BARE_FIELD_LABEL.match(text):
        return True

    in_header = line.box.y0 <= _HEADER_BAND

    # A bare number is a page number at the edge of a page and arithmetic
    # anywhere else. This test used to run before any position check, which made
    # every short number on the sheet furniture: on a mathematics script it
    # deleted 190, 25, 15, 2 and 3 -- pieces of 5(26)+3P=190, 25x=230 and 198/15
    # -- out of the working, and the grader then read an answer with holes in its
    # arithmetic as the student's own mistake.
    #
    # And a question number written in the margin is also a bare number near the
    # top of a page, which is why the row test is here. On a three-page physics
    # script the student's own "1", "3" and "5" each sat at y≈0.09 beside the
    # first line of the answer they labelled, matched this pattern, and were
    # deleted — so the strongest mapping signal available, the student saying
    # which question this answers, was thrown away on every page.
    #
    # A page number sits alone on its row. A margin label has the answer it
    # labels beside it. That is the whole difference, and it is geometric rather
    # than textual, which is what makes it reliable on a bare digit.
    if (
        (in_header or _at_page_edge(line))
        and _PAGE_NUMBER.match(text)
        and not has_writing_beside_it
    ):
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
        if len({line.page for line in group}) < needed:
            continue
        # Position alone is not enough, and trusting it deleted real answers.
        #
        # A tidy script starts an answer at the top of each page, so "short text
        # at the same spot on most pages" describes the *answers* as readily as
        # the furniture. On the physics script that took out "Speed = distance /
        # time" from page one and "R = V / I" from page three — the first line of
        # two answers, and in both cases the line carrying the method mark. The
        # teacher saw a highlight that began at "= 150 / 10".
        #
        # What furniture does and answers do not is repeat the same *words*. The
        # docstring above is right that recognition mangles them — "Rdi No: 37",
        # "RdiNo: 37", "Rdino: 3" — but mangled text still shares most of its
        # character runs, while two unrelated first lines share none. Measured on
        # exactly those examples: the furniture pairs score 0.224 to 1.000 on
        # trigram similarity, and "Speed = distance / time" against "R = V / I"
        # scores 0.000. The gap is not close.
        if _looks_like_the_same_text(group):
            repeated.update(line.line_id for line in group)
    return repeated


#: Trigram similarity above which two lines at the same spot are the same text.
#:
#: Low, because the whole difficulty is that furniture is the text recognition
#: reads worst. It only has to clear zero: unrelated answer lines share no runs
#: at all, and the worst genuine furniture pair measured on real output is 0.224.
_SAME_TEXT = 0.15


def _looks_like_the_same_text(group: list[Line]) -> bool:
    """Whether any two lines in a positional group say the same thing.

    Any pair rather than every pair, and the mangling is why: across four scanned
    pages one roll-number box gave pairs from 1.000 down to 0.224, so requiring
    all of them to agree would let the worst-read page acquit the group. One
    clear pair is enough to establish that this position holds a repeated label.
    """
    from .similarity import CharacterTrigrams

    measure = CharacterTrigrams()
    texts = [line.text.strip() for line in group]
    return any(
        measure.score(texts[i], texts[j]) >= _SAME_TEXT
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    )


#: Share of the shorter box's height two lines must overlap to be one row.
#: The same figure ``reading_order`` uses, and for the same reason: a margin
#: number and the line it labels differ by a fraction of a line height.
_SAME_ROW_OVERLAP = 0.5


def _has_writing_to_the_right(line: Line, lines: list[Line]) -> bool:
    """Whether another line shares this one's row, further right.

    A page number is alone on its row; a question number written in the margin
    has the answer it labels beside it.
    """
    for other in lines:
        if other.line_id == line.line_id or other.page != line.page:
            continue
        if other.box.x0 <= line.box.x1:
            continue
        top, bottom = max(line.box.y0, other.box.y0), min(line.box.y1, other.box.y1)
        shorter = min(line.box.y1 - line.box.y0, other.box.y1 - other.box.y0)
        if shorter > 0 and (bottom - top) / shorter >= _SAME_ROW_OVERLAP:
            return True
    return False


def strip(lines: list[Line]) -> tuple[list[Line], list[Line]]:
    """Split answer-sheet lines into answer text and script details.

    Both halves are returned rather than one discarded, so the caller can report
    what was set aside instead of it vanishing.
    """
    repeated = _repeated_at_the_edge(lines)
    answers: list[Line] = []
    details: list[Line] = []
    for line in lines:
        detail = line.line_id in repeated or is_furniture(
            line, has_writing_beside_it=_has_writing_to_the_right(line, lines)
        )
        (details if detail else answers).append(line)
    return answers, details
