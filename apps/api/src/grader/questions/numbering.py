"""Parsing the labels a question paper prints in front of its questions.

Real boards do not agree on a notation. Taken from official papers: CBSE and
CISCE use ``1.`` and ``11 (a)``, ICSE nests three deep as ``Q2 → (i) → (a)``,
roman numerals appear as ``(i)``, ``(II.)`` and ``(III)``, and some papers prefix
with ``Q.1``.

The design decision that makes this tractable is what this module deliberately
does *not* do: it never decides what a token means. ``(i)`` is stored as the token
``i`` without ruling on whether that is roman one or the letter i — a genuinely
ambiguous question, since ``(h) (i) (j)`` and ``(i) (ii) (iii)`` are both real
sequences and only surrounding context distinguishes them.

Nothing downstream needs the answer. Ordering comes from position on the page,
recorded as ``print_order``, and identity comes from the raw token path. A parser
that tried to interpret the tokens would have to be right about that ambiguity;
this one only has to find where the label ends and the question begins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class LabelStyle(StrEnum):
    """How a label is punctuated.

    Carries structural information even though a token's *value* is ambiguous:
    papers overwhelmingly use bare numerals for top-level questions and
    parenthesised letters or romans for sub-parts.
    """

    NUMERIC = "numeric"
    """``1.`` ``2)`` ``Q3`` — a top-level question."""

    PARENTHESISED = "parenthesised"
    """``(a)`` ``(i)`` ``(II.)`` — conventionally a sub-part."""

    TRAILING_PUNCT = "trailing_punct"
    """``a.`` ``i)`` — a sub-part written without an opening bracket."""


#: One label token: digits, or a short alphabetic run that may be a letter or a
#: roman numeral. Deliberately not distinguishing the two.
_TOKEN_RE = re.compile(r"\d{1,3}|[ivxlcdmIVXLCDM]{1,5}|[a-zA-Z]")

#: A leading Q prefix: ``Q1``, ``Q.1``, ``Q 1``. Requires a digit after it, so a
#: sentence beginning "Quote the formula" is not mistaken for question ``uote``.
_Q_PREFIX = re.compile(r"^\s*[Qq]\s*\.?\s*(?=\d)")

_BARE_NUMBER = re.compile(r"\d{1,3}")


@dataclass(frozen=True)
class ParsedLabel:
    """A label found at the start of a line."""

    raw: str
    """Exactly the characters the paper printed, whitespace-collapsed only.

    Preserved because the requirement is to keep the original numbering, and a
    teacher scanning the list needs the paper's own notation rather than a
    canonicalized rewrite of it."""

    tokens: tuple[str, ...]
    """Raw token values, outermost first. ``11 (a)`` gives ``("11", "a")``."""

    style: LabelStyle
    remainder: str
    """The question text following the label."""

    @property
    def depth_hint(self) -> int:
        """Nesting depth implied by the label alone.

        A hint rather than an answer: ``(a)`` on its own is one token but is
        almost certainly a sub-part of whatever preceded it. Indentation settles
        it in ``extract``.
        """
        return len(self.tokens)

    @property
    def is_top_level_candidate(self) -> bool:
        return self.style is LabelStyle.NUMERIC and self.tokens[0].isdigit()


def _scan_bracketed(text: str, i: int) -> tuple[str, int] | None:
    """Read ``(a)`` or ``(II.)`` starting at i."""
    if i >= len(text) or text[i] != "(":
        return None
    j = i + 1
    while j < len(text) and text[j].isspace():
        j += 1
    match = _TOKEN_RE.match(text, j)
    if match is None:
        return None
    token = match.group(0)
    j = match.end()
    # Allow internal punctuation before the closing bracket, as in "(II.)".
    while j < len(text) and (text[j].isspace() or text[j] == "."):
        j += 1
    if j < len(text) and text[j] == ")":
        return token, j + 1
    return None


def _scan_punctuated(text: str, i: int) -> tuple[str, int] | None:
    """Read ``1.`` or ``a)`` starting at i."""
    match = _TOKEN_RE.match(text, i)
    if match is None:
        return None
    token = match.group(0)
    j = match.end()
    while j < len(text) and text[j] == " ":
        j += 1
    if j < len(text) and text[j] in ".)":
        return token, j + 1
    return None


def parse_label(text: str) -> ParsedLabel | None:
    """Extract a leading question label, or None if the line does not start with one.

    Hand-scanned rather than matched by one regular expression. The notations a
    label can take — ``1.``, ``11 (a)``, ``11(a)``, ``2 (i) (a)``, ``Q.5``,
    ``(II.)`` — do not factor into a single readable pattern, and the regex
    attempted first silently failed on four of those six while appearing to work.

    Three rules earn their place by rejecting things that merely resemble labels:

    * A bare number only begins a label when a bracket follows it, as in
      ``11 (a)``, or when a ``Q`` introduced it. Otherwise a sentence opening
      "In 1947 India..." becomes question 1947.
    * The label must end at whitespace or end of line, which is what stops
      ``1.5 kg of copper`` parsing as question 1 answered by "5 kg of copper".
    * Tokens are never interpreted.
    """
    prefix = _Q_PREFIX.match(text)
    if prefix is not None:
        start = prefix.end()
        had_q = True
    else:
        start = 0
        while start < len(text) and text[start].isspace():
            start += 1
        had_q = False

    tokens: list[str] = []
    style: LabelStyle | None = None
    position = start

    while position < len(text):
        probe = position
        if tokens:
            while probe < len(text) and text[probe] == " ":
                probe += 1

        bracketed = _scan_bracketed(text, probe)
        if bracketed is not None:
            token, position = bracketed
            tokens.append(token)
            if style is None:
                style = LabelStyle.PARENTHESISED
            continue

        punctuated = _scan_punctuated(text, probe)
        if punctuated is not None:
            token, position = punctuated
            tokens.append(token)
            if style is None:
                style = (
                    LabelStyle.NUMERIC
                    if token.isdigit() or had_q
                    else LabelStyle.TRAILING_PUNCT
                )
            continue

        if not tokens:
            bare = _BARE_NUMBER.match(text, probe)
            if bare is not None:
                after = bare.end()
                lookahead = after
                while lookahead < len(text) and text[lookahead] == " ":
                    lookahead += 1
                # A bare number needs corroboration: a bracket following it, or a
                # Q introducing it. Without that, any sentence opening with a
                # year becomes a question label.
                if had_q or (lookahead < len(text) and text[lookahead] == "("):
                    tokens.append(bare.group(0))
                    style = LabelStyle.NUMERIC
                    position = after
                    continue
        break

    if not tokens or style is None:
        return None

    if position < len(text) and not text[position].isspace():
        return None

    remainder = text[position:].strip()
    raw = " ".join(text[:position].split())
    return ParsedLabel(raw=raw, tokens=tuple(tokens), style=style, remainder=remainder)


def looks_like_a_question(label: ParsedLabel, remainder: str) -> bool:
    """Whether a labelled line plausibly introduces a question.

    A bare label with nothing after it is usually a continuation marker or a
    stray page number.

    Deliberately permissive: missing a question is worse than admitting a
    non-question, because the validator can flag a suspicious entry for review
    whereas a question never extracted simply does not exist.
    """
    _ = label
    return len(remainder) >= 3


def canonical_qid(section: str | None, tokens: tuple[str, ...]) -> str:
    """Build a stable identity from a section and a token path.

    Namespaced by section because papers exist whose numbering restarts in each
    one, and an unnamespaced ``5`` would then collide. The path is raw tokens
    joined — no interpretation, no normalizing romans to integers.
    """
    prefix = f"{section}/" if section else ""
    return prefix + "/".join(tokens)


#: Marks printed against a question, as ``[5]``, ``(5)``, ``[5 marks]``.
#:
#: Anchored to the end of the line, because a bracketed number mid-sentence is
#: part of the question — a chemistry paper writing "[Fe(H2O)6]3+" must not have
#: it read as a mark allocation.
_MARKS = re.compile(r"[\[(]\s*(\d{1,3})\s*(?:marks?|m)?\s*[\])]\s*$", re.IGNORECASE)


def extract_marks(text: str) -> tuple[str, int | None]:
    """Split a trailing mark allocation off a line.

    Marks are captured rather than discarded because they are the grading
    denominator, and a printed one is more trustworthy than anything inferred.
    """
    match = _MARKS.search(text)
    if match is None:
        return text, None
    return text[: match.start()].rstrip(), int(match.group(1))
