"""What kind of answer a question asks the student to produce.

Read from the command verb, which is the only thing on the paper that says so.
Two very different parts of the pipeline need it, which is why it lives here
rather than inside either of them:

  * The aligner, to decide whether a region with no readable text could plausibly
    be the answer to a given question. A drawing has no text by nature; an
    explanation that produced no text was simply not read.
  * Grading, to refuse to judge a diagram from a transcription it does not have.
"""

from __future__ import annotations

import re
from enum import StrEnum


class EvidenceKind(StrEnum):
    """What satisfying this question looks like on the page."""

    RECALL = "recall"
    """A definition, statement, or named item."""

    REASONING = "reasoning"
    """An explanation, cause, or justification."""

    WORKING = "working"
    """A calculation: substitution, steps, and a value."""

    CONTRAST = "contrast"
    """A distinction drawn between two things."""

    DRAWING = "drawing"
    """A diagram, graph, or labelled sketch. Not gradable from text."""

    SYMBOLIC = "symbolic"
    """An equation, formula, or balanced reaction."""


#: Asking for ink on the page, in the two shapes papers actually use.
#:
#: **A command to draw.** "Draw", "sketch", "construct a triangle".
#:
#: **A command to put something on a supplied map or figure.** This is how every
#: geography paper phrases its map question — "Mark and locate the following on
#: the outline map of India" — and *none* of the drawing verbs appear in it. Left
#: unmatched it defaulted to RECALL, with two consequences. An unlabelled map
#: answer is a text-free block, and ``_text_free_match_is_plausible`` refuses one
#: unless the question expects a drawing, so it could not even be **placed** — it
#: became an orphan, raised the unassigned-ink total and downgraded every absence
#: claim on the sheet. A labelled map, which is the normal case, reached the
#: *text* grader as scraps like "Delhi" and "Ganga" and was marked against a
#: recall rubric.
#:
#: The placement verbs are deliberately only recognised **together with** the
#: map or figure they place something on. Alone they are far too common to be
#: safe: "name two trace elements" is not a drawing, and neither is "indicate
#: whether the statement is true". Requiring both halves is what makes them
#: usable at all.
#: A verb that puts something somewhere, and the supplied artwork it goes on.
#: Kept apart so both orders can be built from them — a paper writes "mark the
#: river on the outline map" and "on the outline map, mark the river" about
#: equally often, and a single ordered pattern silently handles only one.
#: The placement verb has to be *giving an instruction*, not describing something.
#:
#: Anchored to the start of a clause, because that is what separates the two and
#: nothing else does. "The labels on the graph show rainfall. Describe the trend."
#: pairs an artwork with a placement verb and is answered in words; requiring the
#: verb to open a clause refuses it, while "On the outline map of India, mark the
#: river Ganga" still matches on the comma.
#:
#: Without this the compound rule reintroduced exactly the fault it was written to
#: remove — a text question silently routed to a person and scored zero.
_CLAUSE_START = r"(?:^|[.;:,]\s*|\band\s+|\)\s*)"
_PLACES = r"(?:mark|locate|indicate|trace|show|label)\b"
_ARTWORK = (
    r"\bon (?:the |a |an )?(?:outline |given |political |physical |blank |above )*"
    r"(?:map|figure|diagram|sketch|graph|grid|axes)\b"
)

_DRAWING_PATTERNS = (
    # A bare command to draw. Left unanchored because the word-bounded forms are
    # already safe: "drawn", "drawing", "labelled" and "shaded" do not match.
    r"\b(?:draw|sketch|plot|illustrate|construct|shade|label)\b",
    rf"{_CLAUSE_START}{_PLACES}[^.]{{0,48}}?{_ARTWORK}",
    rf"{_ARTWORK}[^.]{{0,48}}?{_CLAUSE_START}{_PLACES}",
)

#: ``diagram`` is absent from the drawing verbs on purpose, and used to be there.
#:
#: It is a noun, not a command, so it matched every question that merely *refers*
#: to a figure. "The diagram shows a meander. Explain why deposition occurs on the
#: inner bank." is answered in words, carries three marks, and was being routed
#: to ``_needs_a_person`` and scored zero — silently, because a question nobody
#: judged and a question judged as worth nothing are the same number on a report.
#: Every genuine drawing task says draw, sketch, plot, label, construct or
#: illustrate, so nothing is lost by requiring one of those.
_VERBS: tuple[tuple[tuple[str, ...], EvidenceKind], ...] = (
    (("calculate", "compute", "find", "determine", "evaluate", "solve"), EvidenceKind.WORKING),
    (("distinguish", "compare", "differentiate", "contrast"), EvidenceKind.CONTRAST),
    (("balance", "write the formula", "write its formula"), EvidenceKind.SYMBOLIC),
    (("explain", "describe", "justify", "why", "reason", "account for"), EvidenceKind.REASONING),
    (("define", "state", "name", "list", "identify", "what is", "give"), EvidenceKind.RECALL),
)

#: One alternation per group, anchored at word boundaries.
#:
#: Substring matching was the earlier approach and it fires on the inside of
#: ordinary words: "give" inside "given", "state" inside "statement", "find"
#: inside "finding", "mark" inside "marks" — and "marks" appears on nearly every
#: question printed.
_VERB_PATTERNS: tuple[tuple[re.Pattern[str], EvidenceKind], ...] = tuple(
    (re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE), kind)
    for words, kind in _VERBS
)

_DRAWING = tuple(re.compile(p, re.IGNORECASE) for p in _DRAWING_PATTERNS)


def evidence_kind(text: str) -> EvidenceKind:
    """The kind of evidence a question asks for, from its command verb.

    Ordered by specificity rather than by position in the sentence: "Draw a
    labelled diagram and explain how it works" is answered chiefly in ink, and
    treating it as a drawing keeps it out of a text grader's hands.
    """
    if any(pattern.search(text) for pattern in _DRAWING):
        return EvidenceKind.DRAWING
    for pattern, kind in _VERB_PATTERNS:
        if pattern.search(text):
            return kind
    return EvidenceKind.RECALL


def expects_a_drawing(text: str) -> bool:
    """Whether an answer to this would legitimately carry no readable text."""
    return evidence_kind(text) is EvidenceKind.DRAWING
