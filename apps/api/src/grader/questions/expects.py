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


#: Command verbs, mapped to the evidence they ask for. Taken from the wording of
#: real CBSE, ICSE, AQA and Edexcel papers rather than invented.
_VERBS: tuple[tuple[tuple[str, ...], EvidenceKind], ...] = (
    (("draw", "sketch", "plot", "label", "illustrate", "diagram"), EvidenceKind.DRAWING),
    (("calculate", "compute", "find", "determine", "evaluate", "solve"), EvidenceKind.WORKING),
    (("distinguish", "compare", "differentiate", "contrast"), EvidenceKind.CONTRAST),
    (("balance", "write the formula", "write its formula"), EvidenceKind.SYMBOLIC),
    (("explain", "describe", "justify", "why", "reason", "account for"), EvidenceKind.REASONING),
    (("define", "state", "name", "list", "identify", "what is", "give"), EvidenceKind.RECALL),
)


def evidence_kind(text: str) -> EvidenceKind:
    """The kind of evidence a question asks for, from its command verb.

    Ordered by specificity rather than by position in the sentence: "Draw a
    labelled diagram and explain how it works" is answered chiefly in ink, and
    treating it as a drawing keeps it out of a text grader's hands.
    """
    lowered = text.lower()
    for words, kind in _VERBS:
        if any(word in lowered for word in words):
            return kind
    return EvidenceKind.RECALL


def expects_a_drawing(text: str) -> bool:
    """Whether an answer to this would legitimately carry no readable text."""
    return evidence_kind(text) is EvidenceKind.DRAWING
