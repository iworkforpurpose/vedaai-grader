"""Measuring whether a block of writing plausibly answers a given question.

Used to confirm anchors. A student writes ``11 (b)`` in the margin; this is what
decides whether the writing beside it actually belongs to question 11(b) or
whether the student mislabelled it.

Deliberately behind an interface with a dependency-free default. The plan calls
for local ONNX sentence embeddings, and they may well score better — but a
50 MB model is a real cost, and the golden set already contains a mislabelled
case that can say whether it is needed. Building the seam and measuring is
cheaper than assuming either way, and if lexical overlap turns out to be enough
the dependency never has to exist.

The honest limitation of the default: an answer often does not reuse the
question's vocabulary. "Define refraction" is answered by "the bending of light",
which shares no content word with the question at all. So a low lexical score is
weak evidence of mismatch, while a high one is strong evidence of match — an
asymmetry the caller has to respect, and the reason confirmation also accepts
order-consistency as an alternative route.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

#: Words too common to carry topical signal. Kept short on purpose: an
#: aggressive stop list on short exam answers removes most of the text.
_STOPWORDS = frozenset(
    [
        "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to", "for",
        "from", "with", "without", "by", "as", "is", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "this", "that", "these", "those", "it", "its",
        "which", "who", "whom", "what", "when", "where", "how", "why", "not", "no",
        "nor", "so", "than", "then", "there", "here", "their", "them", "they", "he",
        "she", "his", "her", "you", "your", "we", "our", "i", "me", "my",
    ]
)

_WORD = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Similarity(Protocol):
    """Scores how related two pieces of text are, in ``[0, 1]``."""

    def score(self, a: str, b: str) -> float: ...


def tokenize(text: str) -> list[str]:
    """Lowercase content words, stopwords and single characters removed."""
    return [
        word
        for word in _WORD.findall(text.lower())
        if len(word) > 1 and word not in _STOPWORDS
    ]


class LexicalOverlap:
    """Cosine similarity over term-frequency vectors of content words.

    Chosen over plain Jaccard because exam answers repeat key terms, and
    repetition is signal: an answer that says "reflection" three times is more
    likely about reflection than one that mentions it once. Term frequency keeps
    that; set overlap discards it.
    """

    def score(self, a: str, b: str) -> float:
        first = Counter(tokenize(a))
        second = Counter(tokenize(b))
        if not first or not second:
            return 0.0

        shared = set(first) & set(second)
        if not shared:
            return 0.0

        dot = sum(first[word] * second[word] for word in shared)
        norm_a = math.sqrt(sum(count * count for count in first.values()))
        norm_b = math.sqrt(sum(count * count for count in second.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


#: The default. Swapped by passing a different implementation rather than by
#: changing this, so an experiment does not require editing the pipeline.
default_similarity: Similarity = LexicalOverlap()
