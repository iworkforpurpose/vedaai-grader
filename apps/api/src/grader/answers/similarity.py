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


#: Length of the character runs compared by ``CharacterTrigrams``. Three is the
#: usual choice for fuzzy text matching: long enough to carry meaning, short
#: enough that a single misread letter only damages the runs it appears in.
_NGRAM = 3

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def character_ngrams(text: str, n: int = _NGRAM) -> Counter[str]:
    """Overlapping character runs of a normalized string.

    Punctuation and spacing are removed first, so ``for(i=0;i<n;i++)`` and
    ``for (i = 0; i < n; i++)`` are the same text — which matters, because
    handwriting recognition is inconsistent about exactly those characters.
    """
    flat = _NON_ALPHANUMERIC.sub("", text.lower())
    if len(flat) < n:
        return Counter([flat] if flat else [])
    return Counter(flat[i : i + n] for i in range(len(flat) - n + 1))


class CharacterTrigrams:
    """Cosine similarity over character trigrams.

    Adopted after measuring the word-level scorer on real handwritten scripts,
    where it returned **exactly zero for every question against every answer**.
    Not weak — dead. Recognition had turned "sorted array" into "forted armayb"
    and "elements" into "demenbou", and a scorer that needs whole words to match
    has nothing left to work with. The mapping was left resting on position alone,
    which is how a page of code answering one question came to be filed under
    another.

    Trigrams survive that damage because a misread letter only breaks the three
    runs containing it: "forted" and "sorted" still share "ort", "rte" and "ted".
    Word overlap is not lost either — identical words share all of their trigrams,
    so this subsumes the previous measure rather than trading it away.

    The cost is a higher floor, since any two English texts share runs like "the"
    and "ing". That would have been a problem before and is not now: the aligner
    subtracts each block's mean similarity across all questions, so a constant
    shared by every pairing cancels and only the differences between questions
    survive.
    """

    def score(self, a: str, b: str) -> float:
        first = character_ngrams(a)
        second = character_ngrams(b)
        if not first or not second:
            return 0.0

        shared = set(first) & set(second)
        if not shared:
            return 0.0

        dot = sum(first[gram] * second[gram] for gram in shared)
        norm_a = math.sqrt(sum(count * count for count in first.values()))
        norm_b = math.sqrt(sum(count * count for count in second.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class StrongerOf:
    """The higher of two views of the same pair of texts.

    Both measures were built and both were measured, and neither dominates:

      * Word overlap scores slightly better on the synthetic golden set, where the
        generated answers deliberately reuse the question's vocabulary and exact
        matches are therefore available and meaningful.
      * On real handwritten scripts word overlap returns exactly zero for every
        pairing, because recognition had destroyed the words it needs. Trigrams
        still find "ort", "rte" and "ted" shared between "sorted" and "forted".

    Taking the maximum keeps each where it works. It can only raise a score that
    word overlap already found, so the precision of an exact match is never traded
    away — and where exactness is unavailable, something still answers.
    """

    def __init__(self) -> None:
        self._word = LexicalOverlap()
        self._trigram = CharacterTrigrams()

    def score(self, a: str, b: str) -> float:
        return max(self._word.score(a, b), self._trigram.score(a, b))


#: The default. Swapped by passing a different implementation rather than by
#: changing this, so an experiment does not require editing the pipeline.
default_similarity: Similarity = StrongerOf()
