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
import os
import re
from collections import Counter
from collections.abc import Callable
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

    #: Below this score a pair is unrelated, on this measure's own scale.
    #:
    #: Each implementation states its own, because the scales are nothing like each
    #: other: trigram overlap between two unrelated English texts sits around 0.1
    #: and embeddings around 0.15, but embeddings put a genuine match near 0.7
    #: while trigrams may only reach 0.3. A single constant in the aligner would be
    #: wrong for one of them, so the measure that knows the scale supplies it.
    #:
    #: Zero means "no opinion", which is the honest answer for a measure whose
    #: absolute value carries no meaning, and leaves the aligner's behaviour
    #: unchanged.
    unrelated_below: float

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

    #: Word overlap between a question and its own answer is
    #: routinely zero — that is the limitation this class is documented as having —
    #: so its absolute value cannot say whether a pair is unrelated.
    unrelated_below = 0.0


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

    #: Any two English texts share runs like "the" and "ing",
    #: so the floor is high and uninformative; the aligner's centring is what makes
    #: this measure usable, and it discards the absolute value anyway.
    unrelated_below = 0.0


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

    unrelated_below = 0.0

    def __init__(self) -> None:
        self._word = LexicalOverlap()
        self._trigram = CharacterTrigrams()

    def score(self, a: str, b: str) -> float:
        return max(self._word.score(a, b), self._trigram.score(a, b))


#: The embedding model. Small and cheap: this compares short exam answers, not
#: documents, and the larger model's extra capacity is not used by the task.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

#: How many texts to embed per request. One paper's questions and one script's
#: blocks together are tens of items, so a single round trip usually covers a whole
#: submission.
_EMBED_BATCH = 128


class SemanticSimilarity:
    """How close two texts are in meaning, rather than in spelling.

    The measures above compare surfaces — shared words, shared character runs — and
    a good answer does not share the question's surface. Measured on real pairs:

        Name the process by which plants lose water   -> It is called transpiration
        word overlap 0.000, trigrams 0.000

    Both are silent on an answer that is exactly right, because "transpiration" and
    "the process by which plants lose water" have nothing in common as strings. The
    aligner then has only position to go on, which is why a script written out of
    order gets placed by habit rather than by content. That is not a tuning problem;
    it is the limit of comparing characters, and it is the same limit whatever the
    subject — history, geography and English answers restate ideas even more freely
    than science ones do.

    Embeddings put both texts in a space where "transpiration" and "plants losing
    water as vapour" are near each other, which is the property the aligner has
    needed all along.

    Three practical commitments:

    * **Every text is embedded once.** Scoring is quadratic — every question
      against every block — so without a cache one submission would make hundreds
      of identical calls.
    * **Failure is never fatal.** No key, no network, a provider outage: the score
      falls back to the surface measure. Weaker is not useless, and a submission
      that errors is worse than one placed by trigrams.
    * **The seam is unchanged.** This satisfies the same `Similarity` protocol as
      the others, so it is swapped in by configuration and out again by removing a
      key.
    """

    #: Measured on the deployed service. A comprehension paper against a script of
    #: handwritten C scored 0.148 to 0.154 on every block; the same paper against
    #: its own answers scored 0.536 to 0.779. The gap is wide enough that a
    #: threshold in the middle separates them without touching a real match.
    unrelated_below = 0.30

    def __init__(self, *, embed: Callable[[list[str]], list[list[float]]] | None = None,
                 fallback: Similarity | None = None) -> None:
        self._embed = embed or _openai_embed
        self._fallback = fallback or StrongerOf()
        self._cache: dict[str, list[float] | None] = {}
        #: Set once the provider has failed. One failure means the next call will
        #: almost certainly fail too, and a submission should not pay that latency
        #: once per pair.
        self._unavailable = False

    def score(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0

        first, second = self._vectors(a, b)
        if first is None or second is None:
            return self._fallback.score(a, b)

        dot = sum(x * y for x, y in zip(first, second, strict=False))
        norm_a = math.sqrt(sum(x * x for x in first))
        norm_b = math.sqrt(sum(y * y for y in second))
        if not norm_a or not norm_b:
            return self._fallback.score(a, b)

        # Cosine similarity of these models lands in roughly [0, 1] for related
        # text and near zero for unrelated, but is not guaranteed non-negative.
        # Clamped because the aligner's weights assume a score in [0, 1].
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def warm(self, texts: list[str]) -> None:
        """Embed a set of texts in as few requests as possible.

        Called with a paper's questions and a script's blocks before alignment, so
        the whole submission costs one or two round trips instead of one per pair.
        """
        missing = [t for t in dict.fromkeys(texts) if t.strip() and t not in self._cache]
        for start in range(0, len(missing), _EMBED_BATCH):
            self._fetch(missing[start : start + _EMBED_BATCH])

    def _vectors(self, a: str, b: str) -> tuple[list[float] | None, list[float] | None]:
        self.warm([a, b])
        return self._cache.get(a), self._cache.get(b)

    def _fetch(self, texts: list[str]) -> None:
        if not texts:
            return
        if self._unavailable:
            for text in texts:
                self._cache[text] = None
            return
        try:
            vectors = self._embed(texts)
        except Exception:  # noqa: BLE001 - any provider failure degrades the same way
            self._unavailable = True
            for text in texts:
                self._cache[text] = None
            return
        for text, vector in zip(texts, vectors, strict=False):
            self._cache[text] = vector


def _openai_embed(texts: list[str]) -> list[list[float]]:
    """Embed with OpenAI, which the marking path already depends on.

    No new dependency and no new credential: the client is installed for grading
    and the key is already present wherever marking runs.
    """
    from openai import OpenAI

    response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def semantic_available() -> bool:
    """Whether meaning-based scoring can run at all."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return False
    try:
        import openai  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def build_similarity() -> Similarity:
    """The scorer for this deployment.

    Semantics where a key exists, surfaces where one does not — the same shape as
    marking, which degrades to a rubric rather than failing. A developer with no
    key still gets a working mapping, just a weaker one.
    """
    return SemanticSimilarity() if semantic_available() else StrongerOf()


#: The default. Swapped by passing a different implementation rather than by
#: changing this, so an experiment does not require editing the pipeline.
default_similarity: Similarity = build_similarity()
