"""Recovering answer structure from a transcribed, inked answer sheet.

Two stages, in order:

1. ``segment`` — group lines into candidate answer blocks. The important rule is
   negative: a gap in the transcribed text is not a boundary, because at ~90%
   detection recall a missed line leaves a text-shaped hole inside an answer.
   Ink found in the gap settles it.
2. ``anchors`` — find the question numbers the student wrote and decide how far
   each can be trusted, since a wrongly believed label maps an answer to the
   wrong question while reporting high confidence.
"""

from .anchors import confirmed, detect
from .segment import segment_blocks, substantive_writing_ink
from .similarity import LexicalOverlap, Similarity, default_similarity

__all__ = [
    "segment_blocks",
    "substantive_writing_ink",
    "detect",
    "confirmed",
    "Similarity",
    "LexicalOverlap",
    "default_similarity",
]
