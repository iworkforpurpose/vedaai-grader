"""Recovering a question paper's structure from its transcribed lines.

Pipeline order within this package:

1. ``reading_order`` — put lines in the order a person would read them, which on
   a two-column paper is not the order of their vertical positions.
2. ``furniture`` — separate questions from headers, rubric, page numbers, marks
   and competency tags.
3. ``numbering`` — find where a label ends and the question begins, without
   ruling on whether ``(i)`` is roman one or the letter i.
4. ``extract`` — assemble the tree, using indentation to place relative labels.
5. ``validate`` — check the result against the numbering the paper itself printed.
"""

from .expects import EvidenceKind, evidence_kind, expects_a_drawing
from .extract import extract, mark_stems, question_lines, reads_as_a_heading
from .furniture import classify_all, section_label
from .numbering import ParsedLabel, canonical_qid, extract_marks, parse_label
from .optionality import parse_requirement, satisfied
from .validate import find_gaps, suspicious

__all__ = [
    "EvidenceKind",
    "evidence_kind",
    "expects_a_drawing",
    "mark_stems",
    "reads_as_a_heading",
    "extract",
    "question_lines",
    "classify_all",
    "section_label",
    "ParsedLabel",
    "parse_label",
    "canonical_qid",
    "extract_marks",
    "parse_requirement",
    "satisfied",
    "find_gaps",
    "suspicious",
]
