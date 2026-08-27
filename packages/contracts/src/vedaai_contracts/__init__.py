"""Shared contracts for the exam answer-sheet grader.

This package is the single source of truth for every type crossing a boundary —
between the Python pipeline and the TypeScript frontend, and between pipeline
stages. TypeScript types are generated from these models, never hand-written, so
the two languages cannot drift.

The most important thing defined here is the coordinate contract in
``geometry``. Read that module before touching anything spatial.
"""

from .answers import Anchor, AnchorStatus, AnswerBlock, InkRegion, InkRegionKind
from .documents import DocumentKind, Page, SourceFile
from .geometry import HGBENCH_SCALE, RENDER_DPI, BBox, PageBox
from .grading import GradeResult, QuestionGrade, RubricPoint
from .mapping import (
    AnswerStatus,
    Highlight,
    Mapping,
    MappingResult,
    MatchEvidence,
    MatchSignal,
    OrphanAnswer,
)
from .ocr import Line, LineIndex, OcrEngine, Word
from .progress import ProgressEvent, Stage
from .questions import (
    ChoiceGroup,
    LineRole,
    NumberingGap,
    Question,
    QuestionPaper,
    Requirement,
    Section,
    Stem,
)
from .submission import Submission, SubmissionStatus

#: Every model exported to TypeScript. The codegen script walks this list, so a
#: model added here appears on the frontend and one omitted does not exist there.
EXPORTED_MODELS = [
    BBox,
    PageBox,
    SourceFile,
    Page,
    Word,
    Line,
    LineIndex,
    Requirement,
    Section,
    ChoiceGroup,
    Stem,
    Question,
    NumberingGap,
    QuestionPaper,
    InkRegion,
    AnswerBlock,
    Anchor,
    MatchEvidence,
    Highlight,
    Mapping,
    OrphanAnswer,
    MappingResult,
    RubricPoint,
    QuestionGrade,
    GradeResult,
    ProgressEvent,
    Submission,
]

__all__ = [
    "RENDER_DPI",
    "HGBENCH_SCALE",
    "BBox",
    "PageBox",
    "DocumentKind",
    "Page",
    "SourceFile",
    "Word",
    "Line",
    "LineIndex",
    "OcrEngine",
    "LineRole",
    "Requirement",
    "Section",
    "ChoiceGroup",
    "Stem",
    "Question",
    "NumberingGap",
    "QuestionPaper",
    "InkRegion",
    "AnswerBlock",
    "Anchor",
    "AnchorStatus",
    "InkRegionKind",
    "AnswerStatus",
    "MatchSignal",
    "MatchEvidence",
    "Highlight",
    "Mapping",
    "OrphanAnswer",
    "MappingResult",
    "RubricPoint",
    "QuestionGrade",
    "GradeResult",
    "Stage",
    "ProgressEvent",
    "Submission",
    "SubmissionStatus",
    "EXPORTED_MODELS",
]
