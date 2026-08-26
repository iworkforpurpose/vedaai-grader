"""The top-level object tying one grading run together."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

from .answers import Anchor, AnswerBlock, InkRegion
from .documents import Page, SourceFile
from .grading import GradeResult
from .mapping import MappingResult
from .ocr import LineIndex
from .questions import QuestionPaper


class SubmissionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class Submission(BaseModel):
    """One question paper plus one student's answer sheet, and everything derived.

    Held in memory: the brief permits it and at this scale a database would add
    operational surface without buying anything. The consequence is that state
    is lost on restart, which is acceptable for a testing deployment and is
    stated plainly rather than papered over.
    """

    submission_id: str
    status: SubmissionStatus = SubmissionStatus.PENDING

    question_paper_file: SourceFile | None = None
    answer_sheet_file: SourceFile | None = None
    pages: list[Page] = Field(default_factory=list)

    question_paper_lines: LineIndex | None = None
    answer_sheet_lines: LineIndex | None = None
    ink_regions: list[InkRegion] = Field(default_factory=list)

    questions: QuestionPaper | None = None
    blocks: list[AnswerBlock] = Field(default_factory=list)
    anchors: list[Anchor] = Field(default_factory=list)
    mapping: MappingResult | None = None
    grades: GradeResult | None = None

    warnings: list[str] = Field(
        default_factory=list,
        description="Conditions a teacher should know about before trusting the report — "
        "missing pages, suppressed absence claims, low transcription confidence.",
    )
    error: str | None = None

    @computed_field
    @property
    def answer_sheet_page_count(self) -> int:
        from .documents import DocumentKind

        return sum(1 for p in self.pages if p.kind is DocumentKind.ANSWER_SHEET)

    @computed_field
    @property
    def question_count(self) -> int:
        return len(self.questions.questions) if self.questions else 0
