"""The top-level object tying one grading run together."""

from __future__ import annotations

from datetime import datetime
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

    Persisted whole rather than as rows. Every field below is derived from the two
    uploaded documents in one pipeline run, so there is no query that wants them
    apart — and splitting them across tables would mean a half-written submission
    becoming representable, which is a state no reader here knows how to handle.

    The consequence worth knowing is that this object is what gets written on every
    mutation, and it is large: a measured two-page submission serializes to 140 KiB,
    nearly all of it line boxes and ink regions. `grader.persistence` compresses it
    and spills past the item limit for that reason.
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

    #: When this was last written. Stamped by the store on every save.
    #:
    #: Here rather than left to the persistence layer because it answers a
    #: question about the submission and not about the row: how long it has been
    #: since anything happened to it. Ingest runs in a background task, and the
    #: code around it turns every exception it can catch into a stored failure —
    #: but not the process going away. A deploy, an out-of-memory kill or a
    #: replaced host leaves a submission at `processing` with nobody to move it,
    #: and this is what lets a reader tell that apart from one still working.
    updated_at: datetime | None = None

    @computed_field
    @property
    def answer_sheet_page_count(self) -> int:
        from .documents import DocumentKind

        return sum(1 for p in self.pages if p.kind is DocumentKind.ANSWER_SHEET)

    @computed_field
    @property
    def question_count(self) -> int:
        return len(self.questions.questions) if self.questions else 0
