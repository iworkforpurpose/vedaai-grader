"""Progress events streamed to the browser over SSE."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Stage(StrEnum):
    """Pipeline stages, in execution order.

    Named at a granularity a teacher can read. "Reading the answer sheet" is
    useful; "line index construction" is not. A 60-180 second wait needs to show
    real movement, not a spinner.
    """

    UPLOADING = "uploading"
    RENDERING = "rendering"
    PREPROCESSING = "preprocessing"
    TRANSCRIBING = "transcribing"
    EXTRACTING_INK = "extracting_ink"
    EXTRACTING_QUESTIONS = "extracting_questions"
    VALIDATING_QUESTIONS = "validating_questions"
    SEGMENTING_ANSWERS = "segmenting_answers"
    DETECTING_LABELS = "detecting_labels"
    MAPPING = "mapping"
    ADJUDICATING = "adjudicating"
    GRADING = "grading"
    DONE = "done"
    FAILED = "failed"


class ProgressEvent(BaseModel):
    """One progress update.

    Carries page-level counters because per-page work is where the time goes,
    and "page 4 of 11" is far more reassuring than an indeterminate bar.
    """

    model_config = ConfigDict(frozen=True)

    stage: Stage
    message: str
    pages_done: int | None = Field(default=None, ge=0)
    pages_total: int | None = Field(default=None, ge=0)
    questions_found: int | None = Field(default=None, ge=0)
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.stage in {Stage.DONE, Stage.FAILED}
