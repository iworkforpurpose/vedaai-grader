"""Grading output, structured so every mark is traceable to ink on the page."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field


class RubricPoint(BaseModel):
    """One credit-bearing criterion, and whether the answer satisfied it.

    ``cited_line_ids`` is the load-bearing field. A rubric point that awards
    marks must point at the specific lines that earned them, and those IDs are
    validated against the line index before the grade is accepted. Three things
    follow from that constraint:

      * The teacher can click a rubric point and see the sentence behind it,
        which is what makes a grade checkable rather than merely plausible.
      * A fabricated justification fails validation instead of being displayed,
        because invented line IDs do not resolve.
      * Marks cannot be talked into existence by text on the page. Student
        writing reaches the model as untrusted transcription, and a score is
        only assemblable from criteria that cite real, resolvable lines.
    """

    model_config = ConfigDict(frozen=True)

    point_id: str
    criterion: str = Field(description="What the student had to do to earn this.")
    marks_available: float = Field(ge=0.0)
    marks_awarded: float = Field(ge=0.0)
    satisfied: bool
    cited_line_ids: list[str] = Field(
        default_factory=list,
        description="Lines evidencing this judgement. Validated against the line index; "
        "an unresolvable ID invalidates the grade rather than being ignored.",
    )
    comment: str | None = None


class QuestionGrade(BaseModel):
    """Grade for a single question."""

    qid: str
    marks_available: float = Field(ge=0.0)
    marks_awarded: float = Field(ge=0.0)
    rubric_points: list[RubricPoint] = Field(default_factory=list)
    feedback: str | None = None

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    graded_on_partial_text: bool = Field(
        default=False,
        description="True when the underlying transcription was low-confidence. A grade "
        "computed from text we could not read reliably must be labelled as such rather "
        "than presented at face value.",
    )

    @computed_field
    @property
    def fraction(self) -> float | None:
        if self.marks_available <= 0:
            return None
        return self.marks_awarded / self.marks_available

    @computed_field
    @property
    def needs_review(self) -> bool:
        return self.graded_on_partial_text or self.confidence < 0.6


class GradeResult(BaseModel):
    """Whole-submission grading summary.

    ``committed`` is deliberately False by default and is only ever set by an
    explicit human action. The system produces a proposal; a person decides.
    """

    grades: list[QuestionGrade] = Field(default_factory=list)
    overall_feedback: str | None = None
    weak_topics: list[str] = Field(default_factory=list)
    committed: bool = Field(
        default=False,
        description="Whether a human has accepted these marks. Never set by the pipeline.",
    )

    @computed_field
    @property
    def total_awarded(self) -> float:
        return sum(g.marks_awarded for g in self.grades)

    @computed_field
    @property
    def total_available(self) -> float:
        return sum(g.marks_available for g in self.grades)

    @computed_field
    @property
    def review_count(self) -> int:
        return sum(1 for g in self.grades if g.needs_review)
