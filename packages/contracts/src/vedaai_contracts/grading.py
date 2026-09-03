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

    graded_by: str | None = Field(
        default=None,
        description=(
            "Which engine produced this grade, e.g. 'openai:gpt-4o-mini' or "
            "'rubric_only'. Recorded for the same reason transcription records its "
            "engine per line: a mark is only checkable if you know what made it, and "
            "with more than one provider configurable the answer is not obvious from "
            "anything else in the payload."
        ),
    )

    judged: bool = Field(
        default=False,
        description=(
            "True when a marker actually decided this question. False for a rubric "
            "produced without marking, for a judgement whose citations were refused, "
            "and for an answer left to a person to look at.\n\n"
            "Explicit because it is not recoverable from anything else here, and "
            "guessing it was a bug. The obvious inference — that a judged point cites "
            "a line — fails for the commonest interesting case: a genuine zero cites "
            "nothing, because there is no evidence for marks that were not given. So a "
            "0 out of 4 that a marker decided looked identical to a question nobody "
            "had marked, and the difference is the whole point. `confidence` cannot "
            "stand in either: it is derived from cited share, so it is also 0.0 for a "
            "real zero."
        ),
    )

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    graded_on_partial_text: bool = Field(
        default=False,
        description="True when the underlying transcription was low-confidence. A grade "
        "computed from text we could not read reliably must be labelled as such rather "
        "than presented at face value.",
    )

    teacher_marks: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "What a teacher decided this answer was worth, where they said so.\n\n"
            "Beside `marks_awarded` rather than replacing it, on purpose. The whole "
            "product is a proposal a person checks, and a correction that overwrote "
            "the proposal would destroy the only record of what was corrected — which "
            "is the evidence that says whether the marker is getting better or worse. "
            "Marking varies by about a mark between runs and the model is wrong "
            "outright on some answers; both are facts a teacher needs to be able to "
            "act on and this project needs to be able to measure.\n\n"
            "None means the teacher has not said. Zero means they said zero, and the "
            "two are entirely different claims."
        ),
    )

    @computed_field
    @property
    def marks_final(self) -> float:
        """The mark that counts: the teacher's where they gave one."""
        return self.marks_awarded if self.teacher_marks is None else self.teacher_marks

    @computed_field
    @property
    def teacher_decided(self) -> bool:
        return self.teacher_marks is not None

    @computed_field
    @property
    def fraction(self) -> float | None:
        if self.marks_available <= 0:
            return None
        return self.marks_final / self.marks_available

    @computed_field
    @property
    def needs_review(self) -> bool:
        # A question a person has already ruled on is not waiting for one. Leaving
        # it in the review count would send a teacher back to the work they just
        # did, which is the fastest way to make a review queue worth ignoring.
        if self.teacher_decided:
            return False
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
        """The script's total, counting a teacher's corrections."""
        return sum(g.marks_final for g in self.grades)

    @computed_field
    @property
    def total_proposed(self) -> float:
        """What the marker proposed, before any correction.

        Kept beside the total rather than derived away: the gap between the two is
        the measurement of how good the marker is on real scripts, and it is the
        only such measurement that does not need somebody to write truth down
        first.
        """
        return sum(g.marks_awarded for g in self.grades)

    @computed_field
    @property
    def corrected_count(self) -> int:
        return sum(1 for g in self.grades if g.teacher_decided)

    @computed_field
    @property
    def total_available(self) -> float:
        return sum(g.marks_available for g in self.grades)

    @computed_field
    @property
    def review_count(self) -> int:
        return sum(1 for g in self.grades if g.needs_review)
