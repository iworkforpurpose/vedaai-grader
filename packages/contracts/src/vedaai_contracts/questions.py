"""Extracted question-paper structure.

The shapes here are driven by what real papers actually do, taken from official
CBSE, CISCE, AQA, Edexcel and College Board PDFs rather than assumed:

  * ICSE nests three levels deep as ``Q2 → (i) → (a)``, while other boards
    write ``11(a)`` — so level ordering cannot be hardcoded.
  * ICSE numbers continuously across sections (A = Q1-3, B = Q4-10); other
    boards restart per section — so a label is not a unique key.
  * ICSE prints *"Attempt all questions from Section A and any four questions
    from Section B"*; VTU prints *"Answer any FIVE full questions, choosing ONE
    full question from each module"* — so a skipped question can be correct.
  * CBSE prints *"an internal choice has been provided in two questions"* — so
    two questions can be alternatives where answering either one suffices.
  * Papers carry printed furniture that is not a question: competency tags like
    ``[Analysis & Evaluation]``, mark allocations like ``[4]``, and notes like
    ``(for V.I. candidates)``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .geometry import PageBox


class LineRole(StrEnum):
    """What a question-paper line is, structurally.

    Classifying before parsing is what keeps printed furniture out of question
    text. Getting this wrong in either direction is costly: swallowing a mark
    allocation into a question corrupts the text sent for grading, while
    discarding a real question line loses a question outright.
    """

    QUESTION_START = "question_start"
    QUESTION_CONTINUATION = "question_continuation"
    STEM = "stem"  # shared passage / assertion-reason context
    SECTION_HEADER = "section_header"
    INSTRUCTION = "instruction"
    FURNITURE = "furniture"  # headers, footers, page numbers, competency tags
    MATERIAL = "material"
    """Content the paper prints for a question to refer to: a source extract, a
    passage, the rows of a table, the labels on a figure.

    It is not question text and it is not furniture, and collapsing it into either
    loses marks. Read as question text it corrupts the question — a history paper's
    source extract became part of the question above it and cost eight marks. Read
    as furniture it is discarded, which is where it went next: the same extract
    vanished, and the question that asks "what does the source suggest?" was marked
    without the source. An economics table's numbers went the same way, stripped as
    bare page numbers, leaving "calculate the elasticity between the first and
    second rows" with the column headings and no data.

    So it is kept, and attached to the questions that refer to it."""
    MARKS = "marks"


class Requirement(BaseModel):
    """How many questions in a scope must be answered.

    ``answer_any`` of ``None`` means all of them. This is what makes the
    ``NOT_REQUIRED`` answer status possible — without it, a student correctly
    exercising choice looks identical to a student who skipped work.
    """

    model_config = ConfigDict(frozen=True)

    answer_any: int | None = Field(
        default=None,
        ge=1,
        description="Number of questions required from this scope; None means all.",
    )
    source_text: str | None = Field(
        default=None,
        description="The printed instruction this was parsed from, kept verbatim so a "
        "teacher can check our interpretation against the paper.",
    )

    @computed_field
    @property
    def is_optional(self) -> bool:
        return self.answer_any is not None


class Section(BaseModel):
    """A titled division of the paper, e.g. Section A."""

    model_config = ConfigDict(frozen=True)

    section_id: str
    label_raw: str = Field(description="As printed, e.g. 'SECTION B'.")
    requirement: Requirement = Field(default_factory=Requirement)
    total_marks: int | None = None


class ChoiceGroup(BaseModel):
    """A set of questions where answering any ``answer_any`` closes the group.

    Models both CBSE-style internal choice ("Q5 OR Q5-alternative") and
    "attempt any 4 of 7" at question granularity.
    """

    model_config = ConfigDict(frozen=True)

    group_id: str
    member_qids: list[str] = Field(min_length=2)
    answer_any: int = Field(default=1, ge=1)


class Stem(BaseModel):
    """Shared context serving several questions.

    Covers passage-based comprehension, case studies, and assertion-reason
    pairs. Modelled separately rather than duplicated into each question so
    that grading sees the context once and mapping does not mistake the passage
    itself for an answerable question.
    """

    model_config = ConfigDict(frozen=True)

    stem_id: str
    text: str
    line_ids: list[str] = Field(default_factory=list)
    geometry: list[PageBox] = Field(default_factory=list)

    material: list[str] = Field(
        default_factory=list,
        description="Text the paper printed for this question to refer to — a source "
        "extract, a passage, the rows of a table. Kept apart from `text` because it is "
        "not what was asked: the marks come from the question and the material is what "
        "the answer is judged against.",
    )
    material_line_ids: list[str] = Field(default_factory=list)
    material_geometry: list[PageBox] = Field(
        default_factory=list,
        description="Where the material sits, so a figure with no readable text can "
        "still be cropped and shown to something that can see it.",
    )


class Question(BaseModel):
    """One answerable question, or labelled sub-part.

    Identity and ordering are deliberately separate concerns:

      * ``qid`` is the canonical key used to join everything else to this
        question. Stable, unique, namespaced by section so that papers which
        restart numbering per section do not collide.
      * ``label_raw`` is exactly what the paper printed, never normalized. The
        requirement is to *preserve the original numbering*, and a teacher
        scanning the list needs to see the paper's own notation.
      * ``print_order`` is the sole authority on sequence. Labels cannot order
        anything reliably — they restart, they mix roman and alphabetic, and
        ``(ii)`` sorts before ``(i)`` as a string.
      * ``path`` holds raw label tokens with no interpretation of what each
        level *means*, so ``["2", "i", "a"]`` and ``["11", "a"]`` both work
        without the parser deciding whether romans outrank letters.
    """

    model_config = ConfigDict(frozen=True)

    qid: str = Field(description="Canonical identity, e.g. 'B/11/a'.")
    label_raw: str = Field(description="Verbatim as printed, e.g. '11 (a)'.")
    text: str = Field(description="Question text, furniture stripped.")
    path: list[str] = Field(min_length=1, description="Raw label tokens, outermost first.")
    print_order: int = Field(ge=0, description="Position in reading order. The ordering authority.")

    section_id: str | None = None
    stem_ref: str | None = None
    choice_group: str | None = None
    marks: int | None = Field(default=None, description="Printed marks, used as the grading denominator.")

    line_ids: list[str] = Field(default_factory=list)
    geometry: list[PageBox] = Field(default_factory=list)

    material: list[str] = Field(
        default_factory=list,
        description="Text the paper printed for this question to refer to — a source "
        "extract, a passage, the rows of a table. Kept apart from `text` because it is "
        "not what was asked: the marks come from the question and the material is what "
        "the answer is judged against.",
    )
    material_line_ids: list[str] = Field(default_factory=list)
    material_geometry: list[PageBox] = Field(
        default_factory=list,
        description="Where the material sits, so a figure with no readable text can "
        "still be cropped and shown to something that can see it.",
    )

    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    is_stem: bool = Field(
        default=False,
        description=(
            "This introduces the parts beneath it and carries no marks of its own, so it "
            "cannot be answered separately. A fact about the paper, not an inference: the "
            "question has children and printed no allocation. Kept as an extracted question "
            "because the requirement is to preserve the paper's numbering, but excluded from "
            "answer matching — a stem left in the candidate list can absorb the answer to its "
            "own sub-part — and never reported as unanswered, because nothing was asked."
        ),
    )

    @computed_field
    @property
    def depth(self) -> int:
        """Nesting depth. 1 for a top-level question, 2 for ``11(a)``."""
        return len(self.path)

    @computed_field
    @property
    def is_subpart(self) -> bool:
        return len(self.path) > 1

    @computed_field
    @property
    def parent_qid(self) -> str | None:
        """The qid this is a sub-part of, or None at top level."""
        if len(self.path) < 2:
            return None
        prefix = f"{self.section_id}/" if self.section_id else ""
        return prefix + "/".join(self.path[:-1])


class NumberingGap(BaseModel):
    """A suspected missing question, found by the monotonicity validator.

    Surfaced rather than silently tolerated. If a paper runs 1, 2, 4 then either
    question 3 exists and extraction missed it, or the paper genuinely skips it.
    Both are worth a teacher's attention, and only one is our bug.
    """

    model_config = ConfigDict(frozen=True)

    expected_label: str
    after_qid: str | None
    before_qid: str | None
    rescan_attempted: bool = False
    resolved: bool = False


class QuestionPaper(BaseModel):
    """Fully parsed question paper."""

    questions: list[Question] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    stems: list[Stem] = Field(default_factory=list)
    choice_groups: list[ChoiceGroup] = Field(default_factory=list)
    gaps: list[NumberingGap] = Field(default_factory=list)
    total_marks: int | None = None

    def in_print_order(self) -> list[Question]:
        return sorted(self.questions, key=lambda q: q.print_order)

    def by_qid(self) -> dict[str, Question]:
        return {q.qid: q for q in self.questions}
