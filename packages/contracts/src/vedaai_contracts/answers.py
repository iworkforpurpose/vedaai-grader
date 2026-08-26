"""Answer-sheet structure: ink regions, answer blocks, and label anchors."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .geometry import BBox, PageBox


class InkRegion(BaseModel):
    """A connected component of student ink, found without any OCR.

    This is the pipeline's second, independent geometry source, and it exists
    because OCR boxes cannot cover every answer. A hand-drawn diagram, a graph,
    or a chemical structure produces no text at all — yet the requirement is to
    highlight *the answer's region*, whatever the answer is made of.

    It also carries the signal that separates a genuinely blank space from one
    where recognition simply failed. Ink present with no text is an OCR failure;
    reporting it as "unanswered" would be confidently wrong in the single place
    a teacher is least likely to double-check.
    """

    model_config = ConfigDict(frozen=True)

    region_id: str
    page: int = Field(ge=0)
    box: BBox
    ink_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of the box's pixels that are ink. Distinguishes a dense "
        "diagram from a stray speck or scanner noise.",
    )
    pixel_count: int = Field(ge=0)

    @computed_field
    @property
    def is_substantive(self) -> bool:
        """Whether this is large and dense enough to be deliberate marking."""
        return self.pixel_count >= 200 and self.ink_ratio >= 0.02


class AnswerBlock(BaseModel):
    """A contiguous run of answer-sheet lines that appears to be one answer.

    Blocks are the units the alignment operates on. They come from line-gap and
    layout heuristics rather than from the model, so segmentation errors are
    inspectable and fixable in code.

    A block may carry no text at all — a pure-diagram answer is a block whose
    geometry comes entirely from ``ink_region_ids``.
    """

    block_id: str
    line_ids: list[str] = Field(default_factory=list)
    ink_region_ids: list[str] = Field(default_factory=list)
    text: str = Field(default="", description="Concatenated line text, reading order.")
    geometry: list[PageBox] = Field(default_factory=list)

    pages_spanned: list[int] = Field(default_factory=list)
    has_continuation_marker: bool = Field(
        default=False,
        description="Whether the block ends with something like 'cont. on page 7'. "
        "Treated as explicit evidence that the next block continues this answer.",
    )

    @computed_field
    @property
    def is_text_free(self) -> bool:
        """A block with geometry but no transcribed text — typically a drawing."""
        return not self.line_ids and bool(self.ink_region_ids)

    @computed_field
    @property
    def spans_pages(self) -> bool:
        return len(self.pages_spanned) > 1


class AnchorStatus(StrEnum):
    """How much a written question label can be trusted.

    Anchors are the strongest mapping signal available: a student writing "11(b)"
    in the margin is stating the answer's identity outright. They are also
    forgeable and mistakeable, and a wrongly trusted anchor silently mis-maps an
    answer while reporting high confidence — so every anchor starts as a
    hypothesis and must earn confirmation.
    """

    CONFIRMED = "confirmed"
    """Corroborated semantically or by order-consistency. May pin a segment."""

    DISPUTED = "disputed"
    """Contradicted by its own content. Downweighted to a mere scoring term."""

    UNVERIFIED = "unverified"
    """No corroborating evidence available either way."""


class Anchor(BaseModel):
    """A question label the student wrote on the answer sheet.

    ``claimed_qid`` is what the student's label points at; whether the
    surrounding writing actually answers that question is exactly what
    confirmation tests.
    """

    anchor_id: str
    claimed_label: str = Field(description="Raw label as written, e.g. '11 b)'.")
    claimed_qid: str | None = Field(
        default=None,
        description="Resolved question, or None when the label matches no question "
        "in the paper — itself a strong signal the student mislabelled.",
    )
    line_id: str
    page: int = Field(ge=0)
    box: BBox

    status: AnchorStatus = AnchorStatus.UNVERIFIED
    semantic_agreement: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity between the claimed question and the writing that "
        "follows this label. The primary confirmation test.",
    )
    order_consistent: bool | None = Field(
        default=None,
        description="Whether this anchor's position agrees with its neighbours. The "
        "fallback test for answers too short to embed meaningfully.",
    )

    @computed_field
    @property
    def may_pin(self) -> bool:
        """Whether this anchor is trusted enough to constrain the alignment."""
        return self.status is AnchorStatus.CONFIRMED
