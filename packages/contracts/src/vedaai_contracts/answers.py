"""Answer-sheet structure: ink regions, answer blocks, and label anchors."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .geometry import BBox, PageBox


class InkRegionKind(StrEnum):
    """What a region of ink actually is.

    Measured on real scripts, low recognition confidence turns out to conflate
    three quite different things, and they need opposite treatment. Ink darkness
    and density are what separate them — which is why the classification lives
    with the ink, not behind a confidence threshold somewhere downstream.
    """

    WRITING = "writing"
    """Deliberate marking by the student. The default."""

    STRUCK_THROUGH = "struck_through"
    """Work the student crossed out. Excluded from grading and from competing
    for a question — otherwise an abandoned wrong answer can be marked instead
    of the rewrite below it, an error invisible from the score alone. Retained
    in geometry, because it still occupies space on the page."""

    BLEED_THROUGH = "bleed_through"
    """Writing showing through from the reverse side. Not the student's answer on
    this page at all, so excluded entirely — including from the unassigned-ink
    consistency check, since it appears on every double-sided script and would
    otherwise suppress every legitimate absence claim."""

    NOISE = "noise"
    """Speckle, scanner artefacts, a page edge. Too small or too faint to be
    deliberate."""

    @property
    def is_student_answer(self) -> bool:
        """Whether this region may be treated as answer content."""
        return self is InkRegionKind.WRITING

    @property
    def counts_as_page_ink(self) -> bool:
        """Whether this contributes to "there is writing here" evidence.

        Struck-through work counts: the student did write in that space, so its
        presence argues against the page being blank. Bleed-through and noise do
        not.
        """
        return self in {InkRegionKind.WRITING, InkRegionKind.STRUCK_THROUGH}


class InkRegion(BaseModel):
    """A connected component of ink, found without any OCR.

    This is the pipeline's second, independent geometry source, and it exists
    because OCR boxes cannot cover every answer. Two reasons, one anticipated and
    one measured.

    The anticipated one: a hand-drawn diagram, a graph, or a chemical structure
    produces no text at all, yet the requirement is to highlight *the answer's
    region*, whatever the answer is made of.

    The measured one: detection recall on real handwriting is about 90%. A long
    declaration line in the test script produced no box whatsoever. A missed line
    is a missing highlight unless something that does not depend on recognition
    can still find the ink.

    It also carries the signal separating a genuinely blank space from one where
    recognition merely failed. Ink present with no text is an OCR failure;
    reporting it as "unanswered" would be confidently wrong in the one place a
    teacher is least likely to check.
    """

    model_config = ConfigDict(frozen=True)

    region_id: str
    page: int = Field(ge=0)
    box: BBox
    kind: InkRegionKind = InkRegionKind.WRITING

    ink_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of the box's pixels that are ink. Scribbled-over text "
        "carries markedly more ink than ordinary writing, because the original text "
        "and the scribble are both present.",
    )
    mean_darkness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Mean intensity of the region's ink pixels, 0 = black, 1 = white. "
        "This is what identifies bleed-through: it is faint by nature, whereas pen on "
        "the near side of the paper is dark. Density alone cannot tell them apart.",
    )
    has_horizontal_strike: bool = Field(
        default=False,
        description="A long horizontal run through the region, i.e. a clean single-line "
        "crossing out. Catches the tidy case that ink density alone would miss; scribbled "
        "deletions are caught by density instead.",
    )
    pixel_count: int = Field(ge=0)
    covered_by_ocr: bool = Field(
        default=False,
        description="Whether any transcribed line overlaps this region. False means the "
        "recognizer found nothing here — either a diagram, or a line it simply missed. "
        "Both still need to be highlightable.",
    )

    @computed_field
    @property
    def is_substantive(self) -> bool:
        """Whether this is large and dense enough to be deliberate marking."""
        return self.pixel_count >= 200 and self.ink_ratio >= 0.02

    @computed_field
    @property
    def is_orphan_ink(self) -> bool:
        """Student marking that transcription did not account for.

        These are the regions that make incomplete OCR recall survivable: they
        carry geometry for content the recognizer never reported.
        """
        return (
            self.kind is InkRegionKind.WRITING and not self.covered_by_ocr and self.is_substantive
        )


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
