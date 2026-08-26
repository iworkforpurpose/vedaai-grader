"""The question-to-answer mapping, and the highlights it produces."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .geometry import PageBox


class AnswerStatus(StrEnum):
    """The outcome for one question.

    The distinctions here are the product. "Which questions were left
    unanswered" is the stated goal, and a false "unanswered" is the worst error
    available: it is the one claim a teacher will act on without verifying,
    because checking it means re-reading the whole script.

    So the states below separate *reasons* rather than sharing one threshold.
    Only ``UNANSWERED`` asserts absence; everything else admits uncertainty.
    """

    ANSWERED = "answered"

    UNANSWERED = "unanswered"
    """Blank: no ink and no text where an answer would be. The only state that
    asserts the student did not answer."""

    OCR_FAILED = "ocr_failed"
    """Ink is present but nothing was transcribed. Presented as "not found —
    check this region", never as unanswered."""

    NOT_REQUIRED = "not_required"
    """Legitimately skipped under the paper's own choice rules — "answer any four
    of seven". Not an omission, and must never be reported as one."""

    PAGES_MISSING = "pages_missing"
    """Evidence points off the end of the upload: a continuation marker past the
    last page, or printed pagination exceeding the pages provided."""

    UNCERTAIN = "uncertain"
    """Downgraded by a global consistency check — substantial unassigned ink
    exists on the sheet, so absence cannot be claimed for anything."""

    @property
    def asserts_absence(self) -> bool:
        """Whether this status claims the student did not answer."""
        return self is AnswerStatus.UNANSWERED

    @property
    def needs_teacher_attention(self) -> bool:
        return self in {
            AnswerStatus.OCR_FAILED,
            AnswerStatus.PAGES_MISSING,
            AnswerStatus.UNCERTAIN,
        }


class MatchSignal(StrEnum):
    """Which evidence contributed to a mapping decision."""

    WRITTEN_LABEL = "written_label"
    SEMANTIC = "semantic"
    POSITION = "position"
    LENGTH_PLAUSIBILITY = "length_plausibility"
    CONTINUATION_MARKER = "continuation_marker"
    LLM_ADJUDICATION = "llm_adjudication"


class MatchEvidence(BaseModel):
    """Component scores behind one mapping, kept for display and debugging.

    Surfaced in the UI as a "why this matched" breakdown. A teacher deciding
    whether to trust a mapping is better served by "label matched, and the text
    is semantically close" than by a bare confidence percentage — and when a
    mapping is wrong, these are the numbers that say which signal misfired.
    """

    model_config = ConfigDict(frozen=True)

    label_agreement: float = Field(default=0.0, description="Weighted written-label contribution.")
    semantic_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    order_prior: float = Field(default=0.0)
    length_plausibility: float = Field(default=0.0)
    signals: list[MatchSignal] = Field(default_factory=list)
    total_score: float = Field(default=0.0, description="Combined DP match score.")


class Highlight(BaseModel):
    """Where an answer is, on the page.

    A list of per-page boxes rather than one box, which is what makes a
    multi-page answer representable without special-casing: the boxes simply
    carry different page indices. Geometry here is always derived from OCR line
    boxes or ink components, never produced by a model.
    """

    boxes: list[PageBox] = Field(default_factory=list)
    derived_from: str = Field(
        default="ocr_lines",
        description="'ocr_lines' or 'ink_regions' — which geometry source produced "
        "this. Ink-derived highlights are how text-free diagram answers stay "
        "highlightable when transcription returns nothing.",
    )

    @computed_field
    @property
    def pages(self) -> list[int]:
        return sorted({b.page for b in self.boxes})

    @computed_field
    @property
    def spans_pages(self) -> bool:
        return len(self.pages) > 1

    def to_hgbench(self) -> list[dict[str, object]]:
        """Emit in HG-Bench's published annotation format, for benchmarking."""
        return [b.to_hgbench() for b in self.boxes]


class Mapping(BaseModel):
    """One question's resolved answer."""

    qid: str
    status: AnswerStatus

    block_ids: list[str] = Field(default_factory=list)
    start_line_id: str | None = None
    end_line_id: str | None = None
    highlight: Highlight | None = None

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: MatchEvidence = Field(default_factory=MatchEvidence)

    anchor_id: str | None = None
    shares_block_with: list[str] = Field(
        default_factory=list,
        description="Other qids answered inside the same block. Populated when "
        "sub-parts were written as one undivided blob and then split.",
    )
    teacher_override: bool = Field(
        default=False,
        description="Set when a human reassigned this region. Overrides are never "
        "recomputed away by a later pipeline run.",
    )

    @computed_field
    @property
    def needs_review(self) -> bool:
        return self.status.needs_teacher_attention or (
            self.status is AnswerStatus.ANSWERED and self.confidence < 0.55
        )


class OrphanAnswer(BaseModel):
    """A block of writing that matches no question.

    Required by the brief, and worth surfacing prominently rather than
    discarding: an orphan usually means either the student answered something
    the paper does not contain, or — more often, and more importantly — that
    our own question extraction missed a question.
    """

    block_id: str
    text_preview: str
    highlight: Highlight
    best_guess_qid: str | None = None
    best_guess_score: float | None = None


class MappingResult(BaseModel):
    """The complete mapping outcome for a submission."""

    mappings: list[Mapping] = Field(default_factory=list)
    orphans: list[OrphanAnswer] = Field(default_factory=list)

    unassigned_ink_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of substantive ink belonging to no block. The global "
        "consistency check: when this is high, some answer went unmapped, so every "
        "absence claim is downgraded to UNCERTAIN rather than reported as unanswered.",
    )
    absence_claims_suppressed: bool = Field(
        default=False,
        description="True when unassigned ink forced that downgrade. Shown to the "
        "teacher as a banner, because it changes how the whole report should be read.",
    )

    def by_qid(self) -> dict[str, Mapping]:
        return {m.qid: m for m in self.mappings}

    def counts_by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.mappings:
            out[m.status.value] = out.get(m.status.value, 0) + 1
        return out
