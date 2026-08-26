"""Uploaded documents and their rasterized pages."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .geometry import RENDER_DPI


class DocumentKind(StrEnum):
    """Which of the two uploads a page belongs to.

    Kept explicit rather than inferred, because almost every downstream rule
    differs between them: the question paper is printed and parsed for
    structure, the answer sheet is handwritten and parsed for regions.
    """

    QUESTION_PAPER = "question_paper"
    ANSWER_SHEET = "answer_sheet"


class Page(BaseModel):
    """One rasterized page.

    ``image_key`` is an object-store key rather than image bytes. Pages are
    written to object storage and the bitmap freed as soon as OCR and ink
    extraction are done with it — holding 20 pages of 200 DPI bitmaps in
    memory is roughly 220 MB and is the fastest way to OOM a small worker.
    """

    model_config = ConfigDict(frozen=True)

    kind: DocumentKind
    index: int = Field(ge=0, description="0-indexed page number within its document.")
    width: int = Field(gt=0, description="Rendered width in pixels.")
    height: int = Field(gt=0, description="Rendered height in pixels.")
    dpi: int = Field(default=RENDER_DPI, gt=0)
    image_key: str = Field(description="Object-store key for the rendered PNG.")
    rotation_applied: float = Field(
        default=0.0,
        description="Degrees of deskew applied during preprocessing. Recorded so a "
        "highlight can be traced back to the raw scan when debugging.",
    )

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class SourceFile(BaseModel):
    """An uploaded file, before rasterization."""

    model_config = ConfigDict(frozen=True)

    filename: str
    kind: DocumentKind
    content_hash: str = Field(
        description="SHA-256 of the file bytes. This is the cache key: one question "
        "paper shared across a whole class is rendered and OCR'd once, which is what "
        "keeps a 1,000-page-per-month OCR free tier viable."
    )
    byte_size: int = Field(gt=0)
    page_count: int = Field(gt=0)
    has_text_layer: bool = Field(
        default=False,
        description="Whether the PDF carries extractable text. Recorded for diagnostics "
        "only — the pipeline always OCRs the raster and never trusts this layer, which "
        "is also what makes it immune to hidden-text prompt injection.",
    )
