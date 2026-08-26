"""The transcription adapter interface.

Engines differ in what they need: a cloud OCR service wants page pixels, while
a PDF text-layer reader wants the original document. ``PageInput`` carries both
so a single interface covers them, and the pipeline never branches on which
engine is in use.

Every engine is responsible for one thing beyond recognition: converting its own
coordinate system into the normalized geometry contract, using the rendered page
size it is handed. That conversion happens here at the boundary and nowhere else,
which is what keeps engine-specific pixel and point conventions from leaking
into the alignment and rendering code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from vedaai_contracts import BBox, OcrEngine, Word


@dataclass(frozen=True)
class PageInput:
    """Everything an engine might need to transcribe one page."""

    index: int
    """0-indexed page number within its document."""

    width: int
    """Rendered width in pixels. The basis for normalizing pixel coordinates."""

    height: int
    """Rendered height in pixels."""

    png: bytes | None = None
    """Rendered image, for engines that work from pixels."""

    document: bytes | None = None
    """Original file bytes, for engines that read an embedded text layer."""

    filename: str = ""


@dataclass
class TranscribedLine:
    """One recognized line, before it is assigned a line ID.

    IDs are allocated centrally in ``grader.lineindex`` so they run in reading
    order across the whole document. An engine numbering its own lines would
    produce IDs that restart per page.
    """

    text: str
    box: BBox
    confidence: float = 1.0
    words: list[Word] = field(default_factory=list)


@runtime_checkable
class TranscriptionEngine(Protocol):
    """Produces lines with geometry for a single page."""

    @property
    def engine(self) -> OcrEngine: ...

    def available(self) -> bool:
        """Whether this engine can run right now.

        Checked before use so a missing API key degrades to a clear message and
        a fallback engine, rather than an exception from inside a pipeline stage.
        """
        ...

    def transcribe(self, page: PageInput) -> list[TranscribedLine]:
        """Transcribe one page. Order within the page need not be reading order."""
        ...


class EngineUnavailable(RuntimeError):
    """Raised when an engine is selected but cannot run."""
