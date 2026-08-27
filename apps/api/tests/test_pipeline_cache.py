"""Regression tests for transcription across a page-render cache hit.

The bug these guard against is the most dangerous one available to this system.
Rendering is cached by content hash, and a cache hit yields no pixels. An
image-based transcription engine handed no pixels produces no lines — and a
document with no lines is indistinguishable from a genuinely blank script, so it
would be reported as "unanswered". A teacher acts on that claim without
re-reading the page, which is exactly why silent emptiness is worse here than a
loud failure.

A stub engine stands in for the real recognizer so these run in milliseconds
rather than loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from vedaai_contracts import BBox, DocumentKind, OcrEngine, Submission

from grader import pipeline, render
from grader.ocr.base import PageInput, TranscribedLine
from grader.storage import PageStore
from grader.store import SubmissionStore

from .fixtures import question_paper


@dataclass
class PixelHungryEngine:
    """A stub that requires pixels, as every real OCR engine does.

    It records what it was handed, so a test can assert the pipeline supplied
    pixels rather than merely that the output looked plausible.
    """

    calls: list[PageInput] = field(default_factory=list)

    @property
    def engine(self) -> OcrEngine:
        return OcrEngine.PADDLE_OCR_VL

    def available(self) -> bool:
        return True

    def transcribe(self, page: PageInput) -> list[TranscribedLine]:
        self.calls.append(page)
        if page.png is None:
            # Mirrors the real engines: no pixels means no work is possible.
            return []
        return [
            TranscribedLine(
                text=f"page {page.index} line",
                box=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
                confidence=0.8,
            )
        ]


@pytest.fixture
def stores(tmp_path) -> tuple[PageStore, SubmissionStore]:
    return PageStore(root=tmp_path / "pages"), SubmissionStore()


@pytest.fixture
def paper_bytes() -> bytes:
    """One document, built once and reused.

    This matters more than it looks. PyMuPDF embeds a creation timestamp, so
    calling the builder twice yields different bytes, a different content hash,
    and therefore a different cache key — meaning no cache hit ever occurs and a
    test that rebuilds the document silently exercises the cold path while
    appearing to test the warm one. That is how the first version of these tests
    passed even with the fix reverted.

    Uploading identical bytes is also the real-world case: a teacher submits the
    same question paper file once per student.
    """
    data, _ = question_paper()
    return data


def ingest_once(
    stores: tuple[PageStore, SubmissionStore],
    engine: PixelHungryEngine,
    monkeypatch,
    submission_id: str,
    data: bytes,
) -> tuple[int, Submission]:
    pages, submissions = stores
    monkeypatch.setattr(pipeline, "select_engine", lambda source: engine)

    source = render.inspect(data, "paper.pdf", DocumentKind.QUESTION_PAPER)
    submissions.put(Submission(submission_id=submission_id))

    result_pages, index, warnings = pipeline.ingest_document(
        submission=submissions.require(submission_id),
        data=data,
        source=source,
        page_store=pages,
        submission_store=submissions,
    )
    assert not warnings, warnings
    return len(index.lines if index else []), submissions.require(submission_id)


def test_the_cache_is_actually_hit(stores, monkeypatch, paper_bytes) -> None:
    """Guards the guard: prove the warm path is genuinely exercised.

    Without this, every test below could pass while only ever running the cold
    path, which is precisely the trap the first version fell into.
    """
    pages, _ = stores
    engine = PixelHungryEngine()
    ingest_once(stores, engine, monkeypatch, "s1", paper_bytes)

    source = render.inspect(paper_bytes, "paper.pdf", DocumentKind.QUESTION_PAPER)
    warm = list(render.render_pages(paper_bytes, source, pages))
    assert warm, "expected pages"
    assert all(item.png == b"" for item in warm), (
        "render did not report a cache hit, so the warm path is untested"
    )


def test_transcription_survives_a_render_cache_hit(stores, monkeypatch, paper_bytes) -> None:
    engine = PixelHungryEngine()

    first_count, _ = ingest_once(stores, engine, monkeypatch, "s1", paper_bytes)
    assert first_count > 0, "first pass should transcribe"

    # Second pass hits the render cache, so render_pages yields empty bytes.
    second_count, _ = ingest_once(stores, engine, monkeypatch, "s2", paper_bytes)

    assert second_count == first_count, (
        "a cache hit produced fewer lines than a cold run — the engine was handed "
        "no pixels, which would be reported as a blank page"
    )


def test_the_engine_is_always_handed_pixels(stores, monkeypatch, paper_bytes) -> None:
    engine = PixelHungryEngine()
    ingest_once(stores, engine, monkeypatch, "s1", paper_bytes)
    cold_calls = len(engine.calls)
    ingest_once(stores, engine, monkeypatch, "s2", paper_bytes)

    assert len(engine.calls) > cold_calls, "second pass should still call the engine"
    for call in engine.calls:
        assert call.png is not None, f"page {call.index} was transcribed without pixels"
        assert call.png.startswith(b"\x89PNG")


def test_page_dimensions_are_consistent_across_a_cache_hit(
    stores, monkeypatch, paper_bytes
) -> None:
    # Geometry is normalized against these dimensions. If a cached page reported
    # a different size, every box from the second run would be scaled against a
    # different basis than the image the browser displays.
    engine = PixelHungryEngine()
    ingest_once(stores, engine, monkeypatch, "s1", paper_bytes)
    cold = [(c.index, c.width, c.height) for c in engine.calls]

    engine.calls.clear()
    ingest_once(stores, engine, monkeypatch, "s2", paper_bytes)
    warm = [(c.index, c.width, c.height) for c in engine.calls]

    assert cold == warm
