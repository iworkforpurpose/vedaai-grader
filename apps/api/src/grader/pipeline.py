"""Ingest pipeline: render, transcribe, index.

Phase 1 scope. Question extraction, segmentation, alignment and grading attach
to the end of this as later stages; the shape here is what they hook into.

Deliberately not a LangGraph graph yet. With three sequential stages a graph
would be ceremony around a function call. It earns its place once there are
branches, retries and conditional re-reads to express — around Phase 4.
"""

from __future__ import annotations

from vedaai_contracts import (
    DocumentKind,
    LineIndex,
    Page,
    ProgressEvent,
    SourceFile,
    Stage,
    Submission,
    SubmissionStatus,
)

from . import render
from .ocr import EngineUnavailable, PageInput, select_engine, trusts_own_order
from .storage import PageStore
from .store import SubmissionStore


def ingest_document(
    *,
    submission: Submission,
    data: bytes,
    source: SourceFile,
    page_store: PageStore,
    submission_store: SubmissionStore,
) -> tuple[list[Page], LineIndex | None, list[str]]:
    """Render and transcribe one document.

    Returns the page metadata, the line index if a transcription engine was
    available, and any warnings a teacher should see.

    Pages are rendered one at a time and written straight to the page store, so
    only a single bitmap is live at any moment. Twenty pages at 200 DPI is
    roughly 220 MB of raw pixels — accumulating them is the fastest way to get
    a small worker killed.
    """
    warnings: list[str] = []
    pages: list[Page] = []
    per_page_lines: list[list] = []

    engine = None
    try:
        engine = select_engine(source)
    except EngineUnavailable as exc:
        # Not fatal. Rendering still produces the page images the reviewer needs,
        # and the missing transcription is reported rather than crashing the run.
        warnings.append(str(exc))

    submission_store.emit(
        submission.submission_id,
        ProgressEvent(
            stage=Stage.RENDERING,
            message=f"Rendering {source.filename}",
            pages_done=0,
            pages_total=source.page_count,
        ),
    )

    for rendered in render.render_pages(data, source, page_store):
        if rendered.png:
            page_store.put(rendered.page.image_key, rendered.png)
        pages.append(rendered.page)

        if engine is not None:
            # A cached page yields no bytes, but an image-based engine needs
            # pixels. Read them back rather than skipping: silently transcribing
            # nothing would look exactly like a blank page, which is the one
            # wrong answer this product must not give.
            png = rendered.png or (
                page_store.read(rendered.page.image_key)
                if page_store.exists(rendered.page.image_key)
                else None
            )
            page_input = PageInput(
                index=rendered.page.index,
                width=rendered.page.width,
                height=rendered.page.height,
                png=png,
                document=data,
                filename=source.filename,
            )
            per_page_lines.append(engine.transcribe(page_input))

        submission_store.emit(
            submission.submission_id,
            ProgressEvent(
                stage=Stage.TRANSCRIBING if engine else Stage.RENDERING,
                message=f"{source.filename}: page {rendered.page.index + 1}",
                pages_done=len(pages),
                pages_total=source.page_count,
            ),
        )

    if engine is None:
        return pages, None, warnings

    from .lineindex import build_index

    index = build_index(
        source.kind,
        per_page_lines,
        engine.engine,
        trust_engine_order=trusts_own_order(engine.engine),
    )

    if not index.lines:
        warnings.append(
            f"{source.filename}: nothing was transcribed. If this is a scan rather than a "
            "typed document, it needs an OCR engine."
        )

    return pages, index, warnings


def ingest(
    *,
    submission_id: str,
    question_paper: tuple[bytes, SourceFile] | None,
    answer_sheet: tuple[bytes, SourceFile] | None,
    page_store: PageStore,
    submission_store: SubmissionStore,
) -> Submission:
    """Run ingest for both documents and store the result."""
    submission = submission_store.require(submission_id)
    submission.status = SubmissionStatus.PROCESSING
    submission_store.put(submission)

    all_pages: list[Page] = []
    warnings: list[str] = []

    try:
        if question_paper is not None:
            data, source = question_paper
            submission.question_paper_file = source
            pages, index, warns = ingest_document(
                submission=submission,
                data=data,
                source=source,
                page_store=page_store,
                submission_store=submission_store,
            )
            all_pages.extend(pages)
            submission.question_paper_lines = index
            warnings.extend(warns)

        if answer_sheet is not None:
            data, source = answer_sheet
            submission.answer_sheet_file = source
            pages, index, warns = ingest_document(
                submission=submission,
                data=data,
                source=source,
                page_store=page_store,
                submission_store=submission_store,
            )
            all_pages.extend(pages)
            submission.answer_sheet_lines = index
            warnings.extend(warns)

        submission.pages = all_pages
        submission.warnings = warnings
        submission.status = SubmissionStatus.COMPLETE
        submission_store.put(submission)

        question_lines = (
            len(submission.question_paper_lines.lines) if submission.question_paper_lines else 0
        )
        answer_lines = (
            len(submission.answer_sheet_lines.lines) if submission.answer_sheet_lines else 0
        )
        submission_store.emit(
            submission_id,
            ProgressEvent(
                stage=Stage.DONE,
                message=(
                    f"Ingested {len(all_pages)} pages · "
                    f"{question_lines} question lines · {answer_lines} answer lines"
                ),
                pages_done=len(all_pages),
                pages_total=len(all_pages),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - report any stage failure to the client
        submission.status = SubmissionStatus.FAILED
        submission.error = str(exc)
        submission_store.put(submission)
        submission_store.emit(
            submission_id,
            ProgressEvent(stage=Stage.FAILED, message="Ingest failed", error=str(exc)),
        )
        raise

    return submission


def kind_pages(submission: Submission, kind: DocumentKind) -> list[Page]:
    """Pages belonging to one document, in order."""
    return sorted((p for p in submission.pages if p.kind is kind), key=lambda p: p.index)
