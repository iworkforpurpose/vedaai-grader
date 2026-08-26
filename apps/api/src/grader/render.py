"""Rasterize uploaded documents into page images.

Two rules govern this module.

**Memory.** One page bitmap exists at a time. A 200 DPI A4 page is about
1654x2339 RGB, roughly 11 MB raw; twenty of them is 220 MB, which is more than
a small worker can hold alongside Python, OpenCV and a model. Each page is
rendered, written to the page store, and freed before the next is touched. This
is why the entry point is a generator rather than a function returning a list —
the shape of the API makes accumulating pages awkward on purpose.

**Everything gets rasterized.** Even a PDF with a perfectly good text layer is
rendered, because the browser needs page images to draw highlights over. Whether
the *text* comes from that layer or from OCR is a separate decision made in
``grader.ocr``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import fitz  # PyMuPDF
from vedaai_contracts import DocumentKind, Page, SourceFile
from vedaai_contracts.geometry import RENDER_DPI

from .storage import PageStore

#: Refuse documents beyond this many pages. A grading run is one paper and one
#: student's script; anything far larger is a misuse or an accident, and
#: rendering it would occupy the worker for minutes.
MAX_PAGES = 60

#: Refuse uploads beyond this size before any parsing happens.
MAX_BYTES = 40 * 1024 * 1024

#: Formats PyMuPDF opens directly. Images are wrapped into a single-page
#: document so the rest of the pipeline never learns whether the teacher
#: uploaded a PDF or a phone photo.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class UnsupportedDocument(ValueError):
    """Raised for input the pipeline will not attempt to process."""


@dataclass
class RenderedPage:
    """A page plus its pixels, valid only until the next iteration.

    ``png`` is handed to the caller so it can be written or inspected, then
    dropped. Holding a list of these defeats the point of the generator.
    """

    page: Page
    png: bytes


def compute_content_hash(data: bytes) -> str:
    """SHA-256 of the raw upload.

    Used as a cache key so a question paper shared across a class is rendered
    and transcribed once, not once per student. That is the difference between
    fitting inside a 1,000-page monthly OCR quota and blowing through it.
    """
    return hashlib.sha256(data).hexdigest()


def _open_document(data: bytes, filename: str) -> fitz.Document:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix in IMAGE_SUFFIXES:
        # Teachers photograph answer sheets. Wrapping the image in a one-page
        # PDF means downstream code has exactly one input shape to handle.
        try:
            image_doc = fitz.open(stream=data, filetype=suffix.lstrip("."))
            pdf_bytes = image_doc.convert_to_pdf()
            image_doc.close()
            return fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:  # noqa: BLE001 - surface any decode failure uniformly
            raise UnsupportedDocument(f"could not read image {filename!r}: {exc}") from exc

    try:
        return fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedDocument(
            f"{filename!r} is neither a readable PDF nor a supported image"
        ) from exc


def inspect(data: bytes, filename: str, kind: DocumentKind) -> SourceFile:
    """Validate an upload and describe it, without rendering anything.

    Runs before any expensive work so a 60-page or malformed document is
    rejected in milliseconds rather than after a minute of rasterization.
    """
    if not data:
        raise UnsupportedDocument(f"{filename!r} is empty")
    if len(data) > MAX_BYTES:
        raise UnsupportedDocument(
            f"{filename!r} is {len(data) / 1e6:.1f} MB; the limit is {MAX_BYTES / 1e6:.0f} MB"
        )

    doc = _open_document(data, filename)
    try:
        page_count = doc.page_count
        if page_count == 0:
            raise UnsupportedDocument(f"{filename!r} has no pages")
        if page_count > MAX_PAGES:
            raise UnsupportedDocument(
                f"{filename!r} has {page_count} pages; the limit is {MAX_PAGES}"
            )

        # Whether a usable text layer exists. Recorded for diagnostics and to
        # choose a transcription engine; a printed paper with real text is
        # better read directly than OCR'd.
        has_text = any(doc[i].get_text("text").strip() for i in range(min(page_count, 3)))

        return SourceFile(
            filename=filename,
            kind=kind,
            content_hash=compute_content_hash(data),
            byte_size=len(data),
            page_count=page_count,
            has_text_layer=has_text,
        )
    finally:
        doc.close()


def render_pages(
    data: bytes,
    source: SourceFile,
    page_store: PageStore,
    *,
    dpi: int = RENDER_DPI,
) -> Iterator[RenderedPage]:
    """Rasterize each page, yielding one at a time.

    Yields rather than returns so that only one page's pixels are live at any
    moment. Pages already present in the store are skipped and their metadata
    reconstructed, which makes re-processing the same paper nearly free.
    """
    doc = _open_document(data, source.filename)
    try:
        for index in range(doc.page_count):
            key = page_store.key_for(source.content_hash, index)
            page = doc[index]

            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            try:
                meta = Page(
                    kind=source.kind,
                    index=index,
                    width=pixmap.width,
                    height=pixmap.height,
                    dpi=dpi,
                    image_key=key,
                )
                if page_store.exists(key):
                    # Cache hit: dimensions still come from the pixmap so the
                    # geometry contract holds, but no bytes are rewritten.
                    yield RenderedPage(page=meta, png=b"")
                else:
                    yield RenderedPage(page=meta, png=pixmap.tobytes("png"))
            finally:
                # Explicit: PyMuPDF pixmaps hold their buffer outside Python's
                # ordinary refcount rhythm, and relying on GC here is what turns
                # a 20-page document into an out-of-memory kill.
                del pixmap
    finally:
        doc.close()


def page_size(data: bytes, filename: str, index: int, *, dpi: int = RENDER_DPI) -> tuple[int, int]:
    """Rendered pixel size of one page, without keeping the bitmap.

    Needed when converting text-layer coordinates, which PyMuPDF reports in
    points, into the normalized space the geometry contract requires.
    """
    doc = _open_document(data, filename)
    try:
        pixmap = doc[index].get_pixmap(dpi=dpi, alpha=False)
        try:
            return pixmap.width, pixmap.height
        finally:
            del pixmap
    finally:
        doc.close()
