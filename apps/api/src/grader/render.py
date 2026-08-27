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

#: Longest side of a rendered page, in pixels.
#:
#: Chosen to match the internal cap of the handwriting recognizer: it downsizes
#: anything longer than this before detection, so rendering larger produces
#: pixels that are resampled away again. Worse than merely wasteful — the
#: round trip softens pen strokes, which is exactly the detail recognition
#: depends on.
#:
#: This costs nothing in correctness because geometry is normalized. Only the
#: aspect ratio has to survive, and that is preserved.
MAX_RENDER_SIDE = 4000


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

    correction: str | None = None
    """What straightening was applied, phrased for a teacher, or None.

    Surfaced rather than silent: someone comparing the page on screen with the
    paper on their desk should not have to wonder why it looks different.
    """


def compute_content_hash(data: bytes) -> str:
    """SHA-256 of the raw upload.

    Used as a cache key so a question paper shared across a class is rendered
    and transcribed once, not once per student. That is the difference between
    fitting inside a 1,000-page monthly OCR quota and blowing through it.
    """
    return hashlib.sha256(data).hexdigest()


def is_image_upload(filename: str) -> bool:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return suffix in IMAGE_SUFFIXES


def native_pixel_size(data: bytes, filename: str) -> tuple[int, int] | None:
    """True pixel dimensions of an image upload, or None for a PDF.

    Decoded directly from the image bytes. Measuring the wrapped PDF page
    instead would report its size in points at 72 DPI — which is a *smaller*
    number than the real resolution for any modern photo, and would cause the
    caller to downscale a page it was only trying to avoid upscaling.
    """
    if not is_image_upload(filename):
        return None
    try:
        pixmap = fitz.Pixmap(data)
    except Exception:  # noqa: BLE001
        return None
    try:
        return pixmap.width, pixmap.height
    finally:
        del pixmap


def _open_document(data: bytes, filename: str) -> fitz.Document:
    if is_image_upload(filename):
        suffix = filename.rsplit(".", 1)[-1].lower()
        # Teachers photograph answer sheets. Wrapping the image in a one-page
        # PDF means downstream code has exactly one input shape to handle.
        try:
            image_doc = fitz.open(stream=data, filetype=suffix)
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


def _render_scale(
    page: fitz.Page,
    dpi: int,
    native: tuple[int, int] | None,
) -> float:
    """Scale factor for rasterizing one page, capped in two ways.

    PyMuPDF measures pages in points, so the requested DPI becomes a scale of
    ``dpi / 72``. Two ceilings apply on top of that:

    * never exceed ``MAX_RENDER_SIDE``, because the recognizer resamples
      anything larger back down before it looks at the page
    * never upscale a photograph beyond the resolution it was captured at,
      since the extra pixels are interpolation rather than detail

    Both are safe precisely because geometry is normalized: the aspect ratio is
    what has to survive, and scaling preserves it.
    """
    scale = dpi / 72.0

    if native is not None:
        native_longest = max(native)
        page_longest_pt = max(page.rect.width, page.rect.height)
        if page_longest_pt > 0:
            scale = min(scale, native_longest / page_longest_pt)

    projected = max(page.rect.width, page.rect.height) * scale
    if projected > MAX_RENDER_SIDE:
        scale *= MAX_RENDER_SIDE / projected

    # Never scale below 1:1 with the page's own point size, or fine strokes
    # vanish entirely.
    return max(scale, 1.0)


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

    A page with no text layer is a scan or a photograph, and it is straightened
    here before anything else sees it — see ``grader.preprocess`` for why this and
    nowhere else. The consequence to keep in mind: **the corrected bitmap is the
    page**. It is what gets stored, shown and read, its dimensions are what every
    coordinate is normalized against, and correction changes those dimensions. So
    the metadata below is taken from the image that is actually kept, never from
    the pixmap it came from.
    """
    doc = _open_document(data, source.filename)
    native = native_pixel_size(data, source.filename)
    # A text layer means a typed document, rendered square and evenly lit by
    # definition. Correcting one would be looking for a distortion that cannot be
    # there — and `page_size` converts that document's text-layer coordinates
    # without consulting this function, so a correction here would silently put
    # the two into different spaces.
    correcting = not source.has_text_layer

    try:
        for index in range(doc.page_count):
            key = page_store.key_for(source.content_hash, index)
            page = doc[index]

            if page_store.exists(key):
                # Cache hit. Dimensions come from the stored image, which is the
                # corrected one — reading them from a freshly rendered pixmap
                # would report the size before correction and shift every
                # coordinate on the page by the difference.
                stored = page_store.read(key)
                width, height = _png_size(stored)
                yield RenderedPage(
                    page=Page(
                        kind=source.kind,
                        index=index,
                        width=width,
                        height=height,
                        dpi=round(_render_scale(page, dpi, native) * 72),
                        image_key=key,
                    ),
                    png=b"",
                )
                continue

            scale = _render_scale(page, dpi, native)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            try:
                png = pixmap.tobytes("png")
                width, height = pixmap.width, pixmap.height
            finally:
                # Explicit: PyMuPDF pixmaps hold their buffer outside Python's
                # ordinary refcount rhythm, and relying on GC here is what turns
                # a 20-page document into an out-of-memory kill.
                del pixmap

            note: str | None = None
            if correcting:
                png, width, height, note = _corrected_png(png, width, height)

            yield RenderedPage(
                page=Page(
                    kind=source.kind,
                    index=index,
                    width=width,
                    height=height,
                    # The effective density, not the requested one. When a cap
                    # applies these differ, and recording the request would
                    # misreport what the geometry is actually relative to.
                    dpi=round(scale * 72),
                    image_key=key,
                ),
                png=png,
                correction=note,
            )
    finally:
        doc.close()


def _png_size(png: bytes) -> tuple[int, int]:
    """Pixel size of an encoded image, without keeping the decoded pixels."""
    import cv2
    import numpy as np

    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise UnsupportedDocument("a stored page image could not be read back")
    return decoded.shape[1], decoded.shape[0]


def _corrected_png(
    png: bytes, width: int, height: int
) -> tuple[bytes, int, int, str | None]:
    """Straighten a scanned or photographed page, or leave it exactly as it was.

    Failure here is never fatal. Correction improves recognition; it is not a
    prerequisite for it, and a page that cannot be straightened is still a page
    that can be read.
    """
    import cv2
    import numpy as np

    from .preprocess import correct

    try:
        decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return png, width, height, None

        result = correct(decoded, is_photograph=True)
        if not result.changed:
            return png, width, height, None

        ok, buffer = cv2.imencode(".png", result.image)
        if not ok:
            return png, width, height, None

        corrected_height, corrected_width = result.image.shape[:2]
        return buffer.tobytes(), corrected_width, corrected_height, result.describe()
    except Exception:  # noqa: BLE001 - correction is an improvement, not a requirement
        return png, width, height, None


def page_size(data: bytes, filename: str, index: int, *, dpi: int = RENDER_DPI) -> tuple[int, int]:
    """Rendered pixel size of one page, without keeping the bitmap.

    Needed when converting text-layer coordinates, which PyMuPDF reports in
    points, into the normalized space the geometry contract requires.
    """
    doc = _open_document(data, filename)
    native = native_pixel_size(data, filename)
    try:
        page = doc[index]
        # Must apply the same caps as render_pages, or the two disagree about
        # the page size and every normalized coordinate derived from this is
        # scaled by the difference.
        scale = _render_scale(page, dpi, native)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        try:
            return pixmap.width, pixmap.height
        finally:
            del pixmap
    finally:
        doc.close()
