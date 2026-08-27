"""Tests for rasterization and upload validation."""

from __future__ import annotations

import pytest
from vedaai_contracts import DocumentKind
from vedaai_contracts.geometry import RENDER_DPI

from grader import render
from grader.render import UnsupportedDocument
from grader.storage import PageStore

from .fixtures import (
    image_with_known_size,
    question_paper,
    single_page_image,
    skewed_photo_image,
)


@pytest.fixture
def page_store(tmp_path) -> PageStore:
    return PageStore(root=tmp_path / "pages")


class TestInspect:
    def test_describes_a_typed_paper(self) -> None:
        data, _ = question_paper()
        source = render.inspect(data, "science_unit_test.pdf", DocumentKind.QUESTION_PAPER)

        assert source.page_count >= 1
        assert source.byte_size == len(data)
        assert len(source.content_hash) == 64
        # A typed paper carries real text, which is what lets the exact
        # text-layer engine be used instead of OCR.
        assert source.has_text_layer

    def test_content_hash_is_stable_and_content_addressed(self) -> None:
        # This hash is the cache key that makes one paper shared across a class
        # cost one render rather than one per student.
        data, _ = question_paper()
        a = render.inspect(data, "a.pdf", DocumentKind.QUESTION_PAPER)
        b = render.inspect(data, "different_name.pdf", DocumentKind.QUESTION_PAPER)
        assert a.content_hash == b.content_hash

    def test_accepts_an_image_upload(self) -> None:
        # Teachers photograph answer sheets; a photo must be as acceptable as a
        # PDF, and downstream code should not be able to tell the difference.
        source = render.inspect(single_page_image(), "photo.png", DocumentKind.ANSWER_SHEET)
        assert source.page_count == 1

    def test_rejects_an_empty_upload(self) -> None:
        with pytest.raises(UnsupportedDocument, match="is empty"):
            render.inspect(b"", "nothing.pdf", DocumentKind.ANSWER_SHEET)

    def test_rejects_a_file_that_is_not_a_document(self) -> None:
        with pytest.raises(UnsupportedDocument, match="neither a readable PDF"):
            render.inspect(b"this is not a pdf", "notes.pdf", DocumentKind.ANSWER_SHEET)

    def test_rejects_an_oversized_upload_before_parsing(self) -> None:
        oversized = b"x" * (render.MAX_BYTES + 1)
        with pytest.raises(UnsupportedDocument, match="the limit is"):
            render.inspect(oversized, "huge.pdf", DocumentKind.ANSWER_SHEET)

    def test_rejects_too_many_pages(self, monkeypatch) -> None:
        # Guards the worker: rasterizing an accidental 500-page upload would
        # occupy it for minutes.
        data, _ = question_paper()
        monkeypatch.setattr(render, "MAX_PAGES", 1)
        with pytest.raises(UnsupportedDocument, match="the limit is 1"):
            render.inspect(data, "long.pdf", DocumentKind.QUESTION_PAPER)


class TestRenderPages:
    def test_yields_one_page_at_a_time(self, page_store: PageStore) -> None:
        data, _ = question_paper()
        source = render.inspect(data, "paper.pdf", DocumentKind.QUESTION_PAPER)

        rendered = list(render.render_pages(data, source, page_store))
        assert len(rendered) == source.page_count

        for i, item in enumerate(rendered):
            assert item.page.index == i
            assert item.page.dpi == RENDER_DPI
            assert item.page.width > 0 and item.page.height > 0
            assert item.png.startswith(b"\x89PNG")

    def test_page_dimensions_match_the_declared_dpi(self, page_store: PageStore) -> None:
        # The geometry contract is "normalized against the page rendered at
        # RENDER_DPI". If rendering silently used a different density, every
        # normalized box would still be valid but subtly wrong.
        data, _ = question_paper()
        source = render.inspect(data, "paper.pdf", DocumentKind.QUESTION_PAPER)
        first = next(iter(render.render_pages(data, source, page_store)))

        # A4 at 200 DPI is about 1654 x 2339.
        assert 1600 < first.page.width < 1700
        assert 2300 < first.page.height < 2400

    def test_reuses_cached_pages_without_rewriting_bytes(self, page_store: PageStore) -> None:
        data, _ = question_paper()
        source = render.inspect(data, "paper.pdf", DocumentKind.QUESTION_PAPER)

        for item in render.render_pages(data, source, page_store):
            page_store.put(item.page.image_key, item.png)

        # Second pass: metadata still complete, but no pixels handed back, which
        # is what makes reprocessing the same paper cheap.
        second = list(render.render_pages(data, source, page_store))
        assert all(item.png == b"" for item in second)
        assert all(page_store.exists(item.page.image_key) for item in second)

    def test_keys_are_namespaced_by_content(self, page_store: PageStore) -> None:
        data, _ = question_paper()
        source = render.inspect(data, "paper.pdf", DocumentKind.QUESTION_PAPER)
        keys = [r.page.image_key for r in render.render_pages(data, source, page_store)]
        assert len(set(keys)) == len(keys)
        assert all(k.startswith(source.content_hash[:16]) for k in keys)


class TestRenderScaleCaps:
    def test_a4_pdf_renders_at_the_requested_density(self, page_store: PageStore) -> None:
        # A4 at 200 DPI is ~2339px on its long side, comfortably under the cap,
        # so neither ceiling should engage and the density is honoured exactly.
        data, _ = question_paper()
        source = render.inspect(data, "paper.pdf", DocumentKind.QUESTION_PAPER)
        first = next(iter(render.render_pages(data, source, page_store)))
        assert first.page.dpi == RENDER_DPI

    def test_a_photo_is_never_upscaled_past_its_own_resolution(
        self, page_store: PageStore
    ) -> None:
        # Rendering a photo larger than it was taken invents pixels by
        # interpolation, and the recognizer resamples them away again. The round
        # trip softens pen strokes, which is the detail recognition needs.
        data = single_page_image()
        native = render.native_pixel_size(data, "photo.png")
        assert native is not None

        source = render.inspect(data, "photo.png", DocumentKind.ANSWER_SHEET)
        first = next(iter(render.render_pages(data, source, page_store)))

        assert max(first.page.width, first.page.height) <= max(native) + 1

    def test_nothing_renders_beyond_the_recognizer_cap(self, page_store: PageStore) -> None:
        data, _ = question_paper()
        source = render.inspect(data, "paper.pdf", DocumentKind.QUESTION_PAPER)
        for item in render.render_pages(data, source, page_store):
            assert max(item.page.width, item.page.height) <= render.MAX_RENDER_SIDE

    def test_page_size_agrees_with_what_was_rendered(self, page_store: PageStore) -> None:
        # These two must apply identical caps. If they diverge, every coordinate
        # normalized against page_size is scaled by the difference — a uniform
        # drift that looks like a mapping fault, not a rendering one.
        data = single_page_image()
        source = render.inspect(data, "photo.png", DocumentKind.ANSWER_SHEET)
        first = next(iter(render.render_pages(data, source, page_store)))
        assert render.page_size(data, "photo.png", 0) == (first.page.width, first.page.height)

    def test_native_size_is_the_true_decoded_resolution(self) -> None:
        # Regression guard. An earlier version measured the wrapped PDF page at
        # 72 DPI instead of decoding the image, which under-reports the
        # resolution of any modern photo — so the no-upscale rule became a
        # downscale, discarding real detail in the name of preserving it.
        data = image_with_known_size(1448, 2047)
        assert render.native_pixel_size(data, "photo.png") == (1448, 2047)

    def test_a_photo_renders_at_its_native_resolution(self, page_store: PageStore) -> None:
        data = image_with_known_size(1448, 2047)
        source = render.inspect(data, "photo.png", DocumentKind.ANSWER_SHEET)
        first = next(iter(render.render_pages(data, source, page_store)))
        assert (first.page.width, first.page.height) == (1448, 2047)

    def test_native_size_is_none_for_a_pdf(self) -> None:
        data, _ = question_paper()
        assert render.native_pixel_size(data, "paper.pdf") is None

    def test_identifies_image_uploads_by_suffix(self) -> None:
        assert render.is_image_upload("scan.JPEG")
        assert render.is_image_upload("photo.heic") is False  # not yet supported
        assert render.is_image_upload("paper.pdf") is False


class TestPageStore:
    def test_round_trips_bytes(self, page_store: PageStore) -> None:
        key = page_store.key_for("a" * 64, 3)
        page_store.put(key, b"payload")
        assert page_store.read(key) == b"payload"
        assert page_store.exists(key)

    def test_rejects_a_traversal_key(self, page_store: PageStore) -> None:
        # Reachable from a URL path in the image endpoint, so it must not rely
        # on callers being careful.
        with pytest.raises(ValueError, match="escapes the page store"):
            page_store.path_for("../../etc/passwd")

    def test_exists_is_false_for_a_traversal_key(self, page_store: PageStore) -> None:
        assert not page_store.exists("../../etc/passwd")


class TestCorrectionAndTheCoordinateSpace:
    """Correction changes the page's size, which every coordinate depends on.

    The bug these guard against is the worst kind this codebase can have: geometry
    that is internally consistent, confidently drawn, and in the wrong place. It
    appears the moment the size reported for a page stops matching the image
    actually stored for it.
    """

    def test_a_typed_page_is_never_corrected(self, tmp_path) -> None:
        # A text layer means the document is square and evenly lit by definition.
        # It also means `page_size` converts its coordinates without consulting
        # `render_pages`, so a correction here would put the two in different
        # spaces with nothing to reveal it.
        data, _ = question_paper()
        store = PageStore(root=tmp_path)
        source = render.inspect(data, "paper.pdf", DocumentKind.QUESTION_PAPER)

        for rendered in render.render_pages(data, source, store):
            assert rendered.correction is None
            width, height = render.page_size(data, "paper.pdf", rendered.page.index)
            assert (rendered.page.width, rendered.page.height) == (width, height)

    def test_reported_size_matches_the_stored_image(self, tmp_path) -> None:
        # The invariant in one assertion, on a photograph, where correction does
        # run and does resize.
        data = skewed_photo_image()
        store = PageStore(root=tmp_path)
        source = render.inspect(data, "photo.png", DocumentKind.ANSWER_SHEET)

        corrected_any = False
        for rendered in render.render_pages(data, source, store):
            store.put(rendered.page.image_key, rendered.png)
            corrected_any = corrected_any or rendered.correction is not None
            stored_width, stored_height = render._png_size(store.read(rendered.page.image_key))
            assert (rendered.page.width, rendered.page.height) == (stored_width, stored_height)

        assert corrected_any, "fixture must actually trigger a correction, or this proves nothing"

    def test_a_cached_page_reports_the_size_of_what_was_cached(self, tmp_path) -> None:
        # The trap: on a second pass the pixmap is re-rendered at its *original*
        # size, and reading dimensions from it would report the page as it was
        # before straightening. Every coordinate on the page would then be scaled
        # by the difference — invisibly, because both numbers look plausible.
        data = skewed_photo_image()
        store = PageStore(root=tmp_path)
        source = render.inspect(data, "photo.png", DocumentKind.ANSWER_SHEET)

        first = []
        for rendered in render.render_pages(data, source, store):
            store.put(rendered.page.image_key, rendered.png)
            first.append((rendered.page.width, rendered.page.height))

        second = [
            (rendered.page.width, rendered.page.height)
            for rendered in render.render_pages(data, source, store)
        ]
        assert second == first
