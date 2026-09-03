"""Uploading past the service, and refusing keys it did not issue."""

from __future__ import annotations

import pytest

from grader import uploads


class TestKeyShape:
    """The key arrives from the client, so its shape is a security boundary.

    Reading an arbitrary key would let a caller name any object in the bucket —
    another submission's rendered pages, or anything else sharing it. Only the shape
    this module generates is accepted, and everything below is a way of not
    generating it.
    """

    def test_accepts_a_key_it_issued(self) -> None:
        key = uploads.new_key("answer_sheet", "suyash_6c.pdf")
        assert uploads.check(key) == key

    def test_keeps_a_sensible_extension(self) -> None:
        assert uploads.new_key("answer_sheet", "scan.PDF").endswith("/answer_sheet.pdf")
        assert uploads.new_key("question_paper", "paper.png").endswith("/question_paper.png")

    def test_drops_an_implausible_extension(self) -> None:
        # No extension is preferable to a long or non-alphanumeric one reaching a
        # key. The document is identified by its bytes, so nothing needs it.
        assert uploads.new_key("answer_sheet", "sheet.thisisnotanextension").endswith(
            "/answer_sheet"
        )
        assert uploads.new_key("answer_sheet", "sheet.p df").endswith("/answer_sheet")

    @pytest.mark.parametrize(
        "key",
        [
            "pages/058f7ebffc67155a/p0000.png",  # someone else's rendered page
            "../pages/058f7ebffc67155a/p0000.png",  # traversal
            "/etc/passwd",
            "deadbeef/answer_sheet.pdf",  # directory too short to be a uuid
            "0123456789abcdef0123456789abcdef/marks.csv",  # not a document kind
            "0123456789abcdef0123456789abcdef/answer_sheet.pdf/../../x",
            "",
            "0123456789ABCDEF0123456789ABCDEF/answer_sheet.pdf",  # uppercase hex
        ],
    )
    def test_refuses_anything_else(self, key: str) -> None:
        with pytest.raises(uploads.UploadRejected):
            uploads.check(key)

    def test_object_key_is_prefixed_and_checked(self) -> None:
        key = uploads.new_key("question_paper", "p.pdf")
        assert uploads.object_key(key).startswith(uploads.UPLOAD_PREFIX)
        with pytest.raises(uploads.UploadRejected):
            uploads.object_key("pages/x/p0000.png")

    def test_two_attempts_never_collide(self) -> None:
        # A random directory per attempt, not a content hash: the hash is unknown
        # until the bytes arrive, and two students uploading the same paper must not
        # overwrite each other's in-flight upload.
        keys = {uploads.new_key("answer_sheet", "same.pdf") for _ in range(200)}
        assert len(keys) == 200


class TestPresigning:
    def test_signs_the_destination_and_a_size_but_not_the_content_type(
        self, monkeypatch
    ) -> None:
        """Content type is deliberately not signed; size deliberately is.

        Signing the type means a browser that normalises the header, or guesses a
        different type for the same bytes, gets a signature mismatch nobody can
        debug from the client side. Nothing downstream trusts it anyway — the
        document is identified by inspecting its bytes.

        Size is the opposite case. A signed PUT cannot carry a length condition at
        all, which is why this is a POST policy: without one the URL authorised an
        object of any size against the operator's bucket, and the service's own
        40 MB cap did not run until the renderer, long after the bytes had landed.
        """
        seen: dict = {}

        class FakeS3:
            def generate_presigned_post(self, **kwargs):
                seen.update(kwargs)
                return {"url": "https://example.test/post", "fields": {"key": "k"}}

        monkeypatch.setenv("S3_PAGE_BUCKET", "a-bucket")
        slot = uploads.presign("answer_sheet", "sheet.pdf", client=FakeS3())

        assert seen["Bucket"] == "a-bucket"
        assert seen["Key"] == uploads.object_key(slot.key)
        assert seen["ExpiresIn"] == uploads.URL_TTL_SECONDS
        assert seen["Conditions"] == [
            ["content-length-range", 1, uploads.MAX_UPLOAD_BYTES]
        ]
        assert not any("Content-Type" in str(c) for c in seen["Conditions"])
        assert slot.url == "https://example.test/post"
        assert slot.fields == {"key": "k"}

    def test_unavailable_without_a_bucket(self, monkeypatch) -> None:
        # The local case. There is nothing to presign, and saying so lets the client
        # post the file instead of failing at a URL that cannot exist.
        monkeypatch.delenv("S3_PAGE_BUCKET", raising=False)
        assert uploads.available() is False

    def test_available_with_one(self, monkeypatch) -> None:
        monkeypatch.setenv("S3_PAGE_BUCKET", "a-bucket")
        assert uploads.available() is True


class TestDiscard:
    def test_a_failed_tidy_up_is_swallowed(self, monkeypatch) -> None:
        """Cleanup runs after the work succeeded, so it must not be able to fail it.

        An object left behind is expired by the bucket lifecycle rule anyway. A
        submission reporting failure because its cleanup did would be a worse
        outcome than the thing it was cleaning.
        """
        monkeypatch.setenv("S3_PAGE_BUCKET", "a-bucket")

        class Broken:
            def delete_object(self, **_: object) -> None:
                raise RuntimeError("access denied")

        key = uploads.new_key("answer_sheet", "s.pdf")
        uploads.discard(key, client=Broken())  # must not raise

    def test_deletes_the_prefixed_key(self, monkeypatch) -> None:
        monkeypatch.setenv("S3_PAGE_BUCKET", "a-bucket")
        calls: list[dict] = []

        class Recorder:
            def delete_object(self, **kwargs: object) -> None:
                calls.append(kwargs)

        key = uploads.new_key("answer_sheet", "s.pdf")
        uploads.discard(key, client=Recorder())
        assert calls == [{"Bucket": "a-bucket", "Key": uploads.object_key(key)}]


class TestTheSignedDestinationIsBounded:
    """A presigned URL is a capability, and this one had no limit on it.

    `POST /uploads` was unauthenticated and unthrottled, and the URL it returned
    signed only the bucket and the key. So a caller could mint them in a loop and
    PUT objects of any size — up to S3's 5 GB single-object limit — into the
    operator's bucket. The service's own 40 MB cap ran afterwards, in the
    renderer, long after the bytes had landed and been paid for.
    """

    def test_the_service_cap_and_the_signed_cap_are_the_same_number(self) -> None:
        """Two caps that can drift are one cap and one bug."""
        from grader import render, uploads

        assert uploads.MAX_UPLOAD_BYTES == render.MAX_BYTES

    def test_the_signature_carries_a_size_range(self) -> None:
        from grader import uploads

        captured: dict = {}

        class FakeS3:
            def generate_presigned_post(self, **kwargs):
                captured.update(kwargs)
                return {"url": "https://s3.example/bucket", "fields": {"key": "k"}}

        uploads.presign("answer_sheet", "script.pdf", client=FakeS3())

        assert ["content-length-range", 1, uploads.MAX_UPLOAD_BYTES] in captured[
            "Conditions"
        ], "an unbounded signed destination is an unbounded write to the bucket"

    def test_an_oversized_object_is_refused_before_it_is_downloaded(self) -> None:
        """`get_object(...).read()` is one allocation of whatever it finds.

        A cap that runs after the download is a cap on the wrong thing.
        """
        from grader import uploads

        key = "0123456789abcdef0123456789abcdef/answer_sheet.pdf"
        downloaded: list[str] = []

        class FakeS3:
            def head_object(self, **kwargs):
                return {"ContentLength": uploads.MAX_UPLOAD_BYTES + 1}

            def get_object(self, **kwargs):
                downloaded.append(kwargs["Key"])
                raise AssertionError("must not download an object it has refused")

        with pytest.raises(uploads.UploadRejected):
            uploads.read(key, client=FakeS3())
        assert downloaded == []

    def test_an_acceptable_object_is_read(self) -> None:
        from grader import uploads

        key = "0123456789abcdef0123456789abcdef/answer_sheet.pdf"

        class Body:
            def read(self) -> bytes:
                return b"%PDF-1.4"

        class FakeS3:
            def head_object(self, **kwargs):
                return {"ContentLength": 8}

            def get_object(self, **kwargs):
                return {"Body": Body()}

        assert uploads.read(key, client=FakeS3()) == b"%PDF-1.4"
