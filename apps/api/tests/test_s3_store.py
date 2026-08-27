"""Tests for the S3 page store.

No bucket and no network: the store is a thin translation over four S3 calls, and
a fake client exercises the translation exactly. What is worth testing is not that
boto3 works but the two decisions layered on top of it — that a key cannot escape
the prefix, and that a failure is never mistaken for an absent object.
"""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from grader.storage import PageStore, S3PageStore


class FakeS3:
    """An in-memory stand-in for the parts of the S3 client the store uses."""

    def __init__(self, *, head_error: ClientError | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.head_error = head_error
        self.puts: list[dict] = []

    def head_object(self, *, Bucket: str, Key: str) -> dict:  # noqa: N803 - boto3 casing
        if self.head_error is not None:
            raise self.head_error
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"ContentLength": len(self.objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs) -> dict:  # noqa: N803
        self.puts.append({"Bucket": Bucket, "Key": Key, **kwargs})
        self.objects[Key] = Body
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:  # noqa: N803
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "nope"}}, "GetObject")

        class Body:
            def __init__(self, data: bytes) -> None:
                self.data = data

            def read(self) -> bytes:
                return self.data

        return {"Body": Body(self.objects[Key])}

    def list_objects_v2(self, *, Bucket: str, Prefix: str, **kwargs) -> dict:  # noqa: N803
        keys = [k for k in self.objects if k.startswith(Prefix)]
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def delete_objects(self, *, Bucket: str, Delete: dict) -> dict:  # noqa: N803
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
        return {}


def store(client: FakeS3, prefix: str = "pages/") -> S3PageStore:
    return S3PageStore("test-bucket", prefix=prefix, client=client)


class TestRoundTrip:
    def test_a_page_written_can_be_read_back(self) -> None:
        client = FakeS3()
        subject = store(client)
        key = subject.key_for("a" * 64, 3)

        subject.put(key, b"\x89PNG fake")
        assert subject.exists(key) is True
        assert subject.read(key) == b"\x89PNG fake"

    def test_an_absent_page_is_absent(self) -> None:
        assert store(FakeS3()).exists("nothing/p0000.png") is False

    def test_the_prefix_is_applied_to_the_stored_key(self) -> None:
        client = FakeS3()
        subject = store(client, prefix="renders/")
        subject.put("abc/p0001.png", b"data")
        assert list(client.objects) == ["renders/abc/p0001.png"]

    def test_pages_are_stored_as_png(self) -> None:
        # The browser fetches these directly, so the content type has to be right
        # or it downloads instead of rendering.
        client = FakeS3()
        store(client).put("abc/p0000.png", b"data")
        assert client.puts[0]["ContentType"] == "image/png"


class TestKeysMatchTheLocalStore:
    def test_the_two_backends_agree_on_keys(self) -> None:
        # The same upload has to produce the same key either side, or switching
        # backends silently invalidates everything already rendered.
        content_hash = "f" * 64
        assert S3PageStore.key_for(content_hash, 7) == PageStore.key_for(content_hash, 7)


class TestKeySafety:
    @pytest.mark.parametrize("key", ["../secrets/p0000.png", "/etc/passwd", "a/../../b.png"])
    def test_a_key_cannot_escape_the_prefix(self, key: str) -> None:
        # Reachable from a URL path in the image endpoint, so it must not depend
        # on the caller being careful.
        with pytest.raises(ValueError, match="escapes"):
            store(FakeS3()).put(key, b"data")

    def test_a_traversing_key_reads_as_absent_rather_than_raising(self) -> None:
        # `exists` is asked about keys that arrive from outside, and a malformed
        # one is a miss, not a crash.
        assert store(FakeS3()).exists("../elsewhere/p0000.png") is False


class TestFailuresAreNotMistakenForAbsence:
    def test_a_denied_permission_is_raised_not_swallowed(self) -> None:
        # The important one. Treating a permission failure as "not there" would
        # re-render every page of every submission and hide the misconfiguration
        # behind a service that merely got slow.
        denied = ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "HeadObject")
        with pytest.raises(ClientError):
            store(FakeS3(head_error=denied)).exists("abc/p0000.png")

    def test_a_wrong_region_is_raised_not_swallowed(self) -> None:
        moved = ClientError(
            {"Error": {"Code": "PermanentRedirect", "Message": "wrong endpoint"}}, "HeadObject"
        )
        with pytest.raises(ClientError):
            store(FakeS3(head_error=moved)).exists("abc/p0000.png")

    @pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
    def test_the_genuine_absence_codes_read_as_absent(self, code: str) -> None:
        missing = ClientError({"Error": {"Code": code, "Message": "gone"}}, "HeadObject")
        assert store(FakeS3(head_error=missing)).exists("abc/p0000.png") is False


class TestClear:
    def test_it_removes_only_what_is_under_the_prefix(self) -> None:
        client = FakeS3()
        client.objects["pages/abc/p0000.png"] = b"mine"
        client.objects["other/keep.txt"] = b"not mine"

        store(client).clear()
        assert list(client.objects) == ["other/keep.txt"]


class TestSelection:
    def test_no_bucket_means_the_local_store(self, monkeypatch, tmp_path) -> None:
        import grader.storage as storage_module

        monkeypatch.setattr(storage_module, "S3_BUCKET", "")
        monkeypatch.setenv("PAGE_STORE_ROOT", str(tmp_path))
        assert isinstance(storage_module.build_page_store(), PageStore)

    def test_a_bucket_means_the_s3_store(self, monkeypatch) -> None:
        import grader.storage as storage_module

        monkeypatch.setattr(storage_module, "S3_BUCKET", "some-bucket")
        built = storage_module.build_page_store()
        assert isinstance(built, S3PageStore)
        assert built.bucket == "some-bucket"
