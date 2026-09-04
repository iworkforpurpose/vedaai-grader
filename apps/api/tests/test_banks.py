"""Where a derived check bank is kept.

A bank belongs to the question, not the script, and deriving it is the only part
of marking that still costs a provider call. Held in memory it died with the
process — which on Fargate means every deploy — so a class of forty paid forty
times and a day of free-tier budget went on banks that were then discarded.
"""

from __future__ import annotations

import json

import pytest

from grader import banks


@pytest.fixture(autouse=True)
def _fresh():
    banks.reset()
    yield
    banks.reset()


BANK = {
    "qid": "1",
    "traps": ["feathers are insulators"],
    "needs_material": False,
    "checks": [
        {"ask": "Does the answer give 15 m/s?", "claim": "The speed is 15 m/s.",
         "marks": 1.0, "needs_material": False}
    ],
}


class TestOnDisk:
    def test_a_bank_survives_the_process_that_derived_it(self, tmp_path) -> None:
        store = banks.LocalBankStore(root=tmp_path)
        store.write("q1-key", BANK)

        assert banks.LocalBankStore(root=tmp_path).read("q1-key") == BANK

    def test_a_bank_nobody_derived_is_a_miss_not_an_error(self, tmp_path) -> None:
        assert banks.LocalBankStore(root=tmp_path).read("never-seen") is None

    def test_a_read_only_filesystem_is_slower_not_broken(self, tmp_path) -> None:
        """Failing to cache must never fail a submission."""
        store = banks.LocalBankStore(root=tmp_path / "nested" / "deep")
        store.root.mkdir(parents=True)
        store.root.chmod(0o500)
        try:
            store.write("q1-key", BANK)  # must not raise
        finally:
            store.root.chmod(0o700)

    def test_the_key_becomes_a_filename(self) -> None:
        """The key holds the question's full text — newlines, punctuation, length."""
        messy = "1\x00What is 2 + 2?\nShow your working.\x004.0\x00\x00"

        name = banks.slot(messy)

        assert name.endswith(".json")
        assert "/" not in name and "\n" not in name
        assert banks.slot(messy) == name, "the same question must find its own bank"


class FakeS3:
    def __init__(self, fail: bool = False):
        self.objects: dict[str, bytes] = {}
        self.fail = fail

    def put_object(self, *, Bucket, Key, Body, **kw):  # noqa: N803
        if self.fail:
            raise RuntimeError("AccessDenied")
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise RuntimeError("NoSuchKey")
        return {"Body": _Body(self.objects[Key])}


class _Body:
    def __init__(self, data: bytes):
        self.data = data

    def read(self) -> bytes:
        return self.data


class TestInObjectStorage:
    def test_a_bank_outlives_the_task_that_derived_it(self) -> None:
        s3 = FakeS3()
        banks.S3BankStore("b", client=s3).write("q1-key", BANK)

        assert banks.S3BankStore("b", client=s3).read("q1-key") == BANK

    def test_banks_are_not_under_the_page_prefix(self) -> None:
        """The page endpoint serves whatever is under its own prefix, and a bank
        is not an image a browser should be able to ask for."""
        s3 = FakeS3()
        banks.S3BankStore("b", client=s3).write("q1-key", BANK)

        [key] = s3.objects
        assert key.startswith("banks/")
        assert not key.startswith("pages/")

    def test_an_outage_is_a_miss_rather_than_a_failed_submission(self) -> None:
        assert banks.S3BankStore("b", client=FakeS3()).read("absent") is None

    def test_a_refused_write_does_not_reach_the_caller(self) -> None:
        banks.S3BankStore("b", client=FakeS3(fail=True)).write("q1-key", BANK)

    def test_the_payload_is_json(self) -> None:
        s3 = FakeS3()
        banks.S3BankStore("b", client=s3).write("q1-key", BANK)

        assert json.loads(next(iter(s3.objects.values()))) == BANK


class TestChoosingAStore:
    def test_a_bucket_means_object_storage(self, monkeypatch) -> None:
        """Chosen from the same variable the pages use, so a deployment that
        persists its pages persists its banks with nothing else to configure."""
        monkeypatch.setenv("S3_PAGE_BUCKET", "a-bucket")

        assert isinstance(banks.store(), banks.S3BankStore)

    def test_no_bucket_means_a_laptop(self, monkeypatch) -> None:
        monkeypatch.delenv("S3_PAGE_BUCKET", raising=False)

        assert isinstance(banks.store(), banks.LocalBankStore)
