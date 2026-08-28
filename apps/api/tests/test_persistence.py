"""Surviving a restart, and not losing a correction to a race.

Tested against a stub table rather than a real one. What is worth testing here is
this module's own decisions — when to compress, when to spill to object storage,
what to do when a conditional write is rejected — and none of those need a network
round trip to exercise. What a stub cannot check is whether the item shape is one
the service accepts, so the deployed path is verified separately by making a real
submission and reading it back after a restart.
"""

from __future__ import annotations

import gzip
import json

import pytest
from vedaai_contracts import (
    BBox,
    DocumentKind,
    Line,
    LineIndex,
    OcrEngine,
    Submission,
    SubmissionStatus,
)

from grader import persistence
from grader.persistence import ConcurrentUpdate, DynamoPersistence, NoPersistence
from grader.store import SubmissionStore


class FakeTable:
    """The two operations this module uses, and the one failure it cares about."""

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.puts = 0

    def get_item(self, Key, ConsistentRead=False):  # noqa: N803 - boto3's casing
        item = self.items.get(Key["pk"])
        return {"Item": dict(item)} if item else {}

    def put_item(self, Item, ConditionExpression=None, ExpressionAttributeValues=None):  # noqa: N803
        self.puts += 1
        existing = self.items.get(Item["pk"])
        if ConditionExpression == "attribute_not_exists(pk)" and existing is not None:
            raise _rejected()
        if ConditionExpression == "version = :expected":
            expected = (ExpressionAttributeValues or {})[":expected"]
            if existing is None or existing.get("version") != expected:
                raise _rejected()
        self.items[Item["pk"]] = dict(Item)
        return {}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):  # noqa: N803
        class Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": Body(self.objects[Key])}


def _rejected() -> Exception:
    exc = Exception("the condition was not met")
    exc.response = {"Error": {"Code": "ConditionalCheckFailedException"}}  # type: ignore[attr-defined]
    return exc


def _submission(sid: str = "abc123", *, lines: int = 4) -> Submission:
    index = LineIndex(
        kind=DocumentKind.ANSWER_SHEET,
        lines=[
            Line(
                line_id=f"as:{i:04d}",
                kind=DocumentKind.ANSWER_SHEET,
                page=0,
                box=BBox(x0=0.1, y0=0.1 + i * 0.001, x1=0.9, y1=0.11 + i * 0.001),
                text=f"line {i} of an answer about invasive species and their range",
                confidence=0.9,
                engine=OcrEngine.PADDLE_OCR_VL,
            )
            for i in range(lines)
        ],
        engine=OcrEngine.PADDLE_OCR_VL,
    )
    return Submission(
        submission_id=sid,
        status=SubmissionStatus.COMPLETE,
        answer_sheet_lines=index,
    )


@pytest.fixture
def store(monkeypatch) -> tuple[SubmissionStore, FakeTable, FakeS3]:
    monkeypatch.setenv("SUBMISSIONS_TABLE", "a-table")
    monkeypatch.setenv("S3_PAGE_BUCKET", "a-bucket")
    table, s3 = FakeTable(), FakeS3()
    return SubmissionStore(DynamoPersistence(table=table, s3=s3)), table, s3


class TestRoundTrip:
    def test_a_submission_read_back_is_the_one_that_was_written(self, store) -> None:
        st, table, _ = store
        original = _submission()
        st.put(original)

        # A fresh store: the process restarted and its memory is empty. This is the
        # whole point of the change, so it is the first thing asserted.
        after_restart = SubmissionStore(DynamoPersistence(table=table))
        loaded = after_restart.get("abc123")

        assert loaded is not None
        assert loaded.model_dump_json() == original.model_dump_json()

    def test_the_payload_is_compressed(self, store) -> None:
        st, table, _ = store
        st.put(_submission(lines=200))

        item = table.items["sub#abc123"]
        stored = bytes(item["body"])
        plain = json.loads(gzip.decompress(stored))
        assert plain["submission_id"] == "abc123"
        # Not merely smaller — the reason for compressing is the item limit, and a
        # marginal saving would not have been worth the indirection.
        assert len(stored) * 3 < len(json.dumps(plain))

    def test_the_status_is_readable_without_decompressing(self, store) -> None:
        # So that "what happened to this submission?" is answerable from the
        # console rather than needing a script.
        st, table, _ = store
        st.put(_submission())
        assert table.items["sub#abc123"]["status"] == "complete"

    def test_an_unknown_submission_is_absent_rather_than_an_error(self, store) -> None:
        st, _, _ = store
        assert st.get("neverexisted") is None


class TestSpillingToObjectStorage:
    """The path a long script takes.

    Measured: about 16 KiB compressed per page, so roughly twenty pages fit an
    item and `render.MAX_PAGES` allows sixty. This is not a theoretical branch.
    """

    def test_a_large_submission_goes_to_the_bucket_with_a_pointer_in_the_item(
        self, store, monkeypatch
    ) -> None:
        st, table, s3 = store
        monkeypatch.setattr(persistence, "MAX_ITEM_BODY", 1024)

        st.put(_submission(lines=400))

        item = table.items["sub#abc123"]
        assert "body" not in item, "the payload should not be in the item"
        assert item["body_key"] == "submissions/abc123.json.gz"
        assert item["body_key"] in s3.objects

    def test_it_reads_back_through_the_pointer(self, store, monkeypatch) -> None:
        st, table, s3 = store
        monkeypatch.setattr(persistence, "MAX_ITEM_BODY", 1024)
        original = _submission(lines=400)
        st.put(original)

        reloaded = SubmissionStore(DynamoPersistence(table=table, s3=s3)).get("abc123")
        assert reloaded is not None
        assert reloaded.model_dump_json() == original.model_dump_json()


class TestLostUpdates:
    def test_a_stale_writer_is_refused(self, store) -> None:
        """Two teachers correcting the same script.

        Both read version 1, both move a block, and a plain overwrite would keep
        whichever wrote second while reporting success to both. The refusal is what
        makes the second one reload instead of believing a correction landed.
        """
        st, table, s3 = store
        st.put(_submission())

        stale = SubmissionStore(DynamoPersistence(table=table, s3=s3))
        stale.get("abc123")  # reads version 1

        fresh = _submission()
        fresh.warnings.append("someone else got here first")
        st.put(fresh)  # now version 2

        with pytest.raises(ConcurrentUpdate):
            stale.put(_submission())

    def test_the_winner_is_untouched_by_the_refusal(self, store) -> None:
        st, table, s3 = store
        st.put(_submission())
        stale = SubmissionStore(DynamoPersistence(table=table, s3=s3))
        stale.get("abc123")

        winner = _submission()
        winner.warnings.append("kept")
        st.put(winner)
        with pytest.raises(ConcurrentUpdate):
            stale.put(_submission())

        reloaded = SubmissionStore(DynamoPersistence(table=table, s3=s3)).get("abc123")
        assert reloaded is not None
        assert reloaded.warnings == ["kept"]

    def test_memory_is_not_updated_when_the_write_is_refused(self, store) -> None:
        """The reason the table is written before memory.

        Updating memory first would leave this process serving a submission no
        other process and no restart will ever see, which surfaces much later as a
        submission that vanished with nothing near the cause in the logs.
        """
        st, table, s3 = store
        st.put(_submission())
        stale = SubmissionStore(DynamoPersistence(table=table, s3=s3))
        stale.get("abc123")
        st.put(_submission())

        rejected = _submission()
        rejected.warnings.append("should not be visible")
        with pytest.raises(ConcurrentUpdate):
            stale.put(rejected)

        assert stale.get("abc123").warnings == []

    def test_repeated_writes_from_one_owner_are_fine(self, store) -> None:
        # The normal case: ingest writes several times as stages complete.
        st, table, _ = store
        for _ in range(4):
            st.put(_submission())
        assert table.items["sub#abc123"]["version"] == 4


class TestContentCache:
    def test_a_shared_question_paper_is_found_after_a_restart(self, store) -> None:
        """The saving this cache exists for has to survive a deploy too.

        One paper is shared across a class, so a cache that empties on restart
        means re-rendering and re-transcribing the same paper for every student who
        uploads after it.
        """
        st, table, _ = store
        st.remember_content("deadbeef", "abc123")

        after_restart = SubmissionStore(DynamoPersistence(table=table))
        assert after_restart.cached_submission_for("deadbeef") == "abc123"

    def test_the_first_writer_wins(self, store) -> None:
        st, table, _ = store
        st.remember_content("deadbeef", "first")
        other = SubmissionStore(DynamoPersistence(table=table))
        other.remember_content("deadbeef", "second")
        assert other.cached_submission_for("deadbeef") == "first"

    def test_an_unseen_hash_is_a_miss(self, store) -> None:
        st, _, _ = store
        assert st.cached_submission_for("nothinglikethis") is None


class TestWithoutATable:
    """Local development, which must not require a table to exist."""

    def test_nothing_is_persisted_and_nothing_pretends_it_was(self) -> None:
        st = SubmissionStore(NoPersistence())
        assert st.durable is False
        st.put(_submission())
        assert st.get("abc123") is not None  # served from memory, as before

        assert SubmissionStore(NoPersistence()).get("abc123") is None

    def test_the_default_is_chosen_by_whether_a_table_is_configured(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("SUBMISSIONS_TABLE", raising=False)
        assert persistence.default_persistence().available() is False

        monkeypatch.setenv("SUBMISSIONS_TABLE", "a-table")
        assert persistence.default_persistence().available() is True


class TestExpiry:
    def test_every_item_carries_one(self, store) -> None:
        """A record outliving its page images opens to a review with blank pages.

        That is worse than "no such submission": it looks like the pipeline lost
        the work rather than like the link expiring. So the expiry matches the
        lifecycle rule on the rendered pages.
        """
        st, table, _ = store
        st.put(_submission())
        st.remember_content("deadbeef", "abc123")

        for key in ("sub#abc123", "content#deadbeef"):
            assert table.items[key]["expires_at"] > 0
        assert persistence.TTL_SECONDS == 7 * 24 * 3600
