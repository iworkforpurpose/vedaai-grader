"""Where a submission survives a restart.

Until now submissions lived only in the process. The brief permitted it, and at
this scale it cost nothing — until the deployment became something people were
asked to test, at which point every push discarded whatever they were half way
through reviewing. That is not a scale problem, it is an unfinished product.

**Why a table and not just object storage.** The lookups this service makes are
by id, which object storage answers perfectly well. The table earns its place on
three other counts: an expiry the store does not have to police, a status
readable without decompressing anything, and a conditional write. The last is the
one that matters — reassignment is read, mutate, write, and with a plain overwrite
two teachers correcting the same script silently lose one of the corrections. It
cannot happen today, because there is one task and one worker, but "cannot happen
today" is exactly the assumption that a second task quietly invalidates.

**Why the payload is compressed, and why it can spill.** A measured two-page
submission serializes to 140 KiB, about 70 KiB a page, nearly all of it line
boxes and ink regions. The item limit is 400 KB, so a raw item breaks at six
pages while `render.MAX_PAGES` allows sixty. Compression buys 4.3x — enough to
about twenty pages — so the payload is gzipped, and past the limit it goes to
object storage with the item keeping the pointer. Both paths are exercised by
tests; the fallback is not theoretical, a sixty-page script reaches it.

Progress events stay in memory on purpose. They are a live stream for a browser
watching a run in progress, and a run does not survive the restart either — what
has to survive is the result. After a restart the page falls back to polling the
submission, which is the path it already uses.
"""

from __future__ import annotations

import gzip
import os
import time
from dataclasses import dataclass
from typing import Protocol

from vedaai_contracts import Submission

#: The largest compressed payload to put in an item.
#:
#: The hard limit is 400 KB for the whole item, including attribute names and the
#: key. The margin covers those and the status, version and expiry attributes,
#: and it is a margin rather than an exact accounting because being wrong here
#: means a rejected write at the end of a two-minute pipeline run.
MAX_ITEM_BODY = 350 * 1024

#: Where a payload too large for an item goes.
BLOB_PREFIX = "submissions/"

#: How long a submission is kept.
#:
#: Seven days, matching the lifecycle rule that expires rendered pages. A record
#: outliving its page images would open to a review with every page blank, which
#: is worse than opening to "no such submission" — it looks like the pipeline
#: lost the work rather than like the link expiring.
TTL_SECONDS = 7 * 24 * 3600


class ConcurrentUpdate(Exception):
    """Someone else changed this submission since it was read.

    Raised rather than resolved because this store cannot merge two edits: it
    holds whole submissions, so "apply both" is not an operation it has. The
    caller's own state is stale, and the honest response is to say so and let the
    page reload — which the reassignment UI already does, since it reverts an
    optimistic move when the request fails.
    """


@dataclass(frozen=True)
class Stored:
    """A loaded submission and the version it was loaded at."""

    submission: Submission
    version: int


class Persistence(Protocol):
    """The seam. Two implementations: one that does nothing, one that persists."""

    def available(self) -> bool: ...

    def load(self, submission_id: str) -> Stored | None: ...

    def save(self, submission: Submission, *, expect_version: int | None) -> int: ...

    def content_lookup(self, content_hash: str) -> str | None: ...

    def content_remember(self, content_hash: str, submission_id: str) -> None: ...


class NoPersistence:
    """The local case: nothing is written, and nothing pretends it was.

    A developer running this has no table, and standing one up to change a CSS
    file would be absurd. Returning "not found" rather than raising keeps the
    store's own code free of environment checks — a miss from a store with no
    backing looks exactly like a miss from one with backing.
    """

    def available(self) -> bool:
        return False

    def load(self, submission_id: str) -> Stored | None:
        return None

    def save(self, submission: Submission, *, expect_version: int | None) -> int:
        return 0

    def content_lookup(self, content_hash: str) -> str | None:
        return None

    def content_remember(self, content_hash: str, submission_id: str) -> None:
        return None


def table_name() -> str:
    return os.getenv("SUBMISSIONS_TABLE", "").strip()


def bucket() -> str:
    return os.getenv("S3_PAGE_BUCKET", "").strip()


class DynamoPersistence:
    """Submissions in a table, with the payload compressed and spilling if large."""

    def __init__(self, *, table=None, s3=None) -> None:
        self._table = table
        self._s3 = s3

    # -- plumbing ----------------------------------------------------------

    def available(self) -> bool:
        return bool(table_name())

    @property
    def table(self):
        if self._table is None:
            import boto3

            self._table = boto3.resource(
                "dynamodb", region_name=os.getenv("AWS_REGION") or None
            ).Table(table_name())
        return self._table

    @property
    def s3(self):
        if self._s3 is None:
            import boto3

            from .clients import aws_config

            self._s3 = boto3.client(
                "s3", region_name=os.getenv("AWS_REGION") or None, config=aws_config()
            )
        return self._s3

    # -- submissions -------------------------------------------------------

    def load(self, submission_id: str) -> Stored | None:
        item = self.table.get_item(
            Key={"pk": _key(submission_id)}, ConsistentRead=True
        ).get("Item")
        if not item:
            return None

        body = item.get("body")
        if body is None:
            key = item.get("body_key")
            if not key:
                return None
            body = self.s3.get_object(Bucket=bucket(), Key=str(key))["Body"].read()

        raw = gzip.decompress(bytes(body))
        return Stored(
            submission=Submission.model_validate_json(raw),
            version=int(item.get("version", 0)),
        )

    def save(self, submission: Submission, *, expect_version: int | None) -> int:
        body = gzip.compress(submission.model_dump_json().encode(), 6)
        version = (expect_version or 0) + 1

        item: dict[str, object] = {
            "pk": _key(submission.submission_id),
            "version": version,
            # Plain text alongside the compressed payload, so "what happened to
            # this submission?" is answerable from the console without a script.
            "status": submission.status.value,
            "expires_at": int(time.time()) + TTL_SECONDS,
        }

        if len(body) <= MAX_ITEM_BODY:
            item["body"] = body
        else:
            # Too large for an item. The pointer goes in the item so a reader
            # never has to guess which of the two places holds the payload.
            key = f"{BLOB_PREFIX}{submission.submission_id}.json.gz"
            self.s3.put_object(Bucket=bucket(), Key=key, Body=body)
            item["body_key"] = key

        # First write: the item must not already exist. Later writes: the stored
        # version must be the one this caller read. Either way a lost update is a
        # rejected write rather than a silent overwrite.
        if expect_version is None:
            condition = "attribute_not_exists(pk)"
            values = None
        else:
            condition = "version = :expected"
            values = {":expected": expect_version}

        try:
            if values is None:
                self.table.put_item(Item=item, ConditionExpression=condition)
            else:
                self.table.put_item(
                    Item=item,
                    ConditionExpression=condition,
                    ExpressionAttributeValues=values,
                )
        except Exception as exc:  # noqa: BLE001 - narrowed immediately below
            if _is_condition_failure(exc):
                raise ConcurrentUpdate(
                    f"submission {submission.submission_id} changed since it was read"
                ) from exc
            raise
        return version

    # -- the content-addressed cache ---------------------------------------

    def content_lookup(self, content_hash: str) -> str | None:
        item = self.table.get_item(Key={"pk": _content_key(content_hash)}).get("Item")
        if not item:
            return None
        found = item.get("submission_id")
        return str(found) if found else None

    def content_remember(self, content_hash: str, submission_id: str) -> None:
        # First writer wins, matching the in-memory `setdefault`. Two students
        # uploading the same paper at once should not fight over which render the
        # rest of the class reuses.
        try:
            self.table.put_item(
                Item={
                    "pk": _content_key(content_hash),
                    "submission_id": submission_id,
                    "expires_at": int(time.time()) + TTL_SECONDS,
                },
                ConditionExpression="attribute_not_exists(pk)",
            )
        except Exception as exc:  # noqa: BLE001 - narrowed immediately below
            if not _is_condition_failure(exc):
                raise


def _key(submission_id: str) -> str:
    return f"sub#{submission_id}"


def _content_key(content_hash: str) -> str:
    return f"content#{content_hash}"


def _is_condition_failure(exc: Exception) -> bool:
    """Whether a write was rejected by its condition rather than by a fault.

    Matched on the error code rather than the exception class because the class
    lives on the client object at runtime — ``client.exceptions`` — and reaching
    for it here would mean constructing a client to interpret an error, which is
    exactly the wrong direction. A stubbed table in a test can raise anything
    carrying the same code.
    """
    response = getattr(exc, "response", None) or {}
    code = (response.get("Error") or {}).get("Code")
    return code == "ConditionalCheckFailedException"


def default_persistence() -> Persistence:
    """Persist when there is somewhere to persist to, otherwise do not."""
    dynamo = DynamoPersistence()
    return dynamo if dynamo.available() else NoPersistence()
