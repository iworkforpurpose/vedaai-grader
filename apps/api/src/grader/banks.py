"""Where a derived check bank is kept, so it is derived once and not once a day.

A bank belongs to the *question*, not to the script. Every student who sat the
paper is marked against the same one, and deriving it is the only part of marking
that still costs a provider call.

Held in memory it died with the process, which on Fargate means every deploy and
every restart. That is not a cache miss, it is the difference between a pilot that
fits inside a free tier and one that does not: a class of forty paid forty times,
and a day's worth of token budget went on deriving banks that were then discarded.

So the store is the same shape as the page store — local directory on a laptop,
object storage where a bucket is configured — and for the same reason. What is
kept is derived text about a *question*, not a student's writing: a bank contains
the paper's own checks and nothing from any script, which is why it can outlive a
submission's seven-day expiry without outliving anything a student wrote.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

#: Where banks live inside the bucket. A prefix of its own, never the page
#: prefix: the page endpoint serves whatever is under its own, and a bank is not
#: an image a browser should be able to ask for.
S3_PREFIX = os.getenv("S3_BANK_PREFIX", "banks/").strip()

#: Local fallback, for a laptop with no bucket.
LOCAL_ROOT = Path(
    os.getenv("CHECK_BANK_CACHE") or Path(__file__).resolve().parents[2] / ".banks"
)


def slot(key: str) -> str:
    """A stable filename for a bank key.

    Hashed because the key contains the question's full text — which is long,
    contains newlines, and would otherwise become a path.
    """
    return f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"


class BankStore(Protocol):
    def read(self, key: str) -> dict | None: ...
    def write(self, key: str, payload: dict) -> None: ...


class LocalBankStore:
    """Banks on disk. Survives a restart, not a redeploy of a container."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or LOCAL_ROOT

    def read(self, key: str) -> dict | None:
        try:
            return json.loads((self.root / slot(key)).read_text())
        except (OSError, ValueError):
            return None

    def write(self, key: str, payload: dict) -> None:
        # Never fatal. A read-only filesystem is a slower deployment, not a
        # broken one — the bank is simply derived again next time.
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / slot(key)).write_text(json.dumps(payload))
        except OSError:
            pass


class S3BankStore:
    """Banks in object storage, so they outlive the task that derived them."""

    def __init__(self, bucket: str, prefix: str | None = None, client=None) -> None:
        self.bucket = bucket
        self.prefix = (prefix if prefix is not None else S3_PREFIX).lstrip("/")
        self._client = client

    def _s3(self):
        if self._client is None:
            import boto3

            from .clients import aws_config

            self._client = boto3.client(
                "s3", region_name=os.getenv("AWS_REGION") or None, config=aws_config()
            )
        return self._client

    def read(self, key: str) -> dict | None:
        try:
            body = self._s3().get_object(
                Bucket=self.bucket, Key=f"{self.prefix}{slot(key)}"
            )["Body"].read()
            return json.loads(body)
        except Exception:  # noqa: BLE001
            # A miss and an outage are the same answer here: derive it again. The
            # alternative is failing a submission over a cache.
            return None

    def write(self, key: str, payload: dict) -> None:
        # Never fatal, for the same reason as the local store: a bank that fails
        # to persist is derived again, and that is a slower deployment rather
        # than a failed submission.
        with contextlib.suppress(Exception):
            self._s3().put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}{slot(key)}",
                Body=json.dumps(payload).encode(),
                ContentType="application/json",
            )


_store: BankStore | None = None


def store() -> BankStore:
    """The bank store for this deployment.

    Chosen the same way the page store is, from the same bucket variable, so a
    deployment that persists its pages persists its banks and there is no second
    thing to configure and forget.
    """
    global _store
    if _store is None:
        bucket = os.getenv("S3_PAGE_BUCKET", "").strip()
        _store = S3BankStore(bucket) if bucket else LocalBankStore()
    return _store


def reset() -> None:
    """Forget which store was chosen. For tests."""
    global _store
    _store = None
