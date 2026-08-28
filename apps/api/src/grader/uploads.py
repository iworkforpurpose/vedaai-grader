"""Where an uploaded document lands before anything reads it.

The browser sends the file straight to object storage and then tells this service
which keys to process. The service never carries the bytes, which is the point:
every host puts a limit on request bodies — API Gateway 10 MB, a serverless
function less — and routing a 40 MB scan through one means choosing which limit to
live with. Uploading past the host removes the question instead of moving it.

This was anticipated rather than discovered. The original design noted that
"presigned object-storage upload becomes an optimization rather than a
requirement"; the requirement arrived with the first host that had a body cap
smaller than the documents.

There is a direct path too, and it is not dead weight. A developer running this
locally has no bucket, so presigning has nothing to sign — the endpoint says so and
the client posts the file itself. One code path per environment that actually
exists.
"""

from __future__ import annotations

import contextlib
import os
import re
import uuid
from dataclasses import dataclass

#: Where uploads live inside the bucket. A prefix rather than a bucket of its own,
#: so one lifecycle rule expires uploads and rendered pages together — both are
#: student work and neither should outlive the review it was uploaded for.
UPLOAD_PREFIX = os.getenv("S3_UPLOAD_PREFIX", "uploads/").strip()

#: How long the browser has to start the upload. Long enough for someone to pick a
#: file on a slow phone, short enough that a leaked URL is not a standing grant.
URL_TTL_SECONDS = int(os.getenv("UPLOAD_URL_TTL", "900"))

#: Keys this service will agree to read.
#:
#: Narrow on purpose. The key arrives from the client, and reading an arbitrary key
#: would let a caller name any object in the bucket — including another student's
#: rendered pages. Only the shape this module generates is accepted.
_KEY = re.compile(r"^[0-9a-f]{32}/(question_paper|answer_sheet)(\.[A-Za-z0-9]{1,8})?$")


class UploadRejected(Exception):
    """A key the client asked for is not one this service issued."""


@dataclass(frozen=True)
class UploadSlot:
    """One presigned destination, and the key to quote back afterwards."""

    key: str
    url: str
    fields: dict[str, str]


def bucket() -> str:
    return os.getenv("S3_PAGE_BUCKET", "").strip()


def available() -> bool:
    """Whether uploads can bypass this service. False when there is no bucket."""
    return bool(bucket())


def new_key(kind: str, filename: str) -> str:
    """A key for one document of one upload attempt.

    A fresh random directory per attempt rather than a content hash: the hash is not
    known until the bytes arrive, and two students uploading the same paper must not
    be able to overwrite each other's in-flight upload.
    """
    suffix = ""
    if "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
        if ext.isalnum() and len(ext) <= 8:
            suffix = f".{ext}"
    return f"{uuid.uuid4().hex}/{kind}{suffix}"


def check(key: str) -> str:
    """Return the key if this service could have issued it, else refuse."""
    if not _KEY.match(key):
        raise UploadRejected(f"{key!r} is not an upload key issued by this service")
    return key


def object_key(key: str) -> str:
    return f"{UPLOAD_PREFIX.lstrip('/')}{check(key)}"


def presign(kind: str, filename: str, *, client=None) -> UploadSlot:
    """A URL the browser can PUT one document to.

    Deliberately signs only the bucket and key. Including a content type would put
    it in the signature, and then a browser that normalises the header — or guesses
    a different type for the same file — gets a signature mismatch it cannot debug.
    Nothing downstream trusts the declared type anyway: the document is identified
    by inspecting its bytes.
    """
    key = new_key(kind, filename)
    s3 = client or _client()
    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket(), "Key": object_key(key)},
        ExpiresIn=URL_TTL_SECONDS,
    )
    return UploadSlot(key=key, url=url, fields={})


def read(key: str, *, client=None) -> bytes:
    """The bytes the browser uploaded."""
    s3 = client or _client()
    response = s3.get_object(Bucket=bucket(), Key=object_key(key))
    return response["Body"].read()


def discard(key: str, *, client=None) -> None:
    """Remove an upload once it has been rendered.

    Not strictly required — the lifecycle rule would get to it — but the rendered
    pages are the durable artefact and the original scan is the larger object. There
    is no reason to keep a copy of a student's script for a week after the pages
    exist.

    Failure is suppressed on purpose: this is tidying after work that has already
    succeeded, and an undeleted object is a smaller problem than a submission that
    reports failure because its cleanup did.
    """
    with contextlib.suppress(Exception):
        (client or _client()).delete_object(Bucket=bucket(), Key=object_key(key))


def _client():
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise RuntimeError(
            "S3_PAGE_BUCKET is set but boto3 is not installed; install the 'aws' extra"
        ) from exc
    return boto3.client("s3", region_name=os.getenv("AWS_REGION") or None)
