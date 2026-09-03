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


#: The largest document this service accepts, mirrored from the renderer.
#:
#: Imported lazily rather than at module scope because `render` pulls in PyMuPDF
#: and OpenCV, and the upload path is reached before either is needed. The value
#: is asserted against `render.MAX_BYTES` in the tests, so the two cannot drift.
MAX_UPLOAD_BYTES = 40_000_000


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
    """A form the browser can POST one document to, size-bounded.

    A signed POST policy rather than a signed PUT, for one reason: a policy can
    carry a `content-length-range` and a signed PUT cannot. The PUT version signed
    only the bucket and the key, so the URL it handed out authorised an object of
    any size — up to S3's 5 GB single-object limit — against the operator's bucket,
    from an endpoint that had no rate limit either. The service's own cap of 40 MB
    was enforced afterwards, in the renderer, long after the bytes had landed and
    been paid for.

    The content type is still deliberately unsigned. Putting it in the signature
    means a browser that normalises the header, or guesses a different type for the
    same file, gets a signature mismatch it cannot debug — and nothing downstream
    trusts the declared type anyway, because the document is identified by
    inspecting its bytes.
    """
    key = new_key(kind, filename)
    s3 = client or _client()
    signed = s3.generate_presigned_post(
        Bucket=bucket(),
        Key=object_key(key),
        # One byte, so an empty object cannot be uploaded and then reported as a
        # corrupt document from four stages away.
        Conditions=[["content-length-range", 1, MAX_UPLOAD_BYTES]],
        ExpiresIn=URL_TTL_SECONDS,
    )
    return UploadSlot(key=key, url=signed["url"], fields=dict(signed["fields"]))


def read(key: str, *, client=None) -> bytes:
    """The bytes the browser uploaded, refusing an object too large to accept.

    The size is checked before the body is fetched. The signed policy already
    bounds what can be written, but this path also serves objects that were
    written before a policy existed, and `get_object(...).read()` materialises
    whatever it finds in the worker's memory in one allocation. A cap that runs
    after the download is a cap on the wrong thing.
    """
    s3 = client or _client()
    resolved = object_key(key)
    head = s3.head_object(Bucket=bucket(), Key=resolved)
    size = int(head.get("ContentLength", 0))
    if size > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"{key!r} is {size} bytes, over the {MAX_UPLOAD_BYTES}-byte limit"
        )
    response = s3.get_object(Bucket=bucket(), Key=resolved)
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
