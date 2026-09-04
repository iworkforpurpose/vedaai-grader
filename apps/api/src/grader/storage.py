"""Page-image storage.

Rendered pages are written here and the in-memory bitmap freed immediately.
Twenty pages of 200 DPI A4 is roughly 220 MB of raw pixels, which is more than
a small worker has to spare, so nothing holds a whole document's bitmaps at
once.

Two implementations behind one narrow interface: the local filesystem for
development, and S3 for the deployed service. Which one runs is decided by
whether a bucket is configured, and the pipeline never learns the difference.

The interface is four methods — ``exists``, ``put``, ``read``, ``clear`` — and
that is deliberately all. Presigned URLs were considered and left out: the
browser already fetches page images through this service, which is where the
CORS allowance and the submission's own lifetime live, and adding a second path
to the same bytes would mean two places for an access rule to be wrong.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

#: Where rendered pages land in local development. Scoped to this service rather
#: than the repo root so it cannot be mistaken for shared state, and overridable
#: because the deployed worker writes to a mounted volume, not the source tree.
DEFAULT_ROOT = Path(
    os.getenv("PAGE_STORE_ROOT") or Path(__file__).resolve().parents[2] / ".pagestore"
)


#: The only key shape either store accepts, and exactly what `key_for` produces:
#: sixteen hex characters of the content hash, then a four-digit page number.
#:
#: Both stores validate against this rather than against traversal, because the
#: key arrives from a URL path and "does not escape" is a weaker claim than "is
#: one of ours". A bucket shared with `uploads/` and `submissions/` makes the
#: difference between those two claims a student's scanned script.
_PAGE_KEY = re.compile(r"^[0-9a-f]{1,64}/p\d{4}\.png$")


def require_page_key(key: str) -> str:
    """Return the key if this store could have generated it, else refuse."""
    if not _PAGE_KEY.match(key):
        raise ValueError(f"not a page key issued by this store: {key!r}")
    return key


class PageStore:
    """Content-addressed store for rendered page images.

    Keys embed the source file's content hash, so re-uploading the same
    question paper reuses the pages already on disk. One paper shared across a
    class is rendered once rather than once per student.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        # Coerced rather than required as a Path: the root also arrives from an
        # environment variable, where it is a string by definition.
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(content_hash: str, page_index: int) -> str:
        return f"{content_hash[:16]}/p{page_index:04d}.png"

    def path_for(self, key: str) -> Path:
        # Reject anything that is not a key this store generates, before touching
        # the filesystem. The key reaches here from a URL path in the image
        # endpoint, so it must not depend on the caller being careful.
        require_page_key(key)
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root.resolve()):
            raise ValueError(f"key escapes the page store: {key!r}")
        return candidate

    def exists(self, key: str) -> bool:
        try:
            return self.path_for(key).is_file()
        except ValueError:
            return False

    def put(self, key: str, data: bytes) -> str:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def read(self, key: str) -> bytes:
        return self.path_for(key).read_bytes()

    def clear(self) -> None:
        """Drop everything. Used by tests."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)


#: Bucket for rendered pages in the deployed service. Empty means local disk,
#: which is what development wants.
S3_BUCKET = os.getenv("S3_PAGE_BUCKET", "").strip()

#: Prefix inside the bucket, so page images can share a bucket with other objects
#: and still be expired by a single lifecycle rule.
S3_PREFIX = os.getenv("S3_PAGE_PREFIX", "pages/").strip()


class S3PageStore:
    """The same content-addressed store, backed by S3.

    Content addressing earns more here than it does on disk. A question paper
    shared across a class is rendered once and every later submission reuses those
    objects, so the bill and the latency both scale with distinct papers rather
    than with students.

    These objects are student answer scripts. They are fully regenerable from the
    upload, so the bucket should carry a lifecycle rule that expires them — there
    is no reason for a scanned script to outlive the review it was uploaded for.
    """

    def __init__(
        self,
        bucket: str | None = None,
        *,
        prefix: str | None = None,
        client: object | None = None,
    ) -> None:
        self.bucket = bucket or S3_BUCKET
        self.prefix = (prefix if prefix is not None else S3_PREFIX).lstrip("/")
        self._client = client

    @staticmethod
    def key_for(content_hash: str, page_index: int) -> str:
        # Identical to the local store's scheme on purpose: the same upload
        # produces the same key either side, so switching backends does not
        # invalidate anything already rendered.
        return PageStore.key_for(content_hash, page_index)

    def _object_key(self, key: str) -> str:
        # The local store gets its safety from the filesystem: `path_for` resolves
        # and checks containment, so a prefix really is a directory. Here the
        # prefix is a string, one bucket holds the students' original uploads and
        # the spilled submission payloads as well as the pages, and string
        # concatenation confines nothing. Rejecting `..` was not the same guard,
        # and `uploads/{id}/answer_sheet.pdf` went straight through it.
        require_page_key(key)
        return f"{self.prefix}{key}"

    def _s3(self):
        if self._client is None:
            try:
                import boto3
            except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
                raise RuntimeError(
                    "S3_PAGE_BUCKET is set but boto3 is not installed; "
                    "install the 'aws' extra"
                ) from exc
            from .clients import aws_config

            self._client = boto3.client(
                "s3", region_name=os.getenv("AWS_REGION") or None, config=aws_config()
            )
        return self._client

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3().head_object(Bucket=self.bucket, Key=self._object_key(key))
            return True
        except ValueError:
            return False
        except ClientError as error:
            code = str((error.response.get("Error") or {}).get("Code") or "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            # Anything else — a denied permission, a wrong region — is not
            # "absent". Reporting it as absent would silently re-render every page
            # on every submission and hide a misconfiguration behind a slow but
            # working service.
            raise

    def put(self, key: str, data: bytes) -> str:
        self._s3().put_object(
            Bucket=self.bucket,
            Key=self._object_key(key),
            Body=data,
            ContentType="image/png",
        )
        return key

    def read(self, key: str) -> bytes:
        response = self._s3().get_object(Bucket=self.bucket, Key=self._object_key(key))
        return response["Body"].read()

    def clear(self) -> None:
        """Drop every page under the prefix. Used by tests."""
        client = self._s3()
        token: str | None = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": self.prefix}
            if token:
                kwargs["ContinuationToken"] = token
            listing = client.list_objects_v2(**kwargs)
            objects = [{"Key": item["Key"]} for item in listing.get("Contents") or []]
            if objects:
                client.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
            if not listing.get("IsTruncated"):
                return
            token = listing.get("NextContinuationToken")


AnyPageStore = PageStore | S3PageStore


def build_page_store() -> AnyPageStore:
    """The store this process should use, from configuration."""
    return S3PageStore() if S3_BUCKET else PageStore()


store: AnyPageStore = build_page_store()


def get_page_store() -> AnyPageStore:
    return store
