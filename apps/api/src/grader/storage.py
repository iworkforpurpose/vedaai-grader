"""Page-image storage.

Rendered pages are written here and the in-memory bitmap freed immediately.
Twenty pages of 200 DPI A4 is roughly 220 MB of raw pixels, which is more than
a small worker has to spare, so nothing holds a whole document's bitmaps at
once.

This is a local-filesystem implementation behind a narrow interface. The
deployment target is Cloudflare R2 with presigned URLs, and swapping to it means
implementing ``put`` / ``url_for`` against the same two methods rather than
touching the pipeline.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: Where rendered pages land in local development. Scoped to this service rather
#: than the repo root so it cannot be mistaken for shared state, and overridable
#: because the deployed worker writes to a mounted volume, not the source tree.
DEFAULT_ROOT = Path(
    os.getenv("PAGE_STORE_ROOT") or Path(__file__).resolve().parents[2] / ".pagestore"
)


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
        # Reject traversal before touching the filesystem. Keys are generated
        # internally today, but this becomes reachable from a URL path in the
        # image endpoint, so it must not depend on the caller being careful.
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


store = PageStore()


def get_page_store() -> PageStore:
    return store
