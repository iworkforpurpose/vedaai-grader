"""In-memory submission store.

The brief permits in-memory storage and the deployment target is 100 test users,
so a database would add operational surface without buying anything. Two honest
consequences, stated rather than hidden:

  * state is lost on restart
  * a single process owns all state, so the API cannot be horizontally scaled

Both are fine here. Neither is fine in production, and the seam below is what
would be swapped rather than rewritten if that ever changed.

Progress events are kept per submission alongside the data. Subscribers get an
``asyncio.Event`` to wait on rather than polling, so the SSE endpoint stays
idle between stages instead of spinning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from vedaai_contracts import ProgressEvent, Stage, Submission


@dataclass
class _Entry:
    submission: Submission
    events: list[ProgressEvent] = field(default_factory=list)
    #: Pulsed whenever an event is appended, so waiters wake without polling.
    updated: asyncio.Event = field(default_factory=asyncio.Event)


class SubmissionStore:
    """Holds submissions and their progress streams for the process lifetime."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._content_cache: dict[str, str] = {}

    # -- submissions -------------------------------------------------------

    def put(self, submission: Submission) -> None:
        entry = self._entries.get(submission.submission_id)
        if entry is None:
            self._entries[submission.submission_id] = _Entry(submission=submission)
        else:
            entry.submission = submission

    def get(self, submission_id: str) -> Submission | None:
        entry = self._entries.get(submission_id)
        return entry.submission if entry else None

    def require(self, submission_id: str) -> Submission:
        submission = self.get(submission_id)
        if submission is None:
            raise KeyError(f"unknown submission: {submission_id}")
        return submission

    def ids(self) -> list[str]:
        return list(self._entries)

    # -- progress ----------------------------------------------------------

    def emit(self, submission_id: str, event: ProgressEvent) -> None:
        """Append a progress event and wake anyone streaming it."""
        entry = self._entries.get(submission_id)
        if entry is None:
            return
        entry.events.append(event)
        entry.updated.set()

    def events_since(self, submission_id: str, cursor: int) -> tuple[list[ProgressEvent], int]:
        """Return events after ``cursor``, plus the new cursor.

        Cursor-based rather than a queue so that a browser reconnecting
        mid-processing replays what it missed instead of resuming blind. A
        90-second job outliving a flaky connection is the normal case, not an
        edge case.
        """
        entry = self._entries.get(submission_id)
        if entry is None:
            return [], cursor
        pending = entry.events[cursor:]
        return pending, cursor + len(pending)

    async def wait_for_change(self, submission_id: str, timeout: float = 15.0) -> None:
        """Block until a new event arrives, or ``timeout`` elapses.

        The timeout exists so the SSE handler can send a keepalive comment;
        proxies drop connections that go quiet, and OCR stages can legitimately
        run for tens of seconds without producing an event.
        """
        entry = self._entries.get(submission_id)
        if entry is None:
            return
        entry.updated.clear()
        try:
            await asyncio.wait_for(entry.updated.wait(), timeout=timeout)
        except TimeoutError:
            return

    def is_finished(self, submission_id: str) -> bool:
        entry = self._entries.get(submission_id)
        if entry is None:
            return True
        return any(e.stage in {Stage.DONE, Stage.FAILED} for e in entry.events)

    # -- content-addressed cache -------------------------------------------

    def cached_submission_for(self, content_hash: str) -> str | None:
        """Find a prior submission that already processed this exact file.

        One question paper is shared across a whole class, so this turns N
        students into one question-paper render and OCR instead of N. That is
        what keeps a 1,000-page-per-month OCR free tier viable at all, and it
        also makes iterating during development far faster.
        """
        return self._content_cache.get(content_hash)

    def remember_content(self, content_hash: str, submission_id: str) -> None:
        self._content_cache.setdefault(content_hash, submission_id)

    def clear(self) -> None:
        """Drop everything. Used by tests to isolate cases."""
        self._entries.clear()
        self._content_cache.clear()


#: Process-wide store. Injected via a FastAPI dependency so tests can override it.
store = SubmissionStore()


def get_store() -> SubmissionStore:
    return store
