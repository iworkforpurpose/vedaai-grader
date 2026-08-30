"""Submission store: memory in front, a table behind.

This was memory only, on the reasoning that the brief permitted it and a database
would add operational surface without buying anything. That held right up until
the deployment became something people were asked to test, at which point every
push threw away whatever a tester was half way through reviewing. Losing a
reviewer's work is not an acceptable cost of a deploy.

So memory is now a cache in front of `persistence`, not the store itself. A read
that misses loads from the table and repopulates; a write goes through to the
table before memory is updated, so memory never claims something durable that is
not. With no table configured — the local case — persistence does nothing and this
behaves exactly as it did.

One consequence remains and is worth stating: progress events are still per
process. They are a live stream for a browser watching a run, and the run does not
survive a restart either. What has to survive is the result, and after a restart
the page falls back to polling the submission — the path it already uses.

Progress events are kept per submission alongside the data. Subscribers get an
``asyncio.Event`` to wait on rather than polling, so the SSE endpoint stays
idle between stages instead of spinning.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from vedaai_contracts import ProgressEvent, Stage, Submission, SubmissionStatus

from .persistence import Persistence, default_persistence


@dataclass
class _Entry:
    submission: Submission
    events: list[ProgressEvent] = field(default_factory=list)
    #: Pulsed whenever an event is appended, so waiters wake without polling.
    updated: asyncio.Event = field(default_factory=asyncio.Event)
    #: The stored version this submission was read at, or None if never stored.
    #:
    #: Held per entry rather than passed around because it belongs to the copy in
    #: hand: it is the answer to "what did I base this edit on?", which is the
    #: only question a conditional write asks.
    version: int | None = None


#: How long a submission may sit at `processing` before it is treated as lost.
#:
#: Well past what ingest takes. The slowest document in the corpus — a five-page
#: script through recognition, alignment and a marking call per question — runs
#: around two minutes, so this is an order of magnitude of headroom. Calling a
#: working submission abandoned would be the same bug pointed the other way, and
#: the cost of waiting a little longer to say so is nothing.
STALE_AFTER = timedelta(minutes=20)


class SubmissionStore:
    """Holds submissions and their progress streams for the process lifetime."""

    def __init__(self, persistence: Persistence | None = None) -> None:
        self._entries: dict[str, _Entry] = {}
        self._content_cache: dict[str, str] = {}
        self._persistence = persistence if persistence is not None else default_persistence()

    @property
    def durable(self) -> bool:
        """Whether a submission would survive a restart."""
        return self._persistence.available()

    # -- submissions -------------------------------------------------------

    def put(self, submission: Submission) -> None:
        """Store a submission, table first.

        The order is deliberate. Updating memory first and then failing to write
        would leave this process serving a submission that no other process, and
        no restart, will ever see — a difference that shows up much later as a
        submission that vanished, with nothing in the logs near the cause.
        """
        # Stamped here because this is the one place every write passes through,
        # so nothing can store a submission without saying when.
        submission.updated_at = datetime.now(UTC)

        entry = self._entries.get(submission.submission_id)
        version = self._persistence.save(
            submission, expect_version=entry.version if entry else None
        )
        if entry is None:
            self._entries[submission.submission_id] = _Entry(
                submission=submission, version=version
            )
        else:
            entry.submission = submission
            entry.version = version

    def get(self, submission_id: str) -> Submission | None:
        entry = self._entries.get(submission_id)
        if entry is not None:
            return self._settled(entry.submission)

        # A miss is not necessarily absence any more: it is also every submission
        # made before the last restart.
        stored = self._persistence.load(submission_id)
        if stored is None:
            return None
        self._entries[submission_id] = _Entry(
            submission=stored.submission, version=stored.version
        )
        return self._settled(stored.submission)

    def _settled(self, submission: Submission) -> Submission:
        """The submission, with a lost one reported as lost.

        Decided when somebody asks rather than by a sweeper on a schedule. The
        screen waiting on this is the one a person is looking at, so the moment
        the answer is wanted is the moment to work it out — and a scheduler is a
        second thing to deploy, keep running and notice the failure of.

        Written back, not merely reported. Otherwise every reader recomputes it,
        and anything reading the table directly still finds a submission claiming
        to be at work. The write is best-effort: losing the race to a worker that
        turned out to be alive after all means it is not abandoned, which is the
        outcome anybody would want.
        """
        if submission.status not in {SubmissionStatus.PENDING, SubmissionStatus.PROCESSING}:
            return submission
        stamped = submission.updated_at
        if stamped is None or datetime.now(UTC) - stamped < STALE_AFTER:
            return submission

        submission.status = SubmissionStatus.FAILED
        submission.error = (
            "This submission did not finish. The service was interrupted while "
            "reading it — most likely a restart. Upload the two documents again."
        )
        with contextlib.suppress(Exception):  # see the note above
            self.put(submission)
        return submission

    def require(self, submission_id: str) -> Submission:
        submission = self.get(submission_id)
        if submission is None:
            raise KeyError(f"unknown submission: {submission_id}")
        return submission

    def ids(self) -> list[str]:
        """Submissions this process has seen.

        Deliberately not a table scan. Nothing in the product lists submissions —
        a teacher arrives with a link — and a scan is the operation that quietly
        becomes expensive as a table grows.
        """
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
        found = self._content_cache.get(content_hash)
        if found is not None:
            return found
        found = self._persistence.content_lookup(content_hash)
        if found is not None:
            self._content_cache[content_hash] = found
        return found

    def remember_content(self, content_hash: str, submission_id: str) -> None:
        """Record that this content has been processed, first writer winning.

        Written through before memory, and then read back, for a reason a test
        caught: the persistence layer keeps whoever got there first, so a process
        that lost the race and cached its own id would keep re-answering with a
        submission the table does not agree is the canonical one. Ask who won
        rather than assuming it was us — the same ordering rule as `put`.
        """
        if content_hash in self._content_cache:
            return
        self._persistence.content_remember(content_hash, submission_id)
        winner = self._persistence.content_lookup(content_hash) or submission_id
        self._content_cache[content_hash] = winner

    def clear(self) -> None:
        """Drop everything. Used by tests to isolate cases."""
        self._entries.clear()
        self._content_cache.clear()


#: Process-wide store. Injected via a FastAPI dependency so tests can override it.
store = SubmissionStore()


def get_store() -> SubmissionStore:
    return store
