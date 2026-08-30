"""Tests for submissions whose worker never came back.

Ingest runs in a background task, and the code around it is careful: every
exception it can catch ends as a stored `failed` submission carrying a reason.
What it cannot catch is the process going away — a deploy rolling the task, the
container running out of memory, the host being replaced. The submission stays at
`processing` with nobody to move it, and the review page waits on it forever.

The screen showing that is the one a tester is looking at, so the fix is not a
sweeper on a schedule but an answer at the moment somebody asks: a submission
that has been processing for longer than ingest could possibly take is not
processing, and saying so costs nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vedaai_contracts import Submission, SubmissionStatus

from grader.store import STALE_AFTER, SubmissionStore


def _aged(minutes: float) -> datetime:
    return datetime.now(UTC) - timedelta(minutes=minutes)


class TestAWorkerThatNeverCameBack:
    def test_a_long_stalled_submission_reads_as_failed(self) -> None:
        store = SubmissionStore()
        store.put(Submission(submission_id="abc", status=SubmissionStatus.PROCESSING))
        # Reach past the store to age it, because a store that let a caller
        # backdate a write would be a worse thing than the bug.
        store._entries["abc"].submission.updated_at = _aged(STALE_AFTER.total_seconds() / 60 + 5)

        found = store.get("abc")
        assert found is not None
        assert found.status is SubmissionStatus.FAILED

    def test_and_says_what_happened(self) -> None:
        store = SubmissionStore()
        store.put(Submission(submission_id="abc", status=SubmissionStatus.PROCESSING))
        store._entries["abc"].submission.updated_at = _aged(STALE_AFTER.total_seconds() / 60 + 5)

        found = store.get("abc")
        assert found is not None
        assert found.error and "did not finish" in found.error.lower()

    def test_one_still_working_is_left_alone(self) -> None:
        # The threshold has to sit well past how long ingest actually takes. A
        # five-page script is around two minutes, and calling that abandoned would
        # be the same bug pointed the other way.
        store = SubmissionStore()
        store.put(Submission(submission_id="abc", status=SubmissionStatus.PROCESSING))
        store._entries["abc"].submission.updated_at = _aged(3)

        found = store.get("abc")
        assert found is not None
        assert found.status is SubmissionStatus.PROCESSING

    def test_a_finished_submission_never_ages_into_failure(self) -> None:
        # Nothing about being old makes a completed submission wrong, and a
        # reviewer opening last week's script must find it intact.
        store = SubmissionStore()
        store.put(Submission(submission_id="abc", status=SubmissionStatus.COMPLETE))
        store._entries["abc"].submission.updated_at = _aged(60 * 24 * 7)

        found = store.get("abc")
        assert found is not None
        assert found.status is SubmissionStatus.COMPLETE

    def test_the_verdict_is_written_down(self) -> None:
        # Otherwise every reader recomputes it, and anything reading the table
        # directly still sees a submission that claims to be working.
        store = SubmissionStore()
        store.put(Submission(submission_id="abc", status=SubmissionStatus.PROCESSING))
        store._entries["abc"].submission.updated_at = _aged(STALE_AFTER.total_seconds() / 60 + 5)

        store.get("abc")
        assert store._entries["abc"].submission.status is SubmissionStatus.FAILED

    def test_a_write_refreshes_the_clock(self) -> None:
        # A long ingest that reports progress must not be killed by its own
        # duration: each step it stores resets how long it has been quiet.
        store = SubmissionStore()
        submission = Submission(submission_id="abc", status=SubmissionStatus.PROCESSING)
        store.put(submission)
        store._entries["abc"].submission.updated_at = _aged(STALE_AFTER.total_seconds() / 60 + 5)

        store.put(store._entries["abc"].submission.model_copy(
            update={"status": SubmissionStatus.PROCESSING}
        ))
        found = store.get("abc")
        assert found is not None
        assert found.status is SubmissionStatus.PROCESSING
