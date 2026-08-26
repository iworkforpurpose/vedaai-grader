"""Tests for the in-memory store, focused on the progress-replay behaviour."""

from __future__ import annotations

from vedaai_contracts import ProgressEvent, Stage, Submission

from grader.store import SubmissionStore


def make_store() -> tuple[SubmissionStore, str]:
    store = SubmissionStore()
    store.put(Submission(submission_id="s1"))
    return store, "s1"


def test_events_replay_from_a_cursor() -> None:
    # A browser reconnecting mid-job must receive what it missed. A 90-second
    # job outliving a flaky connection is the normal case, not an edge case.
    store, sid = make_store()
    store.emit(sid, ProgressEvent(stage=Stage.RENDERING, message="page 1"))
    store.emit(sid, ProgressEvent(stage=Stage.RENDERING, message="page 2"))

    first, cursor = store.events_since(sid, 0)
    assert [e.message for e in first] == ["page 1", "page 2"]
    assert cursor == 2

    store.emit(sid, ProgressEvent(stage=Stage.TRANSCRIBING, message="page 1"))
    second, cursor = store.events_since(sid, cursor)
    assert [e.message for e in second] == ["page 1"]
    assert cursor == 3


def test_events_for_unknown_submission_are_inert() -> None:
    store, _ = make_store()
    events, cursor = store.events_since("nope", 0)
    assert events == []
    assert cursor == 0


def test_finished_tracks_terminal_stages() -> None:
    store, sid = make_store()
    assert not store.is_finished(sid)
    store.emit(sid, ProgressEvent(stage=Stage.MAPPING, message="aligning"))
    assert not store.is_finished(sid)
    store.emit(sid, ProgressEvent(stage=Stage.DONE, message="complete"))
    assert store.is_finished(sid)


def test_content_cache_returns_the_first_submission_for_a_hash() -> None:
    # One question paper shared across a class should be OCR'd once, not once
    # per student. This is what keeps a 1,000-page monthly OCR quota viable.
    store, sid = make_store()
    store.remember_content("abc123", sid)
    store.remember_content("abc123", "s2")
    assert store.cached_submission_for("abc123") == sid
    assert store.cached_submission_for("missing") is None
