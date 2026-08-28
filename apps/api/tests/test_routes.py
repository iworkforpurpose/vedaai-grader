"""End-to-end tests over the HTTP surface."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from grader import storage
from grader import store as store_module
from grader.main import app
from grader.ocr import PaddleOcrEngine, TextractEngine
from grader.storage import PageStore
from grader.store import SubmissionStore

from .fixtures import answer_sheet_with_text, question_paper, single_page_image


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    """A client with isolated stores and the OCR model disabled.

    The stores are process-wide singletons in production; swapping them per test
    keeps cases from seeing each other's submissions or cached page images.

    Handwriting recognition is switched off here deliberately. These tests cover
    the HTTP surface and the graceful-degradation path, and loading the OCR model
    would add about a minute per case while testing nothing this file is
    responsible for. The real transcription path is covered by
    test_paddle_engine.py and by test_real_handwriting_end_to_end below.
    """
    # Both handwriting engines are switched off. Disabling only the local one used
    # to be enough; now Textract is preferred, and on a machine with any AWS
    # credentials at all it would be selected and then fail against the network.
    monkeypatch.setattr(PaddleOcrEngine, "available", lambda self: False)
    monkeypatch.setattr(TextractEngine, "available", lambda self: False)
    pages = PageStore(root=tmp_path / "pages")
    submissions = SubmissionStore()
    monkeypatch.setattr(storage, "store", pages)
    monkeypatch.setattr(store_module, "store", submissions)
    app.dependency_overrides[storage.get_page_store] = lambda: pages
    app.dependency_overrides[store_module.get_store] = lambda: submissions
    # As a context manager, so the event loop lives across requests.
    #
    # A bare TestClient gives each request its own portal and tears the loop down
    # when the response returns — which orphans the ingest, now that upload
    # schedules it as a background task instead of awaiting it. Every upload test
    # would see a permanently `processing` submission.
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def start_upload(client: TestClient, *, qp: bytes | None = None, ans: bytes | None = None):
    """Post both documents and return the immediate response, unwaited."""
    if qp is None:
        qp, _ = question_paper()
    if ans is None:
        ans, _ = answer_sheet_with_text()
    return client.post(
        "/submissions",
        files={
            "question_paper": ("science_unit_test.pdf", qp, "application/pdf"),
            "answer_sheet": ("suyash_6c.pdf", ans, "application/pdf"),
        },
    )


def upload(client: TestClient, *, qp: bytes | None = None, ans: bytes | None = None):
    """Post both documents and wait for the pipeline to settle.

    Ingest is a background task now, so the upload response is a `processing`
    stub. Everything downstream of this helper is asserting on the *result*, so it
    waits here rather than in thirty tests — and it returns the final GET, which
    carries the same shape the POST used to.

    Waits on the status rather than on a fixed sleep. A sleep long enough to be
    reliable makes the suite slow, and one short enough to be quick makes it flaky.
    """
    started = start_upload(client, qp=qp, ans=ans)
    if started.status_code != 200:
        return started
    return wait_for_ingest(client, started.json()["submission_id"])


def wait_for_ingest(client: TestClient, submission_id: str):
    """Block until a submission stops being `processing`, and return its final GET.

    Waits on the status rather than on a fixed sleep. A sleep long enough to be
    reliable makes the suite slow, and one short enough to be quick makes it flaky.
    """
    for _ in range(600):  # 30s ceiling; the fixtures settle in well under one
        current = client.get(f"/submissions/{submission_id}")
        if current.status_code != 200 or current.json()["status"] != "processing":
            return current
        time.sleep(0.05)
    raise AssertionError(f"submission {submission_id} never left processing")


class TestUpload:
    def test_returns_immediately_so_no_proxy_can_time_the_upload_out(
        self, client: TestClient
    ) -> None:
        """The upload answers before the work is done.

        This is the contract that replaced awaiting ingest in the request. Deployed,
        every upload died at almost exactly thirty seconds — no traceback, worker
        still healthy, a one-page sheet failing at 30.7s and a two-page at 32.0s,
        which is a wall rather than a resource limit. A pipeline measured in tens of
        seconds per page cannot sit inside one HTTP request, because every layer in
        between gets to impose its own timeout.
        """
        response = start_upload(client)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "processing"
        assert body["submission_id"]
        # Nothing derived yet — that is the point.
        assert body["questions"] is None

    def test_ingests_both_documents(self, client: TestClient) -> None:
        response = upload(client)
        assert response.status_code == 200, response.text

        body = response.json()
        assert body["status"] == "complete"
        assert body["question_paper_file"]["filename"] == "science_unit_test.pdf"
        assert body["answer_sheet_file"]["filename"] == "suyash_6c.pdf"
        # Both fixtures span two pages, so four in total.
        assert len(body["pages"]) == 4
        assert body["answer_sheet_page_count"] == 2

    def test_transcribes_the_question_paper(self, client: TestClient) -> None:
        body = upload(client).json()
        lines = body["question_paper_lines"]
        assert lines is not None
        assert lines["engine"] == "pdf_text"
        assert len(lines["lines"]) > 20
        assert any("refraction" in ln["text"].lower() for ln in lines["lines"])

    def test_degrades_honestly_when_no_handwriting_engine_is_available(
        self, client: TestClient
    ) -> None:
        # The failure mode this guards against is the dangerous one: silently
        # producing an empty transcription, which is indistinguishable from a
        # genuinely blank script and would be reported as "unanswered".
        body = upload(client).json()
        assert body["answer_sheet_lines"] is None
        assert any("ocr-local" in w for w in body["warnings"]), body["warnings"]
        # The run still succeeds and still produces page images, so the reviewer
        # remains usable rather than the whole submission failing.
        assert body["status"] == "complete"
        assert len(body["pages"]) == 4

    def test_accepts_a_photographed_answer_sheet(self, client: TestClient) -> None:
        response = upload(client, ans=single_page_image())
        assert response.status_code == 200, response.text
        assert response.json()["answer_sheet_page_count"] == 1

    def test_rejects_a_file_that_is_not_a_document(self, client: TestClient) -> None:
        response = upload(client, ans=b"definitely not a pdf")
        assert response.status_code == 422
        assert "neither a readable PDF" in response.json()["detail"]

    def test_rejects_an_empty_file(self, client: TestClient) -> None:
        response = upload(client, qp=b"")
        assert response.status_code == 422
        assert "is empty" in response.json()["detail"]


class TestRetrieval:
    def test_fetches_a_submission_by_id(self, client: TestClient) -> None:
        submission_id = upload(client).json()["submission_id"]
        response = client.get(f"/submissions/{submission_id}")
        assert response.status_code == 200
        assert response.json()["submission_id"] == submission_id

    def test_unknown_submission_is_a_404(self, client: TestClient) -> None:
        assert client.get("/submissions/deadbeef").status_code == 404

    def test_serves_the_line_index_for_the_overlay(self, client: TestClient) -> None:
        # What the debug overlay draws. If a highlight later lands wrong, this
        # endpoint answers whether geometry or mapping is at fault.
        submission_id = upload(client).json()["submission_id"]
        response = client.get(f"/submissions/{submission_id}/lines/question_paper")
        assert response.status_code == 200

        index = response.json()
        assert index["kind"] == "question_paper"
        for line in index["lines"]:
            box = line["box"]
            assert 0.0 <= box["x0"] < box["x1"] <= 1.0
            assert 0.0 <= box["y0"] < box["y1"] <= 1.0

    def test_missing_line_index_explains_itself(self, client: TestClient) -> None:
        submission_id = upload(client).json()["submission_id"]
        response = client.get(f"/submissions/{submission_id}/lines/answer_sheet")
        assert response.status_code == 404
        # The message names the actual remedy rather than just reporting absence,
        # because "no transcription available" is not actionable on its own.
        assert "ocr-local" in response.json()["detail"]


class TestPageImages:
    def test_serves_a_rendered_page(self, client: TestClient) -> None:
        body = upload(client).json()
        key = body["pages"][0]["image_key"]

        response = client.get(f"/pages/{key}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")

    def test_page_images_are_cached_immutably(self, client: TestClient) -> None:
        # Keys are content-addressed, so bytes never change for a given key. The
        # review surface scrolls every page repeatedly; refetching would make
        # the overlay feel broken.
        body = upload(client).json()
        key = body["pages"][0]["image_key"]
        response = client.get(f"/pages/{key}")
        assert "immutable" in response.headers["cache-control"]

    def test_unknown_page_is_a_404(self, client: TestClient) -> None:
        assert client.get("/pages/abc123/p0000.png").status_code == 404

    def test_rejects_a_traversal_attempt(self, client: TestClient) -> None:
        response = client.get("/pages/..%2f..%2f..%2fetc%2fpasswd")
        assert response.status_code in {400, 404}


class TestProgressStream:
    def test_replays_events_and_terminates(self, client: TestClient) -> None:
        # Replay from the beginning matters: ingest finishes before the browser
        # opens the stream in this test, and in production a reconnect mid-run
        # must not lose the stages it missed.
        submission_id = upload(client).json()["submission_id"]

        with client.stream("GET", f"/submissions/{submission_id}/events") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            payloads = [
                json.loads(line[len("data: ") :])
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

        assert payloads, "expected at least one progress event"
        assert payloads[-1]["stage"] == "done"

        stages = [p["stage"] for p in payloads]
        assert "rendering" in stages
        # Page counters are what make a long wait legible.
        assert any(p["pages_total"] for p in payloads)

    def test_stream_for_unknown_submission_is_a_404(self, client: TestClient) -> None:
        assert client.get("/submissions/deadbeef/events").status_code == 404


class TestContentCache:
    def test_the_same_paper_reuses_rendered_pages(self, client: TestClient, tmp_path) -> None:
        # One paper shared across a class should cost one render, not one per
        # student. This is what keeps a 1,000-page monthly OCR quota viable.
        qp, _ = question_paper()
        first = upload(client, qp=qp).json()
        second = upload(client, qp=qp).json()

        first_keys = [p["image_key"] for p in first["pages"] if p["kind"] == "question_paper"]
        second_keys = [p["image_key"] for p in second["pages"] if p["kind"] == "question_paper"]
        assert first_keys == second_keys
        assert first["submission_id"] != second["submission_id"]


@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("GRADER_SAMPLE_DIR"),
    reason="set GRADER_SAMPLE_DIR to a directory of real handwritten pages",
)
def test_real_handwriting_end_to_end(tmp_path, monkeypatch) -> None:
    """Upload a real handwritten script and confirm the whole path yields geometry.

    Skipped unless real pages are supplied, because student work does not belong
    in this repository. This is the case that proves the pipeline works on the
    input it exists to handle, rather than on a typed stand-in.
    """
    if not PaddleOcrEngine().available():
        pytest.skip("local OCR extra not installed")

    sample_dir = Path(os.environ["GRADER_SAMPLE_DIR"])
    images = sorted(
        p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        pytest.skip(f"no images in {sample_dir}")

    pages = PageStore(root=tmp_path / "pages")
    submissions = SubmissionStore()
    monkeypatch.setattr(storage, "store", pages)
    monkeypatch.setattr(store_module, "store", submissions)
    app.dependency_overrides[storage.get_page_store] = lambda: pages
    app.dependency_overrides[store_module.get_store] = lambda: submissions
    try:
        client = TestClient(app)
        qp, _ = question_paper()
        response = client.post(
            "/submissions",
            files={
                "question_paper": ("paper.pdf", qp, "application/pdf"),
                "answer_sheet": (images[0].name, images[0].read_bytes(), "image/jpeg"),
            },
        )
        assert response.status_code == 200, response.text
        index = response.json()["answer_sheet_lines"]

        assert index is not None, "real handwriting should have been transcribed"
        assert index["engine"] == "paddle"
        assert len(index["lines"]) > 5

        # Geometry is the deliverable here. Transcription quality on handwriting
        # is poor and that is tolerable, because highlights come from boxes.
        for line in index["lines"]:
            box = line["box"]
            assert 0.0 <= box["x0"] < box["x1"] <= 1.0
            assert 0.0 <= box["y0"] < box["y1"] <= 1.0

        low = sum(1 for line in index["lines"] if line["confidence"] < 0.7)
        print(
            f"{images[0].name}: {len(index['lines'])} regions, "
            f"{low} below 0.7 confidence (struck-through work lands here)"
        )
    finally:
        app.dependency_overrides.clear()


class TestGradingEndpoint:
    """The marking endpoint.

    Marking is requested rather than automatic. Locating answers is useful on its
    own and must not wait behind it, and a proposed score is hard to unsee — so
    the teacher asks for it.
    """

    def test_an_unknown_submission_is_a_404(self, client) -> None:
        assert client.post("/submissions/nope/grades").status_code == 404

    def test_an_untranscribed_sheet_is_refused_with_a_reason(self, client) -> None:
        # The state this fixture is actually in: recognition is disabled, so the
        # answer sheet has no text. Marking must say that rather than mark an
        # empty script and report a zero.
        paper, _ = question_paper()
        sheet, _ = answer_sheet_with_text()
        submission_id = client.post(
            "/submissions",
            files={
                "question_paper": ("paper.pdf", paper, "application/pdf"),
                "answer_sheet": ("student.pdf", sheet, "application/pdf"),
            },
        ).json()["submission_id"]
        # Ingest runs behind the response now, and this asserts on what it left.
        wait_for_ingest(client, submission_id)

        refused = client.post(f"/submissions/{submission_id}/grades")
        assert refused.status_code == 409
        assert "never transcribed" in refused.json()["detail"]

    def test_it_returns_a_rubric_with_no_model_configured(self, client, monkeypatch) -> None:
        # The degraded path is the one most likely to run, so it is the one worth
        # testing end to end: rubric and located answer, no invented marks.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        submission_id = _submission_ready_to_mark(client)

        graded = client.post(f"/submissions/{submission_id}/grades")
        assert graded.status_code == 200
        body = graded.json()

        assert body["grades"]["grades"], "expected one grade per question"
        assert body["grades"]["total_awarded"] == 0.0
        assert body["grades"]["committed"] is False
        assert any("not marked automatically" in w for w in body["warnings"])

    def test_the_marks_available_come_from_the_paper(self, client, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        submission_id = _submission_ready_to_mark(client)

        body = client.post(f"/submissions/{submission_id}/grades").json()
        assert body["grades"]["total_available"] > 0
        for grade in body["grades"]["grades"]:
            assert grade["rubric_points"], grade["qid"]

    def test_marking_twice_replaces_rather_than_accumulates(self, client, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        submission_id = _submission_ready_to_mark(client)

        first = client.post(f"/submissions/{submission_id}/grades").json()["grades"]
        second = client.post(f"/submissions/{submission_id}/grades").json()["grades"]
        assert len(second["grades"]) == len(first["grades"])


def _submission_ready_to_mark(client: TestClient) -> str:
    """A submission with answer-sheet lines, which this fixture cannot produce.

    Recognition is disabled for these tests, so the sheet arrives with no text.
    Attaching a small index directly is what lets the marking route be exercised
    over HTTP without a minute of model loading — and the route is worth covering
    because its guards and its degraded path are the parts most likely to be hit.
    """
    from vedaai_contracts import (
        AnswerStatus,
        BBox,
        DocumentKind,
        Line,
        LineIndex,
        Mapping,
        MappingResult,
        OcrEngine,
    )

    paper, _ = question_paper()
    sheet, _ = answer_sheet_with_text()
    submission_id = client.post(
        "/submissions",
        files={
            "question_paper": ("paper.pdf", paper, "application/pdf"),
            "answer_sheet": ("student.pdf", sheet, "application/pdf"),
        },
    ).json()["submission_id"]
    wait_for_ingest(client, submission_id)

    submission = store_module.store.get(submission_id)
    assert submission is not None and submission.questions is not None

    line = Line(
        line_id="as:0001",
        kind=DocumentKind.ANSWER_SHEET,
        page=0,
        box=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.14),
        text="Light bends when it passes from one medium into another.",
        confidence=0.9,
        engine=OcrEngine.PADDLE_OCR_VL,
    )
    submission.answer_sheet_lines = LineIndex(
        kind=DocumentKind.ANSWER_SHEET, lines=[line], engine=OcrEngine.PADDLE_OCR_VL
    )
    first = submission.questions.questions[0]
    submission.mapping = MappingResult(
        mappings=[
            Mapping(
                qid=first.qid,
                status=AnswerStatus.ANSWERED,
                start_line_id=line.line_id,
                end_line_id=line.line_id,
            )
        ],
        orphans=[],
        unassigned_ink_ratio=0.0,
    )
    store_module.store.put(submission)
    return submission_id
