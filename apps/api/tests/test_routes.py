"""End-to-end tests over the HTTP surface."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from grader import storage
from grader import store as store_module
from grader.main import app
from grader.storage import PageStore
from grader.store import SubmissionStore

from .fixtures import answer_sheet_with_text, question_paper, single_page_image


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    """A client with isolated page and submission stores.

    Both are process-wide singletons in production. Swapping them per test keeps
    cases from seeing each other's submissions or cached page images.
    """
    pages = PageStore(root=tmp_path / "pages")
    submissions = SubmissionStore()
    monkeypatch.setattr(storage, "store", pages)
    monkeypatch.setattr(store_module, "store", submissions)
    app.dependency_overrides[storage.get_page_store] = lambda: pages
    app.dependency_overrides[store_module.get_store] = lambda: submissions
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client: TestClient, *, qp: bytes | None = None, ans: bytes | None = None):
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


class TestUpload:
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

    def test_reports_the_missing_handwriting_engine_as_a_warning(self, client: TestClient) -> None:
        # Honest partial state: the answer sheet cannot be transcribed until an
        # OCR engine is configured, and that is surfaced rather than silently
        # producing an empty result that looks like a blank script.
        body = upload(client).json()
        assert body["answer_sheet_lines"] is None
        assert any("answer sheet" in w for w in body["warnings"])
        # Crucially, the run still succeeds and still produces page images, so
        # the reviewer is usable.
        assert body["status"] == "complete"

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
        assert "OCR engine" in response.json()["detail"]


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
