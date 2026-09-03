"""Phase 0 smoke tests: the app boots and the contracts package is reachable."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from vedaai_contracts.geometry import RENDER_DPI

from grader.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_reports_the_geometry_contract() -> None:
    # A DPI mismatch between services shifts every highlight without erroring,
    # so the constant is exposed rather than left implicit.
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["render_dpi"] == RENDER_DPI
    assert body["hgbench_scale"] == 1000
    assert body["contract_model_count"] > 20


def test_health_states_the_upload_limit_the_service_enforces() -> None:
    """The number on the upload screen has to be the number that rejects a file.

    The screen said "Max 10MB" — the request-body cap of a host the uploads no
    longer pass through — while render.py accepted 40 MB. Understating a limit is
    not a harmless label: someone with a 25 MB phone-photo PDF reads it and does
    not try. Asserting the equality is what keeps the two from drifting again.
    """
    from grader.render import MAX_BYTES

    body = TestClient(app).get("/health").json()
    assert body["max_upload_bytes"] == MAX_BYTES


def test_health_says_whether_submissions_survive_a_restart() -> None:
    # False here, because the test process has no table. The point is that the
    # answer is reported rather than assumed — it was silently no for the whole
    # first week this was deployed.
    body = TestClient(app).get("/health").json()
    assert body["submissions_durable"] is False


def test_contracts_are_importable_across_the_package_boundary() -> None:
    # Guards the uv path dependency. If this breaks, everything downstream
    # breaks with a confusing ImportError deep in a pipeline stage.
    from vedaai_contracts import BBox, Highlight, PageBox

    highlight = Highlight(
        boxes=[
            PageBox(page=0, box=BBox(x0=0.1, y0=0.8, x1=0.9, y1=0.95)),
            PageBox(page=1, box=BBox(x0=0.1, y0=0.05, x1=0.9, y1=0.30)),
        ]
    )
    assert highlight.spans_pages
    assert highlight.pages == [0, 1]


class TestCrossOriginAccess:
    """The browser talks to this service directly, so CORS is load-bearing.

    Worth testing because the failure is asymmetric and quiet: a server-rendered
    read never leaves the server and is unaffected, so a blocked origin breaks
    only the interactive writes — and it looks like a dead button, not a network
    error.
    """

    def test_a_loopback_dev_origin_on_any_port_may_write(self) -> None:
        from fastapi.testclient import TestClient

        from grader.main import app

        for origin in ("http://localhost:3001", "http://127.0.0.1:3001"):
            response = TestClient(app).options(
                "/submissions/x/mapping/A%2F1",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "PATCH",
                },
            )
            assert response.status_code == 200, origin
            assert response.headers["access-control-allow-origin"] == origin

    def test_a_configured_deployment_does_not_allow_loopback(self, monkeypatch) -> None:
        # The loopback allowance is a development convenience and must not
        # survive into a deployment that names its own origin.
        import importlib

        monkeypatch.setenv("WEB_ORIGINS", "https://grader.example.com")
        import grader.main as main_module

        reloaded = importlib.reload(main_module)
        try:
            from fastapi.testclient import TestClient

            client = TestClient(reloaded.app)
            allowed = client.options(
                "/submissions/x/mapping/A%2F1",
                headers={
                    "Origin": "https://grader.example.com",
                    "Access-Control-Request-Method": "PATCH",
                },
            )
            assert allowed.headers.get("access-control-allow-origin") == (
                "https://grader.example.com"
            )

            blocked = client.options(
                "/submissions/x/mapping/A%2F1",
                headers={
                    "Origin": "http://localhost:3001",
                    "Access-Control-Request-Method": "PATCH",
                },
            )
            assert "access-control-allow-origin" not in blocked.headers
        finally:
            monkeypatch.delenv("WEB_ORIGINS", raising=False)
            importlib.reload(main_module)


class TestHealthSaysWhetherMarkingWillRun:
    """The signal whose absence let a deploy go green with marking dead.

    `/health` reported render DPI, contract count, upload cap and store
    durability — everything except the one thing a submission is judged on. It
    returns 200 with no API key configured, in which case `select_grader` falls
    back to `RubricOnly` and every question comes back zero with a warning nobody
    reads. The deploy pipeline asserts only that this endpoint answers 200, so a
    rotated key, a dropped secret ARN or an empty account all ship green.

    That is not hypothetical. It is what happened: the account ran out of credit,
    the deploy reported success, and the live service marked nothing while telling
    teachers "3 of 6 answered · rubric only".
    """

    def test_it_reports_that_nothing_will_be_marked(self, client, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        body = client.get("/health").json()

        assert body["grading"]["configured"] is False
        assert body["grading"]["engine"] == "rubric_only"
        assert body["grading"]["model"] is None

    def test_it_names_the_model_that_will_mark(self, client, monkeypatch) -> None:
        """A mark is only checkable if you know what made it — and that has to be
        answerable before a script is uploaded, not only afterwards from a grade."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GRADER_PROVIDER", "openai")

        body = client.get("/health").json()

        assert body["grading"]["configured"] is True
        assert body["grading"]["engine"] == "openai"
        assert body["grading"]["model"]

    def test_it_reports_which_scorer_will_place_answers(self, client, monkeypatch) -> None:
        """The other half of the same question.

        Placement degrades to word overlap without a key, which is a materially
        different product, and until now nothing outside a finished submission
        said which one was running.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert client.get("/health").json()["similarity"] == "lexical"

    def test_asking_does_not_cost_a_request_to_the_provider(self, client) -> None:
        """A health check runs every thirty seconds against a paid API."""
        import grader.answers.similarity as sim

        calls: list[int] = []
        original = sim._openai_embed
        sim._openai_embed = lambda texts: calls.append(1) or [[1.0] for _ in texts]
        try:
            client.get("/health")
        finally:
            sim._openai_embed = original

        assert calls == []
