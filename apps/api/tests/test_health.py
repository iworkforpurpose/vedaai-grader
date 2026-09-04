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

        Placement degrades to word overlap when neither a hosted key nor a local
        model is available, which is a materially
        different product, and until now nothing outside a finished submission
        said which one was running.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("LOCAL_EMBEDDINGS", "0")
        import importlib

        from grader.answers import similarity as similarity_module

        importlib.reload(similarity_module)
        try:
            assert client.get("/health").json()["similarity"] == "lexical"
        finally:
            monkeypatch.delenv("LOCAL_EMBEDDINGS", raising=False)
            importlib.reload(similarity_module)

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


class TestProvingTheMarkerCanActuallyMark:
    """A key that exists is not a provider that answers.

    The first version of this check asked `select_grader` whether a grader could
    be *built*, and a grader is built from a key being present. So a deployment
    whose provider account had run out of credit reported

        "grading": {"configured": true, "engine": "openai", "model": "gpt-4.1"}

    and the release passed — while every submission came back zeros with
    "the provider is rate limiting or the account is out of credit" buried in a
    warnings list nobody reads. That is the exact failure the check was added for,
    and it walked straight through it.

    So readiness is two questions, not one. Is a marker configured, which is free
    to answer and safe on a probe that runs every thirty seconds. And does the
    provider answer, which costs a request and therefore happens only when asked
    for, once per release, cached.
    """

    def test_the_cheap_check_does_not_claim_the_provider_answers(
        self, client, monkeypatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GRADER_PROVIDER", "openai")

        grading = client.get("/health").json()["grading"]

        assert grading["configured"] is True
        assert grading["reachable"] is None, (
            "the default probe must not imply a provider it never contacted"
        )

    def test_the_deep_check_reports_a_provider_that_refuses(
        self, client, monkeypatch
    ) -> None:
        from grader import main as main_module

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        main_module.forget_reachability()

        async def refuses(_grader):
            raise RuntimeError("You have no credits remaining.")

        monkeypatch.setattr(main_module, "_ask_the_provider", refuses)
        body = client.get("/health?deep=1").json()

        assert body["grading"]["reachable"] is False
        assert "credits" in body["grading"]["detail"]

    def test_a_spent_quota_is_not_the_same_answer_as_a_rejected_key(
        self, client, monkeypatch
    ) -> None:
        """Both are `reachable: false`, and they need opposite responses.

        A rejected key means nothing marks until somebody changes a secret, and a
        release carrying it must be red. A rate limit means the deployment is
        correct and marking resumes when the window rolls — on a free tier, daily.
        With only the boolean the release step has to pick which one to get wrong:
        block every deploy for the rest of the day, or ship a dead marker green.
        """
        from grader import main as main_module

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class RateLimitError(RuntimeError):
            pass

        async def throttled(_grader):
            raise RateLimitError("Rate limit reached for model, try again in 2m1s")

        async def rejected(_grader):
            raise RuntimeError("Incorrect API key provided")

        main_module.forget_reachability()
        monkeypatch.setattr(main_module, "_ask_the_provider", throttled)
        busy = client.get("/health?deep=1").json()["grading"]

        main_module.forget_reachability()
        monkeypatch.setattr(main_module, "_ask_the_provider", rejected)
        broken = client.get("/health?deep=1").json()["grading"]

        assert busy["reachable"] is False and broken["reachable"] is False
        assert busy["throttled"] is True
        assert broken["throttled"] is False

    def test_a_provider_that_answers_is_never_reported_throttled(
        self, client, monkeypatch
    ) -> None:
        from grader import main as main_module

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        main_module.forget_reachability()

        async def answers(_grader):
            return None

        monkeypatch.setattr(main_module, "_ask_the_provider", answers)
        assert client.get("/health?deep=1").json()["grading"]["throttled"] is False

    def test_the_deep_check_reports_a_provider_that_answers(
        self, client, monkeypatch
    ) -> None:
        from grader import main as main_module

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        main_module.forget_reachability()

        async def answers(_grader):
            return None

        monkeypatch.setattr(main_module, "_ask_the_provider", answers)
        body = client.get("/health?deep=1").json()

        assert body["grading"]["reachable"] is True
        assert body["grading"]["detail"] is None

    def test_the_answer_is_cached_so_a_probe_cannot_become_a_bill(
        self, client, monkeypatch
    ) -> None:
        """A liveness probe that spends money per call is one somebody turns off."""
        from grader import main as main_module

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        main_module.forget_reachability()
        calls: list[int] = []

        async def answers(_grader):
            calls.append(1)

        monkeypatch.setattr(main_module, "_ask_the_provider", answers)
        for _ in range(5):
            client.get("/health?deep=1")

        assert len(calls) == 1

    def test_an_unconfigured_marker_is_not_asked(self, client, monkeypatch) -> None:
        """There is nothing to ask, and `configured: false` already answers it."""
        from grader import main as main_module

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        main_module.forget_reachability()
        calls: list[int] = []

        async def answers(_grader):
            calls.append(1)

        monkeypatch.setattr(main_module, "_ask_the_provider", answers)
        body = client.get("/health?deep=1").json()

        assert body["grading"]["configured"] is False
        assert body["grading"]["reachable"] is False
        assert calls == []
