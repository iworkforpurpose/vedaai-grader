"""Phase 0 smoke tests: the app boots and the contracts package is reachable."""

from __future__ import annotations

from fastapi.testclient import TestClient
from vedaai_contracts.geometry import RENDER_DPI

from grader.main import app


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
