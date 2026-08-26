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
