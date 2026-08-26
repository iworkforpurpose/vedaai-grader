"""FastAPI application entry point.

Phase 0 scope: the app boots, reports health, and proves the contracts package
is importable across the package boundary. Pipeline endpoints arrive in Phase 1.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vedaai_contracts import EXPORTED_MODELS
from vedaai_contracts.geometry import HGBENCH_SCALE, RENDER_DPI

app = FastAPI(
    title="Vedaai Grader API",
    version="0.1.0",
    description=(
        "Extracts questions from a printed paper, maps handwritten answers to them, "
        "and returns highlight geometry derived from OCR boxes."
    ),
)

# The browser fetches page images and the SSE progress stream directly from this
# service, so the deployed web origin has to be allowed explicitly.
_origins = [o for o in os.getenv("WEB_ORIGINS", "http://localhost:3000").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)


class Health(BaseModel):
    """Health payload.

    Reports the geometry constants rather than just a bare "ok". If the API and
    the frontend ever disagree about render DPI, that mismatch silently shifts
    every highlight — so it is worth making visible at a glance.
    """

    status: str
    version: str
    render_dpi: int
    hgbench_scale: int
    contract_model_count: int


@app.get("/health", response_model=Health, tags=["meta"])
def health() -> Health:
    return Health(
        status="ok",
        version=app.version,
        render_dpi=RENDER_DPI,
        hgbench_scale=HGBENCH_SCALE,
        contract_model_count=len(EXPORTED_MODELS),
    )
