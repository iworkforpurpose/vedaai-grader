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

from .render import MAX_BYTES
from .store import get_store
from .routes import router

app = FastAPI(
    title="Vedaai Grader API",
    version="0.1.0",
    description=(
        "Extracts questions from a printed paper, maps handwritten answers to them, "
        "and returns highlight geometry derived from OCR boxes."
    ),
)

# The browser fetches page images, the SSE progress stream and the mapping
# endpoints directly from this service, so the deployed web origin has to be
# allowed explicitly.
_origins = [o for o in os.getenv("WEB_ORIGINS", "").split(",") if o]

#: Any loopback origin, used only when no explicit list is configured.
#:
#: A fixed dev default was worse than no default. "localhost" and "127.0.0.1" are
#: different origins to a browser, and the dev server takes whichever port is
#: free, so a hardcoded ``localhost:3000`` blocked the real dev setup. The failure
#: is quiet in the worst way: server-rendered reads are unaffected because they
#: never leave the server, so only the interactive writes break, and they present
#: as a button that does nothing.
_LOOPBACK_ORIGIN = r"http://(localhost|127\.0\.0\.1)(:\d+)?"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # Deployment sets WEB_ORIGINS, which switches the loopback allowance off.
    allow_origin_regex=None if _origins else _LOOPBACK_ORIGIN,
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
    #: The largest document render.py will accept, in bytes.
    #:
    #: Reported so the upload screen can state the real limit rather than a
    #: number typed into a label. It said "Max 10MB" — the cap of a host the
    #: uploads no longer pass through — while the service accepts four times
    #: that, so a teacher with a phone-photo PDF would not have tried.
    max_upload_bytes: int
    #: Whether a submission would survive a restart of this service.
    #:
    #: Reported because the answer used to be no, and the consequence — every
    #: deploy discarding whatever a tester was half way through — is invisible
    #: until it happens to someone. Now it is checkable from outside.
    submissions_durable: bool


@app.get("/health", response_model=Health, tags=["meta"])
def health() -> Health:
    return Health(
        status="ok",
        version=app.version,
        render_dpi=RENDER_DPI,
        hgbench_scale=HGBENCH_SCALE,
        contract_model_count=len(EXPORTED_MODELS),
        max_upload_bytes=MAX_BYTES,
        submissions_durable=get_store().durable,
    )


app.include_router(router)
