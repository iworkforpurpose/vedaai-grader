"""FastAPI application entry point.

Phase 0 scope: the app boots, reports health, and proves the contracts package
is importable across the package boundary. Pipeline endpoints arrive in Phase 1.
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vedaai_contracts import EXPORTED_MODELS
from vedaai_contracts.geometry import HGBENCH_SCALE, RENDER_DPI

from .observability import configure as configure_logging
from .render import MAX_BYTES
from .routes import router
from .store import get_store

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


class GradingReadiness(BaseModel):
    """Whether this deployment can mark, and with what."""

    configured: bool
    """A marker is selected. Free to answer, and not the same as it working."""

    engine: str
    model: str | None

    throttled: bool = False
    """The credential works and the quota is spent.

    A distinction worth drawing, because the two failures need opposite responses
    and look identical from a boolean. A rejected key, a dropped secret ARN or a
    wrong model is a misconfiguration: nothing will mark until somebody changes
    something, and a release carrying it should be red. A 429 is capacity: the
    deployment is correct, the credential is good, and marking resumes when the
    window rolls.

    Conflating them means either a spent free-tier budget blocks every deploy, or
    a genuinely broken marker ships green. Neither is acceptable, so the health
    payload says which one it is."""

    reachable: bool | None = None
    """Whether the provider actually answered, when it was asked.

    ``None`` means nobody asked. The distinction is the whole point of this field:
    the first version of this check reported only ``configured``, which is true as
    soon as a key exists — and a deployment whose provider account had run out of
    credit passed its release check while every submission came back zeros. A key
    that exists is not a provider that answers.
    """

    detail: str | None = None
    """Why the provider refused, where it did. Truncated, and never the key."""


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

    #: Whether this deployment will actually mark anything, and with what.
    #:
    #: The gap this closes shipped and cost a day. `/health` reported the render
    #: DPI, the contract count, the upload cap and the store durability —
    #: everything except the one thing a submission is judged on. It answers 200
    #: with no key configured, in which case `select_grader` returns `RubricOnly`
    #: and every question comes back zero with a warning nobody reads. The deploy
    #: pipeline asserts only that this endpoint answers 200, so a rotated key, a
    #: dropped secret ARN or an empty provider account all ship green.
    #:
    #: That is not hypothetical. The account ran out of credit, the deploy
    #: reported success, and the live service marked nothing while telling
    #: teachers "3 of 6 answered · rubric only".
    grading: GradingReadiness

    #: Which measure will place answers. Without a key this degrades to word
    #: overlap, which is a materially different product, and until now nothing
    #: outside a finished submission said which one was running.
    similarity: str


#: How long a reachability answer is reused.
#:
#: The container probes health every thirty seconds. A check that spends a request
#: per probe is one somebody switches off, and then the signal is gone for the
#: reason the signal existed. Five minutes is far shorter than any outage worth
#: reporting and far longer than a probe interval.
_REACHABILITY_TTL = 300.0

_reachability: tuple[float, bool, str | None, bool] | None = None


def forget_reachability() -> None:
    """Drop the cached answer. For tests, and for a deliberate re-probe."""
    global _reachability
    _reachability = None


async def _ask_the_provider(grader) -> None:
    """One minimal request, to prove the provider answers at all.

    Deliberately not a marking call: no rubric, no student text, no schema. The
    question is whether credentials and credit are good, and the cheapest possible
    request answers it exactly as well as an expensive one.
    """
    client = getattr(grader, "_client", None)
    if client is None:
        raise RuntimeError("this grader has no client to ask")
    await client.chat.completions.create(
        model=grader.model,
        messages=[{"role": "user", "content": "ok"}],
        max_tokens=1,
    )


async def _reachable(grader) -> tuple[bool, str | None, bool]:
    """Whether the provider answered, whether it was merely throttled, cached."""
    global _reachability
    now = time.monotonic()
    if _reachability is not None and now - _reachability[0] < _REACHABILITY_TTL:
        return _reachability[1], _reachability[2], _reachability[3]

    try:
        await _ask_the_provider(grader)
    except Exception as exc:  # noqa: BLE001
        # The message, not the key. Provider errors name the account and the
        # limit — "You have no credits remaining" — which is the actionable half,
        # and they do not contain the credential.
        throttled = "ratelimit" in type(exc).__name__.lower() or "429" in str(exc)
        _reachability = (now, False, str(exc)[:200], throttled)
    else:
        _reachability = (now, True, None, False)
    return _reachability[1], _reachability[2], _reachability[3]


def _grading_readiness() -> GradingReadiness:
    """What will mark the next submission, decided without asking the provider.

    Built by the same `select_grader` the marking path uses, so this cannot report
    one answer while the pipeline takes another. No request is made: a health check
    runs every thirty seconds against a paid API, and a probe that spends money is
    a probe that gets switched off.
    """
    from . import grading

    try:
        grader = grading.select_grader()
    except grading.GraderUnavailable:
        return GradingReadiness(configured=False, engine="rubric_only", model=None)

    return GradingReadiness(
        configured=grader.name != "rubric_only",
        engine=grader.name,
        model=getattr(grader, "model", None),
    )


@app.get("/health", response_model=Health, tags=["meta"])
async def health(deep: bool = False) -> Health:
    """Readiness. ``?deep=1`` also asks the provider whether it will answer.

    Two questions, because they have different costs and different answers. Is a
    marker configured — free, safe on a probe every thirty seconds, and true as
    soon as a key exists. Does the provider answer — one request, so it happens
    only when asked for, and the answer is cached.

    The release step asks deeply, once. The container's own probe does not.
    """
    from .answers.similarity import SemanticSimilarity, default_similarity

    grading = _grading_readiness()
    if deep:
        if not grading.configured:
            # Nothing to ask, and `configured: false` has already answered it.
            grading = grading.model_copy(update={"reachable": False})
        else:
            from . import grading as grading_module

            marker = grading_module.select_grader()
            ok, detail, throttled = await _reachable(marker)
            grading = grading.model_copy(
                update={"reachable": ok, "detail": detail, "throttled": throttled}
            )

    return Health(
        status="ok",
        version=app.version,
        render_dpi=RENDER_DPI,
        hgbench_scale=HGBENCH_SCALE,
        contract_model_count=len(EXPORTED_MODELS),
        max_upload_bytes=MAX_BYTES,
        submissions_durable=get_store().durable,
        grading=grading,
        similarity=(
            "semantic" if isinstance(default_similarity, SemanticSimilarity) else "lexical"
        ),
    )


configure_logging()

app.include_router(router)
