"""HTTP surface.

Note on uploads: the browser posts files to this service directly, not through
the Next.js app. That matters because a Vercel function caps its request body at
4.5 MB, which a scanned answer sheet routinely exceeds — but this service runs on
its own host, so the cap never applies and presigned object-storage uploads
become an optimization rather than a requirement.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Depends, File, HTTPException, Path, UploadFile
from fastapi.responses import Response, StreamingResponse
from vedaai_contracts import (
    DocumentKind,
    InkRegion,
    LineIndex,
    Submission,
    SubmissionStatus,
)

from . import align, grading, pipeline, regions, render
from .render import UnsupportedDocument
from .storage import AnyPageStore, get_page_store
from .store import SubmissionStore, get_store

router = APIRouter()

StoreDep = Annotated[SubmissionStore, Depends(get_store)]
PageStoreDep = Annotated[AnyPageStore, Depends(get_page_store)]


@router.post("/submissions", response_model=Submission, tags=["submissions"])
async def create_submission(
    store: StoreDep,
    page_store: PageStoreDep,
    question_paper: Annotated[UploadFile, File(description="Printed question paper.")],
    answer_sheet: Annotated[UploadFile, File(description="One student's handwritten sheet.")],
) -> Submission:
    """Accept both documents and start ingest. Returns immediately.

    Ingest used to be awaited here, on the reasoning that a request taking a few
    seconds is simpler than a job handle. That reasoning had a stated expiry —
    "when transcription grows to a minute-plus this moves to a background task" —
    and it expired in a way worth recording, because it did not look like a
    timeout.

    Deployed behind the Next proxy, every upload failed with a 500 at almost
    exactly thirty seconds. Not an exception: no traceback, no completed request in
    the log, the worker still healthy and never restarted. A one-page sheet failed
    at 30.7s and a two-page at 32.0s, which is what said it was a wall rather than
    a resource limit — the same wall regardless of the work behind it.

    Raising whichever timeout it was would only move the wall. A pipeline that
    takes fifteen seconds a page cannot live inside one HTTP request whatever the
    proxy allows, and every layer between browser and worker gets to impose its own
    limit. So the response is now the submission, immediately, in `processing`; the
    work continues behind it; and the client follows the status it already knows
    how to render.

    The upload is still validated synchronously, so a file that cannot be read is
    still a 422 on this request rather than a job that fails later.
    """
    qp_bytes = await question_paper.read()
    as_bytes = await answer_sheet.read()

    try:
        qp_source = render.inspect(
            qp_bytes, question_paper.filename or "question_paper", DocumentKind.QUESTION_PAPER
        )
        as_source = render.inspect(
            as_bytes, answer_sheet.filename or "answer_sheet", DocumentKind.ANSWER_SHEET
        )
    except UnsupportedDocument as exc:
        # A rejected upload is the user's problem to fix, so the message says
        # what was wrong rather than surfacing a parser exception.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    submission_id = uuid.uuid4().hex[:12]
    accepted = Submission(submission_id=submission_id, status=SubmissionStatus.PROCESSING)
    store.put(accepted)
    store.remember_content(qp_source.content_hash, submission_id)

    task = asyncio.create_task(
        _run_ingest(
            submission_id=submission_id,
            question_paper=(qp_bytes, qp_source),
            answer_sheet=(as_bytes, as_source),
            page_store=page_store,
            store=store,
        )
    )
    # Held, because asyncio keeps only a weak reference to a running task and will
    # happily collect one mid-flight — which would abandon the ingest silently.
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)

    return accepted


#: Strong references to in-flight ingests. See the note where they are added.
_BACKGROUND: set[asyncio.Task] = set()


async def _run_ingest(
    *,
    submission_id: str,
    question_paper: tuple[bytes, object],
    answer_sheet: tuple[bytes, object],
    page_store: AnyPageStore,
    store: SubmissionStore,
) -> None:
    """Render, transcribe, map and mark, off the request.

    Every failure ends as a stored `failed` submission carrying the reason, never
    as an exception nobody sees. There is no request left to return a 500 to, so a
    swallowed error here would present as a page that waits forever.
    """
    loop = asyncio.get_running_loop()
    try:
        submission = await loop.run_in_executor(
            None,
            lambda: pipeline.ingest(
                submission_id=submission_id,
                question_paper=question_paper,
                answer_sheet=answer_sheet,
                page_store=page_store,
                submission_store=store,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        failed = store.get(submission_id) or Submission(submission_id=submission_id)
        failed.status = SubmissionStatus.FAILED
        reason = f"Ingest failed: {exc}"
        if reason not in failed.warnings:
            failed.warnings.append(reason)
        store.put(failed)
        return

    # Marks up front, so the review screen opens with feedback rather than a
    # button. Guarded because marking is the one step that talks to a paid API:
    # a failure here must cost the located answers nothing, and the submission is
    # already stored and returnable without it.
    marking = _auto_mark_enabled() and (
        submission.mapping is not None
        and submission.questions is not None
        and submission.answer_sheet_lines is not None
    )

    if marking:
        # Held at `processing` across marking, because `pipeline.ingest` sets
        # `complete` itself the moment locating is done.
        #
        # That is a seam a client cannot see past: the submission says complete
        # while eight marking calls are still in flight, so anything watching the
        # status opens the review screen with no marks on it and no reason to look
        # again. It read as auto-marking silently not running — the marks arrived
        # perfectly well, roughly half a minute after the only signal saying to
        # stop waiting.
        submission.status = SubmissionStatus.PROCESSING
        store.put(submission)
        try:
            await _apply_marks(submission)
        except BaseException as exc:  # noqa: BLE001
            # BaseException, not Exception: a cancelled task raises CancelledError,
            # which is not an Exception, and losing it here would leave a
            # submission stuck at `processing` with nothing said about why.
            warning = f"Answers were not marked automatically: {exc}"
            if warning not in submission.warnings:
                submission.warnings.append(warning)
        submission.status = SubmissionStatus.COMPLETE

    store.put(submission)


def _auto_mark_enabled() -> bool:
    """Whether ingest marks without being asked. On unless switched off.

    Env-gated because marking is per-question paid API traffic: a deployment that
    wants locating without that bill sets ``AUTO_MARK=0`` and the endpoint above
    still works on request.
    """
    return os.environ.get("AUTO_MARK", "1").strip().lower() not in {"0", "false", "no", "off"}


@router.get("/submissions/{submission_id}", response_model=Submission, tags=["submissions"])
def get_submission(submission_id: str, store: StoreDep) -> Submission:
    submission = store.get(submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail=f"No submission {submission_id!r}")
    return submission


@router.get(
    "/submissions/{submission_id}/lines/{kind}",
    response_model=LineIndex,
    tags=["submissions"],
)
def get_lines(submission_id: str, kind: DocumentKind, store: StoreDep) -> LineIndex:
    """The line index for one document.

    This is what the debug overlay draws. Every box it renders came from the
    transcription engine, so if a highlight later lands in the wrong place, this
    endpoint answers whether the geometry or the mapping is at fault.
    """
    submission = store.get(submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail=f"No submission {submission_id!r}")

    index = (
        submission.question_paper_lines
        if kind is DocumentKind.QUESTION_PAPER
        else submission.answer_sheet_lines
    )
    if index is None:
        raise HTTPException(
            status_code=404,
            detail=f"No transcription for the {kind.value.replace('_', ' ')}. "
            + "; ".join(submission.warnings),
        )
    return index


@router.get(
    "/submissions/{submission_id}/ink",
    response_model=list[InkRegion],
    tags=["submissions"],
)
def get_ink_regions(submission_id: str, store: StoreDep) -> list[InkRegion]:
    """Ink regions for the answer sheet.

    The second geometry source, and the answer to two things transcription
    cannot report: where a diagram is, and where the recognizer missed a line
    that is nonetheless covered in ink.
    """
    submission = store.get(submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail=f"No submission {submission_id!r}")
    return submission.ink_regions


@router.patch(
    "/submissions/{submission_id}/mapping/{qid:path}",
    response_model=Submission,
    tags=["submissions"],
)
def reassign_answer(
    submission_id: str,
    qid: str,
    store: StoreDep,
    block_id: Annotated[str, Body(embed=True, description="Block to move.")],
) -> Submission:
    """Move a region of writing to a different question.

    Manual correction is expected rather than exceptional. Gradescope, the
    established tool for this task, does not locate answer regions automatically
    at all — students mark their own, or a pre-printed template is required — and
    it ships an explicit tool for instructors to correct regions. This endpoint is
    that tool.
    """
    submission = store.get(submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail=f"No submission {submission_id!r}")
    if submission.mapping is None or submission.questions is None:
        raise HTTPException(
            status_code=409,
            detail="This submission has no mapping yet, so there is nothing to reassign.",
        )
    if all(block.block_id != block_id for block in submission.blocks):
        raise HTTPException(status_code=404, detail=f"No block {block_id!r}")
    if all(question.qid != qid for question in submission.questions.questions):
        raise HTTPException(status_code=404, detail=f"No question {qid!r}")

    submission.mapping = align.reassign(
        submission.questions,
        submission.blocks,
        submission.mapping,
        block_id=block_id,
        to_qid=qid,
    )
    store.put(submission)
    return submission


@router.post(
    "/submissions/{submission_id}/grades",
    response_model=Submission,
    tags=["submissions"],
)
async def grade_submission(submission_id: str, store: StoreDep) -> Submission:
    """Propose marks for a submission, citing the lines behind each one.

    Ingest now marks on its own (see ``AUTO_MARK``), so this endpoint is the
    re-run: it is what a teacher uses after correcting a mapping, and the entry
    point when auto-marking is switched off for a deployment.

    It was deliberately *not* part of ingest, on the reasoning that a proposed
    score is hard to unsee and locating answers is useful without it. Overruled on
    request — a teacher opening a marked script and choosing to ignore the numbers
    is a smaller cost than one who never finds the button. The numbers are still
    labelled as proposals and every one carries the line it rests on.

    Without a grading key this still succeeds. It returns the rubric and
    the located answer with every point unjudged, which is a marking aid rather
    than a grade. Inventing a score from keyword overlap would be worse than
    offering none: a plausible wrong mark is the error a teacher is least likely
    to catch.
    """
    submission = store.get(submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail=f"No submission {submission_id!r}")
    if submission.mapping is None or submission.questions is None:
        raise HTTPException(
            status_code=409,
            detail="This submission has no mapping yet, so there are no answers to mark.",
        )
    if submission.answer_sheet_lines is None:
        raise HTTPException(
            status_code=409,
            detail="The answer sheet was never transcribed, so there is no text to mark.",
        )

    await _apply_marks(submission)
    store.put(submission)
    return submission


async def _apply_marks(submission: Submission) -> None:
    """Mark a submission in place, degrading to the rubric when no grader is set.

    Shared by the explicit endpoint and by ingest, so the two cannot diverge in
    what they exclude or how they report a failure.

    Caller checks the preconditions; this assumes a mapping, questions and a
    transcribed answer sheet are present.
    """
    assert submission.questions is not None
    assert submission.mapping is not None
    assert submission.answer_sheet_lines is not None

    try:
        grader: grading.Grader = grading.select_grader()
    except grading.GraderUnavailable as unavailable:
        grader = grading.RubricOnly()
        warning = f"Answers were not marked automatically: {unavailable}"
        if warning not in submission.warnings:
            submission.warnings.append(warning)

    # Computed here rather than stored, so a re-run picks up any change to region
    # classification. The set is small and the calculation is geometric.
    excluded = regions.lines_excluded_from_grading(
        submission.ink_regions, submission.answer_sheet_lines.lines
    )

    submission.grades, marking_failures = await grading.grade_submission(
        paper=submission.questions,
        mapping=submission.mapping,
        index=submission.answer_sheet_lines,
        grader=grader,
        excluded_line_ids=excluded,
    )
    for failure in marking_failures:
        if failure not in submission.warnings:
            submission.warnings.append(failure)


@router.get("/pages/{key:path}", tags=["pages"])
def get_page_image(
    page_store: PageStoreDep,
    key: Annotated[str, Path(description="Page image key from Page.image_key.")],
) -> Response:
    """Serve a rendered page image.

    Cached hard: page images are content-addressed, so a given key's bytes never
    change and the browser can keep them for as long as it likes. The review
    surface scrolls through every page of a script, and re-fetching them on each
    interaction would make the overlay feel broken.
    """
    try:
        if not page_store.exists(key):
            raise HTTPException(status_code=404, detail=f"No page image {key!r}")
        data = page_store.read(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/submissions/{submission_id}/events", tags=["submissions"])
async def stream_events(submission_id: str, store: StoreDep) -> StreamingResponse:
    """Server-sent events for pipeline progress.

    Replays from the beginning on connect, so a browser that reconnects
    mid-processing sees what it missed rather than resuming blind. A keepalive
    comment goes out whenever a stage runs quiet, because proxies drop idle
    connections and transcription legitimately produces nothing for tens of
    seconds at a time.
    """
    if store.get(submission_id) is None:
        raise HTTPException(status_code=404, detail=f"No submission {submission_id!r}")

    async def generate():
        cursor = 0
        while True:
            events, cursor = store.events_since(submission_id, cursor)
            for event in events:
                yield f"data: {json.dumps(event.model_dump(mode='json'))}\n\n"
                if event.is_terminal:
                    return
            if store.is_finished(submission_id):
                return
            await store.wait_for_change(submission_id)
            if not events:
                yield ": keepalive\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Nginx and several PaaS proxies buffer responses by default, which
            # would hold every event until the stream closed.
            "X-Accel-Buffering": "no",
        },
    )
