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
import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Depends, File, HTTPException, Path, UploadFile
from fastapi.responses import Response, StreamingResponse
from vedaai_contracts import DocumentKind, InkRegion, LineIndex, Submission

from . import align, grading, pipeline, regions, render
from .render import UnsupportedDocument
from .storage import PageStore, get_page_store
from .store import SubmissionStore, get_store

router = APIRouter()

StoreDep = Annotated[SubmissionStore, Depends(get_store)]
PageStoreDep = Annotated[PageStore, Depends(get_page_store)]


@router.post("/submissions", response_model=Submission, tags=["submissions"])
async def create_submission(
    store: StoreDep,
    page_store: PageStoreDep,
    question_paper: Annotated[UploadFile, File(description="Printed question paper.")],
    answer_sheet: Annotated[UploadFile, File(description="One student's handwritten sheet.")],
) -> Submission:
    """Upload both documents and run ingest.

    Ingest is awaited rather than backgrounded. At this scale a request that
    takes a few seconds is simpler and more debuggable than a job handle, and
    the SSE stream still carries per-page progress for the client. When
    transcription grows to a minute-plus this moves to a background task and the
    response becomes the submission ID alone.
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
    store.put(Submission(submission_id=submission_id))

    loop = asyncio.get_running_loop()
    try:
        submission = await loop.run_in_executor(
            None,
            lambda: pipeline.ingest(
                submission_id=submission_id,
                question_paper=(qp_bytes, qp_source),
                answer_sheet=(as_bytes, as_source),
                page_store=page_store,
                submission_store=store,
            ),
        )
    except UnsupportedDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    store.remember_content(qp_source.content_hash, submission_id)
    return submission


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

    Explicitly requested rather than part of ingestion, for two reasons. Locating
    answers is useful on its own and must not wait behind marking, and marking is
    the one step whose output a teacher may not want at all — a proposed score is
    hard to unsee once shown.

    Without ``ANTHROPIC_API_KEY`` this still succeeds. It returns the rubric and
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

    try:
        grader: grading.Grader = grading.Claude()
    except grading.ClaudeUnavailable as unavailable:
        grader = grading.RubricOnly()
        warning = f"Answers were not marked automatically: {unavailable}"
        if warning not in submission.warnings:
            submission.warnings.append(warning)

    # Computed here rather than stored, so a re-run picks up any change to region
    # classification. The set is small and the calculation is geometric.
    excluded = regions.lines_excluded_from_grading(
        submission.ink_regions, submission.answer_sheet_lines.lines
    )

    submission.grades = await grading.grade_submission(
        paper=submission.questions,
        mapping=submission.mapping,
        index=submission.answer_sheet_lines,
        grader=grader,
        excluded_line_ids=excluded,
    )
    store.put(submission)
    return submission


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
