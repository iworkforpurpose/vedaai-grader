"use client";

import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "@/lib/api";
import type { Page, PageBox, RubricPoint, Submission } from "@/lib/contracts";
import { firstPage } from "@/lib/geometry";
import {
  applyReassignment,
  buildRows,
  isTeacherPlaced,
  movableBlocks,
  citationHighlight,
  gradeFor,
  highlightByPage,
  questionAtPoint,
  summarize,
  summarizeMarks,
  untranscribedInkByPage,
} from "@/lib/review";
import { useNarrow } from "@/lib/breakpoints";
import { crossFade } from "@/lib/transitions";
import { LoadingStage } from "./LoadingStage";
import { ProcessingStage } from "./ProcessingStage";
import { QuestionCard } from "./QuestionCard";
import { SheetView } from "./SheetView";

/**
 * The question-to-answer mapping screen.
 *
 * Questions on the left, the answer sheet on the right, and clicking a question
 * highlights where it was answered. Below the rail breakpoint the two become tabs
 * rather than a split, which is what the phone frames show — and the right call,
 * since a side-by-side split at 393px gave the sheet 73px and made the highlight
 * impossible to see.
 *
 * All the logic lives in lib/review.ts with no React and no DOM, which is why this
 * component could be rewritten to the frames without touching geometry,
 * hit-testing or mapping. The parts that have to be right regardless of appearance
 * are tested there, 41 of them.
 */

type Tab = "questions" | "sheet";

export function MapSurface({ initial }: { initial: Submission }): React.JSX.Element {
  const [submission, setSubmission] = useState(initial);
  const [selectedQid, setSelectedQid] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [citedPoint, setCitedPoint] = useState<RubricPoint | null>(null);
  const [scrollTarget, setScrollTarget] = useState<{
    page: number;
    y: number;
    nonce: number;
  } | null>(null);
  const [tab, setTab] = useState<Tab>("questions");
  const [marking, setMarking] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const narrow = useNarrow();

  /*
   * The block a teacher is currently finding a home for, if any.
   *
   * One piece of state drives the whole correction flow, because there is only ever
   * one thing being placed. Holding it here rather than in the card means the
   * question list can become a target list without every card needing to know about
   * every other one.
   */
  const [placing, setPlacing] = useState<{ blockId: string; from: string | null } | null>(
    null,
  );
  const [moving, setMoving] = useState(false);

  /*
   * Follow a submission that is still being worked on.
   *
   * The upload used to return the finished submission, so the first render was
   * always the final one. It now returns immediately in `processing`, because a
   * pipeline measured in tens of seconds per page cannot live inside one HTTP
   * request — every layer between the browser and the worker gets to impose its own
   * timeout, and one of them was cutting the connection at thirty seconds.
   *
   * Polling rather than the SSE progress stream: this screen needs the whole
   * submission to render, which is one request either way, and a two-second poll
   * against a job measured in tens of seconds costs nothing. The event stream stays
   * the right source if this ever shows per-page progress.
   */
  useEffect(() => {
    if (submission.status !== "processing") return;

    let live = true;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const response = await fetch(
            `${API_BASE}/submissions/${submission.submission_id}`,
            { cache: "no-store" },
          );
          if (!response.ok || !live) return;
          const next = (await response.json()) as Submission;
          // The waiting screen giving way to the mapping screen, which is the
          // other whole-screen swap in the app.
          if (live && next.status !== "processing") crossFade(() => setSubmission(next));
        } catch {
          // A dropped poll is not a failure — the next one is two seconds away.
        }
      })();
    }, 2000);

    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [submission.status, submission.submission_id]);

  const rows = useMemo(() => buildRows(submission), [submission]);
  const summary = useMemo(() => summarize(submission, rows), [submission, rows]);
  const marks = useMemo(() => summarizeMarks(submission), [submission]);
  const selected = rows.find((r) => r.question.qid === selectedQid) ?? null;
  const ink = useMemo(() => untranscribedInkByPage(submission), [submission]);
  const [showInk, setShowInk] = useState(false);

  // A cited rubric point narrows the highlight to the lines behind that one mark.
  // Without it the whole answer lights up and the teacher still has to find the
  // sentence — which is the work the citation exists to save.
  const highlights = useMemo(
    () =>
      citedPoint
        ? citationHighlight(submission, citedPoint.cited_line_ids)
        : highlightByPage(selected?.mapping),
    [citedPoint, submission, selected],
  );

  const answerPages: Page[] = useMemo(
    () =>
      submission.pages.filter((p) => p.kind === "answer_sheet").sort((a, b) => a.index - b.index),
    [submission.pages],
  );

  /*
   * Bring a region into view: scroll to it, and on the phone switch to the tab it
   * is on.
   *
   * One function because there are two ways to ask for a region — picking a
   * question, and following a citation — and only the first had this. Clicking
   * "show the writing this rests on" moved the highlight and left the reader
   * looking at wherever they already were, which on the phone was the questions
   * tab, so the entire response to the click was invisible. Sharing the path is
   * what stops the two drifting apart again.
   */
  function revealOnSheet(boxes: readonly PageBox[]): void {
    const page = firstPage(boxes);
    if (page === null) return;
    const first = boxes.find((box) => box.page === page);
    // The nonce makes asking twice for the same region scroll twice, which is what
    // someone clicking again is asking for.
    setScrollTarget({ page, y: first?.box.y0 ?? 0, nonce: Date.now() });
    if (narrow) setTab("sheet");
  }

  function select(qid: string): void {
    setSelectedQid(qid);
    setCitedPoint(null);
    const row = rows.find((r) => r.question.qid === qid);
    revealOnSheet(row?.mapping?.highlight?.boxes ?? []);
  }

  /*
   * Follow a citation: narrow the highlight to the lines behind one mark, and go
   * there.
   *
   * Clicking the same citation again clears it and returns the highlight to the
   * whole answer, so it is a toggle — and in that direction there is nothing new
   * to scroll to.
   */
  function cite(point: RubricPoint): void {
    const clearing = citedPoint?.point_id === point.point_id;
    setCitedPoint(clearing ? null : point);
    if (clearing) return;
    const boxes = [...citationHighlight(submission, point.cited_line_ids).values()].flat();
    revealOnSheet(boxes);
  }

  function pickAtPoint(page: number, x: number, y: number): void {
    // Reverse lookup: click a region on the sheet, jump to its question. A teacher
    // reading the script wants to ask "what is this an answer to", not only "where
    // is this answer".
    const hit = questionAtPoint(rows, page, x, y);
    if (hit) {
      setSelectedQid(hit.question.qid);
      setCitedPoint(null);
      setNotice(null);
      if (narrow) setTab("questions");
      return;
    }
    setNotice("No answer is mapped to that spot.");
  }

  /*
   * Move a block of writing to a question, and keep the screen honest if it fails.
   *
   * Applied locally first, because the correction is the teacher's decision and
   * should land the moment they make it — a round trip of a second on a list they
   * are actively working through reads as the interface ignoring them. The local
   * apply mirrors the server's rules exactly, which is why it lives in
   * `lib/review.ts` and is tested there rather than being written twice.
   *
   * On failure the server's response replaces it. That matters more than it looks:
   * this is a correction to a mark a student receives, so a change that silently
   * did not persist is worse than one that visibly refused.
   */
  async function place(toQid: string): Promise<void> {
    const target = placing;
    if (target === null) return;

    const optimistic = applyReassignment(submission, target.blockId, toQid);
    setSubmission(optimistic);
    setPlacing(null);
    setMoving(true);
    setNotice(null);

    try {
      const response = await fetch(
        `${API_BASE}/submissions/${submission.submission_id}/mapping/${toQid}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ block_id: target.blockId }),
        },
      );
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { detail?: string } | null;
        setSubmission(submission);
        setNotice(body?.detail ?? "That move could not be saved, so it has been undone.");
        return;
      }
      setSubmission((await response.json()) as Submission);
      setSelectedQid(toQid);
    } catch {
      setSubmission(submission);
      setNotice("That move could not be saved — the service could not be reached.");
    } finally {
      setMoving(false);
    }
  }

  /*
   * Record what a teacher says an answer is worth.
   *
   * Not applied optimistically, unlike a move. A move is the teacher restating
   * something they can see on the page and the server cannot refuse on grounds
   * they do not already know; a mark can be refused for carrying more than the
   * paper prints, and showing it as accepted first would mean showing a total
   * that never existed. The round trip is one field on one question, so the wait
   * is short and the number that appears is the number that was stored.
   */
  async function correctMark(qid: string, marks: number | null): Promise<void> {
    setNotice(null);
    try {
      const response = await fetch(
        `${API_BASE}/submissions/${submission.submission_id}/grades/${encodeURIComponent(qid)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ marks }),
        },
      );
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { detail?: string } | null;
        setNotice(body?.detail ?? "That mark could not be saved.");
        return;
      }
      setSubmission((await response.json()) as Submission);
    } catch {
      setNotice("That mark could not be saved — the service could not be reached.");
    }
  }

  async function proposeMarks(): Promise<void> {
    setMarking(true);
    setNotice(null);
    try {
      const response = await fetch(
        `${API_BASE}/submissions/${submission.submission_id}/grades`,
        { method: "POST" },
      );
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { detail?: string } | null;
        setNotice(body?.detail ?? "Marking could not be run.");
        return;
      }
      setSubmission((await response.json()) as Submission);
    } catch {
      setNotice("Marking could not be run — the service could not be reached.");
    } finally {
      setMarking(false);
    }
  }

  const allExpanded = expanded.size >= rows.length && rows.length > 0;

  if (submission.status === "failed") {
    /*
     * `error` before `warnings`, because it is the more specific field and the
     * one a submission fills in when it knows exactly what went wrong. A
     * submission the service was restarted mid-read says so there; without this
     * it fell through to the generic card and a tester saw "could not be read"
     * about a document that was perfectly readable.
     */
    return (
      <LoadingStage
        title="This one could not be read"
        note="Nothing was kept — try uploading again."
        detail={submission.error ?? submission.warnings[0]}
      />
    );
  }

  if (submission.status === "processing") {
    return <ProcessingStage submissionId={submission.submission_id} />;
  }

  return (
    <div className="map">
      <div className="map-tabs" role="tablist" aria-label="Mapping view">
        <button
          type="button"
          className="map-tab"
          role="tab"
          aria-selected={tab === "questions"}
          onClick={() => setTab("questions")}
        >
          Questions
        </button>
        <button
          type="button"
          className="map-tab"
          role="tab"
          aria-selected={tab === "sheet"}
          onClick={() => setTab("sheet")}
        >
          Answer Sheet
        </button>
      </div>

      <div className="map-panes">
        <section
          className="q-pane map-pane"
          data-active={tab === "questions"}
          /* Below the rail breakpoint the panes are stacked and crossfaded rather
             than swapped with `display`, which cannot transition. `inert` keeps the
             faded-out pane out of the tab order. */
          inert={tab !== "questions" && narrow}
          aria-label="Extracted questions"
        >
          <div className="q-head">
            <h2>Extracted Questions (from question paper)</h2>

            {/*
              * Marking is a button here, not a link buried in the summary line.
              *
              * It was the latter, and it read as prose: the one control that puts
              * scores and feedback on the screen was the hardest thing on the
              * screen to find. Marking now also runs at ingest, so this is the
              * re-run — which is what a teacher needs after moving an answer
              * between questions, since the old mark was made against the old
              * mapping.
              */}
            {/* Grouped so the pair wraps together and stays right-aligned. Left
                loose, the heading's own wrap dropped one button onto its own row
                against the left edge. */}
            <div className="q-head-actions">
              <button
                type="button"
                className="q-head-action"
                data-primary="true"
                disabled={marking}
                onClick={() => void proposeMarks()}
              >
                {marking ? "Marking…" : marks.marked ? "Re-mark" : "Mark answers"}
              </button>

              <button
                type="button"
                className="q-head-action"
                onClick={() =>
                  setExpanded(allExpanded ? new Set() : new Set(rows.map((r) => r.question.qid)))
                }
              >
                {allExpanded ? "Collapse All" : "Expand All"}
              </button>
            </div>
          </div>

          {/*
            * While placing, the list means something different, so the header says
            * so and offers the way out. Without it a teacher has a list of cards
            * that suddenly respond differently and nothing explaining why.
            */}
          {placing !== null && (
            <div className="placing-banner" role="status">
              <span>
                Choose the question this writing belongs to
                {placing.from && <> · currently on <strong>{placing.from}</strong></>}
              </span>
              <button type="button" className="placing-cancel" onClick={() => setPlacing(null)}>
                Cancel
              </button>
            </div>
          )}

          <p className="q-hint" style={{ whiteSpace: "normal" }}>
            <strong>{summary.answered}</strong> of {summary.total} answered
            {summary.absenceSuppressed ? (
              <> · unanswered count withheld, some writing could not be placed</>
            ) : (
              <>
                {" "}
                · <strong>{summary.notAnswered}</strong> not answered
              </>
            )}
            {summary.needsAttention > 0 && <> · {summary.needsAttention} to check</>}
            {marks.marked && !marks.rubricOnly && (
              <>
                {" "}
                · <strong>{marks.awarded}</strong>/{marks.available} proposed
              </>
            )}
            {marks.rubricOnly && <> · marks not proposed, rubric only</>}
          </p>

          {notice && <p className="q-hint" style={{ whiteSpace: "normal" }}>{notice}</p>}

          <div className="q-list">
            {rows.map((row, index) => (
              <QuestionCard
                key={row.question.qid}
                /* Caps the stagger: past a dozen, the last card would wait most of
                   a second to appear, which reads as slow loading rather than as
                   arrival. */
                index={Math.min(index, 12)}
                row={row}
                grade={gradeFor(submission, row.question.qid)}
                selected={row.question.qid === selectedQid}
                expanded={expanded.has(row.question.qid)}
                onSelect={() => select(row.question.qid)}
                onToggle={() =>
                  setExpanded((current) => {
                    const next = new Set(current);
                    if (next.has(row.question.qid)) next.delete(row.question.qid);
                    else next.add(row.question.qid);
                    return next;
                  })
                }
                onCite={cite}
                placingActive={placing !== null}
                teacherPlaced={isTeacherPlaced(submission, row.question.qid)}
                movable={movableBlocks(submission, row.question.qid)}
                onPlaceHere={() => void place(row.question.qid)}
                onMoveBlock={(blockId) =>
                  setPlacing({ blockId, from: row.question.label_raw })
                }
                onCorrectMark={(marks) => correctMark(row.question.qid, marks)}
              />
            ))}
          </div>
        </section>

        <section
          className="sheet-pane map-pane"
          data-active={tab === "sheet"}
          inert={tab !== "sheet" && narrow}
          aria-label="Answer sheet"
        >
          {answerPages.length === 0 ? (
            <LoadingStage
              title="No pages"
              note="The answer sheet produced no page images."
            />
          ) : (
            <SheetView
              pages={answerPages}
              highlights={highlights}
              highlightLabel={selected ? selected.question.label_raw.replace(/[.)]\s*$/, "") : null}
              untranscribedInk={ink}
              showUntranscribed={showInk}
              onToggleUntranscribed={() => setShowInk((on) => !on)}
              onPointerPick={pickAtPoint}
              scrollTarget={scrollTarget}
            />
          )}
        </section>
      </div>

    </div>
  );
}

export { applyReassignment };
