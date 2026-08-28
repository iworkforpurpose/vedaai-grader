"use client";

import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "@/lib/api";
import type { Page, PageBox, RubricPoint, Submission } from "@/lib/contracts";
import { firstPage } from "@/lib/geometry";
import {
  applyReassignment,
  buildRows,
  citationHighlight,
  gradeFor,
  highlightByPage,
  questionAtPoint,
  summarize,
  summarizeMarks,
  untranscribedInkByPage,
} from "@/lib/review";
import { useNarrow } from "@/lib/breakpoints";
import { LoadingStage } from "./LoadingStage";
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
      if (window.matchMedia("(max-width: 1023px)").matches) setTab("questions");
      return;
    }
    setNotice("No answer is mapped to that spot.");
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

  if (submission.status === "processing") {
    return <LoadingStage detail="Rendering pages and reading the handwriting." />;
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
