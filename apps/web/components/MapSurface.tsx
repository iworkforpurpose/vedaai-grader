"use client";

import { useMemo, useState } from "react";
import { API_BASE } from "@/lib/api";
import type { Page, RubricPoint, Submission } from "@/lib/contracts";
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

  function select(qid: string): void {
    setSelectedQid(qid);
    setCitedPoint(null);

    const row = rows.find((r) => r.question.qid === qid);
    const boxes = row?.mapping?.highlight?.boxes ?? [];
    const page = firstPage(boxes);
    if (page !== null) {
      const first = boxes.find((b) => b.page === page);
      setScrollTarget({ page, y: first?.box.y0 ?? 0, nonce: Date.now() });
      // On the phone the sheet is a separate tab, so a selection that cannot be
      // seen is not a selection. Switching is the point of tapping a question.
      if (window.matchMedia("(max-width: 1023px)").matches) setTab("sheet");
    }
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
          aria-label="Extracted questions"
        >
          <div className="q-head">
            <h2>Extracted Questions (from question paper)</h2>
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
            {!marks.marked && (
              <>
                {" · "}
                <button
                  type="button"
                  className="feedback-cite"
                  disabled={marking}
                  onClick={() => void proposeMarks()}
                >
                  {marking ? "Marking…" : "Propose marks"}
                </button>
              </>
            )}
          </p>

          {notice && <p className="q-hint" style={{ whiteSpace: "normal" }}>{notice}</p>}

          <div className="q-list">
            {rows.map((row) => (
              <QuestionCard
                key={row.question.qid}
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
                onCite={(point) =>
                  setCitedPoint((current) =>
                    current?.point_id === point.point_id ? null : point,
                  )
                }
              />
            ))}
          </div>
        </section>

        <section
          className="sheet-pane map-pane"
          data-active={tab === "sheet"}
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
              onPointerPick={pickAtPoint}
              scrollTarget={scrollTarget}
            />
          )}
        </section>
      </div>

      {/* Kept because it is how a teacher answers "the question says not found —
          where is the writing then?", which the pipeline can only point at. */}
      {ink.size > 0 && (
        <label className="q-hint" style={{ whiteSpace: "normal", cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={showInk}
            onChange={(event) => setShowInk(event.target.checked)}
          />{" "}
          Show writing the recognizer could not read
        </label>
      )}
    </div>
  );
}

export { applyReassignment };
