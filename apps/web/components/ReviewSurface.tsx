"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import type { Page, Submission } from "@/lib/contracts";
import {
  applyReassignment,
  blockPreview,
  blocksOf,
  buildRows,
  highlightByPage,
  questionAtPoint,
  summarize,
  untranscribedInkByPage,
} from "@/lib/review";
import { firstPage } from "@/lib/geometry";
import { AnswerSheetView } from "./AnswerSheetView";
import { QuestionList } from "./QuestionList";
import { StatusChip } from "./StatusChip";

/**
 * The teacher's view: questions on the left, the answer sheet on the right.
 *
 * Visually plain on purpose. Every interaction works and is testable; the real
 * designs are applied in a later pass, which is why the layout reads tokens and
 * the logic lives in `lib/review` rather than in here.
 */
export function ReviewSurface({
  initial,
}: {
  initial: Submission;
}): React.JSX.Element {
  const [submission, setSubmission] = useState(initial);
  const [selectedQid, setSelectedQid] = useState<string | null>(null);
  const [scrollTarget, setScrollTarget] = useState<{
    page: number;
    y: number;
    nonce: number;
  } | null>(null);
  const [reassignBlock, setReassignBlock] = useState<string | null>(null);
  const [showUntranscribed, setShowUntranscribed] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const rows = useMemo(() => buildRows(submission), [submission]);
  const summary = useMemo(() => summarize(submission, rows), [submission, rows]);
  const selected = rows.find((r) => r.question.qid === selectedQid) ?? null;
  const highlights = useMemo(() => highlightByPage(selected?.mapping), [selected]);
  const ink = useMemo(() => untranscribedInkByPage(submission), [submission]);
  const selectedBlocks = useMemo(
    () => blocksOf(submission, selected?.mapping ?? undefined),
    [submission, selected],
  );

  const answerPages: Page[] = useMemo(
    () =>
      submission.pages
        .filter((p) => p.kind === "answer_sheet")
        .sort((a, b) => a.index - b.index),
    [submission.pages],
  );

  function select(qid: string): void {
    setSelectedQid(qid);
    const row = rows.find((r) => r.question.qid === qid);
    const boxes = row?.mapping?.highlight?.boxes ?? [];
    const page = firstPage(boxes);
    if (page !== null) {
      const first = boxes.find((b) => b.page === page);
      setScrollTarget({ page, y: first?.box.y0 ?? 0, nonce: Date.now() });
    }
  }

  function pickAtPoint(page: number, x: number, y: number): void {
    // Reverse lookup: click a region on the sheet, jump to its question. The
    // inverse direction matters because a teacher reading the script wants to
    // ask "what is this an answer to", not only "where is this answer".
    const hit = questionAtPoint(rows, page, x, y);
    if (hit) {
      setSelectedQid(hit.question.qid);
      setNotice(null);
      return;
    }
    setNotice("No answer is mapped to that spot.");
  }

  async function reassign(toQid: string): Promise<void> {
    const blockId = reassignBlock;
    if (blockId === null) return;

    setSubmission((current) => applyReassignment(current, blockId, toQid));
    setReassignBlock(null);
    setSelectedQid(toQid);

    try {
      const response = await fetch(
        `${API_BASE}/submissions/${submission.submission_id}/mapping/${encodeURIComponent(toQid)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ block_id: blockId }),
        },
      );
      if (!response.ok) {
        setNotice("The change was not saved. Reload to see the stored mapping.");
        return;
      }
      // The server's version is authoritative — it recomputes geometry and the
      // statuses of whichever questions the move affected.
      setSubmission((await response.json()) as Submission);
      setNotice(null);
    } catch {
      setNotice("The change was not saved — the service could not be reached.");
    }
  }

  const orphans = submission.mapping?.orphans ?? [];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--sp-4)",
          flexWrap: "wrap",
          padding: "var(--sp-3) var(--sp-5)",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface)",
        }}
      >
        <strong style={{ fontSize: "var(--fs-lg)" }}>
          {submission.answer_sheet_file?.filename ?? "Answer sheet"}
        </strong>

        <span style={{ display: "flex", gap: "var(--sp-4)", fontSize: "var(--fs-sm)" }}>
          <span>
            <strong>{summary.answered}</strong> of {summary.total} answered
          </span>
          {summary.absenceSuppressed ? (
            <span style={{ color: "var(--status-review)" }}>
              Unanswered count withheld — some writing could not be placed
            </span>
          ) : (
            <span>
              <strong>{summary.notAnswered}</strong> not answered
            </span>
          )}
          {summary.needsAttention > 0 && (
            <span style={{ color: "var(--status-review)" }}>
              {summary.needsAttention} need checking
            </span>
          )}
        </span>

        <label
          style={{
            marginLeft: "auto",
            display: "flex",
            alignItems: "center",
            gap: "var(--sp-2)",
            fontSize: "var(--fs-sm)",
          }}
        >
          <input
            type="checkbox"
            checked={showUntranscribed}
            onChange={(e) => setShowUntranscribed(e.target.checked)}
          />
          Show unread writing
        </label>
        <Link
          href={`/review/${submission.submission_id}/inspect`}
          style={{ fontSize: "var(--fs-sm)" }}
        >
          Geometry inspector
        </Link>
      </header>

      {(submission.warnings.length > 0 || notice) && (
        <div
          style={{
            padding: "var(--sp-2) var(--sp-5)",
            borderBottom: "1px solid var(--border)",
            background: "var(--surface)",
            fontSize: "var(--fs-sm)",
            color: "var(--text-2)",
          }}
        >
          {notice && <p style={{ margin: 0, color: "var(--status-review)" }}>{notice}</p>}
          {submission.warnings.map((warning) => (
            <p key={warning} style={{ margin: 0 }}>
              {warning}
            </p>
          ))}
        </div>
      )}

      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <aside
          style={{
            width: "38%",
            minWidth: 320,
            maxWidth: 560,
            borderRight: "1px solid var(--border)",
            overflowY: "auto",
            background: "var(--surface)",
          }}
        >
          {reassignBlock !== null && (
            <div
              style={{
                padding: "var(--sp-3) var(--sp-4)",
                background: "var(--surface-2)",
                borderBottom: "1px solid var(--border)",
                fontSize: "var(--fs-sm)",
              }}
            >
              Choose the question this answer belongs to.{" "}
              <button
                type="button"
                onClick={() => setReassignBlock(null)}
                style={{ font: "inherit", background: "none", border: "none", color: "var(--accent)", cursor: "pointer", padding: 0 }}
              >
                Cancel
              </button>
            </div>
          )}

          <QuestionList
            rows={rows}
            selectedQid={selectedQid}
            onSelect={select}
            reassignTarget={reassignBlock}
            onReassign={(qid) => void reassign(qid)}
          />

          {orphans.length > 0 && (
            <section style={{ padding: "var(--sp-4)" }}>
              <h2 style={{ fontSize: "var(--fs-base)", margin: "0 0 var(--sp-2)" }}>
                Writing that matches no question
              </h2>
              <p
                style={{
                  margin: "0 0 var(--sp-3)",
                  fontSize: "var(--fs-sm)",
                  color: "var(--text-muted)",
                }}
              >
                Often rough work — but sometimes an answer whose question was missed.
                Assign one to a question if it belongs there.
              </p>
              <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
                {orphans.map((orphan) => (
                  <li
                    key={orphan.block_id}
                    style={{
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius-sm)",
                      padding: "var(--sp-2) var(--sp-3)",
                      marginBottom: "var(--sp-2)",
                      fontSize: "var(--fs-sm)",
                    }}
                  >
                    <p style={{ margin: "0 0 var(--sp-2)" }}>
                      {orphan.text_preview || "(writing with no readable text)"}
                    </p>
                    <button
                      type="button"
                      onClick={() => setReassignBlock(orphan.block_id)}
                      style={{
                        font: "inherit",
                        fontSize: "var(--fs-xs)",
                        padding: "2px 8px",
                        border: "1px solid var(--accent)",
                        borderRadius: "var(--radius-sm)",
                        background: "transparent",
                        color: "var(--accent)",
                        cursor: "pointer",
                      }}
                    >
                      Assign to a question
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </aside>

        <main style={{ flex: 1, minWidth: 0 }}>
          {selected !== null && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "var(--sp-3)",
                padding: "var(--sp-2) var(--sp-4)",
                borderBottom: "1px solid var(--border)",
                background: "var(--surface)",
                fontSize: "var(--fs-sm)",
              }}
            >
              <strong style={{ fontFamily: "var(--font-mono)" }}>
                {selected.question.label_raw}
              </strong>
              <StatusChip presentation={selected.presentation} />
              {selected.mapping?.teacher_override && (
                <span style={{ color: "var(--text-muted)" }}>reassigned by you</span>
              )}
              {selected.mapping?.highlight?.spans_pages && (
                <span style={{ color: "var(--text-muted)" }}>
                  spans pages {selected.mapping.highlight.pages.map((p) => p + 1).join(" and ")}
                </span>
              )}
              {!selected.hasHighlight && (
                <span style={{ color: "var(--text-muted)" }}>
                  nothing to highlight for this question
                </span>
              )}

              {reassignBlock === null &&
                selectedBlocks.map((block) => (
                  <button
                    key={block.block_id}
                    type="button"
                    onClick={() => setReassignBlock(block.block_id)}
                    title={blockPreview(block)}
                    style={{
                      font: "inherit",
                      fontSize: "var(--fs-xs)",
                      padding: "2px 8px",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius-sm)",
                      background: "transparent",
                      color: "var(--accent)",
                      cursor: "pointer",
                    }}
                  >
                    {selectedBlocks.length > 1
                      ? `Move “${blockPreview(block)}”`
                      : "Move this answer"}
                  </button>
                ))}
            </div>
          )}

          {answerPages.length === 0 ? (
            <p style={{ padding: "var(--sp-5)", color: "var(--text-muted)" }}>
              No answer-sheet pages were rendered.
            </p>
          ) : (
            <AnswerSheetView
              pages={answerPages}
              highlights={highlights}
              untranscribedInk={ink}
              showUntranscribed={showUntranscribed}
              onPointerPick={pickAtPoint}
              scrollToPage={scrollTarget}
            />
          )}
        </main>
      </div>
    </div>
  );
}
