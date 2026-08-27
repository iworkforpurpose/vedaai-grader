"use client";

import type { QuestionRow } from "@/lib/review";
import { StatusChip } from "./StatusChip";

/**
 * Every question in printed order, with its status.
 *
 * The label is shown exactly as the paper printed it — "11 (a)", not a
 * normalized rewrite — because a teacher is matching this list against the paper
 * in front of them.
 *
 * Rows are buttons rather than divs with click handlers, so the list is
 * navigable by keyboard. The review surface is dense and heavily clickable, and
 * a teacher tabbing through needs to know where they are.
 */
export function QuestionList({
  rows,
  selectedQid,
  onSelect,
  reassignTarget,
  onReassign,
}: {
  rows: readonly QuestionRow[];
  selectedQid: string | null;
  onSelect: (qid: string) => void;
  reassignTarget: string | null;
  onReassign: (qid: string) => void;
}): React.JSX.Element {
  return (
    <ol
      style={{
        listStyle: "none",
        margin: 0,
        padding: 0,
        display: "flex",
        flexDirection: "column",
      }}
    >
      {rows.map((row) => {
        const selected = row.question.qid === selectedQid;
        const confidence = row.mapping?.confidence ?? 0;
        return (
          <li key={row.question.qid}>
            <button
              type="button"
              onClick={() =>
                reassignTarget ? onReassign(row.question.qid) : onSelect(row.question.qid)
              }
              aria-current={selected ? "true" : undefined}
              style={{
                width: "100%",
                textAlign: "left",
                display: "flex",
                flexDirection: "column",
                gap: "var(--sp-1)",
                padding: "var(--sp-3) var(--sp-4)",
                border: "none",
                borderBottom: "1px solid var(--border)",
                borderLeft: `3px solid ${
                  selected ? "var(--accent)" : "transparent"
                }`,
                background: selected ? "var(--surface-2)" : "var(--surface)",
                cursor: reassignTarget ? "copy" : "pointer",
                font: "inherit",
                color: "inherit",
              }}
            >
              <span
                style={{ display: "flex", alignItems: "baseline", gap: "var(--sp-2)" }}
              >
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontWeight: 600,
                    minWidth: "4.5em",
                  }}
                >
                  {row.question.label_raw}
                </span>
                <StatusChip presentation={row.presentation} compact />
                {row.question.marks !== null && (
                  <span
                    style={{
                      marginLeft: "auto",
                      fontFamily: "var(--font-mono)",
                      fontSize: "var(--fs-xs)",
                      color: "var(--text-muted)",
                    }}
                  >
                    {row.question.marks} marks
                  </span>
                )}
              </span>

              <span
                style={{
                  fontSize: "var(--fs-sm)",
                  color: "var(--text-2)",
                  display: "-webkit-box",
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: "vertical",
                  overflow: "hidden",
                }}
              >
                {row.question.text}
              </span>

              {row.status === "answered" && confidence < 0.6 && (
                <span
                  style={{ fontSize: "var(--fs-xs)", color: "var(--status-review)" }}
                >
                  Low confidence — worth checking the highlight.
                </span>
              )}
              {row.presentation.needsAttention && (
                <span style={{ fontSize: "var(--fs-xs)", color: "var(--text-muted)" }}>
                  {row.presentation.hint}
                </span>
              )}
            </button>
          </li>
        );
      })}
    </ol>
  );
}
