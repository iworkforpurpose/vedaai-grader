"use client";

import type { OrphanAnswer } from "@/lib/contracts";
import { ArrowRightIcon } from "./icons";

/**
 * Writing on the sheet that matches no question.
 *
 * There is no frame for this, so it borrows the question card's shape and changes
 * one thing: an amber edge instead of a number badge. The point is that it reads as
 * belonging to the same list while plainly not being a question — because that is
 * exactly what it is, a row in the answer sheet's story with no counterpart in the
 * paper.
 *
 * It is worth surfacing rather than hiding. An orphan means one of two things, and
 * the second matters more: either the student answered something the paper does not
 * ask, or our own extraction missed a question. Quietly dropping it would hide our
 * own failure from the only person able to notice it.
 */
export function OrphanCard({
  orphan,
  suggestion,
  selected,
  placing,
  onShow,
  onStartPlacing,
}: {
  orphan: OrphanAnswer;
  /** The aligner's best guess at a home, if it had one worth showing. */
  suggestion: { qid: string; label: string; score: number } | null;
  selected: boolean;
  placing: boolean;
  onShow: () => void;
  onStartPlacing: () => void;
}): React.JSX.Element {
  const pages = orphan.highlight?.pages ?? [];
  const where =
    pages.length === 0
      ? null
      : pages.length === 1
        ? `page ${pages[0]! + 1}`
        : `pages ${pages.map((p) => p + 1).join(" and ")}`;

  return (
    <div
      className="q-card orphan-card"
      data-selected={selected}
      data-placing={placing}
      onClick={onShow}
    >
      <div className="q-row">
        <span className="orphan-mark" aria-hidden />

        <span className="q-text">
          {orphan.text_preview.trim() || "Writing the recognizer could not read"}
          {where && <span className="orphan-where"> · {where}</span>}
        </span>

        <span className="q-right">
          <span className="score" data-tone="orphan">
            Unplaced
          </span>
        </span>
      </div>

      {/*
        * The aligner's own best guess, offered rather than applied.
        *
        * It did not make this match, which means it was below the threshold that
        * stops a plausible-but-wrong answer being asserted. Showing the guess turns
        * a decision into a confirmation without pretending the machine was sure.
        */}
      {suggestion && (
        <p className="orphan-guess">
          Closest question: <strong>{suggestion.label}</strong>
          <span className="orphan-score"> ({Math.round(suggestion.score * 100)}% similar)</span>
        </p>
      )}

      <div className="orphan-actions">
        <button
          type="button"
          className="orphan-place"
          onClick={(event) => {
            event.stopPropagation();
            onStartPlacing();
          }}
        >
          {placing ? "Choosing a question…" : "Place this writing"}
          {!placing && <ArrowRightIcon size={16} />}
        </button>

        <button
          type="button"
          className="feedback-cite"
          onClick={(event) => {
            event.stopPropagation();
            onShow();
          }}
        >
          Show me where it is
        </button>
      </div>
    </div>
  );
}