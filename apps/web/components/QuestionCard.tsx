"use client";

import type { AnswerBlock } from "@/lib/contracts";
import { blockPreview } from "@/lib/review";
import type { QuestionRow } from "@/lib/review";
import { feedbackFor, parseMark, proposedMark, scoreLabel, scoreTone, splitLabel, trimMark } from "@/lib/review";
import type { QuestionGrade, RubricPoint } from "@/lib/contracts";
import { useState } from "react";
import { ChevronDownIcon } from "./icons";

/**
 * One question, as the mapping frame draws it: number badge, text, score pill,
 * expander — and the AI feedback panel when open.
 *
 * The frame assumes every question carries a score, because it was drawn with
 * marking already done. This product has four states the frame has no pill for:
 * not marked yet, not answered, unreadable, and not required. Rather than invent a
 * fourth pill colour, an unmarked question wears the neutral chip and the status
 * is carried in words underneath — because "nobody has marked this" and "the
 * student scored nothing" are the same number and completely different facts, and
 * a teacher acting on the wrong one marks a student down for the tool's silence.
 */
export function QuestionCard({
  row,
  index,
  grade,
  selected,
  expanded,
  onSelect,
  onToggle,
  onCite,
  placingActive,
  teacherPlaced,
  movable,
  onPlaceHere,
  onMoveBlock,
  onCorrectMark,
}: {
  row: QuestionRow;
  /** Position in the list, for the entrance stagger. */
  index: number;
  grade: QuestionGrade | undefined;
  selected: boolean;
  expanded: boolean;
  onSelect: () => void;
  onToggle: () => void;
  onCite: (point: RubricPoint) => void;
  /** True while some writing is looking for a home, which makes this a target. */
  placingActive: boolean;
  /** Whether this question's mapping was set by hand rather than by the aligner. */
  teacherPlaced: boolean;
  /** The blocks currently mapped here, each individually movable. */
  movable: readonly AnswerBlock[];
  onPlaceHere: () => void;
  onMoveBlock: (blockId: string) => void;
  /** What the teacher says this is worth. Null clears their correction. */
  onCorrectMark: (marks: number | null) => void;
}): React.JSX.Element {
  const [editing, setEditing] = useState<string | null>(null);
  const tone = scoreTone(grade);
  const label = scoreLabel(grade);
  const feedback = feedbackFor(grade);
  const cited = grade?.rubric_points.find((point) => point.cited_line_ids.length > 0);

  const { badge, sub } = splitLabel(row.question.label_raw, row.question.path);
  const showPanel = expanded && Boolean(feedback || row.presentation.hint);

  return (
    <div
      className="q-card"
      data-selected={selected}
      /*
       * A target while something is being placed.
       *
       * The whole card, not a small button in a corner. The teacher has already said
       * what they are moving; the only remaining question is which question, and the
       * answer is a card they are already reading. A stem is excluded because it asks
       * nothing — placing an answer on it would be placing it nowhere.
       */
      data-target={placingActive && !row.question.is_stem}
      style={{ "--stagger": `${index * 28}ms` } as React.CSSProperties}
      onClick={placingActive && !row.question.is_stem ? onPlaceHere : onSelect}
    >
      <div className="q-row">
        <span className="q-num" aria-hidden="true">
          {badge}
        </span>
        {sub && (
          <span className="q-sub" aria-hidden="true">
            {sub}
          </span>
        )}

        {/*
         * The tab stop, and the thing a screen reader announces.
         *
         * The card itself carried the only click handler and was a plain `div`
         * with no role, no tabIndex and no key handler — so the product's primary
         * interaction could not be performed without a mouse. The CSS still holds
         * the reset a button needs (`width: 100%`, `font: inherit`,
         * `text-align: left`) and `.q-list` reserves room for a focus ring, so
         * this was a `<button>` once and regressed.
         *
         * It cannot go back to being one: the card now contains the score pill,
         * the chevron and the move controls, and a button inside a button is
         * invalid and unreachable. So the control is the question text, which is
         * what a person would point at anyway.
         *
         * The badge and sub-label are `aria-hidden` and folded into the label
         * here instead, so the announcement is one sentence rather than three
         * fragments read in layout order.
         */}
        <button
          type="button"
          className="q-open"
          aria-pressed={selected}
          aria-label={`Question ${badge}${sub ? ` ${sub}` : ""}: ${row.question.text}`}
          onClick={(event) => {
            event.stopPropagation();
            if (placingActive && !row.question.is_stem) onPlaceHere();
            else onSelect();
          }}
        >
          {row.question.is_stem ? <strong>{row.question.text}</strong> : row.question.text}
        </button>

        <span className="q-right">
          {/* Whose decision this was. A teacher returning to a script should not have
              to remember what they moved, and re-marking will not undo it. */}
          {teacherPlaced && !placingActive && (
            <span className="placed-badge" title="You placed this answer here">
              Placed by you
            </span>
          )}

          {placingActive && !row.question.is_stem ? (
            <span className="score" data-tone="target">
              Place here
            </span>
          ) : editing !== null && grade ? (
            /*
             * The pill becomes the field, in place.
             *
             * A separate editor somewhere else on the card would make the teacher
             * look away from the number they are changing. Committing on blur as
             * well as on Enter is deliberate: a mark typed and then clicked away
             * from has been decided, and losing it would be the interface
             * discarding a judgement.
             */
            <input
              className="score score-edit"
              type="text"
              inputMode="decimal"
              autoFocus
              aria-label={`Marks out of ${trimMark(grade.marks_available)}`}
              value={editing}
              onClick={(event) => event.stopPropagation()}
              onChange={(event) => setEditing(event.target.value)}
              onBlur={() => {
                const marks = parseMark(editing, grade.marks_available);
                setEditing(null);
                if (marks !== undefined) onCorrectMark(marks);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
                // Escape abandons rather than commits, which is the only way to
                // back out of a mistyped mark without saving it first.
                if (event.key === "Escape") {
                  setEditing(null);
                  event.stopPropagation();
                }
              }}
            />
          ) : label && tone !== "none" ? (
            <button
              type="button"
              className="score"
              data-tone={tone}
              data-corrected={grade?.teacher_decided || undefined}
              title={proposedMark(grade) ?? "Click to change this mark"}
              onClick={(event) => {
                event.stopPropagation();
                setEditing(grade ? trimMark(grade.marks_final) : "");
              }}
            >
              {label}
            </button>
          ) : grade && grade.marks_available > 0 ? (
            /*
             * Unmarked, and still correctable.
             *
             * This is the case the whole feature is for: the marker declined —
             * the writing was crossed out, the transcription was unreadable, the
             * provider was down — and a teacher looking at the page can see what
             * it is worth. Leaving this pill inert would mean the one question
             * most needing a person is the one they cannot answer.
             */
            <button
              type="button"
              className="score"
              data-tone="none"
              data-status={row.status}
              title={row.presentation.hint ?? "Click to set this mark"}
              onClick={(event) => {
                event.stopPropagation();
                setEditing("");
              }}
            >
              {row.presentation.needsAttention && (
                <span className="score-mark" aria-hidden="true">
                  !
                </span>
              )}
              {row.presentation.label}
            </button>
          ) : (
            <span
              className="score"
              data-tone="none"
              data-status={row.status}
              title={row.presentation.hint}
            >
              {/*
                The channel that is not colour.
                `aria-hidden` because the label beside it already carries the
                meaning, and "exclamation Not found" is a worse thing to hear
                than "Not found".
              */}
              {row.presentation.needsAttention && (
                <span className="score-mark" aria-hidden="true">
                  !
                </span>
              )}
              {row.presentation.label}
            </span>
          )}

          {!placingActive && (feedback || row.presentation.hint) && (
            <button
              type="button"
              className="q-chevron"
              aria-label={expanded ? "Hide details" : "Show details"}
              aria-expanded={expanded}
              onClick={(event) => {
                event.stopPropagation();
                onToggle();
              }}
              style={expanded ? { transform: "rotate(180deg)" } : undefined}
            >
              <ChevronDownIcon size={18} />
            </button>
          )}
        </span>
      </div>

      {/*
        * Always rendered, opened by a data attribute.
        *
        * It used to be a conditional render, which cannot animate by
        * construction: the element is absent one frame and present the next, so
        * there is nothing for the browser to interpolate and the panel popped.
        * The wrapper animates `grid-template-rows` from `0fr` to `1fr`, which is
        * the one way to transition to an automatic height without measuring it in
        * JavaScript.
        *
        * `inert` while closed. The panel is still in the tree at zero height, so
        * without it the citation button stays in the tab order and a keyboard
        * reader would land inside a collapsed card.
        */}
      <div className="feedback-wrap" data-open={showPanel} inert={!showPanel}>
        <div className="feedback-clip">
          <div className="feedback">
            <h3>AI Feedback</h3>
            {feedback ? (
              <p>{feedback}</p>
            ) : (
              <p className="feedback-pending">No marker comment for this one yet.</p>
            )}

            {/* Only where it explains something the pill does not. With a mark on
                screen, "an answer was found and located" restates the score. */}
            {row.presentation.hint && tone === "none" && (
              <p className="feedback-flag">{row.presentation.hint}</p>
            )}

            {/*
              * Moving an answer away, one block at a time.
              *
              * Per block rather than per question, because a wrong mapping is often
              * one block of several — a page-spanning answer whose second half went
              * to the next question. Moving the whole answer to fix half of it trades
              * one error for another.
              */}
            {/* Not while something is being placed: offering "move this away" on a
                card that is simultaneously offering "place it here" asks the reader
                to hold two contradictory intentions at once. */}
            {!placingActive && movable.length > 0 && (
              <div className="move-out">
                <h4>Wrong answer here?</h4>
                {movable.map((block) => (
                  <button
                    key={block.block_id}
                    type="button"
                    className="move-out-block"
                    onClick={(event) => {
                      event.stopPropagation();
                      onMoveBlock(block.block_id);
                    }}
                  >
                    Move &ldquo;{blockPreview(block)}&rdquo;
                  </button>
                ))}
              </div>
            )}

            {/* The citation is what makes a mark checkable in two seconds rather
                than something to re-mark from scratch. */}
            {cited && (
              <button
                type="button"
                className="feedback-cite"
                onClick={(event) => {
                  event.stopPropagation();
                  onCite(cited);
                }}
              >
                Show the writing this rests on
              </button>
            )}
          </div>
        </div>
      </div>

    </div>
  );
}
