"use client";

import type { AnswerBlock } from "@/lib/contracts";
import { blockPreview } from "@/lib/review";
import type { QuestionRow } from "@/lib/review";
import { feedbackFor, scoreLabel, scoreTone } from "@/lib/review";
import type { QuestionGrade, RubricPoint } from "@/lib/contracts";
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
}): React.JSX.Element {
  const tone = scoreTone(grade);
  const label = scoreLabel(grade);
  const feedback = feedbackFor(grade);
  const cited = grade?.rubric_points.find((point) => point.cited_line_ids.length > 0);

  const { badge, sub } = splitLabel(row.question.label_raw);
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
        <span className="q-num">{badge}</span>
        {sub && <span className="q-sub">{sub}</span>}

        <span className="q-text">
          {row.question.is_stem ? <strong>{row.question.text}</strong> : row.question.text}
        </span>

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
          ) : label && tone !== "none" ? (
            <span className="score" data-tone={tone}>
              {label}
            </span>
          ) : (
            <span className="score" data-tone="none">
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


/**
 * Split a printed label into the badge and the sub-part column.
 *
 * The frame puts the question number in a circle and any sub-part letter in its
 * own column beside it, so labels stay aligned down the list however deep they
 * nest. Three shapes occur in real papers and all three appear in the sample data:
 *
 *   "4."        -> badge 4
 *   "11 (a)"    -> badge 11, sub a
 *   "(i)"       -> badge i        — a sub-part printed on its own
 *
 * The third is why this is a function rather than one regex. A first attempt only
 * matched the first two and fell back to the raw string, which put "(i" inside a
 * 32px circle — clipped, and wrong.
 */
function splitLabel(labelRaw: string): { badge: string; sub: string | null } {
  const raw = labelRaw.trim();

  const numbered = /^(\d+)\s*[.)]?\s*(?:\(\s*([A-Za-z]+)\s*\)|([A-Za-z]+)[.)])?\s*$/.exec(raw);
  if (numbered) {
    const letter = numbered[2] ?? numbered[3];
    return { badge: numbered[1] ?? "?", sub: letter ? `${letter}.` : null };
  }

  // A bare sub-part: "(i)", "(a)", "ii." — the token itself is the badge, because
  // there is no number to show and an empty circle says nothing.
  const bare = /^\(?\s*([A-Za-z0-9]{1,4})\s*\)?\s*[.)]?$/.exec(raw);
  if (bare) return { badge: bare[1] ?? "?", sub: null };

  // Anything else: keep it short enough to fit the circle rather than clip it.
  return { badge: raw.replace(/[^A-Za-z0-9]/g, "").slice(0, 3) || "?", sub: null };
}
