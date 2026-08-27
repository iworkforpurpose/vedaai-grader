"use client";

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
  grade,
  selected,
  expanded,
  onSelect,
  onToggle,
  onCite,
}: {
  row: QuestionRow;
  grade: QuestionGrade | undefined;
  selected: boolean;
  expanded: boolean;
  onSelect: () => void;
  onToggle: () => void;
  onCite: (point: RubricPoint) => void;
}): React.JSX.Element {
  const tone = scoreTone(grade);
  const label = scoreLabel(grade);
  const feedback = feedbackFor(grade);
  const cited = grade?.rubric_points.find((point) => point.cited_line_ids.length > 0);

  const { badge, sub } = splitLabel(row.question.label_raw);

  return (
    <div className="q-card" data-selected={selected} onClick={onSelect}>
      <div className="q-row">
        <span className="q-num">{badge}</span>
        {sub && <span className="q-sub">{sub}</span>}

        <span className="q-text">
          {row.question.is_stem ? <strong>{row.question.text}</strong> : row.question.text}
        </span>

        <span className="q-right">
          {label && tone !== "none" ? (
            <span className="score" data-tone={tone}>
              {label}
            </span>
          ) : (
            <span className="score" data-tone="none">
              {row.presentation.label}
            </span>
          )}

          {(feedback || row.presentation.hint) && (
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

      {expanded && (feedback || row.presentation.hint) && (
        <div className="feedback">
          <h3>{feedback ? "AI Feedback" : "Why this is flagged"}</h3>
          <p>{feedback ?? row.presentation.hint}</p>

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
      )}
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
