"use client";

import { useEffect, useState } from "react";

/** How long each phase is shown before the next one replaces it. */
const EVERY_MS = 4000;

/**
 * The stages of the wait, one after another.
 *
 * These are timed, not measured. The service does run these steps and it runs
 * them in this order, but nothing here is listening to it — the caption advances
 * on a clock, so on a long paper it will be sitting on the last line while the
 * pipeline is still somewhere in the middle. It is honest about the order and
 * approximate about the position, which is the trade being made deliberately.
 *
 * The API does publish a real progress stream. Nothing in the browser subscribes
 * to it yet, and doing so would make this caption true rather than plausible.
 *
 * It holds on the last phase rather than looping. A caption that returns to
 * "Rendering pages" after "Almost ready" tells the reader the work restarted,
 * which is the one thing it must never imply.
 *
 * The ellipsis is added here rather than written into each phrase, so the list
 * stays a list of stages and cannot end up with one item missing its tail.
 */
export function LoadingPhases({
  phases,
  everyMs = EVERY_MS,
}: {
  phases: readonly string[];
  everyMs?: number;
}): React.JSX.Element | null {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (index >= phases.length - 1) return;
    const timer = window.setTimeout(() => setIndex((n) => n + 1), everyMs);
    return () => window.clearTimeout(timer);
  }, [index, phases.length, everyMs]);

  return (
    /*
     * Every phase is rendered, stacked in one grid cell, and only the active one
     * is opaque. That is what makes this a cross-fade rather than a fade-in: a
     * caption that is unmounted the moment the next arrives has nothing to fade
     * out, so the change lands as a flicker however slow the entrance is.
     *
     * Stacking also fixes the width. The container is as wide as the longest
     * phrase from the first frame, so the line does not jump about underneath
     * the loader as the words change length.
     *
     * `aria-hidden` because the region around this is already `aria-live`, and a
     * caption changing five times would interrupt a screen reader to say nothing
     * it can act on. The stable note beside it carries the meaning.
     */
    <span className="stage-phases" aria-hidden="true">
      {phases.map((phase, i) => (
        <span key={phase} className="stage-phase" data-active={i === index}>
          {phase}…
        </span>
      ))}
    </span>
  );
}

/** The order the service actually works in, which is what makes the copy true. */
export const PIPELINE_PHASES = [
  "Rendering pages",
  "Reading the handwriting",
  "Matching answers to questions",
  "Scoring the answers",
  "Almost ready",
] as const;
