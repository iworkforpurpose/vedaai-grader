"use client";

import { useEffect, useState } from "react";

/** How long each phase is shown before the next one replaces it. */
const EVERY_MS = 2000;

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

  const phase = phases[index];
  if (!phase) return null;

  return (
    // `key` remounts the span, which is what re-runs its entrance so each phase
    // arrives rather than swapping in place.
    //
    // `aria-hidden` because the region around this is already `aria-live`, and a
    // caption that changes every two seconds would interrupt a screen reader five
    // times to say nothing it can act on. The stable note beside it carries the
    // meaning: this is going to take a while.
    <span className="stage-phase" key={index} aria-hidden="true">
      {phase}
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
