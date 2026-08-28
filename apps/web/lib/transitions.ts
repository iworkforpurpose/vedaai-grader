"use client";

import { flushSync } from "react-dom";

/**
 * Cross-fade a state change instead of cutting to it.
 *
 * Every animation in this app so far has been an *entrance*: the arriving screen
 * fades in. That is only half a transition, and the missing half is the visible
 * one — React unmounts the outgoing markup the instant state changes, so the old
 * screen disappears in a single frame and then the new one fades up from nothing.
 * The cut is still there; it just happens before the fade.
 *
 * Nothing in CSS fixes that, because there is no element left to animate. The View
 * Transitions API is the mechanism that does: the browser photographs the current
 * page, lets the DOM change however it likes, photographs the result, and
 * cross-fades between the two. Unmounting stops mattering.
 *
 * `flushSync` is not optional. React batches state updates, so without it the
 * callback returns before the DOM has changed and the browser photographs the same
 * frame twice — a transition that technically runs and visibly does nothing.
 *
 * Degrades by doing the update plainly. A browser without the API gets today's
 * behaviour rather than a broken screen, and `prefers-reduced-motion` is honoured
 * by the stylesheet, which reduces the animation to nothing rather than skipping it
 * here — one place to reason about instead of two.
 */
export function crossFade(update: () => void): void {
  const doc = document as Document & {
    startViewTransition?: (callback: () => void) => { finished: Promise<void> };
  };

  if (typeof doc.startViewTransition !== "function") {
    update();
    return;
  }

  doc.startViewTransition(() => {
    flushSync(update);
  });
}
