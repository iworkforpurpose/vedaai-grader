/**
 * Motion primitives, in one place so the app has one feel rather than a dozen.
 *
 * Everything here is style-agnostic and has no React and no DOM assumptions
 * beyond the element handed to it, which is why the easing and duration rules are
 * unit-testable without a browser.
 */

/** Shared durations, in ms. Mirrors the CSS custom properties of the same names. */
export const DURATION = {
  fast: 120,
  mid: 240,
  slow: 420,
} as const;

/**
 * easeInOutQuart.
 *
 * Chosen over the browser's own smooth scroll for the long travel: it leaves
 * slowly, covers the middle quickly, and settles slowly, so a jump to page 2
 * reads as a journey with a start and an end. `behavior: "smooth"` is roughly
 * 300ms flat and not tunable, which is why a two-page jump felt like a cut.
 */
export function easeInOutQuart(t: number): number {
  return t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2;
}

/**
 * How long a scroll of `distance` pixels should take.
 *
 * Proportional to distance between a floor and a ceiling. A fixed duration makes
 * short hops feel sluggish and long ones feel teleported; unbounded proportion
 * makes a twenty-page document take five seconds.
 */
export function scrollDuration(distance: number): number {
  const proportional = Math.abs(distance) * 0.55;
  return Math.max(450, Math.min(950, proportional));
}

export function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/**
 * Scroll `node` to `top` over an eased tween, and stop the moment the reader
 * takes over.
 *
 * The abort is the part that matters for feel. A tween that keeps running while
 * someone is already scrolling fights their input and reads as the page snatching
 * itself back, which is worse than no animation at all. Wheel, touch and key
 * events all cancel it.
 *
 * Returns a cancel function. Calling it again supersedes any tween in flight, so
 * clicking two questions quickly does not run two animations at once.
 */
export function animateScrollTo(
  node: HTMLElement,
  top: number,
  { onDone }: { onDone?: () => void } = {},
): () => void {
  const start = node.scrollTop;
  const target = Math.max(0, Math.min(top, node.scrollHeight - node.clientHeight));
  const delta = target - start;

  if (prefersReducedMotion() || Math.abs(delta) < 2) {
    node.scrollTop = target;
    onDone?.();
    return () => {};
  }

  const duration = scrollDuration(delta);
  let frame = 0;
  let startedAt = 0;
  let cancelled = false;

  const stop = (): void => {
    if (cancelled) return;
    cancelled = true;
    if (frame) cancelAnimationFrame(frame);
    node.removeEventListener("wheel", stop);
    node.removeEventListener("touchstart", stop);
    node.removeEventListener("keydown", stop);
  };

  // `passive` because these only ever cancel; they never preventDefault. The
  // reader's own scrolling stays entirely native.
  node.addEventListener("wheel", stop, { passive: true, once: true });
  node.addEventListener("touchstart", stop, { passive: true, once: true });
  node.addEventListener("keydown", stop, { once: true });

  const step = (now: number): void => {
    if (cancelled) return;
    if (startedAt === 0) startedAt = now;
    const t = Math.min(1, (now - startedAt) / duration);
    node.scrollTop = start + delta * easeInOutQuart(t);
    if (t < 1) {
      frame = requestAnimationFrame(step);
    } else {
      stop();
      onDone?.();
    }
  };

  frame = requestAnimationFrame(step);
  return stop;
}


/**
 * Whether an image still has a load event coming.
 *
 * Exists as a named, tested predicate because this one line was wrong twice, in
 * opposite directions, and both times it hid the answer sheet — the one thing on
 * the screen that must never be hidden.
 *
 * `complete` is the whole test. It is true once loading has finished, whether that
 * finish was a decoded bitmap or a failure, and false only while a fetch is still
 * outstanding — which is exactly "is a fade still to come".
 *
 * The two ways to get it wrong:
 *
 *   Assume `onLoad` will fire. It does not for an image that completed before
 *   React attached the handler, which is every cached image and usually the first
 *   load as well, since the `src` ships in the server-rendered HTML and the
 *   browser starts fetching during parse.
 *
 *   Add `naturalWidth > 0`, which looks stricter. A failed image reports
 *   `complete` with a zero natural width, so it is treated as still loading, its
 *   `onError` has already been and gone, and it stays hidden — taking the alt text
 *   with it.
 */
export function isStillLoading(image: { complete: boolean }): boolean {
  return !image.complete;
}
