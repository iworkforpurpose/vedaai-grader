/**
 * When a loading state is worth showing, and for how long.
 *
 * The first version of this held every screen open for a second and a half,
 * whatever it was waiting for. That is the wrong shape and it showed: a route
 * that resolves in forty milliseconds was being made forty times slower to
 * display a skeleton nobody needed, and every hot reload during development came
 * with a second and a half of blocks.
 *
 * The shape that works has two numbers rather than one.
 *
 * Nothing appears for the first `APPEAR_AFTER_MS`. Work that finishes inside that
 * window is simply fast, and the reader sees the finished screen with no loading
 * state at all — which is the correct amount of ceremony for something that was
 * never slow.
 *
 * Once the skeleton has been shown it stays for `HOLD_MS`. A loading state that
 * appears and vanishes inside a few frames is worse than none: the eye registers
 * that something happened without being able to read what, which is the flicker
 * the skeleton exists to prevent.
 *
 * The delay is enforced twice, in the two places that can each only do half of
 * it. This module keeps the server from resolving so soon after the threshold
 * that the skeleton flashes; the stylesheet keeps the skeleton from painting
 * before the threshold at all. Neither can do the other's half.
 */

/** How long work may take before a loading state is worth showing. */
export const APPEAR_AFTER_MS = 250;

/** Once shown, the shortest time a loading state stays up. */
export const HOLD_MS = 600;

/**
 * Resolve `work`, delaying only when a loading state will already be on screen.
 *
 * Fast work is not delayed at all. Work that crossed the threshold is held until
 * the skeleton has had `HOLD_MS` to be looked at, so the total is at most
 * `APPEAR_AFTER_MS + HOLD_MS` of padding and usually none.
 *
 * A rejection is never paced. An error should surface as soon as it is known;
 * holding a skeleton over a failure only delays the moment the reader can act.
 */
export async function paced<T>(
  work: Promise<T>,
  {
    appearAfterMs = APPEAR_AFTER_MS,
    holdMs = HOLD_MS,
  }: { appearAfterMs?: number; holdMs?: number } = {},
): Promise<T> {
  const started = Date.now();
  const result = await work;
  const elapsed = Date.now() - started;

  // Never became visible, so there is nothing to hold.
  if (elapsed < appearAfterMs) return result;

  const remaining = appearAfterMs + holdMs - elapsed;
  if (remaining > 0) {
    await new Promise<void>((resolve) => setTimeout(resolve, remaining));
  }
  return result;
}
