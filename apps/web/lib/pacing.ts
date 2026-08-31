/**
 * How long a screen change is held open, and why it is held at all.
 *
 * A skeleton that appears and vanishes inside a hundred milliseconds is worse
 * than no skeleton: the eye registers that something happened without being able
 * to read what, which is the flicker every loading state is trying to avoid. So
 * once a screen commits to loading, it stays loading for a floor.
 *
 * This is a deliberate floor rather than a threshold. The usual shape is to show
 * nothing until the work has already taken 250ms or so, and that is the better
 * default when the work is genuinely fast -- it costs nothing and the reader
 * never sees a loading state at all. Chosen against that here: the screens this
 * paces are ones where something real is being fetched, and a consistent beat on
 * every large change reads as an application that is doing something rather than
 * one that flickers.
 */

/** The floor, in milliseconds. One constant, so it is one edit to change. */
export const SKELETON_MS = 1500;

/**
 * Resolve `work`, but never sooner than `ms`.
 *
 * The two run together rather than in sequence: work that already takes longer
 * than the floor is not delayed at all, and work that finishes early waits out
 * the remainder. A sequential `await sleep()` followed by `await work` would add
 * the floor to every load instead of absorbing it.
 *
 * A rejection is not paced. An error should surface as soon as it is known --
 * holding a skeleton over a failure only delays the moment the reader can act on
 * it, and the error boundary has its own arrival.
 */
export async function atLeast<T>(work: Promise<T>, ms: number = SKELETON_MS): Promise<T> {
  const floor = new Promise<void>((resolve) => setTimeout(resolve, ms));
  const [result] = await Promise.all([work, floor]);
  return result;
}
