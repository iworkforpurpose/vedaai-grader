import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { APPEAR_AFTER_MS, HOLD_MS, paced } from "./pacing";

/**
 * Fake timers throughout, including the clock `paced` measures with. A test that
 * really waits out these delays is a test nobody runs, and one that waits a few
 * milliseconds to prove them is a test that fails on a loaded machine.
 */
describe("paced", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("does not delay work that finished before the threshold", async () => {
    const settled = vi.fn();
    void paced(Promise.resolve("fast")).then(settled);

    // Enough for the promise chain, nowhere near the threshold.
    await vi.advanceTimersByTimeAsync(10);
    expect(settled).toHaveBeenCalledWith("fast");
  });

  it("holds work that crossed the threshold until it has been seen", async () => {
    const slow = new Promise<string>((resolve) => setTimeout(() => resolve("slow"), 300));
    const settled = vi.fn();
    void paced(slow).then(settled);

    // The work is done at 300ms, but the skeleton appeared at 250ms and is owed
    // its hold, so nothing resolves yet.
    await vi.advanceTimersByTimeAsync(320);
    expect(settled).not.toHaveBeenCalled();

    // 250 + 600 = 850ms from the start.
    await vi.advanceTimersByTimeAsync(560);
    expect(settled).toHaveBeenCalledWith("slow");
  });

  it("does not pad work that already outlasted the hold", async () => {
    const verySlow = new Promise<string>((resolve) => setTimeout(() => resolve("v"), 4000));
    const settled = vi.fn();
    void paced(verySlow).then(settled);

    await vi.advanceTimersByTimeAsync(3900);
    expect(settled).not.toHaveBeenCalled();

    // Resolves with the work, not the work plus another hold.
    await vi.advanceTimersByTimeAsync(150);
    expect(settled).toHaveBeenCalledWith("v");
  });

  it("surfaces a rejection without waiting", async () => {
    const caught = vi.fn();
    void paced(Promise.reject(new Error("nope"))).catch(caught);

    await vi.advanceTimersByTimeAsync(10);
    expect(caught).toHaveBeenCalled();
  });

  it("never pads by more than the threshold plus the hold", () => {
    expect(APPEAR_AFTER_MS + HOLD_MS).toBeLessThanOrEqual(1000);
  });
});
