import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { atLeast, SKELETON_MS } from "./pacing";

/**
 * Fake timers throughout. A test that really waits 1.5s to prove a 1.5s floor is
 * a test nobody runs, and one that waits 20ms to prove it is a test that fails on
 * a loaded machine.
 */
describe("atLeast", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("holds work that finishes early until the floor", async () => {
    const settled = vi.fn();
    const promise = atLeast(Promise.resolve("done"), 1500).then((v) => {
      settled(v);
      return v;
    });

    await vi.advanceTimersByTimeAsync(1400);
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(200);
    expect(settled).toHaveBeenCalledWith("done");
    await expect(promise).resolves.toBe("done");
  });

  it("does not delay work that already took longer than the floor", async () => {
    const slow = new Promise<string>((resolve) => setTimeout(() => resolve("slow"), 4000));
    const settled = vi.fn();
    const promise = atLeast(slow, 1500).then((v) => {
      settled(v);
      return v;
    });

    // The floor has long passed; only the work is still outstanding.
    await vi.advanceTimersByTimeAsync(3900);
    expect(settled).not.toHaveBeenCalled();

    // It resolves with the work, not the work plus another floor.
    await vi.advanceTimersByTimeAsync(150);
    expect(settled).toHaveBeenCalledWith("slow");
    await expect(promise).resolves.toBe("slow");
  });

  it("surfaces a rejection without waiting out the floor", async () => {
    const promise = atLeast(Promise.reject(new Error("nope")), 1500);
    const caught = vi.fn();
    void promise.catch(caught);

    await vi.advanceTimersByTimeAsync(10);
    expect(caught).toHaveBeenCalled();
  });

  it("defaults to the shared constant", async () => {
    const settled = vi.fn();
    void atLeast(Promise.resolve(1)).then(settled);

    await vi.advanceTimersByTimeAsync(SKELETON_MS - 100);
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(150);
    expect(settled).toHaveBeenCalled();
  });

  it("is the value the screens are paced to", () => {
    expect(SKELETON_MS).toBe(1500);
  });
});
