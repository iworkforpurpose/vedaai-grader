import { describe, expect, it } from "vitest";
import { easeInOutQuart, isStillLoading, scrollDuration } from "./motion";

describe("easeInOutQuart", () => {
  it("is pinned at both ends", () => {
    expect(easeInOutQuart(0)).toBe(0);
    expect(easeInOutQuart(1)).toBe(1);
  });

  it("passes through the midpoint", () => {
    expect(easeInOutQuart(0.5)).toBeCloseTo(0.5, 10);
  });

  it("is monotonic, so a scroll never reverses mid-tween", () => {
    let previous = -1;
    for (let i = 0; i <= 100; i += 1) {
      const value = easeInOutQuart(i / 100);
      expect(value).toBeGreaterThanOrEqual(previous);
      previous = value;
    }
  });

  it("starts and ends slower than the middle", () => {
    // The whole point of the curve: the first tenth covers far less ground than
    // the tenth around the midpoint.
    const opening = easeInOutQuart(0.1) - easeInOutQuart(0);
    const middle = easeInOutQuart(0.55) - easeInOutQuart(0.45);
    expect(middle).toBeGreaterThan(opening * 4);
  });
});

describe("scrollDuration", () => {
  it("never dips below the floor, however short the hop", () => {
    expect(scrollDuration(0)).toBe(450);
    expect(scrollDuration(12)).toBe(450);
  });

  it("never exceeds the ceiling, however long the document", () => {
    expect(scrollDuration(50_000)).toBe(950);
  });

  it("scales with distance in between", () => {
    expect(scrollDuration(1000)).toBeCloseTo(550, 5);
    expect(scrollDuration(1400)).toBeGreaterThan(scrollDuration(1000));
  });

  it("ignores direction", () => {
    expect(scrollDuration(-1000)).toBe(scrollDuration(1000));
  });
});


describe("isStillLoading", () => {
  it("is false for an image that finished before the handler attached", () => {
    // The cached case, and usually the first load too: this returning true is what
    // left a fully decoded answer sheet at zero opacity.
    expect(isStillLoading({ complete: true })).toBe(false);
  });

  it("is true only while a fetch is outstanding", () => {
    expect(isStillLoading({ complete: false })).toBe(true);
  });

  it("is false for a failed image, so the alt text is not hidden with it", () => {
    // A failed load reports complete with a zero natural width. Testing the width
    // as well treated this as still loading and hid it permanently.
    expect(isStillLoading({ complete: true, naturalWidth: 0 } as HTMLImageElement)).toBe(
      false,
    );
  });
});
