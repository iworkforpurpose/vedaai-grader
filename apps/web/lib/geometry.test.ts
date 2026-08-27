import { describe, expect, it } from "vitest";
import type { BBox, PageBox } from "./contracts";
import { area, boxToStyle, contains, firstPage, groupByPage } from "./geometry";

function box(x0: number, y0: number, x1: number, y1: number): BBox {
  return { x0, y0, x1, y1 };
}

function pageBox(page: number, x0: number, y0: number, x1: number, y1: number): PageBox {
  return { page, box: box(x0, y0, x1, y1) };
}

describe("boxToStyle", () => {
  it("expresses a box as percentages of its container", () => {
    // Percentages rather than pixels is the whole reason coordinates are
    // normalized: the overlay then tracks its page image at any rendered width
    // with no measurement and no resize listener.
    expect(boxToStyle(box(0.1, 0.2, 0.6, 0.5))).toEqual({
      left: "10%",
      top: "20%",
      width: "50%",
      height: "30%",
    });
  });

  it("handles a full-page box", () => {
    expect(boxToStyle(box(0, 0, 1, 1))).toEqual({
      left: "0%",
      top: "0%",
      width: "100%",
      height: "100%",
    });
  });
});

describe("groupByPage", () => {
  it("splits boxes by the page they sit on", () => {
    // A multi-page highlight is just a list whose entries carry different pages,
    // which is what lets a page-spanning answer need no special handling in the UI.
    const grouped = groupByPage([
      pageBox(0, 0.1, 0.8, 0.9, 0.95),
      pageBox(1, 0.1, 0.05, 0.9, 0.3),
      pageBox(0, 0.1, 0.5, 0.9, 0.6),
    ]);

    expect([...grouped.keys()].sort()).toEqual([0, 1]);
    expect(grouped.get(0)).toHaveLength(2);
    expect(grouped.get(1)).toHaveLength(1);
  });

  it("returns an empty map for no boxes", () => {
    expect(groupByPage([]).size).toBe(0);
  });
});

describe("firstPage", () => {
  it("finds the earliest page a highlight touches", () => {
    // Used for scroll targeting: a page-spanning answer should scroll to where it
    // begins, not to whichever box happened to be listed first.
    expect(firstPage([pageBox(3, 0, 0, 1, 1), pageBox(1, 0, 0, 1, 1)])).toBe(1);
  });

  it("returns null when there is nothing to scroll to", () => {
    expect(firstPage([])).toBeNull();
  });
});

describe("contains", () => {
  it("accepts a point inside the box", () => {
    expect(contains(box(0.1, 0.1, 0.9, 0.9), 0.5, 0.5)).toBe(true);
  });

  it("rejects a point outside the box", () => {
    expect(contains(box(0.1, 0.1, 0.4, 0.4), 0.8, 0.8)).toBe(false);
  });

  it("treats the boundary as inside", () => {
    expect(contains(box(0.1, 0.1, 0.9, 0.9), 0.1, 0.9)).toBe(true);
  });
});

describe("area", () => {
  it("multiplies the sides", () => {
    expect(area(box(0.2, 0.2, 0.6, 0.7))).toBeCloseTo(0.4 * 0.5);
  });
});
