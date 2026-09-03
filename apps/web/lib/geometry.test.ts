import { describe, expect, it } from "vitest";
import type { BBox, PageBox } from "./contracts";
import {
  area,
  boxToStyle,
  contains,
  firstPage,
  groupByPage,
  mergeOverlapping,
  stackEdges,
} from "./geometry";

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

describe("mergeOverlapping", () => {
  const at = (x0: number, y0: number, x1: number, y1: number, page = 0) => ({
    page,
    box: { x0, y0, x1, y1 },
  });

  it("keeps the rows of one highlight apart when they only touch", () => {
    // The mapper draws a multi-line answer as one row per line of writing, each
    // extended to meet the row below so the run reads as a single connected
    // shape. Those rows share an edge and share no area, and that distinction is
    // load-bearing: collapsing them would restore the single rectangle the row
    // shape exists to replace, and would do it silently, in the browser, where
    // no eval metric can see it.
    const rows = [
      at(0.12, 0.2, 0.78, 0.23),
      at(0.12, 0.23, 0.78, 0.27),
      at(0.12, 0.27, 0.55, 0.3),
    ];

    expect(mergeOverlapping(rows)).toHaveLength(3);
  });

  it("collapses a band drawn inside another band", () => {
    // The reported bug: a wide band with a tighter one nested in it, which reads
    // as the sheet being highlighted twice.
    const merged = mergeOverlapping([
      at(0.1, 0.1, 0.9, 0.2),
      at(0.15, 0.12, 0.85, 0.18),
    ]);

    expect(merged).toHaveLength(1);
    expect(merged[0]?.box).toEqual({ x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 });
  });

  it("unions two bands that partly overlap", () => {
    const merged = mergeOverlapping([
      at(0.1, 0.1, 0.5, 0.2),
      at(0.4, 0.15, 0.9, 0.25),
    ]);

    expect(merged).toHaveLength(1);
    expect(merged[0]?.box).toEqual({ x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.25 });
  });

  it("keeps bands that are genuinely separate", () => {
    // The code-page case the backend deliberately splits: two runs of writing far
    // apart down the page. Merging these would repaint the gap between them.
    const merged = mergeOverlapping([
      at(0.1, 0.1, 0.9, 0.2),
      at(0.1, 0.6, 0.9, 0.7),
    ]);

    expect(merged).toHaveLength(2);
  });

  it("never merges across pages", () => {
    const merged = mergeOverlapping([
      at(0.1, 0.1, 0.9, 0.2, 0),
      at(0.1, 0.1, 0.9, 0.2, 1),
    ]);

    expect(merged).toHaveLength(2);
  });

  it("collapses a chain, where a joins b and b joins c", () => {
    const merged = mergeOverlapping([
      at(0.1, 0.1, 0.4, 0.2),
      at(0.3, 0.1, 0.6, 0.2),
      at(0.5, 0.1, 0.9, 0.2),
    ]);

    expect(merged).toHaveLength(1);
    expect(merged[0]?.box.x1).toBeCloseTo(0.9);
  });

  it("returns an empty list unchanged", () => {
    expect(mergeOverlapping([])).toEqual([]);
  });
});

describe("stackEdges", () => {
  const at = (x0: number, y0: number, x1: number, y1: number, page = 0) => ({
    page,
    box: { x0, y0, x1, y1 },
  });

  it("names the outer rows of one answer so only its corners are rounded", () => {
    expect(
      stackEdges([
        at(0.12, 0.2, 0.78, 0.23),
        at(0.12, 0.23, 0.78, 0.27),
        at(0.12, 0.27, 0.55, 0.3),
      ]),
    ).toEqual(["top", "middle", "bottom"]);
  });

  it("leaves a single box alone", () => {
    expect(stackEdges([at(0.12, 0.2, 0.78, 0.23)])).toEqual(["only"]);
  });

  it("does not join two regions separated by paper", () => {
    // An answer at the top of the page and another at the bottom. Rounding them
    // as one shape would claim the empty half between them.
    expect(
      stackEdges([at(0.12, 0.1, 0.78, 0.13), at(0.12, 0.7, 0.78, 0.73)]),
    ).toEqual(["only", "only"]);
  });

  it("does not join two columns that merely sit level with each other", () => {
    // The code page. These share no width, so they are two shapes whatever their
    // vertical extents do.
    expect(
      stackEdges([at(0.1, 0.2, 0.4, 0.23), at(0.6, 0.23, 0.9, 0.26)]),
    ).toEqual(["only", "only"]);
  });

  it("keeps pages apart", () => {
    // A continuation is a new shape on the next page, however the numbers line
    // up: the bottom of page 0 does not touch the top of page 1.
    expect(
      stackEdges([at(0.12, 0.9, 0.78, 1.0, 0), at(0.12, 0.0, 0.78, 0.1, 1)]),
    ).toEqual(["only", "only"]);
  });

  it("reports edges in the order it was given, not in reading order", () => {
    // The caller pairs these with its own list by index, so a sort inside must
    // not leak out. Given bottom-first, the answer is still bottom-first.
    expect(
      stackEdges([at(0.12, 0.23, 0.78, 0.27), at(0.12, 0.2, 0.78, 0.23)]),
    ).toEqual(["bottom", "top"]);
  });
});
