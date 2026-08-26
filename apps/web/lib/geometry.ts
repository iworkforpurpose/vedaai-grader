/**
 * The frontend half of the coordinate contract.
 *
 * Boxes arrive normalized to [0,1] with the origin top-left, relative to the
 * page rendered at 200 DPI. This module is the only place that turns them into
 * something CSS understands, so there is exactly one line of code to inspect
 * when a highlight lands in the wrong place.
 *
 * Percentages rather than pixels, deliberately. The overlay then tracks its
 * page image at any rendered size — zoom, window resize, a responsive column —
 * with no measurement, no resize observer, and no re-render. That property is
 * the reason the pipeline normalizes coordinates in the first place.
 */

import type { BBox, PageBox } from "./contracts";

export interface BoxStyle {
  left: string;
  top: string;
  width: string;
  height: string;
}

/** Position a box over its page container. */
export function boxToStyle(box: BBox): BoxStyle {
  return {
    left: `${box.x0 * 100}%`,
    top: `${box.y0 * 100}%`,
    width: `${(box.x1 - box.x0) * 100}%`,
    height: `${(box.y1 - box.y0) * 100}%`,
  };
}

/** Group page-scoped boxes by page, so each page renders only its own.
 *
 * A multi-page highlight is just a list whose entries carry different page
 * indices, which is what lets an answer spanning a page boundary need no
 * special handling anywhere in the UI.
 */
export function groupByPage<T extends { page: number }>(items: readonly T[]): Map<number, T[]> {
  const out = new Map<number, T[]>();
  for (const item of items) {
    const bucket = out.get(item.page);
    if (bucket) bucket.push(item);
    else out.set(item.page, [item]);
  }
  return out;
}

/** The first page a set of boxes touches, for scroll targeting. */
export function firstPage(boxes: readonly PageBox[]): number | null {
  if (boxes.length === 0) return null;
  return boxes.reduce((min, b) => (b.page < min ? b.page : min), boxes[0]!.page);
}

/** Whether a normalized point falls inside a box.
 *
 * Used for reverse lookup: click a region on the sheet and jump to the question
 * it answers.
 */
export function contains(box: BBox, x: number, y: number): boolean {
  return x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1;
}

/** Convert a pointer event into normalized coordinates within an element.
 *
 * The inverse of {@link boxToStyle}, and the only other place the normalized
 * space is crossed.
 */
export function pointerToNormalized(
  event: { clientX: number; clientY: number },
  element: HTMLElement,
): { x: number; y: number } {
  const rect = element.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / rect.width,
    y: (event.clientY - rect.top) / rect.height,
  };
}

/** Area of a box in normalized units. Smaller boxes win hit-tests. */
export function area(box: BBox): number {
  return (box.x1 - box.x0) * (box.y1 - box.y0);
}
