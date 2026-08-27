"use client";

import { useEffect, useRef, useState } from "react";
import type { InkRegion, Page, PageBox } from "@/lib/contracts";
import { boxToStyle, pointerToNormalized } from "@/lib/geometry";
import { API_BASE } from "@/lib/api";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  MinusIcon,
  PlusIcon,
} from "./icons";

/**
 * The answer sheet, with the dark toolbar the frame specifies.
 *
 * Highlights are positioned as percentages of their page container, so they track
 * the image at any rendered width and any zoom with no measurement and no resize
 * listener. That is the entire payoff of normalizing coordinates in the pipeline:
 * the browser never divides by DPI, so it cannot get the arithmetic wrong.
 *
 * The frame shows a page counter and paging arrows. Pages are still rendered as
 * one continuous scroll rather than one at a time, because a page-spanning answer
 * has to read as a single region carrying over the boundary — paging would split
 * it into two rectangles the teacher has to mentally rejoin. The arrows scroll to
 * a page instead of swapping it, which satisfies the design without breaking the
 * one case the product exists to handle.
 */
export function SheetView({
  pages,
  highlights,
  highlightLabel,
  untranscribedInk,
  showUntranscribed,
  onPointerPick,
  scrollTarget,
}: {
  pages: readonly Page[];
  highlights: Map<number, PageBox[]>;
  highlightLabel: string | null;
  untranscribedInk: Map<number, InkRegion[]>;
  showUntranscribed: boolean;
  onPointerPick: (page: number, x: number, y: number) => void;
  scrollTarget: { page: number; y: number; nonce: number } | null;
}): React.JSX.Element {
  const scroller = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [current, setCurrent] = useState(0);

  // Scroll to a page when a question is chosen, and again if the same question is
  // chosen twice — hence the nonce.
  useEffect(() => {
    if (scrollTarget === null || scroller.current === null) return;
    const target = scroller.current.querySelector<HTMLElement>(
      `[data-page="${scrollTarget.page}"]`,
    );
    if (target === null) return;
    const top = target.offsetTop + target.offsetHeight * scrollTarget.y;
    scroller.current.scrollTo({
      top: Math.max(0, top - scroller.current.clientHeight * 0.2),
      behavior: "smooth",
    });
  }, [scrollTarget]);

  // Which page is in view, for the counter. Read from scroll position rather than
  // tracked separately, so it cannot disagree with what is on screen.
  useEffect(() => {
    const node = scroller.current;
    if (node === null) return;
    const onScroll = () => {
      const mid = node.scrollTop + node.clientHeight / 2;
      const pageNodes = Array.from(node.querySelectorAll<HTMLElement>("[data-page]"));
      const index = pageNodes.findIndex(
        (el) => mid >= el.offsetTop && mid < el.offsetTop + el.offsetHeight,
      );
      if (index >= 0) setCurrent(index);
    };
    node.addEventListener("scroll", onScroll, { passive: true });
    return () => node.removeEventListener("scroll", onScroll);
  }, [pages.length]);

  function goToPage(index: number): void {
    const node = scroller.current;
    const target = node?.querySelector<HTMLElement>(`[data-page="${index}"]`);
    if (!node || !target) return;
    node.scrollTo({ top: target.offsetTop, behavior: "smooth" });
  }

  const count = pages.length;

  return (
    <>
      <div className="sheet-bar">
        <h2>Answer Sheet</h2>

        <div className="sheet-tools">
          <div className="tool-group">
            <button
              type="button"
              className="tool-button"
              aria-label="Zoom out"
              disabled={zoom <= 0.5}
              onClick={() => setZoom((z) => Math.max(0.5, Math.round((z - 0.25) * 100) / 100))}
            >
              <MinusIcon size={16} />
            </button>
            <span className="tool-value">{Math.round(zoom * 100)}%</span>
            <button
              type="button"
              className="tool-button"
              aria-label="Zoom in"
              disabled={zoom >= 3}
              onClick={() => setZoom((z) => Math.min(3, Math.round((z + 0.25) * 100) / 100))}
            >
              <PlusIcon size={16} />
            </button>
          </div>

          {count > 1 && (
            <div className="tool-group">
              <button
                type="button"
                className="tool-button"
                aria-label="Previous page"
                disabled={current <= 0}
                onClick={() => goToPage(current - 1)}
              >
                <ChevronLeftIcon size={16} />
              </button>
              <span className="tool-value">
                Page {current + 1} of {count}
              </span>
              <button
                type="button"
                className="tool-button"
                aria-label="Next page"
                disabled={current >= count - 1}
                onClick={() => goToPage(current + 1)}
              >
                <ChevronRightIcon size={16} />
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="sheet-scroll" ref={scroller}>
        {pages.map((page, index) => (
          <div
            key={page.image_key}
            className="sheet-page"
            data-page={index}
            style={{ width: `${zoom * 100}%` }}
            onClick={(event) => {
              const { x, y } = pointerToNormalized(event, event.currentTarget);
              if (x >= 0 && x <= 1 && y >= 0 && y <= 1) onPointerPick(index, x, y);
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${API_BASE}/pages/${page.image_key}`}
              alt={`Page ${index + 1} of the answer sheet`}
              width={page.width}
              height={page.height}
            />

            {showUntranscribed &&
              (untranscribedInk.get(index) ?? []).map((region) => (
                <span
                  key={region.region_id}
                  className="hl"
                  data-kind="ink"
                  style={boxToStyle(region.box)}
                  title="Writing the recognizer did not read. If a question reads 'not found', look here."
                />
              ))}

            {(highlights.get(index) ?? []).map((pageBox, i) => (
              <span key={`${index}-${i}`} className="hl" style={boxToStyle(pageBox.box)}>
                {/*
                 * The tab carries the question label, which is what makes a
                 * highlight self-describing when more than one is on screen.
                 *
                 * It sits above the box as the frame draws it, and flips inside
                 * when there is no room — an answer at the very top of a page put
                 * the tab off the page entirely, which is exactly where a first
                 * answer tends to be.
                 */}
                {i === 0 && highlightLabel && (
                  <span className="hl-tab" data-inside={pageBox.box.y0 < 0.04}>
                    {highlightLabel}
                  </span>
                )}
              </span>
            ))}
          </div>
        ))}
      </div>
    </>
  );
}
