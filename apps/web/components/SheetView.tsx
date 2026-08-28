"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { animateScrollTo, isStillLoading } from "@/lib/motion";
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
  orphanRegions,
  showUntranscribed,
  onToggleUntranscribed,
  onPointerPick,
  scrollTarget,
}: {
  pages: readonly Page[];
  highlights: Map<number, PageBox[]>;
  highlightLabel: string | null;
  untranscribedInk: Map<number, InkRegion[]>;
  /** Writing that matched no question, always shown — it needs a decision. */
  orphanRegions: Map<number, PageBox[]>;
  showUntranscribed: boolean;
  onToggleUntranscribed: () => void;
  onPointerPick: (page: number, x: number, y: number) => void;
  scrollTarget: { page: number; y: number; nonce: number } | null;
}): React.JSX.Element {
  const scroller = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [current, setCurrent] = useState(0);
  /*
   * Which page bitmaps have *not* arrived yet — the inverse of what this tracked.
   *
   * Tracking "loaded" made an invisible sheet the default and relied on `onLoad`
   * to undo it. `onLoad` does not fire for an image that finished before React
   * attached the handler, which is every cached image and, because the `src` is in
   * the server-rendered HTML, usually the first load too: the browser starts
   * fetching during parse and is done before hydration. The result was a fully
   * decoded answer sheet at `opacity: 0`, permanently.
   *
   * Inverted, the failure mode is safe. No entry means no fade, so anything this
   * does not know about — including the case where none of this runs — is visible.
   */
  const [pending, setPending] = useState<Set<string>>(new Set());

  // Scroll to a page when a question is chosen, and again if the same question is
  // chosen twice — hence the nonce.
  useEffect(() => {
    if (scrollTarget === null || scroller.current === null) return;
    const node = scroller.current;
    const target = node.querySelector<HTMLElement>(`[data-page="${scrollTarget.page}"]`);
    if (target === null) return;
    // Our own tween, not `behavior: "smooth"`: that is about 300ms flat and not
    // tunable, so crossing a page boundary read as a cut rather than travel. This
    // eases over 450-950ms by distance, and aborts if the reader starts scrolling.
    return animateScrollTo(node, offsetWithin(node, target, scrollTarget.y));
  }, [scrollTarget]);

  /*
   * Which page is in view, for the counter.
   *
   * An observer rather than a scroll handler. The handler this replaces ran
   * `querySelectorAll` plus a `getBoundingClientRect()` per page on every scroll
   * event, unthrottled — each rect read forces a synchronous layout, so scrolling
   * a two-page script paid a reflow and a React render per event. That is the
   * frame budget any animation elsewhere has to come out of.
   *
   * The root margin collapses the viewport to a band across its middle, so "the
   * page in view" means the one under the middle of the window, which is the same
   * rule as before and the one that matches what a reader would say.
   */
  useEffect(() => {
    const node = scroller.current;
    if (node === null) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const hit = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!hit) return;
        const index = Number((hit.target as HTMLElement).dataset.page);
        if (!Number.isNaN(index)) setCurrent(index);
      },
      { root: node, rootMargin: "-50% 0px -50% 0px", threshold: 0 },
    );

    node.querySelectorAll<HTMLElement>("[data-page]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [pages.length]);

  function goToPage(index: number): void {
    const node = scroller.current;
    const target = node?.querySelector<HTMLElement>(`[data-page="${index}"]`);
    if (!node || !target) return;
    animateScrollTo(node, offsetWithin(node, target, 0));
  }

  // Returns the same Set when nothing changes, so a ref callback firing on every
  // commit cannot loop.
  const mark = useCallback((key: string, isPending: boolean): void => {
    setPending((current) => {
      if (current.has(key) === isPending) return current;
      const next = new Set(current);
      if (isPending) next.add(key);
      else next.delete(key);
      return next;
    });
  }, []);

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

          {/*
            * The unread-ink toggle lives here, with zoom and paging, because it is
            * a control over this view rather than over the submission. It was a
            * checkbox below both panes, where it was both misplaced and the last
            * 24px that pushed the review screen past the viewport.
            */}
          {untranscribedInk.size > 0 && (
            <button
              type="button"
              className="tool-toggle"
              aria-pressed={showUntranscribed}
              onClick={onToggleUntranscribed}
              title="Outline writing the recognizer did not read. If a question reads 'not found', look here."
            >
              Unread ink
            </button>
          )}

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
            {/*
              * Faded in on decode. A scanned page is a large bitmap over the
              * network, and appearing at full opacity the instant the last byte
              * lands is the single most abrupt thing on this screen.
              *
              * eslint-disable-next-line @next/next/no-img-element
              */}
            <img
              src={`${API_BASE}/pages/${page.image_key}`}
              alt={`Page ${index + 1} of the answer sheet`}
              width={page.width}
              height={page.height}
              decoding="async"
              data-pending={pending.has(page.image_key)}
              ref={(node) => {
                // Read at attach, because an image that finished before this ran
                // will never fire onLoad. See isStillLoading for why that test is
                // `complete` and nothing else.
                if (node && isStillLoading(node)) mark(page.image_key, true);
              }}
              onLoad={() => mark(page.image_key, false)}
              // A broken image must not stay hidden, or the alt text goes with it.
              onError={() => mark(page.image_key, false)}
            />

            {/*
              * Unplaced writing, outlined amber and always visible.
              *
              * Not behind a toggle like the unread-ink overlay. Unread ink is a
              * diagnostic a teacher consults when something looks wrong; an orphan is
              * a decision waiting to be made, and one hidden behind a checkbox is one
              * that does not get made.
              */}
            {(orphanRegions.get(index) ?? []).map((pageBox, i) => (
              <span
                key={`orphan-${index}-${i}`}
                className="hl"
                data-kind="orphan"
                style={boxToStyle(pageBox.box)}
                title="This writing matched no question. Place it from the list."
              >
                {i === 0 && <span className="hl-tab" data-kind="orphan">?</span>}
              </span>
            ))}

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
                  <span className="hl-tab">{highlightLabel}</span>
                )}
              </span>
            ))}
          </div>
        ))}
      </div>
    </>
  );
}


/**
 * Where to scroll the sheet so that a fraction `y` down `target` lands just inside
 * the top of `scroller`, with room to see the highlight's label.
 *
 * Measured from bounding rects rather than `offsetTop`, which was the bug this
 * replaces. `offsetTop` is relative to the nearest *positioned* ancestor, and the
 * scroller is not positioned — so the value carried the toolbar and the pane's own
 * offset with it. The scroll overshot by exactly that much, which put the top edge
 * of the highlight above the visible area and hid its label behind the toolbar:
 * the one part of a highlight that has to be visible was the one part reliably
 * scrolled out of view.
 *
 * The inset is a fixed 24px rather than a fraction of the viewport. A fraction
 * reads as "a bit of context above" on a laptop and as three pixels on a phone,
 * and this is the case where being able to see the label is the whole point.
 */
function offsetWithin(scroller: HTMLElement, target: HTMLElement, y: number): number {
  const INSET = 24;
  const host = scroller.getBoundingClientRect();
  const rect = target.getBoundingClientRect();
  const top = scroller.scrollTop + (rect.top - host.top) + rect.height * y;
  return Math.max(0, top - INSET);
}
