"use client";

import { useEffect, useRef } from "react";
import type { InkRegion, Page, PageBox } from "@/lib/contracts";
import { boxToStyle, pointerToNormalized } from "@/lib/geometry";
import { PageCanvas } from "./PageCanvas";

/**
 * The answer sheet, as one continuous scroll of pages.
 *
 * Continuous rather than paged, and that is what makes a page-spanning answer
 * legible: its highlight reads as a single region carrying over the boundary
 * instead of two disconnected rectangles the teacher has to mentally join.
 *
 * Highlights are positioned as percentages of their page container, so they
 * track the image at any rendered width with no measurement and no resize
 * listener. That is the entire payoff of normalizing coordinates in the pipeline.
 */
export function AnswerSheetView({
  pages,
  highlights,
  untranscribedInk,
  showUntranscribed,
  onPointerPick,
  scrollToPage,
}: {
  pages: readonly Page[];
  highlights: Map<number, PageBox[]>;
  untranscribedInk: Map<number, InkRegion[]>;
  showUntranscribed: boolean;
  onPointerPick: (page: number, x: number, y: number) => void;
  scrollToPage: { page: number; y: number; nonce: number } | null;
}): React.JSX.Element {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollToPage === null || container.current === null) return;
    const target = container.current.querySelector<HTMLElement>(
      `[data-page="${scrollToPage.page}"]`,
    );
    if (target === null) return;

    // Scroll so the highlight sits a little below the top edge rather than
    // flush against it, which reads as cut off.
    const offset = target.offsetTop + target.offsetHeight * scrollToPage.y;
    container.current.scrollTo({
      top: Math.max(0, offset - container.current.clientHeight * 0.25),
      behavior: "smooth",
    });
    // nonce is included so clicking the same question twice scrolls again.
  }, [scrollToPage]);

  return (
    <div
      ref={container}
      style={{
        overflowY: "auto",
        overflowX: "hidden",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        gap: "var(--sp-4)",
        padding: "var(--sp-4)",
        background: "var(--bg)",
      }}
    >
      {pages.map((page) => (
        <div
          key={page.image_key}
          onClick={(event) => {
            const host = event.currentTarget.querySelector<HTMLElement>("[data-page]");
            if (host === null) return;
            const { x, y } = pointerToNormalized(event, host);
            if (x >= 0 && x <= 1 && y >= 0 && y <= 1) onPointerPick(page.index, x, y);
          }}
        >
          <PageCanvas page={page} label={`Page ${page.index + 1}`}>
            {showUntranscribed &&
              (untranscribedInk.get(page.index) ?? []).map((region) => (
                <div
                  key={region.region_id}
                  title="Writing the recognizer did not read. If a question reads 'not found', look here."
                  style={{
                    position: "absolute",
                    ...boxToStyle(region.box),
                    border: "2px dashed var(--hl-stroke-ink)",
                    background: "var(--hl-fill-ink)",
                  }}
                />
              ))}

            {(highlights.get(page.index) ?? []).map((pageBox, index) => (
              <div
                key={`${pageBox.page}-${index}`}
                style={{
                  position: "absolute",
                  ...boxToStyle(pageBox.box),
                  border: "2px solid var(--hl-stroke)",
                  background: "var(--hl-fill)",
                  borderRadius: "var(--radius-sm)",
                }}
              />
            ))}
          </PageCanvas>
        </div>
      ))}
    </div>
  );
}
