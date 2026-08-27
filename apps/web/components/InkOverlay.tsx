"use client";

import type { InkRegion, InkRegionKind } from "@/lib/contracts";
import { boxToStyle } from "@/lib/geometry";

/**
 * Draws ink regions, coloured by what they were classified as.
 *
 * Worth having beside the transcription overlay rather than instead of it,
 * because the interesting cases are exactly where the two disagree. Ink with no
 * line over it is either a diagram or a line the recognizer missed — both need
 * highlighting, and neither is visible from the transcription alone.
 */

const STYLE: Record<InkRegionKind, { colour: string; label: string }> = {
  writing: { colour: "var(--debug-ink)", label: "writing" },
  struck_through: { colour: "var(--status-missing)", label: "crossed out" },
  bleed_through: { colour: "var(--status-review)", label: "from reverse side" },
  noise: { colour: "var(--text-muted)", label: "not the page" },
};

export function InkOverlay({
  regions,
  showNoise,
  orphansOnly,
}: {
  regions: readonly InkRegion[];
  showNoise: boolean;
  orphansOnly: boolean;
}): React.JSX.Element {
  const visible = regions.filter((region) => {
    if (orphansOnly) return region.is_orphan_ink;
    if (region.kind === "noise") return showNoise;
    return true;
  });

  return (
    <>
      {visible.map((region) => {
        const style = STYLE[region.kind];
        return (
          <div
            key={region.region_id}
            title={
              `${region.region_id} · ${style.label}` +
              ` · ink ${(region.ink_ratio * 100).toFixed(0)}%` +
              ` · darkness ${region.mean_darkness.toFixed(2)}` +
              (region.covered_by_ocr ? " · transcribed" : " · NOT transcribed") +
              (region.has_horizontal_strike ? " · strike" : "")
            }
            style={{
              position: "absolute",
              ...boxToStyle(region.box),
              border: `2px ${region.covered_by_ocr ? "solid" : "dashed"} ${style.colour}`,
              // Dashed means the recognizer reported nothing here. That is the
              // case ink exists to cover, so it should be visually distinct at a
              // glance rather than only in a tooltip.
              background: region.is_orphan_ink ? "var(--hl-fill-ink)" : "transparent",
              pointerEvents: "auto",
              cursor: "crosshair",
            }}
          />
        );
      })}
    </>
  );
}

/** Counts per classification, for the toolbar summary. */
export function summarizeInk(regions: readonly InkRegion[]): {
  counts: Record<string, number>;
  orphans: number;
  substantive: number;
} {
  const counts: Record<string, number> = {};
  let orphans = 0;
  let substantive = 0;
  for (const region of regions) {
    counts[region.kind] = (counts[region.kind] ?? 0) + 1;
    if (region.is_orphan_ink) orphans += 1;
    if (region.is_substantive) substantive += 1;
  }
  return { counts, orphans, substantive };
}
