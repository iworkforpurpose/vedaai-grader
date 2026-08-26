"use client";

import { useState } from "react";
import type { Line } from "@/lib/contracts";
import { boxToStyle } from "@/lib/geometry";

/**
 * Draws every transcribed box over its page.
 *
 * This is the most important view in the application during development, and it
 * earns that by being the only thing that can falsify the geometry. Every
 * highlight the product will ever draw is derived from these boxes, so if they
 * sit correctly on the words here, a misplaced highlight later is a mapping
 * fault; if they do not, nothing downstream can be trusted.
 *
 * Word boxes are shown separately because line and word geometry fail
 * differently: a line box can be right while the words inside it are shifted,
 * and rubric citations depend on the word boxes.
 */
export function DebugOverlay({
  lines,
  showWords,
  showText,
}: {
  lines: readonly Line[];
  showWords: boolean;
  showText: boolean;
}): React.JSX.Element {
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <>
      {lines.map((line) => {
        const style = boxToStyle(line.box);
        const isHovered = hovered === line.line_id;
        return (
          <div key={line.line_id}>
            <div
              onMouseEnter={() => setHovered(line.line_id)}
              onMouseLeave={() => setHovered(null)}
              title={`${line.line_id} · conf ${line.confidence.toFixed(2)} · ${line.text}`}
              style={{
                position: "absolute",
                ...style,
                border: `1px solid ${
                  line.is_low_confidence ? "var(--status-review)" : "var(--debug-line)"
                }`,
                background: isHovered ? "var(--hl-fill)" : "transparent",
                pointerEvents: "auto",
                cursor: "crosshair",
              }}
            />
            {showWords &&
              line.words.map((word, i) => (
                <div
                  key={`${line.line_id}-w${i}`}
                  style={{
                    position: "absolute",
                    ...boxToStyle(word.box),
                    outline: `1px solid var(--debug-word)`,
                  }}
                />
              ))}
            {showText && (
              <span
                style={{
                  position: "absolute",
                  left: style.left,
                  top: style.top,
                  transform: "translateY(-100%)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 8,
                  lineHeight: 1,
                  color: "var(--debug-line)",
                  background: "var(--surface)",
                  padding: "0 2px",
                  whiteSpace: "nowrap",
                }}
              >
                {line.line_id}
              </span>
            )}
          </div>
        );
      })}
    </>
  );
}
