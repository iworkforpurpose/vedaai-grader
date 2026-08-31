import type { Page } from "@/lib/contracts";
import { API_BASE } from "@/lib/api";

/**
 * One page image with an absolutely-positioned overlay layer above it.
 *
 * Presentational only: it knows how to stack an image and its overlay, and
 * nothing about lines, questions or highlights. Overlays are passed as children
 * so this component never needs to change as new kinds of annotation are added.
 *
 * The image is unoptimized on purpose. Next's image pipeline would resample
 * these, and the overlay's correctness depends on the rendered image being the
 * exact page the geometry was computed against.
 */
export function PageCanvas({
  page,
  children,
  label,
}: {
  page: Page;
  children?: React.ReactNode;
  label?: string;
}): React.JSX.Element {
  return (
    <figure
      style={{
        margin: 0,
        display: "flex",
        flexDirection: "column",
        gap: "var(--sp-2)",
      }}
    >
      {label !== undefined && (
        <figcaption
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "var(--fs-xs)",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "var(--text-muted)",
          }}
        >
          {label}
        </figcaption>
      )}
      <div
        data-page={page.index}
        style={{
          position: "relative",
          // Reserve the correct space before the image loads, so the overlay
          // never renders against a collapsed box and then jump on load.
          aspectRatio: `${page.width} / ${page.height}`,
          width: "100%",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--r-sm)",
          overflow: "hidden",
          boxShadow: "var(--shadow)",
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`${API_BASE}/pages/${page.image_key}`}
          alt={`Page ${page.index + 1}`}
          style={{ display: "block", width: "100%", height: "100%" }}
        />
        <div
          style={{
            position: "absolute",
            inset: 0,
            // Overlays are annotation, not interaction. Individual children
            // opt back in where they need to be clickable.
            pointerEvents: "none",
          }}
        >
          {children}
        </div>
      </div>
    </figure>
  );
}
