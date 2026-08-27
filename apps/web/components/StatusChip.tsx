import type { StatusPresentation } from "@/lib/review";

/**
 * The status of one question, as a chip.
 *
 * Carries a shape as well as a colour. Colour alone excludes anyone who cannot
 * distinguish the hues, and this is the single most consequential thing on the
 * page — a teacher scans these to decide where to look.
 */
export function StatusChip({
  presentation,
  compact,
}: {
  presentation: StatusPresentation;
  compact?: boolean;
}): React.JSX.Element {
  return (
    <span
      title={presentation.hint}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--sp-1)",
        fontFamily: "var(--font-mono)",
        fontSize: "var(--fs-xs)",
        letterSpacing: "0.04em",
        textTransform: "uppercase",
        color: presentation.colour,
        border: `1px solid ${presentation.colour}`,
        borderRadius: "var(--radius-sm)",
        padding: compact ? "0 4px" : "1px 6px",
        whiteSpace: "nowrap",
      }}
    >
      {presentation.needsAttention && <span aria-hidden="true">!</span>}
      {presentation.label}
    </span>
  );
}
