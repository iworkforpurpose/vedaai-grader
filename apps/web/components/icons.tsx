/**
 * Line icons, inline.
 *
 * Inline rather than an icon package because these are the only icons the
 * product uses, and a dependency that ships a thousand of them to deliver
 * fifteen is a poor trade in a container image.
 *
 * They are traced to match the weight and corner treatment of the Figma frames —
 * 1.6 stroke, round caps, 24-unit box — not copied from them. The frames were
 * supplied as flattened PNGs, so the original vectors were not available to
 * export. That is a real limitation and is recorded here rather than hidden: if
 * the exported SVGs arrive later, these are a drop-in replacement point.
 */

type IconProps = {
  size?: number;
  className?: string;
};

function svgProps(size: number) {
  return {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.6,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };
}

export function HomeIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <rect x="3" y="3" width="7.5" height="7.5" rx="2" />
      <rect x="13.5" y="3" width="7.5" height="7.5" rx="2" />
      <rect x="3" y="13.5" width="7.5" height="7.5" rx="2" />
      <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="2" />
    </svg>
  );
}

export function ClassroomIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <rect x="2.5" y="5" width="19" height="13" rx="2.5" />
      <path d="M8 18v2h8v-2" />
      <path d="M7 13.5l3-3 2.5 2.5 3-4" />
    </svg>
  );
}

export function AssignmentsIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
      <path d="M9 13h6M9 17h4" />
    </svg>
  );
}

export function ExamsIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <rect x="5" y="4" width="14" height="17" rx="2.5" />
      <path d="M9.5 4V3h5v1" />
      <path d="M9 11h6M9 15h4" />
    </svg>
  );
}

export function LibraryIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3v9h9" />
    </svg>
  );
}

export function SettingsIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1.03 1.56V21a2 2 0 1 1-4 0v-.09A1.7 1.7 0 0 0 8.9 19.3a1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.56-1.03H3a2 2 0 1 1 0-4h.09A1.7 1.7 0 0 0 4.65 8.9a1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1.03-1.56V3a2 2 0 1 1 4 0v.09A1.7 1.7 0 0 0 15.1 4.65a1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.7 1.7 0 0 0 19.4 9v.09A1.7 1.7 0 0 0 21 10.1h.09a2 2 0 1 1 0 4H21a1.7 1.7 0 0 0-1.6 1.03z" />
    </svg>
  );
}

export function ReviewIcon({ size = 20 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
      <path d="M12 12v5M9.5 14.5h5" />
    </svg>
  );
}

export function AnalyticsIcon({ size = 20 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M4 19.5h16" />
      <path d="M7 19.5v-6M12 19.5V8M17 19.5v-9" />
    </svg>
  );
}

export function BackIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M15 5l-7 7 7 7" />
    </svg>
  );
}

export function ArrowRightIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M4 12h15" />
      <path d="M13 6l6 6-6 6" />
    </svg>
  );
}

export function HelpIcon({ size = 20 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <circle cx="12" cy="12" r="9" />
      <path d="M9.6 9.2a2.5 2.5 0 1 1 3.6 2.24c-.75.42-1.2 1-1.2 1.86v.2" />
      <path d="M12 17.3h.01" />
    </svg>
  );
}

export function BellIcon({ size = 20 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M18 9a6 6 0 1 0-12 0c0 5-2 6-2 6h16s-2-1-2-6" />
      <path d="M10.3 19a2 2 0 0 0 3.4 0" />
    </svg>
  );
}

export function SparkleIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M12 2.5l1.7 4.6 4.6 1.7-4.6 1.7L12 15.1l-1.7-4.6L5.7 8.8l4.6-1.7z" />
      <path d="M18.6 14.4l.85 2.3 2.3.85-2.3.85-.85 2.3-.85-2.3-2.3-.85 2.3-.85z" />
    </svg>
  );
}

export function ChevronDownIcon({ size = 16 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M6 9.5l6 6 6-6" />
    </svg>
  );
}

export function MenuIcon({ size = 20 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M4 7h16M4 12h16M4 17h16" />
    </svg>
  );
}

export function PanelIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <rect x="3" y="4" width="18" height="16" rx="2.5" />
      <path d="M9.5 4v16" />
    </svg>
  );
}

export function UploadIcon({ size = 22 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M12 16V4" />
      <path d="M7.5 8.5L12 4l4.5 4.5" />
      <path d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16" />
    </svg>
  );
}

export function VedaMark({ size = 22 }: IconProps): React.JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4.5 5.5h4.2L12 15l3.3-9.5h4.2L14 20h-4z"
        fill="currentColor"
      />
    </svg>
  );
}

/** Small glyphs for the badges orbiting the hero portrait. */
export function ClockGlyph({ size = 12 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3.5 2" />
    </svg>
  );
}

export function GridGlyph({ size = 12 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <path d="M4 10h16M10 4v16" />
    </svg>
  );
}

export function CloudGlyph({ size = 12 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M6.5 18a4 4 0 0 1 .4-8 5.5 5.5 0 0 1 10.5 1.4A3.6 3.6 0 0 1 17.5 18z" />
    </svg>
  );
}

export function GearGlyph({ size = 12 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <circle cx="12" cy="12" r="3.2" />
      <path d="M12 3v2.4M12 18.6V21M3 12h2.4M18.6 12H21M5.6 5.6l1.7 1.7M16.7 16.7l1.7 1.7M18.4 5.6l-1.7 1.7M7.3 16.7l-1.7 1.7" />
    </svg>
  );
}

export function ChevronLeftIcon({ size = 20 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M14 6l-6 6 6 6" />
    </svg>
  );
}

export function ChevronRightIcon({ size = 20 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M10 6l6 6-6 6" />
    </svg>
  );
}

export function ChevronsRightIcon({ size = 20 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M6 6l6 6-6 6M13 6l6 6-6 6" />
    </svg>
  );
}

export function MinusIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M5 12h14" />
    </svg>
  );
}

export function PlusIcon({ size = 18 }: IconProps): React.JSX.Element {
  return (
    <svg {...svgProps(size)}>
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

/**
 * One four-pointed sparkle, filled.
 *
 * The loading frame composes four of these at 96, 72, 29 and a 12px dot, all in
 * the accent. Drawn as a single reusable shape rather than four hand-tuned paths.
 */
export function SparkleShape({ size = 96 }: IconProps): React.JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="currentColor" aria-hidden>
      <path d="M50 0c3.4 22.4 10.9 34 27.6 38.1C89.4 41 95.6 44.6 100 50c-22.4 3.4-34 10.9-38.1 27.6C59 89.4 55.4 95.6 50 100c-3.4-22.4-10.9-34-27.6-38.1C10.6 59 4.4 55.4 0 50c22.4-3.4 34-10.9 38.1-27.6C41 10.6 44.6 4.4 50 0z" />
    </svg>
  );
}

export function DotShape({ size = 12 }: IconProps): React.JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" fill="currentColor" aria-hidden>
      <circle cx="6" cy="6" r="6" />
    </svg>
  );
}
