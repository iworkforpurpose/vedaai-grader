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

export function LockIcon({ size = 22 }: IconProps): React.JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect
        x="4.5"
        y="10.5"
        width="15"
        height="10"
        rx="3"
        stroke="currentColor"
        strokeWidth="1.8"
      />
      <path
        d="M8 10.5V8a4 4 0 0 1 8 0v2.5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <circle cx="12" cy="15.5" r="1.4" fill="currentColor" />
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
/*
 * The loader mark, exported from the file rather than drawn by eye.
 *
 * One `d` for all three sparkles: the file's own three are the same outline at
 * 1, 0.75 and 0.3 scale, which is why the transforms below are exact rather than
 * fitted. Every number here is read off the node — the 128.1543 x 134.4925 box,
 * the four offsets, the 0.5198 and 0.83 group opacities — and the outline is the
 * node's own `fillGeometry` path.
 *
 * Two things the previous version got wrong, both of them the reason it read as a
 * different graphic rather than a slightly-off one:
 *
 *   The outline was a hand-drawn symmetric four-point star. The file's star is not
 *   symmetric — each arm's control points differ, so the waist of every arm sits
 *   in a different place.
 *
 *   Every sparkle carries a white inner shadow at zero offset, which is what makes
 *   the arms fade out toward their tips. Painted flat, the shape is a hard orange
 *   diamond; the fade is most of what the mark looks like. The API reports the
 *   fill as SOLID, so this only shows up in `effects` — worth naming, because
 *   reading the fills and stopping there is exactly what produced the flat one.
 */
const SPARKLE_PATH =
  "M0 47.8277C37.4479 47.5563 47.4883 15.8295 47.8276 0C47.8276 37.7194 79.7126 " +
  "47.6016 95.6557 47.8277C57.6645 47.285 47.9407 79.7129 47.8276 95.9946C47.8276 " +
  "57.1897 15.9426 47.7147 0 47.8277Z";

/**
 * Figma's zero-offset inner shadow, as a filter.
 *
 * Blur the *inverted* alpha, keep only what falls inside the shape, flood it with
 * the shadow colour, and lay that over the fill. That is what "a shadow cast
 * inward from the edges" reduces to.
 *
 * `stdDeviation` is half the Figma radius, the same relation CSS uses between a
 * box-shadow blur and its Gaussian sigma. It is in viewBox units, which is why the
 * filter sits above the per-sparkle `scale()` rather than inside it — inside, the
 * 0.3 sparkle would have its glow shrunk to a third of the specified width.
 *
 * `color-interpolation-filters="sRGB"` is not optional. Filters default to
 * linearRGB, where a blurred white glow spreads visibly wider and paler than the
 * same glow composited in sRGB, which is what the design tool did.
 */
function InnerGlow({ id, radius }: { id: string; radius: number }): React.JSX.Element {
  return (
    <filter
      id={id}
      x="-40%"
      y="-40%"
      width="180%"
      height="180%"
      colorInterpolationFilters="sRGB"
    >
      <feComponentTransfer in="SourceAlpha" result="inverted">
        <feFuncA type="table" tableValues="1 0" />
      </feComponentTransfer>
      <feGaussianBlur in="inverted" stdDeviation={radius / 2} result="spread" />
      <feFlood floodColor="#FFFFFF" floodOpacity="1" result="tint" />
      <feComposite in="tint" in2="spread" operator="in" result="edge" />
      <feComposite in="edge" in2="SourceAlpha" operator="in" result="glow" />
      <feMerge>
        <feMergeNode in="SourceGraphic" />
        <feMergeNode in="glow" />
      </feMerge>
    </filter>
  );
}

export function LoaderMark(): React.JSX.Element {
  return (
    <svg
      className="loader-mark"
      viewBox="0 0 128.1543 134.4925"
      fill="#FF5623"
      aria-hidden
    >
      <defs>
        <InnerGlow id="mark-glow-7" radius={7.4996} />
        <InnerGlow id="mark-glow-5" radius={4.9997} />
        <InnerGlow id="mark-glow-10" radius={9.9994} />
      </defs>

      {/*
        * Three nested groups per sparkle, each with one job: the effect and group
        * opacity outermost in viewBox units, then placement, then the animation.
        *
        * Split because a `transform-origin` written in viewBox units on the same
        * element that carries `scale(0.75)` resolves in that element's *scaled*
        * space and lands somewhere else — which threw the cluster off position and
        * pushed the small sparkle out of the box entirely. The innermost group has
        * no transform of its own, so one origin is correct for all three.
        */}

      {/* 95.6557 x 95.9946 at 32.498, 0 */}
      <g filter="url(#mark-glow-7)">
        <g transform="translate(32.498 0)">
          <g className="mark-twinkle" data-s="lg">
            <path d={SPARKLE_PATH} />
          </g>
        </g>
      </g>

      {/* 71.7414 x 71.996 at 12.5, 62.4965 — the same outline at 0.75 */}
      <g filter="url(#mark-glow-7)">
        <g transform="translate(12.5 62.4965) scale(0.75)">
          <g className="mark-twinkle" data-s="md">
            <path d={SPARKLE_PATH} />
          </g>
        </g>
      </g>

      {/* 28.6966 x 28.7984 at 89.995, 83.7454 — at 0.3, group opacity 0.5198 */}
      <g filter="url(#mark-glow-5)" opacity="0.5198">
        <g transform="translate(89.995 83.7454) scale(0.3)">
          <g className="mark-twinkle" data-s="sm">
            <path d={SPARKLE_PATH} />
          </g>
        </g>
      </g>

      {/* 12.5 circle at 17.499, 47.4974 — group opacity 0.83, and a radius-10 glow
          on a 12.5px shape, which is why it reads as pale rather than solid */}
      <g filter="url(#mark-glow-10)" opacity="0.83">
        <circle data-s="dot" cx="23.749" cy="53.7474" r="6.25" />
      </g>
    </svg>
  );
}
