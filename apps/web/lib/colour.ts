/**
 * Just enough colour maths to assert that text is readable.
 *
 * Exists because the palette is authored in OKLCH, and a contrast check that
 * cannot read OKLCH would have to run against a stale hex copy — which is the
 * same as not running it. Nothing here is used at runtime; it is test-only
 * support that lives in `lib/` so it typechecks with everything else.
 */

export interface Rgb {
  /** 0..1, sRGB, gamma-encoded. */
  r: number;
  g: number;
  b: number;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function parseHex(input: string): Rgb {
  const h = input.slice(1);
  const full =
    h.length === 3
      ? h
          .split("")
          .map((c) => c + c)
          .join("")
      : h;

  if (!/^[0-9a-f]{6}$/i.test(full)) {
    throw new Error(`Not a hex colour: ${input}`);
  }

  return {
    r: parseInt(full.slice(0, 2), 16) / 255,
    g: parseInt(full.slice(2, 4), 16) / 255,
    b: parseInt(full.slice(4, 6), 16) / 255,
  };
}

/**
 * `oklch(L% C H)`, with an optional `/ alpha` this deliberately ignores.
 *
 * Alpha is dropped rather than composited because a contrast assertion needs to
 * know what the reader sees, and what they see depends on what is behind it —
 * which a single colour string cannot say. Every pair asserted in the tests is
 * opaque for exactly that reason.
 */
function parseOklch(input: string): Rgb {
  const open = input.indexOf("(");
  const close = input.lastIndexOf(")");
  const body = input.slice(open + 1, close);
  const parts = (body.split("/")[0] ?? "").trim().split(/\s+/);

  const [rawL, rawC, rawH] = parts;
  if (rawL === undefined || rawC === undefined || rawH === undefined) {
    throw new Error(`Not an oklch colour: ${input}`);
  }

  const L = parseFloat(rawL) / (rawL.endsWith("%") ? 100 : 1);
  const C = parseFloat(rawC);
  const H = (parseFloat(rawH) * Math.PI) / 180;

  const a = C * Math.cos(H);
  const b = C * Math.sin(H);

  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;

  const l = l_ ** 3;
  const m = m_ ** 3;
  const s = s_ ** 3;

  return {
    r: clamp01(linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s)),
    g: clamp01(linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s)),
    b: clamp01(linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)),
  };
}

/** Accepts `#rgb`, `#rrggbb` and `oklch(L% C H)`. */
export function parseColour(input: string): Rgb {
  const value = input.trim();
  if (value.startsWith("#")) return parseHex(value);
  if (value.startsWith("oklch")) return parseOklch(value);
  throw new Error(`Unsupported colour: ${input}`);
}

/** WCAG 2.1 relative luminance. */
export function luminance(colour: Rgb): number {
  return (
    0.2126 * srgbToLinear(colour.r) +
    0.7152 * srgbToLinear(colour.g) +
    0.0722 * srgbToLinear(colour.b)
  );
}

/** WCAG 2.1 contrast ratio, 1..21. Argument order does not matter. */
export function contrastRatio(a: string, b: string): number {
  const la = luminance(parseColour(a));
  const lb = luminance(parseColour(b));
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}
