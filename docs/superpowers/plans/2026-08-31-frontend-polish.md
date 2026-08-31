# Frontend Polish Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `apps/web` read as a finished product rather than a clean prototype, by giving surfaces a real elevation ladder, making status colour mean something, fixing the contrast failures and undefined tokens found while reading the code, and adding motion that leaves as well as arrives.

**Architecture:** Token spine first in `app/globals.css`, proved end-to-end on the review screen, then swept onto the upload screen. Two new pure modules under `lib/` carry the parts that can be tested without a browser — token integrity and colour contrast — so the defect classes found during design cannot come back silently. Everything else is CSS, verified by eye at four widths.

**Tech Stack:** Next 15 App Router, React 19, plain CSS with custom properties (no Tailwind, no CSS-in-JS build step), Vitest, `next/font/google` for self-hosted faces.

## Global Constraints

- **Local only.** Never run `deploy/deploy.sh`, never `git push`, never trigger CI. The dev server on port 3001 is the delivery mechanism.
- **Branch:** all work lands on `frontend-polish`. Never commit to `main`.
- **Figma is intent, not a pixel target.** Surface treatment may change; measured *geometry* in `design/SPEC.md` may not. Every deviation gets a line in that file's "Divergences, and why" section, in the same commit that causes it.
- **Dark mode is out of scope.** Do not add `light-dark()` pairs or a `prefers-color-scheme` block. Declare `color-scheme: light`.
- **Bricolage Grotesque stays the voice.** The only new face is JetBrains Mono, for numerals and metadata.
- **Accessibility floor:** every foreground/background pair carrying text at under 18.66px or under 700 weight must reach **4.5:1**. Status must never be signalled by colour alone.
- **`prefers-reduced-motion: reduce` must survive every change.** The existing global override stays; no new animation may bypass it.
- Type sizes stay in `rem`. Do not reintroduce `px` font sizes — WCAG 1.4.4.
- Run `pnpm --filter @vedaai/web typecheck` and `pnpm --filter @vedaai/web test` before every commit.

---

### Task 1: Delete the dead components

Three components are imported by nothing. They carry most of the undefined-token references, which is why the defects survived. Deleting them first means the integrity test in Task 2 measures only live code.

**Files:**
- Delete: `apps/web/components/QuestionList.tsx`
- Delete: `apps/web/components/GradePanel.tsx`
- Delete: `apps/web/components/StatusChip.tsx`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. `StatusPresentation.colour` in `lib/review.ts` loses its only consumer here and gains a new one in Task 4.

- [ ] **Step 1: Prove they are unreferenced**

Run:
```bash
cd /Users/vighneshnama/Vedaai-assignment/apps/web
grep -rn "QuestionList\|GradePanel\|StatusChip" --include="*.tsx" --include="*.ts" components/ app/ lib/ \
  | grep -v "^components/QuestionList.tsx\|^components/GradePanel.tsx\|^components/StatusChip.tsx"
```
Expected: no output. Any output means something imports them — stop and report rather than deleting.

- [ ] **Step 2: Delete**

```bash
git rm apps/web/components/QuestionList.tsx apps/web/components/GradePanel.tsx apps/web/components/StatusChip.tsx
```

- [ ] **Step 3: Verify nothing broke**

Run: `pnpm --filter @vedaai/web typecheck && pnpm --filter @vedaai/web test`
Expected: typecheck clean, all existing vitest suites pass.

- [ ] **Step 4: Commit**

```bash
git commit -m "Remove three components nothing renders

QuestionList, GradePanel and StatusChip are imported by no file. Between them
they hold most of the custom properties the stylesheet never defines, which is
why those went unnoticed for so long: a dropped declaration in code that never
runs looks exactly like code that works.

StatusChip's one good idea is kept rather than deleted with it -- a status
needs a channel that is not colour, so the glyph it used moves onto the chip
the review screen actually renders."
```

---

### Task 2: Token integrity test, and define what is missing

**Files:**
- Create: `apps/web/lib/tokens.test.ts`
- Modify: `apps/web/app/globals.css` (token block)
- Modify: `apps/web/components/PageCanvas.tsx:56`

**Interfaces:**
- Consumes: nothing.
- Produces: a guarantee later tasks rely on — any `var(--x)` written without a fallback must resolve. Later tasks add tokens freely knowing the test catches typos.

- [ ] **Step 1: Write the failing test**

Create `apps/web/lib/tokens.test.ts`:

```ts
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const WEB = join(__dirname, "..");

/**
 * Every custom property the app reads must be one the stylesheet defines.
 *
 * `var(--missing)` is not an error anywhere in the platform: the declaration is
 * simply dropped, so a square corner or an uncoloured chip is the only symptom.
 * Four such properties shipped before this test existed.
 *
 * A `var()` carrying a fallback is exempt. That is the documented way to read a
 * property set from somewhere the stylesheet cannot see -- an inline style, or a
 * parent component -- and `--stagger` legitimately does exactly that.
 */
function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (entry === "node_modules" || entry === ".next" || entry.startsWith(".")) continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else if (/\.(tsx?|css)$/.test(entry)) out.push(full);
  }
  return out;
}

const CSS = readFileSync(join(WEB, "app/globals.css"), "utf8");

const defined = new Set(
  [...CSS.matchAll(/(--[a-z0-9-]+)\s*:/gi)].map((m) => m[1].toLowerCase()),
);

/** `var(--x)` with no comma before the closing paren. */
const USE = /var\(\s*(--[a-z0-9-]+)\s*\)/gi;

describe("custom properties", () => {
  it("are all defined before they are read", () => {
    const missing: string[] = [];

    for (const file of [...walk(join(WEB, "components")), ...walk(join(WEB, "app")), ...walk(join(WEB, "lib"))]) {
      const text = readFileSync(file, "utf8");
      for (const match of text.matchAll(USE)) {
        const name = match[1].toLowerCase();
        if (!defined.has(name)) {
          missing.push(`${file.replace(WEB + "/", "")}: ${name}`);
        }
      }
    }

    expect(missing).toEqual([]);
  });
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pnpm --filter @vedaai/web test -- tokens`
Expected: FAIL. The array is non-empty and names `--radius-sm` (in `PageCanvas.tsx`), plus `--border`, `--text-muted`, `--text-2`, `--surface-2` across `DebugReview.tsx`, `InkOverlay.tsx`, `PageCanvas.tsx`, and `--status-unanswered` / `--status-not-required` in `lib/review.ts`.

- [ ] **Step 3: Fix the one typo**

In `apps/web/components/PageCanvas.tsx:56`, `--radius-sm` was never a token. Change:

```tsx
          borderRadius: "var(--radius-sm)",
```
to:
```tsx
          borderRadius: "var(--r-sm)",
```

- [ ] **Step 4: Define the rest**

In `apps/web/app/globals.css`, inside `:root`, immediately after the `--status-neutral-soft` line, add:

```css
  /*
   * Two absence states the review model names but the palette never had.
   *
   * They are deliberately not the same grey. "Not answered" is a fact about the
   * student -- the page was blank -- and "not required" is a fact about the paper.
   * A teacher acts on the first and ignores the second, so they must not look
   * alike. Both are cool, to sit apart from the warm accent family, which is what
   * the paper's own marks use.
   */
  --status-unanswered: #5c6672;
  --status-unanswered-soft: #eef0f3;
  --status-not-required: #6b6f77;
  --status-not-required-soft: #f2f3f5;

  /*
   * Aliases for the debug route at /review/[id]/inspect, which was written
   * against a naming scheme the stylesheet never adopted. Aliased rather than
   * renamed at the call sites: the debug surface is not what this pass is about,
   * and one line each is cheaper than touching three components.
   */
  --border: var(--hairline);
  --surface-2: var(--row-active);
  --text-2: var(--body-solid);
  --text-muted: var(--muted);
```

- [ ] **Step 5: Run the test and watch it pass**

Run: `pnpm --filter @vedaai/web test -- tokens`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/web/lib/tokens.test.ts apps/web/app/globals.css apps/web/components/PageCanvas.tsx
git commit -m "Fail the build when a custom property is read but never defined

var(--missing) is silent everywhere in the platform -- the declaration is
dropped and the only symptom is a square corner or an uncoloured chip. Seven
properties were being read that the stylesheet never defined, across the review
model and the debug route.

The test exempts var() with a fallback, because that is the documented way to
read a property the stylesheet cannot see, and --stagger genuinely does that."
```

---

### Task 3: Colour module, contrast test, and the AA fixes

**Files:**
- Create: `apps/web/lib/colour.ts`
- Create: `apps/web/lib/colour.test.ts`
- Modify: `apps/web/app/globals.css` (palette)

**Interfaces:**
- Consumes: nothing.
- Produces: `contrastRatio(a: string, b: string): number` and `parseColour(input: string): Rgb`, where `Rgb = { r: number; g: number; b: number }` with each channel in `0..1`. Task 4 uses `contrastRatio` in its own assertions.

- [ ] **Step 1: Write the colour module**

Create `apps/web/lib/colour.ts`:

```ts
/**
 * Just enough colour maths to assert that text is readable.
 *
 * Exists because the palette is authored in OKLCH, and a contrast check that
 * cannot read OKLCH would have to be run against a stale hex copy -- which is
 * the same as not running it. Nothing here is used at runtime; it is test-only
 * support that happens to live in lib/ so it typechecks with everything else.
 */

export interface Rgb {
  /** 0..1, sRGB, not gamma-decoded. */
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
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  return {
    r: parseInt(full.slice(0, 2), 16) / 255,
    g: parseInt(full.slice(2, 4), 16) / 255,
    b: parseInt(full.slice(4, 6), 16) / 255,
  };
}

function parseOklch(input: string): Rgb {
  const body = input.slice(input.indexOf("(") + 1, input.lastIndexOf(")"));
  const parts = body.split("/")[0].trim().split(/\s+/);
  const L = parseFloat(parts[0]) / (parts[0].endsWith("%") ? 100 : 1);
  const C = parseFloat(parts[1]);
  const H = (parseFloat(parts[2]) * Math.PI) / 180;

  const a = C * Math.cos(H);
  const bb = C * Math.sin(H);

  const l_ = L + 0.3963377774 * a + 0.2158037573 * bb;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * bb;
  const s_ = L - 0.0894841775 * a - 1.291485548 * bb;

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
  const r = srgbToLinear(colour.r);
  const g = srgbToLinear(colour.g);
  const b = srgbToLinear(colour.b);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG 2.1 contrast ratio, 1..21. Order of arguments does not matter. */
export function contrastRatio(a: string, b: string): number {
  const la = luminance(parseColour(a));
  const lb = luminance(parseColour(b));
  const hi = Math.max(la, lb);
  const lo = Math.min(la, lb);
  return (hi + 0.05) / (lo + 0.05);
}
```

- [ ] **Step 2: Write the failing contrast test**

Create `apps/web/lib/colour.test.ts`:

```ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { contrastRatio, parseColour } from "./colour";

const CSS = readFileSync(join(__dirname, "../app/globals.css"), "utf8");

/** Read one token's declared value out of the `:root` block. */
function token(name: string): string {
  const match = CSS.match(new RegExp(`${name}\\s*:\\s*([^;]+);`));
  if (!match) throw new Error(`token ${name} is not defined`);
  return match[1].trim();
}

describe("colour maths", () => {
  it("agrees with known sRGB conversions", () => {
    expect(contrastRatio("#000000", "#ffffff")).toBeCloseTo(21, 5);
    expect(contrastRatio("#ffffff", "#ffffff")).toBeCloseTo(1, 5);
  });

  it("reads oklch and hex as the same colour", () => {
    const fromHex = parseColour("#ff5623");
    const fromOklch = parseColour("oklch(67.8% 0.213 36.1)");
    expect(fromOklch.r).toBeCloseTo(fromHex.r, 1);
    expect(fromOklch.g).toBeCloseTo(fromHex.g, 1);
    expect(fromOklch.b).toBeCloseTo(fromHex.b, 1);
  });
});

/**
 * Every pair here carries text at a size and weight that WCAG calls normal, so
 * the floor is 4.5:1 rather than 3:1. Five of them were below it.
 */
describe("text contrast", () => {
  const AA = 4.5;

  it("white label on the primary button", () => {
    expect(contrastRatio("#ffffff", token("--accent-ink"))).toBeGreaterThanOrEqual(AA);
  });

  it("accent text on a white surface", () => {
    expect(contrastRatio(token("--accent-ink"), token("--surface"))).toBeGreaterThanOrEqual(AA);
  });

  it("breadcrumb on a white surface", () => {
    expect(contrastRatio(token("--muted"), token("--surface"))).toBeGreaterThanOrEqual(AA);
  });

  it.each([
    ["answered", "--status-answered", "--status-answered-soft"],
    ["unanswered", "--status-unanswered", "--status-unanswered-soft"],
    ["review", "--status-review", "--status-review-soft"],
    ["not required", "--status-not-required", "--status-not-required-soft"],
    ["missing", "--status-missing", "--status-missing-soft"],
    ["neutral", "--status-neutral", "--status-neutral-soft"],
  ])("the %s chip", (_name, ink, fill) => {
    expect(contrastRatio(token(ink), token(fill))).toBeGreaterThanOrEqual(AA);
  });

  it("body text on the page ground", () => {
    expect(contrastRatio(token("--body-solid"), token("--page"))).toBeGreaterThanOrEqual(AA);
  });
});
```

- [ ] **Step 3: Run it and watch it fail**

Run: `pnpm --filter @vedaai/web test -- colour`
Expected: FAIL. `--accent-ink` does not exist yet, so `token()` throws for the first two cases; the answered chip reports ~3.86, review ~3.83, and the breadcrumb ~2.35.

- [ ] **Step 4: Rewrite the palette in OKLCH and fix the failures**

In `apps/web/app/globals.css`, replace the palette block with these values. Every grey keeps its exact measured lightness; only the five noted colours move.

```css
  /* ── palette, from the file, authored in OKLCH ────────────────────────
   *
   * OKLCH rather than hex because the derived shades are the point: a hover or a
   * tint is now the base colour with one number changed, and it stays the same
   * colour while it does it. Hex fallbacks follow at the end of this file for
   * browsers without it.
   *
   * Five values are not the ones in the Figma file. All five were below WCAG AA
   * where they carry text, and all five are logged in design/SPEC.md.
   */
  --accent: oklch(67.8% 0.213 36.1); /* #ff5623 — fills, rings, badges: no text */
  --accent-ink: oklch(59.1% 0.213 36.1); /* #df3500 — the same orange, for text */
  --accent-tint: oklch(72.6% 0.156 42.5 / 0.15);
  --accent-ring-outer: oklch(67.8% 0.213 36.1 / 0.1);
  --accent-ring-inner: oklch(67.8% 0.213 36.1 / 0.26);

  --ink: oklch(30.9% 0 90); /* #303030 */
  --ink-hover: oklch(23.9% 0 90); /* #1f1f1f */
  --title: oklch(28.9% 0 90); /* #2b2b2b */
  --body: oklch(48.2% 0 90 / 0.8);
  --body-solid: oklch(48.2% 0 90); /* #5e5e5e */
  --muted: oklch(56.7% 0 90); /* was #a9a9a9 at 2.35:1 — now 4.51:1 */

  --status-answered: oklch(52.4% 0.128 154.9); /* was #1f8a52 at 3.86:1 */
  --status-answered-soft: #e8f4ed;
  --status-unanswered: oklch(50.6% 0.023 253);
  --status-unanswered-soft: #eef0f3;
  --status-review: oklch(55.1% 0.132 65.1); /* was #b26a00 at 3.83:1 */
  --status-review-soft: #fdf2e2;
  --status-not-required: oklch(54.2% 0.013 259.8);
  --status-not-required-soft: #f2f3f5;
  --status-missing: oklch(54.3% 0.174 29.7); /* #c0392b */
  --status-missing-soft: #fbeae8;
  --status-neutral: oklch(53.5% 0.008 268.5); /* #6b6d72 */
  --status-neutral-soft: #f0f0f1;
```

Delete the four `--status-unanswered*` / `--status-not-required*` lines added in Task 2 — they are folded into this block. Keep every other token in `:root` unchanged.

Add `color-scheme: light;` to the `:root` block.

- [ ] **Step 5: Point the text sites at `--accent-ink`**

`--accent` must no longer carry text. In `apps/web/app/globals.css`, change the CTA fill and any accent-coloured text:

```bash
cd /Users/vighneshnama/Vedaai-assignment/apps/web
grep -n "var(--accent)" app/globals.css
```

For each hit, decide by whether text sits on or in it: a fill behind white text, or text itself, becomes `var(--accent-ink)`. A ring, a badge dot, a highlight band or a border stays `var(--accent)`. The `.q-card[data-selected="true"]` border and `.q-num` selected background stay `--accent`; `.feedback-cite` and any link-coloured rule become `--accent-ink`.

- [ ] **Step 6: Add the hex fallback block**

At the very end of `apps/web/app/globals.css`:

```css
/*
 * Hex fallback for browsers without OKLCH.
 *
 * Only the tokens whose value is a flat colour; the alpha-carrying ones degrade
 * acceptably to their opaque form, which is better than losing them entirely.
 */
@supports not (color: oklch(0% 0 0)) {
  :root {
    --accent: #ff5623;
    --accent-ink: #df3500;
    --ink: #303030;
    --ink-hover: #1f1f1f;
    --title: #2b2b2b;
    --body-solid: #5e5e5e;
    --muted: #767676;
    --status-answered: #077f47;
    --status-unanswered: #5c6672;
    --status-review: #a55e00;
    --status-not-required: #6b6f77;
    --status-missing: #c0392b;
    --status-neutral: #6b6d72;
  }
}
```

- [ ] **Step 7: Run both test files and watch them pass**

Run: `pnpm --filter @vedaai/web test && pnpm --filter @vedaai/web typecheck`
Expected: PASS, including the Task 2 token test — `--accent-ink` is new and must be defined.

- [ ] **Step 8: Commit**

```bash
git add apps/web/lib/colour.ts apps/web/lib/colour.test.ts apps/web/app/globals.css
git commit -m "Measure the palette against WCAG, and fix the five pairs that failed

The primary button was the worst of them: white on #ff5623 is 3.18:1, so the
one control the screen is built around did not meet AA for its own label.

The brand orange is not changed. It keeps every job where nothing sits on it --
rings, the badge, the notification dot, the highlight bands -- and a darker
sibling at the same hue and chroma takes the text and the filled buttons. Side
by side they read as one colour, which is the point.

Authored in OKLCH so a tint is now the base with one number changed rather than
a second colour picked by eye, with a hex fallback for browsers without it."
```

---

### Task 4: Make the status chip mean something

The live card renders `<span className="score" data-tone="none">` for every status, and `.score[data-tone="none"]` is grey by definition. The status is already computed — it is simply not reaching CSS.

**Files:**
- Modify: `apps/web/components/QuestionCard.tsx:105`
- Modify: `apps/web/app/globals.css` (`.score` rules)

**Interfaces:**
- Consumes: `QuestionRow.status: AnswerStatus` and `QuestionRow.presentation: StatusPresentation` from `lib/review.ts`, both already in scope in `QuestionCard`.
- Produces: `.score[data-status]` styling hook. Nothing downstream depends on it.

- [ ] **Step 1: Pass the status through to CSS**

In `apps/web/components/QuestionCard.tsx`, replace the `data-tone="none"` branch at line 105:

```tsx
            <span className="score" data-tone="none">
              {row.presentation.label}
            </span>
```

with:

```tsx
            <span
              className="score"
              data-tone="none"
              data-status={row.status}
              title={row.presentation.hint}
            >
              {row.presentation.needsAttention && (
                <span className="score-mark" aria-hidden="true">
                  !
                </span>
              )}
              {row.presentation.label}
            </span>
```

The `!` is the non-colour channel — it is what a teacher who cannot separate green from amber reads instead. `aria-hidden` because the label beside it already says the same thing, and a screen reader announcing "exclamation Not found" is worse than "Not found".

- [ ] **Step 2: Style each status**

In `apps/web/app/globals.css`, immediately after the existing `.score[data-tone="none"]` rule, add:

```css
/*
 * The status chip, by status.
 *
 * Until now every one of these was the same grey, because the card rendered the
 * label from the review model and dropped the colour it came with. This is the
 * one thing on the screen a teacher scans before reading a word, so it carries
 * three channels rather than one: hue, a hairline border at the same hue, and a
 * glyph on the two states that ask to be looked at.
 */
.score[data-tone="none"][data-status] {
  gap: var(--sp-1);
  border: 1px solid;
  font-size: var(--fs-sm);
  font-weight: 600;
}

.score[data-status="answered"] {
  background: var(--status-answered-soft);
  border-color: oklch(from var(--status-answered) l c h / 0.24);
  color: var(--status-answered);
}

.score[data-status="unanswered"] {
  background: var(--status-unanswered-soft);
  border-color: oklch(from var(--status-unanswered) l c h / 0.24);
  color: var(--status-unanswered);
}

.score[data-status="ocr_failed"],
.score[data-status="uncertain"] {
  background: var(--status-review-soft);
  border-color: oklch(from var(--status-review) l c h / 0.24);
  color: var(--status-review);
}

.score[data-status="not_required"] {
  background: var(--status-not-required-soft);
  border-color: oklch(from var(--status-not-required) l c h / 0.24);
  color: var(--status-not-required);
}

.score[data-status="pages_missing"] {
  background: var(--status-missing-soft);
  border-color: oklch(from var(--status-missing) l c h / 0.24);
  color: var(--status-missing);
}

.score-mark {
  font-weight: 800;
  line-height: 1;
}
```

The six `data-status` values are the `AnswerStatus` union in
`packages/contracts/dist/types.ts:34`, read rather than assumed:
`answered | unanswered | ocr_failed | not_required | pages_missing | uncertain`.
Note `pages_missing` is plural and the "Not found" label belongs to `uncertain` —
an attribute selector that matches nothing fails exactly as silently as the bugs
this pass exists to remove.

- [ ] **Step 3: Give the card a status rail**

In `apps/web/app/globals.css`, in the `.q-card` rule, replace `border: 2px solid transparent;` with:

```css
  border: 2px solid transparent;
  border-left-width: 3px;
```

and add after it:

```css
/*
 * The same hue as the chip, on the card's leading edge.
 *
 * Reading order down a list is vertical, so a colour on the left edge is picked
 * up in peripheral vision before any chip is fixated. It repeats the chip rather
 * than adding information, which is what makes the list scannable at a glance.
 */
.q-card:has(.score[data-status="answered"]) { border-left-color: var(--status-answered-soft); }
.q-card:has(.score[data-status="unanswered"]) { border-left-color: var(--status-unanswered); }
.q-card:has(.score[data-status="ocr_failed"]),
.q-card:has(.score[data-status="uncertain"]) { border-left-color: var(--status-review); }
.q-card:has(.score[data-status="pages_missing"]) { border-left-color: var(--status-missing); }
```

- [ ] **Step 4: Verify by eye**

Run: `pnpm --filter @vedaai/web dev`
Open a review URL. Expected: `Answered` is green on a pale green fill, `Not found` is amber with a leading `!`, and the cards carry a matching left edge. Confirm the chip text is still legible — the contrast test in Task 3 already asserts it, but read it on screen.

- [ ] **Step 5: Commit**

```bash
git add apps/web/components/QuestionCard.tsx apps/web/app/globals.css
git commit -m "Let the status chip show which status it is

The review model has always computed a colour per status. The card took the
label off the same object and dropped the colour, so every question -- answered,
blank, unreadable, missing page -- wore the same grey pill, and the list could
not be scanned without reading it line by line.

Colour is not the only channel. The two states that want attention also carry a
glyph, and the card repeats the hue on its leading edge, where vertical reading
order picks it up before any chip is fixated."
```

---

### Task 5: Elevation and surface ladder

**Files:**
- Modify: `apps/web/app/globals.css`

**Interfaces:**
- Consumes: `--page`, `--surface` from Task 3's palette.
- Produces: `--surface-raised`, `--e1`, `--e2`, `--e3`. Later tasks use `--e2` on the sheet bar.

- [ ] **Step 1: Add the ladder to `:root`**

Replace the single `--shadow-panel` line with:

```css
  /* ── depth ───────────────────────────────────────────────────────────
   *
   * Three steps, because one shadow on everything is the same as no shadow: if
   * the panel and the card inside it are lit identically, neither is above the
   * other and the screen reads flat.
   *
   * Tinted warm rather than neutral black. The page is a warm grey, and a black
   * shadow over it reads as dirt rather than as depth -- the shadow has to be a
   * darker version of the surface it falls on.
   */
  --surface-raised: oklch(99.2% 0.001 90);

  --e1: 0 1px 2px oklch(30% 0.01 60 / 0.05);
  --e2:
    0 1px 2px oklch(30% 0.01 60 / 0.05),
    0 8px 16px -8px oklch(30% 0.01 60 / 0.12);
  --e3:
    0 2px 4px oklch(30% 0.01 60 / 0.06),
    0 24px 48px -16px oklch(30% 0.01 60 / 0.18);

  /* Kept: still referenced by the upload screen until Task 8. */
  --shadow-panel: var(--e2);
```

- [ ] **Step 2: Put the card on its own plane**

In `.q-card`, change `background: var(--surface);` to:

```css
  background: var(--surface-raised);
  box-shadow: var(--e1), inset 0 1px 0 oklch(100% 0 0 / 0.8);
```

The inset is a lit top edge. It is what stops a white card on a white panel reading as a rectangle drawn on the panel rather than as an object sitting on it.

Add hover and selected states after the `.q-card[data-selected="true"]` rule:

```css
.q-card:hover {
  box-shadow: var(--e3), inset 0 1px 0 oklch(100% 0 0 / 0.8);
  transform: translateY(-1px);
}

.q-card[data-selected="true"] {
  box-shadow: var(--e3), inset 0 1px 0 oklch(100% 0 0 / 0.8);
}
```

Add `transform` to the transitioned properties for `.q-card` — it is already in the shared `transition-property` list, so confirm rather than re-add.

- [ ] **Step 3: Put the panes on the middle plane**

In `.q-pane`, add `box-shadow: var(--e2);`. In `.sheet-pane`, replace `border: 1.25px solid rgb(0 0 0 / 0.1);` with `border: 1px solid var(--hairline);` and add `box-shadow: var(--e2);`.

- [ ] **Step 4: Add squircle corners behind a support query**

At the end of the shape section of `apps/web/app/globals.css`:

```css
/*
 * Superelliptical corners where the browser has them.
 *
 * A rounded rectangle changes curvature abruptly where the arc meets the
 * straight edge; a superellipse does not, which is why the same radius reads as
 * softer without reading as rounder. Chrome only, and entirely cosmetic --
 * everywhere else keeps the radius it already had.
 */
@supports (corner-shape: squircle) {
  .q-card,
  .q-pane,
  .sheet-pane,
  .dropzone,
  .stage-inner {
    corner-shape: squircle;
  }
}
```

- [ ] **Step 5: Verify at four widths**

Run the dev server and check 393, 1440, 1536 and 1920. Expected: the card is visibly a distinct object from the pane behind it; hover lifts it; nothing shifts layout, since only `box-shadow` and a 1px `transform` are involved.

- [ ] **Step 6: Commit**

```bash
git add apps/web/app/globals.css
git commit -m "Give the surfaces three planes instead of one

Page, panel and card all shared one white and one shadow token, so nothing on
the screen was above anything else and the whole thing read as a drawing rather
than as a stack.

The shadows are warm rather than neutral. The page is a warm grey and a black
shadow falling on it reads as dirt; a shadow has to be a darker version of what
it falls on to read as depth at all.

Squircle corners are applied where the browser has them and ignored where it
does not. Same radius, softer read, nothing lost in the browsers without it."
```

---

### Task 6: Density, and numerals that hold still

**Files:**
- Modify: `apps/web/app/layout.tsx`
- Modify: `apps/web/app/globals.css`

**Interfaces:**
- Consumes: nothing.
- Produces: `--font-mono` bound to a real face; `.tabular` utility class.

- [ ] **Step 1: Load JetBrains Mono**

In `apps/web/app/layout.tsx`, extend the import and add the loader:

```tsx
import { Bricolage_Grotesque, Inter, JetBrains_Mono } from "next/font/google";
```

```tsx
/**
 * The numerals. Until now `--font-mono` was a system stack, so the counts a
 * teacher reads rendered in a different face on every machine -- and on the two
 * that lack SF Mono, in a proportional fallback that made the digits jitter as
 * they changed.
 */
const mono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono-face",
  weight: ["400", "500"],
});
```

And add it to the class list:

```tsx
    <html lang="en" className={`${bricolage.variable} ${inter.variable} ${mono.variable}`}>
```

- [ ] **Step 2: Bind the token**

In `apps/web/app/globals.css`, replace the `--font-mono` line with:

```css
  --font-mono: var(--font-mono-face), ui-monospace, "SF Mono", Menlo, monospace;
```

- [ ] **Step 3: Add the tabular utility and apply it**

```css
/*
 * Digits that do not move.
 *
 * Proportional numerals give 1 a narrower advance than 8, so a counter ticking
 * from 100% to 90% reflows the row it sits in. Every number here changes while
 * being read, which is exactly the case tabular figures exist for.
 */
.tabular,
.tool-value,
.q-hint,
.score {
  font-variant-numeric: tabular-nums;
}
```

- [ ] **Step 4: Tighten the card**

In `.q-card`, change `padding: var(--sp-3);` to `padding: var(--sp-3) var(--sp-4);` and `gap: var(--sp-3);` to `gap: var(--sp-2);`. In `.q-list`, change `gap: var(--sp-4);` to `gap: var(--sp-2);`. In `.q-num`, change `width`/`height` from `32px` to `28px` and `font-size` from `1.25rem` to `1rem`.

- [ ] **Step 5: Verify**

Run the dev server. Expected: roughly a third more questions visible without scrolling, the zoom counter no longer reflows the toolbar as it changes, and the counts row renders in JetBrains Mono.

- [ ] **Step 6: Commit**

```bash
git add apps/web/app/layout.tsx apps/web/app/globals.css
git commit -m "Bind the mono token to a real face, and stop the digits moving

--font-mono resolved to a system stack, so the counts rendered in a different
typeface on every machine, and on the ones without SF Mono in a proportional
fallback -- which made the zoom counter reflow its own toolbar every time it
changed. Loaded through next/font like the other two faces, so it is still
self-hosted at build time and still not a runtime dependency on anyone.

The card also gives back about a third of its height. It was ninety-six pixels
tall to hold one line of text."
```

---

### Task 7: Motion that leaves as well as arrives

**Files:**
- Modify: `apps/web/app/globals.css`

**Interfaces:**
- Consumes: `--ease`, `--ease-out`, `--fast`, `--mid`, `--slow`.
- Produces: `--ease-spring`. `lib/transitions.ts` is deliberately unchanged — see Step 5.

- [ ] **Step 1: Add the spring curve**

In the motion block of `apps/web/app/globals.css`, after `--ease-out`:

```css
  /*
   * A real spring, sampled.
   *
   * linear() interpolates between as many points as it is given, and values past
   * 1 overshoot -- which is what a spring does and what no cubic-bezier can. Used
   * only where the reader asked for the change, never on hover and never on
   * anything ambient: an overshoot the reader did not cause reads as instability.
   */
  --ease-spring: linear(
    0, 0.0033, 0.0505, 0.1685, 0.3348, 0.5108, 0.6702, 0.7987,
    0.8929, 0.9542, 0.9887, 1.0052, 1.0107, 1.0104, 1.0077,
    1.0045, 1.0018, 1
  );
```

- [ ] **Step 2: Spring the press and the chip**

Replace the press-feedback rule:

```css
button:active:not(:disabled),
.dropzone:active {
  transform: scale(0.97);
  transition-duration: 60ms;
}
```

with:

```css
button:active:not(:disabled),
.dropzone:active {
  transform: scale(0.97);
  transition-duration: 60ms;
}

/* The release, not the press. A spring on the way down fights the finger. */
button:not(:active),
.dropzone:not(:active) {
  transition-timing-function: var(--ease-spring);
  transition-duration: var(--mid);
}
```

- [ ] **Step 3: Let the feedback panel exit**

The panel animates open on a grid row and is removed the instant it closes. Add after the `.feedback-wrap` rules:

```css
/*
 * The half of the transition that was missing.
 *
 * Everything here animated in and then vanished in one frame, because a closing
 * panel is unmounted before any transition can run. allow-discrete keeps it
 * present for the duration, and @starting-style gives the arriving state
 * somewhere to come from.
 */
.feedback-wrap {
  transition:
    grid-template-rows var(--mid) var(--ease),
    opacity var(--mid) var(--ease),
    display var(--mid) allow-discrete;
}

@starting-style {
  .feedback-wrap[data-open="true"] {
    grid-template-rows: 0fr;
    opacity: 0;
  }
}
```

- [ ] **Step 4: Stagger without JavaScript**

Replace the `animation-delay: var(--stagger, 0ms);` line inside the `@media (prefers-reduced-motion: no-preference)` block with:

```css
    animation-delay: var(--stagger, 0ms);
  }

  /*
   * Where the browser can count siblings, it does its own staggering and the
   * caller's cap is unnecessary. Capped at twelve either way: past that the last
   * card waits a third of a second for no reason a reader can perceive.
   */
  @supports (animation-delay: calc(sibling-index() * 1ms)) {
    .q-card {
      animation-delay: calc(min(sibling-index(), 12) * 28ms);
    }
```

- [ ] **Step 5: Leave `lib/transitions.ts` alone**

The design proposed scoping the View Transition to the question pane, on the
assumption that selecting a question cross-fades the whole root and flickers the
answer sheet. Reading the call sites shows that is not what happens. `crossFade`
is called in exactly three places — `UploadForm.tsx:91`, `UploadForm.tsx:132` and
`MapSurface.tsx:100` — and all three are whole-screen swaps: upload giving way to
the waiting screen, and the waiting screen giving way to the mapping screen.
Selection goes through `setSelectedQid` (`MapSurface.tsx:159`) with no transition
at all.

Naming `root` is therefore correct for every existing caller, and adding a `scope`
parameter no caller passes would be dead code of the same kind Task 1 deletes.
**No change.** This step exists so the next reader knows the question was asked
and answered, rather than missed.

- [ ] **Step 6: Verify, including reduced motion**

Run the dev server. Expected: the feedback panel closes by collapsing rather than vanishing, and cards arrive staggered. Then enable **System Settings → Accessibility → Display → Reduce motion** and reload. Expected: none of it animates, and nothing is broken or invisible.

- [ ] **Step 7: Commit**

```bash
git add apps/web/app/globals.css
git commit -m "Animate the leaving as well as the arriving

Every animation in the app was an entrance. A closing panel is unmounted before
a transition can run, so it disappeared in one frame however carefully it had
opened -- the cut was still there, just moved to the end.

The spring goes on the release rather than the press. On the way down it fights
the finger; on the way back it reads as the control having weight."
```

---

### Task 8: Rebalance the panes, and carry the atmosphere across

**Files:**
- Modify: `apps/web/app/globals.css`

**Interfaces:**
- Consumes: `--surface-glass`, `--e2`, `--hairline`.
- Produces: nothing.

- [ ] **Step 1: Lift the black bar**

The `Answer Sheet` bar is the heaviest object on the screen and it is a toolbar. Replace the colour half of `.sheet-bar`:

```css
  background: var(--ink);
  color: #fff;
```

with:

```css
  /*
   * Glass, not ink.
   *
   * A black toolbar above a scanned white page put the screen's strongest
   * contrast on a control strip, which is the one thing on it nobody came to
   * look at. The paper should be the darkest-to-lightest range in the viewport;
   * everything else gets out of its way.
   */
  background: var(--surface-glass);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--hairline);
  color: var(--title);
```

Then invert the controls that assumed a dark ground — in `.sheet-bar h2` change `color: rgb(255 255 255 / 0.8);` to `color: var(--body-solid);`; in `.tool-group` change `background: rgb(255 255 255 / 0.14);` to `background: var(--chip);`; in `.tool-button` change `color: #fff;` to `color: var(--title);`; in `.tool-button:hover:not(:disabled)` change `background: rgb(255 255 255 / 0.18);` to `background: var(--row-active);`.

- [ ] **Step 2: Give the top bar something to blur**

`.topbar` is already `--surface-glass` with nothing behind it. Add to the `.topbar` rule:

```css
  backdrop-filter: blur(12px);
  box-shadow: var(--e2);
```

- [ ] **Step 3: Carry the atmosphere onto the review screen**

`body::before` is fixed and already applies to every route, so the ellipses are present but the review screen's opaque panes cover them. Soften the question pane so they show through — in `.q-pane`, confirm `background: var(--surface-veil);` and add:

```css
  backdrop-filter: blur(20px);
  border: 1px solid var(--hairline);
```

- [ ] **Step 4: Verify**

Run the dev server on both the upload screen and a review URL. Expected: the answer sheet is now the highest-contrast region; the two panes read as one family rather than two apps; the warm ground is visible behind the question pane.

- [ ] **Step 5: Commit**

```bash
git add apps/web/app/globals.css
git commit -m "Stop the toolbar being the loudest thing on the screen

A black bar sat above a scanned white page, which put the strongest contrast in
the viewport on a strip of zoom controls. The two panes also read as two
different applications: one had a white header and the other a black one.

The bar becomes glass like the top bar, its controls invert, and the paper ends
up carrying the contrast -- which is right, because the paper is the product.

The blurred ellipses behind the page were already fixed and already applied to
every route; the review panes were simply opaque over them."
```

---

### Task 9: Log the divergences and verify the whole pass

**Files:**
- Modify: `design/SPEC.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

- [ ] **Step 1: Add the divergences**

In `design/SPEC.md`, under "Divergences, and why", append:

```markdown
5. **Five colours are darker than the file.** Measured against WCAG 2.1, five
   pairs carrying text were below the 4.5:1 the size and weight require: white on
   the `#FF5623` primary button (3.18:1), the same orange as link text (3.18:1),
   the `#A9A9A9` breadcrumb on white (2.35:1), and the answered (3.86:1) and
   review (3.83:1) status inks on their own fills. The accent itself is unchanged
   and keeps every job where nothing sits on it — the rings, the badge, the
   notification dot, the highlight bands. Text and filled buttons use
   `--accent-ink`, the same hue and chroma at lower lightness.
6. **Two status colours the file does not have.** The review model names six
   answer states; the frame drew three score pills. `--status-unanswered` and
   `--status-not-required` are additions, deliberately cool so they sit apart from
   the warm accent family the paper's own marks use.
7. **Surfaces are on three planes, not one.** The file draws panel and card in the
   same white with one shadow. Geometry is unchanged; the card gains its own
   near-white, a lighter shadow and a lit top edge, so it reads as an object on
   the panel rather than a shape drawn on it.
8. **The answer-sheet bar is glass, not `#303030`.** Drawn as ink in the frame,
   which put the screen's strongest contrast on a control strip above a scanned
   page. It matches the top bar instead.
```

- [ ] **Step 2: Confirm no measured geometry moved**

Run:
```bash
cd /Users/vighneshnama/Vedaai-assignment && git diff main -- apps/web/app/globals.css | grep -E "^\+.*(width|height|padding|margin|gap|border-radius):" 
```
Expected: only the `.q-card`, `.q-list` and `.q-num` density changes from Task 6, all on the review screen, which has no Figma frame. Any hit inside the upload screen's rules means a measured value moved — either revert it or add a divergence line.

- [ ] **Step 3: Full verification**

Run:
```bash
pnpm --filter @vedaai/web typecheck && pnpm --filter @vedaai/web test
```
Expected: clean, all suites pass.

Then render at 393, 1440, 1536, 1920. Then reload with reduce-motion enabled and confirm nothing animates and nothing is hidden.

- [ ] **Step 4: Commit**

```bash
git add design/SPEC.md
git commit -m "Record where the polish pass left the Figma file

Four divergences, all surface rather than geometry: five colours darkened to
clear WCAG AA, two status colours the frame never had, three surface planes
where the file drew one, and a glass toolbar where it drew ink.

The measured values are all still the measured values. That is the guarantee
this file exists to make, and a silent change to any of them would be worth
more than the polish."
```

---

## Self-review

**Spec coverage.** §1 defects → Tasks 1, 2, 3. §2 token spine → Tasks 3, 5, 6. §3 depth → Task 5. §4 colour meaning → Task 4. §5 density → Task 6. §6 motion → Task 7. §7 weight rebalance → Task 8. §8 atmosphere → Task 8. Verification and divergence logging → Task 9. No section is unaddressed.

**Known deviations from the spec, decided during planning.** The spec's `--radius-sm` fix listed three files; two of them are deleted in Task 1, so only `PageCanvas.tsx` is repaired. The spec did not mention `--border`, `--surface-2`, `--text-2` or `--text-muted`; those were found afterwards and are handled in Task 2. Both are recorded in the spec document itself.

**Two spec items that did not survive contact with the code.**

The spec's §6 asked for the View Transition to be scoped to the question pane, to
stop the answer sheet flickering when a question is selected. Reading the three
`crossFade` call sites shows selection does not use it — all three are whole-screen
swaps, for which naming `root` is correct. Task 7 Step 5 records the finding and
changes nothing. The flicker in the spec was inferred, not observed.

The spec's §4 named the status values `not_found` and `page_missing`. The real
`AnswerStatus` union (`packages/contracts/dist/types.ts:34`) has `uncertain` and
`pages_missing`. Task 4 uses the real ones. Had this been carried through as
written, all six selectors would have silently matched nothing and the chips would
have stayed grey — the pass would have shipped having fixed nothing at all.
