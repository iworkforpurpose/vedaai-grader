# Frontend polish pass — design

Date: 2026-08-31
Scope: `apps/web` only. No API, no pipeline, no deploy.

## Why

The app reads as a clean prototype rather than as a product someone paid for. The
cause is not missing animation — the stylesheet already has a shared easing pair,
three duration roles, `prefers-reduced-motion` handling, and a working View
Transitions cross-fade. The causes are narrower than that:

1. **Every surface sits on one plane.** The page is `#ececec`, the panel is
   `#ffffff`, and the card inside the panel is also `#ffffff`. One shadow token,
   `--shadow-panel`, is applied to every panel, so nothing is above anything else.
2. **Colour carries no meaning where it matters most.** The status slot on a
   question card is the one thing a teacher scans, and it renders as grey pills.
3. **Three tokens referenced in code are never defined**, so they silently fail.
4. **Several foreground/background pairs fail WCAG AA**, including the primary
   button's own label.

## Decisions taken before design

| Question | Decision |
|---|---|
| Figma fidelity | Figma is **intent**. Surface treatment may evolve; every deviation gets a line in `design/SPEC.md` → "Divergences, and why", same commit. |
| Dark mode | **Out of scope.** Colour is authored through `light-dark()` with identical pairs so adding it later is one file, not a rewrite. |
| Typography | Keep **Bricolage Grotesque** as the voice. Add one **self-hosted mono** for numerals and metadata. |
| Approach | Token spine first, proved end-to-end on the review screen, then swept onto upload. |

Figma only ever covered the upload screen (`design/SPEC.md`, frames `1:8744` and
`1:10442`). The review/mapping screens were designed by extension and have no file
to drift from — they carry the larger share of the change.

---

## 1. Defects to fix first

These are bugs, not taste. They ship today.

**Undefined custom properties.** `var()` with no definition and no fallback
resolves to nothing, and the declaration is dropped:

| Property | Referenced in | Effect |
|---|---|---|
| `--radius-sm` | `StatusChip.tsx:30`, `PageCanvas.tsx:56`, `GradePanel.tsx:94` | `border-radius` dropped — corners render square. The defined token is `--r-sm`. |
| `--status-unanswered` | `lib/review.ts:53` | `color` dropped — the "Not answered" chip inherits its parent's colour. |
| `--status-not-required` | `lib/review.ts:65` | Same, for "Not required". |

**Contrast failures.** Measured, sRGB, WCAG 2.1 non-large text (needs 4.5:1):

| Pair | Now | Fix | After |
|---|---|---|---|
| White label on `--accent` `#ff5623` (the primary button, "Re-mark") | 3.18:1 | new `--accent-ink` `#df3500` | 4.52:1 |
| `--accent` as link text on white ("Show me where it is") | 3.18:1 | `--accent-ink` | 4.52:1 |
| `--muted` `#a9a9a9` on white (breadcrumb — its only use) | 2.35:1 | darken to `#767676` | 4.51:1 |
| `--status-answered` on its soft fill | 3.86:1 | `#077f47` | 4.52:1 |
| `--status-review` on its soft fill | 3.83:1 | `#a55e00` | 4.51:1 |

`--accent-ink` is the same hue and chroma as the brand orange with lightness pulled
down — `oklch(59.1% 0.213 36.1)` against the accent's `oklch(67.8% 0.213 36.1)`. It
reads as the same orange. `--accent` itself is unchanged and stays on the rings,
the badge, the notification dot and the highlight bands, none of which carry text.

Three divergence lines go into `design/SPEC.md`: `--accent-ink`, the darkened
`--muted`, and the two status colours.

---

## 2. Token spine — `app/globals.css`

**Colour authoring.** OKLCH, each value wrapped in `light-dark()` with both slots
identical for now. Hex fallbacks retained under `@supports not (color: oklch(0% 0 0))`
for the small tail of browsers without it. Measured conversions of the existing
palette:

```
accent            #ff5623   oklch(67.8% 0.213  36.1)
accent-ink        #df3500   oklch(59.1% 0.213  36.1)   ← new, AA-safe
ink               #303030   oklch(30.9% 0      89.9)
title             #2b2b2b   oklch(28.9% 0      89.9)
body-solid        #5e5e5e   oklch(48.2% 0      89.9)
muted             #767676   oklch(56.7% 0      89.9)   ← darkened from #a9a9a9
page              #ececec   oklch(94.3% 0      89.9)
status-answered   #077f47   oklch(52.4% 0.128 154.9)
status-unanswered #5c6672   oklch(50.6% 0.023 253.0)   ← new
status-review     #a55e00   oklch(55.1% 0.132  65.1)
status-not-req.   #6b6f77   oklch(54.2% 0.013 259.8)   ← new
status-missing    #c0392b   oklch(54.3% 0.174  29.7)
```

Once authored in OKLCH, hover/active/tint variants derive from the base with
relative colour syntax rather than being hand-picked, which is what stops the
"why is this shade slightly wrong" class of bug.

**Surface ladder.** Replaces the single white.

```
--page             the ground
--surface          panels
--surface-raised   cards on panels
```

**Elevation ladder.** Replaces the single `--shadow-panel`. Shadows are tinted
warm rather than pure black — a neutral-black shadow over a warm grey page reads
as dirt, not depth.

```
--e1   card at rest
--e2   panel
--e3   hover / raised
```

**Motion.** `--ease` and `--ease-out` stay; the duration roles stay. One curve is
added: `--ease-spring`, a `linear()` sampled from a real spring, used only on state
changes the user asked for. It never goes on hover or on anything ambient.

**Type.** `--font-mono` today resolves to `ui-monospace, "SF Mono", Menlo` — a
system stack, so the numerals a teacher reads render differently on every machine.
It is bound instead to **JetBrains Mono**, loaded through `next/font/google` in
`app/layout.tsx` alongside the two existing faces. That route self-hosts at build
time, so this adds no runtime dependency on a third party — the same reasoning the
file already gives for Bricolage and Inter. Weights 400 and 500 only.

## 3. Depth

Card and panel stop sharing a fill. The card takes `--surface-raised`, `--e1` and an
inner top highlight (`inset 0 1px 0`) that reads as a lit edge; the panel takes
`--surface` and `--e2`; hover lifts to `--e3`. `corner-shape: squircle` is applied
behind `@supports` at the existing radii — Chrome renders a superellipse, every
other browser renders today's rounded corner, and nothing is lost.

## 4. Colour that means something

`.score[data-tone]` and `StatusChip` currently solve the same problem twice, in two
places, with two visual languages. They collapse into one chip primitive: soft
fill, solid text, hairline border, hue chosen by status. The question card gains a
3px left rail in the same hue so the list is scannable without reading a word.

The `!` glyph and the shape affordance in `StatusChip` **stay**. That is the
non-colour channel for anyone who cannot separate the hues, and this is the most
consequential element on the page.

## 5. Density

The question card is ~96px tall for one line of text with half its width empty.
Target ~68px, padding placed on a 4px grid. The counts line
("14 of 18 answered · 0 not answered · 4 to check") becomes a summary strip in
mono with tabular numerals — it is the screen's headline number and currently has
the same weight as everything around it.

## 6. Motion

- Spring easing on state changes the user triggered — chip transitions, press
  release, panel open.
- `@starting-style` + `transition-behavior: allow-discrete` so the feedback panel
  **exits** instead of cutting. Every animation in the app today is an entrance.
- `sibling-index()` for stagger on `card-in`, with an `nth-child` fallback for the
  first ~12 rows where support is missing.
- The View Transition is scoped to the question pane rather than to `root`, so the
  answer sheet stops flickering on a state change that never touched it.

## 7. Weight rebalance

The black `Answer Sheet` bar is currently the heaviest object on screen, and it is
a toolbar. It becomes glass, matching the left pane; its controls become chips. The
scanned paper ends up the highest-contrast thing in the viewport, which is correct —
it is the content, and the whole product exists to look at it.

## 8. Atmosphere

The two blurred ellipses that give the upload screen its warmth are ported to the
review screen as fixed radial gradients. `backdrop-filter` goes on the top bar,
which is already 75% white and currently has nothing to blur.

---

## Non-goals

Dark theme. Layout or IA changes. New features. Any change to upload-screen
**geometry** — the measured values in `design/SPEC.md` stay as measured; only
surface treatment moves.

## Verification

- `pnpm typecheck` and `vitest` green.
- Rendered at 393, 1440, 1536 and 1920 wide.
- `prefers-reduced-motion: reduce` pass — no motion survives.
- Every contrast pair in §1 re-measured at 4.5:1 or better.
- Every measured value in `design/SPEC.md` either unchanged, or changed with a
  divergence line added in the same commit.

## Risks

| Risk | Handling |
|---|---|
| `corner-shape` is Chrome-only | Behind `@supports`; degrades to the current radius. |
| `sibling-index()` support is thin | `nth-child` fallback for the visible rows. |
| OKLCH conversion shifts the brand orange | Hex fallback retained; verified side by side before commit. |
| Elevation tokens touch every surface at once | Landed on the review screen first and reviewed there before the upload sweep. |

## Rollout

Local only. Nothing is pushed or deployed; the dev server on port 3001 is the
delivery mechanism for this pass.
