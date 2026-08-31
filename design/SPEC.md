# VedaAI upload screen — measured spec

Read from the Figma REST API, file `GEjt1rt1s7AXvkcr4t8muE`. Not measured off the
PNG exports: those were rendered at 1.389× and 2×, so every pixel taken from them
was wrong by that factor, and the desktop frame is 1440 wide rather than the 1000
the raster implied.

- Desktop `1:8744` — *Upload Screen - Empty State*, **1440 × 787**
- Phone `1:10442` — *Upload Screen - Empty State (phone)*, **393 × 853**

The phone frame includes iOS chrome — a 105px status-and-address bar and a home
indicator. That is device furniture, not part of the product, and is not built.

## Type

**Bricolage Grotesque** throughout, with two exceptions found in the tree: the
toolkit button label is **Inter 500 16/28**, and the help glyph is **Agrandir 700**
(commercial, one character, substituted).

| Role | Value |
|---|---|
| Page title | 700 · 40/48 · `#2B2B2B` |
| Subtitle | 400 · 20/28 · `#303030` |
| Brand wordmark | 700 · 28/20 · `#303030` |
| Nav item | 400 · 16/22 · `#5E5E5E` @80% |
| Nav item, active | 500 · 16/22 · `#303030` |
| Breadcrumb | 600 · 16/19 · `#A9A9A9` |
| User name | 600 · 16/19 · `#303030` |
| Button label | 500 · 14/20 · `#FFFFFF` |
| Caption | 400 · 14/22 · `#5E5E5E` @80% |

## Colour

| Token | Value | Use |
|---|---|---|
| accent | `#FF5623` | badge, notification dot, hero rings |
| accent tint | `#FF9350` @15% | block behind the accent half of the title |
| hero ring outer | `#FF5623` @10% | 138px circle |
| hero ring inner | `#FF5623` @26% | 108px circle |
| ink | `#303030` | primary button, active text |
| title | `#2B2B2B` | |
| body | `#5E5E5E` @80% | |
| muted | `#A9A9A9` | breadcrumb |
| surface | `#FFFFFF` | sidebar |
| surface @75% | `#FFFFFF` @75% | top bar — translucent |
| surface @50% | `#FFFFFF` @50% | drop-zone container — translucent |
| chip | `#F6F6F6` | circular icon buttons |
| active row | `#F0F0F0` | selected nav item, school card |

Two blurred ellipses sit behind everything — `#171717` @40% at 1318×428 and
`#4C4C4C` @40% at 1113×428. They are what makes the page read as warm grey rather
than flat, and why the translucent panels have something to be translucent over.

## Geometry

| Element | Desktop | Phone |
|---|---|---|
| Sidebar | 304 × 763, pad 24, gap 32, r16 | not present |
| Top bar | 1100 × 56, pl24 pr8, gap 10, r16 | — |
| Content frame | 1103 × 694, gap 36, r40 | 373 × 694, gap 24, pt32, r40 |
| Drop-zone container | 789 × 205, pad 12, gap 24, r24 | 373 × 290, pad 12, gap 16, r24 |
| Drop-zone row | 765 × 181, **horizontal**, gap 16 | **vertical** |
| Hero | 138 × 138 | 110 × 110 |
| Primary button | 161 × 44, pl24 pr20 pt12 pb12, r64, 2px `#FFFFFF` @15% stroke | same |
| Nav row | 254 × 38, pl12 pr12 pt9 pb9, r8 | — |
| Toolkit button | 251 × 42, pl43 pr43, r100, `#272727` | — |
| Icon button | 36 × 36, r100, `#F6F6F6` | — |

Radii in use: 8, 12, 16, 24, 40, 64, 100.

## Navigation

Seven items, all `visible: true` in the file. The supplied PNG shows only five —
it is a stale export, so the API is taken as authoritative:

Home · My Classroom · Assignments · **Exams** (active) · My Library `32` ·
Review · Analytics — then Settings, then the school card.

Only Exams is reachable. The rest render as designed and are inert three
independent ways: no href, `aria-disabled`, and `pointer-events: none`.

`My Library` carries a badge: 37 × 20, r8, fill `#FF5623`, label `32`.

## Assets

Exported at 3× into `apps/web/public/brand/`:

| File | Node | Size |
|---|---|---|
| `teacher.png` | `1:8760` | 79 × 97 |
| `teacher-photo.png` | `1:8763` | 79 × 97 |
| `logo.png` | `I1:8796;14328:26657` | 40 × 40 |
| `school-crest.png` | `I1:8796;17584:35143;17584:35129` | 59 × 60 |
| `avatar.png` | `I1:8795;17584:38945;17584:23797` | 32 × 32 |

## Responsive intent

The design gives two fixed widths, 1440 and 393. Neither is what most users have —
1536 × 864 and 1920 × 1080 dominate, and 1440 sits between the two frames — so the
frames are read as intent rather than as targets:

- Sidebar **304px fixed**. As a percentage of 1440 it is 21%, which would be 403px
  at 1920 — a rail with nothing more to say at twice the width.
- Content column capped, so it does not sprawl on a wide monitor.
- Type fluid via `clamp()` between the phone and desktop values, in `rem` so a
  reader's font-size preference is honoured.
- Breakpoints where the content breaks, not at device names: **600px** the drop
  zones stop being cramped side by side, **1024px** there is room for a permanent
  rail.

## Divergences, and why

1. **Caption typo.** The frame reads "you'll able to map answers". Shipped as
   "you'll be able to".
2. **Primary button.** The frame fills it `#303030` even in the empty state, where
   nothing is uploaded yet. Rendered disabled until both files are chosen —
   otherwise the control invites a click that cannot work.
3. **Agrandir** for the `?` glyph is commercial and used for one character.
   Substituted.
4. **Drop zones are drop targets.** The design draws a dashed rectangle, which is
   the conventional signal, so they accept a drop as well as a click.
5. **Five colours are darker than the file.** Measured against WCAG 2.1, five
   pairs carrying text sat below the 4.5:1 their size and weight require: white on
   the `#FF5623` primary button (3.18:1), the same orange as link text (3.18:1),
   the `#A9A9A9` breadcrumb on white (2.35:1), and the answered (3.86:1) and
   review (3.83:1) status inks on their own fills. The accent itself is unchanged
   and keeps every job where nothing sits on it — the focus ring, the notification
   dot, the hero badge, the selected card's border, the large display title. Text
   and the fills that carry a label use `--accent-ink`, the same hue and chroma at
   lower lightness. `apps/web/lib/colour.test.ts` asserts all of it against the
   stylesheet, so the table cannot go stale.
6. **Two status colours the file does not have.** The review model names six
   answer states; the frame drew three score pills. `--status-unanswered` and
   `--status-not-required` are additions, deliberately cool so they sit apart from
   the warm accent family the paper's own marks use.
7. **Surfaces sit on three planes, not one.** The file draws panel and card in the
   same white with one shadow. Geometry is unchanged; the card gains its own
   near-white, a lighter shadow and a lit top edge, so it reads as an object on
   the panel rather than a shape drawn on it. Shadows are tinted warm — a neutral
   black shadow on a warm grey page reads as dirt.
8. **The answer-sheet bar is glass, not `#303030`.** Drawn as ink in the frame,
   which put the screen's strongest contrast on a control strip above a scanned
   page and made the two panes read as two applications. It matches the top bar
   instead.
9. **The mapping screen is denser than the frame.** The question card was 96px
   tall to hold one line of text. Padding and the number badge are tightened to
   put roughly a third more of the list on screen. The upload frame is untouched;
   the mapping screen has no Figma frame to diverge from.
