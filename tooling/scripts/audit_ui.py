"""Sweep the running app for layout faults, across viewports and screen states.

Written because a whole class of bug here is invisible to unit tests and easy to
miss by eye: a scroller that can be dragged past its content, a control clipped
outside its pane, an image stretched by a missing `height: auto`. Each one reads as
"the site feels buggy" without ever failing an assertion.

The check that earns its place is `dead-scroll` — scrollable distance beyond the
last painted pixel inside the box. It found a phone review screen whose entire
content column could be dragged 810px into empty background, because a stale
`overflow: visible` on the question list sent its overflow up to the first ancestor
that would take it.

Two lessons are baked into the metric itself. It measures the deepest painted
bottom among *all* descendants rather than the direct children's boxes, because a
child shorter than its own content otherwise makes real content below the fold look
like emptiness — which produced a 265px false positive on the upload screen. And it
reports padding on scrollers separately rather than counting it as a fault, because
a deliberate top inset and 810px of nothing are not the same finding.

Usage:
    python tooling/scripts/audit_ui.py <submission_id> [--base http://127.0.0.1:3001]

Needs a marked submission to audit the review screen, and the dev servers running.
Exits non-zero if anything but scroller padding is reported.
"""

import asyncio
from playwright.async_api import async_playwright
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("submission", help="A submission id that has been marked.")
parser.add_argument("--base", default="http://127.0.0.1:3001")
ARGS = parser.parse_args()
SUB = ARGS.submission
BASE = ARGS.base
PROBE = (Path(__file__).parent / "audit_ui.js").read_text()
SIZES = [("phone", 393, 852), ("phone-sm", 320, 568), ("tablet", 768, 1024),
         ("laptop", 1440, 900), ("laptop-short", 1440, 600), ("wide", 1920, 1080)]

async def audit(pg, label, extra=""):
    findings = await pg.evaluate(PROBE)
    if not findings:
        print(f"  {label}{extra}: clean")
    else:
        seen = {}
        for f in findings:
            seen.setdefault(f["kind"], []).append(f["detail"])
        print(f"  {label}{extra}:")
        for kind, details in seen.items():
            for det in details[:3]:
                print(f"     [{kind}] {det}")
            if len(details) > 3:
                print(f"     [{kind}] ... +{len(details)-3} more")
    return findings

async def main():
    all_f = {}
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for label, w, h in SIZES:
            ctx = await b.new_context(viewport={"width": w, "height": h})
            pg = await ctx.new_page()

            print(f"\n=== {label} {w}x{h} — UPLOAD")
            await pg.goto(f"{BASE}/", wait_until="networkidle", timeout=60000)
            await pg.wait_for_timeout(900)
            all_f[f"{label}/upload"] = await audit(pg, "upload")

            print(f"=== {label} {w}x{h} — REVIEW")
            await pg.goto(f"{BASE}/review/{SUB}", wait_until="networkidle", timeout=60000)
            await pg.wait_for_timeout(2200)
            all_f[f"{label}/review-top"] = await audit(pg, "review", " (at top)")

            # Sheet tab on narrow, then scrolled to the very bottom.
            tab = pg.locator('.map-tab', has_text="Answer Sheet")
            if await tab.count() and await tab.first.is_visible():
                await tab.first.click(); await pg.wait_for_timeout(700)
            await pg.evaluate("()=>{const n=document.querySelector('.sheet-scroll'); if(n) n.scrollTop=n.scrollHeight;}")
            await pg.wait_for_timeout(700)
            all_f[f"{label}/review-bottom"] = await audit(pg, "review", " (sheet scrolled to end)")

            # Expand everything, which is where clipping tends to show up.
            ea = pg.locator('.q-head-action', has_text="Expand All")
            qt = pg.locator('.map-tab', has_text="Questions")
            if await qt.count() and await qt.first.is_visible():
                await qt.first.click(); await pg.wait_for_timeout(500)
            if await ea.count() and await ea.first.is_visible():
                await ea.first.click(); await pg.wait_for_timeout(800)
            all_f[f"{label}/review-expanded"] = await audit(pg, "review", " (all expanded)")

            # The navigation drawer, opened.
            #
            # This was the audit's blind spot: every check skipped elements inside a
            # closed rail, on the grounds that an off-canvas drawer is off-canvas by
            # design — and then nothing ever opened it. A 304px drawer containing
            # eight unlabelled icons sat there through a clean run.
            opener = pg.locator('button[aria-label="Open navigation"]')
            if await opener.count() and await opener.first.is_visible():
                await opener.first.click()
                await pg.wait_for_timeout(700)
                all_f[f"{label}/nav-drawer"] = await audit(pg, "nav drawer", " (open)")
            await ctx.close()
        await b.close()
    kinds = {}
    for v in all_f.values():
        for f in v:
            kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    total = sum(kinds.values())
    print(f"\n===== {total} findings across {len(all_f)} states")
    for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {n:>4}  {k}")
    # Scroller padding is reported, not failed: a deliberate inset is not a bug.
    return 1 if any(k != "scroller-padding" for k in kinds) else 0


raise SystemExit(asyncio.run(main()))
