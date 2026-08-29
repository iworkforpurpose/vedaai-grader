"""Look at every corpus document after a change, and measure the three things.

The user found the defects in this product by opening review pages and looking at
them, not by reading a metric — so this does both. It computes the three quantities
the current work is about, and it drives a real browser so the pages can be seen.

**Label binding.** For each margin number, is the very next line in reading order
the text sitting beside it? On an answer sheet the number `1.` and its first line
share a row, so if reading order emits the text first, every block boundary lands
one line off and every answer picks up its neighbour's words. Reported as the share
of margin labels that bind correctly.

**Ink coverage.** For each answered question, the painted highlight area against the
area of the lines it actually covers. A highlight built as one bounding box per page
scores around 35% here — meaning two thirds of what is drawn is blank paper.

**Question labels.** Just the list, because a phantom question is obvious once the
labels are printed side by side and impossible to see in a count.

    python tooling/scripts/inspect_corpus.py --run before
    python tooling/scripts/inspect_corpus.py --run fix1 --against before --shots

The measurements are standard library only, so any interpreter runs them. `--shots`
additionally drives a browser and therefore needs Playwright, which on this machine
lives at /Users/vighneshnama/Renate-backend/.venv/bin/python — the same interpreter
`audit_ui.py` has always assumed. Stated here because a missing import three minutes
into a run is a worse way to find that out.

Screenshots land in data/corpus/shots/<run>/, which is not tracked.
"""

# ruff: noqa: E501 - report lines read better unwrapped.

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
RUNS = CORPUS / "runs"


def area(box: dict) -> float:
    return max(0.0, box["x1"] - box["x0"]) * max(0.0, box["y1"] - box["y0"])


def overlaps_vertically(a: dict, b: dict) -> float:
    """Share of the shorter box's height that the two boxes share."""
    top, bottom = max(a["y0"], b["y0"]), min(a["y1"], b["y1"])
    if bottom <= top:
        return 0.0
    shorter = min(a["y1"] - a["y0"], b["y1"] - b["y0"])
    return (bottom - top) / shorter if shorter > 0 else 0.0


#: A line no wider than this share of the page is a candidate margin label.
_LABEL_MAX_WIDTH = 0.08

#: Two boxes sharing at least this much of their height are on the same row.
_SAME_ROW = 0.5


def label_binding(submission: dict) -> tuple[int, int, list[str]]:
    """How many margin labels are immediately followed by their own line."""
    lines = (submission.get("answer_sheet_lines") or {}).get("lines") or []
    bound = total = 0
    broken: list[str] = []

    for index, line in enumerate(lines):
        box = line["box"]
        if (box["x1"] - box["x0"]) > _LABEL_MAX_WIDTH:
            continue
        if len(line["text"].strip()) > 6:
            continue

        # Its row-mate: the widest line on the same page sharing this line's row.
        mates = [
            other for other in lines
            if other is not line
            and other["page"] == line["page"]
            and overlaps_vertically(box, other["box"]) >= _SAME_ROW
        ]
        if not mates:
            continue
        total += 1
        mate = max(mates, key=lambda other: area(other["box"]))
        following = lines[index + 1] if index + 1 < len(lines) else None
        if following is not None and following["line_id"] == mate["line_id"]:
            bound += 1
        else:
            broken.append(f"{line['text'].strip()!r} p{line['page']} -> reads {(following or {}).get('text', '')[:34]!r}")

    return bound, total, broken


def ink_coverage(submission: dict) -> tuple[float, list[tuple[str, float]]]:
    """Painted highlight area versus the area of the lines under it."""
    lines = {line["line_id"]: line for line in (submission.get("answer_sheet_lines") or {}).get("lines", [])}
    blocks = {block["block_id"]: block for block in submission.get("blocks") or []}

    per_question: list[tuple[str, float]] = []
    for entry in (submission.get("mapping") or {}).get("mappings", []):
        block_ids = entry.get("block_ids") or []
        if not block_ids:
            continue
        painted = sum(area(pb["box"]) for pb in (entry.get("highlight") or {}).get("boxes", []))
        if painted <= 0:
            continue
        ink = 0.0
        for block_id in block_ids:
            for line_id in blocks.get(block_id, {}).get("line_ids", []):
                if line_id in lines:
                    ink += area(lines[line_id]["box"])
        per_question.append((entry["qid"], min(1.0, ink / painted)))

    average = sum(c for _q, c in per_question) / len(per_question) if per_question else 0.0
    return average, per_question


async def shoot(base: str, run: str, name: str, sid: str, worst: list[str]) -> None:
    """Open the review page and photograph it, plus its loosest highlights."""
    from playwright.async_api import async_playwright

    out = CORPUS / "shots" / run
    out.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        try:
            await page.goto(f"{base}/review/{sid}", wait_until="networkidle", timeout=90_000)
            await page.wait_for_timeout(1500)

            # Refuse to save a picture of an error page. One run against the
            # deployed task produced seven screenshots of "Service Unavailable",
            # which look like results until you open them.
            body = (await page.inner_text("body"))[:200]
            if "Service Unavailable" in body or "Application error" in body:
                raise RuntimeError(f"{base} returned an error page for {sid}: {body[:80]}")
            await page.screenshot(path=str(out / f"{name}.png"), full_page=False)

            # The two loosest highlights, selected and photographed, because a
            # coverage number does not show *where* the extra rectangle went.
            for rank, qid in enumerate(worst[:2]):
                cards = page.locator(".q-card")
                count = await cards.count()
                for index in range(count):
                    card = cards.nth(index)
                    text = (await card.inner_text())[:40]
                    if qid.split("/")[-1] in text:
                        await card.click()
                        await page.wait_for_timeout(900)
                        sheet = page.locator(".sheet-pane, .map-pane").first
                        target = sheet if await sheet.count() else page
                        await target.screenshot(path=str(out / f"{name}-{rank}-{qid.replace('/', '_')}.png"))
                        break
        finally:
            await browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Which run label to inspect")
    parser.add_argument("--against", default=None, help="An earlier run to compare with")
    parser.add_argument("--base", default="https://wvqyfdkpl1.execute-api.ap-south-1.amazonaws.com",
                        help="Where to read submission JSON from")
    parser.add_argument("--web", default="http://localhost:3001",
                        help="Where to point the browser. Defaults to the local dev server: "
                             "the deployed task is a single 1 GB container and it recycled "
                             "under a whole-corpus run, which turned every screenshot into "
                             "a 503 page.")
    parser.add_argument("--shots", action="store_true", help="Also drive a browser and screenshot")
    args = parser.parse_args()

    run = json.loads((RUNS / f"{args.run}.json").read_text())
    earlier = {}
    if args.against and (RUNS / f"{args.against}.json").exists():
        earlier = json.loads((RUNS / f"{args.against}.json").read_text())

    totals: list[tuple[str, float, float]] = []

    for name, summary in run.items():
        sid = summary["submission_id"]
        with urllib.request.urlopen(f"{args.base}/api/submissions/{sid}", timeout=40) as response:
            submission = json.load(response)

        bound, total, broken = label_binding(submission)
        coverage, per_question = ink_coverage(submission)
        # None, not 1.0, when the script carries no margin numbers at all. Scoring
        # "nothing to bind" as perfect would put three of the seven at 100% and
        # drag the mean up while measuring nothing.
        share = (bound / total) if total else None
        totals.append((name, share, coverage))

        print(f"\n{'=' * 74}\n{name}  [{sid}]")
        print(f"  questions: {summary['questions']}  {summary.get('question_labels')}")
        if total:
            print(f"  label binding: {bound}/{total} margin labels bind to their own line ({share:.0%})")
        else:
            print("  label binding: n/a — this script writes no question numbers")
        for note in broken[:3]:
            print(f"      broken: {note}")
        print(f"  ink coverage: {coverage:.0%} of highlighted area is actually text")
        worst = sorted(per_question, key=lambda pair: pair[1])[:3]
        for qid, value in worst:
            print(f"      {qid:10} {value:.0%}")

        if args.shots:
            asyncio.run(shoot(args.web, args.run, name, sid, [q for q, _ in worst]))
            print(f"  shot: data/corpus/shots/{args.run}/{name}.png")

        if name in earlier:
            print(f"  earlier run: questions={earlier[name]['questions']} mapping={earlier[name]['mapping']}")

    print(f"\n{'=' * 74}\nAcross the corpus")
    print(f"  {'example':22} {'label binding':>14} {'ink coverage':>14}")
    for name, share, coverage in totals:
        binding = "         n/a" if share is None else f"{share:12.0%}"
        print(f"  {name:22} {binding} {coverage:13.0%}")

    measurable = [s for _n, s, _c in totals if s is not None]
    if measurable:
        print(f"  {'MEAN (of ' + str(len(measurable)) + ' with labels)':22} "
              f"{sum(measurable) / len(measurable):12.0%} "
              f"{sum(c for _n, _s, c in totals) / len(totals):13.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
