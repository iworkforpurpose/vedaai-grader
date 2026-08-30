"""Put the fresh papers through the service and print what came back.

Separate from `rerun_corpus.py` on purpose. That script re-runs documents the
system has already been fixed against, which makes it a regression check and not
evidence about anything new. This one runs papers nothing has been tuned to, and
prints enough per question to judge each one by eye rather than by a total.

    uv run python tooling/scripts/run_fresh.py                       # against live
    uv run python tooling/scripts/run_fresh.py --base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from build_corpus import LIVE  # noqa: E402  - same directory
from rerun_corpus import api_prefix, submit, wait  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FRESH = ROOT / "data" / "fresh"

#: What the student actually did, written down before running anything. Without
#: this the output is just a list of statuses with nothing to check them against.
EXPECTED: dict[str, str] = {
    "history": "7 questions. Q.3 answered last and out of order, both its parts in one run. Q.5 never answered. The source extract is not a question.",
    "geography": "6 questions. A figure sits between question 2's stem and its parts. Question 3 asks for a sketch and is blank.",
    "english": "5 questions. Section B asks for any two of three; 5 and 3 are answered, in that order, so 4 is not required.",
    "economics": "5 questions. Q2 prints [4] while the section says 3. One answer is labelled Q4 in the margin but answers Q3. Q4 is never answered.",
}


def report(base: str, name: str, sid: str) -> None:
    prefix = api_prefix(base)
    with urllib.request.urlopen(f"{base}{prefix}/submissions/{sid}", timeout=90) as response:
        sub = json.load(response)

    questions = (sub.get("questions") or {}).get("questions", [])
    grades = {g["qid"]: g for g in (sub.get("grades") or {}).get("grades", [])}
    mapping = {m["qid"]: m for m in (sub.get("mapping") or {}).get("mappings", [])}
    blocks = {b["block_id"]: b for b in sub.get("blocks", [])}

    print(f"\n{'═' * 78}\n{name.upper()}   {base}/review/{sid}")
    print(f"  expected: {EXPECTED.get(name, '')}")
    print(f"  extracted {len(questions)} questions: {[q['label_raw'] for q in questions]}")

    for warning in sub.get("warnings") or []:
        print(f"  ! {warning[:150]}")

    print(f"\n  {'label':10} {'status':13} {'marks':>9}  {'boxes':>5}  first words of what it points at")
    for q in questions:
        m = mapping.get(q["qid"], {})
        g = grades.get(q["qid"])
        marks = f"{g['marks_awarded']:g}/{g['marks_available']:g}" if g else "—"
        boxes = len((m.get("highlight") or {}).get("boxes") or [])
        text = " ".join(
            blocks[b]["text"] for b in (m.get("block_ids") or []) if b in blocks
        )[:58]
        print(f"  {q['label_raw']:10} {m.get('status', ''):13} {marks:>9}  {boxes:>5}  {text}")

    orphans = (sub.get("mapping") or {}).get("orphans") or []
    if orphans:
        print(f"  unplaced: {[o['text_preview'][:40] for o in orphans]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=LIVE)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    names = [args.only] if args.only else sorted(p.name for p in FRESH.iterdir() if p.is_dir())
    for name in names:
        folder = FRESH / name
        sid = submit(args.base, folder / "paper.pdf", folder / "script.pdf")
        wait(args.base, sid)
        report(args.base, name, sid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
