"""How many blocks landed on the question they actually answer.

The corpus already measures label binding and highlight tightness, and neither
says whether an answer reached the right question — which is the thing the product
is for. This does, against ground truth read off the pages by eye and written down
below.

Only documents whose correct mapping is unambiguous are included. The programming
and mathematics scripts are left out on purpose: their transcription is too damaged
to say what a block "should" match without guessing, and a number that depends on
my guess is not a measurement.

    python tooling/scripts/score_mapping.py data/corpus/runs/<label>.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

#: Which question the writing starting on each page answers.
#:
#: Established by reading the pages, not by running the pipeline — the whole point
#: is to have something the pipeline cannot talk its way out of.
TRUTH: dict[str, dict[int, str]] = {
    # Four ASAP pages: two answer the pandas/koalas question, two the "invasive"
    # question.
    "real-ink": {0: "A/1", 1: "A/1", 2: "A/2", 3: "A/2"},
    # Answered out of order — the "invasive" answer is on page one and the
    # pandas answer on page two. Written down the wrong way round at first, which
    # scored a correct mapping as a failure and made a fix look like a regression;
    # ground truth is only ground truth if you read the pages.
    "comprehension-user": {0: "A/2", 1: "A/1"},
    "prose-matched": {0: "A/1", 1: "A/2"},
}

#: Shorter than this and a block is a fragment — a trailing word cut off by a page
#: break — which carries no opinion about which question it belongs to.
_MIN_BLOCK_CHARS = 25


def api_prefix(base: str) -> str:
    return "" if ":8000" in base else "/api"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="A run file from rerun_corpus.py")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    prefix = api_prefix(args.base)
    run = json.loads(args.run.read_text())
    correct = total = 0

    for name, truth in TRUTH.items():
        if name not in run:
            continue
        sid = run[name]["submission_id"]
        with urllib.request.urlopen(
            f"{args.base}{prefix}/submissions/{sid}", timeout=40
        ) as response:
            submission = json.load(response)

        lines = {
            line["line_id"]: line
            for line in (submission.get("answer_sheet_lines") or {}).get("lines", [])
        }
        owner: dict[str, list[str]] = {}
        for entry in (submission.get("mapping") or {}).get("mappings", []):
            for block_id in entry.get("block_ids") or []:
                owner.setdefault(block_id, []).append(entry["qid"])

        hits = seen = 0
        for block in submission.get("blocks") or []:
            line_ids = block.get("line_ids") or []
            if not line_ids or len(block.get("text", "").strip()) < _MIN_BLOCK_CHARS:
                continue
            pages = sorted({lines[i]["page"] for i in line_ids if i in lines})
            wanted = truth.get(pages[0]) if pages else None
            if wanted is None:
                continue
            seen += 1
            hits += wanted in owner.get(block["block_id"], [])

        print(f"  {name:22} {hits}/{seen}")
        correct += hits
        total += seen

    if total:
        print(f"  {'TOTAL':22} {correct}/{total}  ({correct / total:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
