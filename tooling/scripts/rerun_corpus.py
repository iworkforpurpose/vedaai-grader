"""Put the whole corpus through the pipeline again, and say what changed.

The point of a fix is not that a number went up somewhere; it is that a specific
thing that was wrong on a specific document is now right, and that nothing else
moved. So this re-runs all seven documents, stores the result under a label, and
diffs it against a previous run field by field — questions found, what each was
labelled, how the mapping came out, orphans, marks.

Runs against the deployed service by default, because that is where the corpus's
page images and the marking credential live. Point `--base` at a local API to run
against uncommitted code.

    # after a fix
    uv run python tooling/scripts/rerun_corpus.py --label fix1-reading-order
    uv run python tooling/scripts/rerun_corpus.py --label fix1-reading-order --against baseline

Results land in data/corpus/runs/<label>.json, which is not tracked.
"""

# ruff: noqa: E501 - the comparison table is easier to read unwrapped.

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

from access import unlock  # noqa: E402  - same directory
from build_corpus import EXAMPLES, LIVE, summarise  # noqa: E402

#: The deployed service sits behind Next, which owns the /api prefix and proxies
#: past it. A local uvicorn is the bare API and has no prefix. Deriving it from the
#: address avoids a flag that has to be remembered in step with --base.
#:
#: Keyed on the host rather than the port. Port 8000 was the same statement while
#: only one local API ever ran, and stopped being one the first time a second had
#: to come up beside it on 8001 — a bare API is bare wherever it is listening.
def api_prefix(base: str) -> str:
    return "" if ("127.0.0.1" in base or "localhost" in base) else "/api"


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
RUNS = CORPUS / "runs"


def submit(base: str, paper: Path, script: Path) -> str:
    """Upload one pair the way the browser does, and return the submission id."""
    API = api_prefix(base)
    request = urllib.request.Request(
        f"{base}{API}/uploads", method="POST",
        data=json.dumps({"question_paper_name": paper.name,
                         "answer_sheet_name": script.name}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        plan = json.load(response)

    body_parts: list[str] = []
    boundary = "----" + uuid.uuid4().hex

    if plan.get("mode") == "s3":
        for kind, path in (("question_paper", paper), ("answer_sheet", script)):
            slot = plan["slots"][kind]
            urllib.request.urlopen(
                urllib.request.Request(slot["url"], data=path.read_bytes(), method="PUT"),
                timeout=300,
            )
            body_parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; '
                f'name="{kind}_key"\r\n\r\n{slot["key"]}\r\n'
            )
        payload = ("".join(body_parts) + f"--{boundary}--\r\n").encode()
    else:
        # The local path, where there is no bucket to presign into.
        chunks: list[bytes] = []
        for kind, path in (("question_paper", paper), ("answer_sheet", script)):
            chunks.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{kind}"; '
                f'filename="{path.name}"\r\nContent-Type: application/pdf\r\n\r\n'.encode()
            )
            chunks.append(path.read_bytes())
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        payload = b"".join(chunks)

    request = urllib.request.Request(
        f"{base}{API}/submissions", data=payload, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["submission_id"]


def wait(base: str, sid: str, *, limit: int = 100) -> dict:
    API = api_prefix(base)
    for _ in range(limit):
        with urllib.request.urlopen(f"{base}{API}/submissions/{sid}", timeout=40) as response:
            body = json.load(response)
        if body["status"] in ("complete", "failed"):
            return body
        time.sleep(5)
    return body


def compare(name: str, before: dict | None, after: dict) -> list[str]:
    """The fields that moved, in words rather than as a diff."""
    if before is None:
        return ["(no earlier run to compare against)"]

    notes: list[str] = []
    for field in ("status", "questions", "answer_pages", "paper_lines", "sheet_lines",
                  "blocks", "orphans", "warnings"):
        old, new = before.get(field), after.get(field)
        if old != new:
            arrow = "→"
            direction = ""
            if isinstance(old, int) and isinstance(new, int):
                direction = "  (better)" if new > old and field == "questions" else ""
            notes.append(f"{field}: {old} {arrow} {new}{direction}")

    if before.get("mapping") != after.get("mapping"):
        notes.append(f"mapping: {before.get('mapping')} → {after.get('mapping')}")
    if before.get("marks") != after.get("marks"):
        notes.append(f"marks: {before.get('marks')} → {after.get('marks')}")

    old_labels = before.get("question_labels") or []
    new_labels = after.get("question_labels") or []
    if old_labels != new_labels:
        gained = [x for x in new_labels if x not in old_labels]
        lost = [x for x in old_labels if x not in new_labels]
        if gained:
            notes.append(f"questions gained: {gained}")
        if lost:
            notes.append(f"questions LOST: {lost}")

    return notes or ["unchanged"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=LIVE)
    parser.add_argument("--label", required=True, help="Name this run, e.g. fix1-reading-order")
    parser.add_argument("--against", default=None, help="An earlier run label to diff against")
    parser.add_argument("--only", default=None, help="Run one example by name")
    args = parser.parse_args()
    unlock(args.base)

    RUNS.mkdir(parents=True, exist_ok=True)
    previous: dict = {}
    if args.against:
        path = RUNS / f"{args.against}.json"
        if path.exists():
            previous = json.loads(path.read_text())
        elif args.against == "baseline":
            raw = json.loads((CORPUS / "baseline.json").read_text())
            previous = {k: v["original"] for k, v in raw.items()}
        else:
            print(f"  no such run: {args.against}")

    results: dict[str, dict] = {}
    for name, _sid, why in EXAMPLES:
        if args.only and args.only != name:
            continue
        folder = CORPUS / name
        paper, script = folder / "paper.pdf", folder / "script.pdf"
        if not (paper.exists() and script.exists()):
            print(f"\n{name}: not built, skipped")
            continue

        started = time.time()
        new_id = submit(args.base, paper, script)
        body = wait(args.base, new_id)
        summary = summarise(body)
        summary["submission_id"] = new_id
        summary["seconds"] = round(time.time() - started)
        results[name] = summary

        print(f"\n{name}  [{new_id}]   {summary['seconds']}s")
        print(f"  {why}")
        print(f"  questions={summary['questions']}  mapping={summary['mapping']}  "
              f"orphans={summary['orphans']}  marks={summary['marks']}")
        for note in compare(name, previous.get(name), summary):
            print(f"    · {note}")

    (RUNS / f"{args.label}.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\n  written: {(RUNS / f'{args.label}.json').relative_to(ROOT)}")
    print("  review pages:")
    for name, summary in results.items():
        print(f"    {name:22} {args.base}/review/{summary['submission_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
