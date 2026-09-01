"""What the parser thinks every line of every paper is, frozen.

Nearly every regression this project has shipped came from one place: the line
classifier. Eight rules decide what a line is — the page-number pattern, the
identity fields, repetition at the page edge, quotation marks, the material scope,
the label grammar, ``reads_as_a_heading``, the row test — and they interact only
through the order they run in. Nothing states that order as a fact, so a new rule
changes an old answer silently.

The list of what that cost, all of it found by a person looking at output rather
than by a test:

* "Read the following carefully." opened a material scope that swallowed "Each
  question carries 4 marks", so a paper's questions had no denominator at all.
* ``Page : 03`` sat one hundredth of a page below the header band, escaped every
  positional rule, and led a block — and anchors are read from a block's first
  line.
* A source extract became the question above it, then became furniture, losing
  eight marks in one direction and then in the other.
* An economics table's rows matched the bare-page-number pattern and vanished, so
  "calculate the elasticity between the first and second rows" was marked with no
  rows.
* The student's own margin numbers matched the same pattern and were deleted from
  every page.

Every one of those is a line whose role changed. So the roles are snapshotted: one
file per paper, one line per line, role and text. A change that reclassifies
anything fails and prints which line and from what to what. That turns an
invisible interaction into a visible diff, at the moment it is made rather than
three commits later.

**Only papers with a text layer.** The classifier is deterministic given lines, so
the snapshot has to be too — and a paper that needs recognition is not. The real
scanned paper is therefore absent here and covered by the marking gate instead,
which runs the whole pipeline anyway.

Regenerate deliberately, never casually::

    uv run python -m vedaai_evals.classification --update

and read the diff it prints before committing it. A snapshot updated without
looking is worse than no snapshot: it records whatever the code now does as
though it were intended.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

#: This file sits at ``packages/evals/src/vedaai_evals/classification.py``, so 0 is
#: the module directory, 1 is ``src``, 2 is the package and 4 is the repository.
#: Counted out because getting it wrong here is silent: every paper reports "no
#: paper, skipped" and the run exits as though there were nothing to check.
PACKAGE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[4]
SNAPSHOTS = PACKAGE / "snapshots" / "classification"

#: Papers whose question paper carries a text layer, so classification is exact.
#:
#: Named rather than globbed: a snapshot suite that silently covers fewer papers
#: than it used to is a suite that stops catching things, and a missing directory
#: should fail loudly instead of shrinking the run.
PAPERS = {
    "history": "data/fresh/history",
    "geography": "data/fresh/geography",
    "english": "data/fresh/english",
    "economics": "data/fresh/economics",
    "physics": "data/fresh/physics",
    "asap-clean": "data/asap-real/asap-clean",
    "asap-middling": "data/asap-real/asap-middling",
    "asap-worst": "data/asap-real/asap-worst",
}


@dataclass(frozen=True)
class Row:
    role: str
    text: str

    def rendered(self) -> str:
        # Whitespace collapsed so a rendering change in the generator does not
        # register as a classification change.
        return f"{self.role}\t{' '.join(self.text.split())}"


def classify_paper(folder: Path) -> list[Row]:
    """Every line of one question paper, with the role the parser gives it."""
    import fitz
    from grader.lineindex import build_index
    from grader.ocr import PdfTextLayerEngine
    from grader.ocr.base import PageInput
    from grader.questions import furniture
    from vedaai_contracts import DocumentKind

    data = (folder / "paper.pdf").read_bytes()
    document = fitz.open(stream=data, filetype="pdf")
    count = document.page_count
    document.close()

    engine = PdfTextLayerEngine()
    per_page = [
        engine.transcribe(
            PageInput(index=i, width=1000, height=1400, document=data, filename="paper.pdf")
        )
        for i in range(count)
    ]
    index = build_index(
        DocumentKind.QUESTION_PAPER, per_page, engine.engine, trust_engine_order=True
    )
    roles = furniture.classify_all(index.lines)
    return [Row(role=roles[ln.line_id].value, text=ln.text) for ln in index.lines]


def snapshot_path(name: str) -> Path:
    return SNAPSHOTS / f"{name}.txt"


def write(name: str, rows: list[Row]) -> None:
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    snapshot_path(name).write_text("\n".join(r.rendered() for r in rows) + "\n")


def read(name: str) -> list[str] | None:
    path = snapshot_path(name)
    if not path.is_file():
        return None
    return path.read_text().splitlines()


def diff(expected: list[str], actual: list[str]) -> list[str]:
    """Where the two disagree, phrased so the fault is readable.

    Line-by-line rather than a unified diff, because the interesting fact is
    always "this line changed role" and a unified diff buries that in context.
    """
    out: list[str] = []
    for i in range(max(len(expected), len(actual))):
        want = expected[i] if i < len(expected) else None
        got = actual[i] if i < len(actual) else None
        if want == got:
            continue
        if want is None:
            out.append(f"  line {i + 1}: appeared      {got}")
        elif got is None:
            out.append(f"  line {i + 1}: disappeared   {want}")
        else:
            was_role, _, was_text = want.partition("\t")
            now_role, _, now_text = got.partition("\t")
            if was_text == now_text:
                out.append(
                    f"  line {i + 1}: {was_role} -> {now_role}    {was_text[:64]!r}"
                )
            else:
                out.append(f"  line {i + 1}: was  {want[:78]}")
                out.append(f"  line {i + 1}: now  {got[:78]}")
    return out


def check(name: str) -> list[str]:
    """Problems with one paper. Empty means it matches."""
    folder = ROOT / PAPERS[name]
    if not (folder / "paper.pdf").is_file():
        return [f"  no paper at {folder} — build the fixtures first"]

    actual = [r.rendered() for r in classify_paper(folder)]
    expected = read(name)
    if expected is None:
        return [f"  no snapshot yet; run with --update to record {len(actual)} lines"]
    return diff(expected, actual)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="Rewrite the snapshots. Read the diff it prints first.")
    parser.add_argument("papers", nargs="*", default=None)
    args = parser.parse_args(argv)

    names = args.papers or list(PAPERS)
    failures = 0
    for name in names:
        if name not in PAPERS:
            print(f"{name}: not a snapshotted paper")
            failures += 1
            continue
        problems = check(name)
        if args.update:
            folder = ROOT / PAPERS[name]
            if not (folder / "paper.pdf").is_file():
                print(f"{name}: no paper, skipped")
                continue
            rows = classify_paper(folder)
            if problems:
                print(f"{name}: {len(problems)} change(s)")
                for line in problems:
                    print(line)
            write(name, rows)
            print(f"{name}: recorded {len(rows)} lines")
            continue

        if problems:
            failures += 1
            print(f"\n{name}: CLASSIFICATION CHANGED")
            for line in problems:
                print(line)
        else:
            print(f"{name}: unchanged")

    if failures and not args.update:
        print(
            f"\n{failures} paper(s) classify differently than recorded.\n"
            "If the change is intended, run with --update and commit the new snapshot\n"
            "in the same change as the code — but read the diff above first: every\n"
            "regression this file exists to catch looked like an unrelated line moving."
        )
    return 1 if failures and not args.update else 0


if __name__ == "__main__":
    sys.exit(main())
