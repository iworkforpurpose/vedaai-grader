"""The gate. Run it before every commit; it fails rather than reports.

The problem it solves is not accuracy, it is churn. Fix after fix landed today
having been verified against part of the set, and the part that was not verified
regressed — the real handwritten script went from exactly its truth to less than
half of it and nobody noticed for three turns, because the numbers that were
checked were the five generated papers.

So this checks every document, and it exits non-zero. Five properties, and each
one is a failure the project has actually shipped:

**Every script's total inside its band.** The band is what a competent marker
could defend, written before any run. Outside it is not a difference of opinion.

**No false credit.** A mark awarded to an answer a human called wrong is the
failure that destroys a teacher's trust, because it is the one they get
challenged on. It has been zero since marking moved to binary checks and it must
stay zero.

**No denominator mismatch.** A wrong denominator makes every mark on a question
wrong at once. One rubric line swallowed by a material scope did this to a whole
paper.

**Nothing answered-but-never-judged.** An unjudged zero and a judged zero are the
same number and completely different facts. A signature mismatch and a
contradictory citation rule have each produced a run of these silently.

**No truth missing from the run.** A question the harness has marks for and the
pipeline never produced is an extraction failure, and scoring the intersection
would let a paper that lost half its questions report a flawless mark error.

**On repeated passes.** Marks move between identical runs, so a single pass cannot
tell a fix from noise — several conclusions today were drawn from one run and one
mark. With ``--passes 3`` each question's median is taken and the spread is
reported, and the band is checked against the median. A question whose spread is
wide is flagged whatever its median, because a mark a teacher cannot reproduce is
not a mark.

    uv run python tooling/scripts/gate.py                 # one pass, every document
    uv run python tooling/scripts/gate.py --passes 3      # before a commit
    uv run python tooling/scripts/gate.py --engine text-layer

Costs paid calls, which is why it is a command and not a unit test. The offline
half of the same job is ``vedaai_evals.classification``, which runs in the test
suite and needs no keys.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "evals" / "src"))
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_marks import DOCUMENTS, SOURCES, facts_from, run_document  # noqa: E402
from vedaai_evals import marks as marks_mod  # noqa: E402
from vedaai_evals import metrics  # noqa: E402

#: How far a question's marks may move across identical passes before the mark is
#: called unreproducible.
#:
#: Half a mark: boards award those, so a swing of one is a different judgement
#: rather than rounding. A teacher who looks twice and sees two numbers has no
#: mark at all, whichever is nearer the truth.
MAX_SPREAD = 0.5


@dataclass
class Failure:
    doc: str
    kind: str
    detail: str


@dataclass
class DocRun:
    doc: str
    reports: list[metrics.ScoringReport] = field(default_factory=list)
    error: str | None = None

    @property
    def median_awarded(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for qid in self.reports[0].awarded if self.reports else {}:
            values = [r.awarded[qid] for r in self.reports if qid in r.awarded]
            if values:
                out[qid] = statistics.median(values)
        return out

    @property
    def spreads(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for qid in self.reports[0].awarded if self.reports else {}:
            values = [r.awarded[qid] for r in self.reports if qid in r.awarded]
            if len(values) > 1:
                out[qid] = max(values) - min(values)
        return out


def evaluate(doc: str, *, engine: str, passes: int, pages: Path) -> DocRun:
    run = DocRun(doc=doc)
    truth = marks_mod.find(doc, extra_roots=SOURCES)
    if truth is None:
        run.error = "no mark truth"
        return run
    for _ in range(passes):
        try:
            submission = run_document(doc, engine=engine, page_root=pages)
        except Exception as exc:  # noqa: BLE001 - one document must not stop the gate
            run.error = f"{type(exc).__name__}: {exc}"
            return run
        run.reports.append(metrics.scoring_scores(truth, facts_from(submission)))
    return run


def failures_for(doc: str, run: DocRun, truth) -> list[Failure]:
    if run.error:
        return [Failure(doc, "did not run", run.error)]
    if not run.reports:
        return [Failure(doc, "no result", "the document produced nothing")]

    first = run.reports[0]
    out: list[Failure] = []

    # The band is checked against the median so that one noisy pass cannot fail a
    # document that is really inside it, nor pass one that is really outside.
    medians = run.median_awarded
    total = sum(medians.values()) if medians else first.awarded_total
    low, high = first.total_band
    if not (low - 1e-6 <= total <= high + 1e-6):
        out.append(
            Failure(doc, "out of band", f"{total:g} against a band of {low:g}-{high:g}")
        )

    if first.false_credit:
        out.append(
            Failure(doc, "false credit", f"marks given to {', '.join(first.false_credit)}")
        )
    if first.denominator_mismatch:
        out.append(
            Failure(
                doc,
                "denominator",
                "; ".join(
                    f"{q}: rubric {a:g} vs printed {b:g}"
                    for q, a, b in first.denominator_mismatch
                ),
            )
        )
    if first.unjudged:
        out.append(
            Failure(doc, "never judged", f"answered but not judged: {', '.join(first.unjudged)}")
        )
    if first.missing_from_run:
        out.append(
            Failure(
                doc,
                "truth missing",
                f"the run never produced {', '.join(first.missing_from_run)}",
            )
        )

    wide = {q: s for q, s in run.spreads.items() if s > MAX_SPREAD}
    if wide:
        out.append(
            Failure(
                doc,
                "unreproducible",
                ", ".join(f"{q} moved {s:g}" for q, s in sorted(wide.items())),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="*", default=None)
    parser.add_argument("--engine", default="text-layer", choices=["default", "text-layer"],
                        help="text-layer removes recognition from the gate, so a marking "
                             "regression cannot hide behind an OCR difference. The scanned "
                             "paper ignores this and uses the recognizer either way.")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--pages", type=Path, default=ROOT / "packages" / "generated" / ".gate")
    args = parser.parse_args(argv)

    documents = args.documents or DOCUMENTS
    print(f"gate · {len(documents)} document(s) · {args.passes} pass(es) · engine {args.engine}")

    all_failures: list[Failure] = []
    rows: list[tuple[str, str, str, str]] = []

    for doc in documents:
        truth = marks_mod.find(doc, extra_roots=SOURCES)
        run = evaluate(doc, engine=args.engine, passes=args.passes, pages=args.pages)
        problems = failures_for(doc, run, truth)
        all_failures.extend(problems)

        if run.error or not run.reports:
            rows.append((doc, "—", "—", "did not run"))
            continue
        first = run.reports[0]
        low, high = first.total_band
        total = sum(run.median_awarded.values()) or first.awarded_total
        rows.append(
            (
                doc,
                f"{total:g}",
                f"{first.truth_total:g} ({low:g}-{high:g})",
                "ok" if not problems else f"{len(problems)} FAIL",
            )
        )

    print(f"\n  {'document':16}{'marks':>7}{'truth (band)':>16}   result")
    for doc, got, want, verdict in rows:
        print(f"  {doc:16}{got:>7}{want:>16}   {verdict}")

    if not all_failures:
        print(f"\n  GATE PASSED — {len(documents)} document(s), every property held.")
        return 0

    print(f"\n  GATE FAILED — {len(all_failures)} problem(s)\n")
    for f in all_failures:
        print(f"  {f.doc:16} {f.kind:16} {f.detail}")
    print(
        "\n  A band is what a competent marker could defend, decided before the run.\n"
        "  Leaving one is a regression, not a difference of opinion. Fix it or change\n"
        "  the truth deliberately and say why in the commit."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
