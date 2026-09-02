"""How close the proposed marks came, against marks decided before the run.

Every figure this project publishes measures where an answer is or which question
it belongs to. None of them measures whether the marks are right, so "scoring
accuracy" has never had a number. This is that number.

Runs in-process rather than against a live service, for the same reason the eval
runner does: a metric that needs a server running is a metric nobody runs.

    uv run python tooling/scripts/score_marks.py                    # all five
    uv run python tooling/scripts/score_marks.py economics --verbose
    uv run python tooling/scripts/score_marks.py english --engine text-layer
    uv run python tooling/scripts/score_marks.py economics --repeat 3

**On ``--engine``.** The default puts the answer sheet through the recognizer the
deployed service uses, which is the honest end-to-end number. ``text-layer`` reads
a generated script from its PDF text layer instead, which is exact -- so mark
error measured that way is the marker's alone, with recognition removed. Quoting
one and meaning the other is the mistake this flag exists to prevent, and it is
the same device ``pipeline.ingest_document`` already offers the eval harness.

Marking costs a paid call per question, so this spends money. Roughly thirty
questions across the five documents on one pass.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "evals" / "src"))
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))

from grader import grading, pipeline, regions, render  # noqa: E402
from grader.ocr import PdfTextLayerEngine  # noqa: E402
from grader.storage import PageStore  # noqa: E402
from grader.store import SubmissionStore  # noqa: E402
from vedaai_contracts import DocumentKind, Submission  # noqa: E402
from vedaai_evals import marks as marks_mod  # noqa: E402
from vedaai_evals import metrics  # noqa: E402

#: Where the documents live. Fresh papers are generated; the corpus is real work.
SOURCES = [ROOT / "data" / "fresh", ROOT / "data" / "corpus", ROOT / "data" / "asap-real"]

#: The five documents that carry mark truth today.
DOCUMENTS = ["history", "geography", "english", "economics", "physics", "math-paper",
             "asap-clean", "asap-middling", "asap-worst"]


def locate(doc: str) -> tuple[Path, Path] | None:
    for root in SOURCES:
        paper, script = root / doc / "paper.pdf", root / doc / "script.pdf"
        if paper.is_file() and script.is_file():
            return paper, script
    return None


@dataclass
class Outcome:
    submission: Submission
    report: metrics.ScoringReport


def run_document(doc: str, *, engine: str, page_root: Path) -> Submission:
    """Ingest and mark one document, in process."""
    found = locate(doc)
    if found is None:
        raise FileNotFoundError(f"no paper.pdf/script.pdf for {doc!r} under data/")
    paper_path, script_path = found

    paper_bytes, script_bytes = paper_path.read_bytes(), script_path.read_bytes()
    paper = render.inspect(paper_bytes, "paper.pdf", DocumentKind.QUESTION_PAPER)
    script = render.inspect(script_bytes, "script.pdf", DocumentKind.ANSWER_SHEET)

    store = SubmissionStore()
    store.put(Submission(submission_id=doc))

    # Exact transcription, so mark error is the marker's alone. Refused by normal
    # engine selection because a real scanned sheet's text layer is spurious; on a
    # generated script it is the ground truth.
    #
    # And only where there *is* one. A scanned sheet has no text layer, so forcing
    # the reader on it returns zero lines, nothing is marked, and the document
    # scores zero — which reads as a catastrophic regression rather than as the
    # harness asking for something impossible. The gate reported exactly that on
    # four documents the first time it ran.
    override = None
    if engine == "text-layer":
        if script.has_text_layer:
            override = PdfTextLayerEngine()
        else:
            print(f"    {doc}: no text layer on the script, using the recognizer")

    submission = pipeline.ingest(
        submission_id=doc,
        question_paper=(paper_bytes, paper),
        answer_sheet=(script_bytes, script),
        page_store=PageStore(root=page_root / doc),
        submission_store=store,
        answer_engine_override=override,
    )

    if submission.mapping is None or submission.questions is None:
        return submission
    if submission.answer_sheet_lines is None:
        return submission

    excluded = regions.lines_excluded_from_grading(
        submission.ink_regions, submission.answer_sheet_lines.lines
    )

    async def mark():
        """One loop per document, and the client closed inside it.

        The grader is built here rather than outside so that its HTTP client
        lives and dies with the loop it was used on. Built outside, it was
        finalised by the garbage collector after ``asyncio.run`` had closed the
        loop, which printed a bare ``RuntimeError: Event loop is closed`` above
        otherwise correct output.
        """
        grader = grading.select_grader()
        try:
            return await grading.grade_submission(
                paper=submission.questions,
                mapping=submission.mapping,
                index=submission.answer_sheet_lines,
                grader=grader,
                excluded_line_ids=excluded,
            )
        finally:
            close = getattr(grader, "aclose", None)
            if close is not None:
                await close()

    submission.grades, failures = asyncio.run(mark())
    submission.warnings.extend(f for f in failures if f not in submission.warnings)
    return submission


def facts_from(submission: Submission) -> dict[str, metrics.GradedFact]:
    """Flatten one submission into what the metric needs.

    The rubric is re-derived here rather than read off the grade, because the
    thing being checked is whether the derived criteria sum to the marks the paper
    printed -- and a grade that inherited a wrong split would agree with itself.
    """
    statuses = {m.qid: m for m in (submission.mapping.mappings if submission.mapping else [])}
    grades = {g.qid: g for g in (submission.grades.grades if submission.grades else [])}
    questions = {q.qid: q for q in (submission.questions.questions if submission.questions else [])}

    out: dict[str, metrics.GradedFact] = {}
    for qid, question in questions.items():
        grade = grades.get(qid)
        mapping = statuses.get(qid)
        spec = grading.derive(question)

        points = grade.rubric_points if grade else []
        awarded_points = [p for p in points if p.marks_awarded > 0]
        incoherent = [
            p
            for p in points
            if (p.satisfied and p.marks_awarded < p.marks_available)
            or (not p.satisfied and p.marks_available > 0 and p.marks_awarded >= p.marks_available)
        ]
        # The reason a question went unmarked is written into every point's
        # comment by the grading package, so the first one carries it.
        reason = points[0].comment if points and not (grade and grade.judged) else ""

        out[qid] = metrics.GradedFact(
            qid=qid,
            status=mapping.status.value if mapping else "unanswered",
            marks_available=float(grade.marks_available if grade else (question.marks or 0)),
            marks_awarded=float(grade.marks_awarded if grade else 0.0),
            judged=bool(grade and grade.judged),
            unmarked_reason=(reason or "").strip(),
            rubric_marks_sum=float(sum(c.marks for c in spec.criteria)),
            criteria=len(spec.criteria),
            awarded_points=len(awarded_points),
            cited_points=sum(1 for p in awarded_points if p.cited_line_ids),
            incoherent_points=len(incoherent),
        )
    return out


def pct(value: float | None) -> str:
    return "   n/a" if value is None else f"{value * 100:5.1f}%"


def num(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.2f}"


def report_one(doc: str, truth: marks_mod.MarkSet, outcome: Outcome, *, verbose: bool) -> None:
    r = outcome.report
    low, high = r.total_band

    print(f"\n{'─' * 76}\n{doc.upper()}   {truth.source} truth · {len(r.scored)} scored, "
          f"{len(r.excluded)} excluded")

    if r.missing_from_run:
        print(f"  ! truth names {len(r.missing_from_run)} question(s) the run never produced: "
              f"{', '.join(r.missing_from_run)}")
        print("    extraction lost them — mark error below is over what survived")

    print(f"\n  false zeros              {len(r.false_zeros):>3}   {pct(r.false_zero_rate)}"
          f"   {', '.join(r.false_zeros) if r.false_zeros else ''}")
    print(f"  false credit             {len(r.false_credit):>3}          "
          f"   {', '.join(r.false_credit) if r.false_credit else ''}")
    print(f"  mark error per question  {num(r.mae)}")
    print(f"    where placement right  {num(r.mae_placed_right)}   "
          f"({len(r.placed_right)} question(s))")
    print(f"    where placement wrong  {num(r.mae_placed_wrong)}   "
          f"({len(r.placed_wrong)} question(s)) — {r.marks_lost_to_placement:g} marks")
    if r.hedged:
        print(f"  hedged on a real blank   {len(r.hedged):>3}   "
              f"{', '.join(r.hedged)}  (safe, not a placement error)")
    print(f"  within the marker's band {pct(r.within_band_rate)}")
    print(f"  script total             {r.awarded_total:g} / {r.available_total:g}"
          f"   truth {r.truth_total:g}   band {low:g}-{high:g}"
          f"   {'in band' if r.total_within_band else 'OUT OF BAND'}")

    print(f"\n  denominator              {'ok' if r.denominator_ok else 'MISMATCH'}", end="")
    if r.denominator_mismatch:
        print("   " + "; ".join(
            f"{q}: rubric {a:g} vs printed {b:g}" for q, a, b in r.denominator_mismatch
        ))
    else:
        print()
    print(f"  citation rate            {pct(r.citation_rate)}"
          f"   ({r.cited_points}/{r.awarded_points} mark-bearing points)")
    if r.incoherent_points:
        print(f"  ! {r.incoherent_points} point(s) claim satisfied without full marks")
    if r.unjudged:
        print(f"  ! answered but never judged: {', '.join(r.unjudged)}")
    if r.unmarked_reasons:
        print("  unmarked:")
        for reason, count in sorted(r.unmarked_reasons.items(), key=lambda kv: -kv[1]):
            print(f"      {count:>2}  {reason[:66]}")

    if verbose:
        by_qid = truth.by_qid()
        print(f"\n  {'qid':10} {'truth':>7} {'run':>7} {'err':>6} {'band':>8}  placement")
        for qid in r.scored:
            entry = by_qid[qid]
            lo, hi = entry.band
            placed = "right" if qid in r.placed_right else "WRONG"
            flag = "" if qid in r.within_band else "  <-"
            print(f"  {qid:10} {entry.marks_awarded:>7g} "
                  f"{r.awarded[qid]:>7g} {r.errors[qid]:>6.1f} "
                  f"{f'{lo:g}-{hi:g}':>8}  {placed}{flag}")
        for qid in r.excluded:
            print(f"  {qid:10} {'—':>7} {'—':>7} {'—':>6} {'—':>8}  excluded: "
                  f"{(by_qid[qid].excluded_reason or '')[:40]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="*", default=None,
                        help="Which documents to score. Default: all five with truth.")
    parser.add_argument("--engine", choices=["default", "text-layer"], default="default",
                        help="'text-layer' removes recognition from the measurement.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Mark each document N times and report per-question spread.")
    parser.add_argument("--verbose", action="store_true", help="Per-question table.")
    parser.add_argument("--pages", type=Path, default=ROOT / "packages" / "generated" / ".marks",
                        help="Where rendered pages are cached.")
    args = parser.parse_args(argv)

    documents = args.documents or DOCUMENTS
    grader_name = type(grading.select_grader()).__name__
    print(f"grader {grader_name} · engine {args.engine} · {args.repeat} pass(es)")
    if grader_name == "RubricOnly":
        print("! No grading key is configured, so every mark will be zero and every\n"
              "  figure below will be meaningless. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

    outcomes: dict[str, Outcome] = {}
    repeats: dict[str, list[dict[str, float]]] = {}

    for doc in documents:
        truth = marks_mod.find(doc, extra_roots=SOURCES)
        if truth is None:
            print(f"\n{doc}: no mark truth, skipped")
            continue

        for attempt in range(args.repeat):
            try:
                submission = run_document(doc, engine=args.engine, page_root=args.pages)
            except Exception as exc:  # noqa: BLE001 - one document must not stop the rest
                print(f"\n{doc}: failed to run ({type(exc).__name__}: {exc})")
                break

            facts = facts_from(submission)
            repeats.setdefault(doc, []).append(
                {qid: fact.marks_awarded for qid, fact in facts.items()}
            )
            if attempt == 0:
                outcomes[doc] = Outcome(
                    submission=submission,
                    report=metrics.scoring_scores(truth, facts),
                )

        if doc in outcomes:
            report_one(doc, truth, outcomes[doc], verbose=args.verbose)

    if not outcomes:
        print("\nNothing scored.")
        return 1

    print(f"\n{'═' * 76}\nACROSS {len(outcomes)} DOCUMENT(S)")
    reports = [o.report for o in outcomes.values()]
    errors = [e for r in reports for e in r.errors.values()]
    right = [r.errors[q] for r in reports for q in r.placed_right if q in r.errors]
    wrong = [r.errors[q] for r in reports for q in r.placed_wrong if q in r.errors]
    deserving = sum(1 for r in reports for m in r.truth_marks.values() if m > 0)
    false_zeros = sum(len(r.false_zeros) for r in reports)

    print(f"  questions scored         {len(errors)}")
    print(f"  FALSE ZERO RATE          {pct(false_zeros / deserving if deserving else None)}"
          f"   ({false_zeros} of {deserving} genuinely-earned answers scored zero)")
    print(f"  false credit             {sum(len(r.false_credit) for r in reports)}")
    print(f"  mark error per question  {num(metrics._mean(errors))}")
    print(f"    placement right        {num(metrics._mean(right))}   ({len(right)})")
    print(f"    placement wrong        {num(metrics._mean(wrong))}   ({len(wrong)})")
    print(f"  within band              "
          f"{pct(sum(len(r.within_band) for r in reports) / len(errors) if errors else None)}")
    print(f"  marks lost to placement  "
          f"{sum(r.marks_lost_to_placement for r in reports):g}"
          f" of {sum(sum(r.errors.values()) for r in reports):g} total error")
    print(f"  total marks              {sum(r.awarded_total for r in reports):g}"
          f" / {sum(r.available_total for r in reports):g}"
          f"   truth {sum(r.truth_total for r in reports):g}")
    print(f"  denominator mismatches   {sum(len(r.denominator_mismatch) for r in reports)}")
    cited = sum(r.cited_points for r in reports)
    total_points = sum(r.awarded_points for r in reports)
    print(f"  citation rate            {pct(cited / total_points if total_points else None)}")
    print(f"  hedged on a real blank   {sum(len(r.hedged) for r in reports)}"
          f"   (uncertain where unanswered was available)")
    print(f"  never judged             {sum(len(r.unjudged) for r in reports)}")
    print(f"  truth missing from runs  {sum(len(r.missing_from_run) for r in reports)}")

    if args.repeat > 1:
        print(f"\n  STABILITY over {args.repeat} passes")
        for doc, runs in repeats.items():
            spread = metrics.mark_stability(runs)
            moved = {q: s for q, s in spread.items() if s > 0}
            worst = max(spread.values()) if spread else 0.0
            print(f"    {doc:12} {len(moved)} question(s) moved, worst {worst:g} mark(s)"
                  + (f"   {', '.join(f'{q}:{s:g}' for q, s in moved.items())}" if moved else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
