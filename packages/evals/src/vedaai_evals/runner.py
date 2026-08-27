"""Run the pipeline over the golden set and report accuracy.

Invoked by ``pnpm turbo eval``.

The scoring path is written in full even though most of it cannot fire yet:
question extraction lands in Phase 4 and mapping in Phase 6. Each metric is
guarded on the pipeline output it needs and reports itself as pending otherwise.
That is deliberate — printing a page of zeros would be indistinguishable from
total failure, and wiring the metrics later would mean writing them against
whatever the pipeline happened to produce rather than against what accuracy means.

What is measured today is ingest: how many lines were transcribed, how much ink
was found, and how much of that ink the recognizer never accounted for. Those are
descriptive rather than scored, because scoring them needs ground-truth boxes on
real pages that nobody has drawn yet.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from grader import pipeline, regions, render
from grader.ocr import PdfTextLayerEngine
from grader.storage import PageStore
from grader.store import SubmissionStore
from vedaai_contracts import AnswerStatus, DocumentKind, PageBox, Submission

from . import metrics
from .generate import adopt_real_pages, generate_all
from .schema import GoldenSample, load_set

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "generated"


@dataclass
class IngestObservation:
    """What ingest produced for one sample. Descriptive, not scored."""

    sample_id: str
    origin: str
    question_lines: int
    answer_lines: int
    ink_regions: int
    orphan_ink: int
    excluded_from_grading: int
    warnings: list[str]


@dataclass
class SampleScore:
    """Scored metrics for one sample, where the pipeline supports them."""

    sample_id: str
    origin: str = "synthetic"
    extraction: metrics.ExtractionReport | None = None
    mapping: metrics.MappingReport | None = None
    ious: list[float] | None = None
    recall: metrics.RecallReport | None = None
    agreement: metrics.AgreementReport | None = None


def run_sample(
    sample: GoldenSample,
    directory: Path,
    *,
    page_root: Path,
) -> tuple[Submission, IngestObservation]:
    """Ingest one golden sample."""
    qp_bytes = (directory / sample.question_paper).read_bytes()
    as_bytes = (directory / sample.answer_sheet).read_bytes()

    pages = PageStore(root=page_root / sample.sample_id)
    submissions = SubmissionStore()
    submissions.put(Submission(submission_id=sample.sample_id))

    qp_source = render.inspect(qp_bytes, sample.question_paper, DocumentKind.QUESTION_PAPER)
    as_source = render.inspect(as_bytes, sample.answer_sheet, DocumentKind.ANSWER_SHEET)

    # Synthetic answer sheets are read from their text layer, which normal
    # engine selection refuses to do because a real scanned sheet's layer is
    # spurious. On a generated sheet it is exact, and that is precisely what these
    # samples are for: they measure mapping and highlighting, so transcription
    # should be perfect rather than merely good. Otherwise a mapping regression
    # and a recognition regression are indistinguishable in the report.
    #
    # Real samples get the real engine, since recognition is the thing they
    # exist to measure.
    override = PdfTextLayerEngine() if sample.origin == "synthetic" else None

    submission = pipeline.ingest(
        submission_id=sample.sample_id,
        question_paper=(qp_bytes, qp_source),
        answer_sheet=(as_bytes, as_source),
        page_store=pages,
        submission_store=submissions,
        answer_engine_override=override,
    )

    answer_lines = submission.answer_sheet_lines.lines if submission.answer_sheet_lines else []
    excluded = regions.lines_excluded_from_grading(submission.ink_regions, answer_lines)

    observation = IngestObservation(
        sample_id=sample.sample_id,
        origin=sample.origin,
        question_lines=(
            len(submission.question_paper_lines.lines)
            if submission.question_paper_lines
            else 0
        ),
        answer_lines=len(answer_lines),
        ink_regions=sum(1 for r in submission.ink_regions if r.is_substantive),
        orphan_ink=len(regions.orphan_ink(submission.ink_regions)),
        excluded_from_grading=len(excluded),
        warnings=list(submission.warnings),
    )
    return submission, observation


def score_sample(sample: GoldenSample, submission: Submission) -> SampleScore:
    """Score whatever the pipeline currently produces for this sample."""
    score = SampleScore(sample_id=sample.sample_id, origin=sample.origin)

    # Needs no labelling, so this is the one detection signal available on real
    # pages today. Only meaningful where transcription and ink were produced by
    # genuinely different mechanisms — on synthetic samples the text layer reads
    # everything perfectly, so agreement is trivially total and says nothing.
    substantive = [r for r in submission.ink_regions if r.is_substantive]
    if substantive and sample.origin != "synthetic":
        score.agreement = metrics.detector_agreement([r.covered_by_ocr for r in substantive])

    # Phase 4 onward.
    if submission.questions is not None and submission.questions.questions:
        predicted = [q.label_raw for q in submission.questions.in_print_order()]
        truth = [
            q.label_raw
            for q in sorted(sample.questions, key=lambda q: q.print_order)
        ]
        score.extraction = metrics.extraction_scores(truth, predicted)

    # Phase 6 onward.
    if submission.mapping is not None and submission.mapping.mappings:
        predicted_by_qid = submission.mapping.by_qid()
        cases: list[metrics.MappingCase] = []
        for answer in sample.answers:
            predicted = predicted_by_qid.get(answer.qid)
            predicted_boxes: list[PageBox] = []
            predicted_status = AnswerStatus.UNANSWERED
            if predicted is not None:
                predicted_status = predicted.status
                if predicted.highlight is not None:
                    predicted_boxes = list(predicted.highlight.boxes)
            cases.append(
                metrics.MappingCase(
                    qid=answer.qid,
                    truth_status=answer.status,
                    truth_boxes=list(answer.complete_answer_box),
                    predicted_status=predicted_status,
                    predicted_boxes=predicted_boxes,
                )
            )
        score.mapping, score.ious = metrics.mapping_scores(cases)

    # Needs a human-labelled real sample. Synthetic pages cannot measure this:
    # they are rendered from fonts, so recognition on them says nothing about
    # recognition on handwriting.
    if sample.has_line_truth and submission.answer_sheet_lines is not None:
        truth_lines = [(ln.page, ln.box, ln.text) for ln in sample.lines]
        predicted_lines = [
            (ln.page, ln.box) for ln in submission.answer_sheet_lines.lines
        ]
        score.recall = metrics.line_recall(truth_lines, predicted_lines)

    return score


def validate_truth(sample: GoldenSample) -> list[str]:
    """Self-check the ground truth before trusting any number derived from it.

    A golden set is a measuring instrument, and an uncalibrated instrument is
    worse than none: it produces numbers that look authoritative. These checks
    catch the mistakes that would otherwise show up as inexplicable scores.
    """
    problems: list[str] = []
    qids = {q.qid for q in sample.questions}

    for answer in sample.answers:
        if answer.qid not in qids:
            problems.append(f"answer for unknown question {answer.qid!r}")
        if answer.status is AnswerStatus.ANSWERED and not answer.complete_answer_box:
            problems.append(f"{answer.qid} is marked answered but has no box")
        if answer.status is not AnswerStatus.ANSWERED and answer.complete_answer_box:
            problems.append(f"{answer.qid} is not answered yet carries a box")

    orders = [q.print_order for q in sample.questions]
    if len(set(orders)) != len(orders):
        problems.append("print_order values are not unique")

    labels = [q.label_raw for q in sample.questions]
    if len(set(labels)) != len(labels):
        problems.append("question labels are not unique")

    return problems


def _fmt_pct(value: float | None) -> str:
    return "  n/a " if value is None else f"{value * 100:5.1f}%"


def report(
    observations: list[IngestObservation],
    scores: list[SampleScore],
    samples: list[GoldenSample],
) -> int:
    """Print the report. Returns a process exit code."""
    out = sys.stdout.write
    synthetic = sum(1 for s in samples if s.origin == "synthetic")
    real = sum(1 for s in samples if s.origin == "real")
    labelled = sum(1 for s in samples if s.has_line_truth)

    out("\n=== GOLDEN SET ===\n")
    out(f"  synthetic cases      {synthetic}\n")
    out(f"  real samples         {real}\n")
    out(f"  with line-level truth {labelled}\n")

    problems = [(s.sample_id, p) for s in samples for p in validate_truth(s)]
    if problems:
        out("\n  TRUTH PROBLEMS (metrics from these are not trustworthy)\n")
        for sample_id, problem in problems:
            out(f"    {sample_id}: {problem}\n")

    out("\n=== INGEST (descriptive) ===\n")
    out(f"  {'case':<18} {'q-lines':>8} {'a-lines':>8} {'ink':>6} {'orphan':>7} {'excl':>6}\n")
    for o in observations:
        out(
            f"  {o.sample_id:<18} {o.question_lines:>8} {o.answer_lines:>8} "
            f"{o.ink_regions:>6} {o.orphan_ink:>7} {o.excluded_from_grading:>6}\n"
        )

    warned = [o for o in observations if o.warnings]
    if warned:
        out("\n  warnings\n")
        for o in warned:
            for warning in o.warnings:
                out(f"    {o.sample_id}: {warning}\n")

    out("\n=== SCORED ===\n")

    extractions = [s.extraction for s in scores if s.extraction is not None]
    if extractions:
        f1 = sum(e.f1 for e in extractions) / len(extractions)
        taus = [e.order_tau for e in extractions if e.order_tau is not None]
        out(f"  question extraction F1   {_fmt_pct(f1)}\n")
        out(
            "  reading order (Kendall)  "
            + (f"{sum(taus) / len(taus):+.3f}\n" if taus else "  n/a\n")
        )
        missed = [label for e in extractions for label in e.missed]
        if missed:
            out(f"    missed labels: {', '.join(sorted(set(missed))[:12])}\n")
    else:
        out("  question extraction      pending — Phase 4\n")

    mappings = [s.mapping for s in scores if s.mapping is not None]
    if mappings:
        total_correct = sum(m.correct + m.correctly_unanswered for m in mappings)
        total_scored = sum(m.scored for m in mappings)
        all_missed = [q for m in mappings for q in m.missed]
        answered = sum(
            m.correct + len(m.wrong_region) + len(m.missed) for m in mappings
        )
        out(
            f"  mapping accuracy         "
            f"{_fmt_pct(total_correct / total_scored if total_scored else None)}\n"
        )
        out(
            "  FALSE UNANSWERED         "
            + f"{_fmt_pct(len(all_missed) / answered if answered else 0.0)}"
            + "   <- the error a teacher acts on unchecked\n"
        )
        violated = [q for m in mappings for q in m.not_required_violated]
        if violated:
            out(f"    optional questions wrongly reported missing: {len(violated)}\n")

        ious = [i for s in scores if s.ious for i in s.ious]
        summary = metrics.summarize_iou(ious)
        out(
            f"  highlight IoU            mean {summary['mean']:.3f}  "
            f"@0.5 {summary['at_50'] * 100:.0f}%  @0.75 {summary['at_75'] * 100:.0f}%\n"
        )
    else:
        out("  answer mapping           pending — Phase 6\n")
        out("  highlight IoU            pending — Phase 6\n")

    agreements = [s.agreement for s in scores if s.agreement is not None]
    if agreements:
        ink = sum(a.ink_regions for a in agreements)
        uncovered = sum(a.uncovered for a in agreements)
        out(
            f"  ink accounted for by text {_fmt_pct((ink - uncovered) / ink if ink else None)}"
            f"   ({uncovered} of {ink} regions untranscribed)\n"
        )
        out(
            "    proxy for recall, not ground truth — overstates it, since ink\n"
            "    misses very faint writing too. Use as a regression signal.\n"
        )

    recalls = [s.recall for s in scores if s.recall is not None]
    if recalls:
        matched = sum(r.matched for r in recalls)
        total = sum(r.total_truth for r in recalls)
        out(
            f"  OCR line recall          "
            f"{_fmt_pct(matched / total if total else None)}   <- the binding ceiling\n"
        )
        lost = [t for r in recalls for t in r.missed_text]
        if lost:
            out(f"    missed lines: {len(lost)}\n")
            for text in lost[:5]:
                out(f"      {text[:66]!r}\n")
    else:
        out(
            "  OCR line recall          unavailable — needs ground-truth boxes on real\n"
            "                           pages. The ink-agreement figure above is the\n"
            "                           closest substitute that requires no labelling.\n"
        )

    out("\n")
    return 1 if problems else 0


#: Records which version of the generator produced the fixtures on disk.
_STAMP = ".generator"


def _generator_fingerprint() -> str:
    """A hash of the code that builds the synthetic cases."""
    import hashlib

    from . import generate as generate_module
    from . import schema as schema_module

    digest = hashlib.sha256()
    for module in (generate_module, schema_module):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def _generator_changed(root: Path) -> bool:
    """Whether the fixtures on disk were built by a different generator.

    Regenerating on a source change rather than only when asked, because the
    alternative silently reports the wrong number. These fixtures exist to answer
    "did that change help", and scoring a new algorithm against a paper generated
    before it — with a structure it was written to handle absent from the set —
    produces a figure that looks like a result and is not one. Observed exactly
    that way: a question shape was added to the paper and the reported accuracy
    did not move, because nothing had been rebuilt.

    Fingerprinted rather than compared by timestamp, so a checkout or a copy
    cannot make stale fixtures look current.
    """
    if not root.is_dir():
        return True
    stamp = root / _STAMP
    if not stamp.is_file():
        return True
    return stamp.read_text().strip() != _generator_fingerprint()


def _record_generator(root: Path) -> None:
    (root / _STAMP).write_text(_generator_fingerprint() + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the grader against the golden set.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Rebuild the synthetic cases before running.",
    )
    parser.add_argument(
        "--real",
        type=Path,
        default=None,
        help="Directory of prepared real samples to include.",
    )
    parser.add_argument(
        "--adopt-real",
        type=Path,
        default=None,
        help=(
            "Directory of raw handwritten images to register as unlabelled samples. "
            "Scores detection only, since answer-level truth is unknown."
        ),
    )
    parser.add_argument("--only", default=None, help="Run one case by id.")
    args = parser.parse_args(argv)

    synthetic_root = args.root / "synthetic"
    if args.regenerate or _generator_changed(synthetic_root):
        synthetic_root.mkdir(parents=True, exist_ok=True)
        generate_all(synthetic_root)
        _record_generator(synthetic_root)

    samples: list[tuple[GoldenSample, Path]] = [
        (s, synthetic_root / s.sample_id) for s in load_set(synthetic_root)
    ]

    if args.adopt_real is not None:
        real_root = args.root / "real"
        images = sorted(
            p
            for p in args.adopt_real.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        adopted = adopt_real_pages(images, real_root)
        samples.extend((s, real_root / s.sample_id) for s in adopted)
    if args.real is not None:
        samples.extend((s, args.real / s.sample_id) for s in load_set(args.real))

    if args.only:
        samples = [(s, d) for s, d in samples if s.sample_id == args.only]
    if not samples:
        sys.stderr.write("no samples found\n")
        return 1

    page_root = args.root / ".pages"
    observations: list[IngestObservation] = []
    scores: list[SampleScore] = []

    for sample, directory in samples:
        submission, observation = run_sample(sample, directory, page_root=page_root)
        observations.append(observation)
        scores.append(score_sample(sample, submission))

    return report(observations, scores, [s for s, _ in samples])


if __name__ == "__main__":
    raise SystemExit(main())
