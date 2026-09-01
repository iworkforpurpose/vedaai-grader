"""How the marker behaves on questions it has never seen.

Every scoring figure this project has published was measured on the six documents
the fixes were built against, so all of them are training numbers. This is the
held-out instrument.

**SciEntsBank** (SemEval-2013 Task 7, CC-BY-4.0). Real school science questions,
each with a **human-written reference answer** and a **human label** on the
student's response, drawn from five classes:

    correct · partially_correct_incomplete · contradictory · irrelevant · non_domain

Four things make it the right test rather than merely another dataset.

**The labels name our two failure directions separately.** Crediting a
``contradictory`` answer is exactly the fault that started this work — a fluent,
on-topic, wrong answer earning full marks. Under-crediting ``correct`` is the
opposite drift the fix introduced. A single accuracy number would average them;
here they are two columns.

**The questions are unseen.** ``test_uq`` holds questions absent from training and
``test_ud`` whole domains, which is the closest available proxy for a teacher
uploading a paper nobody has tuned against.

**It carries a real reference answer**, so the experiment that matters can be run
directly: derive the checks from the question alone, as the product does today,
and again from the teacher's reference. A model writing its own reference guessed
confidently and tripled the false-zero rate; a reference a person wrote should
not. If the second arm is much better, the product should ask teachers for a mark
scheme, and that is a finding about the product rather than a tuning result.

**Published baselines exist**, so the number stops being self-referential.

    uv run python tooling/scripts/score_scientsbank.py --n 60
    uv run python tooling/scripts/score_scientsbank.py --n 60 --arm reference
    uv run python tooling/scripts/score_scientsbank.py --split test_ud --n 40

Costs two model calls per item, and check banks are cached per question.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))

from grader.grading import rubric as rubric_mod  # noqa: E402
from grader.grading import scheme as scheme_mod  # noqa: E402
from grader.grading import select_grader  # noqa: E402
from vedaai_contracts import (  # noqa: E402
    BBox,
    DocumentKind,
    Line,
    LineIndex,
    OcrEngine,
    Question,
)

CACHE = ROOT / "packages" / "generated" / ".scientsbank"

#: Marks given to every question, since SciEntsBank prints none.
#:
#: Two, so that partial credit is expressible as one. A denominator of one would
#: force every judgement to be all-or-nothing and would flatter the binary checks
#: by removing the case they are least sure of.
MARKS = 2.0

#: The ClassLabel order the dataset declares. Rows carry the *index*, not the
#: name, and the first run silently sampled nothing because the code compared
#: "0" against "correct" — a mismatch that produces an empty report rather than an
#: error, which is the worst way for a harness to be wrong.
LABEL_NAMES = [
    "correct",
    "contradictory",
    "partially_correct_incomplete",
    "irrelevant",
    "non_domain",
]


def label_of(row: dict) -> str:
    raw = row.get("label")
    if isinstance(raw, int) and 0 <= raw < len(LABEL_NAMES):
        return LABEL_NAMES[raw]
    return str(raw)


#: What each human label implies the award should be.
#:
#: ``partially_correct_incomplete`` deliberately has no single right answer — any
#: award strictly between zero and full is defensible, and that is what is scored.
EXPECTED = {
    "correct": "full",
    "partially_correct_incomplete": "partial",
    "contradictory": "zero",
    "irrelevant": "zero",
    "non_domain": "zero",
}


@dataclass
class Outcome:
    label: str
    awarded: float
    available: float
    judged: bool
    deferred: int = 0
    checks: int = 0

    @property
    def band(self) -> str:
        if self.awarded <= 0:
            return "zero"
        if self.awarded >= self.available:
            return "full"
        return "partial"

    @property
    def credited(self) -> bool:
        """Whether any mark at all was given. The false-credit test."""
        return self.awarded > 0


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)

    def by_label(self, label: str) -> list[Outcome]:
        return [o for o in self.outcomes if o.label == label]

    @property
    def wrong_answers(self) -> list[Outcome]:
        return [o for o in self.outcomes if EXPECTED.get(o.label) == "zero"]

    @property
    def false_credit_rate(self) -> float | None:
        """Share of answers a human called wrong that were given marks anyway.

        The headline. This is the failure that destroys a teacher's trust, because
        a mark awarded to a wrong answer is one they get challenged on.
        """
        wrong = self.wrong_answers
        if not wrong:
            return None
        return sum(1 for o in wrong if o.credited) / len(wrong)

    @property
    def full_credit_on_correct(self) -> float | None:
        """Share of human-correct answers given full marks."""
        right = self.by_label("correct")
        if not right:
            return None
        return sum(1 for o in right if o.band == "full") / len(right)

    @property
    def any_credit_on_correct(self) -> float | None:
        """Share of human-correct answers given at least something.

        Reported beside the full-credit figure because they answer different
        questions: a correct answer given half marks is a mild drift, and one given
        nothing is a false zero.
        """
        right = self.by_label("correct")
        if not right:
            return None
        return sum(1 for o in right if o.credited) / len(right)

    @property
    def agreement(self) -> float | None:
        """Share of items whose award band matches what the label implies.

        Partial is counted as agreement for ``partially_correct_incomplete``, since
        any award between the extremes is defensible there.
        """
        scored = [o for o in self.outcomes if o.label in EXPECTED]
        if not scored:
            return None
        return sum(1 for o in scored if o.band == EXPECTED[o.label]) / len(scored)

    @property
    def binary_agreement(self) -> float | None:
        """Agreement after collapsing to correct-or-not, which is the shape the
        literature reports and the shape the checks are built for."""
        scored = [o for o in self.outcomes if o.label in EXPECTED]
        if not scored:
            return None
        hits = 0
        for o in scored:
            human_correct = o.label == "correct"
            ours_correct = o.band == "full"
            hits += human_correct == ours_correct
        return hits / len(scored)


def load(split: str) -> list[dict]:
    path = CACHE / f"{split}.json"
    if not path.is_file():
        raise SystemExit(
            f"no cached {split}. Fetch it first — see the note at the top of this file."
        )
    return json.loads(path.read_text())


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """A stratified sample that deliberately over-weights the wrong answers.

    The classes are very unbalanced toward ``correct``, and a proportional sample
    would spend most of its budget confirming that a right answer gets marks. The
    figure that matters is what happens to the wrong ones, so they get the room.
    """
    wanted = {
        "correct": 0.34,
        "partially_correct_incomplete": 0.22,
        "contradictory": 0.22,
        "irrelevant": 0.16,
        "non_domain": 0.06,
    }
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[label_of(row)].append(row)

    rng = random.Random(seed)
    out: list[dict] = []
    for label, share in wanted.items():
        pool = buckets.get(label, [])
        rng.shuffle(pool)
        out.extend(pool[: max(1, round(n * share))])
    rng.shuffle(out)
    return out[:n]


def as_index(text: str) -> tuple[LineIndex, list[str]]:
    """The student's answer as a line index, so citations resolve.

    Split on sentences rather than handed over as one line: a single line would let
    every check cite the same id and would not exercise the citation validation at
    all.
    """
    parts = [p.strip() for p in text.replace("\n", " ").split(". ") if p.strip()] or [text]
    lines: list[Line] = []
    for i, part in enumerate(parts, start=1):
        top = min(0.05 + i * 0.04, 0.90)
        lines.append(
            Line(
                line_id=f"as:{i:04d}",
                kind=DocumentKind.ANSWER_SHEET,
                page=0,
                box=BBox(x0=0.1, y0=top, x1=0.9, y1=top + 0.03),
                text=part,
                confidence=1.0,
                engine=OcrEngine.PDF_TEXT_LAYER,
            )
        )
    index = LineIndex(
        kind=DocumentKind.ANSWER_SHEET, lines=lines, engine=OcrEngine.PDF_TEXT_LAYER
    )
    return index, [ln.line_id for ln in lines]


async def run_one(row: dict, *, arm: str, grader) -> Outcome | None:
    label = label_of(row)
    question = Question(
        qid=str(row.get("id", "q")),
        label_raw="1.",
        text=str(row.get("question", "")).strip(),
        path=["1"],
        print_order=0,
        marks=MARKS,
    )
    spec = rubric_mod.derive(question)
    index, line_ids = as_index(str(row.get("student_answer", "")))
    reference = str(row.get("reference_answer", "")) if arm == "reference" else ""

    try:
        bank = await scheme_mod.derive(question, spec, reference=reference)
    except Exception:  # noqa: BLE001
        bank = None

    try:
        grade = await grader.grade(
            question=question, rubric=spec, index=index, line_ids=line_ids, scheme=bank
        )
    except Exception:  # noqa: BLE001 - one item must not stop the run
        return None

    deferred = sum(
        1
        for p in grade.rubric_points
        if p.marks_awarded == 0 and "needs your eye" in (p.comment or "")
    )
    return Outcome(
        label=label,
        awarded=float(grade.marks_awarded),
        available=float(grade.marks_available),
        judged=bool(grade.judged),
        deferred=deferred,
        checks=len(bank.checks) if bank else 0,
    )


def pct(v: float | None) -> str:
    return "   n/a" if v is None else f"{v * 100:5.1f}%"


async def main_async(args) -> int:
    rows = sample(load(args.split), args.n, args.seed)
    grader = select_grader()
    print(
        f"{type(grader).__name__} · split {args.split} · arm {args.arm} · "
        f"{len(rows)} items · {MARKS:g} marks each"
    )
    if type(grader).__name__ == "RubricOnly":
        print("! no grading key configured — every award will be zero and the figures")
        print("  below will be meaningless. Set OPENAI_API_KEY.")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(row):
        async with semaphore:
            return await run_one(row, arm=args.arm, grader=grader)

    results = await asyncio.gather(*(guarded(r) for r in rows))
    report = Report(outcomes=[o for o in results if o is not None])
    failed = len(results) - len(report.outcomes)

    print(f"\n{'═' * 72}\nHELD-OUT RESULT · {len(report.outcomes)} items scored"
          + (f" · {failed} failed" if failed else ""))
    print(f"\n  FALSE CREDIT RATE        {pct(report.false_credit_rate)}"
          f"   marks given to answers a human called wrong")
    print(f"  full credit on correct   {pct(report.full_credit_on_correct)}")
    print(f"  any credit on correct    {pct(report.any_credit_on_correct)}"
          f"   (1 - this = false-zero rate)")
    print(f"  band agreement           {pct(report.agreement)}")
    print(f"  binary agreement         {pct(report.binary_agreement)}"
          f"   correct-or-not, the shape the literature reports")

    print(f"\n  {'human label':32}{'n':>4}{'zero':>7}{'partial':>9}{'full':>6}   want")
    for label in EXPECTED:
        got = report.by_label(label)
        if not got:
            continue
        bands = Counter(o.band for o in got)
        print(
            f"  {label:32}{len(got):>4}{bands['zero']:>7}{bands['partial']:>9}"
            f"{bands['full']:>6}   {EXPECTED[label]}"
        )

    deferred = sum(o.deferred for o in report.outcomes)
    unjudged = sum(1 for o in report.outcomes if not o.judged)
    checks = [o.checks for o in report.outcomes if o.checks]
    print(f"\n  checks per question      {sum(checks) / len(checks):.1f}" if checks else "")
    print(f"  marks deferred to a teacher   {deferred}")
    print(f"  never judged                  {unjudged}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test_uq", choices=["test_uq", "test_ud"])
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--arm", default="question", choices=["question", "reference"],
                        help="'question' derives checks from the question alone, as the "
                             "product does. 'reference' uses the teacher's answer too.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--concurrency", type=int, default=6)
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
