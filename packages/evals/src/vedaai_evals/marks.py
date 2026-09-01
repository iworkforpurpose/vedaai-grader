"""Mark-level ground truth: what a question *should* have been awarded.

This is the piece the harness never had. ``GoldenAnswer`` can say where an answer
is and whether it was answered at all, and every published figure measures one of
those two things — extraction, placement, highlight IoU, label binding, line
recall. None of them says whether the marks are right, so "scoring accuracy" has
never had a number attached to it.

Three commitments, and each one exists because of how marking actually behaves.

**A mark is a band, not a point.** Two competent teachers disagree by a mark on a
four-mark answer, and a harness that demands an exact match reports that
disagreement as a failure. The existing hand-written key for the mathematics
script already says so in as many words -- "treat 14-17 as agreement; a grader is
entitled to argue T1 down to 2 or T2 up to 3" -- so the band is what truth records
and the midpoint is what error is measured against.

**Some questions are deliberately outside scoring truth.** A question whose
correct answer depends on a figure that was never drawn cannot be marked by
anybody, and inventing a number for it would put my guess into the denominator of
every figure downstream. ``marks_awarded=None`` with an ``excluded_reason`` is a
first-class state, and the report counts exclusions rather than hiding them.

**The key is written down beside the mark.** A number with no statement of what
the correct answer was is not checkable, which makes it indistinguishable from a
number rationalised after a run. ``key`` carries the answer the mark was decided
against.

**On where these files live.** ``data/`` is untracked in this repository, on the
stated ground that it holds real student work. So truth for the four generated
papers ships here, tracked, because those scripts were authored by
``tooling/scripts/build_fresh_papers.py`` and contain nobody's handwriting. Truth
for a real script stays beside the script, untracked, and is loaded from there.
The consequence worth knowing: the real-script truth is not in version control and
is not backed up by this repository.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from vedaai_contracts import AnswerStatus

#: Where truth for the generated papers ships. Tracked, beside the harness.
#:
#: ``parents[2]`` is the package root: this file sits at
#: ``packages/evals/src/vedaai_evals/marks.py``, so 0 is the module directory,
#: 1 is ``src`` and 2 is ``packages/evals``.
PACKAGED_MARKS = Path(__file__).resolve().parents[2] / "marks"


class MarkTruth(BaseModel):
    """What one question should have been awarded, and why.

    Frozen, because a truth object that downstream code can adjust is not truth.
    """

    model_config = ConfigDict(frozen=True)

    qid: str = Field(
        description="Canonical qid as the extractor builds it. A truth entry naming "
        "a qid the run does not contain is reported rather than skipped -- it is "
        "usually an extraction failure, which is worth hearing about here."
    )
    label_raw: str = Field(default="", description="As printed, for reading the report.")
    marks_available: float = Field(ge=0, description="The denominator the paper printed.")

    status: AnswerStatus = Field(
        description="What the student actually did. Carried here as well as in "
        "GoldenAnswer so that scoring can be partitioned by whether placement was "
        "right, which is the only way to tell a marking error from an aligner error "
        "without line-level truth."
    )

    marks_awarded: float | None = Field(
        default=None,
        ge=0,
        description="What a teacher would award. None means this question is "
        "deliberately outside scoring truth -- see excluded_reason.",
    )
    band_low: float | None = Field(default=None, ge=0)
    band_high: float | None = Field(default=None, ge=0)

    key: str = Field(
        default="",
        description="The correct answer, decided before any run. A mark without one "
        "is not checkable.",
    )
    note: str = Field(default="", description="Why this mark and not one either side.")
    excluded_reason: str | None = Field(
        default=None,
        description="Why this question carries no mark truth. Required when "
        "marks_awarded is None, so that an exclusion is always a decision and never "
        "an omission.",
    )

    @model_validator(mode="after")
    def _coherent(self) -> MarkTruth:
        if self.marks_awarded is None:
            if not self.excluded_reason:
                raise ValueError(
                    f"{self.qid}: no marks_awarded and no excluded_reason. An "
                    "unlabelled question has to say why, or it is indistinguishable "
                    "from one somebody forgot."
                )
            return self

        if self.excluded_reason:
            raise ValueError(f"{self.qid}: has both a mark and an excluded_reason")
        if self.marks_awarded > self.marks_available:
            raise ValueError(
                f"{self.qid}: awarded {self.marks_awarded} of "
                f"{self.marks_available} available"
            )
        low, high = self.band
        if not (low <= self.marks_awarded <= high):
            raise ValueError(
                f"{self.qid}: awarded {self.marks_awarded} sits outside its own "
                f"band {low}-{high}"
            )
        if high > self.marks_available:
            raise ValueError(f"{self.qid}: band reaches {high}, above the available marks")
        # A question the student did not attempt cannot carry marks, and a question
        # they did attempt correctly cannot be worth nothing by definition -- but
        # the second is a real possibility (a wrong answer), so only the first is
        # an error.
        if self.status is AnswerStatus.UNANSWERED and self.marks_awarded > 0:
            raise ValueError(f"{self.qid}: unanswered but awarded {self.marks_awarded}")
        return self

    @property
    def is_scored(self) -> bool:
        """Whether this question contributes to any scoring figure."""
        return self.marks_awarded is not None

    @property
    def band(self) -> tuple[float, float]:
        """The range a competent marker could defend.

        Defaults to the exact mark. That is the right default rather than a
        tolerance guessed on the question's behalf: a band is a claim about how
        much room the answer leaves, and it should be stated where it exists.
        """
        awarded = self.marks_awarded or 0.0
        low = self.band_low if self.band_low is not None else awarded
        high = self.band_high if self.band_high is not None else awarded
        return low, high


class MarkSet(BaseModel):
    """Mark truth for one document."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    source: str = Field(
        description="How this truth was arrived at. 'authored' means the script was "
        "generated from a known answer, so correctness is decided by construction. "
        "'read' means somebody read the pages and marked them, which is judgement "
        "and carries a band."
    )
    notes: str = ""
    questions: list[MarkTruth] = Field(default_factory=list)

    #: The whole script's defensible range, where it differs from the sum of the
    #: per-question bands.
    #:
    #: Stated separately because it does differ. Summing the per-question bands on
    #: the mathematics script gives 14-16, while the hand-written key says 14-17 --
    #: the totals band allows for one more mark of drift somewhere unspecified,
    #: which is a real statement about the script and not an arithmetic slip.
    total_low: float | None = None
    total_high: float | None = None

    @property
    def scored(self) -> list[MarkTruth]:
        return [q for q in self.questions if q.is_scored]

    @property
    def excluded(self) -> list[MarkTruth]:
        return [q for q in self.questions if not q.is_scored]

    @property
    def truth_total(self) -> float:
        """Marks the student earned, over the questions truth covers."""
        return sum(q.marks_awarded or 0.0 for q in self.scored)

    @property
    def available_total(self) -> float:
        """Marks on offer, over the questions truth covers.

        Deliberately not the paper's printed maximum. A question excluded from
        scoring truth must leave the denominator as well as the numerator, or the
        script's percentage silently penalises it for something nobody measured.
        """
        return sum(q.marks_available for q in self.scored)

    @property
    def total_band(self) -> tuple[float, float]:
        if self.total_low is not None and self.total_high is not None:
            return self.total_low, self.total_high
        lows, highs = zip(*(q.band for q in self.scored), strict=False) or ((), ())
        return (sum(lows), sum(highs)) if lows else (0.0, 0.0)

    def by_qid(self) -> dict[str, MarkTruth]:
        return {q.qid: q for q in self.questions}


def load(path: Path) -> MarkSet:
    """Read one truth file.

    Accepts either a dedicated marks file or a corpus ``truth.json`` that has
    grown mark fields, so the real script's truth does not have to be duplicated
    into a second file that can then disagree with the first.
    """
    raw = json.loads(Path(path).read_text())

    if "questions" in raw and raw.get("questions") and "marks_available" in (
        raw["questions"][0] or {}
    ):
        return MarkSet.model_validate(raw)

    return MarkSet.model_validate(_from_corpus_truth(raw))


def _from_corpus_truth(raw: dict) -> dict:
    """Fold a corpus ``truth.json`` into a ``MarkSet``.

    A corpus file keeps questions and answers in separate lists -- questions carry
    the denominator, answers carry the status -- because that is the shape the
    mapping metrics wanted. Marks are read from whichever list states them, so a
    file can grow ``marks_awarded`` on either side without this caring which.
    """
    answers = {a["qid"]: a for a in raw.get("answers", [])}
    questions: list[dict] = []

    for question in raw.get("questions", []):
        qid = question["qid"]
        answer = answers.get(qid, {})
        marked = {**question, **answer}
        entry = {
            "qid": qid,
            "label_raw": question.get("label_raw", ""),
            "marks_available": float(question.get("marks") or 0.0),
            "status": answer.get("status", AnswerStatus.UNANSWERED.value),
            "key": marked.get("key", ""),
            "note": marked.get("note", ""),
        }
        if marked.get("marks_awarded") is None:
            entry["excluded_reason"] = marked.get(
                "excluded_reason", "no mark truth recorded for this question"
            )
        else:
            entry["marks_awarded"] = float(marked["marks_awarded"])
            for key in ("band_low", "band_high"):
                if marked.get(key) is not None:
                    entry[key] = float(marked[key])
        questions.append(entry)

    return {
        "doc_id": raw.get("sample_id", "unknown"),
        "source": "read",
        "notes": raw.get("notes", ""),
        "questions": questions,
        "total_low": raw.get("total_low"),
        "total_high": raw.get("total_high"),
    }


def find(doc_id: str, *, extra_roots: list[Path] | None = None) -> MarkSet | None:
    """Locate truth for a document, packaged first then beside the document.

    Packaged first so a generated paper's truth cannot be shadowed by a stale copy
    someone left in ``data/``.
    """
    candidates = [PACKAGED_MARKS / f"{doc_id}.json"]
    for root in extra_roots or []:
        candidates.extend([root / doc_id / "truth.json", root / f"{doc_id}.json"])

    for candidate in candidates:
        if candidate.is_file():
            marks = load(candidate)
            if marks.scored:
                return marks
    return None
