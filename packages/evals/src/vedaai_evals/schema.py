"""Ground-truth format for the golden set.

Two tiers, and they measure different things. Keeping them in one schema with
optional fields lets the runner report whatever a sample can support rather than
demanding everything from every sample.

**Synthetic samples** carry complete truth, because the generator drew the page
and therefore knows where every answer is. They measure mapping and highlighting,
which are geometric and structural — handwriting realism is irrelevant to whether
an answer was assigned to the right question.

**Real samples** carry whatever a human has labelled. They measure the things
synthetic cannot fake: OCR line recall and character error rate on actual
handwriting, plus paper texture, bleed-through and camera distortion.

The annotation format follows HG-Bench (arXiv 2606.25491) exactly — boxes on a
0-1000 integer grid, one ``complete_answer_box`` per question and an optional
ordered list of ``step_box`` entries with one-indexed ``step_id``, every step box
contained within its parent. Adopting a published format costs nothing now and
means our numbers are directly comparable to its baselines when the benchmark is
released, instead of requiring a conversion nobody will want to write later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from vedaai_contracts import AnswerStatus, BBox, PageBox


class GoldenStep(BaseModel):
    """One step of a multi-step answer, in writing order."""

    model_config = ConfigDict(frozen=True)

    step_id: int = Field(ge=1, description="One-indexed, per HG-Bench.")
    box: PageBox


class GoldenAnswer(BaseModel):
    """Ground truth for one question's answer."""

    qid: str
    status: AnswerStatus

    complete_answer_box: list[PageBox] = Field(
        default_factory=list,
        description="Where the answer is, as one box per page. A list rather than one "
        "box, because an answer spanning a page boundary has no single meaningful "
        "rectangle. This is HG-Bench's shape, kept so its baselines stay comparable.",
    )
    written_lines: list[PageBox] = Field(
        default_factory=list,
        description="The individual lines the answer occupies, before they are "
        "collapsed into a region. This is what the product actually has to highlight, "
        "and scoring against the region instead rewards a highlight for covering the "
        "blank paper between the lines.",
    )
    steps: list[GoldenStep] = Field(default_factory=list)
    text: str | None = Field(
        default=None,
        description="What the answer says, where a human transcribed it. Used for "
        "character error rate; absent on synthetic samples where it is trivially known.",
    )

    @model_validator(mode="after")
    def _steps_within_parent(self) -> GoldenAnswer:
        # HG-Bench's hierarchical constraint. Enforced here rather than trusted,
        # because a step box outside its parent means the annotation tool or the
        # importer has a coordinate bug, and that is far cheaper to catch at load
        # time than to discover as an inexplicable IoU score.
        if not self.complete_answer_box or not self.steps:
            return self
        parents = {pb.page: pb.box for pb in self.complete_answer_box}
        for step in self.steps:
            parent = parents.get(step.box.page)
            if parent is None:
                raise ValueError(
                    f"{self.qid} step {step.step_id} is on page {step.box.page}, "
                    "which the answer does not cover"
                )
            if not parent.contains(step.box.box, tolerance=1e-3):
                raise ValueError(
                    f"{self.qid} step {step.step_id} escapes its answer box"
                )
        return self

    @property
    def is_answered(self) -> bool:
        return self.status is AnswerStatus.ANSWERED


class GoldenQuestion(BaseModel):
    """Ground truth for one extracted question."""

    model_config = ConfigDict(frozen=True)

    qid: str
    label_raw: str = Field(description="Exactly as printed. Extraction is scored on this.")
    print_order: int = Field(ge=0)
    marks: int | None = None


class GoldenLine(BaseModel):
    """Ground truth for one line of handwriting on the answer sheet.

    Only these carry OCR recall and CER. They are also the expensive part to
    produce, which is why they are optional: a sample without them still scores
    mapping and highlighting.
    """

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=0)
    box: BBox
    text: str


class GoldenSample(BaseModel):
    """One question paper plus one answer sheet, with whatever truth is known."""

    sample_id: str
    origin: str = Field(
        description="'synthetic' or 'real'. Reported alongside every metric, because "
        "a number from synthetic data means something different from the same number "
        "on a real script and averaging them together would hide that."
    )

    question_paper: str = Field(description="Filename within the sample directory.")
    answer_sheet: str

    questions: list[GoldenQuestion] = Field(default_factory=list)
    answers: list[GoldenAnswer] = Field(default_factory=list)
    lines: list[GoldenLine] = Field(default_factory=list)

    notes: str | None = Field(
        default=None,
        description="What this sample is testing, for a human reading a failure report.",
    )

    @property
    def has_line_truth(self) -> bool:
        return bool(self.lines)

    @property
    def answered_qids(self) -> set[str]:
        return {a.qid for a in self.answers if a.is_answered}

    @property
    def unanswered_qids(self) -> set[str]:
        return {a.qid for a in self.answers if a.status is AnswerStatus.UNANSWERED}

    # -- HG-Bench interop --------------------------------------------------

    def to_hgbench(self) -> list[dict[str, Any]]:
        """Emit answers in HG-Bench's published annotation format."""
        out: list[dict[str, Any]] = []
        for answer in self.answers:
            if not answer.is_answered:
                continue
            entry: dict[str, Any] = {
                "question_id": answer.qid,
                "complete_answer_box": [pb.to_hgbench() for pb in answer.complete_answer_box],
            }
            if answer.steps:
                entry["step_boxes"] = [
                    {"step_id": s.step_id, **s.box.to_hgbench()} for s in answer.steps
                ]
            out.append(entry)
        return out

    @staticmethod
    def answers_from_hgbench(entries: list[dict[str, Any]]) -> list[GoldenAnswer]:
        """Read answers from HG-Bench's format.

        The path that matters once the benchmark is released: its 500 annotated
        samples become usable without touching anything downstream.
        """
        answers: list[GoldenAnswer] = []
        for entry in entries:
            boxes = [
                PageBox(page=int(b["page"]), box=BBox.from_hgbench(b["box"]))
                for b in entry.get("complete_answer_box", [])
            ]
            steps = [
                GoldenStep(
                    step_id=int(s["step_id"]),
                    box=PageBox(page=int(s["page"]), box=BBox.from_hgbench(s["box"])),
                )
                for s in entry.get("step_boxes", [])
            ]
            answers.append(
                GoldenAnswer(
                    qid=str(entry["question_id"]),
                    status=AnswerStatus.ANSWERED,
                    complete_answer_box=boxes,
                    steps=steps,
                )
            )
        return answers


def load_sample(directory: Path) -> GoldenSample:
    """Load one sample from its directory."""
    truth = directory / "truth.json"
    if not truth.is_file():
        raise FileNotFoundError(f"no truth.json in {directory}")
    return GoldenSample.model_validate_json(truth.read_text())


def save_sample(directory: Path, sample: GoldenSample) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "truth.json").write_text(
        json.dumps(sample.model_dump(mode="json"), indent=2) + "\n"
    )


def load_set(root: Path) -> list[GoldenSample]:
    """Load every sample under a root directory, in a stable order.

    Sorted so that a report can be diffed against a previous run — an unstable
    ordering makes a regression look like churn.
    """
    if not root.is_dir():
        return []
    samples = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if (directory / "truth.json").is_file():
            samples.append(load_sample(directory))
    return samples
