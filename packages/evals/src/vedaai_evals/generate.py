"""Synthetic question papers and answer sheets, with exact ground truth.

The generator draws the page, so it knows where every answer is. Ground truth
costs nothing, which is what makes it possible to test the graded edge cases in
volume — out-of-order answers, unanswered questions, orphan blocks, page-spanning
answers, merged sub-parts, mislabelled answers — rather than hoping a handful of
real scripts happen to contain them.

What this measures and what it does not is worth being precise about, because
overclaiming here would quietly invalidate every number the project reports.

**Measured well:** answer-to-question mapping and highlight geometry. Both are
structural. Whether an answer written third belongs to question one is a question
about labels, order and content, not about how convincingly the ink resembles
handwriting.

**Not measured at all:** recognition. These pages use handwriting *fonts*, which
are uniform, cleanly separated and perfectly baselined. Real handwriting is none
of those. OCR line recall and character error rate mean nothing here and must come
from real labelled pages — which is exactly the division of labour the two-tier
golden set exists to express.

Also not measured: skew, shadow, bleed-through and camera perspective. Those are
render-time distortions that would invalidate the axis-aligned truth boxes unless
the boxes were transformed alongside them, and real pages already exercise them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import fitz
from vedaai_contracts import AnswerStatus, BBox, PageBox

from .schema import GoldenAnswer, GoldenQuestion, GoldenSample, save_sample

A4 = fitz.paper_rect("a4")

#: Handwriting faces to draw answers with, tried in order.
#:
#: These are macOS system fonts and will be absent on the Linux worker and in CI.
#: That is tolerable precisely because realism is not what synthetic pages
#: measure — the fallback is a plain face with the same jitter, and every
#: structural property being tested survives unchanged. Silently producing
#: different-looking pages on different machines would be a problem if the metrics
#: depended on appearance; they do not.
_HANDWRITING_FONTS = [
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
]


def available_handwriting_font() -> str | None:
    for path in _HANDWRITING_FONTS:
        if Path(path).is_file():
            return path
    return None


@dataclass(frozen=True)
class SyntheticQuestion:
    qid: str
    label: str
    text: str
    marks: int | None
    section: str
    indent: float = 0.0
    answer: str = ""
    required: bool = True

    is_stem: bool = False
    """Introduces the parts below and asks nothing itself.

    Included because real papers are full of them — "2. Answer the following:" —
    and because the harness could not see the failure they cause. A stem left in
    the matching candidate list can absorb the answer to its own sub-part, which
    costs two mappings rather than one, and it is reported unanswered when nothing
    was ever asked.
    """


#: The canonical paper. Structures here mirror what official CBSE, CISCE, AQA and
#: College Board papers actually do: continuous numbering across sections,
#: three-level nesting, printed marks, and a section whose questions are optional.
PAPER: list[SyntheticQuestion] = [
    SyntheticQuestion(
        qid="A/1",
        label="1.",
        text="Define refraction of light.",
        marks=2,
        section="A",
        answer="Refraction is the bending of light when it passes from one medium to another.",
    ),
    SyntheticQuestion(
        qid="A/2",
        label="2.",
        text="Answer the following:",
        marks=None,
        section="A",
        is_stem=True,
    ),
    SyntheticQuestion(
        qid="A/2/i",
        label="2 (i)",
        text="State the laws of reflection.",
        marks=2,
        section="A",
        indent=16,
        answer=(
            "The angle of incidence equals the angle of reflection, "
            "and both rays lie in one plane."
        ),
    ),
    SyntheticQuestion(
        qid="A/2/i/a",
        label="2 (i) (a)",
        text="Draw a labelled ray diagram.",
        marks=3,
        section="A",
        indent=32,
        answer="Diagram drawn with incident ray, normal and reflected ray labelled.",
    ),
    SyntheticQuestion(
        qid="A/2/ii",
        label="2 (ii)",
        text="What is the SI unit of power?",
        marks=1,
        section="A",
        indent=16,
        answer="The watt.",
    ),
    SyntheticQuestion(
        qid="A/3",
        label="3.",
        text="Sound travels faster in water than in air. Give a reason.",
        marks=2,
        section="A",
        answer="Water is denser than air, so sound waves travel through it more quickly.",
    ),
    SyntheticQuestion(
        qid="B/4",
        label="4.",
        text="Explain the working of an electric motor.",
        marks=5,
        section="B",
        answer=(
            "A current-carrying coil placed in a magnetic field experiences a force, "
            "which turns the coil and converts electrical energy into rotation."
        ),
        required=False,
    ),
    SyntheticQuestion(
        qid="B/5/a",
        label="5 (a)",
        text="State Ohm's law.",
        marks=3,
        section="B",
        answer="Current through a conductor is proportional to the potential difference across it.",
        required=False,
    ),
    SyntheticQuestion(
        qid="B/5/b",
        label="5 (b)",
        text="A resistor carries 2 A at 10 V. Find its resistance.",
        marks=2,
        section="B",
        indent=16,
        answer="R = V / I = 10 / 2 = 5 ohm.",
        required=False,
    ),
    SyntheticQuestion(
        qid="B/6",
        label="6.",
        text="Describe an experiment to show that air has mass.",
        marks=5,
        section="B",
        answer="Weigh a sealed flask, pump the air out, and weigh it again. The mass decreases.",
        required=False,
    ),
]

#: Blocks that answer nothing in the paper. Required by the brief, and in real
#: scripts this is usually rough working or a note to the examiner.
ORPHAN_BLOCKS = [
    "Rough work: 12 x 4 = 48, then divide by 6 to get 8.",
    "Sir, I have attempted question 5 on the last page.",
    "Formula sheet: v = u + at, s = ut + half a t squared.",
]


@dataclass
class CaseConfig:
    """One synthetic scenario.

    Each field switches on a structure the requirements call out, so a case is a
    statement about which edge cases a run is being held to.
    """

    case_id: str
    notes: str

    answer_order: str = "in_order"
    """'in_order', 'shuffled' or 'reversed'. Out-of-order answers are a stated
    requirement, and reversed is the adversarial extreme for a monotone aligner."""

    omit: tuple[str, ...] = ()
    """Questions left genuinely unanswered."""

    skip_optional: tuple[str, ...] = ()
    """Optional questions not attempted. Correct behaviour, and must be reported
    as not-required rather than as an omission."""

    orphans: int = 0
    """Blocks of writing answering no question."""

    span_pages: tuple[str, ...] = ()
    """Answers deliberately split across a page boundary."""

    merge_subparts: tuple[tuple[str, ...], ...] = ()
    """Groups of sub-parts written as one undivided block, which the mapper has to
    split. The hardest supported case."""

    mislabel: tuple[tuple[str, str], ...] = ()
    """(qid, wrong label) — the student writes the wrong question number. This
    attacks label anchors directly."""

    unlabelled: bool = False
    """Answers written with no question numbers at all, so only order and content
    can map them."""

    seed: int = 0


#: The matrix. Each case isolates one structure so a failure names its own cause;
#: the last combines several, because real scripts do not arrive one problem at a
#: time.
CASES: list[CaseConfig] = [
    CaseConfig("baseline", "Every question answered in order, all labelled."),
    CaseConfig("out_of_order", "Answers shuffled.", answer_order="shuffled", seed=7),
    CaseConfig("reversed", "Answers in reverse order.", answer_order="reversed"),
    CaseConfig("unanswered", "Two questions left blank.", omit=("A/2/ii", "B/6")),
    CaseConfig(
        "optional_skipped",
        "Optional section questions not attempted; must not read as omissions.",
        skip_optional=("B/5/a", "B/5/b", "B/6"),
    ),
    CaseConfig("orphans", "Three blocks answering nothing.", orphans=3),
    CaseConfig("page_spanning", "Two answers cross a page break.", span_pages=("B/4", "A/3")),
    CaseConfig(
        "merged_subparts",
        "Sub-parts of 2 and 5 written as single blocks.",
        merge_subparts=(("A/2/i", "A/2/i/a", "A/2/ii"), ("B/5/a", "B/5/b")),
    ),
    CaseConfig(
        "mislabelled",
        "Two answers carry the wrong question number.",
        mislabel=(("A/3", "8."), ("B/4", "2 (iii)")),
    ),
    CaseConfig("unlabelled", "No question numbers written at all.", unlabelled=True),
    CaseConfig(
        "everything",
        "Shuffled, two unanswered, orphans, a page-spanning answer and a mislabel.",
        answer_order="shuffled",
        omit=("A/2/ii",),
        orphans=2,
        span_pages=("B/4",),
        mislabel=(("A/1", "9."),),
        seed=11,
    ),
]


class _Sheet:
    """Draws lines top-to-bottom and records the normalized box of each."""

    def __init__(self, *, font_path: str | None, rng: random.Random) -> None:
        self.doc = fitz.open()
        self.font_path = font_path
        self.rng = rng
        self.margin = 56.0
        self._page = self.doc.new_page(width=A4.width, height=A4.height)
        self._y = self.margin
        self._font_registered: set[int] = set()

    @property
    def page_index(self) -> int:
        return self.doc.page_count - 1

    def _ensure_font(self) -> str:
        if self.font_path is None:
            return "helv"
        if self.page_index not in self._font_registered:
            self._page.insert_font(fontname="hw", fontfile=self.font_path)
            self._font_registered.add(self.page_index)
        return "hw"

    def page_break(self) -> None:
        self._page = self.doc.new_page(width=A4.width, height=A4.height)
        self._y = self.margin

    def gap(self, amount: float = 12.0) -> None:
        self._y += amount
        self._wrap()

    def _wrap(self) -> None:
        if self._y > A4.height - self.margin:
            self.page_break()

    def line(
        self,
        text: str,
        *,
        size: float = 11.0,
        indent: float = 0.0,
        handwritten: bool = False,
        bold: bool = False,
    ) -> PageBox:
        """Draw one line, returning where it landed."""
        self._wrap()

        jitter_x = self.rng.uniform(-2.0, 2.0) if handwritten else 0.0
        jitter_y = self.rng.uniform(-1.5, 1.5) if handwritten else 0.0
        actual_size = size * (self.rng.uniform(0.94, 1.07) if handwritten else 1.0)

        font = self._ensure_font() if handwritten else ("hebo" if bold else "helv")
        x = self.margin + indent + jitter_x
        baseline = self._y + actual_size + jitter_y

        self._page.insert_text(
            fitz.Point(x, baseline), text, fontsize=actual_size, fontname=font
        )
        try:
            width = fitz.get_text_length(text, fontname=font, fontsize=actual_size)
        except Exception:  # noqa: BLE001 - embedded fonts may not support measurement
            width = actual_size * 0.5 * len(text)

        box = PageBox(
            page=self.page_index,
            box=BBox(
                x0=max(0.0, x / A4.width),
                y0=max(0.0, (baseline - actual_size) / A4.height),
                x1=min(1.0, (x + max(width, 4.0)) / A4.width),
                y1=min(1.0, (baseline + actual_size * 0.28) / A4.height),
            ),
        )
        self._y = baseline + actual_size * 0.75
        return box

    def to_bytes(self) -> bytes:
        return self.doc.tobytes()

    def close(self) -> None:
        self.doc.close()


def _wrap_text(text: str, width: int = 58) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_question_paper() -> tuple[bytes, list[GoldenQuestion]]:
    """Render the canonical paper and describe its questions."""
    sheet = _Sheet(font_path=None, rng=random.Random(0))
    sheet.line("SCIENCE - UNIT TEST", size=15, bold=True)
    sheet.line("Time allowed: 1 hour     Maximum Marks: 25", size=9.5)
    sheet.line("The marks for questions are shown in brackets.", size=9.5)
    sheet.gap()

    truth: list[GoldenQuestion] = []
    current_section = ""
    for order, q in enumerate(PAPER):
        if q.section != current_section:
            current_section = q.section
            sheet.gap(6)
            sheet.line(f"SECTION {q.section}", size=12.5, bold=True)
            instruction = (
                "(Attempt all questions from this Section)"
                if q.required
                else "(Attempt any two questions from this Section)"
            )
            sheet.line(instruction, size=9.5)
            sheet.gap(6)

        # A stem prints no allocation, which is exactly what makes it a stem —
        # the marks live on the parts below it.
        allocation = "" if q.marks is None else f"  [{q.marks}]"
        sheet.line(f"{q.label} {q.text}{allocation}", indent=q.indent)
        truth.append(
            GoldenQuestion(qid=q.qid, label_raw=q.label, print_order=order, marks=q.marks)
        )

    data = sheet.to_bytes()
    sheet.close()
    return data, truth


def build_answer_sheet(config: CaseConfig) -> tuple[bytes, list[GoldenAnswer]]:
    """Render an answer sheet per the case, recording exactly where each answer is."""
    rng = random.Random(config.seed)
    sheet = _Sheet(font_path=available_handwriting_font(), rng=rng)

    merged_lookup: dict[str, tuple[str, ...]] = {}
    for group in config.merge_subparts:
        for qid in group:
            merged_lookup[qid] = group

    mislabels = dict(config.mislabel)
    by_qid = {q.qid: q for q in PAPER}

    # Stems are excluded from the ordering rather than skipped inside the loop.
    # A stem has no answer, so its place in an answer order is meaningless — and
    # leaving it in makes the shuffled and reversed cases depend on how many
    # headings the paper happens to have, which silently changes every one of
    # those fixtures whenever the paper gains one.
    order = [q.qid for q in PAPER if not q.is_stem]
    if config.answer_order == "shuffled":
        rng.shuffle(order)
    elif config.answer_order == "reversed":
        order.reverse()

    answers: dict[str, GoldenAnswer] = {
        # Recorded directly: nothing was asked, so nothing is expected.
        q.qid: GoldenAnswer(qid=q.qid, status=AnswerStatus.NOT_REQUIRED)
        for q in PAPER
        if q.is_stem
    }
    written_groups: set[tuple[str, ...]] = set()

    sheet.line("Name: Test Student        Class: 6C", size=10)
    sheet.gap()

    for qid in order:
        if qid in config.omit:
            answers[qid] = GoldenAnswer(qid=qid, status=AnswerStatus.UNANSWERED)
            continue
        if qid in config.skip_optional:
            answers[qid] = GoldenAnswer(qid=qid, status=AnswerStatus.NOT_REQUIRED)
            continue

        group = merged_lookup.get(qid)
        if group is not None:
            if group in written_groups:
                continue
            written_groups.add(group)
            boxes = _write_merged(sheet, group, by_qid, config, mislabels)
            # Every sub-part in the group shares the block, which is the property
            # the mapper has to notice and then split.
            for member in group:
                answers[member] = GoldenAnswer(
                    qid=member,
                    status=AnswerStatus.ANSWERED,
                    complete_answer_box=boxes,
                    text=by_qid[member].answer,
                )
            continue

        boxes = _write_single(sheet, by_qid[qid], config, mislabels)
        answers[qid] = GoldenAnswer(
            qid=qid,
            status=AnswerStatus.ANSWERED,
            complete_answer_box=boxes,
            text=by_qid[qid].answer,
        )

    for i in range(config.orphans):
        sheet.gap()
        sheet.line(ORPHAN_BLOCKS[i % len(ORPHAN_BLOCKS)], handwritten=True)

    data = sheet.to_bytes()
    sheet.close()
    return data, [answers[q.qid] for q in PAPER if q.qid in answers]


def _label_for(qid: str, config: CaseConfig, mislabels: dict[str, str], label: str) -> str | None:
    if config.unlabelled:
        return None
    return mislabels.get(qid, label)


def _boxes_union_per_page(boxes: list[PageBox]) -> list[PageBox]:
    """Collapse per-line boxes into one box per page.

    Ground truth for an answer is its region, not its individual lines, and a
    page-spanning answer must stay as one entry per page it touches.
    """
    per_page: dict[int, list[BBox]] = {}
    for pb in boxes:
        per_page.setdefault(pb.page, []).append(pb.box)
    return [
        PageBox(page=page, box=BBox.union_all(bs)) for page, bs in sorted(per_page.items())
    ]


def _write_single(
    sheet: _Sheet,
    q: SyntheticQuestion,
    config: CaseConfig,
    mislabels: dict[str, str],
) -> list[PageBox]:
    sheet.gap()
    boxes: list[PageBox] = []
    label = _label_for(q.qid, config, mislabels, q.label)

    body = _wrap_text(q.answer)
    split_at = len(body) // 2 if q.qid in config.span_pages and len(body) > 1 else None

    for i, text in enumerate(body):
        prefix = f"{label} " if (i == 0 and label) else ""
        if split_at is not None and i == split_at:
            boxes.append(sheet.line("cont. on next page", handwritten=True))
            sheet.page_break()
        boxes.append(sheet.line(f"{prefix}{text}", handwritten=True))

    return _boxes_union_per_page(boxes)


def _write_merged(
    sheet: _Sheet,
    group: tuple[str, ...],
    by_qid: dict[str, SyntheticQuestion],
    config: CaseConfig,
    mislabels: dict[str, str],
) -> list[PageBox]:
    """Write several sub-parts as one undivided block.

    Only the first sub-part's label appears, and the answers run together without
    a break — which is what makes this hard: there is no structural cue marking
    where one sub-part ends and the next begins, only meaning.
    """
    sheet.gap()
    boxes: list[PageBox] = []
    first = by_qid[group[0]]
    label = _label_for(first.qid, config, mislabels, first.label)

    combined = " ".join(by_qid[qid].answer for qid in group)
    for i, text in enumerate(_wrap_text(combined)):
        prefix = f"{label} " if (i == 0 and label) else ""
        boxes.append(sheet.line(f"{prefix}{text}", handwritten=True))

    return _boxes_union_per_page(boxes)


def generate_case(config: CaseConfig, root: Path) -> GoldenSample:
    """Write one synthetic sample to disk and return its ground truth."""
    directory = root / config.case_id
    directory.mkdir(parents=True, exist_ok=True)

    paper_bytes, questions = build_question_paper()
    sheet_bytes, answers = build_answer_sheet(config)

    (directory / "question_paper.pdf").write_bytes(paper_bytes)
    (directory / "answer_sheet.pdf").write_bytes(sheet_bytes)

    sample = GoldenSample(
        sample_id=config.case_id,
        origin="synthetic",
        question_paper="question_paper.pdf",
        answer_sheet="answer_sheet.pdf",
        questions=questions,
        answers=answers,
        notes=config.notes,
    )
    save_sample(directory, sample)
    return sample


def generate_all(root: Path, cases: list[CaseConfig] | None = None) -> list[GoldenSample]:
    return [generate_case(config, root) for config in (cases or CASES)]


def adopt_real_pages(images: list[Path], root: Path) -> list[GoldenSample]:
    """Register real handwritten pages as unlabelled golden samples.

    Their answer-level truth is unknown, so mapping and highlight metrics cannot
    score them. What they can carry is detection: how much ink the recognizer
    accounted for, which needs no labelling and is the only signal available on
    real handwriting until ground-truth boxes exist.

    Each is paired with the synthetic question paper. That pairing is not
    meaningful — the answers do not correspond to those questions — and nothing
    that depends on the correspondence is scored. It exists because the pipeline
    ingests a pair, and the answer sheet is the half being measured.
    """
    paper_bytes, questions = build_question_paper()
    samples: list[GoldenSample] = []

    for image in images:
        sample_id = f"real-{image.stem}"
        directory = root / sample_id
        directory.mkdir(parents=True, exist_ok=True)

        (directory / "question_paper.pdf").write_bytes(paper_bytes)
        answer_name = f"answer_sheet{image.suffix.lower()}"
        (directory / answer_name).write_bytes(image.read_bytes())

        sample = GoldenSample(
            sample_id=sample_id,
            origin="real",
            question_paper="question_paper.pdf",
            answer_sheet=answer_name,
            questions=questions,
            answers=[],
            lines=[],
            notes=(
                f"Real handwriting from {image.name}. Unlabelled: detection metrics only. "
                "Paired with the synthetic paper, which the answers do not correspond to."
            ),
        )
        save_sample(directory, sample)
        samples.append(sample)

    return samples
