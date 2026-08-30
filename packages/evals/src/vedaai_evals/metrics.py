"""Metric definitions.

Pure functions over ground truth and prediction, with no knowledge of the
pipeline. That separation is deliberate: a metric that reaches into the system it
measures tends to drift toward measuring what the system happens to produce.

Two decisions here shape everything the project will later claim about accuracy.

**OCR line recall is reported first and on its own.** It is the binding ceiling —
measured at roughly 90% — and a regression in it presents as a mapping
regression, sending you to debug the wrong module. Separating it makes the
attribution unambiguous.

**The false-unanswered rate is reported separately from mapping accuracy.**
Averaged into a single accuracy figure it disappears: getting 19 of 20 questions
right looks excellent whether the failure was a slightly-off highlight or a
confident claim that an answered question was left blank. The second is the error
a teacher acts on without checking, so it does not get to hide inside an average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vedaai_contracts import AnswerStatus, BBox, PageBox

#: Overlap at which a predicted box is considered to have found a truth line.
#:
#: Generous on purpose. This measures *detection* — whether the recognizer saw
#: the line at all — not how tightly it framed it. A loosely-framed line is still
#: highlightable; a missed line is not.
DETECTION_IOU = 0.30

#: Overlap at which a highlight is considered correct. The stricter figure,
#: because this is the graded criterion.
HIGHLIGHT_IOU = 0.50


# -- OCR line recall ------------------------------------------------------


@dataclass
class RecallReport:
    matched: int = 0
    missed: int = 0
    spurious: int = 0
    missed_text: list[str] = field(default_factory=list)

    @property
    def total_truth(self) -> int:
        return self.matched + self.missed

    @property
    def recall(self) -> float:
        return self.matched / self.total_truth if self.total_truth else 1.0

    @property
    def precision(self) -> float:
        predicted = self.matched + self.spurious
        return self.matched / predicted if predicted else 1.0


def line_recall(
    truth: list[tuple[int, BBox, str]],
    predicted: list[tuple[int, BBox]],
    *,
    iou_threshold: float = DETECTION_IOU,
) -> RecallReport:
    """How many ground-truth lines the recognizer found.

    Matching is greedy per truth line and each prediction is consumed once, so a
    single sprawling predicted box cannot claim credit for every line it happens
    to overlap.

    ``missed_text`` is populated because the interesting question after a low
    recall is always *which* lines were lost — long ones, faint ones, ones near
    the margin — and that is not answerable from a percentage.
    """
    report = RecallReport()
    available = list(range(len(predicted)))

    for page, box, text in truth:
        best_index: int | None = None
        best_iou = 0.0
        for i in available:
            pred_page, pred_box = predicted[i]
            if pred_page != page:
                continue
            iou = box.iou(pred_box)
            if iou > best_iou:
                best_iou, best_index = iou, i

        if best_index is not None and best_iou >= iou_threshold:
            report.matched += 1
            available.remove(best_index)
        else:
            report.missed += 1
            report.missed_text.append(text)

    report.spurious = len(available)
    return report


# -- character error rate -------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Edit distance. Implemented rather than imported to avoid a dependency
    for thirty lines of dynamic programming."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(truth: str, predicted: str) -> float:
    """Edit distance normalized by truth length.

    Can exceed 1.0 when the prediction is longer than the truth and wrong, which
    is not a bug — a recognizer that hallucinates text is worse than one that
    returns nothing, and the metric should say so.
    """
    if not truth:
        return 0.0 if not predicted else 1.0
    return levenshtein(truth, predicted) / len(truth)


# -- question extraction --------------------------------------------------


@dataclass
class ExtractionReport:
    matched: int = 0
    missed: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)
    order_tau: float | None = None

    @property
    def precision(self) -> float:
        predicted = self.matched + len(self.spurious)
        return self.matched / predicted if predicted else 1.0

    @property
    def recall(self) -> float:
        total = self.matched + len(self.missed)
        return self.matched / total if total else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def normalize_label(label: str) -> str:
    """Collapse whitespace for comparison, and nothing else.

    Deliberately minimal. The requirement is to preserve the original numbering,
    so ``11 (a)`` and ``11(a)`` compare equal on spacing but ``11a`` does not
    become ``11(a)`` — the brackets are what the paper printed.
    """
    return " ".join(label.split())


def extraction_scores(
    truth_labels: list[str],
    predicted_labels: list[str],
) -> ExtractionReport:
    """Precision, recall, F1 and ordering on extracted question labels.

    Compared on exact labels rather than on question text, because the graded
    requirement is that every question is found and its numbering preserved. Text
    similarity would let a run that silently merged ``11(a)`` and ``11(b)`` into
    one entry score well.
    """
    report = ExtractionReport()

    remaining = [normalize_label(p) for p in predicted_labels]
    truth_norm = [normalize_label(t) for t in truth_labels]

    matched_positions: list[tuple[int, int]] = []
    for truth_index, label in enumerate(truth_norm):
        if label in remaining:
            predicted_index = remaining.index(label)
            remaining[predicted_index] = "\x00consumed"  # keep indices stable
            report.matched += 1
            matched_positions.append((truth_index, predicted_index))
        else:
            report.missed.append(label)

    report.spurious = [r for r in remaining if r != "\x00consumed"]
    report.order_tau = kendall_tau(
        [t for t, _ in matched_positions], [p for _, p in matched_positions]
    )
    return report


def kendall_tau(a: list[int], b: list[int]) -> float | None:
    """Rank correlation between two orderings of the same items.

    Measures whether questions came out in the printed order, which is a stated
    requirement and is invisible to precision and recall — a run can find every
    question and still present them scrambled, which on a multi-column paper is
    exactly what a naive reading order does.

    Returns None for fewer than two pairs, where correlation is undefined.
    """
    n = len(a)
    if n < 2 or n != len(b):
        return None

    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = (a[i] > a[j]) - (a[i] < a[j])
            sign_b = (b[i] > b[j]) - (b[i] < b[j])
            product = sign_a * sign_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    total = concordant + discordant
    return (concordant - discordant) / total if total else None


# -- answer mapping -------------------------------------------------------


@dataclass
class MappingReport:
    """Outcome counts for one submission's mapping."""

    correct: int = 0
    wrong_region: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    false_answer: list[str] = field(default_factory=list)
    correctly_unanswered: int = 0
    not_required_respected: int = 0
    not_required_violated: list[str] = field(default_factory=list)

    @property
    def scored(self) -> int:
        return (
            self.correct
            + len(self.wrong_region)
            + len(self.missed)
            + len(self.false_answer)
            + self.correctly_unanswered
        )

    @property
    def accuracy(self) -> float:
        return (self.correct + self.correctly_unanswered) / self.scored if self.scored else 1.0

    @property
    def placement_accuracy(self) -> float:
        """Share of questions whose answer reached the right question at all.

        `accuracy` above requires the highlight to be tight as well, and the two
        move independently: changing the shape of a highlight cannot move an
        answer to a different question, but it moves `accuracy` by twenty-five
        points. Reported separately so a change to one is never read as a change
        to the other — this file already records that confusion happening once,
        when gating on the region marked highlights sitting exactly on the ink as
        the wrong region and mapping accuracy read 46.9% for a set where nothing
        was mapped wrongly at all.
        """
        placed = self.correct + len(self.wrong_region) + self.correctly_unanswered
        return placed / self.scored if self.scored else 1.0

    @property
    def false_unanswered_rate(self) -> float:
        """Share of genuinely answered questions reported as unanswered.

        The headline safety number. A teacher acts on "unanswered" without
        re-reading the script, so this error is the one that reaches a student's
        grade uncorrected.
        """
        answered = self.correct + len(self.wrong_region) + len(self.missed)
        return len(self.missed) / answered if answered else 0.0


@dataclass(frozen=True)
class MappingCase:
    """One question's truth and prediction, paired.

    Primitives rather than the golden-set models, so that these functions stay
    independent of how truth happens to be stored.
    """

    qid: str
    truth_status: AnswerStatus
    truth_boxes: list[PageBox]
    predicted_status: AnswerStatus
    predicted_boxes: list[PageBox]
    #: The lines the answer occupies, where truth records them.
    #:
    #: Scored separately from the region because they measure different things and
    #: the difference is not small. A highlight drawn as one rectangle around four
    #: spread-out lines scored a perfect 1.0 against a region stored the same way,
    #: while covering 60 per cent blank paper — so the metric was rewarding exactly
    #: the fault a teacher complained about, and a tight highlight would have
    #: scored *worse*. The region stays because HG-Bench is defined on it and its
    #: baselines are the only external comparison available.
    truth_lines: list[PageBox] = field(default_factory=list)


def mapping_scores(
    cases: list[MappingCase],
    *,
    iou_threshold: float = HIGHLIGHT_IOU,
) -> tuple[MappingReport, list[float], list[float]]:
    """Score mapping outcomes, and collect the IoU of every correct-ish mapping.

    Returns the report plus per-question IoU values for the questions that were
    correctly identified as answered, so highlight quality can be summarized
    separately from whether the mapping picked the right question at all. Those
    are different failures with different fixes: a wrong question is an alignment
    problem, a poor box is a geometry or segmentation problem.
    """
    report = MappingReport()
    ious: list[float] = []
    line_ious: list[float] = []

    for case in cases:
        truth_answered = case.truth_status is AnswerStatus.ANSWERED
        # Any status that does not assert absence is treated as a claim that
        # something is there — "needs review" is not the same as "unanswered",
        # and scoring it as absence would credit the system for hedging.
        predicted_answered = case.predicted_status not in {
            AnswerStatus.UNANSWERED,
            AnswerStatus.NOT_REQUIRED,
        }

        if case.truth_status is AnswerStatus.NOT_REQUIRED:
            # Legitimately skipped under the paper's own choice rules. Reporting
            # it as an omission is a product error, not a mapping one.
            if case.predicted_status in {AnswerStatus.NOT_REQUIRED, AnswerStatus.UNANSWERED}:
                report.not_required_respected += 1
            else:
                report.not_required_violated.append(case.qid)
            continue

        if truth_answered and predicted_answered:
            iou = multi_box_iou(case.truth_boxes, case.predicted_boxes)
            ious.append(iou)

            # Correctness is judged against the writing where truth records it.
            #
            # Comparing a per-line highlight to a region drawn around those same
            # lines scores about 0.4 — the gaps between the lines count against it
            # — so gating on the region marked a highlight sitting exactly on the
            # ink as the wrong region, and mapping accuracy read 46.9% for a set
            # where nothing had been mapped to the wrong question at all.
            #
            # The region number is still computed and still reported, because
            # HG-Bench is defined on it. It is simply not the thing that decides
            # whether a highlight found the answer.
            judged = iou
            if case.truth_lines:
                against_writing = multi_box_iou(case.truth_lines, case.predicted_boxes)
                line_ious.append(against_writing)
                judged = against_writing

            if judged >= iou_threshold:
                report.correct += 1
            else:
                report.wrong_region.append(case.qid)
        elif truth_answered and not predicted_answered:
            report.missed.append(case.qid)
        elif not truth_answered and predicted_answered:
            report.false_answer.append(case.qid)
        else:
            report.correctly_unanswered += 1

    return report, ious, line_ious


def summarize_iou(ious: list[float]) -> dict[str, float]:
    """Mean IoU plus hit rates at the two thresholds that matter."""
    if not ious:
        return {"mean": 0.0, "at_50": 0.0, "at_75": 0.0, "n": 0.0}
    return {
        "mean": sum(ious) / len(ious),
        "at_50": sum(1 for i in ious if i >= 0.50) / len(ious),
        "at_75": sum(1 for i in ious if i >= 0.75) / len(ious),
        "n": float(len(ious)),
    }


def multi_box_iou(truth: list[PageBox], predicted: list[PageBox]) -> float:
    """IoU over a multi-page region.

    Intersection and union are accumulated per page and summed, which is what
    makes a page-spanning answer scoreable at all: one box across two pages has
    no meaningful rectangle, and comparing per-page then combining does.

    Assumes boxes within each side do not overlap each other — true for
    highlights, which are built as one union box per page.
    """
    if not truth and not predicted:
        return 1.0
    if not truth or not predicted:
        return 0.0

    pages = {pb.page for pb in truth} | {pb.page for pb in predicted}
    intersection = 0.0
    union = 0.0

    for page in pages:
        t = [pb.box for pb in truth if pb.page == page]
        p = [pb.box for pb in predicted if pb.page == page]
        t_area = sum(b.area for b in t)
        p_area = sum(b.area for b in p)
        inter = sum(tb.intersection_area(pb) for tb in t for pb in p)
        intersection += inter
        union += t_area + p_area - inter

    return intersection / union if union > 0 else 0.0


# -- does a written label reach its own answer? ---------------------------


#: A line no wider than this share of the page is a margin token, not a line of
#: writing.
_LABEL_MAX_WIDTH = 0.08

#: Two boxes sharing this much of the shorter one's height are on the same row.
_SAME_ROW = 0.5


@dataclass
class BindingReport:
    """Whether each question number reaches the line it was written beside.

    Measured directly rather than inferred from the mapping, because a mapping is
    an outcome and this is a property — and the difference is not academic. A fault
    that made every margin number bind to the *following* line, so that every
    answer began with its neighbour's last sentence, changed no number in this
    harness at all: the answers here are two lines long and semantically distinct,
    so the aligner still picked the right question from shifted text and the
    outcome looked fine. On a real script with longer answers it mangled all of
    them.

    An outcome metric cannot see a defect that does not change the outcome on easy
    data. This is the property itself.
    """

    bound: int = 0
    total: int = 0
    broken: list[str] = field(default_factory=list)

    @property
    def share(self) -> float | None:
        """None, not 1.0, when the script writes no numbers — there is nothing to
        measure, and scoring that as perfect would flatter the average."""
        return (self.bound / self.total) if self.total else None


def label_binding(lines: list[tuple[int, BBox, str]]) -> BindingReport:
    """How many margin numbers are immediately followed by their own line."""
    report = BindingReport()

    for i, (page, box, text) in enumerate(lines):
        if (box.x1 - box.x0) > _LABEL_MAX_WIDTH or len(text.strip()) > 6:
            continue

        mates = [
            (p, b, t)
            for j, (p, b, t) in enumerate(lines)
            if j != i and p == page and _row_overlap(box, b) >= _SAME_ROW
        ]
        if not mates:
            continue

        report.total += 1
        widest = max(mates, key=lambda m: (m[1].x1 - m[1].x0))
        following = lines[i + 1] if i + 1 < len(lines) else None
        if following is not None and following[2] == widest[2]:
            report.bound += 1
        else:
            report.broken.append(f"{text.strip()!r} p{page}")

    return report


def _row_overlap(a: BBox, b: BBox) -> float:
    top, bottom = max(a.y0, b.y0), min(a.y1, b.y1)
    if bottom <= top:
        return 0.0
    shorter = min(a.y1 - a.y0, b.y1 - b.y0)
    return (bottom - top) / shorter if shorter > 0 else 0.0


# -- cross-detector agreement (no labelling required) ---------------------


@dataclass
class AgreementReport:
    """Agreement between two independent detectors on where content is.

    A proxy for OCR line recall that needs no human labelling, which matters
    because ground-truth boxes on real pages are expensive and may never exist.

    The two detectors are genuinely independent in mechanism: one recognizes
    text, the other thresholds ink. So ink found where no line was reported is
    real evidence the recognizer missed something, not a restatement of what the
    recognizer already believes. Bootstrapping truth from the recognizer's own
    output would score near-perfect recall by construction and measure nothing.

    What this is NOT is ground truth, and the difference has teeth:

    * Ink has its own blind spots — very faint pen, writing that merges with a
      printed rule — so content both detectors miss is invisible here. The proxy
      therefore *overstates* recall.
    * Ink and OCR segment differently. One line of writing can be several ink
      regions, or vice versa, so the ratio is not a like-for-like count.

    Treat it as a regression signal — a change that increases uncovered ink has
    probably made detection worse — rather than as an absolute figure to quote.
    """

    ink_regions: int = 0
    covered_by_text: int = 0
    uncovered: int = 0

    @property
    def coverage(self) -> float:
        """Share of substantive ink that the recognizer accounted for."""
        return self.covered_by_text / self.ink_regions if self.ink_regions else 1.0


def detector_agreement(
    ink_covered_flags: list[bool],
) -> AgreementReport:
    """Summarize how much substantive ink the recognizer accounted for.

    Takes the already-reconciled flags rather than recomputing overlap, so the
    metric and the pipeline cannot disagree about what "covered" means — which is
    exactly the kind of drift that produces a metric measuring its own
    reimplementation.
    """
    report = AgreementReport(ink_regions=len(ink_covered_flags))
    report.covered_by_text = sum(1 for covered in ink_covered_flags if covered)
    report.uncovered = report.ink_regions - report.covered_by_text
    return report
