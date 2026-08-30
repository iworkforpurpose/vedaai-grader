"""Mapping answers to questions.

Two stages, in strict order of authority.

**Confirmed labels are honoured directly, in any order.** A student writing
``11 (b)`` in the margin, corroborated by evidence that did not come from the
label, has stated the answer. Order is then irrelevant, exactly as it is for a
human marker flipping to the labelled page.

**Everything left over is aligned by a monotone dynamic program**, scored on
semantics, position and length. Monotonicity is a reasonable prior for unlabelled
answers, which are usually written in order, and it is what makes the recurrence
tractable.

Getting that order of authority wrong was the phase's main error. The first
version treated confirmed anchors as *pins* for the DP, which required them to be
monotone — so on the fully-reversed golden case it kept one pin and turned eight
correctly-labelled answers into orphans, leaving 84% of the sheet unassigned.
Monotonicity is a convenience of the algorithm; it must not override direct
evidence.

Every structure the brief calls out then falls out of the formulation rather than
needing a rule of its own:

* an **unanswered question** is a gap on the question axis
* an **orphan answer** is a gap on the block axis
* a **page-spanning answer** is one question consuming several blocks
* **merged sub-parts** are one block serving several questions

Gap penalties are deliberately asymmetric. Skipping a question is cheap because
unanswered questions are ordinary. Emitting an orphan is dearer, because an orphan
is more often our own segmentation splitting one answer in two than the student
writing something extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import inf

from vedaai_contracts import (
    Anchor,
    AnchorStatus,
    AnswerBlock,
    AnswerStatus,
    BBox,
    Highlight,
    InkRegion,
    Mapping,
    MappingResult,
    MatchEvidence,
    MatchSignal,
    OrphanAnswer,
    PageBox,
    Question,
    QuestionPaper,
)

from .answers.similarity import Similarity, default_similarity
from .questions.expects import expects_a_drawing

#: Weights on the match score.
#:
#: The label term dominates by design: a student naming the question is stating
#: the answer directly, and no amount of semantic drift should outvote a
#: confirmed label.
#:
#: Order is meant to be weak, so that a real signal beats it — answering out of
#: order is permitted, and position is only a habit. At 0.3 it was not weak: the
#: prior spans [0, 1], so it contributed up to +0.30 while observed semantic
#: deviations on real prose span about ±0.22, which makes position the *dominant*
#: term rather than the tiebreaker the comment claimed.
#:
#: A real script showed the cost. A student's answer beginning "Invasive is a
#: significant word in the article because…" scored +0.218 against the question
#: that asks exactly that, and +0.063 against a later question about invasive
#: species — semantics preferred the right question by more than three times, and
#: the order prior handed the answer to the wrong one anyway.
#:
#: Halved. The golden set does not move at all across a sixfold sweep of this
#: weight, which is itself the finding: there, an answer's vocabulary always agrees
#: with its position, so the prior never has to decide anything and cannot be
#: measured. It earns its place only where semantics is silent — on scripts whose
#: recognition is too damaged to carry meaning — and for that a tiebreaker is
#: enough.
W_LABEL = 3.0
W_SEMANTIC = 1.0
W_ORDER = 0.15
W_LENGTH = 0.2

#: Cost of leaving a question unanswered, and of emitting an orphan block.
#:
#: Asymmetric, and the asymmetry is the point. Unanswered questions are ordinary.
#: An orphan is more often our own segmentation error than a real extra answer,
#: so the aligner should prefer attaching a stray block to a plausible question
#: over declaring it unattached.
SKIP_QUESTION = -0.40
SKIP_BLOCK = -0.80

#: Cost of extending an answer across another block, before evidence.
#: Cheap, because answers spanning blocks and pages is normal.
CONTINUE_BASE = -0.10

#: Cost of one block serving an additional question, before evidence.
#:
#: Tuning this alone could not work, and the two failed attempts show why. At
#: -0.55 the aligner preferred reporting sub-parts unanswered over sharing the
#: block they were plainly written in — false-unanswered claims, the worst error
#: available. At -0.30 it shared blocks onto questions the student had skipped
#: entirely, inventing answers for three optional questions.
#:
#: The missing signal was length. A block holding three sub-answers is roughly
#: three times the length of one, so length is what distinguishes a genuinely
#: merged answer from an unrelated block being spread across questions. The base
#: cost is therefore moderate and ``_share_support`` supplies the evidence.
SHARE_BASE = -0.45

#: Length beyond one question's expectation before sharing looks plausible. A
#: merged block must be substantially longer than a single answer, or there is no
#: second answer inside it to find.
SHARE_LENGTH_FACTOR = 1.4

#: Bonus when the block is long enough to hold another answer, and penalty when
#: it plainly is not. The penalty is what stops a short unrelated block being
#: spread across questions the student never attempted.
SHARE_LENGTH_BONUS = 0.55
SHARE_LENGTH_PENALTY = -0.9

#: Below this combined score a match is not worth making at all, and the aligner
#: must prefer a gap.
#:
#: Enforced by making such a pairing *unavailable*, which is not what the first
#: version did. It clamped low scores up to this value with ``max``, and since a
#: clamped -0.35 beats ``SKIP_QUESTION`` at -0.40, the floor guaranteed the DP
#: could never choose a gap at all — the precise opposite of the intent stated
#: here. On a real script with far more blocks than questions the effect was
#: total: every question received an answer, including ones nothing on the page
#: addressed.
#:
#: The golden set could not see it. There, blocks and questions are roughly equal
#: in number, so the DP has no surplus blocks it must place somewhere; the fault
#: only appears when a page carries several times more writing than the paper has
#: questions, which is the ordinary case for a real answer sheet.
MATCH_MINIMUM = -0.35

#: How much worse than a block's own best a question may score and still be
#: allowed to claim it.
#:
#: A ratio rather than a difference, so it means the same thing whatever scale the
#: measure works on. Seventy per cent: a question scoring less than seven tenths of
#: what the block's best question scores is not a plausible home for it.
#:
#: Found by looking at a review page. A real script answered question 2 across two
#: pages; the first page took question 2, and the second — scoring 0.689 for
#: question 2 and 0.439 for question 5 — was placed on question 5 and highlighted
#: there, under a question it plainly does not answer. The aligner had no notion of
#: "too much worse", only of "best available", and once the right question was
#: taken it settled for half as good.
SETTLE_RATIO = 0.70

#: Share of substantive ink left unassigned that suppresses absence claims.
#:
#: When this much writing belongs to no block, some answer went unmapped and the
#: system is in no position to tell a teacher a question was left blank.
UNASSIGNED_INK_SUPPRESSES = 0.18

#: How far a block's best question must lead the rest before that preference is
#: treated as a statement rather than a lean.
#:
#: This exists because maximising the whole path and getting each block right are
#: different objectives, and on a real script they disagree. Four pages of genuine
#: handwriting answered two questions: pages one and two answered question 1, pages
#: three and four answered question 2. Semantics said so unambiguously — +0.603 and
#: +0.608 for question 1, +0.525 and +0.556 for question 2, with every runner-up
#: half that or less.
#:
#: The aligner spread them across six questions anyway, and its arithmetic was
#: sound. Continuing an answer consumes a block without advancing to the next
#: question, so every question after it has fewer blocks to draw on; whereas
#: advancing collects a positive match now and avoids a -0.40 skip later. Six
#: mediocre matches beat two good ones plus four honest gaps. The DP was rewarded
#: for inventing answers.
#:
#: A margin this wide is not a lean. When a block prefers one question by this
#: much, that preference is evidence, and no amount of downstream bookkeeping
#: should be able to trade it away. Below it the DP's global view is still the
#: better judge, which is why this is a gate on strong cases and not a new weight.
DECISIVE_MARGIN = 0.25

#: Similarity to any block above which an unassigned question is reported
#: uncertain rather than unanswered.
#:
#: A targeted guard on the worst error this system can make, and it exists because
#: moving the global share and skip costs only traded false-unanswered against
#: false-answered — one knob, two errors, no setting that fixes both.
#:
#: The reasoning is direct: if some writing on the sheet plausibly answers this
#: question, then the honest report is "found writing I could not place", not "the
#: student left this blank". The unassigned-ink check cannot cover this case,
#: because the block may well be assigned — to a different question.
PLAUSIBLE_ANSWER_EXISTS = 0.18


class _State(Enum):
    """What the previous move left open."""

    FRESH = auto()
    """Nothing open. Any move is available."""

    IN_MULTI = auto()
    """A question is accumulating blocks, so another block may continue it."""

    IN_SHARE = auto()
    """A block is serving several questions, so another question may share it."""


@dataclass(frozen=True)
class _Move:
    kind: str
    question_index: int | None
    block_index: int | None


@dataclass
class _Cell:
    score: float
    move: _Move | None
    previous: tuple[int, int, _State] | None


@dataclass
class Assignment:
    """One question's resolved blocks, before status is decided."""

    qid: str
    block_ids: list[str]
    evidence: MatchEvidence
    shared_with: list[str]


def align(
    paper: QuestionPaper,
    blocks: list[AnswerBlock],
    anchors: list[Anchor],
    *,
    similarity: Similarity | None = None,
) -> list[Assignment]:
    """Assign blocks to questions.

    Confirmed anchors are honoured first and unconditionally, in any order. That
    is a correction of the original design, which treated them as pins for a
    monotone dynamic program and therefore kept only the longest monotone subset
    of them. On the fully-reversed golden case that left one pin standing and
    turned eight correctly-labelled answers into orphans — 84% of the writing on
    the sheet unassigned.

    The mistake was conceptual. A confirmed anchor is not a hint about order; it
    is the student stating which question this answers, corroborated by evidence
    that did not come from the label. Order is simply irrelevant once that is
    established, exactly as it is for a human marker flipping to the labelled
    page. Monotonicity is a convenience of the recurrence, and it must not
    override direct evidence.

    Whatever the anchors do not claim is then aligned by the DP, where
    monotonicity is a reasonable prior because unlabelled answers are usually
    written in order.
    """
    similarity = similarity or default_similarity

    # Give the scorer everything it will be asked about, in one go. Scoring is
    # every question against every block, so a scorer that fetches per pair makes
    # hundreds of calls for one submission; one that is told the whole set up front
    # makes one. Scorers with nothing to prepare do not implement this.
    warm = getattr(similarity, "warm", None)
    if callable(warm):
        warm(
            [q.text for q in paper.in_print_order() if not q.is_stem]
            + [b.text for b in blocks]
        )

    # Stems are excluded from candidacy. "2. Answer the following:" is a heading
    # with no marks and no answer of its own, and leaving it in the candidate list
    # lets it absorb the answer to its own sub-part — which costs two mappings,
    # not one, since the sub-part then reads as unanswered.
    questions = [q for q in paper.in_print_order() if not q.is_stem]
    if not questions or not blocks:
        return []

    # How well each block does against the best question on the whole paper.
    # Fixed before anything is claimed, so that what counts as "settling" does not
    # change as questions are taken.
    block_best = {
        block.block_id: max(
            (similarity.score(question.text, block.text) for question in questions),
            default=0.0,
        )
        for block in blocks
    }

    assignments: list[Assignment] = []
    claimed_qids: set[str] = set()
    claimed_blocks: set[str] = set()

    block_of_line: dict[str, str] = {}
    for block in blocks:
        for line_id in block.line_ids:
            block_of_line[line_id] = block.block_id
    known_qids = {q.qid for q in questions}

    for anchor in anchors:
        if not anchor.may_pin or anchor.claimed_qid is None:
            continue
        if anchor.claimed_qid in claimed_qids or anchor.claimed_qid not in known_qids:
            continue
        block_id = block_of_line.get(anchor.line_id)
        if block_id is None or block_id in claimed_blocks:
            continue

        claimed_qids.add(anchor.claimed_qid)
        claimed_blocks.add(block_id)
        assignments.append(
            Assignment(
                qid=anchor.claimed_qid,
                block_ids=[block_id],
                evidence=MatchEvidence(
                    label_agreement=W_LABEL,
                    semantic_agreement=anchor.semantic_agreement,
                    total_score=W_LABEL,
                    signals=[MatchSignal.WRITTEN_LABEL],
                ),
                shared_with=[],
            )
        )

    # Blocks whose own content names one question, honoured the same way.
    #
    # This module already holds that "monotonicity is a convenience of the
    # recurrence, and it must not override direct evidence" — the lesson of the
    # reversed golden case, where treating confirmed anchors as pins turned eight
    # correctly-labelled answers into orphans. A block narrowed by its own meaning
    # to exactly one question is the same kind of evidence as a label the student
    # wrote, and was being subjected to the same constraint.
    #
    # A real script showed it. Two answers, written in the order 2 then 1. Both were
    # narrowed correctly and unambiguously, and the DP placed the first and dropped
    # the second, because taking question 2 first left no monotone path back to
    # question 1. Answering out of order is something the brief requires handling,
    # and it is ordinary student behaviour.
    open_questions = [q for q in questions if q.qid not in claimed_qids]
    open_blocks = [b for b in blocks if b.block_id not in claimed_blocks]
    decided_by_qid: dict[str, Assignment] = {}
    for qid, block_id, agreement in _decided_pairs(
        open_questions, open_blocks, similarity, block_best
    ):
        if block_id in claimed_blocks:
            continue

        # A second block decided on the same question is the rest of that answer,
        # not a competitor for it. Dropping it was what a page-spanning answer
        # looked like from here: a real script answered two questions across four
        # pages, two pages each, and claiming the strongest block per question left
        # the other two placed nowhere.
        existing = decided_by_qid.get(qid)
        if existing is not None:
            existing.block_ids.append(block_id)
            claimed_blocks.add(block_id)
            continue

        if qid in claimed_qids:
            continue

        claimed_qids.add(qid)
        claimed_blocks.add(block_id)
        assignment = Assignment(
            qid=qid,
            block_ids=[block_id],
            evidence=MatchEvidence(
                semantic_agreement=agreement,
                total_score=agreement,
                signals=[MatchSignal.SEMANTIC],
            ),
            shared_with=[],
        )
        decided_by_qid[qid] = assignment
        assignments.append(assignment)

    # Document order within each answer, so `start_line_id` and `end_line_id` still
    # name the span a reader would follow. The pairs arrive strongest-first, which
    # is right for deciding and wrong for reading.
    order = {block.block_id: i for i, block in enumerate(blocks)}
    for assignment in decided_by_qid.values():
        assignment.block_ids.sort(key=lambda bid: order.get(bid, 0))

    # A block that carries on from the answer directly above it.
    #
    # Page four of a real script holds one answer to question 2, written down the
    # page with a paragraph break in the middle that the segmenter read as a
    # boundary. The second half opens mid-sentence and quotes the passage, and
    # question 2 scores 0.444 against it, the highest on the paper. It went to
    # question 6 at 0.348, because question 2 had already been claimed by the
    # block above and a claimed question leaves the alignment entirely — so what
    # was left to compete for the tail did not include the right answer.
    #
    # `continue` is the move for exactly this, and it had nothing to carry the
    # tail to. Adjacency is therefore settled before the rest runs, on the same
    # terms the rest uses.
    hints = _label_hints(anchors, block_of_line, known_qids)
    labelled_blocks = {block_id for _qid, block_id in hints}
    owner_of_block: dict[str, Assignment] = {
        block_id: assignment
        for assignment in assignments
        for block_id in assignment.block_ids
    }
    question_by_qid = {question.qid: question for question in questions}

    for position, block in enumerate(blocks):
        if position == 0 or block.block_id in claimed_blocks:
            continue
        if block.block_id in labelled_blocks:
            # The student wrote a question number beside it. Their statement of
            # intent outranks what the block above happens to be.
            continue
        if block.is_text_free or not block.text.strip():
            # A region with no readable text has nothing to be the rest of. It may
            # be a diagram, and `_text_free_match_is_plausible` is where that is
            # decided, on the two honest reasons — a drawing question, or the
            # student's own label. Reaching around it attached an unreadable
            # region to the answer above purely because it sat under it, which is
            # how a highlight grows past the writing it marks and how the
            # unassigned-ink total that qualifies every absence claim on the page
            # quietly goes to zero.
            continue

        previous = blocks[position - 1]
        above = owner_of_block.get(previous.block_id)
        if above is None:
            continue
        question = question_by_qid.get(above.qid)
        if question is None:
            continue
        if not _runs_on_from(previous, block):
            continue

        # A tail still has to be about the question it is joining, in absolute
        # terms and not merely relative to its own best. Without that, "Rough
        # work: 12 x 4 = 48" and "Sir, I have attempted question 5 on the last
        # page" were both swallowed by the answer above them: writing that
        # answers nothing has no meaningful best, so asking whether it prefers
        # somewhere else is a question with no answer, and the noise says yes.
        #
        # The real tail clears this comfortably — 0.444 against the question it
        # belongs to, where 0.30 is the line below which a pair counts as
        # unrelated. What it does cost is a four-word fragment scoring zero
        # against its own answer, which stays an orphan. That is the better trade:
        # a highlight four words short is a smaller error than stray writing
        # presented to a teacher as part of an answer.
        fit = _semantic(question, block, similarity)
        floor = float(getattr(similarity, "unrelated_below", 0.0) or 0.0)
        if floor > 0.0 and fit < floor:
            continue
        best = block_best.get(block.block_id, 0.0)
        if best > 0.0 and fit < best * SETTLE_RATIO:
            # It prefers somewhere else clearly enough that this is a new answer,
            # not the rest of the one above.
            continue
        above.block_ids.append(block.block_id)
        claimed_blocks.add(block.block_id)
        # Recorded so a run of three blocks carries through, not only two.
        owner_of_block[block.block_id] = above

    blocks_by_id = {block.block_id: block for block in blocks}

    # A block whose own question has already been taken joins it, rather than
    # settling for whatever is left.
    #
    # A student answered question 1 on page one and again on page two. Both blocks
    # prefer question 1 and prefer it clearly — 0.626 and 0.656, against 0.493 and
    # 0.351 for the next best. The stronger one took it, and a claimed question
    # leaves the alignment entirely, so the page-one block came back to a field
    # with its own answer missing from it and settled on "State whether a python
    # is a specialist or a generalist", worth one mark. Clicking that question lit
    # up the whole pandas paragraph, which is what a teacher reported.
    #
    # Nothing in the scoring was wrong. What was wrong is that being taken removed
    # question 1 instead of making it something to join — and a few lines above,
    # for blocks the evidence has *narrowed* to one question, this module already
    # says the opposite: a second block on the same question is the rest of that
    # answer, not a competitor for it. This is the same statement for a block that
    # prefers a question without having been narrowed to it.
    for block in blocks:
        if block.block_id in claimed_blocks or block.block_id in labelled_blocks:
            continue
        if block.is_text_free or not block.text.strip():
            continue

        scored = sorted(
            ((_semantic(question, block, similarity), question) for question in questions),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not scored:
            continue
        top_score, top = scored[0]
        floor = float(getattr(similarity, "unrelated_below", 0.0) or 0.0)
        if top.qid not in claimed_qids or (floor > 0.0 and top_score < floor):
            continue

        owner = next((a for a in assignments if a.qid == top.qid), None)
        if owner is None:
            continue

        # Only where it answers that question about as well as the writing already
        # placed on it. Two halves of one answer look alike to the scorer — 0.626
        # and 0.656 here — and rough work sitting on the same paper does not.
        #
        # Judged against the answer already there rather than against the
        # runner-up, because the runner-up says nothing about whether this is more
        # of the same answer, and handing every leftover to whichever answer
        # already exists is how a highlight quietly grows to cover the page.
        placed = [
            _semantic(top, blocks_by_id[bid], similarity)
            for bid in owner.block_ids
            if bid in blocks_by_id
        ]
        if not placed or top_score < max(placed) * SETTLE_RATIO:
            continue

        owner.block_ids.append(block.block_id)
        claimed_blocks.add(block.block_id)
        owner.block_ids.sort(key=lambda bid: order.get(bid, 0))

    remaining_questions = [q for q in questions if q.qid not in claimed_qids]
    remaining_blocks = [b for b in blocks if b.block_id not in claimed_blocks]

    assignments.extend(
        _align_segment(
            remaining_questions,
            remaining_blocks,
            similarity=similarity,
            label_hints=hints,
            block_best=block_best,
        )
    )
    return assignments


def _decided_pairs(
    questions: list[Question],
    blocks: list[AnswerBlock],
    similarity: Similarity,
    block_best: dict[str, float] | None = None,
) -> list[tuple[str, str, float]]:
    """Block-question pairs the evidence has already settled.

    A pair qualifies when the block's own content leaves exactly one question
    standing after the score matrix has applied its floors — unrelated pairs and
    pairs the block would be settling for are already gone by then, so what remains
    single is a statement rather than a preference.

    Strongest first, so that when two blocks are both settled on one question the
    better one takes it and the other falls to the DP, where `continue` can attach
    it to the same question as a second block of the same answer.
    """
    if not questions or not blocks:
        return []

    matrix = _score_matrix(questions, blocks, similarity, {}, block_best)
    pairs: list[tuple[str, str, float]] = []
    for j, block in enumerate(blocks):
        live = [i for i in range(len(questions)) if matrix[i][j] != -inf]
        if len(live) != 1:
            continue
        i = live[0]
        pairs.append((questions[i].qid, block.block_id, similarity.score(
            questions[i].text, block.text
        )))

    pairs.sort(key=lambda pair: pair[2], reverse=True)
    return pairs


#: Weight of a label that was written but not confirmed.
#:
#: Well below W_LABEL, because the whole point of confirmation is that an
#: unverified label has not earned that authority. Well above zero, because a
#: student writing a question number is still the most direct statement of intent
#: available and ignoring it entirely throws away real information.
W_LABEL_UNCONFIRMED = 0.9

#: Weight of a label the evidence contradicts. Small and still positive: a
#: disputed label is usually a slip rather than a fiction, so it remains faint
#: evidence for the question it names.
W_LABEL_DISPUTED = 0.25


def _label_hints(
    anchors: list[Anchor],
    block_of_line: dict[str, str],
    known_qids: set[str],
) -> dict[tuple[str, str], float]:
    """Scoring weight for labels the confirmation step did not accept."""
    hints: dict[tuple[str, str], float] = {}
    for anchor in anchors:
        if anchor.may_pin or anchor.claimed_qid is None:
            continue
        if anchor.claimed_qid not in known_qids:
            continue
        block_id = block_of_line.get(anchor.line_id)
        if block_id is None:
            continue
        weight = (
            W_LABEL_DISPUTED
            if anchor.status is AnchorStatus.DISPUTED
            else W_LABEL_UNCONFIRMED
        )
        hints[(anchor.claimed_qid, block_id)] = weight
    return hints


def _align_segment(
    questions: list[Question],
    blocks: list[AnswerBlock],
    *,
    similarity: Similarity,
    label_hints: dict[tuple[str, str], float] | None = None,
    block_best: dict[str, float] | None = None,
) -> list[Assignment]:
    """Monotone DP over the questions and blocks no anchor claimed."""
    if not questions or not blocks:
        return []

    n, m = len(questions), len(blocks)
    scores = _score_matrix(questions, blocks, similarity, label_hints, block_best)

    table: dict[tuple[int, int, _State], _Cell] = {
        (0, 0, _State.FRESH): _Cell(score=0.0, move=None, previous=None)
    }

    def best_at(i: int, j: int) -> tuple[float, _State] | None:
        options = [
            (table[(i, j, state)].score, state)
            for state in _State
            if (i, j, state) in table
        ]
        return max(options) if options else None

    for i in range(n + 1):
        for j in range(m + 1):
            current = best_at(i, j)
            if current is None:
                continue
            base, base_state = current

            if i < n and j < m:
                _offer(
                    table,
                    (i + 1, j + 1, _State.FRESH),
                    base + scores[i][j],
                    _Move("match", i, j),
                    (i, j, base_state),
                )
            if i < n:
                _offer(
                    table,
                    (i + 1, j, _State.FRESH),
                    base + SKIP_QUESTION,
                    _Move("skip_question", i, None),
                    (i, j, base_state),
                )
            if j < m:
                _offer(
                    table,
                    (i, j + 1, _State.FRESH),
                    base + SKIP_BLOCK,
                    _Move("skip_block", None, j),
                    (i, j, base_state),
                )

            # Continue and share are only available immediately after a pairing,
            # which is what keeps them from chaining arbitrarily.
            last = table[(i, j, base_state)].move
            if (
                last is not None
                and last.kind in {"match", "continue"}
                and j < m
                and i > 0
                # A region with no readable text needs the same reason to continue
                # an answer as it needs to start one. Without this the move became
                # the loophole for exactly what the match gate was added to stop:
                # unreadable margin fragments were swept into whichever answer
                # preceded them, and one answer collected the leftover ink of two
                # pages while every other question read as blank.
                and _may_continue_into(questions[i - 1], blocks[j])
            ):
                bonus = 0.0
                if blocks[j - 1].has_continuation_marker if j > 0 else False:
                    bonus = 0.5  # the student said so themselves
                # Two blocks in a row that both point unmistakably at the same
                # question are one answer that segmentation split, not two
                # answers. Left to its own arithmetic the DP prefers to advance —
                # a match now plus a skip avoided later beats continuing — so it
                # spread two pages about pandas across four questions while the
                # scores said "question 1" twice, at +0.36 against +0.06 for the
                # runner-up. The bonus makes the evidence decide it.
                if _agrees_with_the_answer_so_far(scores, j, i - 1, blocks):
                    bonus += 1.0
                _offer(
                    table,
                    (i, j + 1, _State.IN_MULTI),
                    # The same centred score `match` uses, not the raw one.
                    #
                    # This was raw similarity, and against surface scorers the
                    # difference was small enough to hide. Embeddings made it
                    # decisive: any two English texts sit around 0.4 apart, so a
                    # raw score carried a constant floor that `match` had already
                    # subtracted out, and continuing an answer therefore looked
                    # better than matching one no matter which question it was.
                    # A real script showed the cost — two blocks whose meaning
                    # pointed unambiguously at question 1, at 0.66 against 0.35
                    # for the runner-up, were placed on questions 3 and 4.
                    base + CONTINUE_BASE + bonus + scores[i - 1][j],
                    _Move("continue", i - 1, j),
                    (i, j, base_state),
                )
            if (
                last is not None
                and last.kind in {"match", "share"}
                and i < n
                and j > 0
                and last.question_index is not None
                and _may_share(questions[last.question_index], questions[i], blocks[j - 1])
            ):
                _offer(
                    table,
                    (i + 1, j, _State.IN_SHARE),
                    base
                    + SHARE_BASE
                    # Centred, for the same reason as `continue` above.
                    + scores[i][j - 1]
                    + _share_support(questions[i], blocks[j - 1]),
                    _Move("share", i, j - 1),
                    (i, j, base_state),
                )

    end = best_at(n, m)
    if end is None:
        return []

    return _traceback(table, (n, m, end[1]), questions, blocks)


def _offer(
    table: dict[tuple[int, int, _State], _Cell],
    key: tuple[int, int, _State],
    score: float,
    move: _Move,
    previous: tuple[int, int, _State],
) -> None:
    existing = table.get(key)
    if existing is None or score > existing.score:
        table[key] = _Cell(score=score, move=move, previous=previous)


def _score_matrix(
    questions: list[Question],
    blocks: list[AnswerBlock],
    similarity: Similarity,
    label_hints: dict[tuple[str, str], float] | None = None,
    block_best: dict[str, float] | None = None,
) -> list[list[float]]:
    """Score every question-block pairing.

    ``label_hints`` carries labels that were *not* confirmed — disputed or
    merely uncorroborated. Those still belong in the score: an uncorroborated
    label is weak evidence, not no evidence, and the original version omitted
    them entirely. On the reversed case that lost two correctly-labelled answers
    whose text happens to share no vocabulary with their questions, leaving the
    DP with nothing but a position prior that reversal had inverted.
    """
    hints = label_hints or {}
    raw = [
        [_semantic(question, block, similarity) for block in blocks] for question in questions
    ]
    # Each block's similarities re-expressed as deviations from its own mean.
    #
    # Centring says the necessary thing: a block equally similar to every question
    # is evidence for none of them. It also makes the flat value a text-free block
    # receives self-cancelling — as an absolute score that constant was a free win
    # the block could spend against a gap penalty on every question in the paper.
    #
    # Standardizing as well — dividing by each block's spread — was tried and
    # measured, because on real scripts the deviations are around ±0.06 against an
    # order prior worth ±0.3, which is genuine evidence far too quiet to be heard.
    # It cost the golden set five points of accuracy and four of mean IoU. The
    # reason is instructive: dividing by the spread discards magnitude, so a block
    # that barely prefers one question shouts as loudly as one that clearly does.
    #
    # The two regimes differ by nearly an order of magnitude in how much signal
    # they carry, and no single weight or rescaling serves both. That is not a
    # scoring problem to be solved but a recognition ceiling to be reported: where
    # the writing cannot be read, the mapping rests on position, and the honest
    # response is to say so in the confidence rather than to amplify noise until it
    # outvotes a real prior.
    baselines = [
        sum(raw[i][j] for i in range(len(questions))) / len(questions)
        for j in range(len(blocks))
    ]

    matrix: list[list[float]] = []
    for i, question in enumerate(questions):
        row: list[float] = []
        for j, block in enumerate(blocks):
            hint = hints.get((question.qid, block.block_id), 0.0)

            if not _text_free_match_is_plausible(question, block, hint):
                row.append(-inf)
                continue

            score = (
                W_SEMANTIC * (raw[i][j] - baselines[j])
                + W_ORDER * _order_prior(i, j, len(questions), len(blocks))
                + W_LENGTH * _length_plausibility(question, block)
                + hint
            )
            # Unavailable, not clamped. A pairing this weak must lose to a gap,
            # and clamping would instead make it beat one.
            row.append(score if score > MATCH_MINIMUM else -inf)
        matrix.append(row)

    # Two rules on the raw similarities, before anything is centred.
    #
    # Centring answers "which question does this block prefer?", which always has
    # an answer even when the block prefers all of them equally little. Neither of
    # the faults below is visible in a deviation, and both were found by opening a
    # review page rather than by reading a number.
    floor = float(getattr(similarity, "unrelated_below", 0.0) or 0.0)
    for j, block in enumerate(blocks):
        # The best across *every* question on the paper, not merely the ones still
        # unclaimed. Recomputing it here would mean that removing the right answer
        # from the pool makes a wrong one look acceptable — a block scoring 0.689
        # for the question it answers and 0.439 for another would settle for 0.439
        # the moment the first was taken, which is the exact fault this rule exists
        # to prevent.
        column = [raw[i][j] for i in range(len(questions))]
        best = (block_best or {}).get(block.block_id) or (max(column) if column else 0.0)
        # The question this block prefers among those on offer here, but only when
        # it prefers it enough to count as related at all. Its own parts are exempt
        # from both rules below, and the exemption reads "a part of the question
        # this block *plainly answers*" — so where there is no such question there
        # is nothing to be a part of.
        #
        # Without that condition the exemption fires on noise. An answer sheet of
        # handwritten C uploaded against a comprehension paper scores about 0.15
        # against every question, which makes the preference arbitrary, and 3(ii)
        # was made a sibling of whatever won and handed the code. The floor exists
        # to stop exactly that.
        preferred = max(range(len(questions)), key=lambda i: column[i]) if column else None
        if preferred is not None and floor > 0.0 and column[preferred] < floor:
            preferred = None
        for i in range(len(questions)):
            # Unrelated to this question outright. An answer sheet belonging to a
            # different paper scored 0.15 against every question while reporting
            # five of seven answered, and highlighted handwritten C as an essay
            # about pandas.
            unrelated = floor > 0.0 and raw[i][j] < floor
            # Or related, but not as well as somewhere else — settling for this
            # question because the one it answers has been taken.
            settling = best > 0.0 and raw[i][j] < best * SETTLE_RATIO
            if not (unrelated or settling):
                continue

            # A part of the question this block plainly answers is neither
            # unrelated to it nor somewhere to settle for. It is the other half of
            # the same answer, and the evidence tying the two together is the
            # paper's own numbering rather than anything the scorer can see.
            #
            # A history script answered Q.3(a) and Q.3(b) in one forty-four-word
            # run. Most of it is (a)'s answer, so the block scores 0.195 against
            # (b) — under the 0.30 that marks a pair as unrelated — and (b) was
            # discarded here, on a question the student had answered. The science
            # script's 11(a) and 11(b) survive only because that run happens to
            # score 0.752 against (b).
            #
            # Whether the run really covers both parts stays `share`'s decision,
            # with the length evidence behind it.
            if (
                preferred is not None
                and preferred != i
                and _is_part_of_the_same_question(questions[preferred], questions[i], block)
            ):
                continue

            matrix[i][j] = -inf

    # Judged on the semantic deviations alone, not on the assembled score.
    #
    # The order prior narrows the very margins this gate is asking about: on a real
    # script, two blocks whose meaning preferred question 2 by 0.281 and 0.250 came
    # out of the full score at 0.226 and 0.184, under the threshold. Which is
    # circular — position is exactly the signal being overruled, so letting it
    # decide whether the evidence is strong enough guarantees it wins whenever it
    # disagrees.
    deviations = [
        [raw[i][j] - baselines[j] if matrix[i][j] != -inf else -inf for j in range(len(blocks))]
        for i in range(len(questions))
    ]
    return _withhold_decided_blocks(matrix, deviations, questions, blocks)


def _withhold_decided_blocks(
    matrix: list[list[float]],
    evidence: list[list[float]],
    questions: list[Question],
    blocks: list[AnswerBlock],
) -> list[list[float]]:
    """Take away the pairings a block's own content has already ruled out.

    Where one question leads the rest by ``DECISIVE_MARGIN``, the block has said
    which question it answers, and the DP is not entitled to spend it elsewhere to
    balance its books. Making the alternatives unavailable is the same device the
    match floor uses, and for the same reason: a preference the DP can outvote is
    not a constraint.

    Deliberately narrow. It fires only on a clear margin, so a block that genuinely
    could belong to either of two questions is still settled by the DP's global
    view, which is better at that than any single pairing can be.
    """
    if not matrix:
        return matrix

    for j in range(len(matrix[0])):
        column = [(evidence[i][j], i) for i in range(len(matrix))]
        finite = [(s, i) for s, i in column if s != -inf]
        if len(finite) < 2:
            continue

        finite.sort(reverse=True)
        (best, winner), (runner_up, _) = finite[0], finite[1]
        if best - runner_up < DECISIVE_MARGIN:
            continue

        for _score, i in finite:
            if i == winner:
                continue
            # A part of the winning question is not a rival for the block; it is
            # the other half of the same answer, and a block leading decisively
            # towards 3(a) is not evidence against 3(b).
            if evidence[winner][j] > 0.0 and _is_part_of_the_same_question(
                questions[winner], questions[i], blocks[j]
            ):
                continue
            matrix[i][j] = -inf

    return matrix


def _decided_question(matrix: list[list[float]], j: int) -> int | None:
    """The question a block has been narrowed to, if it has been."""
    live = [i for i in range(len(matrix)) if matrix[i][j] != -inf]
    return live[0] if len(live) == 1 else None


#: A block shorter than this is a fragment, not an answer: a trailing word that
#: segmentation cut off, a stray mark. Measured on a real script, "there." — the
#: last word of an answer, separated from it by a page break.
_FRAGMENT_MAX_CHARS = 12


def _agrees_with_the_answer_so_far(
    matrix: list[list[float]], j: int, question: int, blocks: list[AnswerBlock]
) -> bool:
    """Whether this block continues an answer the previous block already stated.

    Adjacency by index was the first attempt, and a real script broke it at once:
    between two pages that both plainly answered the same question sat the block
    "there." — the tail of the first page's last sentence, cut off by the page
    break. Three blocks, not two, so nothing was adjacent and the evidence was
    ignored.

    Fragments are therefore looked past. A block of a dozen characters carries no
    opinion about which question it belongs to, and letting one break the chain
    means the commonest segmentation artefact there is can silently undo this.
    """
    if _decided_question(matrix, j) != question:
        return False

    for previous in range(j - 1, -1, -1):
        if len(blocks[previous].text.strip()) <= _FRAGMENT_MAX_CHARS:
            continue
        return _decided_question(matrix, previous) == question
    return False


def _may_continue_into(question: Question, block: AnswerBlock) -> bool:
    """Whether an answer can plausibly carry on into this block.

    Text blocks always may — that is what the move is for, and the score decides.
    A text-free block may only when the question expects a drawing or the block
    says it is a continuation, since otherwise "carries on the same answer" is an
    assertion with nothing behind it.
    """
    if not block.is_text_free and block.text.strip():
        return True
    return block.has_continuation_marker or expects_a_drawing(question.text)


def _text_free_match_is_plausible(
    question: Question, block: AnswerBlock, label_hint: float
) -> bool:
    """Whether a region with no readable text could be this question's answer.

    A block with no text carries no evidence about *which* question it answers,
    only that something is written. Left free to compete, such a block attaches to
    whichever question the position prior happens to favour — on real scripts,
    unreadable margin fragments were landing on whatever questions were left over,
    including "Explain why a sorted array...".

    So it needs a reason, and there are exactly two honest ones:

      * The question asks for a drawing. A diagram legitimately has no text, and
        refusing these would lose the case this whole ink pipeline exists for.
      * The student wrote a label pointing at it. Their own say-so outranks
        anything inferred from the text they did not manage to write legibly.

    With neither, the region becomes an orphan — which is not a loss of
    information but a more accurate report of it. The teacher sees it under
    "writing that matches no question", the unassigned-ink total rises, and that
    total is what downgrades every absence claim on the page from "not answered"
    to "check this". An unreadable region attached to an arbitrary question would
    have hidden all of that behind a confident answer.
    """
    if not block.is_text_free and block.text.strip():
        return True
    if label_hint > 0.0:
        return True
    return expects_a_drawing(question.text)


#: Vertical space, as a share of page height, that a continuation may leave below
#: the answer it carries on from. About two blank lines of ordinary handwriting —
#: a paragraph break, not a fresh start further down the page.
_TAIL_MAX_GAP = 0.12

#: How far down a page a continuation may begin when it carries on from the page
#: before. An answer running over the page break resumes at the top of the next
#: one; writing that starts halfway down it is something else.
_TAIL_PAGE_TOP = 0.25


def _runs_on_from(previous: AnswerBlock, block: AnswerBlock) -> bool:
    """Whether this block sits where the rest of the one above it would.

    Adjacency in the block list is not enough on its own. A block of rough work at
    the foot of the page is also "next", and joining it to the answer at the top
    would attach working to a question the student never meant it for.
    """
    if not previous.geometry or not block.geometry:
        return False

    ends = max(previous.geometry, key=lambda box: (box.page, box.box.y1))
    starts = min(block.geometry, key=lambda box: (box.page, box.box.y0))

    if starts.page == ends.page:
        return starts.box.y0 - ends.box.y1 <= _TAIL_MAX_GAP
    if starts.page == ends.page + 1:
        return starts.box.y0 <= _TAIL_PAGE_TOP
    return False


def _semantic(question: Question, block: AnswerBlock, similarity: Similarity) -> float:
    if not block.text.strip():
        # A text-free block is a diagram or an unreadable region. It carries no
        # semantic signal, and scoring it zero would make it lose every contest
        # against a gap — leaving a question answered by a drawing unanswered.
        return 0.25
    return similarity.score(question.text, block.text)


def _is_part_of_the_same_question(
    owner: Question, candidate: Question, block: AnswerBlock
) -> bool:
    """Whether these two are parts of one numbered question — 11 (a) and 11 (b).

    Stricter than `_may_share`, deliberately. That one calls two questions
    siblings when their labels have the same parent, and the parent of every
    top-level question is the same empty root, so under it question 1, question 4
    and question 6 are all siblings. In the alignment proper that is held in check
    by what sharing costs and by the length evidence supporting it. Used to excuse
    a question from a rule it is not held in check by anything, and an earlier
    version of this let a block about refraction through to "the chemical formula
    of washing soda" and "the tissue that transports water in a plant".

    So this asks the narrower question: are they parts of one question, rather
    than merely two questions at the same level.
    """
    if block.is_text_free or not block.text.strip():
        return False

    owner_path, candidate_path = tuple(owner.path), tuple(candidate.path)
    if len(owner_path) < 2 and len(candidate_path) < 2:
        # Two whole questions. Answering both in one run is possible and is what
        # `share` is for; it is not a reason to exempt either from a floor.
        return False
    if owner_path[:-1] and owner_path[:-1] == candidate_path[:-1]:
        return True
    # A part and the question it belongs to.
    shorter, longer = sorted((owner_path, candidate_path), key=len)
    return len(longer) > len(shorter) and longer[: len(shorter)] == shorter


def _may_share(owner: Question, candidate: Question, block: AnswerBlock) -> bool:
    """Whether one block can legitimately answer both of these questions.

    Sharing means a student answered several sub-parts of one question in a single
    run of writing — "11 (a)" and "11 (b)" answered as one paragraph. Two
    conditions make that plausible, and without them the move does real damage.

    They must be **relatives**: siblings under the same parent, or a parent and its
    own child. Unconstrained, the move chained across a whole paper — on a real
    script one unreadable ink region was assigned to five questions spanning three
    sections, which is not a shared answer but the same evidence spent five times.

    And the block must have **text**. Splitting a shared answer between questions
    means dividing its lines; a region with no readable text has nothing to divide,
    so claiming it answers several questions asserts something unfounded rather
    than something merely uncertain.
    """
    if block.is_text_free or not block.text.strip():
        return False

    owner_path, candidate_path = tuple(owner.path), tuple(candidate.path)
    if owner_path[:-1] == candidate_path[:-1]:
        return True
    # Parent and child, in either direction.
    shorter, longer = sorted((owner_path, candidate_path), key=len)
    return longer[: len(shorter)] == shorter


def _share_support(question: Question, block: AnswerBlock) -> float:
    """Whether this block is long enough to hold another answer.

    The discriminator between a genuinely merged answer and an unrelated block
    being spread across questions. Sub-parts written as one paragraph produce a
    block several times the length of a single answer; a short block shared
    across three questions is the aligner inventing answers.

    Without marks printed on the paper there is no expectation to compare
    against, so this stays neutral rather than guessing — a paper that omits marks
    should not have sharing suppressed on that account alone.
    """
    words = len(block.text.split())
    if not words:
        return 0.0
    if question.marks is None:
        return 0.0
    expected = max(6, question.marks * 12)
    return (
        SHARE_LENGTH_BONUS
        if words >= expected * SHARE_LENGTH_FACTOR
        else SHARE_LENGTH_PENALTY
    )


def _order_prior(i: int, j: int, n: int, m: int) -> float:
    """Preference for pairings that keep relative position.

    Weak by construction. Students do answer out of order, so this nudges rather
    than decides, and a strong label or a strong semantic match should beat it.
    """
    if n <= 1 or m <= 1:
        return 1.0
    return 1.0 - abs((i / (n - 1)) - (j / (m - 1)))


def _length_plausibility(question: Question, block: AnswerBlock) -> float:
    """Whether the answer's length suits the marks on offer.

    A five-mark question answered in four words, or a one-mark question answered
    in two hundred, is worth a small nudge away from. Small, because students are
    not consistent and a terse correct answer is still correct.
    """
    if question.marks is None or not block.text.strip():
        return 0.0
    words = len(block.text.split())
    expected = max(6, question.marks * 12)
    ratio = min(words, expected) / max(words, expected)
    return ratio


def _traceback(
    table: dict[tuple[int, int, _State], _Cell],
    end: tuple[int, int, _State],
    questions: list[Question],
    blocks: list[AnswerBlock],
) -> list[Assignment]:
    """Walk the backpointers and collect per-question assignments."""
    moves: list[_Move] = []
    key: tuple[int, int, _State] | None = end
    while key is not None:
        cell = table[key]
        if cell.move is not None:
            moves.append(cell.move)
        key = cell.previous
    moves.reverse()

    by_qid: dict[str, list[str]] = {}
    shared: dict[str, set[str]] = {}

    for move in moves:
        if move.kind in {"match", "continue"} and move.question_index is not None:
            qid = questions[move.question_index].qid
            block_id = blocks[move.block_index].block_id  # type: ignore[index]
            by_qid.setdefault(qid, []).append(block_id)
        elif move.kind == "share" and move.question_index is not None:
            qid = questions[move.question_index].qid
            block_id = blocks[move.block_index].block_id  # type: ignore[index]
            by_qid.setdefault(qid, []).append(block_id)
            shared.setdefault(block_id, set()).add(qid)

    out: list[Assignment] = []
    for qid, block_ids in by_qid.items():
        partners: set[str] = set()
        for block_id in block_ids:
            partners |= shared.get(block_id, set())
        partners.discard(qid)
        out.append(
            Assignment(
                qid=qid,
                block_ids=block_ids,
                evidence=MatchEvidence(signals=[MatchSignal.SEMANTIC, MatchSignal.POSITION]),
                shared_with=sorted(partners),
            )
        )
    return out


def resolve(
    paper: QuestionPaper,
    blocks: list[AnswerBlock],
    anchors: list[Anchor],
    ink_regions: list[InkRegion],
    *,
    similarity: Similarity | None = None,
    pages_uploaded: int = 0,
) -> MappingResult:
    """Produce the full mapping, including status for every question."""
    assignments = align(paper, blocks, anchors, similarity=similarity)
    by_qid = {a.qid: a for a in assignments}
    blocks_by_id = {b.block_id: b for b in blocks}

    used_block_ids = {bid for a in assignments for bid in a.block_ids}
    unassigned_ink = _unassigned_ink_share(blocks, used_block_ids, ink_regions)
    suppress = unassigned_ink >= UNASSIGNED_INK_SUPPRESSES

    satisfied_sections = _sections_satisfied(paper, set(by_qid))
    pages_missing = _pages_appear_missing(blocks, pages_uploaded)
    resolver = similarity or default_similarity

    mappings: list[Mapping] = []
    for question in paper.in_print_order():
        assignment = by_qid.get(question.qid)
        if assignment is not None:
            used = [blocks_by_id[bid] for bid in assignment.block_ids if bid in blocks_by_id]
            highlight = _highlight(used)
            status = AnswerStatus.ANSWERED
            if all(b.is_text_free for b in used) and used:
                # Ink with no readable text. Answered, but the text cannot be
                # graded, and saying so is very different from saying blank.
                status = AnswerStatus.OCR_FAILED
            mappings.append(
                Mapping(
                    qid=question.qid,
                    status=status,
                    block_ids=list(assignment.block_ids),
                    start_line_id=used[0].line_ids[0] if used and used[0].line_ids else None,
                    end_line_id=used[-1].line_ids[-1] if used and used[-1].line_ids else None,
                    highlight=highlight,
                    confidence=0.75,
                    evidence=assignment.evidence,
                    shares_block_with=assignment.shared_with,
                )
            )
            continue

        if question.is_stem:
            # A heading, not a question. Nothing was asked here, so there is
            # nothing to be absent — and the absence logic below would otherwise
            # report a blank the paper never invited.
            mappings.append(
                Mapping(
                    qid=question.qid,
                    status=AnswerStatus.NOT_REQUIRED,
                    confidence=1.0,
                )
            )
            continue

        mappings.append(
            Mapping(
                qid=question.qid,
                status=_absent_status(
                    question,
                    satisfied_sections=satisfied_sections,
                    pages_missing=pages_missing,
                    suppress=suppress,
                    plausible_answer_exists=(
                        _plausible_answer_exists(question, blocks, resolver, paper, by_qid)
                        or _sibling_was_answered(question, paper, set(by_qid))
                    ),
                ),
                confidence=0.5,
            )
        )

    orphans = [
        OrphanAnswer(
            block_id=block.block_id,
            text_preview=block.text[:160],
            highlight=_highlight([block]),
        )
        for block in blocks
        if block.block_id not in used_block_ids
    ]

    return MappingResult(
        mappings=mappings,
        orphans=orphans,
        unassigned_ink_ratio=unassigned_ink,
        absence_claims_suppressed=suppress,
    )


#: How much better a block must fit the question it was placed on before it stops
#: counting as writing that might belong somewhere else.
#:
#: Not a large margin, because the claim it guards is modest: this block has a
#: home, so its faint resemblance to another question is not evidence that
#: question was answered.
_SETTLED_ELSEWHERE = 1.5


def _plausible_answer_exists(
    question: Question,
    blocks: list[AnswerBlock],
    similarity: Similarity,
    paper: QuestionPaper | None = None,
    assignments: dict[str, Assignment] | None = None,
) -> bool:
    """Whether any *loose* writing on the sheet plausibly answers this question.

    Checked before claiming a question was left blank. A block may already belong
    to another question and still be this one's answer — that is precisely the
    case the unassigned-ink check cannot see, and where a false "unanswered" is
    most likely.

    But a block placed on a question it fits half again as well is not loose. It
    has a home, and its faint resemblance to this one says nothing about whether
    this one was answered. An English paper offering "any two of three" showed
    what the difference costs: the student answered two and skipped the third,
    and their answer about two brothers scored 0.305 against "the effect of
    telling the story through a child's eyes" — one thousandth over the line at
    which a pair counts as related at all. That was enough to stop the paper's own
    rule ever being asked, and a question the student was invited to skip came
    back "uncertain" rather than "not required".

    The order of the tests around this is not the mistake and is not changed:
    evidence about what the student wrote outranks inference from the paper's
    rules, and getting that backwards once reported an answered sub-part as
    optional.
    """
    owner_of: dict[str, str] = {
        block_id: qid
        for qid, assignment in (assignments or {}).items()
        for block_id in assignment.block_ids
    }
    text_of: dict[str, str] = {q.qid: q.text for q in (paper.questions if paper else [])}

    for block in blocks:
        if not block.text.strip():
            continue
        score = similarity.score(question.text, block.text)
        if score < PLAUSIBLE_ANSWER_EXISTS:
            continue

        owner = owner_of.get(block.block_id)
        settled = text_of.get(owner or "")
        if (
            owner is not None
            and owner != question.qid
            and settled
            and similarity.score(settled, block.text) >= score * _SETTLED_ELSEWHERE
        ):
            continue
        return True
    return False


def _sibling_was_answered(
    question: Question,
    paper: QuestionPaper,
    answered: set[str],
) -> bool:
    """Whether a sub-part of the same parent question was answered.

    Structural evidence, and it works where semantics does not. The remaining
    false-unanswered case was ``5 (b)``, answered "R = V / I = 10 / 2 = 5 ohm" —
    almost entirely symbols, so lexical overlap with its question is near zero and
    no similarity threshold could rescue it.

    But ``5 (a)`` was answered, and a student who answers one sub-part very
    rarely leaves the next silently blank. That makes a blank claim for ``5 (b)``
    unsafe regardless of what the text looks like.
    """
    if len(question.path) < 2:
        return False
    parent = tuple(question.path[:-1])
    for other in paper.questions:
        if other.qid == question.qid:
            continue
        if tuple(other.path[:-1]) == parent and other.qid in answered:
            return True
    return False


def _absent_status(
    question: Question,
    *,
    satisfied_sections: set[str],
    pages_missing: bool,
    suppress: bool,
    plausible_answer_exists: bool = False,
) -> AnswerStatus:
    """Decide why a question has no answer.

    The order of these tests is the product decision. Only the last of them
    asserts the student left something blank, because that is the claim a teacher
    acts on without re-reading the script — so every other explanation is
    preferred when it fits.
    """
    if plausible_answer_exists:
        # Checked first, and the ordering matters. Evidence about what the student
        # actually wrote outranks inference from the paper's rules: a question
        # whose answer appears to be on the sheet must not be filed as
        # "not required" merely because its section's quota was already met.
        # Getting this order wrong reported an answered sub-part as optional.
        return AnswerStatus.UNCERTAIN

    if question.section_id is not None and question.section_id in satisfied_sections:
        # The paper's own rules let this be skipped, and the student skipped it.
        return AnswerStatus.NOT_REQUIRED

    if pages_missing:
        return AnswerStatus.PAGES_MISSING

    if suppress:
        # Substantial writing on the sheet belongs to no block, so some answer
        # went unmapped and absence cannot honestly be claimed for anything.
        return AnswerStatus.UNCERTAIN

    return AnswerStatus.UNANSWERED


def _sections_satisfied(paper: QuestionPaper, answered: set[str]) -> set[str]:
    """Sections whose choice requirement the student has already met.

    Once a section demanding "any two" has two answers, its remaining questions
    are not omissions. Below that threshold they still are — the student owes
    answers, and reporting them as optional would hide a real gap.
    """
    satisfied: set[str] = set()
    for section in paper.sections:
        required = section.requirement.answer_any
        if required is None:
            continue
        count = sum(
            1
            for question in paper.questions
            if question.section_id == section.section_id and question.qid in answered
        )
        if count >= required:
            satisfied.add(section.section_id)
    return satisfied


def _pages_appear_missing(blocks: list[AnswerBlock], pages_uploaded: int) -> bool:
    """Whether the evidence points off the end of what was uploaded.

    A continuation marker in the last block is the student saying the answer
    carries on somewhere that is not here.
    """
    if not blocks or pages_uploaded <= 0:
        return False
    last_page = max((page for block in blocks for page in block.pages_spanned), default=0)
    trailing = [b for b in blocks if last_page in b.pages_spanned]
    return any(b.has_continuation_marker for b in trailing) and last_page >= pages_uploaded - 1


def _unassigned_ink_share(
    blocks: list[AnswerBlock],
    used_block_ids: set[str],
    ink_regions: list[InkRegion],
) -> float:
    """Share of substantive marking that no assigned block accounts for.

    Bleed-through and noise are excluded, because they appear on most scripts and
    counting them would suppress every absence claim the product exists to make.
    """
    substantive = [r for r in ink_regions if r.kind.counts_as_page_ink and r.is_substantive]
    if not substantive:
        return 0.0

    assigned_ink: set[str] = set()
    for block in blocks:
        if block.block_id in used_block_ids:
            assigned_ink.update(block.ink_region_ids)

    total = sum(r.box.area * r.ink_ratio for r in substantive)
    if total <= 0:
        return 0.0
    unassigned = sum(
        r.box.area * r.ink_ratio for r in substantive if r.region_id not in assigned_ink
    )
    return min(1.0, unassigned / total)


#: How far apart two lines may be, and how little they may share horizontally,
#: before they stop being one band of writing.
#:
#: The gap is measured in line heights, and a full one is allowed because ruled
#: paper is written on every other line: on a real script the gap between lines was
#: 0.82 of a line height, and at the old threshold of 0.35 every line of every
#: answer became a rectangle of its own. Question 16 came out as ten stripes down
#: the page, each cut to the ragged end of its line.
#:
#: The horizontal condition is what stops a band reaching across the page. It is
#: kept, loosely: writing down the left of a sheet and more down the right are two
#: regions, and one rectangle over both paints the empty middle. Measured on a page
#: of handwritten code, a single page-wide box covered 0.77 of the sheet to mark
#: 0.28 of writing.
_MERGE_GAP_SHARE = 1.2
_MERGE_OVERLAP_SHARE = 0.15


def _highlight(blocks: list[AnswerBlock]) -> Highlight | None:
    """Where the writing is: one band per region of it.

    Two shapes were tried before this one and both were wrong in opposite
    directions.

    One box per page reads cleanly and covers far too much. On a multi-line answer
    60 to 74 per cent of the rectangle was blank paper, and on a page of
    handwritten code it painted 0.77 of the sheet to mark 0.28 of writing —
    including, on a page where a student answers one question at the top and
    another at the bottom, the answer in between.

    One box per line is tight and unreadable. Question 16 of a real script came out
    as ten separate rectangles down a page of ruled paper, each cut to the ragged
    end of its own line, and a teacher said so. Worse, each was drawn round the
    recognised text, so the first started after the "I" of "In" and clipped the
    letter it was meant to mark.

    So lines that sit under one another and overlap horizontally become one band,
    and writing elsewhere on the page gets a band of its own. On ruled prose that is
    a single rectangle. On the code page it stays several, because there the
    separation is real.
    """
    boxes: list[PageBox] = [pb for block in blocks for pb in block.geometry]
    if not boxes:
        return None

    per_page: dict[int, list[BBox]] = {}
    for pb in boxes:
        per_page.setdefault(pb.page, []).append(pb.box)

    merged: list[PageBox] = []
    for page, page_boxes in sorted(per_page.items()):
        for run in _runs(page_boxes):
            merged.append(PageBox(page=page, box=BBox.union_all(run)))

    derived = "ink_regions" if all(b.is_text_free for b in blocks) else "ocr_lines"
    return Highlight(boxes=merged, derived_from=derived)


def _runs(boxes: list[BBox]) -> list[list[BBox]]:
    """Group boxes on one page into bands of writing.

    Every open run is offered the line, not merely the most recent one. Two columns
    of writing interleave when sorted by height — left, right, left, right — so
    comparing against the last run only meant each line started a run of its own and
    a two-column page came out as one rectangle per line.
    """
    runs: list[list[BBox]] = []

    for box in sorted(boxes, key=lambda b: (b.y0, b.x0)):
        for run in runs:
            if _joins(box, run):
                run.append(box)
                break
        else:
            runs.append([box])

    return runs


def _joins(box: BBox, run: list[BBox]) -> bool:
    """Whether this line continues that band."""
    bottom = max(b.y1 for b in run)
    # Measured against the taller of the two lines, so a short line does not make
    # an ordinary gap look enormous.
    height = max(box.y1 - box.y0, max(b.y1 - b.y0 for b in run))
    if box.y0 - bottom > height * _MERGE_GAP_SHARE:
        return False
    return _shares_width(box, run)


def _shares_width(box: BBox, run: list[BBox]) -> bool:
    """Whether a line lines up with the run above it well enough to join it."""
    left, right = min(b.x0 for b in run), max(b.x1 for b in run)
    overlap = min(right, box.x1) - max(left, box.x0)
    if overlap <= 0:
        return False
    # Against the narrower of the two, so a short final line still joins the
    # paragraph it belongs to, while an indented line under a long one does not.
    narrower = min(right - left, box.x1 - box.x0)
    return narrower > 0 and overlap / narrower >= _MERGE_OVERLAP_SHARE


def reassign(
    paper: QuestionPaper,
    blocks: list[AnswerBlock],
    mapping: MappingResult,
    *,
    block_id: str,
    to_qid: str,
) -> MappingResult:
    """Move one block to a question because a teacher said so.

    A teacher's correction outranks everything the aligner computed, and is
    recorded as ``teacher_override`` so a later re-run does not quietly undo it.

    The question losing the block does *not* become unanswered. The teacher moved
    an answer; that says nothing about whether the original question was
    attempted, and asserting a blank on the strength of a correction elsewhere
    would be exactly the unfounded absence claim the rest of this module works to
    avoid.
    """
    blocks_by_id = {b.block_id: b for b in blocks}
    if block_id not in blocks_by_id or all(q.qid != to_qid for q in paper.questions):
        return mapping

    position = {b.block_id: i for i, b in enumerate(blocks)}

    updated: list[Mapping] = []
    for entry in mapping.mappings:
        if entry.qid == to_qid:
            # Added to what the question already holds, not substituted for it.
            #
            # Replacing would make the commonest correction unexpressible. When an
            # answer is split across two blocks and the aligner gives one to the
            # neighbouring question, moving that block back must leave the question
            # holding both — under replace semantics the other block is displaced
            # to the orphan list, and putting it back displaces the first, so a
            # question could never hold two blocks after any manual edit.
            merged = sorted({*entry.block_ids, block_id}, key=lambda bid: position.get(bid, 0))
            owned = [blocks_by_id[bid] for bid in merged if bid in blocks_by_id]
            lines = [lid for block in owned for lid in block.line_ids]
            updated.append(
                entry.model_copy(
                    update={
                        "status": AnswerStatus.OCR_FAILED
                        if all(block.is_text_free for block in owned)
                        else AnswerStatus.ANSWERED,
                        "block_ids": merged,
                        "start_line_id": lines[0] if lines else None,
                        "end_line_id": lines[-1] if lines else None,
                        "highlight": _highlight(owned),
                        "confidence": 1.0,
                        "teacher_override": True,
                        "evidence": MatchEvidence(
                            total_score=W_LABEL,
                            signals=[MatchSignal.WRITTEN_LABEL],
                        ),
                    }
                )
            )
            continue

        if block_id in entry.block_ids:
            remaining = [bid for bid in entry.block_ids if bid != block_id]
            kept = [blocks_by_id[bid] for bid in remaining if bid in blocks_by_id]
            updated.append(
                entry.model_copy(
                    update={
                        "block_ids": remaining,
                        "highlight": _highlight(kept) if kept else None,
                        "status": entry.status if kept else AnswerStatus.UNCERTAIN,
                    }
                )
            )
            continue

        updated.append(entry)

    used = {bid for entry in updated for bid in entry.block_ids}
    orphans = [
        OrphanAnswer(
            block_id=block.block_id,
            text_preview=block.text[:160],
            highlight=_highlight([block]) or Highlight(),
        )
        for block in blocks
        if block.block_id not in used
    ]

    return mapping.model_copy(update={"mappings": updated, "orphans": orphans})
