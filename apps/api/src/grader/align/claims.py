"""The passes that claim blocks before the dynamic program runs.

Each pass answers a question the recurrence cannot: who wrote a label, whose
meaning names exactly one question, what carries on from the answer above, and
what belongs to a question already claimed. They run in this order and the order
matters - the dynamic program only sees what is left, so a block claimed here is
a block it can no longer misplace.

They share the state they are building up, which is what `Board` is for. Passing
three mutable collections between six functions is how a claim gets recorded in
one place and forgotten in another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vedaai_contracts import (
    Anchor,
    AnswerBlock,
    MatchEvidence,
    MatchSignal,
    Question,
)

from ..answers.similarity import Similarity
from .matrix import Assignment, _decided_pairs, _label_hints
from .signals import _runs_on_from, _semantic
from .tuning import _PREFERENCE_MARGIN, SETTLE_RATIO, W_LABEL


@dataclass
class Board:
    """What has been claimed so far, as the passes build it up.

    A question or a block may be claimed once. Both sets exist because the two
    constraints are different: a question can hold several blocks (an answer
    running over a page break), but a block belongs to exactly one answer.
    """

    assignments: list[Assignment] = field(default_factory=list)
    claimed_qids: set[str] = field(default_factory=set)
    claimed_blocks: set[str] = field(default_factory=set)

    def owner_of(self, block_id: str) -> Assignment | None:
        return next(
            (a for a in self.assignments if block_id in a.block_ids),
            None,
        )


def warm_the_scorer(
    similarity: Similarity, questions: list[Question], blocks: list[AnswerBlock]
) -> None:
    """Give the scorer everything it will be asked about, in one go.

    Scoring is every question against every block, so a scorer that fetches per
    pair makes hundreds of calls for one submission and one told the whole set up
    front makes one. Scorers with nothing to prepare do not implement this.
    """
    warm = getattr(similarity, "warm", None)
    if callable(warm):
        warm([q.text for q in questions] + [b.text for b in blocks])


def ceilings(
    questions: list[Question], blocks: list[AnswerBlock], similarity: Similarity
) -> tuple[dict[str, float], dict[str, float | None]]:
    """Each block's best and second-best score against the whole paper.

    Fixed before anything is claimed, so that what counts as "settling" does not
    change as questions are taken.

    The runner-up is not a lesser version of the best. It measures whether a block
    *prefers* somewhere, which is a different question from how well it scores
    anywhere - and on damaged handwriting it is the only one of the two still
    carrying signal.
    """
    scored = {
        block.block_id: sorted(
            (similarity.score(question.text, block.text) for question in questions),
            reverse=True,
        )
        for block in blocks
    }
    best = {bid: (s[0] if s else 0.0) for bid, s in scored.items()}
    runner_up = {bid: (s[1] if len(s) > 1 else None) for bid, s in scored.items()}
    return best, runner_up


def claim_written_labels(
    board: Board,
    anchors: list[Anchor],
    block_of_line: dict[str, str],
    known_qids: set[str],
) -> None:
    """Honour confirmed labels directly, in any order.

    A confirmed anchor is not a hint about order; it is the student stating which
    question this answers, corroborated by evidence that did not come from the
    label. Order is irrelevant once that is established, exactly as it is for a
    marker flipping to the labelled page.
    """
    for anchor in anchors:
        if not anchor.may_pin or anchor.claimed_qid is None:
            continue
        if anchor.claimed_qid in board.claimed_qids or anchor.claimed_qid not in known_qids:
            continue
        block_id = block_of_line.get(anchor.line_id)
        if block_id is None or block_id in board.claimed_blocks:
            continue

        board.claimed_qids.add(anchor.claimed_qid)
        board.claimed_blocks.add(block_id)
        board.assignments.append(
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


def claim_decided_blocks(
    board: Board,
    questions: list[Question],
    blocks: list[AnswerBlock],
    similarity: Similarity,
    block_best: dict[str, float],
    order: dict[str, int],
) -> None:
    """Honour blocks whose own content narrows them to exactly one question.

    That is the same kind of evidence as a label the student wrote, and it must
    not be subjected to the monotonicity the recurrence needs: two answers written
    in the order 2 then 1 are both narrowed correctly, and requiring a monotone
    path drops the second. Answering out of order is ordinary student behaviour.
    """
    open_questions = [q for q in questions if q.qid not in board.claimed_qids]
    open_blocks = [b for b in blocks if b.block_id not in board.claimed_blocks]
    decided_by_qid: dict[str, Assignment] = {}

    for qid, block_id, agreement in _decided_pairs(
        open_questions, open_blocks, similarity, block_best
    ):
        if block_id in board.claimed_blocks:
            continue

        # A second block decided on the same question is the rest of that answer,
        # not a competitor for it. Dropping it is what a page-spanning answer
        # looks like from here: claiming the strongest block per question leaves
        # the other half of a two-page answer placed nowhere.
        existing = decided_by_qid.get(qid)
        if existing is not None:
            existing.block_ids.append(block_id)
            board.claimed_blocks.add(block_id)
            continue

        if qid in board.claimed_qids:
            continue

        board.claimed_qids.add(qid)
        board.claimed_blocks.add(block_id)
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
        board.assignments.append(assignment)

    # Document order within each answer, so `start_line_id` and `end_line_id`
    # still name the span a reader would follow. The pairs arrive strongest-first,
    # which is right for deciding and wrong for reading.
    for assignment in decided_by_qid.values():
        assignment.block_ids.sort(key=lambda bid: order.get(bid, 0))


def attach_continuations(
    board: Board,
    blocks: list[AnswerBlock],
    questions: list[Question],
    anchors: list[Anchor],
    similarity: Similarity,
    block_best: dict[str, float],
    block_runner_up: dict[str, float | None],
    block_of_line: dict[str, str],
    known_qids: set[str],
) -> set[str]:
    """Attach a block that carries on from the answer directly above it.

    An answer written down a page with a paragraph break in it is one answer, and
    the segmenter reads the break as a boundary. Once the question is claimed by
    the first half it leaves the alignment entirely, so the tail has nothing left
    to compete for and lands on whatever is still free.

    Settled before the rest runs, and on the same terms the rest uses. Returns the
    blocks carrying a written label, which the later passes must also leave alone.
    """
    hints = _label_hints(anchors, block_of_line, known_qids)
    labelled_blocks = {block_id for _qid, block_id in hints}
    # Which assignment currently owns each block. Kept as a local map rather than
    # searched per block, and updated as blocks are attached so that a run of
    # three carries through rather than only two.
    owner_of_block: dict[str, Assignment] = {
        block_id: assignment
        for assignment in board.assignments
        for block_id in assignment.block_ids
    }

    question_by_qid = {question.qid: question for question in questions}

    for position, block in enumerate(blocks):
        if position == 0 or block.block_id in board.claimed_blocks:
            continue
        if block.block_id in labelled_blocks:
            # The student wrote a question number beside it. Their statement of
            # intent outranks what the block above happens to be.
            continue
        if block.is_text_free or not block.text.strip():
            # A region with no readable text has nothing to be the rest of. It may
            # be a diagram, and `_text_free_match_is_plausible` is where that is
            # decided, on the two honest reasons - a drawing question, or the
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
        # The real tail clears this comfortably - 0.444 against the question it
        # belongs to, where 0.30 is the line below which a pair counts as
        # unrelated. What it does cost is a four-word fragment scoring zero
        # against its own answer, which stays an orphan. That is the better trade:
        # a highlight four words short is a smaller error than stray writing
        # presented to a teacher as part of an answer.
        fit = _semantic(question, block, similarity)
        floor = float(getattr(similarity, "unrelated_below", 0.0) or 0.0)
        best = block_best.get(block.block_id, 0.0)
        # Below the floor, join only where the block plainly *prefers* this
        # question over every other on the paper.
        #
        # The floor catches writing that answers nothing - rough work, a note to
        # the marker - and it has to, because such writing has no meaningful best
        # and the noise will say yes to whatever sits above it. But it is an
        # absolute threshold calibrated on prose: the note above quotes a real
        # tail at 0.444, and a mathematics tail cannot reach it. On a real script
        # the tail carrying "Value of a note book = 26, Value of a Pen = 20"
        # scored 0.1555 against its own question - its best on that paper by a
        # factor of 3.5, and nowhere near 0.30. Both repair paths refused it on
        # the floor, and the fallback that places whatever is left applies no
        # floor at all, so it went to a question about a father and his son's
        # coins. The answer it belonged to was then marked zero for having nothing
        # left on it.
        #
        # So preference stands in for the floor where the floor cannot reach:
        # this question is the block's best, and beats the runner-up by a clear
        # multiple. Rough work does not concentrate like that - it scores its
        # small score against everything.
        #
        # With a single question on the paper there is no runner-up and no
        # preference to measure, so the floor governs alone. That is the case the
        # golden set covers, and relaxing it there would let rough work in.
        runner_up = block_runner_up.get(block.block_id)
        prefers_here = (
            fit >= best
            and runner_up is not None
            and fit >= runner_up * _PREFERENCE_MARGIN
        )
        if floor > 0.0 and fit < floor and not prefers_here:
            continue
        if best > 0.0 and fit < best * SETTLE_RATIO:
            # It prefers somewhere else clearly enough that this is a new answer,
            # not the rest of the one above.
            continue
        above.block_ids.append(block.block_id)
        board.claimed_blocks.add(block.block_id)
        # Recorded so a run of three blocks carries through, not only two.
        owner_of_block[block.block_id] = above
    return labelled_blocks


def join_claimed_questions(
    board: Board,
    blocks: list[AnswerBlock],
    questions: list[Question],
    similarity: Similarity,
    labelled_blocks: set[str],
    order: dict[str, int],
) -> None:
    """Let a block join the question it prefers, even once that is claimed.

    Being taken should make a question something to join rather than something to
    remove. A student answering one question on two pages produces two blocks that
    both prefer it clearly; the stronger takes it, and under removal the other
    settles on an unrelated question and highlights the wrong answer there.

    This is the statement `claim_decided_blocks` already makes for blocks the
    evidence has narrowed, applied to a block that prefers a question without
    having been narrowed to it.
    """
    blocks_by_id = {block.block_id: block for block in blocks}

    # A block whose own question has already been taken joins it, rather than
    # settling for whatever is left.
    #
    # A student answered question 1 on page one and again on page two. Both blocks
    # prefer question 1 and prefer it clearly - 0.626 and 0.656, against 0.493 and
    # 0.351 for the next best. The stronger one took it, and a claimed question
    # leaves the alignment entirely, so the page-one block came back to a field
    # with its own answer missing from it and settled on "State whether a python
    # is a specialist or a generalist", worth one mark. Clicking that question lit
    # up the whole pandas paragraph, which is what a teacher reported.
    #
    # Nothing in the scoring was wrong. What was wrong is that being taken removed
    # question 1 instead of making it something to join - and a few lines above,
    # for blocks the evidence has *narrowed* to one question, this module already
    # says the opposite: a second block on the same question is the rest of that
    # answer, not a competitor for it. This is the same statement for a block that
    # prefers a question without having been narrowed to it.
    for block in blocks:
        if block.block_id in board.claimed_blocks or block.block_id in labelled_blocks:
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
        if top.qid not in board.claimed_qids or (floor > 0.0 and top_score < floor):
            continue

        owner = next((a for a in board.assignments if a.qid == top.qid), None)
        if owner is None:
            continue

        # Only where it answers that question about as well as the writing already
        # placed on it. Two halves of one answer look alike to the scorer - 0.626
        # and 0.656 here - and rough work sitting on the same paper does not.
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
        board.claimed_blocks.add(block.block_id)
        owner.block_ids.sort(key=lambda bid: order.get(bid, 0))
