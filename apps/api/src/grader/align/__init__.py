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

The order of authority is the part worth guarding. Treating confirmed anchors as
*pins* for the dynamic program requires them to be monotone, which on a fully
reversed script turned eight correctly-labelled answers into orphans and left 84%
of the sheet unassigned. Monotonicity is a convenience of the algorithm and must
never override direct evidence.

This module is the public surface. `align` assigns blocks to questions, `resolve`
decides what each question's outcome means, and `reassign` applies a teacher's
correction. The parts live next door:

* `tuning`    every weight and threshold, with the reason it holds its value
* `signals`   how well one question matches one block
* `matrix`    the dynamic program itself
* `status`    what an absence means, and which of the four kinds it is
* `highlight` turning a run of lines into the region a teacher sees
"""

from __future__ import annotations

from vedaai_contracts import (
    Anchor,
    AnswerBlock,
    AnswerStatus,
    Highlight,
    InkRegion,
    Mapping,
    MappingResult,
    MatchEvidence,
    MatchSignal,
    OrphanAnswer,
    QuestionPaper,
)

from ..answers.similarity import Similarity, default_similarity
from .highlight import _highlight
from .matrix import Assignment, _align_segment, _decided_pairs, _label_hints
from .signals import _runs_on_from, _semantic
from .status import (
    _absent_status,
    _pages_appear_missing,
    _plausible_answer_exists,
    _sections_satisfied,
    _sibling_was_answered,
    _unassigned_ink_share,
)
from .tuning import _PREFERENCE_MARGIN, SETTLE_RATIO, UNASSIGNED_INK_SUPPRESSES, W_LABEL


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
    block_scores = {
        block.block_id: sorted(
            (similarity.score(question.text, block.text) for question in questions),
            reverse=True,
        )
        for block in blocks
    }
    block_best = {
        block_id: (scores[0] if scores else 0.0)
        for block_id, scores in block_scores.items()
    }
    #: The second-best question for each block, or None when the paper holds only
    #: one. It measures whether a block *prefers* somewhere, which is a different
    #: question from how well it scores anywhere — and on damaged handwriting the
    #: only one of the two that still carries signal.
    block_runner_up = {
        block_id: (scores[1] if len(scores) > 1 else None)
        for block_id, scores in block_scores.items()
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
        best = block_best.get(block.block_id, 0.0)
        # Below the floor, join only where the block plainly *prefers* this
        # question over every other on the paper.
        #
        # The floor catches writing that answers nothing — rough work, a note to
        # the marker — and it has to, because such writing has no meaningful best
        # and the noise will say yes to whatever sits above it. But it is an
        # absolute threshold calibrated on prose: the note above quotes a real
        # tail at 0.444, and a mathematics tail cannot reach it. On a real script
        # the tail carrying "Value of a note book = 26, Value of a Pen = 20"
        # scored 0.1555 against its own question — its best on that paper by a
        # factor of 3.5, and nowhere near 0.30. Both repair paths refused it on
        # the floor, and the fallback that places whatever is left applies no
        # floor at all, so it went to a question about a father and his son's
        # coins. The answer it belonged to was then marked zero for having nothing
        # left on it.
        #
        # So preference stands in for the floor where the floor cannot reach:
        # this question is the block's best, and beats the runner-up by a clear
        # multiple. Rough work does not concentrate like that — it scores its
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


# `default_similarity` is re-exported here rather than reached through a
# submodule so that `grader.align.default_similarity` remains the single place to
# patch it. The eval harness and the alignment tests both pin the scorer there,
# and a submodule-local import would silently ignore them.
__all__ = ["align", "resolve", "reassign", "Assignment", "Similarity", "default_similarity"]
