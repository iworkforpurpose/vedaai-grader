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
from .claims import (
    Board,
    attach_continuations,
    ceilings,
    claim_decided_blocks,
    claim_written_labels,
    join_claimed_questions,
    warm_the_scorer,
)
from .highlight import _highlight
from .matrix import Assignment, _align_segment, _label_hints
from .status import (
    _absent_status,
    _pages_appear_missing,
    _plausible_answer_exists,
    _sections_satisfied,
    _sibling_was_answered,
    _unassigned_ink_share,
)
from .tuning import UNASSIGNED_INK_SUPPRESSES, W_LABEL


def align(
    paper: QuestionPaper,
    blocks: list[AnswerBlock],
    anchors: list[Anchor],
    *,
    similarity: Similarity | None = None,
) -> list[Assignment]:
    """Assign blocks to questions.

    Direct evidence is honoured first and unconditionally, in any order, and only
    what is left goes to the dynamic program. That order of authority is the part
    to preserve: a confirmed anchor is the student stating which question this
    answers, while monotonicity is a convenience of the recurrence. Treating
    anchors as pins for the program keeps only the longest monotone subset of
    them, which on a fully reversed script turned eight correctly-labelled
    answers into orphans.

    The claiming passes run in order and each narrows what the next can see. They
    live in `claims`, and `Board` is the state they build up together.
    """
    similarity = similarity or default_similarity

    # Stems are excluded from candidacy. "2. Answer the following:" is a heading
    # with no marks and no answer of its own, and leaving it in the candidate
    # list lets it absorb the answer to its own sub-part - which costs two
    # mappings rather than one, since the sub-part then reads as unanswered.
    questions = [q for q in paper.in_print_order() if not q.is_stem]
    if not questions or not blocks:
        return []

    warm_the_scorer(similarity, questions, blocks)
    block_best, block_runner_up = ceilings(questions, blocks, similarity)

    board = Board()
    order = {block.block_id: i for i, block in enumerate(blocks)}
    block_of_line = {
        line_id: block.block_id for block in blocks for line_id in block.line_ids
    }
    known_qids = {q.qid for q in questions}

    claim_written_labels(board, anchors, block_of_line, known_qids)
    claim_decided_blocks(board, questions, blocks, similarity, block_best, order)
    labelled_blocks = attach_continuations(
        board,
        blocks,
        questions,
        anchors,
        similarity,
        block_best,
        block_runner_up,
        block_of_line,
        known_qids,
    )
    join_claimed_questions(board, blocks, questions, similarity, labelled_blocks, order)

    # Whatever nothing has claimed goes to the monotone recurrence, which is a
    # reasonable prior for unlabelled answers because they are usually written in
    # the order they were asked.
    board.assignments.extend(
        _align_segment(
            [q for q in questions if q.qid not in board.claimed_qids],
            [b for b in blocks if b.block_id not in board.claimed_blocks],
            similarity=similarity,
            label_hints=_label_hints(anchors, block_of_line, known_qids),
            block_best=block_best,
        )
    )
    return board.assignments


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
            # nothing to be absent - and the absence logic below would otherwise
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
            # holding both - under replace semantics the other block is displaced
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
