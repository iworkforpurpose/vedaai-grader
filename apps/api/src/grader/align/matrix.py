"""The monotone dynamic program that assigns blocks to questions.

Answers written without a usable label are assumed to appear in the order the
questions were asked, which turns assignment into a shortest-path problem over a
grid of (question, block) prefixes. Every structure the product needs falls out
of the recurrence rather than needing a rule of its own:

* an unanswered question is a gap on the question axis
* an orphan answer is a gap on the block axis
* a page-spanning answer is one question consuming several blocks
* merged sub-parts are one block serving several questions

Gap penalties are asymmetric on purpose. Skipping a question is cheap, because
unanswered questions are ordinary; emitting an orphan is dearer, because an
orphan is more often our own segmentation splitting one answer in two than a
student writing something extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import inf

from vedaai_contracts import (
    Anchor,
    AnchorStatus,
    AnswerBlock,
    MatchEvidence,
    MatchSignal,
    Question,
)

from ..answers.similarity import Similarity
from .signals import (
    _is_part_of_the_same_question,
    _length_plausibility,
    _may_continue_into,
    _may_share,
    _order_prior,
    _semantic,
    _share_support,
    _text_free_match_is_plausible,
)
from .tuning import (
    _FRAGMENT_MAX_CHARS,
    CONTINUE_BASE,
    DECISIVE_MARGIN,
    MATCH_MINIMUM,
    SETTLE_RATIO,
    SHARE_BASE,
    SKIP_BLOCK,
    SKIP_QUESTION,
    W_LABEL_DISPUTED,
    W_LABEL_UNCONFIRMED,
    W_LENGTH,
    W_ORDER,
    W_SEMANTIC,
)


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
