"""How well one question matches one block.

Each function answers a narrow question about a candidate pairing and returns a
number. Nothing here decides an assignment - they only supply evidence, which is
what makes them testable one at a time. Anything needing the state of the
alignment so far lives in `matrix` instead.
"""

from __future__ import annotations

from vedaai_contracts import (
    AnswerBlock,
    Question,
)

from ..answers.similarity import Similarity
from ..questions.expects import expects_a_drawing
from .tuning import (
    _TAIL_MAX_GAP,
    _TAIL_PAGE_TOP,
    SHARE_LENGTH_BONUS,
    SHARE_LENGTH_FACTOR,
    SHARE_LENGTH_PENALTY,
)


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
