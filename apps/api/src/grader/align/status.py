"""What it means when a question has no answer.

The four absence states are not interchangeable and must never collapse into a
single threshold: `unanswered` says the student left it blank, `ocr_failed` says
we could not read what they wrote, `not_required` says they were not asked to,
and `pages_missing` says the upload is incomplete. A false `unanswered` is the
worst error this product makes, so everything here is biased toward doubt.
"""

from __future__ import annotations

from vedaai_contracts import (
    AnswerBlock,
    AnswerStatus,
    InkRegion,
    Question,
    QuestionPaper,
)

from ..answers.similarity import Similarity
from .matrix import Assignment
from .tuning import _SETTLED_ELSEWHERE, PLAUSIBLE_SHARE_OF_BAND


def _plausible_threshold(similarity) -> float:
    """The score above which a block plausibly answers something.

    Read from the measure, because the measures disagree by a factor of four
    about where their noise stops. See ``PLAUSIBLE_SHARE_OF_BAND``.
    """
    floor = float(getattr(similarity, "unrelated_below", 0.0) or 0.0)
    confident = float(getattr(similarity, "confident_above", 1.0) or 1.0)
    return floor + max(0.0, confident - floor) * PLAUSIBLE_SHARE_OF_BAND


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
        if score < _plausible_threshold(similarity):
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
