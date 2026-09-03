"""Question numbers the student wrote, and whether to believe them.

A label in the margin is the strongest mapping signal available: the student is
stating outright which question this answers. It is also the signal most capable
of silently doing damage, because a wrongly trusted label maps an answer to the
wrong question *and* reports high confidence while doing it. The teacher sees a
confident wrong answer, which is worse than a hedged one.

So every anchor starts as a hypothesis and must earn confirmation from evidence
that does not come from the label itself. Two routes, and either suffices:

**Semantic agreement.** Does the writing beside the label actually discuss what
that question asked? Strong evidence when present.

**Order consistency.** Does this label's position among the other labels make
sense? A student answering 1, 2, 3, 4 in order corroborates every one of them.

Both are needed because each fails alone. Semantic agreement is weak on short
answers and on answers that reuse none of the question's vocabulary — "Define
refraction" answered by "the bending of light" shares no content word. Order
consistency cannot condemn anything on its own, because answering out of order is
explicitly permitted and common.

An anchor that fails both is *disputed*, which does not discard it: it stops
pinning the alignment and becomes one weighted signal among several. That is the
defence against a mislabelled answer, and it degrades rather than breaking.
"""

from __future__ import annotations

from dataclasses import dataclass

from vedaai_contracts import Anchor, AnchorStatus, AnswerBlock, Line, Question

from ..questions.numbering import detect_section_prefixes, parse_label
from .similarity import Similarity, default_similarity

#: How far above a scorer's own unrelated floor an agreement must sit before it
#: confirms an anchor, as a share of the range that remains.
#:
#: A share rather than a number, because the number was wrong on both scales at
#: once and in opposite directions.
#:
#: It was 0.18 absolute. Against embeddings, whose own unrelated floor is 0.30,
#: that confirmed pairs the model had classified as unrelated — inside the
#: 0.148-0.154 band it was calibrated on — and a confirmed anchor pins the
#: alignment with ``W_LABEL = 3.0``. Confirmation was a weaker claim than the
#: aligner's own bar for two texts being related at all.
#:
#: Against trigrams, whose floor is zero and which score nonzero on any two
#: English texts, 0.18 is close to noise on prose and unreachable on a symbolic
#: answer like ``R=V/I``. The same constant was simultaneously too permissive and
#: too strict.
#:
#: Low by the standards of sentence similarity, and still deliberately so: lexical
#: overlap between a question and its answer is modest even when they plainly
#: correspond, so a high bar refuses correct labels, and the cost of that is
#: losing a reliable signal rather than gaining safety.
_CONFIRM_SHARE_OF_RANGE = 0.25

#: How much better another question must score before the written label is
#: treated as contradicted.
#:
#: Comparative rather than absolute, because an absolute floor cannot support
#: this judgement. On the out-of-order golden case, "Explain the working of an
#: electric motor" is answered with "a current-carrying coil in a magnetic
#: field" — a correct answer sharing no content word with its question, scoring
#: zero. An absolute threshold disputed that correct label and threw away a good
#: anchor.
#:
#: What does carry meaning is a *rival*: if the writing matches some other
#: question markedly better than the one the student named, the label is
#: suspect. If nothing matches well, the measure is simply uninformative and
#: says nothing either way.
#: Expressed as a share of the usable range for the same reason as above. It was
#: 0.12 absolute, on a scale where word overlap "returns exactly zero for every
#: pairing" on real handwriting and trigram spreads between questions are
#: hundredths. A 0.12 gap essentially never occurred, so ``_outscored_by_a_rival``
#: returned False for practically every anchor and ``_decide``'s only route to
#: DISPUTED became dead code — the mislabelled-answer defence this module was
#: written for had silently stopped existing.
_RIVAL_SHARE_OF_RANGE = 0.15


def _band(similarity: Similarity) -> tuple[float, float]:
    """The range of scores this measure actually uses: noise floor to match floor.

    Not floor-to-one. A measure's ceiling is where *it* puts a genuine match, and
    the two scorers here disagree by a factor of nearly three about where that is —
    embeddings at 0.55, trigrams at 0.30. Deriving a threshold from ``1 - floor``
    is right for one and three times too wide for the other, which is exactly how
    an absolute margin of 0.12 came to be unreachable on the surface scorers.
    """
    floor = float(getattr(similarity, "unrelated_below", 0.0) or 0.0)
    confident = float(getattr(similarity, "confident_above", 1.0) or 1.0)
    return floor, max(floor, confident) - floor


def semantic_confirms(agreement: float, similarity: Similarity) -> bool:
    """Whether an agreement is strong enough to confirm a written label."""
    floor, band = _band(similarity)
    return agreement >= floor + band * _CONFIRM_SHARE_OF_RANGE


def rival_margin(similarity: Similarity) -> float:
    """How much better a rival question must score to contradict a label.

    Always smaller than the band the measure works in, so it is reachable on every
    scale. An absolute 0.12 was larger than the whole trigram spread between two
    questions, so nothing ever cleared it.
    """
    _floor, band = _band(similarity)
    return band * _RIVAL_SHARE_OF_RANGE

#: A block shorter than this carries too little text for similarity to mean
#: anything. "The watt." is a correct answer and an unusable embedding.
_MIN_TEXT_FOR_SEMANTICS = 24


#: Sentinel for "any section", so that filtering on ``section_id=None`` — a real
#: value, meaning a question outside every section — stays distinguishable from
#: not filtering at all.
_ANY_SECTION = object()


@dataclass
class _Candidate:
    block: AnswerBlock
    line: Line
    tokens: tuple[str, ...]
    raw: str


def detect(
    blocks: list[AnswerBlock],
    lines: list[Line],
    questions: list[Question],
    *,
    similarity: Similarity | None = None,
) -> list[Anchor]:
    """Find written question labels and decide how far each can be trusted."""
    similarity = similarity or default_similarity
    by_id = {line.line_id: line for line in lines}

    # The paper's own numbering styles, so a written label can be read the way the
    # paper writes them. A section that prefixes its numbers -- "T1".."T5" beside
    # "Q1".."Q4" -- is unreadable without them: `parse_label("T2")` returns None,
    # so the student's own question number is discarded and the block is placed on
    # whatever its wording happens to favour. On a real script that put a
    # congruence proof onto a question the student never attempted, and marked it.
    prefixes = detect_section_prefixes([question.label_raw for question in questions])

    candidates: list[_Candidate] = []
    for block in blocks:
        if not block.line_ids:
            continue
        first = by_id.get(block.line_ids[0])
        if first is None:
            continue
        parsed = parse_label(first.text, prefixes=prefixes)
        if parsed is None:
            continue
        candidates.append(
            _Candidate(block=block, line=first, tokens=parsed.tokens, raw=parsed.raw)
        )

    resolved = _resolve(candidates, questions)
    order_ok = _order_consistent([qid for _c, qid, _ambiguous in resolved])

    anchors: list[Anchor] = []
    for index, (candidate, qid, ambiguous) in enumerate(resolved):
        question = next((q for q in questions if q.qid == qid), None)
        agreement = _semantic_agreement(candidate.block, question, similarity)
        consistent = order_ok[index]
        outscored = _outscored_by_a_rival(candidate.block, qid, questions, similarity)
        status = _decide(
            qid, agreement, consistent, outscored, similarity, ambiguous=ambiguous
        )

        anchors.append(
            Anchor(
                anchor_id=f"anc:{index:03d}",
                claimed_label=candidate.raw,
                claimed_qid=qid,
                line_id=candidate.line.line_id,
                page=candidate.line.page,
                box=candidate.line.box,
                status=status,
                semantic_agreement=agreement,
                order_consistent=consistent,
            )
        )

    return anchors


def _label_key(text: str) -> str:
    """A label reduced to the characters that identify it.

    Alphanumerics only, so ``Q4.`` meets ``Q4`` and ``2 a)`` meets ``2 (a)``:
    students and typesetters punctuate differently and neither is saying anything
    by it.
    """
    return "".join(character for character in text.lower() if character.isalnum())


def _resolve(
    candidates: list[_Candidate],
    questions: list[Question],
) -> list[tuple[_Candidate, str | None, bool]]:
    """Match each written label to a question in the paper.

    Two indexes, and the order between them is the whole point.

    **The label the paper printed is tried first.** A student writing ``Q4.``
    where the paper printed ``Q4`` has named the question outright, and the
    printed label carries the section — which ``path`` does not, because the
    section lives in the qid instead.

    **The token path is tried second, and may be ambiguous.** A paper whose
    numbering restarts per section holds two questions whose path is ``("2",)``,
    so this index cannot be single-valued. It was, once, and the later section
    silently overwrote the earlier: on a mathematics paper with a ``Q`` section
    and a ``T`` section that left every one of Q1-Q4 unreachable by any written
    label, and a student's own correct ``Q4.`` resolved to ``T/4``, was confirmed
    against it, and pinned the alignment onto a question they never attempted.
    Their correct label was the thing that moved their answer.

    A lone sub-part label such as ``(b)`` is resolved against the most recent
    resolved parent, which is how the student meant it: having written ``11 (a)``
    above, they see no need to repeat the 11. That lookup is confined to the
    parent's own section, since the parent is what says which section this is.

    A label matching more than one question resolves to None and is reported
    ambiguous — which is a different fact from naming no question at all, and is
    why the two are distinguished rather than both collapsing to None. The
    section the student was last writing in would often settle such a label, and
    is deliberately *not* used for it: on the paper above, the student's own list
    item ``2.`` inside their Q3 working would then resolve to Q2 and report an
    untouched question as answered. A hedge costs a teacher a second look; a
    confident wrong answer costs them the mark.
    """
    by_path: dict[tuple[str, ...], list[Question]] = {}
    by_label: dict[str, list[Question]] = {}
    for question in questions:
        by_path.setdefault(tuple(question.path), []).append(question)
        by_label.setdefault(_label_key(question.label_raw), []).append(question)

    def sole(matches: list[Question], *, section: str | None = _ANY_SECTION) -> Question | None:
        if section is not _ANY_SECTION:
            matches = [q for q in matches if q.section_id == section]
        return matches[0] if len(matches) == 1 else None

    out: list[tuple[_Candidate, str | None, bool]] = []
    recent_parent: tuple[str, ...] = ()
    recent_section: str | None = None

    for candidate in candidates:
        by_this_label = by_label.get(_label_key(candidate.raw), [])
        by_this_path = by_path.get(candidate.tokens, [])

        found = sole(by_this_label) or sole(by_this_path)

        if found is None and len(candidate.tokens) == 1 and recent_parent:
            for depth in range(len(recent_parent), 0, -1):
                combined = recent_parent[:depth] + candidate.tokens
                found = sole(by_path.get(combined, []), section=recent_section)
                if found is not None:
                    break

        if found is None:
            # Ambiguous only if something matched and more than one thing did.
            # A label naming nothing on the paper is a different report.
            ambiguous = len(by_this_label) > 1 or len(by_this_path) > 1
            out.append((candidate, None, ambiguous))
            continue

        out.append((candidate, found.qid, False))
        # A multi-token label names its own parent; a single-token one *is* the
        # parent for anything that follows beneath it.
        path = tuple(found.path)
        recent_parent = path[:-1] if len(path) > 1 else path
        recent_section = found.section_id
    return out


def _semantic_agreement(
    block: AnswerBlock,
    question: Question | None,
    similarity: Similarity,
) -> float | None:
    """How well the writing matches what the question asked.

    Returns None when the block is too short to judge, which is a real and
    frequent case — a one-word answer is correct and unmeasurable. None means "no
    evidence", never "no agreement", and the caller must not treat the two alike.
    """
    if question is None:
        return None
    if len(block.text.strip()) < _MIN_TEXT_FOR_SEMANTICS:
        return None
    return similarity.score(question.text, block.text)


def _outscored_by_a_rival(
    block: AnswerBlock,
    qid: str | None,
    questions: list[Question],
    similarity: Similarity,
) -> bool:
    """Whether some other question matches this writing markedly better.

    This is the evidence that can actually contradict a written label. A low
    absolute score means the measure is uninformative; a *rival* scoring well
    above the claimed question means the student probably wrote the wrong number.
    """
    if qid is None or len(block.text.strip()) < _MIN_TEXT_FOR_SEMANTICS:
        return False

    claimed = next((q for q in questions if q.qid == qid), None)
    if claimed is None:
        return False

    claimed_score = similarity.score(claimed.text, block.text)
    margin = rival_margin(similarity)
    for question in questions:
        if question.qid == qid:
            continue
        if similarity.score(question.text, block.text) - claimed_score >= margin:
            return True
    return False


def _order_consistent(qids: list[str | None]) -> list[bool]:
    """Which anchors sit in the longest run that respects the paper's order.

    Anchors are in document order; their claimed questions have a printed order.
    The longest increasing subsequence is the largest set of labels that could all
    be right if the student worked through the paper forwards — so membership is
    corroboration, and exclusion is a reason to look closer.

    Exclusion is emphatically not proof of error. Answering out of order is
    permitted, so a correctly-labelled answer written early can fall outside the
    run. That is why this only ever confirms, and never condemns.
    """
    positions = [_position(qid) for qid in qids]
    known = [(i, p) for i, p in enumerate(positions) if p is not None]
    if len(known) < 2:
        # Nothing to be consistent with. Treated as uncorroborated rather than
        # inconsistent, so a single anchor is not condemned for being alone.
        return [False] * len(qids)

    values = [p for _i, p in known]
    keep = _longest_increasing(values)
    consistent = [False] * len(qids)
    for offset in keep:
        consistent[known[offset][0]] = True
    return consistent


def _position(qid: str | None) -> int | None:
    """A sortable position for a qid, from the numeric parts of its path."""
    if qid is None:
        return None
    parts = qid.split("/")
    scale = 1000
    total = 0
    for depth, part in enumerate(parts):
        if part.isdigit():
            total += int(part) * (scale ** (3 - min(depth, 3)))
    return total


def _longest_increasing(values: list[int]) -> list[int]:
    """Indices of a longest non-decreasing subsequence."""
    if not values:
        return []

    best_length = [1] * len(values)
    predecessor = [-1] * len(values)

    for i in range(1, len(values)):
        for j in range(i):
            if values[j] <= values[i] and best_length[j] + 1 > best_length[i]:
                best_length[i] = best_length[j] + 1
                predecessor[i] = j

    end = max(range(len(values)), key=lambda i: best_length[i])
    chain: list[int] = []
    while end != -1:
        chain.append(end)
        end = predecessor[end]
    return list(reversed(chain))


def _decide(
    qid: str | None,
    agreement: float | None,
    consistent: bool,
    outscored_by_a_rival: bool,
    similarity: Similarity,
    *,
    ambiguous: bool = False,
) -> AnchorStatus:
    """Decide how far one anchor can be trusted."""
    if qid is None:
        if ambiguous:
            # The label names a question, and more than one of them: the paper
            # numbers two sections alike and the student wrote no section letter.
            # Nothing here is contradicted, so this is an absence of evidence and
            # not evidence against — and the teacher is told the difference,
            # since "this number does not match the writing beside it" would be
            # a false statement about a label that may well be right.
            return AnchorStatus.UNVERIFIED
        # The label names a question the paper does not contain. Strong evidence
        # the student mislabelled, and nothing to pin an alignment to regardless.
        return AnchorStatus.DISPUTED

    if outscored_by_a_rival and not consistent:
        # The writing matches a different question better, and the label's
        # position offers no support. This is the mislabelled case.
        return AnchorStatus.DISPUTED

    if agreement is not None and semantic_confirms(agreement, similarity):
        return AnchorStatus.CONFIRMED

    if consistent:
        return AnchorStatus.CONFIRMED

    # No corroboration, but nothing against it either. Not trusted to pin an
    # alignment, and not condemned — a correct label can land here simply because
    # its answer reuses none of the question's words.
    return AnchorStatus.UNVERIFIED


def confirmed(anchors: list[Anchor]) -> list[Anchor]:
    """Anchors trusted enough to constrain the alignment."""
    return [anchor for anchor in anchors if anchor.may_pin]
