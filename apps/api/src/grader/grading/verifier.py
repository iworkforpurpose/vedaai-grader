"""Answering a bank of binary checks by entailment rather than by generation.

This is the payoff of a decision this project already made and did not collect on.

Marking here is not "read this answer and judge it". `scheme.py` turns a question
into atomic yes/no checks — *"Does the answer state there is no potential
difference between the bird's feet?"* — and the marker answers each one against
the student's text. That is **entailment**: does this text assert this claim? The
subject knowledge lives in the check, which a capable model wrote once for the
whole class; answering it needs no physics, only reading.

The fact-checking literature is direct about what decomposition buys, and it is
the opposite of the intuition that a harder task needs a bigger model:
decomposition *benefits weaker verifiers* and does little for strong ones, because
without it a single judgement has to weigh several sub-claims at once and the
supported ones drown the contradicted one. Having already decomposed, this project
was paying a generative model to do a classification job — five times per
question, at roughly 95% of its model spend.

Three things follow, and the third is the one worth the change on its own.

**The citation cannot be invented.** Every mark must cite a line that resolves
inside the answer, and asking a model to do that produced fabricated ids, which
`citations.check` then refused — losing the whole question rather than one mark.
Here the citation *is* the line that best entails the check. There is nothing to
fabricate.

**The mark stops moving.** A cross-encoder is deterministic. The five-sample panel
exists to cancel decode noise, and an encoder has none, so the panel and its 5x
cost disappear together and "a mark a teacher cannot reproduce is not a mark"
becomes exactly true instead of true 42 times in 45.

**The student's handwriting stops leaving the machine** for this step, which for a
school is a property rather than an optimisation.

**What this is not.** It is not a claim that small models grade as well as large
ones. Measured on generic short-answer grading they do not generalise: fine-tuned
~1B models beat GPT-4o in-domain and score 0.31-0.43 against its 0.67 across
domains, and an unseen paper is the cross-domain case. The argument here is that
this is not that task, because the hypothesis is supplied. That argument is
exactly the kind that has to be measured rather than asserted, which is what
`tooling/scripts/score_scientsbank.py` is for — unseen questions, unseen domains,
false credit and false zeros reported apart.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from vedaai_contracts import Line

#: Entailment at or above this answers a check "yes".
#:
#: High, deliberately. A false "yes" is a mark awarded for something the student
#: did not write, which is the error a teacher gets challenged on; a check that
#: lands between the two thresholds is deferred to them instead, which costs one
#: named check rather than trust.
ENTAILED = float(os.getenv("NLI_ENTAILED") or 0.65)

#: Below this, the check is answered "no".
#:
#: The gap between the two is not slack, it is the deferral band, and it is the
#: mechanism that replaces a generative model's "unsure".
NOT_ENTAILED = float(os.getenv("NLI_NOT_ENTAILED") or 0.25)

#: How many lines either side of the best one are read together as the premise.
#:
#: A check often needs two lines to be satisfied — a value on one and its unit on
#: the next — and scoring lines only in isolation refuses those. Scoring only the
#: whole answer, on the other hand, gives no citation and lets an unrelated
#: sentence carry an unsupported claim.
WINDOW = 2


@dataclass(frozen=True)
class Verdict:
    """One check, answered."""

    met: bool | None
    cited_line_ids: list[str]
    score: float
    error: str | None = None


class Entailment(Protocol):
    """Scores whether a premise asserts a hypothesis, in ``[0, 1]``."""

    name: str

    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


#: "Does the answer give the value 15 m/s?" is a question. NLI wants a statement.
#:
#: Done in code rather than by asking the bank for both forms: the transform is
#: deterministic, and a second field is a second thing a model can get wrong.
_OPENERS = re.compile(
    r"^\s*(does|do|did)\s+(the\s+)?(answer|response|student|it|they)\s+", re.IGNORECASE
)
_IS_OPENER = re.compile(r"^\s*(is|are|was|were)\s+", re.IGNORECASE)


def hypothesis_for(check) -> str | None:
    """The proposition to test, or None where the check is not a proposition.

    The bank writes it. Deriving it from the question mechanically was tried and
    measured: "Does the answer state that field lines never intersect?" becomes
    "The answer states that field lines never intersect", which scores 0.057
    against a student who wrote exactly that, because an entailment model reads
    the hypothesis as a claim about the world and has no referent for "the
    answer". The bare claim is the only form that can be entailed.

    ``None`` for a check with no proposition behind it — that a definition is not
    circular, that two reasons are *distinct*, that each is correct rather than
    merely plausible. Those are properties of the writing, not facts, and an
    entailment model asked to judge one answers confidently and wrongly: it scored
    a textbook circular definition at 0.842 for being non-circular. A check like
    that is deferred to someone who can read.
    """
    claim = (getattr(check, "claim", "") or "").strip()
    if claim:
        return claim
    # A bank derived before this field existed. The transform is worse than the
    # bank's own wording and is here so an old cache does not fail outright.
    return as_statement(check.ask) or None


def as_statement(ask: str) -> str:
    """The check phrased as the claim it is testing.

    "Does the answer give the value 15 m/s?" -> "The answer gives the value 15 m/s"

    Third-person agreement is restored on the leading verb only, which is where
    the question form moved it. Anything that does not match a known opener is
    returned with its question mark removed and nothing else done to it — a
    slightly awkward hypothesis is a much smaller error than a mangled one.
    """
    text = ask.strip().rstrip("?").strip()

    match = _OPENERS.match(text)
    if match:
        rest = text[match.end() :].strip()
        verb, _, tail = rest.partition(" ")
        return f"The answer {_third_person(verb)} {tail}".strip()

    if _IS_OPENER.match(text):
        verb, _, tail = text.strip().partition(" ")
        return f"{tail} {verb.lower()}".strip()

    return text


def _third_person(verb: str) -> str:
    """`give` -> `gives`. Only the shapes an English check actually opens with."""
    lower = verb.lower()
    if not lower.isalpha():
        return verb
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return f"{lower}es"
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return f"{lower[:-1]}ies"
    return f"{lower}s"


def verify(
    checks: list,
    lines: list[Line],
    entailment: Entailment,
    question: str = "",
) -> list[Verdict]:
    """Answer every check against the answer's lines.

    The question is part of the premise, and it is not decoration. An answer line
    is a fragment written in the context of the question it answers, and an
    entailment model reads it without that context: "The lines never cross each
    other" does not entail "Magnetic field lines never intersect", because nothing
    in the premise says which lines. Measured on eight real cases from the physics
    script, the answer alone decided 5 of 8; prefixing the question decided 8 of 8,
    and the three that flipped went from 0.001 to 0.98.

    Crucially the negatives did not move with them — "feathers are insulators"
    stayed at 0.005 against "there is no potential difference between the bird's
    feet". Supplying the referent sharpened the discrimination rather than
    loosening it, which is the difference between fixing this and breaking it.

    The literature calls the missing step decontextualization, and this is the
    cheap form of it: rather than rewriting each line to stand alone, hand the
    model the context the line was written in.

    One batch of pairs for the whole question, because a cross-encoder is far
    faster batched and the per-call overhead otherwise dominates a short answer.
    """
    readable = [ln for ln in lines if ln.text.strip()]
    if not checks:
        return []
    if not readable:
        return [
            Verdict(met=None, cited_line_ids=[], score=0.0,
                    error="there is no readable text for this answer")
            for _ in checks
        ]

    premises = _premises(readable, question)
    hypotheses = [hypothesis_for(c) for c in checks]

    testable = [(i, h) for i, h in enumerate(hypotheses) if h]
    pairs = [(text, h) for _i, h in testable for text, _ids in premises]
    scores = entailment.score(pairs)

    verdicts: list[Verdict | None] = [None] * len(checks)
    width = len(premises)
    for slot, (i, _h) in enumerate(testable):
        window = scores[slot * width : (slot + 1) * width]
        best = max(range(len(window)), key=lambda j: window[j])
        score = window[best]
        _text, cited = premises[best]

        if score >= ENTAILED:
            verdicts[i] = Verdict(met=True, cited_line_ids=list(cited), score=score)
        elif score <= NOT_ENTAILED:
            verdicts[i] = Verdict(
                met=False,
                cited_line_ids=[],
                score=score,
                error="the answer does not state this",
            )
        else:
            # The deferral band. Neither given nor taken; one named check for the
            # teacher rather than a guess in either direction.
            verdicts[i] = Verdict(
                met=None, cited_line_ids=list(cited), score=score,
                error="too close to call from the writing",
            )

    # A check with no proposition behind it — "is the definition non-circular",
    # "are the two reasons distinct" — is not an entailment question, and the
    # measured cost of pretending otherwise is a circular definition scored 0.842
    # for being non-circular. Deferred to a person, by name.
    for i, verdict in enumerate(verdicts):
        if verdict is None:
            verdicts[i] = Verdict(
                met=None,
                cited_line_ids=[],
                score=0.0,
                error="this check is about how the answer is written rather than "
                "what it states, so it needs a person",
            )
    return [v for v in verdicts if v is not None]


def _premises(lines: list[Line], question: str = "") -> list[tuple[str, list[str]]]:
    """The spans of the answer a check may be entailed by, with their line ids.

    Every window of up to ``WINDOW`` consecutive lines, plus the whole answer. The
    windows are what produce a citation; the whole answer is what catches a claim
    the student spread across the page, where no window carries it alone.

    Each span is prefixed with the question, which is what gives the student's
    pronouns and bare nouns a referent. See ``verify``.
    """
    stem = f"Question: {question.strip()} Answer: " if question.strip() else ""
    spans: list[tuple[str, list[str]]] = []
    for start in range(len(lines)):
        for size in range(1, WINDOW + 1):
            chunk = lines[start : start + size]
            if len(chunk) < size:
                break
            spans.append(
                (
                    stem + " ".join(ln.text.strip() for ln in chunk),
                    [ln.line_id for ln in chunk],
                )
            )
    whole = " ".join(ln.text.strip() for ln in lines)
    if len(lines) > WINDOW:
        spans.append((stem + whole, [ln.line_id for ln in lines]))
    return spans


def as_judgement(verdicts: list[Verdict]) -> dict:
    """The verdicts in the shape ``assemble_checks`` already reads.

    Deliberately the same dict a generative marker returns. Everything downstream
    — the marks, the deferral handling, the citation validation, the unverifiable
    credit — is reused rather than reimplemented, so the two markers cannot drift
    apart in how a judgement becomes a grade.
    """
    return {
        "checks": [
            {
                "index": i,
                "met": v.met,
                "cited_line_ids": v.cited_line_ids,
                "error": v.error,
            }
            for i, v in enumerate(verdicts, start=1)
        ],
        "feedback": None,
        "uncertain": all(v.met is None for v in verdicts) if verdicts else False,
    }
