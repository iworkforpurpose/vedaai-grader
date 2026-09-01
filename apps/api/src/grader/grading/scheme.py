"""Marking as a bank of binary checks, not a score out of five.

Two measured failures sit behind this module, and they pull in opposite
directions, which is why neither could be fixed by tuning.

**Fluent nonsense earned full marks.** A physics answer explaining that a bird is
safe on a power line because feathers are insulators — on topic, confident, and
not the reason — earned 3 of 3 with citations that all resolved. Every safety
mechanism in this package worked; the judgement was wrong. This is the documented
failure of LLM judges: they lean on fluency, verbosity and polish rather than on
whether the substance is there.

**Handing the marker a model-written "correct answer" made it worse.** Measured on
five papers, the false-zero rate went from 4.5% to 31.8% and the script total from
three marks over truth to twenty-one under. The scheme is written by the same
model from the question alone, so where it guesses it guesses with conviction — it
decided "the landform at position A" was a glacier and refused a waterfall — and
the marker then enforced the guess. Replacing no reference with a wrong reference
is not an improvement.

**What the literature says to do instead, and it resolves both.** Rubric-conditioned
grading aligns with human markers on *binary* judgements and degrades as rubric
granularity grows; moving from partial credit to binary raises agreement by around
twenty points, because partial credit introduces ambiguity without improving
discriminative power. So a question becomes a bank of atomic yes/no checks, each
worth a fixed mark and each requiring cited evidence.

That is not a compromise between the two failures — it removes the mechanism both
used. A binary check cannot be talked into a mark by fluency, because it names the
substance that has to be present. And it cannot lose a mark it earned, because a
question that asks two things becomes two checks that stand or fall separately:
"Define resistance and state its SI unit" scored 0 as one scalar criterion and
scores 1 as two checks, which is what a teacher gives.

Three rules make it safe rather than merely stricter:

* **A check that cannot be verified is credited, not refused.** Where the question
  refers to a passage, figure or source the paper never supplied, nothing
  downstream can confirm a quotation. Withholding there would punish a student for
  our missing input.
* **A check may be answered "unsure".** That mark is neither given nor taken; it is
  deferred to the teacher, naming the one check in doubt rather than the whole
  question.
* **The marks are the paper's.** Checks are rescaled to the printed total, because a
  wrong denominator makes every mark on a question wrong at once.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from vedaai_contracts import Question

from .rubric import Rubric


@dataclass(frozen=True)
class Check:
    """One atomic yes/no question about the answer, worth a fixed mark."""

    ask: str
    """Phrased so that yes means the mark is earned. Names the substance that has
    to be present — "gives the value 15 m/s", not "correct calculation"."""

    marks: float
    needs_material: bool = False
    """True only when answering needs material the paper did not supply — a
    passage, a figure, a table. Such a check is credited rather than refused,
    because the missing input is ours and not the student's.

    Default False, and the polarity matters. The field was first written as
    ``verifiable: bool = True`` and the model set it false on *every* check,
    including "does the answer give the value 15 m/s" — which needs nothing but the
    answer. Everything then fell into the credit-anyway branch. A flag whose unsafe
    value is the one a model reaches for by default is the wrong way round."""

    @property
    def verifiable(self) -> bool:
        return not self.needs_material


@dataclass(frozen=True)
class CheckBank:
    """The checks for one question."""

    qid: str
    checks: list[Check] = field(default_factory=list)
    traps: list[str] = field(default_factory=list)
    """Wrong answers that read as right. Stated so the marker has to look past
    fluency for the substance the check names."""

    needs_material: bool = False

    @property
    def usable(self) -> bool:
        return bool(self.checks)

    @property
    def total(self) -> float:
        return sum(c.marks for c in self.checks)


#: Banks by question, because one paper is marked for a whole class: the checks
#: for question 4 are derived once and reused for every script.
_CACHE: dict[str, CheckBank] = {}

BANK_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["checks", "traps", "needs_material"],
    "properties": {
        "checks": {
            "type": "array",
            "description": "Atomic yes/no checks. Each must be answerable by looking at "
            "the answer and finding one specific thing. They must sum to the marks "
            "available.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ask", "marks", "needs_material"],
                "properties": {
                    "ask": {
                        "type": "string",
                        "description": "A yes/no question where yes earns the mark. Name "
                        "the substance: 'Does the answer give 15 m/s?' not 'Is the "
                        "calculation correct?'. Never bundle two things into one check — "
                        "'defines resistance AND gives the unit' must be two checks.",
                    },
                    "marks": {"type": "number"},
                    "needs_material": {
                        "type": "boolean",
                        "description": "Almost always false. Set true ONLY if this check "
                        "cannot be answered without reading a passage, figure, table or "
                        "source that is absent from the question text above. A question "
                        "that is self-contained — a calculation, a definition, a "
                        "named property, an explanation of a mechanism — is false. If you "
                        "can tell whether the answer is right by reading the answer, it is "
                        "false.",
                    },
                },
            },
        },
        "traps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Wrong answers that read as right — a plausible mechanism that "
            "is not the right one, a definition that restates the term, an arithmetic "
            "slip inside sound working.",
        },
        "needs_material": {
            "type": "boolean",
            "description": "True if the question as a whole refers to material not "
            "supplied in its text.",
        },
    },
}

SYSTEM = """\
You are turning one exam question into a bank of marking checks, before any \
student answer is seen. You are not marking anything and you are not writing a \
model answer.

Each check is a yes/no question about the student's answer where **yes earns the \
mark**. A check must be atomic and concrete:

  good: "Does the answer give the value 15 m/s?"
  good: "Does the answer state that there is no potential difference between the \
bird's feet?"
  bad:  "Is the calculation correct?"            (not concrete)
  bad:  "Does it define resistance and give the unit?"   (two things in one check)

**Never bundle.** If the question asks for two things, or for a definition and a \
unit, or for three properties, that is two or three separate checks. A student who \
gets one and misses the other must be able to earn one mark.

**Never demand content the question does not.** This is the mistake that matters \
most, and it is the opposite of the one above. Many questions have more than one \
correct answer: "give two reasons why people settle on floodplains" has six or \
seven valid reasons and the student needs any two. A check naming one of them \
fails every student who chose a different valid one.

So decide first whether the question is closed or open.

**Closed** — one right answer. A value, a unit, a named term, a specific \
mechanism, a definition. Write content-specific checks; they are exact and safe:

  "Does the answer give the value 15 m/s?"
  "Does the answer state the SI unit as the ohm?"
  "Does the answer say there is no potential difference between the bird's feet?"

**Open** — many right answers. "Describe", "explain", "give two reasons", \
"discuss". Write checks that test *validity and sufficiency*, never named content:

  "Does the answer give at least two distinct and correct reasons why people \
settle on floodplains?"
  "Does the answer describe at least two correct ways railways contributed to \
urban growth?"
  "Is each reason given actually correct, rather than merely plausible?"

  NOT: "Does the answer mention economic benefits?"        (one valid answer of many)
  NOT: "Does the answer give a time scale for the feature?" (the question never asked)

If you would have to guess which valid answer the student chose, the check is \
wrong. Ask whether what they wrote is correct and sufficient instead.

The checks must sum to exactly the marks available. Use half marks only if the \
total forces it.

`needs_material` is almost always **false**. Set it true only when the check \
genuinely cannot be answered by reading the student's answer, because it depends \
on something the question text does not contain:

  false: "Does the answer give the value 15 m/s?"          (self-contained)
  false: "Does the answer state the SI unit as the ohm?"    (self-contained)
  false: "Does the answer say there is no potential difference between the feet?"
  true:  "Does the quoted phrase actually appear in the poem?"   (no poem supplied)
  true:  "Is the landform at position A correctly named?"        (no figure supplied)

If you can tell whether the answer is right by reading the answer, it is false.

Then list the traps: wrong answers a marker would be tempted to credit because \
they are fluent, confident and on topic.\
"""


def _user_message(question: Question, rubric: Rubric, reference: str = "") -> str:
    # A reference answer, where the teacher supplied one.
    #
    # This is the difference between checks derived from the question alone and
    # checks derived from a real mark scheme, and it is worth keeping separable
    # because the two are not equivalent: a model writing its own reference guesses
    # confidently where it does not know, and enforcing that guess tripled the
    # false-zero rate. A reference a person wrote does not have that failure.
    known = (
        f"\n\nThe teacher's reference answer, which is correct and authoritative:\n"
        f"{reference.strip()}\n\nBase the checks on it. Do not add requirements it "
        f"does not contain."
        if reference.strip()
        else ""
    )
    return f"""\
QUESTION {question.label_raw} ({rubric.marks_available:g} marks total)
{question.text}{known}

Write the checks. They must sum to {rubric.marks_available:g}.\
"""


def available() -> bool:
    """Whether checks can be derived at all."""
    if os.environ.get("MARK_CHECKS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


async def derive(
    question: Question, rubric: Rubric, *, reference: str = "", client=None
) -> CheckBank | None:
    """The checks for one question, or None if they could not be worked out.

    None rather than raising: marking falls back to the criteria the paper printed,
    which is what it did before this module existed.
    """
    if not available() and client is None:
        return None
    if rubric.marks_available <= 0:
        return None

    key = f"{question.qid}\x00{question.text}\x00{rubric.marks_available}\x00{reference}"
    if key in _CACHE:
        return _CACHE[key]

    try:
        raw = await _ask(question, rubric, reference=reference, client=client)
    except Exception:  # noqa: BLE001 - checks are an improvement, not a requirement
        return None

    bank = _assemble(question, rubric, raw)
    if bank.usable:
        _CACHE[key] = bank
    return bank if bank.usable else None


async def _ask(
    question: Question, rubric: Rubric, *, reference: str = "", client=None
) -> dict:
    if client is None:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

    completion = await client.chat.completions.create(
        model=os.getenv("MARK_CHECKS_MODEL") or os.getenv("GRADER_MODEL") or "gpt-4o-mini",
        temperature=0.0,
        seed=20240817,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _user_message(question, rubric, reference)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "check_bank", "strict": True, "schema": BANK_SCHEMA},
        },
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("no checks returned")
    return json.loads(content)


def _assemble(question: Question, rubric: Rubric, raw: dict) -> CheckBank:
    """Build the bank, with the paper's total enforced on it."""
    checks: list[Check] = []
    for entry in raw.get("checks") or []:
        if not isinstance(entry, dict):
            continue
        ask = str(entry.get("ask", "")).strip()
        if not ask:
            continue
        checks.append(
            Check(
                ask=ask,
                marks=max(0.0, float(entry.get("marks", 0.0) or 0.0)),
                needs_material=bool(entry.get("needs_material", False)),
            )
        )

    checks = _rescaled(checks, rubric.marks_available)
    return CheckBank(
        qid=question.qid,
        checks=checks,
        traps=[str(t).strip() for t in (raw.get("traps") or []) if str(t).strip()],
        needs_material=bool(raw.get("needs_material", False)),
    )


def _rescaled(checks: list[Check], available_marks: float) -> list[Check]:
    """Make the checks sum to what the paper printed.

    An even share is used when the model's own split is unusable — zero, or wildly
    off — because the count of checks is the useful part of its answer and the
    division is something a rule can do exactly.
    """
    if not checks or available_marks <= 0:
        return checks

    total = sum(c.marks for c in checks)
    if abs(total - available_marks) <= 1e-6:
        return checks

    n = len(checks)
    if total <= 0:
        share = round((available_marks / n) * 2) / 2
        marks = [share] * n
    else:
        marks = [round((c.marks / total) * available_marks * 2) / 2 for c in checks]

    # Never a zero-mark check: a check worth nothing cannot earn or lose anything
    # and only adds noise to the report.
    marks = [max(0.5, m) for m in marks]
    drift = round(available_marks - sum(marks), 2)
    marks[0] = max(0.5, round(marks[0] + drift, 2))
    return [
        Check(ask=c.ask, marks=m, needs_material=c.needs_material)
        for c, m in zip(checks, marks, strict=False)
    ]


def render(bank: CheckBank) -> str:
    """The bank as the marking prompt shows it."""
    lines = ["CHECKS — answer each yes, no, or unsure. Yes earns the mark."]
    for i, c in enumerate(bank.checks, start=1):
        note = "" if c.verifiable else "   [cannot be verified from the paper — see below]"
        lines.append(f"  {i}. [{c.marks:g}] {c.ask}{note}")
    if bank.traps:
        lines.append("")
        lines.append(
            "TRAPS — these read as right and are not. A fluent, confident, on-topic "
            "answer saying one of these has NOT met the check it looks like it meets:"
        )
        lines.extend(f"  - {t}" for t in bank.traps)
    if any(not c.verifiable for c in bank.checks) or bank.needs_material:
        lines.append("")
        lines.append(
            "A check marked as unverifiable refers to a passage, figure or source the "
            "question paper does not contain. Answer those YES unless the answer is "
            "plainly self-contradictory: the missing material is our omission and the "
            "student must not lose a mark for it. Say so in the comment."
        )
    return "\n".join(lines)


def clear_cache() -> None:
    """Forget every derived bank. For tests, and for a changed paper."""
    _CACHE.clear()
