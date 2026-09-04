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

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from vedaai_contracts import Question

from .rubric import Rubric


@dataclass(frozen=True)
class Check:
    """One atomic yes/no question about the answer, worth a fixed mark."""

    ask: str
    """Phrased so that yes means the mark is earned. Names the substance that has
    to be present — "gives the value 15 m/s", not "correct calculation"."""

    marks: float

    claim: str = ""
    """The same check as a statement about the world, for a verifier to test.

    Not a duplicate of ``ask``: they are addressed to different readers, and the
    difference is load-bearing. ``ask`` is a question about the answer and is what
    a teacher reads. ``claim`` is the proposition the answer has to assert, and is
    what an entailment model is given as its hypothesis.

    Measured, deriving one from the other mechanically does not work. "Does the
    answer state that field lines never intersect?" transformed into "The answer
    states that field lines never intersect" scores 0.057 against a student who
    wrote exactly that — because an NLI model judges the hypothesis as a statement
    about the world and has no referent for "the answer". The bare claim, "Field
    lines never intersect", is the thing that can be entailed.

    Empty where the bank predates this field, in which case the verifier falls
    back to transforming ``ask`` and is worse at it."""

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
                "required": ["ask", "claim", "marks", "needs_material"],
                "properties": {
                    "ask": {
                        "type": "string",
                        "description": "A yes/no question where yes earns the mark. Name "
                        "the substance: 'Does the answer give 15 m/s?' not 'Is the "
                        "calculation correct?'. Never bundle two things into one check — "
                        "'defines resistance AND gives the unit' must be two checks.",
                    },
                    "claim": {
                        "type": "string",
                        "description": "The same check written as a plain statement "
                        "about the world, which the student's answer would make true. "
                        "NOT a sentence about the answer. 'Field lines never "
                        "intersect.' not 'The answer states that field lines never "
                        "intersect.' 'The speed is 15 m/s.' not 'The answer gives 15 "
                        "m/s.' Where the check tests a property of the writing rather "
                        "than a fact — that a definition is not circular, that two "
                        "distinct reasons are given — leave this empty, because there "
                        "is no fact to state.",
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

**Order the checks as a ladder, easiest first.** A real mark scheme is a ladder: one \
mark for identifying the right thing, another for explaining it, another for \
linking it to the question. Write them the same way, so that an answer which is \
partly right meets the early checks and fails the later ones.

  1. "Does the answer name the correct process?"          <- any partial answer has this
  2. "Does it say what drives that process?"
  3. "Does it link the process to the outcome the question asks about?"  <- complete only

This is the difference between a scheme that grades and one that only passes or \
fails. Two checks that each need most of the answer means a half-right answer \
scores nothing, which no teacher would do — measured, 10 of 13 partially-correct \
answers scored zero when the checks were written that way.

So the first check must be one that **any** answer showing partial understanding \
would meet. If every check requires the complete answer, split them differently.

**Write only the checks a real mark scheme would contain, and weight them.** The \
checks must sum to the marks available — but the *number* of checks is decided by \
the question, not by the total. A check may be worth 2 or 3 marks. Two or three \
checks on a five-mark question is normal and correct.

This is measured, not stylistic. When the count was pinned to the total, a \
five-mark question got five checks: the model wrote the two or three the mark \
scheme really contains and then invented the rest to fill the number. The \
inventions were always of the same kind, and always refusable:

  NOT: "Does the answer provide a conclusion that summarizes the interpretation?"
  NOT: "Does the answer link the changes to the overall themes of the story?"
  NOT: "Does the answer explain how the reasons relate to the overall tone?"
  NOT: "Does the answer provide a clear explanation linking the similarities and \
differences to the information from the article?"

No mark scheme asks for a summarising conclusion, or for a link to "the overall \
theme", or for an explanation of an explanation. Each of those cost a student a \
mark on an answer a teacher gave full marks to: "comment on the ending, is it \
hopeful or bleak, give reasons" was marked 2 of 5 against a human's 5 of 5, and \
three of the five checks were requirements nobody set.

So: one check per distinct thing the answer must contain, and nothing else. If \
the question supports three checks and carries five marks, give them 2, 2 and 1 — \
weight the rungs, do not add rungs. Use half marks only where the split forces it.

A test before you write each check: could you point at the phrase in the question, \
or the line in a published mark scheme, that requires this? If not, delete it.

`needs_material` is almost always **false**. Set it true only when the check \
genuinely cannot be answered by reading the student's answer, because it depends \
on something the question text does not contain:

  false: "Does the answer give the value 15 m/s?"          (self-contained)
  false: "Does the answer state the SI unit as the ohm?"    (self-contained)
  false: "Does the answer say there is no potential difference between the feet?"
  true:  "Does the quoted phrase actually appear in the poem?"   (no poem supplied)
  true:  "Is the landform at position A correctly named?"        (no figure supplied)

If you can tell whether the answer is right by reading the answer, it is false.

**Write the claim as well as the question.** The two are read by different things.
The question is what a teacher reads. The claim is the bare proposition the \
student's answer has to make true, and it is handed to a verifier as a hypothesis:

  ask:   "Does the answer state that field lines never intersect?"
  claim: "Field lines never intersect."

  ask:   "Does the answer give the value 15 m/s?"
  claim: "The speed is 15 m/s."

Never write the claim as a sentence about the answer. "The answer states that \
field lines never intersect" is a fact about a document, and a verifier asked to \
test it has no idea what document is meant — measured, that phrasing scores 0.057 \
against a student who wrote exactly the right thing.

Leave the claim **empty** where the check tests a property of the writing rather \
than a fact about the world: that a definition is not circular, that two *distinct* \
reasons are given, that each reason is correct rather than merely plausible. Those \
are real checks and there is no proposition to state, so say nothing rather than \
inventing one.

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
    # Material the paper printed with the question. Where it is present a check
    # that would otherwise be unanswerable becomes answerable — the table's rows are
    # here, so "does the answer use 10 and 100 from the first row" is checkable —
    # and `needs_material` should be false for it.
    printed = (
        "\n\nMATERIAL PRINTED WITH THE QUESTION, which you may rely on:\n"
        + "\n".join(f"  {m}" for m in question.material)
        if question.material
        else ""
    )
    return f"""\
QUESTION {question.label_raw} ({rubric.marks_available:g} marks total)
{question.text}{printed}{known}

Write the checks. They must sum to {rubric.marks_available:g}.\
"""


def available() -> bool:
    """Whether checks can be derived at all."""
    if os.environ.get("MARK_CHECKS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    from ..clients import openai_provider

    return bool(openai_provider()[1] or os.getenv("ANTHROPIC_API_KEY"))


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

    key = (
        f"{question.qid}\x00{question.text}\x00{rubric.marks_available}"
        f"\x00{reference}\x00{'|'.join(question.material)}"
    )
    if key in _CACHE:
        return _CACHE[key]

    stored = _restore(key)
    if stored is not None:
        _CACHE[key] = stored
        return stored

    try:
        raw = await _ask(question, rubric, reference=reference, client=client)
    except Exception as exc:  # noqa: BLE001 - checks are an improvement, not a requirement
        # Logged here rather than by the caller, because this function swallows
        # the exception and returns `None` — so the caller's own handler never
        # runs and its `scheme_failed` event never fired. That gap cost an hour:
        # every question reported "no check bank, nothing to verify" and nothing
        # anywhere said the provider had refused, or why.
        from ..observability import log_event

        log_event(
            "scheme_failed",
            qid=question.qid,
            error=type(exc).__name__,
            detail=str(exc),
        )
        return None

    bank = _assemble(question, rubric, raw)
    if bank.usable:
        _CACHE[key] = bank
        _persist(key, bank)
    return bank if bank.usable else None


def _model() -> str:
    """Which model writes the checks.

    Follows the marker by default. The checks are the mark scheme, so a weak model
    writing them limits a strong model marking against them — the measured failure
    was checks that demanded content the question never asked for, and no marker
    can award a mark a check refuses. ``MARK_CHECKS_MODEL`` exists to separate the
    two when that is the experiment being run.
    """
    from ..clients import openai_provider
    from .engine import DEFAULT_MODELS

    return (
        os.getenv("MARK_CHECKS_MODEL")
        or os.getenv("GRADER_MODEL")
        or DEFAULT_MODELS[openai_provider()[0]]
    )


async def _ask(
    question: Question, rubric: Rubric, *, reference: str = "", client=None
) -> dict:
    from . import sampling

    owned = None
    if client is None:
        from openai import AsyncOpenAI

        from ..clients import openai_kwargs

        owned = client = AsyncOpenAI(**openai_kwargs())

    try:
        return await sampling.structured_completion(
            client,
            model=_model(),
            system=SYSTEM,
            user=_user_message(question, rubric, reference),
            schema_name="check_bank",
            schema=BANK_SCHEMA,
            # Zero, and no panel. One bank is derived per question and reused for
            # every script in the class, so a bank that varied would make two
            # students marked an hour apart incomparable in a way no downstream
            # vote could repair.
            temperature=0.0,
            seed=20240817,
        )
    finally:
        # Closed because this function owns it. Left open, the client is collected
        # later against an event loop that no longer exists, and a harness that
        # marks nine documents with one `asyncio.run` each buries its own report
        # under a screen of "Event loop is closed" tracebacks.
        if owned is not None:
            await owned.close()


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
                claim=str(entry.get("claim", "") or "").strip(),
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
        Check(ask=c.ask, claim=c.claim, marks=m, needs_material=c.needs_material)
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


#: Where derived banks are kept between processes.
#:
#: A bank is a property of a question, not of a script, and it is the same for
#: every student who sat the paper. In memory only, it died with the task — so a
#: class of forty paid forty times for one paper's schemes, a restart threw away
#: everything, and an experiment could not be repeated without paying again.
#:
#: That last one is what made this urgent rather than tidy: a day's worth of
#: provider budget went on deriving banks that were then discarded, and the
#: measurement it was meant to enable could not run.
_STORE = Path(os.getenv("CHECK_BANK_CACHE") or Path(__file__).resolve().parents[3] / ".banks")


def _slot(key: str) -> Path:
    return _STORE / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"


def _persist(key: str, bank: CheckBank) -> None:
    """Write a bank where the next process can find it. Never fatal."""
    try:
        _STORE.mkdir(parents=True, exist_ok=True)
        _slot(key).write_text(
            json.dumps(
                {
                    "qid": bank.qid,
                    "traps": bank.traps,
                    "needs_material": bank.needs_material,
                    "checks": [
                        {
                            "ask": c.ask,
                            "claim": c.claim,
                            "marks": c.marks,
                            "needs_material": c.needs_material,
                        }
                        for c in bank.checks
                    ],
                }
            )
        )
    except OSError:
        # A read-only filesystem is a slower deployment, not a broken one.
        pass


def _restore(key: str) -> CheckBank | None:
    """A bank derived by an earlier process, if one is on disk."""
    slot = _slot(key)
    try:
        raw = json.loads(slot.read_text())
    except (OSError, ValueError):
        return None
    bank = CheckBank(
        qid=str(raw.get("qid", "")),
        checks=[
            Check(
                ask=str(c.get("ask", "")),
                marks=float(c.get("marks", 0.0)),
                claim=str(c.get("claim", "") or ""),
                needs_material=bool(c.get("needs_material", False)),
            )
            for c in raw.get("checks", [])
        ],
        traps=[str(x) for x in raw.get("traps", [])],
        needs_material=bool(raw.get("needs_material", False)),
    )
    return bank if bank.usable else None


def clear_cache() -> None:
    """Forget every derived bank. For tests, and for a changed paper."""
    _CACHE.clear()
    with contextlib.suppress(OSError):
        for slot in _STORE.glob("*.json"):
            slot.unlink()
