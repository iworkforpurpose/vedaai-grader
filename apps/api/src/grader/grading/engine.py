"""Grading engines behind one interface.

Three, for the same reason the transcription layer has more than one: the
interesting decisions should be measurable against each other rather than assumed.
Here the split also does something else — it keeps grading useful when no model is
configured at all.

``RubricOnly`` proposes no marks. It derives the rubric from the paper, locates
the answer, and hands the teacher both with every point unjudged. That is a
genuine product: a marking aid that structures the work without inventing a
score. It is also the honest fallback, because a grade nobody can justify is
worth less than no grade.

``Claude`` and ``OpenAIGrader`` judge the answer and cite the lines behind each
mark. Their output is validated before it is shown; a fabricated or out-of-scope
citation invalidates the grade rather than being quietly dropped.

Which one runs is configuration, and the answer is recorded on every grade. That
matters more with two providers than it did with one: a mark is only checkable if
you know what made it, and a small model and a large one are not interchangeable
evidence.

**On using a small model here, and why it stopped.** It was a reasonable thing to
try, and the reason was structural rather than optimistic. This task is not a
reasoning showcase — it is "read a rubric, read numbered lines, decide, and cite
the line IDs" — and the two ways a weak model fails it are both already
contained. Malformed output is prevented by asking for a schema rather than for
prose. Invented citations are caught by validation, which refuses the grade
instead of displaying it, so the failure mode is *no mark* rather than a wrong
one.

What is not contained is judgement: deciding whether a student's own words satisfy
a criterion, in text a recognizer has already damaged. Nothing in the architecture
rescues that, and the gate eventually measured it — five of nine documents outside
the mark range a teacher would defend, every one of them under-marking. The
default moved up and three of the five came back inside. See ``DEFAULT_MODELS``
for the table.

That leaves the architecture doing what it was built to do and the model doing
what only a model can. It is also why the provider and model are recorded on every
grade: a mark is only checkable if you know what made it, and a small model and a
large one are not interchangeable evidence.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import Protocol

from vedaai_contracts import LineIndex, Question, QuestionGrade, RubricPoint

from ..observability import log_event
from . import citations, prompt, sampling
from .rubric import Rubric

#: Default model per provider, used when GRADER_MODEL names none.
#:
#: OpenAI's default used to be ``gpt-4o-mini``, on the argument that marking is a
#: short, highly constrained call — a rubric, a few lines, a schema to fill — and
#: that the expensive part of a large model is capability this task mostly does
#: not use. The argument ended by saying that if it proved not good enough, that
#: would show up in the grades rather than in the bill.
#:
#: It showed up in the grades. Measured on the gate, one pass, nine documents
#: whose defensible mark range was written down before any run:
#:
#: ===============  =============  ===========  =========
#: document         truth (band)   gpt-4o-mini  gpt-4.1
#: ===============  =============  ===========  =========
#: history          20 (15-20)     17           18
#: geography        15 (13-15)     15           15
#: english          20 (16-20)     15  FAIL     20
#: economics        13 (11-13)     10  FAIL     10  FAIL
#: physics           5  (4-7)       3  FAIL      6
#: math-paper       15 (14-17)      7  FAIL     10  FAIL
#: asap-clean        3  (3-3)       2  FAIL      3
#: asap-middling     2  (2-3)       3            3
#: asap-worst        3  (3-3)       3            3
#: ===============  =============  ===========  =========
#:
#: Five documents outside their band become two, and every one of the five was
#: *under* marking — the small model withholding marks a teacher gave. That is
#: the failure this package cannot contain by construction: malformed output is
#: prevented by demanding a schema, and invented citations are refused by
#: validation, but whether a student's own wording satisfies a criterion is
#: judgement, and nothing in the architecture substitutes for it.
#:
#: `physics` is the check on the other direction. It is the one paper in the set
#: answered badly on purpose, and it lands at 6 against a truth of 5 in a band of
#: 4-7 — so the larger model is not simply more generous, which would have shown
#: here first.
#:
#: The two that remain are not marking faults. `economics` loses three marks to a
#: margin label the student wrote against the wrong question, and `math-paper` to
#: handwritten mathematics that recognition cannot read.
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4.1",
}

#: Explicit provider choice. With nothing set, whichever key is present is used —
#: convenient, and safe only because the provider is recorded on every grade, so
#: "which engine judged this" is never a guess.
GRADER_PROVIDER = os.getenv("GRADER_PROVIDER", "").strip().lower()

#: Cap on the answer text handed to the model, in characters. A whole script is
#: small; this exists to bound a pathological OCR result rather than to save cost.
MAX_ANSWER_CHARS = 20_000

class Grader(Protocol):
    """Grades one answer, having been given the rubric and the lines to use."""

    name: str

    async def grade(
        self,
        *,
        question: Question,
        rubric: Rubric,
        index: LineIndex,
        line_ids: list[str],
        scheme=None,
    ) -> QuestionGrade: ...


def _unjudged_points(rubric: Rubric, *, comment: str) -> list[RubricPoint]:
    return [
        RubricPoint(
            point_id=f"{rubric.qid}#{i + 1}",
            criterion=criterion.criterion,
            marks_available=criterion.marks,
            marks_awarded=0.0,
            satisfied=False,
            cited_line_ids=[],
            comment=comment,
        )
        for i, criterion in enumerate(rubric.criteria)
    ]


class RubricOnly:
    """Produces the rubric and locates the answer, awarding nothing.

    Deliberately not a scoring heuristic. Keyword overlap would produce numbers
    that look like marks and are not, and a teacher shown a plausible wrong score
    is worse off than one shown none — the first visible mistake costs more than
    the help was worth.
    """

    name = "rubric_only"

    async def grade(
        self,
        *,
        question: Question,
        rubric: Rubric,
        index: LineIndex,
        line_ids: list[str],
        scheme=None,
    ) -> QuestionGrade:
        return QuestionGrade(
            qid=question.qid,
            marks_available=rubric.marks_available,
            marks_awarded=0.0,
            rubric_points=_unjudged_points(
                rubric, comment="Not marked automatically — awaiting the teacher."
            ),
            feedback=None,
            confidence=0.0,
            graded_by="rubric_only",
            graded_on_partial_text=False,
        )


#: Fixed sampling seed, so two runs over one script agree.
#:
#: A constant rather than a config knob: there is no version of this product where
#: a teacher wants marking to vary run to run, and a configurable seed would be a
#: setting whose only effect is to make results harder to compare.
GRADER_SEED = 20240817

#: How many independent judgements are taken before a mark is settled.
#:
#: One was flaky, and the gate measured how flaky. Over three identical passes of
#: nine documents and 45 scored questions, four questions returned a different
#: mark — 8.9% — and the worst of them moved 3 marks out of 5. Over the same
#: passes the pipeline never placed an answer differently once, so the marker was
#: the only unstable component in the product.
#:
#: Five, measured rather than chosen. Three identical passes of the same nine
#: documents and the same 45 questions, at each setting:
#:
#: =========================  ==============  ===========  =======
#: arm                        unreproducible  clean         in band
#: =========================  ==============  ===========  =======
#: 1 sample, temperature 0    8.9%  (4/45)    3 of 9        6 of 9
#: 3 samples, temperature 0   11.1% (5/45)    3 of 9        6 of 9
#: 5 samples, temperature .7  2.2%  (1/45)    6 of 9        7 of 9
#: =========================  ==============  ===========  =======
#:
#: The middle row is why the temperature matters and why three is not enough: a
#: panel sampled greedily made things *worse*. See ``MARK_TEMPERATURE``.
#:
#: Five marking calls per question rather than one. That is the cheapest part of
#: the run — recognition and rendering dominate — and it buys the difference
#: between a mark a teacher can check and one that moves when they look again.
#:
#: An earlier note in this repository demoted this idea on the evidence that 20 of
#: 21 questions were stable across three passes. That was one paper set, measured
#: before marking moved to binary checks, and 20 of 21 cannot tell 0% from 9%.
MARK_SAMPLES = max(1, int(os.getenv("MARK_SAMPLES") or 5))

#: How much of the panel must agree before a check is settled.
#:
#: ``majority`` settles on two of three and defers only a genuine three-way split.
#: ``unanimous`` defers whenever the samples disagree at all, trading marks
#: settled for marks a teacher can rely on without checking.
MARK_AGREEMENT = (os.getenv("MARK_AGREEMENT") or "majority").strip().lower()

#: The temperature each member of the panel is sampled at.
#:
#: Zero for one sample, above zero for a panel, and this is the counter-intuitive
#: part of the whole change, so it is worth stating why.
#:
#: The first attempt at a panel kept temperature at zero and measured *worse*:
#: 8.9% of questions unreproducible became 11.1%. Voting reduces variance only
#: across independent samples, and at temperature zero the decode is near-greedy,
#: so varying the seed changed almost nothing — three samples were three
#: correlated draws from one process. Worse, the vote threshold added a flip of
#: its own, because a check landing 2-1 one run and 3-0 the next moves between
#: deferred and awarded.
#:
#: Raising it to 0.7 and taking five samples brought the same measurement to 2.2%.
#: A panel pays for its independence with per-sample noise: each sample is
#: deliberately worse so that the aggregate can be better. Anyone tempted to set
#: this back to zero "for determinism" should re-read the middle row of the table
#: above — that experiment has been run.
#:
#: With one sample it must stay at zero, because there the noise has nothing to
#: cancel against.
def _default_temperature(samples: int) -> float:
    return 0.0 if samples <= 1 else 0.7


MARK_TEMPERATURE = float(
    os.getenv("MARK_TEMPERATURE") or _default_temperature(MARK_SAMPLES)
)


def _seeds(count: int) -> list[int]:
    """The seeds the panel uses, fixed rather than random.

    The same set every run, so where the provider honours a seed the panel is
    reproducible by construction, and where it does not the vote absorbs what is
    left. Randomising here would make the marker *less* reproducible, not more.
    """
    return [GRADER_SEED + offset for offset in range(count)]


def _agreement(votes: int) -> int:
    """How many of ``votes`` samples have to agree.

    Computed from the samples that actually returned rather than from
    ``MARK_SAMPLES``, because a panel that quietly shrinks is how "unanimous"
    becomes "whatever the one surviving sample said".
    """
    if MARK_AGREEMENT == "unanimous":
        return votes
    return votes // 2 + 1


def vote_checks(samples: list[dict], *, need: int) -> dict:
    """One check judgement out of several, decided check by check.

    Per check rather than by picking a whole sample, because the checks are
    independent questions about the answer and a sample that is wrong about one is
    not thereby wrong about the others. Citations travel with the check that
    earned them, so a voted judgement is still internally consistent.

    A check the panel splits on comes out as ``met: null`` — deferred to the
    teacher rather than guessed. That is the point: the mark a panel cannot agree
    on is exactly the mark a single sample was flipping between runs, and one
    named check to settle is a smaller ask than a number that moves.
    """
    order: list[int] = []
    for sample in samples:
        for entry in sample.get("checks") or []:
            position = entry.get("index")
            if isinstance(position, int) and position not in order:
                order.append(position)

    checks: list[dict] = []
    for position in sorted(order):
        entries = [
            entry
            for sample in samples
            for entry in (sample.get("checks") or [])
            if entry.get("index") == position
        ]
        yes = [entry for entry in entries if entry.get("met") is True]
        no = [entry for entry in entries if entry.get("met") is False]

        if len(yes) >= need:
            met: bool | None = True
        elif len(no) >= need:
            met = False
        else:
            met = None

        cited: list[str] = []
        if met is True:
            for entry in yes:
                for line_id in entry.get("cited_line_ids") or []:
                    if line_id not in cited:
                        cited.append(line_id)

        # The fault is carried over from the samples that named one, preferring the
        # side that won. Withholding a mark without naming the fault is treated as
        # a shrug downstream, so losing the wording here would turn a decided "no"
        # back into an unsure.
        error = next(
            (entry.get("error") for entry in (no or entries) if entry.get("error")),
            None,
        )

        checks.append(
            {
                "index": position,
                "met": met,
                "cited_line_ids": cited,
                "error": None if met is True else error,
            }
        )

    return {
        "checks": checks,
        "feedback": next(
            (sample.get("feedback") for sample in samples if sample.get("feedback")),
            None,
        ),
        "uncertain": sum(1 for sample in samples if sample.get("uncertain")) >= need,
    }


def median_sample(samples: list[dict]) -> dict:
    """The sample whose total sits in the middle, taken whole.

    Whole rather than merged point by point, because on the scalar path the marks
    and the comments justifying them are written together: recombining them can
    produce a judgement whose comment argues for a mark it did not award. The
    median also cannot be a value no sample proposed, which a mean can.
    """

    def total(sample: dict) -> float:
        return sum(
            float(point.get("marks_awarded") or 0.0)
            for point in (sample.get("points") or [])
        )

    return sorted(samples, key=total)[len(samples) // 2]


async def _panel(judge, count: int) -> list[dict]:
    """``count`` judgements, concurrently, minus any that failed.

    Failures are dropped rather than fatal: one refused call out of three should
    cost a vote, not the question. Everything failing is still an error, and the
    surviving count is what sets the agreement threshold.
    """
    results = await asyncio.gather(
        *(judge(seed) for seed in _seeds(count)), return_exceptions=True
    )
    samples = [r for r in results if isinstance(r, dict)]
    if len(samples) < count:
        # A panel that quietly shrinks is how "unanimous" becomes "whatever the
        # one surviving sample said". The agreement threshold is recomputed from
        # the survivors, so a question marked by one of five is indistinguishable
        # downstream from one marked five-nil.
        for result in results:
            if isinstance(result, BaseException):
                log_event(
                    "panel_sample_failed",
                    returned=len(samples),
                    requested=count,
                    error=type(result).__name__,
                    detail=str(result),
                )
    if not samples:
        first = next((r for r in results if isinstance(r, BaseException)), None)
        raise first or ValueError("the model returned no judgement")
    return samples


#: The shape the model must return. Written by hand rather than generated from the
#: contract so that the model is asked for exactly the fields it should decide —
#: it never sets identity, availability, or anything downstream code computes.
JUDGEMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based position of the rubric point being judged.",
                    },
                    "marks_awarded": {"type": "number", "minimum": 0},
                    "satisfied": {"type": "boolean"},
                    "cited_line_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Line IDs from inside the answer fence. Required "
                        "whenever marks are awarded.",
                    },
                    "comment": {"type": "string"},
                    "error": {
                        "type": "string",
                        "description": "Where marks are withheld, what specifically is "
                        "wrong. Naming it is required; 'incomplete' is not naming it.",
                    },
                },
                "required": ["index", "marks_awarded", "satisfied", "cited_line_ids"],
            },
        },
        "feedback": {
            "type": "string",
            "description": "One or two sentences addressed to the student.",
        },
        "uncertain": {
            "type": "boolean",
            "description": "True when the transcription was too damaged to judge fairly.",
        },
    },
    "required": ["points", "uncertain"],
}


#: The response shape when the question came with a bank of binary checks.
#:
#: `met` is a nullable boolean on purpose. True earns the mark, False refuses it,
#: and null is "unsure" — that mark is deferred to the teacher rather than guessed
#: in either direction. A scalar `marks_awarded` is deliberately absent: asking for
#: a number is what let a fluent wrong answer be talked into three marks, and
#: rubric-conditioned grading is measured to agree with human markers on binary
#: judgements and to degrade as granularity grows.
CHECK_JUDGEMENT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["checks", "feedback", "uncertain"],
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "met", "cited_line_ids", "error"],
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based position of the check being answered.",
                    },
                    "met": {
                        "type": ["boolean", "null"],
                        "description": "true = yes, earns the mark. false = no. "
                        "null = unsure, defer this mark to the teacher.",
                    },
                    "cited_line_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Line IDs evidencing the answer. Required when met "
                        "is true.",
                    },
                    "error": {
                        "type": ["string", "null"],
                        "description": "When met is false, what specifically is missing or "
                        "wrong. Naming it is required.",
                    },
                },
            },
        },
        "feedback": {"type": ["string", "null"]},
        "uncertain": {"type": "boolean"},
    },
}


class GraderUnavailable(RuntimeError):
    """Raised when a model-backed grader is asked for but cannot be built.

    Not fatal anywhere it is raised: the caller falls back to the rubric-only
    grader, which is a working product rather than an error path.
    """


class Claude:
    """Judges an answer with Claude, citing the lines behind each mark."""

    name = "claude"

    def __init__(self, *, model: str | None = None, client=None) -> None:
        self.model = model or os.getenv("GRADER_MODEL") or DEFAULT_MODELS["anthropic"]
        if client is not None:
            self._client = client
            return

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise GraderUnavailable(
                "ANTHROPIC_API_KEY is not set, so answers cannot be marked automatically. "
                "The rubric and the located answer are still produced."
            )
        try:
            from anthropic import AsyncAnthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise GraderUnavailable(
                "the anthropic package is not installed; install the 'grading' extra"
            ) from exc
        self._client = AsyncAnthropic()

    async def grade(
        self,
        *,
        question: Question,
        rubric: Rubric,
        index: LineIndex,
        line_ids: list[str],
        scheme=None,
    ) -> QuestionGrade:
        # A drawing is not gradable from a transcription it does not have. Asking
        # anyway produces a confident zero for a correct answer.
        if not rubric.gradable_from_text:
            return _needs_a_person(rubric, question, graded_by=self.provenance)

        # Whether a mark scheme was derived decides the shape of the answer, and it
        # has to decide the tool as well as the prompt. It did not: this path asked
        # for rubric-point marks while the prompt asked for binary checks, so the
        # two disagreed about what was being requested the moment the scheme path
        # landed. Unreachable with no Anthropic key configured, which is why it
        # survived, and wrong the moment one is.
        binary = scheme is not None and getattr(scheme, "usable", False)
        text = prompt.build(
            question=question, rubric=rubric, index=index,
            line_ids=line_ids, scheme=scheme,
        )
        samples = await _panel(lambda _seed: self._judge(text, binary=binary), MARK_SAMPLES)
        need = _agreement(len(samples))

        if binary:
            return assemble_checks(
                question=question,
                rubric=rubric,
                bank=scheme,
                index=index,
                line_ids=line_ids,
                judgement=vote_checks(samples, need=need),
                graded_by=self.provenance,
            )

        return assemble(
            question=question,
            rubric=rubric,
            index=index,
            line_ids=line_ids,
            judgement=median_sample(samples),
            graded_by=self.provenance,
        )

    async def _judge(self, text: str, *, binary: bool) -> dict:
        """One member of the panel. This provider exposes no seed."""
        message = await self._client.messages.create(
            model=self.model,
            # Same reason as the OpenAI path, and the same trade: a lone sample is
            # taken greedily, a panel pays per-sample noise for independence.
            temperature=MARK_TEMPERATURE,
            max_tokens=2048,
            system=prompt.SYSTEM,
            tools=[
                {
                    "name": "record_judgement",
                    "description": "Record the answer to each check."
                    if binary
                    else "Record the judgement for each rubric point.",
                    "input_schema": CHECK_JUDGEMENT_SCHEMA if binary
                    else JUDGEMENT_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "record_judgement"},
            messages=[{"role": "user", "content": text}],
        )
        return _tool_input(message)

    @property
    def provenance(self) -> str:
        return f"anthropic:{self.model}"

    async def aclose(self) -> None:
        """Release the HTTP client.

        Worth having because the failure it prevents is confusing rather than
        harmful: a harness that marks several documents calls ``asyncio.run`` once
        per document, and an unclosed client is cleaned up by the garbage
        collector later, against a loop that no longer exists. That surfaces as a
        bare ``RuntimeError: Event loop is closed`` traceback with no reference to
        anything in this project, printed above output that is otherwise correct.
        """
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


#: The same judgement, expressed for OpenAI's structured-output mode.
#:
#: A separate schema rather than a shared one, because strict mode is genuinely
#: stricter and the differences are not cosmetic: every property must appear in
#: ``required``, ``additionalProperties`` must be false, and numeric bounds like
#: ``minimum`` are not supported. An optional field is therefore expressed as a
#: nullable type instead of an absent key.
#:
#: Worth the duplication. Strict mode guarantees the response parses and matches,
#: which removes the failure a small model is most likely to produce.
STRICT_JUDGEMENT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["points", "feedback", "uncertain"],
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "marks_awarded", "satisfied", "cited_line_ids",
                             "comment", "error"],
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based position of the rubric point being judged.",
                    },
                    "marks_awarded": {"type": "number"},
                    "satisfied": {"type": "boolean"},
                    "cited_line_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Line IDs copied from inside the answer fence. "
                        "Required whenever marks are awarded.",
                    },
                    "comment": {"type": ["string", "null"]},
                    "error": {
                        "type": ["string", "null"],
                        "description": "Where marks are withheld, what specifically is "
                        "wrong. Naming it is required; 'incomplete' is not naming it.",
                    },
                },
            },
        },
        "feedback": {
            "type": ["string", "null"],
            "description": "One or two sentences addressed to the student.",
        },
        "uncertain": {
            "type": "boolean",
            "description": "True when the transcription was too damaged to judge fairly.",
        },
    },
}


class OpenAIGrader:
    """Judges an answer with an OpenAI model, citing the lines behind each mark."""

    name = "openai"

    def __init__(self, *, model: str | None = None, client=None) -> None:
        self.model = model or os.getenv("GRADER_MODEL") or DEFAULT_MODELS["openai"]
        if client is not None:
            self._client = client
            return

        if not os.getenv("OPENAI_API_KEY"):
            raise GraderUnavailable(
                "OPENAI_API_KEY is not set, so answers cannot be marked automatically. "
                "The rubric and the located answer are still produced."
            )
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise GraderUnavailable(
                "the openai package is not installed; install the 'grading' extra"
            ) from exc
        self._client = AsyncOpenAI()

    async def grade(
        self,
        *,
        question: Question,
        rubric: Rubric,
        index: LineIndex,
        line_ids: list[str],
        scheme=None,
    ) -> QuestionGrade:
        # A drawing is not gradable from a transcription it does not have. Asking
        # anyway produces a confident zero for a correct answer.
        if not rubric.gradable_from_text:
            return _needs_a_person(rubric, question, graded_by=self.provenance)

        binary = scheme is not None and getattr(scheme, "usable", False)
        message = prompt.build(
            question=question, rubric=rubric, index=index,
            line_ids=line_ids, scheme=scheme,
        )
        samples = await _panel(
            lambda seed: self._judge(message, binary=binary, seed=seed), MARK_SAMPLES
        )
        need = _agreement(len(samples))

        if binary:
            return assemble_checks(
                question=question,
                rubric=rubric,
                bank=scheme,
                index=index,
                line_ids=line_ids,
                judgement=vote_checks(samples, need=need),
                graded_by=self.provenance,
            )

        return assemble(
            question=question,
            rubric=rubric,
            index=index,
            line_ids=line_ids,
            judgement=median_sample(samples),
            graded_by=self.provenance,
        )

    async def _judge(self, message: str, *, binary: bool, seed: int) -> dict:
        """One member of the panel.

        Temperature and seed are asked for and given up if the model refuses
        them. Marking the same script twice must give the same marks: temperature
        was once unset, the API default is 1.0, and re-marking one submission five
        times produced totals of 3, 3, 2, 2 and 0. The seed asks the provider for
        reproducible sampling — best-effort rather than guaranteed, which is why
        the panel exists at all, and the seeds being fixed is what keeps the panel
        itself reproducible.

        A reasoning model accepts neither, and refusing the whole request over an
        optimisation would mark nothing at all. See ``sampling``.
        """
        return await sampling.structured_completion(
            self._client,
            model=self.model,
            system=prompt.SYSTEM,
            user=message,
            schema_name="judgement",
            schema=CHECK_JUDGEMENT_SCHEMA if binary else STRICT_JUDGEMENT_SCHEMA,
            temperature=MARK_TEMPERATURE,
            seed=seed,
        )

    @property
    def provenance(self) -> str:
        return f"openai:{self.model}"

    async def aclose(self) -> None:
        """Release the HTTP client.

        Worth having because the failure it prevents is confusing rather than
        harmful: a harness that marks several documents calls ``asyncio.run`` once
        per document, and an unclosed client is cleaned up by the garbage
        collector later, against a loop that no longer exists. That surfaces as a
        bare ``RuntimeError: Event loop is closed`` traceback with no reference to
        anything in this project, printed above output that is otherwise correct.
        """
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def select_grader() -> Grader:
    """The grader this deployment should use.

    An explicit ``GRADER_PROVIDER`` wins. Otherwise whichever key is present is
    used, and if neither is, the rubric-only grader — which is a working product,
    not an error path: it structures the marking without inventing a score.
    """
    providers: list[tuple[str, type]] = [("anthropic", Claude), ("openai", OpenAIGrader)]
    if GRADER_PROVIDER == "openai":
        providers.reverse()
    elif GRADER_PROVIDER == "none":
        return RubricOnly()

    reasons: list[str] = []
    for _name, engine in providers:
        try:
            return engine()
        except GraderUnavailable as unavailable:
            reasons.append(str(unavailable))

    raise GraderUnavailable(" ".join(reasons))


def _needs_a_person(rubric: Rubric, question: Question, *, graded_by: str) -> QuestionGrade:
    """A question whose answer is a drawing, left for someone who can see it."""
    return QuestionGrade(
        qid=question.qid,
        marks_available=rubric.marks_available,
        marks_awarded=0.0,
        rubric_points=_unjudged_points(
            rubric, comment="Answered by a drawing. Needs a person to look at the page."
        ),
        confidence=0.0,
        graded_by=graded_by,
        graded_on_partial_text=True,
    )


def _tool_input(message) -> dict:
    """The judgement from a tool-use response, whatever else the message holds."""
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            data = getattr(block, "input", None)
            if isinstance(data, str):
                return json.loads(data)
            if isinstance(data, dict):
                return data
    raise ValueError("the model returned no judgement")


def assemble(
    *,
    question: Question,
    rubric: Rubric,
    index: LineIndex,
    line_ids: list[str],
    judgement: dict,
    graded_by: str | None = None,
) -> QuestionGrade:
    """Turn a model judgement into a grade, or refuse it.

    Separated from the transport so the validation can be tested without a
    network call — it is the part that decides whether a grade is trustworthy,
    which makes it the part most worth testing.

    A judgement whose citations do not hold is not repaired. Every point reverts
    to unjudged and the reason is recorded, because a grade assembled from
    partially fabricated evidence is not a smaller grade, it is an unfounded one.
    """
    allowed = set(line_ids)
    points: list[RubricPoint] = []
    #: Points that claimed credit and cited nothing. They earn nothing, and they
    #: are counted because the student may have deserved them — see _confidence.
    unevidenced: list[str] = []

    for i, criterion in enumerate(rubric.criteria):
        raw = _point_for(judgement, i + 1)
        if raw is None:
            points.append(
                RubricPoint(
                    point_id=f"{rubric.qid}#{i + 1}",
                    criterion=criterion.criterion,
                    marks_available=criterion.marks,
                    marks_awarded=0.0,
                    satisfied=False,
                    cited_line_ids=[],
                    comment="The model did not judge this point.",
                )
            )
            continue

        satisfied = bool(raw.get("satisfied", False))
        awarded = max(0.0, min(float(raw.get("marks_awarded", 0.0)), criterion.marks))

        # A point the marker calls satisfied is worth what the paper allotted it.
        #
        # The two fields are returned independently and could disagree, and on a
        # real script they did: question 11(a), "Define atomic number and mass
        # number", worth 2. The student defined both. The marker set satisfied,
        # wrote "You provided clear definitions for both... Great job!" and awarded
        # 1 of 2. A teacher reading praise beside half marks cannot tell which half
        # of the grade to believe, which makes the whole grade useless.
        #
        # Partial credit is still expressible and is what an unsatisfied point with
        # a positive mark means: partly there, not met.
        #
        # Neither reading applies to a point that credited itself and cited
        # nothing. Question 16 of a real script, worth 5: two points cited four
        # lines each, and the third claimed credit with an empty citation list —
        # satisfied on one run, unsatisfied but carrying marks on the next. Either
        # way the citation check saw "marks awarded with no line cited", and it
        # refuses a question whole rather than in part, so a correct and fully
        # evidenced 3.5 became 0 and unjudged. Twice, by two different routes.
        #
        # A missing citation is an omission, not a fabrication. The model did not
        # invent evidence; it failed to give any, and the answer to an unevidenced
        # claim is to not credit that claim — not to throw away the claims that
        # were evidenced. An invented or out-of-scope line id is the other thing
        # entirely, and still refuses the question below: a model making evidence
        # up for one point has said nothing trustworthy about the others.
        cited = citations.resolve_all(raw.get("cited_line_ids") or [], index)
        comment = raw.get("comment")
        # The named fault, put in front of the comment. A teacher checking a
        # withheld mark wants "150/10 is 15, not 1.5" before the encouragement.
        named = str(raw.get("error") or "").strip()
        if named and awarded < criterion.marks:
            comment = f"{named} — {comment}" if comment else named
        if not cited and (satisfied or awarded > 0):
            # Not shown as met either. "Satisfied, nought marks" recreates in the
            # other direction the contradiction the promotion exists to remove.
            satisfied = False
            awarded = 0.0
            unevidenced.append(f"{rubric.qid}#{i + 1}")
            comment = (
                "The marker credited this point but cited no line for it, so it "
                "could not be checked and earned nothing. Worth reading yourself."
            )
        elif satisfied:
            awarded = criterion.marks
        points.append(
            RubricPoint(
                point_id=f"{rubric.qid}#{i + 1}",
                criterion=criterion.criterion,
                marks_available=criterion.marks,
                marks_awarded=awarded,
                satisfied=satisfied,
                cited_line_ids=cited,
                comment=comment,
            )
        )

    problems = citations.check(points, index, allowed_line_ids=allowed)
    if problems:
        reason = "; ".join(str(p) for p in problems[:3])
        return QuestionGrade(
            qid=question.qid,
            marks_available=rubric.marks_available,
            marks_awarded=0.0,
            rubric_points=_unjudged_points(
                rubric, comment=f"Marking was refused — evidence did not check out ({reason})."
            ),
            feedback=None,
            confidence=0.0,
            graded_by=graded_by,
            graded_on_partial_text=True,
        )

    uncertain = bool(judgement.get("uncertain", False))
    return QuestionGrade(
        qid=question.qid,
        marks_available=rubric.marks_available,
        marks_awarded=sum(p.marks_awarded for p in points),
        rubric_points=points,
        # The only place this is true. Every other constructor in this module
        # returns a grade nobody judged, and the difference is not otherwise
        # recoverable from the payload — see the field's own note.
        judged=True,
        feedback=judgement.get("feedback"),
        graded_by=graded_by,
        # Confidence is not asked of the model — a self-reported number is not
        # evidence. It is derived from whether the model flagged the transcription
        # as unreadable and from how much of the rubric it managed to cite.
        confidence=0.35 if uncertain else _confidence(points, unevidenced=len(unevidenced)),
        graded_on_partial_text=uncertain,
    )


def _point_for(judgement: dict, index: int) -> dict | None:
    for raw in judgement.get("points", []) or []:
        if isinstance(raw, dict) and raw.get("index") == index:
            return raw
    return None


def _confidence(points: list[RubricPoint], *, unevidenced: int) -> float:
    """How far a teacher can take this grade on trust.

    A point that claimed credit and cited nothing was dropped to zero. Whether
    the student had actually earned it is exactly what cannot be established
    here, so the grade goes to a person however clean the rest of it looks — a
    quiet zero on an answer nobody re-reads is how marking goes wrong without
    anyone noticing.
    """
    share = _cited_share(points)
    return min(share, 0.5) if unevidenced else share


def _cited_share(points: list[RubricPoint]) -> float:
    """Share of awarded points that pointed at specific lines.

    A grade whose marks are all traceable is one a teacher can check in seconds;
    one whose marks rest on unstated evidence needs re-reading. That difference is
    what the number is for.
    """
    awarded = [p for p in points if p.marks_awarded > 0]
    if not awarded:
        # Nothing awarded is a definite judgement, not an uncertain one: a zero
        # needs no citation to be checkable, since the whole answer is the
        # evidence.
        return 0.8
    cited = sum(1 for p in awarded if p.cited_line_ids)
    return round(0.5 + 0.5 * (cited / len(awarded)), 2)


def assemble_checks(
    *,
    question: Question,
    rubric: Rubric,
    bank,
    index: LineIndex,
    line_ids: list[str],
    judgement: dict,
    graded_by: str | None = None,
) -> QuestionGrade:
    """Turn binary check answers into a grade.

    One rule per state, and each exists because of a measured failure.

    **Yes earns the mark, and must cite a line.** Same as before: a mark with no
    resolvable evidence is unfounded rather than small.

    **No earns nothing, and must name the fault.** A refusal with no named fault is
    not a judgement, it is a shrug — so it is treated as *unsure* rather than as a
    zero. This is what stops the strictness that a model-written mark scheme
    introduced, where the false-zero rate tripled because the marker withheld marks
    it could not explain.

    **Unsure defers the mark.** Neither awarded nor refused. It is reported as a
    single named check for the teacher to settle, which is a smaller ask than
    re-reading the answer, and it is the honest state for damaged transcription.

    **An unverifiable check is credited.** Where the question refers to material the
    paper never supplied, nothing downstream can confirm a quotation or a position.
    The missing input is ours; the student does not lose a mark for it.
    """
    answers = {}
    for raw in judgement.get("checks", []) or []:
        if isinstance(raw, dict) and isinstance(raw.get("index"), int):
            answers[raw["index"]] = raw

    points: list[RubricPoint] = []
    deferred: list[str] = []
    credited_unverifiable: list[str] = []

    for i, check in enumerate(bank.checks, start=1):
        raw = answers.get(i) or {}
        met = raw.get("met")
        cited = citations.resolve_all(raw.get("cited_line_ids") or [], index)
        fault = str(raw.get("error") or "").strip()
        point_id = f"{rubric.qid}#{i}"

        if not check.verifiable and met is not True:
            # Cannot be checked against material nobody supplied. Credited, and
            # said out loud so the teacher knows which marks rest on it.
            credited_unverifiable.append(point_id)
            points.append(
                RubricPoint(
                    point_id=point_id,
                    criterion=check.ask,
                    marks_available=check.marks,
                    marks_awarded=check.marks,
                    satisfied=True,
                    cited_line_ids=cited,
                    comment="Given. This needs the passage or figure the question paper "
                    "does not contain, so it could not be checked — read it yourself.",
                )
            )
            continue

        if met is True:
            points.append(
                RubricPoint(
                    point_id=point_id,
                    criterion=check.ask,
                    marks_available=check.marks,
                    marks_awarded=check.marks,
                    satisfied=True,
                    cited_line_ids=cited,
                    comment=fault or None,
                )
            )
            continue

        if met is False and fault:
            points.append(
                RubricPoint(
                    point_id=point_id,
                    criterion=check.ask,
                    marks_available=check.marks,
                    marks_awarded=0.0,
                    satisfied=False,
                    cited_line_ids=cited,
                    comment=fault,
                )
            )
            continue

        # Unsure, or a refusal with no fault named. Deferred either way.
        deferred.append(point_id)
        points.append(
            RubricPoint(
                point_id=point_id,
                criterion=check.ask,
                marks_available=check.marks,
                marks_awarded=0.0,
                satisfied=False,
                cited_line_ids=cited,
                comment="Not decided — this one needs your eye."
                if met is None
                else "Refused without a stated reason, so it was not applied. Check this one.",
            )
        )

    # Validated over the points decided on evidence, not over all of them.
    #
    # A credited-unverifiable check has nothing to cite *by construction*: it was
    # given precisely because the material needed to check it is absent. Passing it
    # to a rule that requires a citation for any mark refused the whole question —
    # on the physics paper every question came back "evidence did not check out"
    # and `judged=False`, which read as the model failing when it was this function
    # contradicting itself.
    #
    # The invariant that matters is unchanged: a mark awarded *on the evidence of
    # the answer* must cite a line that resolves inside that answer. A mark given
    # because we could not look is a different thing, and it is labelled as such in
    # the comment rather than dressed up with evidence it does not have.
    evidenced = [p for p in points if p.point_id not in credited_unverifiable]
    problems = citations.check(evidenced, index, allowed_line_ids=set(line_ids))
    if problems:
        reason = "; ".join(str(p) for p in problems[:3])
        return QuestionGrade(
            qid=question.qid,
            marks_available=rubric.marks_available,
            marks_awarded=0.0,
            rubric_points=_unjudged_points(
                rubric, comment=f"Marking was refused — evidence did not check out ({reason})."
            ),
            confidence=0.0,
            graded_by=graded_by,
            graded_on_partial_text=True,
        )

    uncertain = bool(judgement.get("uncertain", False))
    awarded = sum(p.marks_awarded for p in points)
    settled = [p for p in points if p.point_id not in deferred]

    return QuestionGrade(
        qid=question.qid,
        marks_available=rubric.marks_available,
        marks_awarded=awarded,
        rubric_points=points,
        judged=True,
        feedback=judgement.get("feedback"),
        graded_by=graded_by,
        # Confidence is the share of the marks that were actually settled. A
        # question with a deferred check is not a confident grade however clean the
        # rest of it looks, and a teacher deciding where to spend their attention
        # needs that to be visible in the number.
        confidence=0.35
        if uncertain
        else round(
            (sum(p.marks_available for p in settled) / rubric.marks_available)
            if rubric.marks_available
            else 0.0,
            2,
        ),
        graded_on_partial_text=uncertain or bool(deferred) or bool(credited_unverifiable),
    )
