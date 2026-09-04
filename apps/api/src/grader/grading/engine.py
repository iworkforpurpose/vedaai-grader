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
    # An open-weight model, and the reason it is here is arithmetic. Marking a
    # class of forty on `gpt-4.1` costs about $27; the same class on this costs
    # about $2, and a full nine-document gate run costs six cents rather than
    # seventy-eight. A product that marks a whole class at a time cannot be
    # priced like one that answers a single question.
    #
    # The larger open-weight size first, with the smaller one behind it as a
    # second allowance rather than a second charge. See `sampling.FALLBACK_MODELS`.
    #
    # Measured, both on the nine-document gate: the larger marks 8 of 9 documents
    # inside their band, the smaller 4 of 9 — and every one of the smaller's
    # misses is under-marking, never false credit. Under-marking is the safer
    # direction but it is not a safe result: two of its misses are answers a
    # student earned marks for that scored zero, and a false zero is the worst
    # error this product makes.
    #
    # So the smaller model is where marking goes when the day's budget on the
    # larger one is gone, and not before. That is a real degradation and it is
    # recorded as one on every grade.
    #
    # Three further notes on why the smaller one is nevertheless worth having:
    #
    # **It is the task that was made small, not the model that was made weak.**
    # `scheme.py` turns a question into atomic yes/no checks, so marking is
    # answering a supplied binary question against a supplied text rather than
    # judging an answer from scratch. The fact-checking literature is consistent
    # that decomposition benefits weaker verifiers most and does little for
    # strong ones, because an undecomposed judgement has to weigh several
    # sub-claims at once and the supported ones drown the contradicted one. This
    # codebase already paid for the decomposition and was not collecting on it.
    #
    # **The daily budget is per model, not per account.** A free tier gives each
    # model its own allowance, so the choice is not "smaller and weaker" against
    # "larger and stronger" but against a specific number of scripts a day. The
    # 120B allowance is spent by roughly one full gate run; the 20B allowance
    # buys several times that, because the tokens cost less and the pool is its
    # own.
    #
    # **The schema guarantee is not a differentiator.** It was the stated reason
    # for the larger default, and it does not hold: both sizes accept
    # `response_format: json_schema` with `strict: true`, which is what makes a
    # small model safe here — the marker fills a schema rather than writing
    # prose, so it cannot answer in a shape the citation checker will refuse.
    #
    # `GRADER_MODEL` still names any of them, and the grade records which one
    # marked it, because a small model and a large one are not interchangeable
    # evidence.
    "groq": "openai/gpt-oss-120b",
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
        from ..clients import anthropic_kwargs

        self._client = AsyncAnthropic(**anthropic_kwargs())

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
        from ..clients import openai_provider

        # The default follows the provider. A Groq key with no `GRADER_MODEL`
        # should mark on a model Groq actually serves, rather than asking it for
        # `gpt-4.1` and failing with a model-not-found nobody expected.
        self.model = (
            model
            or os.getenv("GRADER_MODEL")
            or DEFAULT_MODELS[openai_provider()[0]]
        )
        if client is not None:
            self._client = client
            return

        from ..clients import openai_provider

        provider, key, _base = openai_provider()
        if not key:
            raise GraderUnavailable(
                "No marking key is set, so answers cannot be marked automatically. "
                "Set GROQ_API_KEY or OPENAI_API_KEY. The rubric and the located "
                "answer are still produced."
            )
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise GraderUnavailable(
                "the openai package is not installed; install the 'grading' extra"
            ) from exc
        from ..clients import openai_kwargs

        self._client = AsyncOpenAI(**openai_kwargs())

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
        """Which host and model judged this, recorded on every grade.

        The provider, not the SDK. Groq speaks the OpenAI API, so without this a
        grade marked by `gpt-oss-20b` on Groq and one marked by `gpt-4.1` on
        OpenAI would both read `openai:...` — and "a mark is only checkable if
        you know what made it" stops being true the moment two hosts are
        configurable.
        """
        from ..clients import openai_provider

        # The model that will answer, not the one that was configured. They
        # differ once a daily budget is spent and marking falls back to the
        # second allowance, and a grade that named the configured model would
        # make two scripts marked by two different models look identical.
        return f"{openai_provider()[0]}:{sampling.effective_model(self.model)}"

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


class LocalChecks:
    """Answers a bank of binary checks with a local cross-encoder.

    The cheap half of the design, and the one that carries almost all the volume.
    Deriving the checks is generative, needs subject knowledge, and happens once
    per question for a whole class. *Answering* them is entailment — does this
    text assert this claim — and it happens roughly ninety times per script.

    Everything after the judgement is the generative path's code, unchanged:
    ``assemble_checks`` turns the same dict into the same grade, so the marks, the
    deferral handling, the unverifiable credit and the citation validation cannot
    drift between the two markers.

    Without a bank there is nothing to verify. That is not a failure — it is the
    scalar rubric path, and the caller falls back to it exactly as it did before
    this class existed.
    """

    name = "local_checks"

    def __init__(self, *, entailment=None) -> None:
        if entailment is not None:
            self._entailment = entailment
            return
        from . import nli

        if not nli.available():
            raise GraderUnavailable(
                "local entailment is not installed, so checks cannot be answered "
                "on this machine; install the 'nli' extra"
            )
        self._entailment = nli.shared()

    @property
    def model(self) -> str:
        return getattr(self._entailment, "name", "nli")

    @property
    def provenance(self) -> str:
        return f"local:{self.model}"

    async def grade(
        self,
        *,
        question: Question,
        rubric: Rubric,
        index: LineIndex,
        line_ids: list[str],
        scheme=None,
    ) -> QuestionGrade:
        if not rubric.gradable_from_text:
            return _needs_a_person(rubric, question, graded_by=self.provenance)
        if scheme is None or not scheme.usable:
            raise GraderUnavailable(
                "no check bank for this question, so there is nothing to verify"
            )

        from . import verifier

        by_id = index.by_id()
        lines = [by_id[i] for i in line_ids if i in by_id]

        # Synchronous and CPU-bound, so it goes to a thread rather than blocking
        # the loop the way `reread` did. One question is a few hundred pairs and
        # a fraction of a second, but four run at once and the SSE stream is on
        # this loop.
        verdicts = await asyncio.to_thread(
            verifier.verify, scheme.checks, lines, self._entailment, question.text
        )

        return assemble_checks(
            question=question,
            rubric=rubric,
            bank=scheme,
            index=index,
            line_ids=line_ids,
            judgement=verifier.as_judgement(verdicts),
            graded_by=self.provenance,
        )

    async def aclose(self) -> None:
        """Nothing to release. The model is shared and outlives one submission."""
        return None


def _local_checks_preferred() -> bool:
    """Whether to answer checks locally when nothing has been asked for.

    Opt-in for now. The claim that entailment on a supplied hypothesis survives
    the cross-domain regime — where generic short-answer grading with small models
    does not — is exactly the kind that has to be measured before it is defaulted
    to, and `score_scientsbank.py` is the measurement.
    """
    return os.getenv("LOCAL_CHECKS", "").strip().lower() in {"1", "true", "yes", "on"}


def select_grader() -> Grader:
    """The grader this deployment should use.

    An explicit ``GRADER_PROVIDER`` wins. Otherwise whichever key is present is
    used, and if neither is, the rubric-only grader — which is a working product,
    not an error path: it structures the marking without inventing a score.
    """
    # Local entailment first where it is installed, because it is the right shape
    # for the job and not merely the cheap one: a cross-encoder is deterministic,
    # so the mark stops moving between runs, and its citation is the line that
    # entailed the check rather than a line id a model was asked to produce and
    # could invent. It still needs a provider to *derive* the checks; what it
    # removes is the five generative calls per question that answer them.
    if GRADER_PROVIDER == "none":
        return RubricOnly()

    # `LOCAL_CHECKS` and `GRADER_PROVIDER` are not the same switch, and treating
    # them as one meant a run asking for local marking silently got the generative
    # marker instead — which then spent 175 provider calls being rate-limited, and
    # the resulting gate looked like a verdict on entailment when it was a verdict
    # on nothing at all.
    #
    # They name different roles. `GRADER_PROVIDER` is the host that *derives* the
    # check bank, which is generative and needs subject knowledge. `LOCAL_CHECKS`
    # decides who *answers* those checks, which is entailment. Deriving on Groq and
    # answering on this machine is the intended configuration, not a contradiction.
    if _local_checks_preferred():
        try:
            return LocalChecks()
        except GraderUnavailable:
            if os.getenv("LOCAL_CHECKS", "").strip().lower() in {"1", "true", "yes", "on"}:
                raise

    # `OpenAIGrader` covers every OpenAI-shaped host, Groq included; which one it
    # reaches is decided by `clients.openai_provider` from the keys present.
    providers: list[tuple[str, type]] = [("anthropic", Claude), ("openai", OpenAIGrader)]
    if GRADER_PROVIDER in {"openai", "groq"}:
        providers.reverse()

    reasons: list[str] = []
    for _name, engine in providers:
        try:
            return engine()
        except GraderUnavailable as unavailable:
            reasons.append(str(unavailable))

    # One sentence, written for the person reading it.
    #
    # Joining the providers' own messages put this on a teacher's screen:
    #
    #   "No marking key is set, so answers cannot be marked automatically. Set
    #   GROQ_API_KEY or OPENAI_API_KEY. The rubric and the located answer are
    #   still produced. ANTHROPIC_API_KEY is not set, so answers cannot be marked
    #   automatically. The rubric and the located answer are still produced."
    #
    # Two providers, the same fact twice, and two environment variables a teacher
    # can do nothing about. The per-provider reasons are for whoever deployed
    # this, so they go to the log, where they can name variables freely and where
    # somebody is actually looking for them.
    log_event("no_marker_available", reasons=" | ".join(reasons))
    raise GraderUnavailable(
        "Answers were located and the rubric was produced, but no marks were "
        "proposed: this deployment has no marking model configured."
    )


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
