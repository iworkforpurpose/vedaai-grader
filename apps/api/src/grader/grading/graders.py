"""The markers themselves, and how one is chosen.

Every grader answers the same `Grader` protocol, so the pipeline never learns
which is in use - it only records which one answered, because a mark is only
checkable if you know what made it.

`RubricOnly` is not an error path. A deployment with no marking credential still
extracts the paper, places every answer and produces the rubric; what it declines
to do is invent a score. That is a working product, and it is what a submission
degrades to rather than failing.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import Protocol

from vedaai_contracts import LineIndex, Question, QuestionGrade

from ..clients import client_for, peers_for
from ..observability import log_event
from . import prompt, sampling
from .assembly import _needs_a_person, _tool_input, _unjudged_points, assemble, assemble_checks
from .panel import (
    MARK_SAMPLES,
    MARK_TEMPERATURE,
    _agreement,
    _panel,
    median_sample,
    vote_checks,
)
from .rubric import Rubric
from .sampling import weighted_hosts as _weighted
from .schemas import (
    CHECK_JUDGEMENT_SCHEMA,
    JUDGEMENT_SCHEMA,
    STRICT_JUDGEMENT_SCHEMA,
)

#: Default model per provider, used when GRADER_MODEL names none.
#:
#: OpenAI's default used to be ``gpt-4o-mini``, on the argument that marking is a
#: short, highly constrained call - a rubric, a few lines, a schema to fill - and
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
#: *under* marking - the small model withholding marks a teacher gave. That is
#: the failure this package cannot contain by construction: malformed output is
#: prevented by demanding a schema, and invented citations are refused by
#: validation, but whether a student's own wording satisfies a criterion is
#: judgement, and nothing in the architecture substitutes for it.
#:
#: `physics` is the check on the other direction. It is the one paper in the set
#: answered badly on purpose, and it lands at 6 against a truth of 5 in a band of
#: 4-7 - so the larger model is not simply more generous, which would have shown
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
    # inside their band, the smaller 4 of 9 - and every one of the smaller's
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
    # small model safe here - the marker fills a schema rather than writing
    # prose, so it cannot answer in a shape the citation checker will refuse.
    #
    # `GRADER_MODEL` still names any of them, and the grade records which one
    # marked it, because a small model and a large one are not interchangeable
    # evidence.
    "groq": "openai/gpt-oss-120b",
}


#: Explicit provider choice. With nothing set, whichever key is present is used -
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


class RubricOnly:
    """Produces the rubric and locates the answer, awarding nothing.

    Deliberately not a scoring heuristic. Keyword overlap would produce numbers
    that look like marks and are not, and a teacher shown a plausible wrong score
    is worse off than one shown none - the first visible mistake costs more than
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
                rubric, comment="Not marked automatically - awaiting the teacher."
            ),
            feedback=None,
            confidence=0.0,
            graded_by="rubric_only",
            graded_on_partial_text=False,
        )


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
        samples = await _panel(
            lambda _seed, index: self._judge(text, binary=binary, index=index),
            MARK_SAMPLES,
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

    async def _judge(
        self, text: str, *, binary: bool, index: int = 0
    ) -> dict:
        # `index` says which member of the panel this is, so a grader can send it
        # to a different host. Anthropic serves these weights nowhere else, so
        # there is no peer to reach and the argument exists only to keep one
        # signature across the graders.
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


class OpenAIGrader:
    """Judges an answer with an OpenAI model, citing the lines behind each mark."""

    name = "openai"

    def __init__(self, *, model: str | None = None, client=None) -> None:
        from ..clients import marking_chain, openai_provider

        # Where marking starts is the first entry of the chain this deployment
        # can actually reach, not a per-provider default. The two differ once
        # more than one host is configured, and the chain is the ordering that
        # was measured; a per-provider default is the ordering that happened to
        # be written down.
        chain = marking_chain()
        self.provider, chain_model = chain[0] if chain else (openai_provider()[0], "")
        self.model = (
            model
            or os.getenv("GRADER_MODEL")
            or chain_model
            or DEFAULT_MODELS.get(self.provider, DEFAULT_MODELS["openai"])
        )
        if client is not None:
            self._client = client
            return

        from ..clients import key_for

        if not key_for(self.provider):
            raise GraderUnavailable(
                "No marking key is set, so answers cannot be marked automatically. "
                "Set GROQ_API_KEY, CEREBRAS_API_KEY or OPENAI_API_KEY. The rubric "
                "and the located answer are still produced."
            )
        try:
            import openai  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
            raise GraderUnavailable(
                "the openai package is not installed; install the 'grading' extra"
            ) from exc
        self._client = client_for(self.provider)

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
            lambda seed, index: self._judge(
                message, binary=binary, seed=seed, index=index
            ),
            MARK_SAMPLES,
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

    def _peer_for(self, index: int) -> tuple[object, str, str]:
        """The client, provider and model this sample should go to.

        Spread across every host serving these weights, in proportion to how fast
        each one accepts requests. Same model throughout, so the samples stay
        comparable and the panel is still a panel on one marker; what changes is
        that the requests-per-minute ceilings add up.

        In proportion, not evenly. An even split hands the slowest host as much
        work as the fastest, so the whole panel waits on it: measured on the live
        deployment, half of ninety calls sent to a host allowing five a minute is
        nine minutes, while the host allowing thirty finished its half in ninety
        seconds and then sat idle. Weighting by rate makes them finish together.
        """
        peers = _weighted(peers_for(getattr(self, "provider", ""), self.model))
        provider, model = peers[index % len(peers)]
        if provider == getattr(self, "provider", ""):
            return self._client, provider, model
        # Instance-local and lazy. A class attribute here would be one dict
        # shared by every grader ever built, handing one instance's clients to
        # another.
        cache: dict[str, object] = self.__dict__.setdefault("_peers", {})
        client = cache.get(provider)
        if client is None:
            client = client_for(provider)
            cache[provider] = client
        return client, provider, model

    async def _judge(
        self, message: str, *, binary: bool, seed: int, index: int = 0
    ) -> dict:
        """One member of the panel.

        Temperature and seed are asked for and given up if the model refuses
        them. Marking the same script twice must give the same marks: temperature
        was once unset, the API default is 1.0, and re-marking one submission five
        times produced totals of 3, 3, 2, 2 and 0. The seed asks the provider for
        reproducible sampling - best-effort rather than guaranteed, which is why
        the panel exists at all, and the seeds being fixed is what keeps the panel
        itself reproducible.

        A reasoning model accepts neither, and refusing the whole request over an
        optimisation would mark nothing at all. See ``sampling``.
        """
        client, provider, model = self._peer_for(index)
        return await sampling.structured_completion(
            client,
            model=model,
            provider=provider,
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
        OpenAI would both read `openai:...` - and "a mark is only checkable if
        you know what made it" stops being true the moment two hosts are
        configurable.
        """

        # The marker that will answer, not the one that was configured. They
        # differ once a daily budget is spent and marking continues down the
        # chain - possibly onto a different host - and a grade naming the
        # configured entry would make two scripts marked by two different models
        # look identical.
        provider, model = sampling.effective_marker(
            getattr(self, "provider", "openai"), self.model
        )
        return f"{provider}:{model}"

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
    per question for a whole class. *Answering* them is entailment - does this
    text assert this claim - and it happens roughly ninety times per script.

    Everything after the judgement is the generative path's code, unchanged:
    ``assemble_checks`` turns the same dict into the same grade, so the marks, the
    deferral handling, the unverifiable credit and the citation validation cannot
    drift between the two markers.

    Without a bank there is nothing to verify. That is not a failure - it is the
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
    the cross-domain regime - where generic short-answer grading with small models
    does not - is exactly the kind that has to be measured before it is defaulted
    to, and `score_scientsbank.py` is the measurement.
    """
    return os.getenv("LOCAL_CHECKS", "").strip().lower() in {"1", "true", "yes", "on"}


def select_grader() -> Grader:
    """The grader this deployment should use.

    An explicit ``GRADER_PROVIDER`` wins. Otherwise whichever key is present is
    used, and if neither is, the rubric-only grader - which is a working product,
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
    # marker instead - which then spent 175 provider calls being rate-limited, and
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
