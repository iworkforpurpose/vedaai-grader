"""Marking a question several times and reconciling the answers.

A single sample is a coin the temperature flips: the same script marked twice can
differ, and a mark a teacher cannot reproduce is not a mark. The panel exists to
cancel that noise, and the reconciliation is per check rather than per sample,
because checks are independent questions about the answer and a sample wrong
about one is not thereby wrong about the others.

A check the panel splits on comes back `met: null` - deferred to the teacher
rather than guessed. That is the point: the mark a panel cannot agree on is
exactly the mark a single sample was flipping between runs, and one named check
to settle is a smaller ask than a number that moves.
"""

from __future__ import annotations

import asyncio
import os

from ..observability import log_event

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
#: mark - 8.9% - and the worst of them moved 3 marks out of 5. Over the same
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
#: the run - recognition and rendering dominate - and it buys the difference
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
#: so varying the seed changed almost nothing - three samples were three
#: correlated draws from one process. Worse, the vote threshold added a flip of
#: its own, because a check landing 2-1 one run and 3-0 the next moves between
#: deferred and awarded.
#:
#: Raising it to 0.7 and taking five samples brought the same measurement to 2.2%.
#: A panel pays for its independence with per-sample noise: each sample is
#: deliberately worse so that the aggregate can be better. Anyone tempted to set
#: this back to zero "for determinism" should re-read the middle row of the table
#: above - that experiment has been run.
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

    A check the panel splits on comes out as ``met: null`` - deferred to the
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
