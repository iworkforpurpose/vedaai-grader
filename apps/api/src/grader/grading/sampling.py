"""Asking a model for one schema-shaped answer, whatever it accepts.

Marking sends the same request shape everywhere: a system prompt, one message, a
strict JSON schema, a temperature and a seed. The last two are load-bearing — the
panel is five independent samples, and independence is what makes voting cancel
anything — so they are not decoration that can be dropped on a whim.

They are also not universally accepted. Reasoning models refuse a temperature
other than the default:

    BadRequestError: Unsupported value: 'temperature' does not support 0.7 with
    this model. Only the default (1) value is supported.

Without this module that refusal is per-request, so every member of the panel
fails, ``_panel`` finds no samples, the per-question guard catches it, and the
teacher is told the answer "could not be marked automatically". Setting
``GRADER_MODEL`` to a reasoning model therefore marked *nothing*, and said so in
words that name the key rather than the setting that was actually changed.

So an unsupported parameter is dropped and the call retried once, and the model
is remembered as refusing it so the rest of the panel and every later question go
straight to the shape that works. Dropping rather than failing is right because
the parameter is an optimisation: a model that samples at its own temperature
still produces five samples, and the vote still has something to count.

What is deliberately *not* silent is the effect on reproducibility. A model that
will not take a seed cannot promise the same marks twice, and the caller is told
which parameters were refused so that a report can say so rather than implying a
determinism it does not have.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
from typing import Any

from ..clients import client_for
from ..observability import log_event

#: How many model calls this process will have in flight at once.
#:
#: Not a performance knob. Free and low tier quotas are per minute, and this
#: service fans out four questions at a time, each of which used to fan out to
#: five samples — twenty concurrent calls against an allowance of thirty a minute,
#: with nothing client-side to slow it down. Measured on a nine-document gate run,
#: that produced 53 dropped panel samples and a set of documents that scored zero
#: for reasons that had nothing to do with marking.
#:
#: A pilot on a free tier is exactly this shape, so this is a correctness setting
#: rather than a courtesy.
MAX_IN_FLIGHT = int(os.getenv("MODEL_MAX_IN_FLIGHT") or 2)

#: How many times a refused-for-rate request is waited out before giving up.
RATE_LIMIT_RETRIES = int(os.getenv("MODEL_RATE_LIMIT_RETRIES") or 5)

#: The longest single wait. A provider asking for twenty minutes is telling you
#: the daily budget is gone, and a submission should fail honestly rather than
#: hold a teacher's browser open until it arrives.
MAX_BACKOFF_SECONDS = float(os.getenv("MODEL_MAX_BACKOFF") or 90.0)

_in_flight: asyncio.Semaphore | None = None
_loop_owning_semaphore: object | None = None


def _gate() -> asyncio.Semaphore:
    """One semaphore per event loop.

    Created lazily rather than at import: a semaphore binds to the loop that made
    it, and this module is imported long before any loop exists — and the eval
    harness runs one `asyncio.run` per document, so there is more than one.
    """
    global _in_flight, _loop_owning_semaphore
    loop = asyncio.get_running_loop()
    if _in_flight is None or _loop_owning_semaphore is not loop:
        _in_flight = asyncio.Semaphore(MAX_IN_FLIGHT)
        _loop_owning_semaphore = loop
    return _in_flight


#: "Please try again in 21m24.336s" / "try again in 6.2s".
_RETRY_AFTER = re.compile(r"try again in\s+(?:(\d+)m)?([\d.]+)s", re.IGNORECASE)


#: A schema-validation 400 names the property the model dropped. Matched on the
#: provider's stable wording rather than on the status code alone, because a 400
#: is also what a genuinely malformed request returns — a bad model name, an
#: unsupported parameter — and retrying one of those is a loop.
_DECODE_SLIP = re.compile(
    r"does not (validate|match)|missing propert|failed_generation", re.IGNORECASE
)

#: How many times a dropped field is re-rolled. Two: a third attempt on a model
#: that cannot hold the schema is a slower failure, not a different one.
DECODE_RETRIES = int(os.getenv("MODEL_DECODE_RETRIES") or 2)


#: A daily exhaustion, as opposed to a per-minute burst.
#:
#: The two arrive as the same 429 and need opposite handling. A per-minute limit
#: clears in seconds and waiting is correct. A daily limit clears in hours: the
#: provider says "try again in 8m47s" only because that is when the rolling
#: window next admits one request, and waiting it out means a teacher's browser
#: holds open for a submission that will be rate limited again on the next
#: question. There is nothing to wait for.
_DAILY_LIMIT = re.compile(r"per day|\bTPD\b|\bRPD\b", re.IGNORECASE)

#: Which (provider, model) entries have spent their day, in this process.
#:
#: Sticky, because a spent allowance does not come back inside one process's
#: lifetime, and keyed by both halves because the same model on two hosts is two
#: separate budgets - which is the entire reason the chain is worth walking.
_spent: set[tuple[str, str]] = set()


def _is_a_daily_limit(error: Exception) -> bool:
    """Whether this 429 is the day's budget rather than this minute's burst."""
    return bool(_DAILY_LIMIT.search(str(error)))


def next_marker(after: tuple[str, str] | None = None) -> tuple[str, str] | None:
    """The next reachable (provider, model) the day has not been spent on.

    Walks `clients.marking_chain`, which is ordered by measured accuracy on the
    gate. Passing `after` continues past a specific entry rather than from the
    top, so a caller that has just been refused does not immediately retry the
    thing that refused it.
    """
    from ..clients import marking_chain

    chain = marking_chain()
    start = 0
    if after is not None and after in chain:
        start = chain.index(after) + 1
    for entry in chain[start:]:
        if entry not in _spent:
            return entry
    return None


def forget_spent_budgets() -> None:
    """For tests, and for a process that outlives a quota window."""
    _spent.clear()


def effective_marker(provider: str, model: str) -> tuple[str, str]:
    """The marker that will actually answer, after any budget already spent.

    A grade records what marked it, and after a fallback the configured entry is
    no longer that. Reporting the configured one would let two scripts marked by
    two different models claim the same provenance, which is exactly the
    comparison a teacher would make and exactly the one that would be wrong.
    """
    if (provider, model) not in _spent:
        return provider, model
    return next_marker((provider, model)) or (provider, model)


def _is_a_decode_slip(error: Exception) -> bool:
    """Whether the provider rejected the model's output rather than our request."""
    return bool(_DECODE_SLIP.search(str(error)))


def _wait_for(error: Exception, attempt: int) -> float | None:
    """How long to wait before retrying, or None if this is not a rate limit.

    The provider's own number is preferred over a guess, because it knows when the
    window rolls and a shorter guess just spends another request being refused.
    Jitter on the fallback so that four questions refused together do not all come
    back at the same instant and refuse each other again.
    """
    if "ratelimit" not in type(error).__name__.lower() and "429" not in str(error):
        return None

    match = _RETRY_AFTER.search(str(error))
    if match:
        minutes = float(match.group(1) or 0)
        seconds = minutes * 60 + float(match.group(2))
        return seconds if seconds <= MAX_BACKOFF_SECONDS else None

    return min(MAX_BACKOFF_SECONDS, (2**attempt) + random.random())


#: Parameters known to be refused, by model. Populated on the first refusal and
#: consulted before every later call, so one 400 is paid per model per process
#: rather than one per request.
_REFUSED: dict[str, set[str]] = {}

#: The optional parameters this module knows how to give up.
_OPTIONAL = ("temperature", "seed")


def refused_by(model: str) -> set[str]:
    """Which optional parameters this model has been seen to reject."""
    return set(_REFUSED.get(model, set()))


def forget() -> None:
    """Drop what was learned about every model. For tests."""
    _REFUSED.clear()


def _names_a_refused_parameter(error: Exception) -> str | None:
    """The optional parameter an error is complaining about, if it is one.

    Matched on the message because the provider expresses this several ways —
    "Unsupported value: 'temperature'", "Unsupported parameter: 'seed'",
    "Unrecognized request argument supplied: seed" — and all of them are 400s
    that name the parameter. Anything else is a real failure and is re-raised.
    """
    if "badrequest" not in type(error).__name__.lower():
        return None
    message = str(error).lower()
    if not any(
        phrase in message
        for phrase in ("unsupported value", "unsupported parameter", "unrecognized request")
    ):
        return None
    return next((name for name in _OPTIONAL if name in message), None)


async def _create(
    client: Any,
    *,
    model: str,
    provider: str = "",
    messages: list[dict],
    schema_name: str,
    schema: dict,
    temperature: float | None,
    seed: int | None,
) -> dict:
    """The request, retried without whatever the model turns out to refuse."""
    optional: dict[str, Any] = {}
    if temperature is not None:
        optional["temperature"] = temperature
    if seed is not None:
        optional["seed"] = seed

    attempt = 0
    if not provider:
        from ..clients import openai_provider

        provider = openai_provider()[0]
    exhausted: set[tuple[str, str]] = set()
    while True:
        sending = {k: v for k, v in optional.items() if k not in refused_by(model)}
        try:
            async with _gate():
                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                    **sending,
                )
            break
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it is ours
            refused = _names_a_refused_parameter(exc)
            if refused is not None and refused in sending:
                _REFUSED.setdefault(model, set()).add(refused)
                continue

            # A decode slip is a re-roll, not a failure. The provider validates
            # the model's output against our schema and returns 400 when it does
            # not fit — "missing properties: 'cited_line_ids'". That is the
            # model dropping a field on one sample, not a request this service
            # got wrong, and the fix for a bad sample is another sample.
            #
            # It mattered because the panel used to absorb this and no longer
            # does. At five samples a dropped field cost one vote; at one it
            # costs the question, which comes back "never judged" and reads to a
            # teacher as an answer nobody looked at. A false absence is the worst
            # error this product makes, and losing a question to a JSON field is
            # the cheapest possible way to cause one.
            #
            # Retried at a raised temperature, because a re-roll at the same
            # settings on a near-deterministic decode reproduces the same slip.
            if _is_a_decode_slip(exc) and attempt < DECODE_RETRIES:
                attempt += 1
                optional["temperature"] = min(
                    1.0, (optional.get("temperature") or 0.0) + 0.3
                )
                continue

            # A spent day is a different marker, not a longer wait. Waiting out
            # a daily limit means holding a teacher's browser open for a
            # submission that will be refused again on the very next question,
            # while an untouched allowance sits one entry down the chain.
            #
            # The next entry may be on a different host, so the client is
            # rebuilt rather than reused. That is the point of keying the spent
            # set by (provider, model): free tiers meter per model *and* per
            # host, so the same model on a second host is a second budget, and
            # crossing to it is how capacity is bought without losing accuracy.
            if _is_a_daily_limit(exc):
                _spent.add((provider, model))
                spare = next_marker((provider, model))
                if spare is not None and spare not in exhausted:
                    exhausted.add(spare)
                    log_event(
                        "model_budget_spent",
                        provider=provider,
                        model=model,
                        falling_back_to=f"{spare[0]}:{spare[1]}",
                        detail=str(exc),
                    )
                    if spare[0] != provider:
                        client = client_for(spare[0])
                    provider, model = spare
                    attempt = 0
                    continue
                raise

            # A rate limit is a wait, not a failure. Left unhandled it became one:
            # the question came back "could not be marked automatically", the
            # document scored zero, and the gate reported it as a marking result.
            wait = _wait_for(exc, attempt) if attempt < RATE_LIMIT_RETRIES else None
            if wait is None:
                raise
            attempt += 1
            await asyncio.sleep(wait)

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("the model returned no judgement")
    return json.loads(content)


async def structured_completion(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    schema_name: str,
    schema: dict,
    provider: str = "",
    temperature: float | None = None,
    seed: int | None = None,
) -> dict:
    """One JSON object matching ``schema``, from whichever call shape works."""
    return await _create(
        client,
        model=model,
        provider=provider,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        schema_name=schema_name,
        schema=schema,
        temperature=temperature,
        seed=seed,
    )


async def structured_completion_with_image(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    image_png: bytes,
    schema_name: str,
    schema: dict,
    provider: str = "",
    temperature: float | None = None,
    seed: int | None = None,
) -> dict:
    """The same, with one image attached to the user's turn.

    Inline as a data URL rather than by reference. The crop exists only in memory
    for the length of this call, and uploading a student's handwriting somewhere
    to obtain a URL for it would give the page a second life nobody asked for and
    nothing deletes.
    """
    encoded = base64.b64encode(image_png).decode("ascii")
    return await _create(
        client,
        model=model,
        provider=provider,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ],
        schema_name=schema_name,
        schema=schema,
        temperature=temperature,
        seed=seed,
    )
