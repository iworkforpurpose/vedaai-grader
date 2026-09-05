"""How long this service is willing to wait, in one place.

Every external client was constructed bare. No `botocore.Config` existed anywhere
in the tree, and the model clients took the SDK defaults — which for both OpenAI
and Anthropic is **600 seconds**.

That number is the whole problem. Marking runs four questions at a time
(`run.CONCURRENCY`), each fanning out to five panel samples, so one stalled
provider holds a submission at `processing` for ten minutes per wave with no
reaper and nothing logged. A teacher watches a spinner; an operator sees an idle
task. Nothing in the service disagrees with waiting.

AWS was worse in a quieter way. `ocr/textract.py` translates every failure —
including `ThrottlingException` and `ProvisionedThroughputExceededException`, both
of which carry a hand-written "try again" message — into a terminal
`EngineUnavailable`, and nothing retried. A throttle on one page ends the
transcription of the whole document.

The numbers here are chosen against what the work actually takes, not against a
round figure:

* **Textract** answers a page in a second or two. Thirty seconds is a page that is
  not coming, and adaptive retries are what turn a throttle into a pause rather
  than a lost document.
* **A marking call** is a short structured completion. A minute is generous; ten
  minutes is a hang.
* **Embeddings** are one batched round trip that the aligner blocks on, and the
  scorer already has its own outage cooldown, so a short timeout costs a degraded
  placement rather than a failed one.
"""

from __future__ import annotations

import os
from typing import Any

#: Seconds to wait for a connection, everywhere. A TCP handshake that takes longer
#: than this is a network that is not going to serve a document either.
CONNECT_TIMEOUT = 5.0

#: Textract, per page.
TEXTRACT_READ_TIMEOUT = 30.0

#: S3 and DynamoDB. Longer than Textract because a spilled submission payload can
#: be a few hundred kilobytes and page images are larger still.
STORAGE_READ_TIMEOUT = 60.0

#: One marking, scheme or re-read call.
MODEL_TIMEOUT = float(os.getenv("MODEL_TIMEOUT_SECONDS") or 60.0)

#: How many times a model call is retried by the SDK before it reaches us. Two,
#: because the panel already tolerates a lost sample and the per-question guard
#: already tolerates a lost question; the retries are for a blip, not for an
#: outage.
MODEL_RETRIES = 2


def aws_config(read_timeout: float = STORAGE_READ_TIMEOUT) -> Any:
    """Timeouts and adaptive retries for a boto3 client.

    `adaptive` rather than `standard`: it rate-limits client-side on a throttle
    instead of retrying straight into the same wall, which is the behaviour a
    per-page loop over sixty pages needs.
    """
    from botocore.config import Config

    return Config(
        connect_timeout=CONNECT_TIMEOUT,
        read_timeout=read_timeout,
        retries={"mode": "adaptive", "max_attempts": 5},
    )


#: Where an OpenAI-shaped client should point, and what key it should carry.
#:
#: Every provider worth using for this speaks the OpenAI chat-completions API, so
#: "which provider" is a base URL and a key rather than a second code path. The
#: marking call is the same call either way: a system prompt, one message, and a
#: strict JSON schema.
#:
#: Groq is chosen when its key is present because it is the one open-weight host
#: confirmed to support `response_format: json_schema` with `strict: true`, which
#: every marking call here depends on — and its schema rules (all fields required,
#: `additionalProperties: false`, nullables as unions) are the ones this codebase
#: already emits, so nothing had to be rewritten to move.
#:
#: An explicit `GRADER_BASE_URL` overrides the lot, so any other OpenAI-compatible
#: host is a config change rather than a release.
PROVIDERS: dict[str, tuple[str, str]] = {
    # name: (env var holding the key, base URL)
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    "cerebras": ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1"),
    # Google's OpenAI-compatible endpoint. Here for one reason, which is that its
    # free tier is metered in requests per day rather than tokens per day, and
    # marking is many small requests. That difference is worth more than it
    # sounds: an allowance of 1,500 requests buys roughly seventeen times the
    # marking of a 200,000-token one, because a marking call is about 2,300
    # tokens and a token budget is spent long before a request budget is.
    "gemini": (
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "openai": ("OPENAI_API_KEY", ""),
}

#: What each host's free tier is worth, in scripts a day.
#:
#: Allowances are quoted in units that are not comparable by eye - one host
#: meters tokens per day, another requests per day - so they are converted here.
#: A marking call is about 4,400 tokens and an eighteen-question script at five
#: samples is ninety calls, so the same paper costs ~396,000 tokens or 90
#: requests depending on who is counting.
#:
#: **These numbers have been wrong before and will be again.** A provider's
#: published free tier said 1,500 requests a day; the account's actual limit for
#: that model was twenty, and only a 429 said so. So this table decides the
#: *order* markers are tried in and nothing else. Whether marking survives a
#: spent allowance is decided by the chain walking on when it meets one, which
#: does not consult these numbers at all. Getting them wrong costs a slower first
#: attempt, never a failed submission.
#:
#: `rpm` is here because it decides how long one script takes rather than how
#: many fit in a day, and the two pull in opposite directions.
FREE_TIER: dict[str, dict[str, float]] = {
    # Measured by trying, not by reading. Two full gate attempts on two Google
    # models produced 153 and 97+ rate-limit errors and could not finish either
    # run, so the figure below is an upper bound on a good day rather than a
    # capacity anyone should plan against.
    "gemini": {"scripts_per_day": 3.0, "rpm": 10},
    # 1,000,000 tokens a day against Groq's 200,000 per model, and it serves the
    # only marker measured at eight of nine. Read from the provider's own
    # `x-ratelimit-*` response headers rather than from a documentation page,
    # which is the one source that cannot be out of date.
    #
    # Tokens bind well before requests here: the daily request budget is 2,400
    # against roughly ninety calls a script, but the token budget runs out at
    # about five. The five-a-minute cap is separate again and decides how long
    # one script takes rather than how many fit in a day - around eighteen
    # minutes, which is fine for a queue and slow for somebody watching.
    "cerebras": {"scripts_per_day": 5.5, "rpm": 5},
    "groq": {"scripts_per_day": 2.0, "rpm": 30},
    "openai": {"scripts_per_day": 0.0, "rpm": 60},
}


#: Markers that hold a strict JSON schema, and what the gate measured.
#:
#: Membership is earned by scoring, not by allowance. A host with a large free
#: tier and a bad marker is worse than no host, because the marks look the same
#: either way. Every model here has been confirmed to accept
#: `response_format: json_schema` with `strict: true`, which is the property that
#: makes a weak model safe: it fills a schema rather than writing prose, so
#: malformed output is impossible and invented citations are refused downstream.
#:
#: `in_band` is documents inside their band out of nine, five samples per
#: question, from a run with no rate limiting. `None` means never measured, which
#: sorts below every measured entry: an unmeasured marker is not a bad one, but
#: it is not evidence either.
MEASURED: dict[tuple[str, str], int | None] = {
    # Six of nine, five samples, a complete run with no rate limiting. The three
    # misses are all under-marking, and one of them - economics - is a known
    # aligner fault rather than a marking one: the student labelled their working
    # `Q4` in the margin and question 3 gets nothing.
    ("cerebras", "gpt-oss-120b"): 6,
    # Eight of nine, but from an earlier session, and the same weights on another
    # host scored six here. Two measurements from different sessions are not
    # comparable in this project - the scorer depends on a hosted embedding
    # service, so the golden set is not a closed system - and this one is kept
    # only so the entry stays in the scored tier. It needs re-running before
    # anybody quotes it.
    ("groq", "openai/gpt-oss-120b"): 8,
    # Marks correctly when it gets through: on the documents that were not rate
    # limited it landed in band. Neither model could complete a gate run on the
    # free tier, so neither has a score, and without one they cannot lead.
    ("gemini", "gemini-2.5-flash"): None,
    ("gemini", "gemini-flash-lite-latest"): None,
    ("gemini", "gemini-3.1-flash-lite"): None,
    ("gemini", "gemini-2.5-flash-lite"): None,
    ("groq", "qwen/qwen3.8-27b"): 4,
    ("cerebras", "qwen-3.8-27b"): None,
    ("groq", "openai/gpt-oss-20b"): 4,
}

#: The minimum gate score a marker needs before it leads the chain.
#:
#: Six of nine. The bar is about what the misses are, not only how many.
#:
#: At six the failures are close calls in the safe direction - under-marking by a
#: mark or three, on questions the marker did engage with - and one of them is a
#: known aligner fault rather than a marking one. At four they are not:
#: `gpt-oss-20b` puts two answers a student earned marks for at zero, and a false
#: zero is the worst error this product makes. No amount of free allowance pays
#: for one, which is why the bar sits between those two results rather than at a
#: round number.
MIN_IN_BAND = int(os.getenv("GRADER_MIN_IN_BAND") or 6)


def _rank(entry: tuple[str, str]) -> tuple[int, float, str, str]:
    """Sort key: accuracy tier first, then free allowance, then a stable name.

    Two levels rather than one. Ordering purely by allowance would put a weaker
    marker in front of a better one the moment somebody's free tier grew;
    ordering purely by accuracy would waste the largest allowance on the entry
    least able to use it. So a gate score decides who leads, and among markers
    that measured the same, the one that buys the most scripts a day goes first.

    Everything unmeasured shares the lower tier and is ordered by allowance
    within it, which is the right default for a marker nobody has scored yet: try
    the one there is most of.
    """
    provider, model = entry
    scored = MEASURED.get(entry)
    tier = 0 if scored is not None and scored >= MIN_IN_BAND else 1
    allowance = FREE_TIER.get(provider, {}).get("scripts_per_day", 0.0)
    return (tier, -allowance, provider, model)


#: Where marking goes, in order, until something answers.
#:
#: Every entry whose host has a key is tried before marking gives up, so a
#: submission fails only once every allowance is spent. The order is computed -
#: see `_rank` - so recording a new measurement re-sorts it rather than requiring
#: somebody to re-sort a list by hand and get it subtly wrong.
FALLBACK_CHAIN: list[tuple[str, str]] = sorted(MEASURED, key=_rank)


def configured_providers() -> set[str]:
    """Which hosts this deployment actually has a key for."""
    return {
        name
        for name, (env_var, _base) in PROVIDERS.items()
        if os.getenv(env_var, "").strip()
    }


def base_url_for(provider: str) -> str | None:
    """Where a provider's OpenAI-shaped API lives."""
    override = os.getenv("GRADER_BASE_URL", "").strip()
    if override:
        return override
    _env, base = PROVIDERS.get(provider, ("", ""))
    return base or None


def key_for(provider: str) -> str | None:
    env_var, _base = PROVIDERS.get(provider, ("", ""))
    return os.getenv(env_var, "").strip() or None


def marking_chain() -> list[tuple[str, str]]:
    """The (provider, model) entries this deployment can actually reach.

    Pinning either half collapses the chain to what was pinned. That is
    deliberate: a measurement is only worth anything if it names one marker, and
    the eval harness pins `GRADER_MODEL` for exactly that reason.
    """
    pinned_model = os.getenv("GRADER_MODEL", "").strip()
    pinned_host = os.getenv("GRADER_PROVIDER", "").strip().lower()
    have = configured_providers()

    chain = [
        (provider, model)
        for provider, model in FALLBACK_CHAIN
        if provider in have
        and (not pinned_host or provider == pinned_host)
        and (not pinned_model or model == pinned_model)
    ]
    if pinned_model and not chain:
        # A model this table has never heard of. Honour it on whichever host is
        # configured rather than silently marking with something else: a run that
        # asked for one marker and measured another is worse than a run that
        # fails.
        host = pinned_host or next(
            (p for p in ("groq", "cerebras", "openai") if p in have), "openai"
        )
        return [(host, pinned_model)]
    return chain


def openai_provider() -> tuple[str, str | None, str | None]:
    """Which OpenAI-shaped provider to use: (name, api_key, base_url).

    Groq first when its key is present, because it is roughly thirteen times
    cheaper per script than the OpenAI default and this product marks a class of
    forty at a time. `GRADER_PROVIDER` still wins where it names one.
    """
    forced = os.getenv("GRADER_PROVIDER", "").strip().lower()
    order = ["groq", "openai"]
    if forced in PROVIDERS:
        order = [forced]

    override = os.getenv("GRADER_BASE_URL", "").strip()
    for name in order:
        env_var, base = PROVIDERS[name]
        key = os.getenv(env_var, "").strip()
        if key:
            return name, key, override or base or None
    return "openai", None, override or None


def openai_kwargs() -> dict[str, Any]:
    """Timeout, retry and destination settings for an OpenAI-shaped client."""
    _name, key, base = openai_provider()
    kwargs: dict[str, Any] = {"timeout": MODEL_TIMEOUT, "max_retries": MODEL_RETRIES}
    if key:
        kwargs["api_key"] = key
    if base:
        kwargs["base_url"] = base
    return kwargs


def anthropic_kwargs() -> dict[str, Any]:
    """Timeout and retry settings for an Anthropic client."""
    return {"timeout": MODEL_TIMEOUT, "max_retries": MODEL_RETRIES}


def client_for(provider: str) -> Any:
    """An OpenAI-shaped async client pointed at one specific host.

    Needed because the marking chain can cross hosts mid-submission: a client
    carries its key and base URL, so continuing on the next entry after a spent
    allowance means a different client, not a different argument.
    """
    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {"timeout": MODEL_TIMEOUT, "max_retries": MODEL_RETRIES}
    key = key_for(provider)
    base = base_url_for(provider)
    if key:
        kwargs["api_key"] = key
    if base:
        kwargs["base_url"] = base
    return AsyncOpenAI(**kwargs)
