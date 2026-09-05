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

#: Where marking goes, in order, until something answers.
#:
#: A chain of (provider, model) rather than a single choice, because on a free
#: tier the binding constraint is not price per token but a daily allowance, and
#: allowances are metered **per model and per provider**. The same model on two
#: hosts is two budgets; two models on one host is also two budgets. Refusing to
#: mark while an untouched allowance sits one entry down is the only genuinely
#: wrong answer available here.
#:
#: Ordered by measured accuracy on the nine-document gate, never by speed or
#: cost, because every entry below the first is a worse marker and the point of
#: the order is to reach them as late as possible. What the chain buys is
#: capacity at the top of it: the same accurate model on a second host is more
#: scripts a day at unchanged accuracy, which is the only kind of extra capacity
#: worth having.
#:
#: Entries whose provider has no key are skipped, so this is also how a
#: deployment says which hosts it has. `GRADER_MODEL` still pins one model and
#: `GRADER_PROVIDER` still pins one host, both of which collapse the chain.
FALLBACK_CHAIN: list[tuple[str, str]] = [
    # Measured on the nine-document gate, five samples per question, and the two
    # entries are the same weights on two hosts - so reaching the second costs
    # nothing but latency. `gpt-oss-120b` is the only marker measured at 8 of 9.
    ("cerebras", "gpt-oss-120b"),
    ("groq", "openai/gpt-oss-120b"),
    # Unmeasured, and here for capacity rather than for judgement. Its free tier
    # is metered in requests rather than tokens, which is the shape marking
    # actually has, so it is the largest permanently-free allowance available.
    # It sits below the measured entries until somebody runs the gate on it.
    ("gemini", "gemini-3-flash-preview"),
    # Measured at 4 of 9 at one sample, with six questions lost to rate limiting
    # rather than judged - so that figure is a floor, not a verdict.
    ("groq", "qwen/qwen3.8-27b"),
    ("cerebras", "qwen-3.8-27b"),
    # Measured at 4 of 9, complete run, every miss under-marking and two of them
    # earned answers scoring zero. Last on purpose.
    ("groq", "openai/gpt-oss-20b"),
]


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
