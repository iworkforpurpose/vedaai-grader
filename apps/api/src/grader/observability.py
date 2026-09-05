"""One JSON object per line, so an operator can answer a question.

There was no logging in this service at all — not a `logging` import, not a
`print`. Everything the pipeline knew about its own failures went into
``submission.warnings``, which lives in DynamoDB behind a seven-day TTL and is
read by exactly one screen.

The consequences were not subtle. "How many submissions failed to mark today" was
unanswerable without enumerating submissions and reading prose. A provider that
had run out of credit produced a green deploy, a `complete` submission, and marks
of zero, with the reason visible only to whoever happened to open that one script.
A stalled provider left submissions at `processing` for as long as the SDK's
ten-minute default allowed, with nothing to look at meanwhile.

**Structured, not prose.** These lines are read by a machine before they are read
by a person: the point of them is a CloudWatch metric filter that can count
`marking_failed` without matching English. A log line whose shape changes when
somebody rewords it is a metric that silently goes to zero.

**Correlated.** Every line carries the ``submission_id`` where there is one, which
until now never reached a log at all — so two failures in the same run could not
be told from two failures in two runs.

**No secrets, and no student writing.** Field values are truncated, credential-
shaped substrings are redacted, and the transcribed text of an answer is never a
field. The redaction is not belt-and-braces: providers echo the credential back in
their own error messages, and those messages are the most useful thing to log. A
dead key produces, verbatim,

    Incorrect API key provided: sk-abc123... You can find your API key at ...

so logging provider errors without redaction writes the key to CloudWatch, where
it outlives the submission, the task and the key rotation that was supposed to
end it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from typing import Any

#: The longest a single field is allowed to be.
#:
#: Provider errors carry whole JSON bodies and OCR text can be a page long. A log
#: line is a signal, not an archive, and an unbounded field turns one bad
#: submission into a log bill.
MAX_FIELD = 300

#: Credential shapes that appear inside provider error messages.
#:
#: Only shapes with a distinctive prefix and no plausible false positive. A
#: pattern loose enough to catch every conceivable secret would redact question
#: text and make the logs useless, which is the failure mode in the other
#: direction.
_SECRETS = re.compile(
    r"""(
        sk-[A-Za-z0-9_\-]{8,}          # OpenAI
      | sk-ant-[A-Za-z0-9_\-]{8,}      # Anthropic
      | AKIA[0-9A-Z]{16}               # AWS access key id
      | ASIA[0-9A-Z]{16}               # AWS session key id
      | Bearer\s+[A-Za-z0-9._\-]{12,}  # any bearer token
    )""",
    re.VERBOSE,
)

_logger = logging.getLogger("grader")


def configure(stream=None) -> None:
    """Send our lines to stdout, once, without touching anybody else's."""
    if _logger.handlers:
        return
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    # Ours only. Propagating would duplicate every line into uvicorn's handler,
    # which formats it as prose and defeats the point.
    _logger.propagate = False


def redact(text: str) -> str:
    """Replace credential-shaped substrings, keeping enough to identify which."""
    return _SECRETS.sub(lambda m: f"{m.group(0)[:6]}...[redacted]", text)


#: Fragments worth keeping even when the field around them is trimmed.
#:
#: A provider's error message puts the useful part last. Google's 429 opens with
#: three lines of documentation links and names the quota that was actually
#: exceeded at around character 340, so a 300-character field logged the boiler-
#: plate and cut the answer off. That difference is "rate limited, unclear why"
#: against "GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit 20" - which
#: is the whole diagnosis, and it took a direct probe to recover.
_KEEP = re.compile(
    r"(quota[A-Za-z]*\s*[:=]\s*\S+|limit:?\s*\d+|Limit \d+|Used \d+"
    r"|per (?:day|minute)|TP[DM]|RP[DM]|retry in [\d.ms]+|try again in [\d.ms]+)",
    re.IGNORECASE,
)


def _trim(text: str) -> str:
    """Trim to ``MAX_FIELD``, keeping the diagnostic fragments.

    A log line is a signal rather than an archive, so the cap stays. What changes
    is which end of the message survives: the head is kept for context, and any
    quota or limit the tail named is appended, because that is the part somebody
    reading this at three in the morning is looking for.
    """
    if len(text) <= MAX_FIELD:
        return text
    kept = " ".join(dict.fromkeys(_KEEP.findall(text)))
    if not kept:
        return text[:MAX_FIELD]
    head = text[: max(0, MAX_FIELD - len(kept) - 5)]
    return f"{head} ... {kept}"[: MAX_FIELD + len(kept) + 5]


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        collapsed = redact(" ".join(value.split()))
        return _trim(collapsed)
    if isinstance(value, float):
        return round(value, 4)
    return value


def log_event(event: str, *, submission_id: str | None = None, **fields: Any) -> None:
    """Record that something happened, in a shape a metric filter can count.

    ``event`` is a stable identifier and never a sentence. Everything variable is
    a field, so `{ "event": "marking_failed" }` remains countable however the
    human-facing message is later reworded.
    """
    payload: dict[str, Any] = {"event": event, "ts": round(time.time(), 3)}
    if submission_id:
        payload["submission_id"] = submission_id
    payload.update({k: _clean(v) for k, v in fields.items() if v is not None})
    _logger.info(json.dumps(payload, default=str))


@contextmanager
def timed(event: str, *, submission_id: str | None = None, **fields: Any):
    """Log how long a stage took, and whether it finished.

    The failure path logs too. A stage that raised is the interesting one, and a
    timer that only reports success is a timer that goes quiet exactly when
    somebody starts looking at it.
    """
    started = time.monotonic()
    try:
        yield
    except BaseException as exc:
        log_event(
            f"{event}_failed",
            submission_id=submission_id,
            seconds=time.monotonic() - started,
            error=type(exc).__name__,
            detail=str(exc),
            **fields,
        )
        raise
    else:
        log_event(
            event,
            submission_id=submission_id,
            seconds=time.monotonic() - started,
            **fields,
        )
