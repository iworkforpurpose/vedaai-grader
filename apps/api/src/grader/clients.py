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


def openai_kwargs() -> dict[str, Any]:
    """Timeout and retry settings for an OpenAI client."""
    return {"timeout": MODEL_TIMEOUT, "max_retries": MODEL_RETRIES}


def anthropic_kwargs() -> dict[str, Any]:
    """Timeout and retry settings for an Anthropic client."""
    return {"timeout": MODEL_TIMEOUT, "max_retries": MODEL_RETRIES}
