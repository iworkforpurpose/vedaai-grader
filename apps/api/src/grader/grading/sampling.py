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

import base64
import json
from typing import Any

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

    while True:
        sending = {k: v for k, v in optional.items() if k not in refused_by(model)}
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
                **sending,
            )
            break
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it is ours
            refused = _names_a_refused_parameter(exc)
            if refused is None or refused not in sending:
                raise
            _REFUSED.setdefault(model, set()).add(refused)

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
    temperature: float | None = None,
    seed: int | None = None,
) -> dict:
    """One JSON object matching ``schema``, from whichever call shape works."""
    return await _create(
        client,
        model=model,
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
