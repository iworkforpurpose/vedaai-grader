"""Turn a /health payload into the four values the release step reports.

A file rather than a `python3 -c` one-liner in the workflow, because the one-liner
version did not survive its own quoting. It was four separate invocations with
escaped double quotes inside a YAML string inside a shell command substitution,
and the one that needed an f-string failed silently — so the release summary
reported `grading: unknown` while the payload it had just fetched plainly said
`openai:gpt-4.1`. A `|| echo unknown` fallback that fires because of quoting is
worse than no fallback at all, because it looks like an answer.

Reads the payload on stdin, writes five lines on stdout:

    marking     yes | no
    marker      engine:model
    placing     semantic | lexical | unknown
    why         the provider's own message, on one line, or empty
    throttled   yes | no

The last line is the difference between a broken release and a busy one. A
rejected key and a spent quota both answer `marking: no`, and they need opposite
responses: one is a misconfiguration that nothing will fix until somebody changes
a secret, the other is a correct deployment whose free-tier window has rolled
over. Without the distinction the release step has to choose which of the two to
get wrong, and both choices are bad — block every deploy once a day, or ship a
genuinely dead marker green.

Exits zero whatever it is given. The release step decides what to do about the
values; this only has to be unable to lie about them.
"""

from __future__ import annotations

import json
import sys


def summarise(payload: str) -> list[str]:
    try:
        health = json.loads(payload)
    except (ValueError, TypeError):
        health = {}
    if not isinstance(health, dict):
        health = {}

    grading = health.get("grading")
    if not isinstance(grading, dict):
        grading = {}

    return [
        # `reachable`, not `configured`. A key that exists is not a provider that
        # answers, and the whole reason this check exists is the difference: a
        # deployment whose account had run out of credit reported `configured:
        # true` and shipped green while every submission came back zeros.
        "yes" if grading.get("reachable") else "no",
        f"{grading.get('engine') or '?'}:{grading.get('model') or '-'}",
        str(health.get("similarity") or "unknown"),
        # Collapsed to one line: the caller reads these with five `read` calls,
        # and a multi-line provider message would shift every field after it.
        " ".join(str(grading.get("detail") or "").split()),
        "yes" if grading.get("throttled") else "no",
    ]


if __name__ == "__main__":
    print("\n".join(summarise(sys.stdin.read())))
