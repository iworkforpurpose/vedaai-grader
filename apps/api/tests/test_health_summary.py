"""The release step's reading of /health.

Tested because the previous version of this logic was four `python3 -c` calls
with escaped quotes inside a YAML string inside a shell substitution, and it was
wrong in production without anybody noticing: the summary reported
`grading: unknown` while the payload it had just fetched said `openai:gpt-4.1`.
Nothing could have caught that, because there was nothing to point a test at.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "deploy"))

from health_summary import summarise  # noqa: E402


def test_a_provider_that_refused_reports_no_and_says_why() -> None:
    """The case that was live: a valid key against an empty account."""
    payload = json.dumps(
        {
            "grading": {
                "configured": True,
                "engine": "openai",
                "model": "gpt-4.1",
                "reachable": False,
                "detail": "Error code: 429 - You have no credits remaining.",
            },
            "similarity": "semantic",
        }
    )

    marking, marker, placing, why = summarise(payload)

    assert marking == "no"
    assert marker == "openai:gpt-4.1"
    assert placing == "semantic"
    assert "no credits remaining" in why


def test_a_working_provider_reports_yes_with_nothing_to_explain() -> None:
    payload = json.dumps(
        {
            "grading": {
                "configured": True,
                "engine": "openai",
                "model": "gpt-4.1",
                "reachable": True,
                "detail": None,
            },
            "similarity": "semantic",
        }
    )

    assert summarise(payload) == ["yes", "openai:gpt-4.1", "semantic", ""]


def test_configured_without_reachable_is_not_good_enough() -> None:
    """A key that exists is not a provider that answers.

    This distinction is the entire reason the check exists, and reading the wrong
    field is how the first version of it passed a deployment that could not mark.
    """
    payload = json.dumps(
        {"grading": {"configured": True, "engine": "openai", "model": "gpt-4.1"}}
    )

    assert summarise(payload)[0] == "no"


def test_a_multi_line_message_is_collapsed() -> None:
    """The caller reads these with four `read` calls.

    A provider message containing a newline would shift every field after it, so
    the failure detail would be interpreted as the similarity mode.
    """
    payload = json.dumps(
        {"grading": {"reachable": False, "detail": "Error 429 -\n  no credits\n  left"}}
    )

    assert summarise(payload)[3] == "Error 429 - no credits left"
    assert len(summarise(payload)) == 4


def test_nothing_it_is_given_can_make_it_raise() -> None:
    """A curl that timed out yields `{}`; a gateway error yields HTML.

    Neither is a reason for the release step to die before it has reported
    anything at all.
    """
    for payload in ["", "{}", "not json", "[]", "null", '{"grading": "surprise"}']:
        assert len(summarise(payload)) == 4
