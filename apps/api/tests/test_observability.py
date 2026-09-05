"""The lines an operator reads when something has gone wrong.

There were none. Every failure this service knew about went into
`submission.warnings`, which lives behind a seven-day TTL and is read by one
screen — so "how many submissions failed to mark today" was unanswerable without
enumerating submissions and reading prose.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from grader import observability


@pytest.fixture(autouse=True)
def stream() -> io.StringIO:
    """A fresh logger writing somewhere the test can read."""
    logger = logging.getLogger("grader")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    buffer = io.StringIO()
    observability.configure(stream=buffer)
    return buffer


def lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


class TestTheShapeIsCountable:
    def test_an_event_is_one_json_object_per_line(self, stream) -> None:
        observability.log_event("marking_failed", submission_id="abc", questions=3)

        assert lines(stream) == [
            {
                "event": "marking_failed",
                "ts": lines(stream)[0]["ts"],
                "submission_id": "abc",
                "questions": 3,
            }
        ]

    def test_the_event_name_carries_no_prose(self, stream) -> None:
        """A metric filter counts `marking_failed`, not a sentence.

        A log line whose shape changes when somebody rewords the human-facing
        message is a metric that silently goes to zero.
        """
        observability.log_event(
            "marking_failed", submission_id="abc", reason="the account is out of credit"
        )

        entry = lines(stream)[0]
        assert entry["event"] == "marking_failed"
        assert entry["reason"] == "the account is out of credit"

    def test_a_submission_id_is_carried_wherever_there_is_one(self, stream) -> None:
        """Until now it never reached a log at all, so two failures in one run
        could not be told from two failures in two runs."""
        observability.log_event("ingest_started", submission_id="abc123")

        assert lines(stream)[0]["submission_id"] == "abc123"

    def test_an_event_without_a_submission_omits_the_field(self, stream) -> None:
        observability.log_event("service_started")

        assert "submission_id" not in lines(stream)[0]


class TestNothingUnboundedReachesTheLog:
    def test_a_long_field_is_truncated(self, stream) -> None:
        """Provider errors carry whole JSON bodies and OCR text can be a page.

        A log line is a signal, not an archive.
        """
        observability.log_event("marking_failed", detail="x" * 5000)

        assert len(lines(stream)[0]["detail"]) == observability.MAX_FIELD

    def test_a_multi_line_field_becomes_one_line(self, stream) -> None:
        """One JSON object per line is the whole contract with the log reader."""
        observability.log_event("marking_failed", detail="Error 429 -\n  no credits\n")

        assert lines(stream)[0]["detail"] == "Error 429 - no credits"
        assert len(stream.getvalue().strip().splitlines()) == 1

    def test_a_field_that_is_none_is_left_out(self, stream) -> None:
        observability.log_event("graded", submission_id="abc", model=None, questions=2)

        entry = lines(stream)[0]
        assert "model" not in entry
        assert entry["questions"] == 2


class TestTimingAStage:
    def test_it_reports_how_long_a_stage_took(self, stream) -> None:
        with observability.timed("ingest", submission_id="abc", pages=3):
            pass

        entry = lines(stream)[0]
        assert entry["event"] == "ingest"
        assert entry["pages"] == 3
        assert entry["seconds"] >= 0

    def test_a_stage_that_raised_still_reports(self, stream) -> None:
        """The interesting case. A timer that only reports success goes quiet
        exactly when somebody starts looking at it."""
        with pytest.raises(ValueError), observability.timed("ingest", submission_id="abc"):
            raise ValueError("no pages")

        entry = lines(stream)[0]
        assert entry["event"] == "ingest_failed"
        assert entry["error"] == "ValueError"
        assert entry["detail"] == "no pages"

    def test_the_exception_is_not_swallowed(self, stream) -> None:
        with pytest.raises(KeyError), observability.timed("ingest"):
            raise KeyError("gone")


class TestConfiguration:
    def test_configuring_twice_does_not_double_every_line(self, stream) -> None:
        observability.configure(stream=stream)
        observability.log_event("once")

        assert len(lines(stream)) == 1


class TestCredentialsDoNotReachTheLog:
    """Providers echo the credential back in their own error messages.

    A dead OpenAI key produces, verbatim: "Incorrect API key provided: sk-abc123
    ... You can find your API key at ...". Those messages are the single most
    useful thing to log — they say *why* — so redaction is not belt-and-braces.
    Without it, logging provider errors writes the key to CloudWatch, where it
    outlives the submission, the task, and the rotation that was supposed to end
    it.
    """

    def test_an_openai_key_echoed_by_the_provider_is_redacted(self, stream) -> None:
        observability.log_event(
            "similarity_degraded",
            detail=(
                "Error code: 401 - Incorrect API key provided: "
                "sk-proj-abc123def456ghi789. You can find your API key at ..."
            ),
        )

        detail = lines(stream)[0]["detail"]
        assert "sk-proj-abc123def456ghi789" not in detail
        assert "[redacted]" in detail
        # Enough left to say which credential it was talking about.
        assert "sk-pro" in detail

    def test_other_credential_shapes_are_redacted(self, stream) -> None:
        for secret in [
            "sk-ant-api03-XYZabc123456",
            "AKIAIOSFODNN7EXAMPLE",
            "Bearer eyJhbGciOiJIUzI1NiJ9",
        ]:
            observability.log_event("marking_failed", detail=f"refused: {secret} bad")

        for entry, secret in zip(
            lines(stream),
            ["sk-ant-api03-XYZabc123456", "AKIAIOSFODNN7EXAMPLE", "eyJhbGciOiJIUzI1NiJ9"],
            strict=True,
        ):
            assert secret not in entry["detail"]
            assert "[redacted]" in entry["detail"]

    def test_ordinary_text_is_left_alone(self, stream) -> None:
        """A pattern loose enough to catch every conceivable secret would redact
        question text and make the logs useless — the failure in the other
        direction."""
        observability.log_event(
            "marking_failed",
            detail="Define resistance and state its SI unit. Sketch the field lines.",
        )

        assert "[redacted]" not in lines(stream)[0]["detail"]

    def test_redaction_runs_before_truncation(self, stream) -> None:
        """Otherwise a key at character 290 survives in the kept prefix."""
        observability.log_event("marking_failed", detail="x" * 250 + " sk-abc123def456ghi")

        assert "sk-abc123def456ghi" not in lines(stream)[0]["detail"]


class TestATruncatedErrorStillNamesTheQuota:
    """The useful half of a provider error is at the end of it.

    Google's 429 opens with three lines of documentation links and names the
    quota that was actually exceeded around character 340. A flat 300-character
    field logged the boilerplate and cut the answer off, which is the difference
    between "rate limited, unclear why" and "requests per day per model, limit
    20" - and recovering it took a direct probe against the provider.
    """

    GOOGLE_429 = (
        "Error code: 429 - You exceeded your current quota, please check your plan "
        "and billing details. For more information on this error, head to: "
        "https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current "
        "usage, head to: https://ai.dev/rate-limit. * Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 20, model: gemini-3-flash. Please retry in 17.5s."
    )

    def test_the_quota_survives_the_cut(self):
        from grader import observability

        line = observability._clean(self.GOOGLE_429)
        assert "limit: 20" in line
        assert "quota" in line.lower()

    def test_the_head_of_the_message_is_still_there(self):
        from grader import observability

        assert observability._clean(self.GOOGLE_429).startswith("Error code: 429")

    def test_a_groq_daily_limit_keeps_its_numbers(self):
        from grader import observability

        groq = (
            "Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` in "
            "organization `org_abc` service tier `on_demand` on tokens per day (TPD): "
            + ("padding " * 40)
            + "Limit 200000, Used 199900. Please try again in 15m34s."
        )
        line = observability._clean(groq)
        assert "Limit 200000" in line and "Used 199900" in line

    def test_a_short_message_is_untouched(self):
        from grader import observability

        assert observability._clean("just a message") == "just a message"

    def test_redaction_still_wins_over_keeping_things(self):
        """Trimming must never resurrect a credential the redactor removed."""
        from grader import observability

        leaky = (
            "Incorrect API key provided: sk-abcdefghijklmnop. " + ("padding " * 60)
            + " quota: exceeded limit: 20"
        )
        line = observability._clean(leaky)
        assert "sk-abcdefghijklmnop" not in line
        assert "limit: 20" in line
