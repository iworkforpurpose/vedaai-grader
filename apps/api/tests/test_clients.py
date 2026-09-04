"""How long this service is willing to wait.

Every external client was constructed bare. No `botocore.Config` existed anywhere
in the tree, and both model SDKs default to **600 seconds** — so a stalled
provider held a submission at `processing` for ten minutes per wave of four
questions, with no reaper and nothing logged.

These tests exist because a timeout is invisible until the day it is needed, and
because the failure it prevents looks like the service merely being slow.
"""

from __future__ import annotations

from grader import clients


class TestNothingWaitsForever:
    def test_every_aws_client_has_both_timeouts(self) -> None:
        config = clients.aws_config()

        assert config.connect_timeout == clients.CONNECT_TIMEOUT
        assert config.read_timeout == clients.STORAGE_READ_TIMEOUT

    def test_textract_gets_a_shorter_read_timeout_than_storage(self) -> None:
        """A page Textract has not answered in thirty seconds is not coming.

        Storage is longer because a spilled submission payload runs to hundreds of
        kilobytes and page images are larger still.
        """
        assert clients.TEXTRACT_READ_TIMEOUT < clients.STORAGE_READ_TIMEOUT
        assert clients.aws_config(clients.TEXTRACT_READ_TIMEOUT).read_timeout == 30.0

    def test_aws_retries_adaptively(self) -> None:
        """`adaptive`, not `standard`.

        Adaptive rate-limits client-side on a throttle instead of retrying
        straight back into the same wall, which is what a per-page loop over
        sixty pages needs. Nothing retried at all before: `ocr/textract.py`
        translates `ThrottlingException` — which carries its own hand-written
        "try again" message — into a terminal `EngineUnavailable`, and that ends
        the transcription of every page including the ones already paid for.
        """
        retries = clients.aws_config().retries

        assert retries["mode"] == "adaptive"
        assert retries["max_attempts"] >= 3

    def test_a_model_call_gives_up_long_before_the_sdk_would(self) -> None:
        """600 seconds is the default. Four questions at a time, five samples
        each, is a submission that hangs for the rest of the afternoon."""
        assert clients.MODEL_TIMEOUT <= 120.0
        assert clients.openai_kwargs()["timeout"] == clients.MODEL_TIMEOUT
        assert clients.anthropic_kwargs()["timeout"] == clients.MODEL_TIMEOUT

    def test_a_model_call_retries_a_blip_but_not_an_outage(self) -> None:
        """The panel already tolerates a lost sample and the per-question guard
        already tolerates a lost question. Retries here are for a blip."""
        assert 1 <= clients.openai_kwargs()["max_retries"] <= 3

    def test_the_model_timeout_is_configurable_without_a_release(self) -> None:
        """A provider that has become slower should not need a deploy to survive."""
        import importlib
        import os

        os.environ["MODEL_TIMEOUT_SECONDS"] = "15"
        try:
            reloaded = importlib.reload(clients)
            assert reloaded.MODEL_TIMEOUT == 15.0
        finally:
            del os.environ["MODEL_TIMEOUT_SECONDS"]
            importlib.reload(clients)


class TestTheClientsActuallyUseIt:
    def test_textract_builds_its_client_with_the_config(self, monkeypatch) -> None:
        # `boto3` is imported inside the method, so the patch goes on the real
        # module rather than on a name the engine holds.
        import boto3

        from grader.ocr import textract as textract_module

        seen: dict = {}

        def fake_client(service, **kwargs):
            seen.update(service=service, **kwargs)
            return object()

        monkeypatch.setattr(boto3, "client", fake_client)
        engine = textract_module.TextractEngine()
        engine._ensure_client()

        assert seen["service"] == "textract"
        assert seen["config"].read_timeout == clients.TEXTRACT_READ_TIMEOUT
        assert seen["config"].retries["mode"] == "adaptive"
