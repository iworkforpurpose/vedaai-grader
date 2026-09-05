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


class TestWhichProviderMarks:
    """One marking call, several hosts that speak the same API.

    Every provider worth using here speaks the OpenAI chat-completions API, so
    "which provider" is a base URL and a key rather than a second code path. What
    has to be right is the selection, because getting it wrong is silent: a key
    for one host and a model name from another produces a model-not-found nobody
    was expecting.
    """

    def _only(self, monkeypatch, **env) -> None:
        for var in ["GROQ_API_KEY", "OPENAI_API_KEY", "GRADER_PROVIDER", "GRADER_BASE_URL"]:
            monkeypatch.delenv(var, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)

    def test_groq_is_preferred_when_its_key_is_present(self, monkeypatch) -> None:
        """Roughly thirteen times cheaper per script, marking a class at a time."""
        self._only(monkeypatch, GROQ_API_KEY="gsk_test", OPENAI_API_KEY="sk-test")

        name, key, base = clients.openai_provider()

        assert name == "groq"
        assert key == "gsk_test"
        assert base == "https://api.groq.com/openai/v1"

    def test_openai_needs_no_base_url(self, monkeypatch) -> None:
        self._only(monkeypatch, OPENAI_API_KEY="sk-test")

        name, key, base = clients.openai_provider()

        assert (name, key, base) == ("openai", "sk-test", None)

    def test_an_explicit_choice_wins(self, monkeypatch) -> None:
        """`GRADER_PROVIDER` exists so a deployment cannot drift silently when a
        second key appears in its environment."""
        self._only(
            monkeypatch,
            GROQ_API_KEY="gsk_test",
            OPENAI_API_KEY="sk-test",
            GRADER_PROVIDER="openai",
        )

        assert clients.openai_provider()[0] == "openai"

    def test_an_explicit_base_url_overrides_the_provider_default(self, monkeypatch) -> None:
        """Any other OpenAI-compatible host is a config change, not a release."""
        self._only(monkeypatch, GROQ_API_KEY="gsk_test",
                   GRADER_BASE_URL="https://openrouter.ai/api/v1")

        assert clients.openai_provider()[2] == "https://openrouter.ai/api/v1"

    def test_no_key_at_all_yields_no_key(self, monkeypatch) -> None:
        """The rubric-only path is a working product, not an error path."""
        self._only(monkeypatch)

        assert clients.openai_provider()[1] is None

    def test_the_client_carries_the_key_and_the_destination(self, monkeypatch) -> None:
        self._only(monkeypatch, GROQ_API_KEY="gsk_test")

        kwargs = clients.openai_kwargs()

        assert kwargs["api_key"] == "gsk_test"
        assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
        assert kwargs["timeout"] == clients.MODEL_TIMEOUT


class TestTheGradeSaysWhichHostJudgedIt:
    def test_provenance_names_the_provider_not_the_sdk_shape(self, monkeypatch) -> None:
        """Groq speaks the OpenAI API. Without this, a grade from `gpt-oss-120b`
        on Groq and one from `gpt-4.1` on OpenAI both read `openai:...`, and
        "a mark is only checkable if you know what made it" stops being true the
        moment two hosts are configurable.
        """
        from grader.grading.engine import OpenAIGrader

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("GRADER_PROVIDER", "groq")

        grader = OpenAIGrader(client=object())

        assert grader.model == "openai/gpt-oss-120b"
        assert grader.provenance == "groq:openai/gpt-oss-120b"


class TestTheChainIsOrderedByEvidenceThenByAllowance:
    """Two rules, and the order between them is the whole design.

    Ordering purely by free allowance puts a weaker marker in front of a better
    one the moment somebody's free tier grows. Ordering purely by accuracy wastes
    the largest allowance on the entry least able to use it. So accuracy decides
    who is allowed to mark, and among markers that measured the same, allowance
    decides who marks first.
    """

    def test_a_marker_below_the_bar_never_outranks_one_above_it(self):
        from grader import clients

        chain = clients.FALLBACK_CHAIN
        good = [e for e in chain if (clients.MEASURED.get(e) or 0) >= clients.MIN_IN_BAND]
        weak = [e for e in chain if (clients.MEASURED.get(e) or 0) < clients.MIN_IN_BAND]
        assert good, "no marker cleared the accuracy bar"
        assert max(chain.index(e) for e in good) < min(chain.index(e) for e in weak)

    def test_among_equals_the_larger_free_tier_comes_first(self):
        """The user-facing rule: the biggest allowance is reached first.

        Applied within an accuracy tier rather than across all of them, because
        a free tier buys scripts and only a gate score says whether those scripts
        are marked correctly.
        """
        from grader import clients

        for tier in (0, 1):
            entries = [e for e in clients.FALLBACK_CHAIN if clients._rank(e)[0] == tier]
            allowances = [
                clients.FREE_TIER.get(p, {}).get("scripts_per_day", 0.0)
                for p, _m in entries
            ]
            assert allowances == sorted(allowances, reverse=True), (
                f"tier {tier} is not ordered by free allowance: {entries}"
            )

    def test_an_unmeasured_marker_sorts_below_a_measured_one(self):
        """Not evidence of being bad, but not evidence either.

        A large free tier is a reason to measure a marker, never a reason to put
        it in front of one that has already scored.
        """
        from grader import clients

        assert clients.MEASURED[("gemini", "gemini-2.5-flash")] is None
        assert clients._rank(("gemini", "gemini-2.5-flash"))[0] == 1
        assert clients._rank(("groq", "openai/gpt-oss-120b"))[0] == 0

    def test_measuring_a_marker_promotes_it_into_allowance_order(self, monkeypatch):
        """The order is computed, so a new score re-sorts without anyone editing a list.

        Asserted as the rule rather than as a position, because the position
        depends on numbers that move: this test used to say "measuring the
        largest free tier puts it first", which stopped being true the moment a
        host with a larger allowance started answering. The rule did not change;
        only the arithmetic did.
        """
        from grader import clients

        entry = ("gemini", "gemini-2.5-flash")
        assert clients.MEASURED[entry] is None
        before = sorted(clients.MEASURED, key=clients._rank).index(entry)

        monkeypatch.setitem(clients.MEASURED, entry, 8)
        after_chain = sorted(clients.MEASURED, key=clients._rank)
        after = after_chain.index(entry)

        assert after < before, "a measured marker did not move up"
        assert clients._rank(entry)[0] == 0, "it did not join the scored tier"

        # And it sits exactly where its allowance puts it among the scored.
        scored = [e for e in after_chain if clients._rank(e)[0] == 0]
        allowances = [
            clients.FREE_TIER.get(p, {}).get("scripts_per_day", 0.0) for p, _m in scored
        ]
        assert allowances == sorted(allowances, reverse=True)

    def test_every_configured_host_is_reachable_before_marking_gives_up(
        self, monkeypatch
    ):
        """A submission should fail only once every host has spent its day."""
        from grader import clients

        for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY"):
            monkeypatch.setenv(var, "k")
        monkeypatch.delenv("GRADER_MODEL", raising=False)
        monkeypatch.delenv("GRADER_PROVIDER", raising=False)

        assert {p for p, _m in clients.marking_chain()} == {
            "groq",
            "gemini",
            "cerebras",
        }
