"""Asking for a judgement in whatever shape the model will accept.

The failure behind this module is quiet and total: a model that refuses
``temperature`` refuses every member of the panel, ``_panel`` finds no samples,
and the question comes back "could not be marked automatically" — words that
send an operator to look at their API key rather than at the model name they
just changed.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from grader.grading import sampling

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


class FakeBadRequest(Exception):
    """Shaped like the provider's error: the class name is what is matched."""

    __name__ = "BadRequestError"


# The provider's own name for the class is what `sampling` inspects, so the test
# double has to carry it rather than merely quack like it.
FakeBadRequest.__name__ = "BadRequestError"


class Client:
    """Records what it was asked for, and refuses the named parameters."""

    def __init__(self, *, refuses: set[str] | None = None, content: str = '{"ok": true}'):
        self.refuses = refuses or set()
        self.content = content
        self.calls: list[dict] = []
        self.chat = self

    @property
    def completions(self):
        return self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        for name in sorted(self.refuses):
            if name in kwargs:
                raise FakeBadRequest(
                    f"Unsupported value: '{name}' does not support "
                    f"{kwargs[name]} with this model."
                )
        return _Completion(self.content)


class _Completion:
    def __init__(self, content: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


@pytest.fixture(autouse=True)
def _forget():
    sampling.forget()
    yield
    sampling.forget()


async def _ask(client, model="m"):
    return await sampling.structured_completion(
        client,
        model=model,
        system="s",
        user="u",
        schema_name="judgement",
        schema=SCHEMA,
        temperature=0.7,
        seed=11,
    )


class TestWhenEverythingIsAccepted:
    @pytest.mark.asyncio
    async def test_sends_the_temperature_and_seed_it_was_given(self) -> None:
        client = Client()

        assert await _ask(client) == {"ok": True}
        assert client.calls[0]["temperature"] == 0.7
        assert client.calls[0]["seed"] == 11

    @pytest.mark.asyncio
    async def test_asks_for_the_schema_strictly(self) -> None:
        """A schema rather than prose is what makes a weak model's output safe."""
        client = Client()
        await _ask(client)

        fmt = client.calls[0]["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True
        assert fmt["json_schema"]["schema"] == SCHEMA


class TestWhenTheModelRefusesAParameter:
    @pytest.mark.asyncio
    async def test_drops_it_and_still_returns_a_judgement(self) -> None:
        """The parameter is an optimisation. The judgement is the product."""
        client = Client(refuses={"temperature"})

        assert await _ask(client) == {"ok": True}
        assert "temperature" not in client.calls[-1]
        assert client.calls[-1]["seed"] == 11, "only the refused one is given up"

    @pytest.mark.asyncio
    async def test_remembers_so_the_rest_of_the_panel_pays_nothing(self) -> None:
        """One 400 per model per process, not one per request.

        A panel is five calls and a script is many questions; relearning this on
        every one of them would multiply both the latency and the bill by the
        length of the paper.
        """
        client = Client(refuses={"temperature"})
        await _ask(client)
        first_round = len(client.calls)

        await _ask(client)

        assert len(client.calls) == first_round + 1
        assert sampling.refused_by("m") == {"temperature"}

    @pytest.mark.asyncio
    async def test_gives_up_each_refused_parameter_in_turn(self) -> None:
        client = Client(refuses={"temperature", "seed"})

        assert await _ask(client) == {"ok": True}
        assert "temperature" not in client.calls[-1]
        assert "seed" not in client.calls[-1]

    @pytest.mark.asyncio
    async def test_what_it_learned_is_per_model(self) -> None:
        """Changing GRADER_MODEL must not inherit the last model's limits."""
        await _ask(Client(refuses={"temperature"}), model="reasoning")

        assert sampling.refused_by("reasoning") == {"temperature"}
        assert sampling.refused_by("ordinary") == set()


class TestWhatItRefusesToSwallow:
    @pytest.mark.asyncio
    async def test_a_bad_request_about_something_else_is_raised(self) -> None:
        """Only an unsupported *parameter* is recoverable.

        A rejected key, an unknown model or a malformed schema are all 400s too,
        and retrying them without a temperature would turn a clear error into a
        silent one.
        """

        class Rejecting(Client):
            async def create(self, **kwargs):
                raise FakeBadRequest("The model `nope` does not exist.")

        with pytest.raises(FakeBadRequest):
            await _ask(Rejecting())

    @pytest.mark.asyncio
    async def test_an_ordinary_failure_is_raised(self) -> None:
        class Failing(Client):
            async def create(self, **kwargs):
                raise TimeoutError("the provider timed out")

        with pytest.raises(TimeoutError):
            await _ask(Failing())

    @pytest.mark.asyncio
    async def test_an_empty_answer_is_an_error_rather_than_an_empty_grade(self) -> None:
        """No judgement must not arrive as a judgement awarding nothing."""
        with pytest.raises(ValueError):
            await _ask(Client(content=""))

    @pytest.mark.asyncio
    async def test_the_content_is_parsed_as_the_object_it_promised(self) -> None:
        client = Client(content=json.dumps({"checks": [{"index": 1, "met": True}]}))

        assert await _ask(client) == {"checks": [{"index": 1, "met": True}]}


class FakeRateLimit(Exception):
    """Shaped like the provider's, including the wait it names."""


FakeRateLimit.__name__ = "RateLimitError"


class TestARateLimitIsAWaitNotAFailure:
    """The failure this prevents cost a whole day of measurement.

    Free and low tiers meter per minute, and this service fans out four questions
    at a time. Nothing client-side slowed that down, so a nine-document run
    produced 53 dropped panel samples, every affected question came back "could
    not be marked automatically", and the documents scored zero — which the gate
    then reported as a marking result. It was a queueing result.
    """

    @pytest.mark.asyncio
    async def test_it_waits_and_retries(self, monkeypatch) -> None:
        monkeypatch.setattr(sampling.asyncio, "sleep", _record_sleep := _Recorder())
        attempts = []

        class Limited(Client):
            async def create(self, **kwargs):
                attempts.append(1)
                if len(attempts) == 1:
                    raise FakeRateLimit(
                        "Error code: 429 - Rate limit reached. Please try again in 2.5s"
                    )
                return await super().create(**kwargs)

        assert await _ask(Limited()) == {"ok": True}
        assert len(attempts) == 2
        assert _record_sleep.waited == [2.5], "the provider's own number, not a guess"

    @pytest.mark.asyncio
    async def test_it_reads_a_wait_given_in_minutes(self, monkeypatch) -> None:
        """Groq phrases a daily-budget refusal as "try again in 21m24.336s"."""
        assert sampling._wait_for(
            FakeRateLimit("429 - try again in 1m30.0s"), attempt=0
        ) == pytest.approx(90.0)

    @pytest.mark.asyncio
    async def test_a_wait_longer_than_the_cap_fails_honestly(self) -> None:
        """A provider asking for twenty minutes is saying the budget is gone.

        Holding a teacher's browser open until then is worse than telling them.
        """
        assert sampling._wait_for(FakeRateLimit("429 - try again in 21m24.3s"), 0) is None

    @pytest.mark.asyncio
    async def test_it_gives_up_rather_than_retrying_for_ever(self, monkeypatch) -> None:
        monkeypatch.setattr(sampling.asyncio, "sleep", _Recorder())

        class AlwaysLimited(Client):
            async def create(self, **kwargs):
                raise FakeRateLimit("429 - Rate limit reached. Please try again in 0.1s")

        with pytest.raises(FakeRateLimit):
            await _ask(AlwaysLimited())

    @pytest.mark.asyncio
    async def test_an_ordinary_failure_is_not_treated_as_a_wait(self) -> None:
        assert sampling._wait_for(ValueError("nope"), attempt=0) is None

    @pytest.mark.asyncio
    async def test_calls_are_bounded_in_flight(self, monkeypatch) -> None:
        """Twenty concurrent calls against thirty a minute is how this began."""
        monkeypatch.setattr(sampling, "MAX_IN_FLIGHT", 2)
        sampling._in_flight = None
        live, peak = 0, 0

        class Slow(Client):
            async def create(self, **kwargs):
                nonlocal live, peak
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0)
                live -= 1
                return await super().create(**kwargs)

        client = Slow()
        await asyncio.gather(*(_ask(client) for _ in range(8)))

        assert peak <= 2, f"{peak} calls were in flight at once"


class _Recorder:
    """Stands in for `asyncio.sleep` and remembers what it was asked to wait."""

    def __init__(self) -> None:
        self.waited: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waited.append(seconds)


class TestADroppedFieldIsARerollNotALostQuestion:
    """The provider validates the model's output and 400s when a field is missing.

    At five samples that cost one vote and the panel absorbed it. At one sample it
    costs the whole question, which surfaces as "never judged" — an answer the
    teacher wrote that nobody marked. That is a false absence, which is the worst
    error this product can make, arrived at through a missing JSON key.
    """

    @pytest.mark.asyncio
    async def test_a_schema_validation_400_is_retried(self):
        from grader.grading import sampling

        calls: list[dict] = []

        class Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs):
                        calls.append(kwargs)
                        if len(calls) == 1:
                            raise RuntimeError(
                                "Error code: 400 - Generated JSON does not match the "
                                "expected schema. Error: jsonschema: '/checks/0' does "
                                "not validate with /properties/checks/items/required: "
                                "missing properties: 'cited_line_ids'"
                            )
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content='{"ok": true}')
                                )
                            ]
                        )

        out = await sampling._create(
            Client(),
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "x"}],
            schema_name="s",
            schema={"type": "object"},
            temperature=0.0,
            seed=1,
        )

        assert out == {"ok": True}
        assert len(calls) == 2, "the slip should have been re-rolled, not raised"

    @pytest.mark.asyncio
    async def test_the_reroll_changes_temperature(self):
        """A re-roll at the same settings reproduces the same slip."""
        from grader.grading import sampling

        temps: list[float | None] = []

        class Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs):
                        temps.append(kwargs.get("temperature"))
                        if len(temps) == 1:
                            raise RuntimeError("400 missing properties: 'x'")
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content="{}")
                                )
                            ]
                        )

        await sampling._create(
            Client(),
            model="m",
            messages=[],
            schema_name="s",
            schema={},
            temperature=0.0,
            seed=None,
        )
        assert temps[1] > temps[0]

    @pytest.mark.asyncio
    async def test_a_genuinely_bad_request_is_not_retried(self):
        """A 400 is also what a wrong model name returns. Retrying that is a loop."""
        from grader.grading import sampling

        calls = []

        class Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs):
                        calls.append(kwargs)
                        raise RuntimeError(
                            "Error code: 400 - The model `nope` does not exist"
                        )

        with pytest.raises(RuntimeError):
            await sampling._create(
                Client(),
                model="nope",
                messages=[],
                schema_name="s",
                schema={},
                temperature=None,
                seed=None,
            )
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_a_model_that_never_holds_the_schema_gives_up(self):
        """Two re-rolls, then raise. A third is a slower failure, not a different one."""
        from grader.grading import sampling

        calls = []

        class Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs):
                        calls.append(kwargs)
                        raise RuntimeError("400 does not validate with required")

        with pytest.raises(RuntimeError):
            await sampling._create(
                Client(),
                model="m",
                messages=[],
                schema_name="s",
                schema={},
                temperature=0.0,
                seed=None,
            )
        assert len(calls) == sampling.DECODE_RETRIES + 1


class TestASpentDayIsADifferentMarkerNotALongerWait:
    """A free tier meters per model *and* per host, so the chain is real capacity.

    Waiting out a daily limit holds a teacher's browser open for a submission
    that will be refused again on the very next question, while an untouched
    allowance sits one entry down. Walking the chain marks the script; waiting
    does not.
    """

    CHAIN = [
        ("cerebras", "gpt-oss-120b"),
        ("groq", "openai/gpt-oss-120b"),
        ("groq", "qwen/qwen3.8-27b"),
    ]

    @pytest.fixture(autouse=True)
    def _chain(self, monkeypatch):
        from grader import clients
        from grader.grading import sampling

        sampling.forget_spent_budgets()
        monkeypatch.setattr(clients, "marking_chain", lambda: list(self.CHAIN))
        yield
        sampling.forget_spent_budgets()

    @staticmethod
    def _client(refuse: set, seen: list):
        class Client:
            def __init__(self, tag="start"):
                self.tag = tag

            class chat:  # noqa: N801
                pass

        def make(tag):
            c = Client(tag)

            class completions:  # noqa: N801
                @staticmethod
                async def create(**kwargs):
                    seen.append((c.tag, kwargs["model"]))
                    if kwargs["model"] in refuse:
                        raise RuntimeError(
                            f"Error code: 429 - Rate limit reached for model "
                            f"`{kwargs['model']}` on tokens per day (TPD): Limit "
                            f"200000, Used 199900. Please try again in 15m34s."
                        )
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(message=SimpleNamespace(content='{"ok": 1}'))
                        ]
                    )

            c.chat = SimpleNamespace(completions=completions)
            return c

        return make

    @pytest.mark.asyncio
    async def test_a_daily_limit_walks_to_the_next_entry(self, monkeypatch):
        from grader.grading import sampling

        seen: list = []
        make = self._client({"gpt-oss-120b"}, seen)
        monkeypatch.setattr(sampling, "client_for", lambda p: make(p))

        out = await sampling._create(
            make("cerebras"),
            model="gpt-oss-120b",
            provider="cerebras",
            messages=[],
            schema_name="s",
            schema={},
            temperature=0.0,
            seed=None,
        )
        assert out == {"ok": 1}
        assert [m for _tag, m in seen] == ["gpt-oss-120b", "openai/gpt-oss-120b"]

    @pytest.mark.asyncio
    async def test_crossing_hosts_builds_a_client_for_the_new_host(self, monkeypatch):
        """A client carries its key and base URL, so a new host is a new client.

        Reusing the old one sends a Groq key to Cerebras, which is not a fallback
        but a second failure wearing the first one's clothes.
        """
        from grader.grading import sampling

        seen: list = []
        built: list[str] = []
        make = self._client({"gpt-oss-120b"}, seen)

        def client_for(provider):
            built.append(provider)
            return make(provider)

        monkeypatch.setattr(sampling, "client_for", client_for)
        await sampling._create(
            make("cerebras"),
            model="gpt-oss-120b",
            provider="cerebras",
            messages=[],
            schema_name="s",
            schema={},
            temperature=None,
            seed=None,
        )
        assert built == ["groq"], "the next host did not get its own client"

    @pytest.mark.asyncio
    async def test_a_per_minute_limit_still_waits(self, monkeypatch):
        """Only a daily limit is a different marker. A burst clears in seconds."""
        from grader.grading import sampling

        slept: list[float] = []

        async def no_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(sampling.asyncio, "sleep", no_sleep)
        calls: list = []

        class Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs):
                        calls.append(kwargs["model"])
                        if len(calls) == 1:
                            raise RuntimeError(
                                "429 Rate limit reached on tokens per minute (TPM). "
                                "Please try again in 2.5s"
                            )
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(message=SimpleNamespace(content="{}"))
                            ]
                        )

        await sampling._create(
            Client(),
            model="openai/gpt-oss-120b",
            provider="groq",
            messages=[],
            schema_name="s",
            schema={},
            temperature=None,
            seed=None,
        )
        assert calls == ["openai/gpt-oss-120b", "openai/gpt-oss-120b"]
        assert slept, "a per-minute limit should have been waited out"

    @pytest.mark.asyncio
    async def test_the_end_of_the_chain_raises_rather_than_waiting_hours(
        self, monkeypatch
    ):
        from grader.grading import sampling

        seen: list = []
        every = {model for _p, model in self.CHAIN}
        make = self._client(every, seen)
        monkeypatch.setattr(sampling, "client_for", lambda p: make(p))

        with pytest.raises(RuntimeError):
            await sampling._create(
                make("cerebras"),
                model="gpt-oss-120b",
                provider="cerebras",
                messages=[],
                schema_name="s",
                schema={},
                temperature=None,
                seed=None,
            )
        assert [m for _t, m in seen] == [m for _p, m in self.CHAIN]

    def test_provenance_names_the_marker_that_will_actually_answer(self):
        """Two scripts marked by two different models must not claim the same thing."""
        from grader.grading import sampling

        assert sampling.effective_marker("cerebras", "gpt-oss-120b") == (
            "cerebras",
            "gpt-oss-120b",
        )
        sampling._spent.add(("cerebras", "gpt-oss-120b"))
        assert sampling.effective_marker("cerebras", "gpt-oss-120b") == (
            "groq",
            "openai/gpt-oss-120b",
        )

    def test_the_same_model_on_two_hosts_is_two_budgets(self):
        """The reason the spent set is keyed by both halves.

        Keyed by model alone, a spent Cerebras allowance would also write off the
        Groq entry for a differently-named build of the same weights, and the
        chain would skip an allowance that was never touched.
        """
        from grader.grading import sampling

        sampling._spent.add(("cerebras", "gpt-oss-120b"))
        assert sampling.next_marker() == ("groq", "openai/gpt-oss-120b")

    def test_the_chain_cannot_loop_or_run_off_the_end(self):
        from grader.grading import sampling

        sampling._spent.update(set(self.CHAIN))
        assert sampling.next_marker() is None
        # And the marker of last resort is still reported honestly rather than
        # claiming an entry that refused.
        assert sampling.effective_marker("groq", "qwen/qwen3.8-27b") == (
            "groq",
            "qwen/qwen3.8-27b",
        )


class TestTheMarkingChainReflectsTheDeployment:
    """Which entries exist is decided by keys and by what was pinned."""

    def test_a_host_with_no_key_is_not_in_the_chain(self, monkeypatch):
        from grader import clients

        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GRADER_MODEL", raising=False)
        monkeypatch.delenv("GRADER_PROVIDER", raising=False)

        chain = clients.marking_chain()
        assert chain, "a configured host produced no chain"
        assert {p for p, _m in chain} == {"groq"}

    def test_pinning_a_model_collapses_the_chain(self, monkeypatch):
        """A measurement is only worth something if it names one marker.

        The eval harness pins `GRADER_MODEL` for exactly this reason: a gate run
        that started on one model and silently finished on another would report a
        number belonging to neither.
        """
        from grader import clients

        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test")
        monkeypatch.setenv("GRADER_MODEL", "qwen/qwen3.8-27b")
        monkeypatch.delenv("GRADER_PROVIDER", raising=False)

        assert clients.marking_chain() == [("groq", "qwen/qwen3.8-27b")]

    def test_pinning_a_host_keeps_only_that_host(self, monkeypatch):
        from grader import clients

        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test")
        monkeypatch.setenv("GRADER_PROVIDER", "cerebras")
        monkeypatch.delenv("GRADER_MODEL", raising=False)

        assert {p for p, _m in clients.marking_chain()} == {"cerebras"}

    def test_an_unknown_pinned_model_is_still_honoured(self, monkeypatch):
        """Better to mark with what was asked for than to substitute silently."""
        from grader import clients

        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        monkeypatch.setenv("GRADER_MODEL", "some/experimental-model")
        monkeypatch.delenv("GRADER_PROVIDER", raising=False)

        assert clients.marking_chain() == [("groq", "some/experimental-model")]


class TestTheChainIsOrderedByMeasurement:
    """The order is a claim about accuracy, so it is worth pinning.

    Measured on the nine-document gate: `gpt-oss-120b` reaches 8 of 9 at five
    samples and 5 of 9 at one; `gpt-oss-20b` reaches 4 of 9 with every miss
    under-marking, two of them earned answers scoring zero. An order that put the
    cheaper or faster entry first would be trading marks for latency without
    saying so.
    """

    def test_the_best_measured_marker_is_reached_first(self, monkeypatch):
        from grader import clients

        for var in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.setenv(var, "k")
        monkeypatch.delenv("GRADER_MODEL", raising=False)
        monkeypatch.delenv("GRADER_PROVIDER", raising=False)

        chain = clients.marking_chain()
        assert chain[0][1].endswith("gpt-oss-120b")
        assert chain[1][1].endswith("gpt-oss-120b")

    def test_a_marker_that_scores_zeros_on_earned_answers_is_reached_late(
        self, monkeypatch
    ):
        """`gpt-oss-20b` is behind every measured-good and every unmeasured entry.

        Its four-of-nine is a complete run, and two of its misses are answers a
        student earned marks for that came back zero. Reaching it early would
        trade a false zero for some latency.
        """
        from grader import clients

        for var in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.setenv(var, "k")
        monkeypatch.delenv("GRADER_MODEL", raising=False)
        monkeypatch.delenv("GRADER_PROVIDER", raising=False)

        chain = clients.marking_chain()
        weak = chain.index(("groq", "openai/gpt-oss-20b"))
        best = chain.index(("groq", "openai/gpt-oss-120b"))
        assert weak > best
        assert weak >= len(chain) - 3, "the weakest marker is reached too early"

    def test_the_same_weights_appear_on_more_than_one_host(self, monkeypatch):
        """The point of the chain: capacity that costs no accuracy.

        Two hosts serving the same model is a second daily allowance for the same
        judgement, which is the only kind of extra capacity worth having. Every
        entry below them is a worse marker.
        """
        from grader import clients

        for var in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.setenv(var, "k")
        monkeypatch.delenv("GRADER_MODEL", raising=False)
        monkeypatch.delenv("GRADER_PROVIDER", raising=False)

        hosts = [p for p, m in clients.marking_chain() if m.endswith("gpt-oss-120b")]
        assert len(set(hosts)) > 1


class TestThePanelIsWhatBuysAccuracy:
    """`MARK_SAMPLES` is an accuracy setting, and it had been used as a budget one.

    Measured on the same nine documents with the same model, `gpt-oss-120b`:
    five samples put 8 of 9 inside their band, one sample put 5 of 9, and all
    four of the extra failures were under-marking. Cutting the panel to fit a
    free tier is therefore not a cost saving, it is three documents of accuracy
    spent without saying so - and the right response is more allowance, which is
    what the chain is for.
    """

    def test_the_default_panel_is_the_measured_one(self, monkeypatch):
        import importlib

        monkeypatch.delenv("MARK_SAMPLES", raising=False)
        from grader.grading import engine

        importlib.reload(engine)
        assert engine.MARK_SAMPLES == 5, (
            "the default panel was reduced; that is an accuracy change, not a "
            "cost change, and the gate figures above say what it costs"
        )
        importlib.reload(engine)


class TestAHostThatRefusesAParameterIsLearnedFrom:
    """Every OpenAI-shaped host implements a different subset of the parameters.

    A refused parameter is silent in the worst way: it comes back as a 400 on
    every sample, so the question is never judged and the document simply scores
    zero. A whole nine-document gate run was lost to Google refusing `seed`, and
    the message it refuses with is worded unlike any other host's.
    """

    def setup_method(self):
        from grader.grading import sampling

        sampling.forget()

    @pytest.mark.parametrize(
        "message",
        [
            "Unsupported value: 'seed' is not supported with this model",
            "Unsupported parameter: 'seed'",
            "Unrecognized request argument supplied: seed",
            # Google, via its OpenAI-compatible endpoint.
            'Invalid JSON payload received. Unknown name "seed": Cannot find field.',
        ],
    )
    def test_every_hosts_wording_is_recognised(self, message):
        from grader.grading import sampling

        class BadRequestError(Exception):
            pass

        assert sampling._names_a_refused_parameter(BadRequestError(message)) == "seed"

    def test_a_real_bad_request_is_not_mistaken_for_a_refused_parameter(self):
        """Otherwise the retry loop strips parameters forever and never succeeds."""
        from grader.grading import sampling

        class BadRequestError(Exception):
            pass

        assert (
            sampling._names_a_refused_parameter(
                BadRequestError("The model `nope` does not exist")
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_the_request_is_retried_without_what_was_refused(self):
        from grader.grading import sampling

        sent: list[dict] = []

        class BadRequestError(Exception):
            pass

        class Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs):
                        sent.append(kwargs)
                        if "seed" in kwargs:
                            raise BadRequestError(
                                'Invalid JSON payload received. Unknown name "seed": '
                                "Cannot find field."
                            )
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(message=SimpleNamespace(content="{}"))
                            ]
                        )

        await sampling._create(
            Client(),
            model="gemini-3-flash-preview",
            provider="gemini",
            messages=[],
            schema_name="s",
            schema={},
            temperature=0.2,
            seed=7,
        )
        assert "seed" in sent[0] and "seed" not in sent[-1]
        assert sampling.refused_by("gemini-3-flash-preview") == {"seed"}


class TestRequestsArePacedNotJustLimitedInFlight:
    """Concurrency and rate are different limits, and only the first was governed.

    A cap of two in flight says nothing about how many requests start in a
    minute: short calls finish and are replaced immediately, so two workers
    comfortably issue sixty requests a minute against an allowance of ten.
    Measured on Google's free tier, a nine-document gate produced 155 rate-limit
    errors and eight documents scoring zero, while the one document that got
    through landed inside its band. The marking was never the problem.
    """

    def setup_method(self):
        from grader.grading import sampling

        sampling.forget_pacing()

    def teardown_method(self):
        from grader.grading import sampling

        sampling.forget_pacing()

    @pytest.mark.asyncio
    async def test_starting_more_than_the_allowance_waits(self, monkeypatch):
        from grader.grading import sampling

        monkeypatch.setitem(sampling.REQUESTS_PER_MINUTE, "gemini", 3)
        waits: list[float] = []
        real_sleep = sampling.asyncio.sleep

        async def record(seconds):
            waits.append(seconds)
            await real_sleep(0)

        # The first three go straight through; the fourth has to wait for the
        # window to roll.
        for _ in range(3):
            await sampling._pace("gemini")
        assert not waits

        # Ages the window by one slot on each sleep, so the fourth call waits
        # once and then proceeds. Without this the stubbed sleep returns
        # instantly, the window never moves, and the test spins for a real
        # minute - which is the loop working correctly against a clock that is
        # not advancing rather than a bug in the pacer.
        async def age_one(seconds):
            waits.append(seconds)
            sampling._starts["gemini"][0] -= 61.0
            await real_sleep(0)

        monkeypatch.setattr(sampling.asyncio, "sleep", age_one)
        await sampling._pace("gemini")
        assert len(waits) == 1, "the fourth request did not wait exactly once"

    @pytest.mark.asyncio
    async def test_each_host_is_paced_on_its_own_allowance(self, monkeypatch):
        """A shared pacer would slow the fast host to the slow one's rate."""
        from grader.grading import sampling

        monkeypatch.setitem(sampling.REQUESTS_PER_MINUTE, "gemini", 1)
        monkeypatch.setitem(sampling.REQUESTS_PER_MINUTE, "groq", 30)

        await sampling._pace("gemini")
        waits: list[float] = []
        real_sleep = sampling.asyncio.sleep

        async def record(seconds):
            waits.append(seconds)
            await real_sleep(0)

        monkeypatch.setattr(sampling.asyncio, "sleep", record)
        # Groq has its own budget and must not be held up by Gemini's.
        await sampling._pace("groq")
        assert not waits

    @pytest.mark.asyncio
    async def test_the_window_slides_rather_than_bucketing(self, monkeypatch):
        """A fixed wall-clock bucket lets a burst straddle two of them.

        The provider's own limit is a rolling sixty seconds, so bucketing would
        allow twice the allowance across a boundary while looking compliant.
        """
        from grader.grading import sampling

        monkeypatch.setitem(sampling.REQUESTS_PER_MINUTE, "groq", 2)
        await sampling._pace("groq")
        await sampling._pace("groq")

        # Age the recorded starts past the window; they should be discarded.
        sampling._starts["groq"] = [t - 61.0 for t in sampling._starts["groq"]]
        waits: list[float] = []
        real_sleep = sampling.asyncio.sleep

        async def record(seconds):
            waits.append(seconds)
            await real_sleep(0)

        monkeypatch.setattr(sampling.asyncio, "sleep", record)
        await sampling._pace("groq")
        assert not waits, "starts older than the window were still counted"

    @pytest.mark.asyncio
    async def test_an_unknown_host_is_not_throttled_to_a_standstill(self):
        from grader.grading import sampling

        assert sampling.rpm_for("some-paid-host") == sampling.DEFAULT_RPM
