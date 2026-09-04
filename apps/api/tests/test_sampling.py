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


class TestASpentDayIsADifferentModelNotALongerWait:
    """A free tier budgets per model, so the second model is a second allowance.

    Waiting out a daily limit holds a teacher's browser open for a submission that
    will be refused again on the very next question, while an untouched allowance
    sits on the smaller model. Falling back marks the script; waiting does not.
    """

    def setup_method(self):
        from grader.grading import sampling

        sampling.forget_spent_budgets()

    def teardown_method(self):
        from grader.grading import sampling

        sampling.forget_spent_budgets()

    @pytest.mark.asyncio
    async def test_a_daily_limit_switches_model_rather_than_sleeping(self):
        from grader.grading import sampling

        used: list[str] = []

        class Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs):
                        used.append(kwargs["model"])
                        if kwargs["model"] == "openai/gpt-oss-120b":
                            raise RuntimeError(
                                "Error code: 429 - Rate limit reached for model "
                                "`openai/gpt-oss-120b` on tokens per day (TPD): "
                                "Limit 200000, Used 199900. Please try again in 15m34s."
                            )
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content='{"ok": 1}')
                                )
                            ]
                        )

        out = await sampling._create(
            Client(),
            model="openai/gpt-oss-120b",
            messages=[],
            schema_name="s",
            schema={},
            temperature=0.0,
            seed=None,
        )
        assert out == {"ok": 1}
        assert used == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

    @pytest.mark.asyncio
    async def test_a_per_minute_limit_still_waits(self, monkeypatch):
        """Only the daily limit is a different model. A burst clears in seconds."""
        from grader.grading import sampling

        slept: list[float] = []

        async def no_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(sampling.asyncio, "sleep", no_sleep)
        used: list[str] = []

        class Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs):
                        used.append(kwargs["model"])
                        if len(used) == 1:
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
            messages=[],
            schema_name="s",
            schema={},
            temperature=None,
            seed=None,
        )
        assert used == ["openai/gpt-oss-120b", "openai/gpt-oss-120b"]
        assert slept, "a per-minute limit should have been waited out"

    @pytest.mark.asyncio
    async def test_a_model_with_no_fallback_raises_rather_than_waiting_hours(self):
        from grader.grading import sampling

        class Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs):
                        raise RuntimeError("429 tokens per day (TPD) exhausted")

        with pytest.raises(RuntimeError):
            await sampling._create(
                Client(),
                model="openai/gpt-oss-20b",
                messages=[],
                schema_name="s",
                schema={},
                temperature=None,
                seed=None,
            )

    def test_provenance_names_the_model_that_will_actually_answer(self):
        """Two scripts marked by two different models must not claim the same thing."""
        from grader.grading import sampling

        assert sampling.effective_model("openai/gpt-oss-120b") == "openai/gpt-oss-120b"
        sampling._spent.add("openai/gpt-oss-120b")
        assert sampling.effective_model("openai/gpt-oss-120b") == "openai/gpt-oss-20b"

    def test_the_fallback_chain_cannot_loop(self):
        from grader.grading import sampling

        sampling._spent.update({"openai/gpt-oss-120b", "openai/gpt-oss-20b"})
        assert sampling.effective_model("openai/gpt-oss-120b") == "openai/gpt-oss-20b"

    def test_the_fallback_is_overridable(self, monkeypatch):
        from grader.grading import sampling

        monkeypatch.setenv("GRADER_FALLBACK_MODEL", "qwen/qwen3.8-27b")
        assert sampling.fallback_for("openai/gpt-oss-120b") == "qwen/qwen3.8-27b"
        assert sampling.fallback_for("qwen/qwen3.8-27b") is None
