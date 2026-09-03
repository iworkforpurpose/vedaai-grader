"""Asking for a judgement in whatever shape the model will accept.

The failure behind this module is quiet and total: a model that refuses
``temperature`` refuses every member of the panel, ``_panel`` finds no samples,
and the question comes back "could not be marked automatically" — words that
send an operator to look at their API key rather than at the model name they
just changed.
"""

from __future__ import annotations

import json

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
