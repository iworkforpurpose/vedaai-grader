"""Tests for the rate limit on the endpoints that cost money.

The service is behind a public URL with no account behind it. Ingesting one
submission renders every page, runs recognition over all of them, embeds the
paper and the script, and calls a marking model once per question — so a loop
pointed at `POST /submissions` spends real money on somebody else's key, and
nothing in the design noticed.

These tests are about the shape of the limit rather than the numbers: that it
counts per client rather than in total, that it lets the count age out, and that
it says when to come back.
"""

from __future__ import annotations

import pytest

from grader.throttle import Throttle


class TestCountingPerClient:
    def test_a_client_may_spend_its_allowance(self) -> None:
        clock = [0.0]
        throttle = Throttle(limit=3, window=60.0, now=lambda: clock[0])
        assert [throttle.check("a") for _ in range(3)] == [None, None, None]

    def test_and_is_refused_after_it(self) -> None:
        clock = [0.0]
        throttle = Throttle(limit=2, window=60.0, now=lambda: clock[0])
        throttle.check("a")
        throttle.check("a")
        retry = throttle.check("a")
        assert retry is not None and retry > 0

    def test_one_client_cannot_spend_another_s(self) -> None:
        # The failure this shape exists to avoid: a single counter would let one
        # person looping lock out everybody else, which is worse than the abuse.
        clock = [0.0]
        throttle = Throttle(limit=1, window=60.0, now=lambda: clock[0])
        assert throttle.check("a") is None
        assert throttle.check("b") is None
        assert throttle.check("a") is not None

    def test_the_count_ages_out(self) -> None:
        clock = [0.0]
        throttle = Throttle(limit=1, window=60.0, now=lambda: clock[0])
        assert throttle.check("a") is None
        clock[0] = 59.0
        assert throttle.check("a") is not None
        clock[0] = 61.0
        assert throttle.check("a") is None

    def test_it_says_how_long_to_wait(self) -> None:
        # A 429 with no Retry-After leaves a caller to guess, and guessing means
        # retrying immediately and being refused again.
        clock = [0.0]
        throttle = Throttle(limit=1, window=60.0, now=lambda: clock[0])
        throttle.check("a")
        clock[0] = 20.0
        assert throttle.check("a") == pytest.approx(40.0)

    def test_a_limit_of_zero_refuses_nothing(self) -> None:
        # How the limit is switched off: a deployment that does not want one sets
        # it to zero rather than having to reason about a very large number.
        throttle = Throttle(limit=0, window=60.0, now=lambda: 0.0)
        assert all(throttle.check("a") is None for _ in range(50))

    def test_clients_that_stop_calling_are_forgotten(self) -> None:
        # Otherwise the table is a slow memory leak keyed by anything a caller
        # can vary, which is the whole internet.
        clock = [0.0]
        throttle = Throttle(limit=5, window=60.0, now=lambda: clock[0])
        for i in range(200):
            throttle.check(f"client-{i}")
        clock[0] = 3600.0
        throttle.check("someone-else")
        assert throttle.tracked() <= 2
