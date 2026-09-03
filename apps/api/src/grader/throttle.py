"""A ceiling on how often one caller may spend money.

The service answers a public URL and asks nobody who they are. Ingesting one
submission renders every page, runs recognition across all of them, embeds the
paper and the script, and calls a marking model once per question — so a loop
pointed at ``POST /submissions`` spends real money on the operator's key, and
until this existed nothing in the design would have noticed.

Two things this deliberately is not.

It is not per-process-safe across a fleet. The count lives in memory, so with two
tasks running a caller gets the limit twice. That is stated rather than solved
because the deployment runs one task, and a distributed counter is a database
round trip on the hot path for a problem that does not exist yet. When it does,
the seam is `Throttle` and the change is behind it.

It is not authentication. It slows a stranger down; it does not keep them out.
The passcode in the web layer is what keeps them out, and the two are worth
having separately: the gate stops people who should not be here at all, and this
stops the ones who should from costing more than intended by accident.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class Throttle:
    """How many times a caller may do something in a window.

    A sliding count rather than a fixed window, because a fixed one lets a caller
    spend the whole allowance at the end of one window and the whole of the next
    at its start — twice the intended rate, at the moment it matters most.
    """

    def __init__(
        self,
        *,
        limit: int,
        window: float,
        now: Callable[[], float] | None = None,
    ) -> None:
        #: Zero switches the limit off, which is how a deployment declines one
        #: without having to reason about a very large number.
        self.limit = limit
        self.window = window
        self._now = now or time.monotonic
        self._seen: dict[str, deque[float]] = {}

    def forget(self) -> None:
        """Drop every caller's history.

        For tests. These limiters are module-level singletons with a one-hour
        window, so without this every case in a file shares one allowance and the
        thirty-first upload in the file fails — as a 429 whose body has no
        `submission_id`, which reads as the upload endpoint being broken rather
        than as the suite having outgrown its budget.
        """
        self._seen.clear()

    def check(self, key: str) -> float | None:
        """Record a call, or say how many seconds until one would be allowed.

        Returning the wait rather than a bare refusal is the difference between a
        caller backing off and a caller retrying immediately and being refused
        again.
        """
        if self.limit <= 0:
            return None

        moment = self._now()
        self._forget_the_quiet(moment)

        calls = self._seen.setdefault(key, deque())
        cutoff = moment - self.window
        while calls and calls[0] <= cutoff:
            calls.popleft()

        if len(calls) >= self.limit:
            return calls[0] + self.window - moment

        calls.append(moment)
        return None

    def tracked(self) -> int:
        """How many callers are being remembered. For tests and diagnostics."""
        return len(self._seen)

    def _forget_the_quiet(self, moment: float) -> None:
        """Drop callers whose whole history has aged out.

        Without this the table grows for the lifetime of the process, keyed by
        something the caller controls, which is a slow memory leak with the whole
        internet holding the pen.
        """
        cutoff = moment - self.window
        stale = [key for key, calls in self._seen.items() if not calls or calls[-1] <= cutoff]
        for key in stale:
            del self._seen[key]
