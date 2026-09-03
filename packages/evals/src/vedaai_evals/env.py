"""Load the repository's ``.env`` before anything reads ``os.environ``.

Every accuracy number this project publishes comes from a script in this package
or in ``tooling/scripts``, and not one of them could see ``.env``. The documented
commands are ``pnpm turbo eval`` and ``pnpm --filter @vedaai/evals gate``, which
run ``uv run`` without ``--env-file``, so ``OPENAI_API_KEY`` was absent unless a
developer happened to have exported it into their shell.

That is not a cosmetic gap. Two things read the key at import time and degrade
silently without it:

* ``grader.answers.similarity.default_similarity`` falls back from embeddings to
  word overlap, so mapping is scored by spelling while the service scores by
  meaning. The README records that this mismatch already cost three points of
  accuracy and caused a revert, and the harness has been printing "NOT what the
  service uses" on every run since.
* ``grading.select_grader`` falls back to ``RubricOnly``, which awards nothing —
  so the gate would mark every document zero and report nine failures that are
  really one missing variable.

Loaded here rather than by adding ``--env-file`` to each package script, for two
reasons. ``uv run --env-file`` is a hard error when the file is absent, which
would break a fresh clone and CI, where there is no ``.env`` and none is wanted.
And a loader in the harness covers every entry point at once, including the ones
invoked as ``python tooling/scripts/...`` rather than through pnpm.

**Never overrides.** A variable already in the environment wins, so CI, a
one-off ``OPENAI_API_KEY=... uv run ...``, and an A/B run that sets
``GRADER_MODEL`` on the command line all behave the way they read.

**Call it before importing ``grader``.** ``default_similarity`` is built at import
time, so a load that happens afterwards changes nothing and looks like it worked.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Filename searched for, walking up from the caller towards the filesystem root.
ENV_FILENAME = ".env"


def find_repo_env(start: Path | None = None) -> Path | None:
    """The nearest ``.env`` at or above ``start``, or None if there is none."""
    here = (start or Path(__file__)).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / ENV_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_repo_env(start: Path | None = None) -> Path | None:
    """Read the repository ``.env`` into ``os.environ``. Returns the file used.

    Absent is not an error: a fresh clone and CI both run without one, and the
    pipeline is designed to work with no credentials at all.
    """
    path = find_repo_env(start)
    if path is None:
        return None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Quotes are stripped, and only as a matching pair. An unquoted value
        # keeps everything after the '=' including any '#', because a secret may
        # legitimately contain one and guessing wrong silently corrupts a key.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)

    return path
