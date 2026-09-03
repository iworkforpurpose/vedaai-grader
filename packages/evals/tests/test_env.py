"""The loader that lets the harness see the same credentials the service does.

These are small tests for a small function, and they exist because the failure it
prevents is silent: with no key the scorer degrades to word overlap and the
grader degrades to awarding nothing, and both look like results rather than like
a missing file.
"""

from __future__ import annotations

import os

from vedaai_evals.env import find_repo_env, load_repo_env


def test_reads_key_and_value(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-test\nGRADER_MODEL=gpt-4.1\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GRADER_MODEL", raising=False)

    assert load_repo_env(tmp_path) == tmp_path / ".env"
    assert os.environ["OPENAI_API_KEY"] == "sk-test"
    assert os.environ["GRADER_MODEL"] == "gpt-4.1"


def test_never_overrides_what_is_already_set(tmp_path, monkeypatch):
    """The environment wins.

    An A/B run sets ``GRADER_MODEL`` on the command line and expects that model to
    be the one measured. A loader that overrode it would report the comparison as
    a null result, which is the most expensive kind of wrong answer here.
    """
    (tmp_path / ".env").write_text("GRADER_MODEL=from-the-file\n")
    monkeypatch.setenv("GRADER_MODEL", "from-the-command-line")

    load_repo_env(tmp_path)

    assert os.environ["GRADER_MODEL"] == "from-the-command-line"


def test_absent_file_is_not_an_error(tmp_path):
    """CI and a fresh clone have no ``.env`` and are meant to work anyway."""
    assert find_repo_env(tmp_path) is None
    assert load_repo_env(tmp_path) is None


def test_ignores_comments_blanks_and_export(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# a comment\n"
        "\n"
        "   # an indented comment\n"
        "export OCR_ENGINE=textract\n"
        "NOT_AN_ASSIGNMENT\n"
    )
    monkeypatch.delenv("OCR_ENGINE", raising=False)

    load_repo_env(tmp_path)

    assert os.environ["OCR_ENGINE"] == "textract"
    assert "NOT_AN_ASSIGNMENT" not in os.environ


def test_strips_matching_quotes_but_keeps_a_hash_inside_a_value(tmp_path, monkeypatch):
    """A secret may contain ``#``, so an unquoted value is taken whole.

    Guessing that everything after a ``#`` is a comment corrupts such a key into
    one that authenticates against nothing, and the resulting 401 names the key
    rather than the parser.
    """
    (tmp_path / ".env").write_text('ACCESS_CODE="quoted value"\nSECRET=abc#def\n')
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    monkeypatch.delenv("SECRET", raising=False)

    load_repo_env(tmp_path)

    assert os.environ["ACCESS_CODE"] == "quoted value"
    assert os.environ["SECRET"] == "abc#def"


def test_finds_the_file_above_the_starting_directory(tmp_path, monkeypatch):
    """Scripts run from anywhere in the tree; the file lives at the root."""
    (tmp_path / ".env").write_text("S3_PAGE_PREFIX=pages/\n")
    deep = tmp_path / "packages" / "evals" / "src"
    deep.mkdir(parents=True)
    monkeypatch.delenv("S3_PAGE_PREFIX", raising=False)

    assert find_repo_env(deep) == tmp_path / ".env"
    load_repo_env(deep)
    assert os.environ["S3_PAGE_PREFIX"] == "pages/"
