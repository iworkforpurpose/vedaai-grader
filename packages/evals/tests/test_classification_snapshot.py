"""The classification snapshot, as a test that runs on every push.

This is the gate the project did not have. Nearly every regression it shipped was
a line changing role — a table's numbers becoming page numbers, a source extract
becoming the question above it, a margin label being deleted, a rubric line being
swallowed by a material scope so a paper lost its denominators. All of them were
found by a person reading output, days later.

Deliberately offline and deliberately fast. It needs no API key, no AWS
credentials and no network, because a gate that needs any of those is a gate that
gets skipped, and the classifier is pure given lines. The papers whose question
paper is a scan cannot be covered here for the same reason — recognition is not
deterministic — and the marking gate covers those instead.

If this fails, read the diff before doing anything. The intended change and the
accidental one look identical in a summary and completely different in the diff.
"""

from __future__ import annotations

import pytest
from vedaai_evals import classification


def available(name: str) -> bool:
    return (classification.ROOT / classification.PAPERS[name] / "paper.pdf").is_file()


@pytest.mark.parametrize("name", sorted(classification.PAPERS))
def test_no_line_changed_role(name: str) -> None:
    if not available(name):
        pytest.skip(f"{name} fixture not built — run the paper builders")
    problems = classification.check(name)
    assert not problems, (
        f"\n{name}: the parser now classifies these lines differently.\n"
        + "\n".join(problems)
        + "\n\nIf that is intended, run "
        "`python -m vedaai_evals.classification --update` and commit the snapshot "
        "with the code change. If it is not, this is the regression."
    )


def test_the_snapshot_suite_has_not_quietly_shrunk() -> None:
    """A suite that covers less than it did stops catching things.

    Cheap, and it exists because the failure it guards is invisible: a fixture
    directory that stops being built makes every test above skip, the run stays
    green, and the classifier is then unguarded without anybody deciding that.
    """
    built = [n for n in classification.PAPERS if available(n)]
    recorded = list(classification.SNAPSHOTS.glob("*.txt"))
    assert len(recorded) >= 8, (
        f"only {len(recorded)} snapshot(s) recorded; 8 papers carry a text layer. "
        "A missing snapshot is an unguarded paper."
    )
    if built:
        assert len(built) >= 5, (
            f"only {len(built)} of {len(classification.PAPERS)} fixtures are built, so "
            "most of this suite is skipping. Rebuild them."
        )


def test_material_reaches_the_questions_that_refer_to_it() -> None:
    """The property behind the snapshot, asserted directly.

    A snapshot catches a change; this catches the thing being wrong in the first
    place. Both papers named here had a question marked without its own input —
    the history source extract and the economics table — and neither failure was
    visible in any number the project reported at the time.
    """
    from grader.questions.extract import extract as extract_paper

    for name, label, least in (("history", "Q.4", 3), ("economics", "Q3.", 6)):
        if not available(name):
            pytest.skip(f"{name} fixture not built")
        folder = classification.ROOT / classification.PAPERS[name]
        rows = classification.classify_paper(folder)  # noqa: F841 - ensures it parses
        paper = extract_paper(_index_for(folder))
        question = next((q for q in paper.questions if q.label_raw == label), None)
        assert question is not None, f"{name}: {label} was not extracted at all"
        assert len(question.material) >= least, (
            f"{name} {label} should carry at least {least} lines of printed material "
            f"and carries {len(question.material)}. The paper prints it; if it is not "
            "attached, the marker is judging an answer without the input it needs."
        )


def _index_for(folder):
    """The paper's line index, built the way the snapshot builds it."""
    import fitz
    from grader.lineindex import build_index
    from grader.ocr import PdfTextLayerEngine
    from grader.ocr.base import PageInput
    from vedaai_contracts import DocumentKind

    data = (folder / "paper.pdf").read_bytes()
    document = fitz.open(stream=data, filetype="pdf")
    count = document.page_count
    document.close()
    engine = PdfTextLayerEngine()
    per_page = [
        engine.transcribe(
            PageInput(index=i, width=1000, height=1400, document=data, filename="paper.pdf")
        )
        for i in range(count)
    ]
    return build_index(
        DocumentKind.QUESTION_PAPER, per_page, engine.engine, trust_engine_order=True
    )
