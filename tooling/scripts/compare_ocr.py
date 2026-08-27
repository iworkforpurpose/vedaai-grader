#!/usr/bin/env python3
"""Put two handwriting engines side by side on real pages.

Recall is the binding ceiling on everything downstream — a line no engine detects
is a highlight that cannot exist — so swapping recognizers without measuring would
be swapping the ceiling blind.

    tooling/scripts/compare_ocr.py data/handwriting --limit 6
    tooling/scripts/compare_ocr.py "data/asap/Handwritten ASAP SAS/prompt-3" --limit 8

Two figures per page, and they answer different questions.

**Detected regions** is the recall proxy. It needs no ground truth, which is what
makes it usable on any page: an engine that finds twenty lines where the other
finds twelve is seeing writing the other misses. It is a proxy and not a
measurement — an engine could also be splitting one line into three — so it is
reported per page rather than averaged, where a suspicious jump stays visible.

**Character error rate** needs ground truth and is therefore only computed where
a transcription file exists, which for these corpora means the ASAP prose set. It
is the figure that says whether the text is usable for matching answers to
questions, and it is the one that mattered most: word-level similarity scored
exactly zero on the code scripts because recognition had destroyed the words.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "apps" / "api" / "src"))


def ground_truth(directory: Path) -> dict[str, str]:
    """Transcriptions for a directory, if the corpus ships them.

    The ASAP release keeps them one level up in ``information/<prompt>.txt`` as
    tab-separated rows.
    """
    prompt = directory.name
    path = directory.parent / "information" / f"{prompt}.txt"
    if not path.is_file():
        return {}

    out: dict[str, str] = {}
    for row in path.read_text(errors="replace").splitlines():
        if row.startswith("#") or "\t" not in row:
            continue
        fields = row.split("\t")
        if len(fields) >= 3:
            out[fields[0]] = fields[2].strip()
    return out


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over characters, divided by the reference length."""
    reference = " ".join(reference.split())
    hypothesis = " ".join(hypothesis.split())
    if not reference:
        return 0.0 if not hypothesis else 1.0

    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, start=1):
        current = [i]
        for j, got in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (expected != got),
                )
            )
        previous = current
    return previous[-1] / len(reference)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    from grader import render
    from grader.ocr import PaddleOcrEngine, TextractEngine
    from grader.ocr.base import EngineUnavailable, PageInput
    from vedaai_contracts import DocumentKind

    directory = args.directory if args.directory.is_absolute() else REPO / args.directory
    images = sorted(
        p
        for p in directory.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )[: args.limit]
    if not images:
        print(f"no images in {directory}", file=sys.stderr)
        return 1

    truth = ground_truth(directory)
    engines = [("textract", TextractEngine()), ("paddle", PaddleOcrEngine())]
    usable = [(name, engine) for name, engine in engines if engine.available()]

    for name, engine in engines:
        if (name, engine) not in usable:
            hint = (
                "fill in AWS_REGION and a key pair in .env, then re-run with "
                "`uv run --env-file .env`"
                if name == "textract"
                else "install the local model with `uv sync --extra ocr-local` in apps/api"
            )
            print(f"skipping {name}: not available — {hint}", file=sys.stderr)

    if not usable:
        return 1

    header = f"{'page':26}"
    for name, _ in usable:
        header += f"{name + ' regions':>16}{name + ' conf':>13}"
        if truth:
            header += f"{name + ' CER':>12}"
    print(header)
    print("-" * len(header))

    totals: dict[str, list[float]] = {name: [] for name, _ in usable}
    region_totals: dict[str, int] = {name: 0 for name, _ in usable}
    # Reported once at the end rather than per page: the same credential problem
    # repeats on every page and burying the table in it helps nobody.
    failures: dict[str, str] = {}

    for image in images:
        data = image.read_bytes()
        source = render.inspect(data, image.name, DocumentKind.ANSWER_SHEET)
        width, height = render.page_size(data, image.name, 0)

        row = f"{image.stem[:25]:26}"
        for name, engine in usable:
            try:
                lines = engine.transcribe(
                    PageInput(
                        index=0, width=width, height=height, png=data, filename=image.name
                    )
                )
            except EngineUnavailable as exc:
                row += f"{'failed':>16}{'':>13}"
                if truth:
                    row += f"{'':>12}"
                failures.setdefault(name, str(exc))
                continue

            confidences = [line.confidence for line in lines]
            mean = sum(confidences) / len(confidences) if confidences else 0.0
            region_totals[name] += len(lines)
            row += f"{len(lines):>16}{mean:>13.3f}"

            if truth:
                expected = truth.get(image.name, "")
                if expected:
                    got = " ".join(line.text for line in lines)
                    cer = character_error_rate(expected, got)
                    totals[name].append(cer)
                    row += f"{cer:>12.3f}"
                else:
                    row += f"{'-':>12}"
        print(row)

    print()
    for name, message in failures.items():
        print(f"  {name} could not run: {message}", file=sys.stderr)
    if failures:
        print(file=sys.stderr)

    for name, _ in usable:
        summary = f"  {name:10} {region_totals[name]:4d} regions total"
        if totals[name]:
            summary += f"   mean CER {sum(totals[name]) / len(totals[name]):.3f}"
        print(summary)

    if truth:
        print(
            "\n  CER is over the whole page joined into one string, so it also "
            "penalises\n  reading order, not only character accuracy."
        )
    elif not truth:
        print(
            "\n  No ground truth in this directory, so only the region count is "
            "shown.\n  It is a recall proxy: more regions usually means less missed "
            "writing, but\n  an engine splitting one line into three would look the "
            "same."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
