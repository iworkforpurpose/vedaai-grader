"""How well real handwriting is actually read.

Named the binding ceiling on everything downstream since the first assessment,
and never given a number. Every mapping and highlight figure the project
publishes was measured on generated pages read from their text layer, so all of
them assume recognition works and none of them test it.

**The data was already on disk.** ``data/asap/Handwritten ASAP SAS`` (Gold & Zesch,
ICFHR 2020) holds 350 photographed pages of genuine student handwriting in blue,
black and green pen — and, in ``information/prompt-3.txt``, a **human
transcription of every one**. That transcription column is the ground truth that
was supposedly missing.

**346 of the 350 are untouched.** ``build_full_papers.py`` uses exactly four
images, so the rest have never been looked at, tuned against, or fixed for. This
is a genuine held-out set for the half of the pipeline SciEntsBank cannot reach.

What this measures and what it does not:

* **Character error rate** — measurable exactly, because the transcription is
  word-for-word. The README quotes 0.027 for Textract from five pages; this says
  what it is on fifty.
* **Word recall** — the share of ground-truth words that appear anywhere in the
  output. A proxy for line recall, and an honest one as long as it is named as a
  proxy: it cannot tell a line that was missed from a line that was read in the
  wrong place.
* **Line-level detection recall is still not measurable.** That needs ground-truth
  *boxes*, and this dataset has transcriptions only. Nothing here closes that gap
  and the report says so rather than implying otherwise.

    uv run python tooling/scripts/score_recognition.py --n 50
    uv run python tooling/scripts/score_recognition.py --n 20 --prompt prompt-4
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "evals" / "src"))

# The repository's `.env`, before `grader` is imported. Both the scorer and the
# grader are selected from the environment at import time, so a load placed with
# the imports below would silently leave this run measuring word overlap and
# marking nothing. Absent is fine; nothing here is overridden if already set.
from vedaai_evals.env import load_repo_env  # noqa: E402

load_repo_env()

from grader import render  # noqa: E402
from grader.lineindex import build_index  # noqa: E402
from grader.ocr import EngineUnavailable, PageInput, select_engine  # noqa: E402
from grader.storage import PageStore  # noqa: E402
from vedaai_contracts import DocumentKind  # noqa: E402
from vedaai_evals import metrics  # noqa: E402

ASAP = ROOT / "data" / "asap" / "Handwritten ASAP SAS"

#: The four images the corpus already uses, excluded so this stays held out.
ALREADY_USED = {"SAS_3_6809", "SAS_3_6812", "SAS_4_10002", "SAS_4_10003"}


@dataclass
class PageResult:
    name: str
    truth: str
    read: str
    lines: int
    status: str

    @property
    def cer(self) -> float:
        return metrics.character_error_rate(normalise(self.truth), normalise(self.read))

    @property
    def word_recall(self) -> float:
        """Share of ground-truth words that appear in the output.

        Multiset rather than set, so an answer that repeats a word does not get
        credit for it twice, and a recognizer that drops one of two occurrences
        is penalised for it.
        """
        want = words(self.truth)
        if not want:
            return 1.0
        got = words(self.read)
        pool = list(got)
        hit = 0
        for w in want:
            if w in pool:
                pool.remove(w)
                hit += 1
        return hit / len(want)


@dataclass
class Report:
    pages: list[PageResult] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def _mean(self, values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @property
    def cer(self) -> float | None:
        return self._mean([p.cer for p in self.pages])

    @property
    def word_recall(self) -> float | None:
        return self._mean([p.word_recall for p in self.pages])

    @property
    def median_cer(self) -> float | None:
        vals = sorted(p.cer for p in self.pages)
        return vals[len(vals) // 2] if vals else None

    @property
    def worst(self) -> list[PageResult]:
        return sorted(self.pages, key=lambda p: -p.cer)[:5]


_PUNCT = re.compile(r"[^a-z0-9\s]+")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, punctuation stripped, whitespace collapsed.

    Compared this way because the product does not care about punctuation: the
    aligner scores meaning and the grader is told to judge the answer rather than
    the spelling of the transcription. A CER that counted a missing comma would
    describe a problem nobody has.
    """
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def words(text: str) -> list[str]:
    return normalise(text).split()


def truth_for(prompt: str) -> dict[str, str]:
    """Ground-truth transcriptions, keyed by image stem.

    Tab-separated: file, ASAP-ID, transcription, pen colour, status. Rows whose
    status is not ``ok`` are kept and reported separately — ``oob`` means the
    writer ran outside the box so text is genuinely absent from the image, and
    scoring those as recognition failures would blame the recognizer for the
    dataset.
    """
    path = ASAP / "information" / f"{prompt}.txt"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        out[Path(parts[0]).stem] = "\t".join([parts[2], parts[4]])
    return out


def read_page(path: Path, store: PageStore) -> tuple[str, int]:
    """Put one image through the real ingest path and return what was read."""
    data = path.read_bytes()
    source = render.inspect(data, path.name, DocumentKind.ANSWER_SHEET)
    engine = select_engine(source)

    texts: list[str] = []
    count = 0
    for rendered in render.render_pages(data, source, store):
        png = rendered.png or (
            store.read(rendered.page.image_key)
            if store.exists(rendered.page.image_key)
            else None
        )
        lines = engine.transcribe(
            PageInput(
                index=rendered.page.index,
                width=rendered.page.width,
                height=rendered.page.height,
                png=png,
                document=data,
                filename=source.filename,
            )
        )
        # Through the real index so reading order is the product's, not the
        # engine's — a page read correctly but ordered wrongly is a different
        # fault and this measures the one the product actually ships.
        index = build_index(DocumentKind.ANSWER_SHEET, [lines], engine.engine)
        texts.extend(ln.text for ln in index.lines)
        count += len(index.lines)
    return " ".join(texts), count


def pct(v: float | None) -> str:
    return "   n/a" if v is None else f"{v * 100:5.1f}%"


def num(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:5.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="prompt-3", choices=["prompt-3", "prompt-4"])
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pages", type=Path, default=ROOT / "packages" / "generated" / ".recog")
    args = parser.parse_args(argv)

    folder = ASAP / args.prompt
    if not folder.is_dir():
        raise SystemExit(f"no such folder: {folder}")

    truth = truth_for(args.prompt)
    if not truth:
        raise SystemExit(
            f"no ground-truth transcriptions for {args.prompt} — only prompt-3 ships them"
        )

    candidates = sorted(
        p for p in folder.glob("*.png") if p.stem not in ALREADY_USED and p.stem in truth
    )
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    chosen = candidates[: args.n]

    store = PageStore(root=args.pages)
    report = Report()
    print(f"{args.prompt} · {len(chosen)} of {len(candidates)} held-out pages with truth")
    print("(the 4 images build_full_papers.py uses are excluded)\n")

    for i, path in enumerate(chosen, start=1):
        record = truth[path.stem]
        want, status = record.split("\t", 1) if "\t" in record else (record, "ok")
        try:
            read, lines = read_page(path, store)
        except EngineUnavailable as exc:
            report.failed.append((path.stem, f"engine unavailable: {exc}"))
            break
        except Exception as exc:  # noqa: BLE001 - one page must not stop the run
            report.failed.append((path.stem, f"{type(exc).__name__}: {exc}"))
            continue
        result = PageResult(
            name=path.stem, truth=want, read=read, lines=lines, status=status.strip()
        )
        report.pages.append(result)
        print(f"  {i:>3}/{len(chosen)}  {path.stem:16} lines={lines:>3} "
              f"cer={result.cer:.3f} recall={result.word_recall * 100:.0f}% [{result.status}]")

    ok = [p for p in report.pages if p.status == "ok"]
    clean = Report(pages=ok)

    print(f"\n{'═' * 72}\nREAL-PAGE RECOGNITION · {len(report.pages)} pages read"
          + (f" · {len(report.failed)} failed" if report.failed else ""))
    print(f"\n  pages with status ok     {len(ok)} of {len(report.pages)}")
    print(f"  CHARACTER ERROR RATE     {num(clean.cer)}   mean over ok pages")
    print(f"    median                 {num(clean.median_cer)}")
    print(f"  word recall              {pct(clean.word_recall)}"
          f"   proxy for line recall, not a substitute")
    print(f"  all pages incl. oob/er   cer {num(report.cer)}  recall {pct(report.word_recall)}")

    if clean.worst:
        print("\n  worst pages")
        for p in clean.worst:
            print(f"    {p.name:16} cer={p.cer:.3f}  truth: {normalise(p.truth)[:52]!r}")
            print(f"    {'':16}              read : {normalise(p.read)[:52]!r}")
    for name, why in report.failed[:5]:
        print(f"  ! {name}: {why}")

    print("\n  Line-level detection recall is still unmeasured: that needs ground-truth")
    print("  boxes and this dataset carries transcriptions only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
