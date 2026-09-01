"""Realistic papers around real student handwriting.

Everything else in the corpus is either a generated script — handwriting fonts,
cleanly baselined, nothing like a scan — or one of four ASAP pages the pipeline
has already been fixed against. This builds papers around the **untouched** ones.

The source is ``data/asap/Handwritten ASAP SAS`` (Gold & Zesch, ICFHR 2020): 350
photographed pages of genuine student writing in blue, black and green pen, with
a human transcription of each in ``information/prompt-3.txt``. Four are used by
``build_full_papers.py``; the rest have never been looked at.

The question is the real one. ASAP-SAS set 3 gives students an article about the
exotic pet trade that distinguishes *generalist* species, such as pythons, from
*specialist* species, such as pandas and koalas, and then asks:

    Explain how pandas in China are similar to koalas in Australia and how they
    both are different from pythons. Support your response with information from
    the article.

So the paper carries an extract of that article as printed **material**, which is
the first time the material path is exercised on real handwriting rather than on a
generated table. It also carries two questions nobody answered, so blank detection
and placement have something to get wrong.

Three scripts are built, chosen across the range the recognition harness measured
rather than at random — one page it read perfectly, one middling, one its worst.
Each is one page, which is what a one-question paper honestly produces.

    uv run python tooling/scripts/build_asap_papers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_full_papers import MARGIN, PAGE_H, PAGE_W, _wrap  # noqa: E402

ASAP = ROOT / "data" / "asap" / "Handwritten ASAP SAS"
OUT = ROOT / "data" / "asap-real"

#: An extract of the article the question refers to.
#:
#: Written from the published description of the prompt's stimulus rather than
#: reproduced, because the article itself is not in this dataset. It carries the
#: two facts every answer is scored against — what a specialist is and what a
#: generalist is — so the material is doing real work rather than decorating the
#: page.
ARTICLE = [
    '"The trade in exotic pets has grown quickly, and nowhere faster than in',
    'reptiles. Traders say a snake in a tank harms nobody. Biologists disagree:',
    'released or escaped, some of these animals thrive where they do not belong.',
    'The pythons now breeding in Florida are the example everyone reaches for.',
    'What makes them so successful is that they are generalists. A generalist',
    'tolerates a wide range of habitats, temperatures and foods, and does best in',
    'and around humans. A specialist does the opposite. It depends on a narrow',
    'food source and a stable place to live: the panda, which eats almost nothing',
    'but bamboo, or the koala bear, which eats eucalyptus leaves almost',
    'exclusively. Specialists are the first to suffer when their habitat shifts."',
    '                              — adapted from a magazine article on the pet trade',
]

#: The paper. Q1 is the real ASAP prompt; the other two are there so that a blank
#: is available to be detected and a wrong placement is available to be made.
PAPER = [
    ("title", "Springfield Middle School"),
    ("sub", "Grade 8  ·  Reading and Science  ·  Unit Assessment"),
    ("sub", "Time: 40 minutes                                Maximum Marks: 9"),
    ("rule", ""),
    ("instr", "Answer all questions. Each question carries 3 marks."),
    ("gap", ""),
    ("stem", "Read the source below and answer the questions that follow."),
    ("article", ""),
    ("gap", ""),
    ("q", "1.  Explain how pandas in China are similar to koalas in Australia and how "
          "they both are different from pythons. Support your response with information "
          "from the article."),
    ("gap", ""),
    ("q", "2.  Using the article, explain why pythons have been able to spread in "
          "Florida."),
    ("gap", ""),
    ("q", "3.  Give one reason a biologist might oppose the trade in exotic pets."),
]

#: Untouched pages, picked across the measured range rather than at random so the
#: three links are a spread and not three samples of the same thing.
SCRIPTS = [
    ("clean", "SAS_3_6854.png", "the recognition harness read this one at CER 0.000"),
    ("middling", "SAS_3_6996.png", "CER 0.027 — quotes the article, spelling errors"),
    ("worst", "SAS_3_6859.png", "CER 0.096 — the worst of the thirty pages measured"),
]


def build_paper(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN

    for kind, text in PAPER:
        if y > PAGE_H - MARGIN - 80:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = MARGIN

        if kind == "title":
            page.insert_text(fitz.Point(MARGIN, y), text, fontsize=17, fontname="hebo")
            y += 24
        elif kind == "sub":
            page.insert_text(fitz.Point(MARGIN, y), text, fontsize=10.5, fontname="helv")
            y += 16
        elif kind == "rule":
            page.draw_line(fitz.Point(MARGIN, y), fitz.Point(PAGE_W - MARGIN, y),
                           color=(0.4, 0.4, 0.4), width=0.6)
            y += 18
        elif kind == "instr":
            page.insert_text(fitz.Point(MARGIN, y), text, fontsize=9.5, fontname="helv")
            y += 15
        elif kind == "stem":
            page.insert_text(fitz.Point(MARGIN, y), text, fontsize=10, fontname="helv")
            y += 16
        elif kind == "article":
            # Indented and italic, as a printed source extract is set. The indent is
            # what a reader uses to tell it from the questions, and the pipeline has
            # to manage without it.
            for line in ARTICLE:
                page.insert_text(fitz.Point(MARGIN + 22, y), line,
                                 fontsize=9.5, fontname="hebi")
                y += 14
        elif kind == "gap":
            y += 10
        else:
            for i, line in enumerate(_wrap(text, 92)):
                page.insert_text(fitz.Point(MARGIN + (0 if i == 0 else 22), y),
                                 line, fontsize=10, fontname="helv")
                y += 15

    doc.save(path)
    doc.close()


def build_script(image: Path, path: Path) -> bool:
    """One real page, fitted to A4 with its aspect preserved."""
    if not image.is_file():
        return False
    doc = fitz.open()
    pixmap = fitz.Pixmap(str(image))
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    scale = min(PAGE_W / pixmap.width, PAGE_H / pixmap.height)
    w, h = pixmap.width * scale, pixmap.height * scale
    page.insert_image(
        fitz.Rect((PAGE_W - w) / 2, (PAGE_H - h) / 2, (PAGE_W + w) / 2, (PAGE_H + h) / 2),
        pixmap=pixmap,
    )
    del pixmap
    doc.save(path)
    doc.close()
    return True


def main() -> int:
    if not ASAP.is_dir():
        print(f"  no ASAP data at {ASAP}")
        return 1

    used_elsewhere = {"SAS_3_6809", "SAS_3_6812", "SAS_4_10002", "SAS_4_10003"}
    for _, name, _ in SCRIPTS:
        if Path(name).stem in used_elsewhere:
            print(f"  ! {name} is already used by build_full_papers.py — not held out")
            return 1

    for label, name, note in SCRIPTS:
        folder = OUT / f"asap-{label}"
        folder.mkdir(parents=True, exist_ok=True)
        build_paper(folder / "paper.pdf")
        ok = build_script(ASAP / "prompt-3" / name, folder / "script.pdf")
        state = "built" if ok else "MISSING IMAGE"
        print(f"  asap-{label:9} {state:14} {name:18} {note}")

    print(f"\n  written to {OUT.relative_to(ROOT)}")
    print("  paper: 3 questions, 9 marks, with the article printed as material")
    print("  script: one real handwritten page answering question 1 only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
