"""Build two full-scale test submissions, one synthetic and one on real ink.

The samples this project has been demonstrated with are small — eight questions,
two pages — and small samples hid a failure that a real Class 9 mathematics paper
found immediately. These two are meant to be the opposite: full papers, full
scripts, and every edge case the design claims to handle present at once.

**Case A — scale.** An eighteen-entry science paper across three sections, with
sub-parts, printed marks and an "attempt any three" rule, answered by a five-page
script that is deliberately awkward: answers out of order, four questions skipped,
one answer crossed out and rewritten below, one merged blob covering two sub-parts,
one answer running across a page break, and a block of rough working that answers
nothing. Every label style used here is one the extractor already handles, so that
what this measures is everything *other* than label parsing — which
`label_matrix.py` covers separately and exhaustively.

**Case B — real ink.** The reading-comprehension paper, answered with genuine
handwritten pages from the Handwritten ASAP SAS corpus. Its two prompts happen to
be exactly this paper's first two questions, which is why the paper was written
against them in the first place. Nothing on those pages is synthetic: it is real
pencil, real scanning, real spelling mistakes. It is the only one of the two whose
transcription figures mean anything.

    uv run python tooling/scripts/build_full_papers.py

Writes into data/, which is not tracked — the ASAP corpus is licensed for
non-commercial research use and its images stay out of the repository.
"""

# ruff: noqa: E501 - the long lines are verbatim exam text and label strings;
# wrapping them would make it harder to compare against a real paper.

from __future__ import annotations

import random
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "full"
ASAP = ROOT / "data" / "asap" / "Handwritten ASAP SAS"

HANDWRITING = "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf"

PAGE_W, PAGE_H = 595, 842          # A4 at 72 dpi; rendering happens later at 200
MARGIN = 56


# ══════════════════════════════════════════════════════════════════════════
# Case A — the question paper
# ══════════════════════════════════════════════════════════════════════════

PAPER_A: list[tuple[str, str]] = [
    ("title", "Greenfield Public School"),
    ("sub", "Class 9  ·  Science  ·  Annual Examination 2026"),
    ("sub", "Time: 3 hours                                    Maximum Marks: 70"),
    ("rule", ""),
    ("instr", "General Instructions:"),
    ("instr", "(a) All questions are compulsory except where stated otherwise."),
    ("instr", "(b) In Section C, attempt any three of the four questions."),
    ("instr", "(c) Draw neat diagrams wherever necessary."),
    ("gap", ""),

    ("section", "SECTION A"),
    ("note", "(Each question carries 1 mark)"),
    ("q", "1. Define refraction of light."),
    ("q", "2. State the SI unit of pressure."),
    ("q", "3. Name the process by which plants lose water as water vapour."),
    ("q", "4. Write the chemical formula of washing soda."),
    ("q", "5. Write one difference between speed and velocity."),
    ("q", "6. Name the tissue that transports water in a plant."),
    ("gap", ""),

    ("section", "SECTION B"),
    ("note", "(Each question carries 3 marks)"),
    ("q", "7. Explain why a pencil appears bent when it is partly immersed in water. [3]"),
    ("q", "8. State Newton's second law of motion and derive the relation F = ma. [3]"),
    ("q", "9. Describe the structure of the plant cell wall and give one of its functions. [3]"),
    ("q", "10. Distinguish between an element, a compound and a mixture, with one example of each. [3]"),
    ("q", "11 (a) Define atomic number and mass number. [2]"),
    ("q", "11 (b) An atom has 11 protons and 12 neutrons. Give its atomic number and mass number. [1]"),
    ("q", "12. Explain, with one everyday example, how evaporation causes cooling. [3]"),
    ("gap", ""),

    ("section", "SECTION C"),
    ("note", "(Each question carries 5 marks. Attempt any three questions.)"),
    ("q", "13. Describe the water cycle and explain the part transpiration plays in it. [5]"),
    ("q", "14. (i) State the law of conservation of mass. [2]"),
    ("q", "14. (ii) Six grams of carbon burns completely in sixteen grams of oxygen. Find the mass of carbon dioxide formed and justify your answer. [3]"),
    ("q", "15. Explain the process of photosynthesis, naming the raw materials and the products, and describe one experiment that shows sunlight is necessary. [5]"),
    ("q", "16. Describe the three states of matter in terms of the arrangement of their particles, and explain what happens at the melting point and the boiling point. [5]"),
]


# ══════════════════════════════════════════════════════════════════════════
# Case A — the script, and what it is supposed to look like
# ══════════════════════════════════════════════════════════════════════════

#: (label written in the margin, lines of the answer, page, awkwardness)
#:
#: The awkwardness column is the whole point. Each value names an edge case the
#: design claims to handle, so a failure is attributable rather than merely a lower
#: number.
SCRIPT_A: list[dict] = [
    dict(page=0, label="1.", lines=[
        "Refraction is the bending of light when it passes",
        "from one medium into another of different density.",
    ], case="plain"),
    dict(page=0, label="2.", lines=[
        "The SI unit of pressure is the pascal (Pa).",
    ], case="plain"),
    dict(page=0, label="3.", lines=[
        "The process is called transpiration. Water evaporates",
        "from the stomata on the leaves.",
    ], case="plain"),
    dict(page=0, label="5.", lines=[
        "Speed is a scalar and has only magnitude. Velocity is",
        "a vector so it has magnitude and direction both.",
    ], case="q4 skipped — 4 and 6 unanswered"),

    dict(page=1, label="7.", lines=[
        "The pencil looks bent because light coming from the",
        "part under water bends away from the normal when it",
        "leaves the water and enters air. So the eye sees the",
        "lower part raised and the pencil appears broken.",
    ], case="plain"),
    dict(page=1, label="9.", lines=[
        "The cell wall is made of cellulose and lies outside the",
        "cell membrane. It is rigid and gives the cell its shape",
        "and protects it from bursting when water enters.",
    ], case="OUT OF ORDER — 9 written before 8"),
    dict(page=1, label="8.", lines=[
        "Newton's second law says the rate of change of momentum",
        "is proportional to the force applied. If mass m has",
        "velocity change from u to v in time t then F = m(v-u)/t",
        "which gives F = ma since a = (v-u)/t.",
    ], case="OUT OF ORDER — 8 written after 9"),

    dict(page=2, label="10.", lines=[
        "An element has only one kind of atom, like oxygen.",
        "A compound has two or more elements combined in a fixed",
        "ratio, like water. A mixture has substances just mixed",
        "and can be separated, like air.",
    ], case="plain"),
    dict(page=2, label="11.", lines=[
        "Atomic number is the number of protons in the nucleus and",
        "mass number is the total of protons and neutrons.",
        "For the given atom the atomic number is 11 and the mass",
        "number is 11 + 12 = 23. So it is sodium.",
    ], case="MERGED — one blob answering 11(a) and 11(b)"),
    dict(page=2, label="12.", lines=[
        "When a liquid evaporates the fastest particles escape",
        "first, so the average energy of the rest falls and the",
        "liquid gets cooler. That is why sweating cools us down.",
    ], case="plain"),

    dict(page=3, label="13.", lines=[
        "The water cycle is when water goes up and comes down",
        "again as rain from the sea only.",
    ], case="STRUCK THROUGH — abandoned first attempt", struck=True),
    dict(page=3, label=None, lines=[
        "Water evaporates from oceans, lakes and rivers by the",
        "heat of the sun. Plants also give out water vapour by",
        "transpiration, which adds a large amount to the air.",
        "The vapour rises, cools and condenses into clouds, and",
        "then falls back as rain, hail or snow. Some of it soaks",
        "into the ground and the rest runs back to the sea.",
    ], case="the rewrite of 13, directly below the crossing out"),
    dict(page=3, label="15.", lines=[
        "Photosynthesis is how green plants make their own food.",
        "The raw materials are carbon dioxide from the air and",
        "water from the soil, and sunlight is trapped by the",
        "chlorophyll in the leaves. The products are glucose and",
        "oxygen, which is given out through the stomata.",
        "cont. on next page",
    ], case="SPANS PAGES — continues on page 5"),

    dict(page=4, label=None, lines=[
        "To show sunlight is needed, take a destarched plant and",
        "cover a part of one leaf with black paper. Keep it in",
        "sunlight for some hours, then test that leaf with iodine.",
        "The covered part stays brown and the uncovered part turns",
        "blue-black, which proves starch is made only in light.",
    ], case="the continuation of 15"),
    dict(page=4, label="16.", lines=[
        "In a solid the particles are packed closely and only",
        "vibrate, so a solid has a fixed shape and volume. In a",
        "liquid they are further apart and can slide, so it takes",
        "the shape of the vessel. In a gas they are far apart and",
        "move freely and fill all the space.",
        "At the melting point the solid becomes a liquid and at",
        "the boiling point the liquid becomes a gas. The",
        "temperature does not rise while the change is going on",
        "because the heat is used to break the forces.",
    ], case="answers 3 of 4 in Section C, so 14 is not required"),
    dict(page=4, label=None, lines=[
        "6 + 16 = 22 g",
        "check 12 : 32 ratio",
    ], case="ORPHAN — rough working that answers no question", rough=True),
]


class Sheet:
    """A handwritten answer sheet, with the wobble a font does not have."""

    def __init__(self, seed: int = 7) -> None:
        self.doc = fitz.open()
        self.rng = random.Random(seed)
        self.page = None
        self.y = 0.0

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN + 18
        # Faint ruled lines, as on real answer paper. They also give the deskew
        # step something honest to measure.
        for i in range(28):
            y = MARGIN + 30 + i * 26
            self.page.draw_line(
                fitz.Point(MARGIN - 8, y), fitz.Point(PAGE_W - MARGIN + 8, y),
                color=(0.82, 0.85, 0.92), width=0.4,
            )

    def write(self, text: str, *, x: float, size: float = 12.5, struck: bool = False) -> None:
        # Per-line jitter, because a font baseline is perfectly straight and a
        # hand is not. The structural properties under test do not depend on this,
        # but a page that looks machine-set invites trusting figures it cannot support.
        jitter_x = self.rng.uniform(-1.6, 1.6)
        jitter_y = self.rng.uniform(-1.2, 1.2)
        actual = size * self.rng.uniform(0.95, 1.06)
        point = fitz.Point(x + jitter_x, self.y + jitter_y)
        try:
            self.page.insert_text(point, text, fontsize=actual,
                                  fontfile=HANDWRITING, fontname="hand",
                                  color=(0.13, 0.16, 0.34))
        except Exception:
            self.page.insert_text(point, text, fontsize=actual, fontname="helv",
                                  color=(0.13, 0.16, 0.34))
        if struck:
            width = len(text) * actual * 0.46
            self.page.draw_line(
                fitz.Point(point.x - 2, point.y - actual * 0.32),
                fitz.Point(point.x + width, point.y - actual * 0.30),
                color=(0.13, 0.16, 0.34), width=1.1,
            )
        self.y += 26

    def gap(self, n: float = 1.0) -> None:
        self.y += 26 * n


def build_paper_a(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN

    for kind, text in PAPER_A:
        if y > PAGE_H - MARGIN - 40:
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
                           color=(0.3, 0.3, 0.3), width=0.8)
            y += 18
        elif kind == "instr":
            page.insert_text(fitz.Point(MARGIN, y), text, fontsize=9.5, fontname="helv")
            y += 14
        elif kind == "section":
            y += 8
            page.insert_text(fitz.Point(MARGIN, y), text, fontsize=12, fontname="hebo")
            y += 16
        elif kind == "note":
            page.insert_text(fitz.Point(MARGIN, y), text, fontsize=9, fontname="helv")
            y += 18
        elif kind == "gap":
            y += 12
        else:
            for i, chunk in enumerate(_wrap(text, 88)):
                page.insert_text(fitz.Point(MARGIN + (0 if i == 0 else 14), y),
                                 chunk, fontsize=10.5, fontname="helv")
                y += 15
            y += 7

    doc.save(path)
    doc.close()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_script_a(path: Path) -> None:
    sheet = Sheet()
    current_page = -1

    for entry in SCRIPT_A:
        if entry["page"] != current_page:
            sheet.new_page()
            current_page = entry["page"]
            sheet.gap(0.4)

        x = MARGIN + (40 if entry.get("rough") else 0)
        if entry["label"]:
            sheet.write(entry["label"], x=MARGIN - 26, size=13)
            sheet.y -= 26

        for line in entry["lines"]:
            sheet.write(line, x=x + 14, size=12.5, struck=entry.get("struck", False))
        sheet.gap(0.5)

    sheet.doc.save(path)
    sheet.doc.close()


# ══════════════════════════════════════════════════════════════════════════
# Case B — real handwriting
# ══════════════════════════════════════════════════════════════════════════

#: Pages from the ASAP corpus, in the order a script would carry them.
#:
#: Two answers to the article's first question and two to its second, so the sheet
#: exercises a genuine multi-page answer as well as real transcription. Chosen by
#: id rather than at random so the run is repeatable and the ground-truth
#: transcriptions in `information/prompt-3.txt` line up with what is on the page.
REAL_PAGES = [
    ("prompt-3", "SAS_3_6809.png"),
    ("prompt-3", "SAS_3_6812.png"),
    ("prompt-4", "SAS_4_10002.png"),
    ("prompt-4", "SAS_4_10003.png"),
]


def build_script_b(path: Path) -> int:
    doc = fitz.open()
    used = 0
    for folder, name in REAL_PAGES:
        source = ASAP / folder / name
        if not source.is_file():
            print(f"  missing, skipped: {source}")
            continue
        pixmap = fitz.Pixmap(str(source))
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        # Fit the scan to the page, preserving aspect — the pipeline normalises
        # geometry, so only the ratio has to survive.
        scale = min(PAGE_W / pixmap.width, PAGE_H / pixmap.height)
        w, h = pixmap.width * scale, pixmap.height * scale
        rect = fitz.Rect((PAGE_W - w) / 2, (PAGE_H - h) / 2,
                         (PAGE_W - w) / 2 + w, (PAGE_H - h) / 2 + h)
        page.insert_image(rect, pixmap=pixmap)
        used += 1
    doc.save(path)
    doc.close()
    return used


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    paper_a = OUT / "science_annual_paper.pdf"
    script_a = OUT / "science_annual_script.pdf"
    build_paper_a(paper_a)
    build_script_a(script_a)
    print(f"  case A paper   {paper_a.relative_to(ROOT)}  "
          f"({paper_a.stat().st_size / 1024:.0f} kB)")
    print(f"  case A script  {script_a.relative_to(ROOT)}  "
          f"({script_a.stat().st_size / 1024:.0f} kB)")

    script_b = OUT / "real_ink_script.pdf"
    pages = build_script_b(script_b)
    print(f"  case B script  {script_b.relative_to(ROOT)}  "
          f"({pages} real pages, {script_b.stat().st_size / 1024:.0f} kB)")

    print("\n  case A expects:")
    for entry in SCRIPT_A:
        if entry["case"] != "plain":
            label = entry["label"] or "—"
            print(f"    {label:5} {entry['case']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
