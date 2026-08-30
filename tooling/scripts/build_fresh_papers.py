"""Four papers the system has never seen, in subjects and layouts it has not met.

Every fault found this week was found by a real document, never by the harness,
and the corpus is small enough that fixing against it stops it being evidence. So
these are new: four subjects, four label conventions, four page layouts, and a set
of student behaviours chosen because each one has broken something before or is
the kind of thing that would.

None of the label styles here appear in the existing corpus, which is the point —
the corpus can only ever confirm what it already contains.

**History.** `Q.1` labels with lettered sub-parts, marks stated once on the cover
rather than per section, and a printed source extract in quotation marks that is
not a question and must not be read as one. The student answers question 3 last,
on a page of its own, and skips question 5 entirely.

**Geography.** Numbered questions with roman sub-parts under a stem, and a figure
sitting between the stem and its sub-parts — the shape that turned a mathematics
question into the text "D C". One question asks for a labelled sketch, which the
student leaves blank while answering everything around it.

**English.** A section with "answer any two of the following", answered two of
three, so the third is not unanswered but not required. Every answer quotes the
passage in inverted commas, which is what sent a tail of one answer to a question
about quoting a phrase in support.

**Economics.** Marks printed against each question *and* stated for the section,
which disagree on one question — the paper says three and the question says four.
A table between two questions. And a margin number the student wrote wrongly.

    uv run python tooling/scripts/build_fresh_papers.py

Writes into data/fresh/, which is not tracked.
"""

# ruff: noqa: E501 - verbatim exam text; wrapping it would make it harder to
# compare a rendered page against what is written here.

from __future__ import annotations

import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_full_papers import MARGIN, PAGE_H, PAGE_W, Sheet, _wrap  # noqa: E402

OUT = ROOT / "data" / "fresh"


# ══════════════════════════════════════════════════════════════════════════
# History — lettered sub-parts, marks on the cover, a source extract
# ══════════════════════════════════════════════════════════════════════════

HISTORY_PAPER: list[tuple[str, str]] = [
    ("title", "St Andrew's High School"),
    ("sub", "Class 10  ·  History  ·  Half-Yearly Examination"),
    ("sub", "Time: 2 hours                                   Maximum Marks: 40"),
    ("rule", ""),
    ("instr", "Read the following carefully."),
    ("instr", "Answer all questions. Each question carries 4 marks."),
    ("instr", "Write your question number clearly in the left margin."),
    ("gap", ""),
    ("q", "Q.1  Explain two reasons why the League of Nations failed to prevent war in the 1930s."),
    ("gap", ""),
    ("q", "Q.2  Describe the part played by the railways in the growth of industrial towns."),
    ("gap", ""),
    ("q", "Q.3 (a)  What is meant by the term 'balance of power'?"),
    ("q", "Q.3 (b)  Give one example of it breaking down before 1914."),
    ("gap", ""),
    ("stem", "Read the source below and answer the question that follows."),
    ("quote", "\"We were told the war would be over by Christmas. By the second winter"),
    ("quote", "nobody said it any more, and the newspapers had stopped printing the"),
    ("quote", "casualty lists in full.\"        — a soldier's letter home, 1915"),
    ("gap", ""),
    ("q", "Q.4  What does the source suggest about how opinion at home changed during the war?"),
    ("gap", ""),
    ("q", "Q.5  Assess how far the Treaty of Versailles was responsible for later instability in Europe."),
    ("gap", ""),
    ("q", "Q.6  Explain why the invention of the printing press changed who could take part in public argument."),
]

HISTORY_SCRIPT: list[dict] = [
    dict(page=0, label="Q.1", lines=[
        "The League failed because the big powers did not all join it.",
        "America never joined at all so it had no army behind it.",
        "It also had no way to make a country obey. When Japan took",
        "Manchuria the League only complained and Japan left instead.",
    ], case="in order"),
    dict(page=0, label="Q.2", lines=[
        "Railways let raw materials come into the towns cheaply and",
        "let finished goods go out. So factories could be built away",
        "from rivers. People moved to the towns for the work and the",
        "towns grew fast around the stations.",
    ], case="in order"),
    dict(page=1, label="Q.4", lines=[
        "The source shows that people at home were hopeful at the start",
        "but they lost that hope. Saying nobody said it any more shows",
        "the mood changed. Not printing the lists in full suggests the",
        "papers were hiding how bad the losses were.",
    ], case="Q.3 skipped here and answered later; Q.5 never answered"),
    dict(page=1, label="Q.6", lines=[
        "Before printing, books were copied by hand so only the church",
        "and rich people had them. Printing made many copies cheaply so",
        "ordinary people could read arguments themselves and did not",
        "have to take the word of a priest.",
    ], case="in order"),
    dict(page=2, label="Q.3", lines=[
        "Balance of power means no single country in Europe is strong",
        "enough to dominate the others, so they keep each other in check.",
        "It broke down when Germany built a large navy and the countries",
        "formed into two alliance blocks instead of many.",
    ], case="OUT OF ORDER — both sub-parts answered as one run, on a later page"),
]


# ══════════════════════════════════════════════════════════════════════════
# Geography — a figure between a stem and its sub-parts, and a sketch question
# ══════════════════════════════════════════════════════════════════════════

GEOGRAPHY_PAPER: list[tuple[str, str]] = [
    ("title", "Riverside Secondary School"),
    ("sub", "Class 9  ·  Geography  ·  Term Test"),
    ("sub", "Time: 90 minutes                                Maximum Marks: 30"),
    ("rule", ""),
    ("instr", "All questions are compulsory."),
    ("gap", ""),
    ("q", "1.  Name the three main types of rainfall and state where each occurs.   [3]"),
    ("gap", ""),
    ("stem", "2.  Study the sketch of the river below and answer the parts that follow."),
    ("figure", ""),
    ("q", "2. (i)  Name the landform marked at position A.   [2]"),
    ("q", "2. (ii)  Explain how the feature at position B is formed.   [4]"),
    ("gap", ""),
    ("q", "3.  Draw a labelled sketch of a meander showing the fastest flow and the deposits.   [5]"),
    ("gap", ""),
    ("q", "4.  Give two reasons why people continue to settle on floodplains despite the risk.   [4]"),
    ("gap", ""),
    ("q", "5.  Distinguish between weathering and erosion, with one example of each.   [4]"),
]

GEOGRAPHY_SCRIPT: list[dict] = [
    dict(page=0, label="1", lines=[
        "The three types are relief rainfall, convectional rainfall and",
        "frontal rainfall. Relief happens where air is pushed up over",
        "hills near the coast. Convectional happens in hot places in the",
        "afternoon. Frontal happens where warm and cold air meet.",
    ], case="in order"),
    dict(page=0, label="2 (i)", lines=[
        "The landform at A is a waterfall.",
    ], case="a one-line answer to a two-mark part"),
    dict(page=1, label="2 (ii)", lines=[
        "At B the river is slower so it drops the material it was carrying.",
        "Over time this builds up as a flat area of new land at the mouth,",
        "which is a delta. It forms where the river meets the sea and the",
        "current cannot carry the load any further.",
    ], case="in order, on the next page"),
    dict(page=1, label="4", lines=[
        "Floodplains have very fertile soil from the silt left by floods so",
        "farming is good there. They are also flat which makes it easier and",
        "cheaper to build on, and rivers give water and transport.",
    ], case="question 3 wants a sketch and is left blank"),
    dict(page=2, label="5", lines=[
        "Weathering is when rock is broken down where it is, without being",
        "moved, for example water freezing in a crack and splitting it.",
        "Erosion is when the broken pieces are carried away by something,",
        "for example a river wearing away its bank and moving the material.",
    ], case="in order"),
]


# ══════════════════════════════════════════════════════════════════════════
# English — an optional section, and quotation-heavy answers
# ══════════════════════════════════════════════════════════════════════════

ENGLISH_PAPER: list[tuple[str, str]] = [
    ("title", "Holy Trinity School"),
    ("sub", "Class 10  ·  English Literature  ·  Annual Examination"),
    ("sub", "Time: 2 hours                                   Maximum Marks: 35"),
    ("rule", ""),
    ("instr", "Section A is compulsory. In Section B, answer any two of the three questions."),
    ("gap", ""),
    ("section", "SECTION A"),
    ("instr", "(Each question carries 5 marks.)"),
    ("q", "1.  How does the poet use the weather to reflect the speaker's state of mind?"),
    ("gap", ""),
    ("q", "2.  Describe the relationship between the two brothers as it changes across the story."),
    ("gap", ""),
    ("section", "SECTION B"),
    ("instr", "(Answer any two. Each question carries 5 marks.)"),
    ("q", "3.  Discuss the writer's attitude to the village, quoting one phrase in support."),
    ("gap", ""),
    ("q", "4.  What is the effect of telling the story through a child's eyes?"),
    ("gap", ""),
    ("q", "5.  Comment on the ending. Is it hopeful or bleak? Give reasons."),
]

ENGLISH_SCRIPT: list[dict] = [
    dict(page=0, label="1", lines=[
        "The poet keeps returning to the rain. At the start it is \"a thin",
        "grey drizzle\" which matches how flat the speaker feels. Later when",
        "he remembers his childhood the sun comes through, and at the end",
        "the storm breaks, which shows his feelings finally coming out.",
    ], case="quotation inside the answer"),
    dict(page=0, label="2", lines=[
        "At first the brothers are close and share everything. After the",
        "father dies the older one becomes distant and the writer says he",
        "\"spoke to me as if across a table of strangers\". By the end they",
        "have made peace but it is quieter than before, not the same.",
    ], case="quotation inside the answer"),
    dict(page=1, label="5", lines=[
        "I think the ending is hopeful but only just. The narrator does not",
        "get what he wanted and the house is still sold. But he chooses to",
        "walk back through the village instead of taking the road out,",
        "which suggests he has not finished with the place.",
    ], case="Section B answered out of order — 5 before 3"),
    dict(page=1, label="3", lines=[
        "The writer is fond of the village but sees it clearly. He calls it",
        "\"a place that had learned to expect very little\", which is critical,",
        "and yet he describes the market in warm detail. So the attitude is",
        "affectionate without pretending the poverty is not there.",
    ], case="the second of two answered in Section B; question 4 not required"),
]


# ══════════════════════════════════════════════════════════════════════════
# Economics — marks stated twice and disagreeing, a table, a wrong margin number
# ══════════════════════════════════════════════════════════════════════════

ECONOMICS_PAPER: list[tuple[str, str]] = [
    ("title", "Kendriya Vidyalaya"),
    ("sub", "Class 11  ·  Economics  ·  Unit Test"),
    ("sub", "Time: 1 hour                                    Maximum Marks: 20"),
    ("rule", ""),
    ("section", "SECTION A"),
    ("instr", "(Each question carries 3 marks.)"),
    ("q", "Q1.  Define opportunity cost and give one example from daily life."),
    ("gap", ""),
    ("q", "Q2.  Distinguish between a movement along a demand curve and a shift of it.   [4]"),
    ("gap", ""),
    ("stem", "Q3.  The table below shows the price and quantity demanded of wheat."),
    ("table", ""),
    ("q", "Q3.  Calculate the price elasticity of demand between the first and second rows."),
    ("gap", ""),
    ("q", "Q4.  State two reasons why a government may impose a tax on a good."),
    ("gap", ""),
    ("q", "Q5.  Explain why the supply curve normally slopes upward."),
]

ECONOMICS_SCRIPT: list[dict] = [
    dict(page=0, label="Q1", lines=[
        "Opportunity cost is the value of the next best thing you give up",
        "when you make a choice. For example if I spend Sunday studying",
        "then the opportunity cost is the cricket match I did not play.",
    ], case="in order"),
    dict(page=0, label="Q2", lines=[
        "A movement along the curve happens when only the price changes,",
        "so the quantity demanded changes but the curve stays where it is.",
        "A shift happens when something else changes, like income or the",
        "price of a substitute, and the whole curve moves left or right.",
    ], case="printed marks say 4, the section says 3 — the question should win"),
    dict(page=1, label="Q4", lines=[
        "PED = percentage change in quantity / percentage change in price",
        "Quantity falls from 100 to 80 so that is a fall of 20 percent.",
        "Price rises from 10 to 12 so that is a rise of 20 percent.",
        "So PED = 20 / 20 = 1, which is unit elastic.",
    ], case="MISLABELLED — the margin says Q4, the working answers Q3"),
    dict(page=1, label="Q5", lines=[
        "The supply curve slopes upward because a higher price means more",
        "profit for the producer, so it is worth making more. It also",
        "covers the higher cost of producing extra units, since firms use",
        "less efficient resources as they expand.",
    ], case="in order; question 4 is never answered"),
]


PAPERS = {
    "history": (HISTORY_PAPER, HISTORY_SCRIPT),
    "geography": (GEOGRAPHY_PAPER, GEOGRAPHY_SCRIPT),
    "english": (ENGLISH_PAPER, ENGLISH_SCRIPT),
    "economics": (ECONOMICS_PAPER, ECONOMICS_SCRIPT),
}


def build_paper(rows: list[tuple[str, str]], path: Path) -> None:
    """Render a printed question paper."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN

    for kind, text in rows:
        if y > PAGE_H - MARGIN - 70:
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
        elif kind == "section":
            y += 8
            page.insert_text(fitz.Point(MARGIN, y), text, fontsize=12, fontname="hebo")
            y += 18
        elif kind == "stem":
            for i, line in enumerate(_wrap(text, 92)):
                page.insert_text(fitz.Point(MARGIN + (0 if i == 0 else 14), y),
                                 line, fontsize=10, fontname="helv")
                y += 15
        elif kind == "quote":
            page.insert_text(fitz.Point(MARGIN + 26, y), text, fontsize=9.5, fontname="hebi")
            y += 15
        elif kind == "gap":
            y += 10
        elif kind == "figure":
            # A river sketch with its positions lettered, sitting between the stem
            # and the sub-parts it belongs to. The stray letters are the point.
            top = y
            page.draw_bezier(fitz.Point(MARGIN + 40, top + 10), fitz.Point(MARGIN + 120, top + 40),
                             fitz.Point(MARGIN + 180, top + 5), fitz.Point(MARGIN + 260, top + 45),
                             color=(0.35, 0.45, 0.7), width=1.4)
            page.draw_bezier(fitz.Point(MARGIN + 40, top + 26), fitz.Point(MARGIN + 120, top + 56),
                             fitz.Point(MARGIN + 180, top + 21), fitz.Point(MARGIN + 260, top + 61),
                             color=(0.35, 0.45, 0.7), width=1.4)
            page.insert_text(fitz.Point(MARGIN + 62, top + 6), "A", fontsize=9.5, fontname="helv")
            page.insert_text(fitz.Point(MARGIN + 246, top + 72), "B", fontsize=9.5, fontname="helv")
            page.insert_text(fitz.Point(MARGIN + 150, top + 78), "N", fontsize=9.5, fontname="helv")
            y = top + 96
        elif kind == "table":
            top = y
            rows_ = [("Price (Rs/kg)", "Quantity (kg)"), ("10", "100"), ("12", "80"), ("14", "65")]
            for r, (a, b) in enumerate(rows_):
                font = "hebo" if r == 0 else "helv"
                page.insert_text(fitz.Point(MARGIN + 30, top + 14 + r * 16), a, fontsize=9, fontname=font)
                page.insert_text(fitz.Point(MARGIN + 170, top + 14 + r * 16), b, fontsize=9, fontname=font)
            page.draw_rect(fitz.Rect(MARGIN + 22, top, MARGIN + 260, top + 14 + len(rows_) * 16),
                           color=(0.5, 0.5, 0.5), width=0.6)
            y = top + 22 + len(rows_) * 16
        else:  # a question
            for i, line in enumerate(_wrap(text, 92)):
                page.insert_text(fitz.Point(MARGIN + (0 if i == 0 else 22), y),
                                 line, fontsize=10, fontname="helv")
                y += 15

    doc.save(path)
    doc.close()


def build_script(entries: list[dict], path: Path, seed: int) -> int:
    """Render a handwritten answer sheet."""
    sheet = Sheet(seed=seed)
    current = -1

    for entry in entries:
        if entry["page"] != current:
            sheet.new_page()
            current = entry["page"]
            sheet.gap(0.4)

        if entry.get("label"):
            sheet.write(entry["label"], x=MARGIN - 34, size=12.5)
            sheet.y -= 26

        for line in entry["lines"]:
            sheet.write(line, x=MARGIN + 4)
        sheet.gap(0.8)

    sheet.doc.save(path)
    pages = sheet.doc.page_count
    sheet.doc.close()
    return pages


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (paper, script) in PAPERS.items():
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        build_paper(paper, folder / "paper.pdf")
        pages = build_script(script, folder / "script.pdf", seed=11 + len(name))
        questions = sum(1 for kind, _ in paper if kind == "q")
        answered = len(script)
        print(f"  {name:10}  {questions} questions  ·  {pages}-page script  ·  {answered} answers written")
        for entry in script:
            print(f"      {str(entry.get('label')):8} {entry['case']}")
    print(f"\n  written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
