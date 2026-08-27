"""Build a question paper that the real answer sheets actually answer.

The handwritten scripts in ``data/handwriting`` are from a B.Tech programming lab
assessment, and reading them shows exactly two problems:

  * the element of a sorted array closest to a given target
  * the length of the longest run of 1s in an array of 0s and 1s, with the input
    rejected if it contains anything else

Answers are in C on some sheets and Python on others, and the sheets are headed
"Set 1" and "Set 3", so the original paper came in variants.

Why this script exists: every earlier end-to-end run paired these sheets with a
school science paper. Nothing was wrong with the pipeline, but nothing about the
result meant anything either — a mapping between unrelated documents is arbitrary
whatever it does, and it looked like success because every question got an answer.
Testing a mapper needs a paper the answers genuinely belong to.

The paper's structure is chosen so that the awkward cases arise honestly rather
than being simulated:

  * Q1 and Q3 are each answered by a whole page of code — the ordinary case.
  * Q2 asks for a time complexity, which no student wrote. A truthful
    "not answered", with no writing anywhere that could be mistaken for it.
  * Q2(ii) asks for a dry run. One sheet has a bare list of numbers in the margin
    that is one, and others have nothing — so the same question is answered on one
    script and absent on another.
  * Q3(a) asks for input validation, which one student did and another did not.
  * Q3(a) and (b) are both satisfied inside the single block of code answering Q3,
    which is the merged sub-part case.
  * Section C is a choice nobody took, so it must read as "not required" rather
    than as three omissions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz

A4 = fitz.paper_rect("a4")
MARGIN = 56.0
BODY = "helv"
BOLD = "hebo"


class Paper:
    """Lays out lines top to bottom, breaking pages as needed."""

    def __init__(self) -> None:
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=A4.width, height=A4.height)
        self.y = MARGIN

    def line(
        self,
        text: str,
        *,
        size: float = 11.0,
        indent: float = 0.0,
        bold: bool = False,
        centre: bool = False,
    ) -> None:
        if self.y > A4.height - MARGIN - 40:
            self.page = self.doc.new_page(width=A4.width, height=A4.height)
            self.y = MARGIN

        font = BOLD if bold else BODY
        width = fitz.get_text_length(text, fontname=font, fontsize=size)
        x = (A4.width - width) / 2 if centre else MARGIN + indent
        self.page.insert_text(fitz.Point(x, self.y + size), text, fontsize=size, fontname=font)
        self.y += size * 1.5

    def wrapped(self, text: str, *, indent: float = 0.0, size: float = 11.0) -> None:
        """One question, wrapped with its continuation lines aligned under it."""
        limit = 88
        words, current = text.split(), ""
        first = True
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > limit and current:
                self.line(current, indent=indent if first else indent + 14, size=size)
                current, first = word, False
            else:
                current = candidate
        if current:
            self.line(current, indent=indent if first else indent + 14, size=size)

    def gap(self, amount: float = 10.0) -> None:
        self.y += amount

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        self.doc.close()


def build() -> Paper:
    paper = Paper()

    paper.line("NATIONAL INSTITUTE OF TECHNOLOGY", size=13, bold=True, centre=True)
    paper.line("Department of Computer Science and Engineering", size=10.5, centre=True)
    paper.line("B.Tech CSE, Semester 8", size=10.5, centre=True)
    paper.gap(6)
    paper.line("PROGRAMMING LAB — INTERNAL ASSESSMENT", size=12.5, bold=True, centre=True)
    paper.line("SET 1", size=11, bold=True, centre=True)
    paper.gap(8)
    paper.line("Time allowed: 1 hour                                    Maximum Marks: 25", size=10)
    paper.line("Answer in either C or Python. Write each program in full.", size=10)
    paper.line("The marks for questions are shown in brackets.", size=10)
    paper.gap(14)

    paper.line("SECTION A", size=12, bold=True)
    paper.line("(Attempt all questions from this Section)", size=10)
    paper.gap(8)

    paper.wrapped(
        "1. Write a program that reads a sorted array of n integers and a target value, "
        "and prints the element of the array closest to the target.  [8]"
    )
    paper.gap(6)
    paper.wrapped("2. Answer the following about the program you wrote for question 1:")
    paper.wrapped("(i) State the time complexity of your solution.  [2]", indent=18)
    paper.wrapped(
        "(ii) Show a dry run for the array 3, 4, 5, 6, 9 with the target 5.2.  [3]", indent=18
    )
    paper.gap(16)

    paper.line("SECTION B", size=12, bold=True)
    paper.line("(Attempt all questions from this Section)", size=10)
    paper.gap(8)

    paper.wrapped(
        "3. Write a program that reads an array of 0s and 1s and prints the length of the "
        "longest run of 1s in that array."
    )
    paper.wrapped(
        "(a) Reject the input and print a message if any element is not 0 or 1.  [3]", indent=18
    )
    paper.wrapped("(b) Print the maximum run length that was found.  [4]", indent=18)
    paper.gap(16)

    paper.line("SECTION C", size=12, bold=True)
    paper.line("(Attempt any one question from this Section)", size=10)
    paper.gap(8)

    paper.wrapped(
        "4. Rewrite your answer to question 3 so that it makes only one pass over the "
        "array.  [5]"
    )
    paper.gap(6)
    paper.wrapped(
        "5. Explain why a sorted array allows question 1 to be solved without examining "
        "every element.  [5]"
    )
    return paper


def build_sheet(images: list[Path], out: Path) -> None:
    """Combine handwritten pages into one answer sheet, one image per page.

    Each source image is a photograph of one page of a student's script. Pages are
    laid out at their own aspect ratio rather than stretched to A4, because a
    distorted page would change what the recognizer sees and every coordinate
    derived from it.
    """
    doc = fitz.open()
    for path in images:
        pixmap = fitz.Pixmap(str(path))
        page = doc.new_page(width=pixmap.width, height=pixmap.height)
        page.insert_image(page.rect, filename=str(path))
        del pixmap
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    paper_path = args.repo / "samples" / "programming_lab_set1.pdf"
    build().save(paper_path)
    print(f"question paper  {paper_path.relative_to(args.repo)}")

    handwriting = args.repo / "data" / "handwriting"
    sheets = args.repo / "data" / "samples"

    # Deliberately different shapes, so a manual pass covers more than one case.
    #
    #   answered_in_order  — Q1 then Q3, the ordinary case
    #   answered_reversed  — the same two answers written in the opposite order
    #   one_question_only  — Q3 attempted and Q1 not, so Q1 is genuinely blank
    plans = {
        "student_a_in_order": ["Closest_value2.JPEG", "Consecutive11.JPEG"],
        "student_b_reversed": ["Consecutive13.JPEG", "Closest10.JPEG"],
        "student_c_partial": ["Consecutive12.JPEG"],
    }

    for name, files in plans.items():
        paths = [handwriting / f for f in files]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            print(f"  skipped {name}: missing {', '.join(missing)}")
            continue
        out = sheets / f"{name}.pdf"
        build_sheet(paths, out)
        print(f"answer sheet    {out.relative_to(args.repo)}  ({len(paths)} page(s))")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
