"""Synthetic documents for tests.

Real papers are not in the repo, and student answer scripts should not be. These
builders produce PDFs shaped like the structures the design has to survive,
taken from official CBSE, CISCE, AQA and College Board papers:

  * continuous numbering across sections (ICSE: Section A = Q1-3, B = Q4-10)
  * three-level nesting as ``Q2 -> (i) -> (a)``
  * printed marks in brackets, and competency tags that are not questions
  * an optionality instruction, which makes a skipped question correct
  * a two-column layout, where naive top-to-bottom ordering interleaves columns

This is the seed of the fuller synthetic generator: once answers are rendered
with handwriting faces, the same code produces ground-truth boxes for free,
because it knows where it drew everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz

A4 = fitz.paper_rect("a4")
_BODY = "helv"
_BOLD = "hebo"


@dataclass
class DrawnLine:
    """A line of text plus where it was drawn, in normalized page space.

    The point of the generator: ground truth costs nothing when you are the one
    placing the ink.
    """

    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class PaperBuilder:
    """Lays out text top-to-bottom and records normalized boxes for each line."""

    def __init__(self, *, columns: int = 1, margin: float = 56.0) -> None:
        self.doc = fitz.open()
        self.columns = columns
        self.margin = margin
        self.drawn: list[DrawnLine] = []
        self._page = self.doc.new_page(width=A4.width, height=A4.height)
        self._column = 0
        self._y = margin

    @property
    def _column_width(self) -> float:
        usable = A4.width - 2 * self.margin
        gutter = 24.0 if self.columns > 1 else 0.0
        return (usable - gutter * (self.columns - 1)) / self.columns

    @property
    def _column_x(self) -> float:
        return self.margin + self._column * (self._column_width + 24.0)

    def _advance(self, height: float) -> None:
        self._y += height
        if self._y > A4.height - self.margin:
            if self._column + 1 < self.columns:
                self._column += 1
                self._y = self.margin
            else:
                self._page = self.doc.new_page(width=A4.width, height=A4.height)
                self._column = 0
                self._y = self.margin

    def text(
        self,
        content: str,
        *,
        size: float = 10.5,
        bold: bool = False,
        indent: float = 0.0,
    ) -> None:
        """Draw one line and record its normalized box."""
        self._advance(0)
        x = self._column_x + indent
        baseline = self._y + size
        self._page.insert_text(
            fitz.Point(x, baseline),
            content,
            fontsize=size,
            fontname=_BOLD if bold else _BODY,
        )
        width = fitz.get_text_length(content, fontname=_BOLD if bold else _BODY, fontsize=size)
        self.drawn.append(
            DrawnLine(
                text=content,
                page=self.doc.page_count - 1,
                x0=x / A4.width,
                y0=(baseline - size) / A4.height,
                x1=min(1.0, (x + width) / A4.width),
                y1=(baseline + size * 0.25) / A4.height,
            )
        )
        self._advance(size * 1.6)

    def gap(self, amount: float = 10.0) -> None:
        self._advance(amount)

    def page_break(self) -> None:
        """Start a new page explicitly.

        Used rather than piling on content until it happens to overflow, so a
        fixture that is meant to span pages provably does, and stays that way
        when its wording changes.
        """
        self._page = self.doc.new_page(width=A4.width, height=A4.height)
        self._column = 0
        self._y = self.margin

    def to_bytes(self) -> bytes:
        return self.doc.tobytes()

    def close(self) -> None:
        self.doc.close()


def question_paper(*, columns: int = 1) -> tuple[bytes, list[DrawnLine]]:
    """A paper exercising the structures that break naive extraction."""
    b = PaperBuilder(columns=columns)

    b.text("SCIENCE — UNIT TEST", size=14, bold=True)
    b.text("Time allowed: 1 hour     Maximum Marks: 40", size=9.5)
    b.text("The marks for questions are shown in brackets.", size=9.5)
    b.gap()

    b.text("SECTION A", size=12, bold=True)
    b.text("(Attempt all questions from this Section)", size=9.5)
    b.gap(6)

    b.text("1. Define refraction of light.  [2]")
    b.gap(4)
    b.text("2. Answer the following:")
    b.text("(i) State the laws of reflection.  [2]", indent=16)
    b.text("(a) Draw a labelled ray diagram.  [3]", indent=32)
    b.text("(b) Give a valid reason for your answer.  [1]", indent=32)
    b.text("(ii) What is the SI unit of power?  [1]", indent=16)
    b.gap(4)
    b.text("3. Assertion: Sound travels faster in water than in air.  [2]")
    b.text("Reason: Water is denser than air.", indent=16)
    b.gap()

    b.text("SECTION B", size=12, bold=True)
    b.text("(Attempt any two questions from this Section)", size=9.5)
    b.gap(6)

    b.text("4. Explain the working of an electric motor.  [5]")
    b.text("[Analysis & Evaluation]", size=8.5, indent=16)
    b.gap(4)
    b.text("5. (a) Balance the chemical equation given below.  [3]", indent=0)
    b.text("(b) Name the type of reaction.  [2]", indent=16)
    b.gap(4)
    b.text("6. Draw a diagram of the human digestive system.  [5]")
    b.gap(4)
    b.text("7. Calculate the resistance of the circuit shown.  [5]")

    # Section C sits on page 2. Multi-page is the normal case for this product,
    # not an edge case, and a single-page fixture would let page-boundary bugs
    # through untested.
    b.page_break()
    b.text("SCIENCE — UNIT TEST (page 2)", size=9.5)
    b.gap()
    b.text("SECTION C", size=12, bold=True)
    b.text("(Attempt any one question from this Section)", size=9.5)
    b.gap(6)

    b.text("8. Answer both parts:")
    b.text("(a) State Ohm's law and write its formula.  [2]", indent=16)
    b.text("(b) A resistor carries 2 A at 10 V. Find its resistance.  [3]", indent=16)
    b.gap(4)
    b.text("9. Describe an experiment to show that air has mass.  [5]")
    b.text("[Application & Analysis]", size=8.5, indent=16)
    b.gap(4)
    b.text("10. Distinguish between the following pairs:")
    b.text("(i) Speed and velocity  [2]", indent=16)
    b.text("(ii) Mass and weight  [2]", indent=16)
    b.text("(iii) Heat and temperature  [1]", indent=16)
    b.gap(4)
    b.text("11. (a) What is meant by the term 'echo'?  [2]")
    b.text("(b) State two conditions necessary to hear an echo.  [3]", indent=16)

    data = b.to_bytes()
    drawn = list(b.drawn)
    b.close()
    return data, drawn


def answer_sheet_with_text() -> tuple[bytes, list[DrawnLine]]:
    """A stand-in answer sheet that carries a text layer.

    Not representative of a real handwritten script — it exists so the pipeline
    can be exercised end-to-end before a handwriting engine is configured.
    Answers are deliberately out of order, one question is skipped, and one
    block matches no question at all, so the structures the mapper has to handle
    are present from the start.
    """
    b = PaperBuilder()
    b.text("Name: Suyash        Class: 6C", size=10)
    b.gap()
    b.text("2 (i) Angle of incidence equals angle of reflection.")
    b.text("Both rays lie in the same plane as the normal.")
    b.gap(6)
    b.text("1. Refraction is the bending of light when it")
    b.text("passes from one medium into another.")
    b.gap(6)
    b.text("4. The motor converts electrical energy into")
    b.text("rotational motion using a current-carrying coil")
    b.text("placed in a magnetic field.")
    b.gap(6)
    b.text("Rough work: 12 x 4 = 48, then divide by 6")
    b.gap(6)

    # Deliberately pushed onto a second page and marked as continuing, so the
    # multi-page span path and the continuation-marker signal are both
    # exercisable from the start.
    b.text("8 (a) Ohm's law states that current is directly")
    b.text("proportional to the potential difference across")
    b.text("a conductor, provided temperature is constant.")
    b.text("V = I R")
    b.text("cont. on next page")

    b.page_break()
    b.text("(continued) 8 (b) R = V / I = 10 / 2 = 5 ohm")
    b.text("Therefore the resistance is 5 ohm.")

    data = b.to_bytes()
    drawn = list(b.drawn)
    b.close()
    return data, drawn


def image_with_known_size(width: int, height: int) -> bytes:
    """A PNG of exactly the requested pixel dimensions.

    Exists so that native-resolution detection can be tested against a known
    ground truth rather than against another call to the same library.
    """
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height))
    pixmap.set_rect(pixmap.irect, (255, 255, 255))
    return pixmap.tobytes("png")


def single_page_image() -> bytes:
    """A PNG, to check that photographed uploads are accepted."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text(fitz.Point(40, 80), "Q1. photographed page", fontsize=14, fontname=_BODY)
    pixmap = page.get_pixmap(dpi=96, alpha=False)
    data = pixmap.tobytes("png")
    doc.close()
    return data


def pdf_with_hidden_text(visible: str, hidden: str) -> bytes:
    """A PDF whose text layer holds content positioned off the page.

    The mechanism behind hidden-prompt injection, and a plain correctness hazard
    too: text that no human can see would otherwise be extracted as though it
    were part of a question.
    """
    doc = fitz.open()
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_text(fitz.Point(56, 100), visible, fontsize=11, fontname=_BODY)
    # Placed beyond the page rectangle: present in the text layer, absent from
    # anything rendered.
    page.insert_text(fitz.Point(56, A4.height + 400), hidden, fontsize=11, fontname=_BODY)
    data = doc.tobytes()
    doc.close()
    return data
