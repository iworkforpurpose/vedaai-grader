"""Which printed question-label styles this extractor recognises, and which it drops.

Written after a real Class 9 mathematics paper produced one question out of nine.
The paper was not unusual — it used ``Q1 (5 Marks)`` as a heading on its own line,
and a second section labelled ``T1``..``T5``. Both were dropped, and because the
golden set only ever contains label styles the parser already handles, every
accuracy figure the project quotes was blind to it.

So this enumerates label styles taken from real papers rather than from the
generator, and reports each as recognised or dropped. It is deliberately not a
test: a failing row here is a known gap, and the list of gaps is the fix plan.

    uv run python tooling/scripts/label_matrix.py
"""

# ruff: noqa: E501 - the long lines are verbatim exam text and label strings;
# wrapping them would make it harder to compare against a real paper.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps/api/src"))

from vedaai_contracts import BBox, DocumentKind, Line, LineRole, OcrEngine  # noqa: E402

from grader.questions import furniture  # noqa: E402
from grader.questions.numbering import detect_section_prefixes  # noqa: E402

#: (label line, following line, board or source, what a reader would call it)
#:
#: The second element matters: a heading style puts the question text on the *next*
#: line, and whether the extractor copes with that is the whole point of the case.
CASES: list[tuple[str, str, str, str]] = [
    # ── styles the generator produces, and the golden set therefore covers ──
    ("1. Define refraction.", "", "generic", "numeric, period, inline"),
    ("2) State two uses of a lever.", "", "generic", "numeric, paren, inline"),
    ("11 (a) Describe the process.", "", "CBSE", "numbered sub-part, inline"),
    ("(i) Name the food source.", "", "ICSE", "roman sub-part, inline"),
    ("Q3. Explain the water cycle.", "", "CBSE", "Q-prefix, period, inline"),
    ("5. Explain the reaction. [3]", "", "generic", "inline with marks"),

    # ── the styles the user's real paper used ──
    ("Q1 (5 Marks)", "Two taps A and B fill a tank.", "SmartLearners.ai", "Q-prefix heading, marks only"),
    ("T1 (5 Marks)", "AB is a line segment and P is its mid-point.", "SmartLearners.ai", "letter-prefix heading"),
    ("T2 (5 Marks)", "ABC and DBC are isosceles triangles on the same base.", "SmartLearners.ai", "letter-prefix heading"),
    ("T3 (5 Marks)", "Bisectors of angle B and angle C meet at O.", "SmartLearners.ai", "letter-prefix heading"),

    # ── other real-world heading styles ──
    ("Question 4", "Describe the carbon cycle.", "AQA", "spelled-out heading"),
    ("Q7", "State Newton's second law.", "generic", "bare Q heading"),
    ("4.", "Describe the carbon cycle.", "generic", "bare numeric heading"),
    ("3 (a)", "Define momentum.", "CBSE", "sub-part heading"),
    ("Q.8", "Explain osmosis.", "CBSE", "Q-dot prefix"),
    ("SECTION B", "Attempt any four questions.", "CBSE", "!section heading"),
    ("A1. Discuss the causes of the famine.", "", "Edexcel", "letter+number inline"),
    ("A2. Explain two consequences.", "", "Edexcel", "letter+number inline"),
    ("A3. Assess the response of government.", "", "Edexcel", "letter+number inline"),
    ("13 [4 marks]", "Compare the two accounts.", "AQA", "heading, bracketed marks"),
    ("(b) Give one example.", "", "ICSE", "letter sub-part, inline"),
    # Expected NOT to be a question. "Answer any two" is rubric, and a roman
    # numeral in front of it does not stop it being rubric — this row was written
    # as a question by mistake and the extractor was right to disagree.
    ("II. Answer any two.", "", "ICSE", "!rubric behind a roman numeral"),
    ("Q 6 (i)", "Name the reagent used.", "ICSE", "spaced Q with sub-part heading"),
    ("5(n)", "State the principle.", "generic", "malformed OCR of 5(a)"),
]


#: Section letters this paper is observed to use, learned from the whole table
#: exactly as the extractor learns them from a whole document. A single `T1` in
#: isolation is not evidence of anything; `T1`, `T2`, `T3` together are.
PREFIXES = detect_section_prefixes([case[0] for case in CASES])


def _as_line(text: str) -> Line:
    return Line(
        line_id="qp:0001",
        kind=DocumentKind.QUESTION_PAPER,
        page=0,
        box=BBox(x0=0.09, y0=0.3, x1=0.9, y1=0.32),
        text=text,
        confidence=1.0,
        engine=OcrEngine.PDF_TEXT_LAYER,
    )


def verdict(label_line: str, next_line: str) -> tuple[bool, str]:
    """Whether the extractor would treat this as the start of a question.

    Asks the classifier the pipeline actually uses, not the label parser beneath
    it. An earlier version of this script called `parse_label` directly and so
    could not see the heading layout at all — it reported eleven styles broken
    after seven of them had been fixed, because the fix lives one level up.
    """
    role = furniture.classify(
        _as_line(label_line),
        repeated=set(),
        previous_role=None,
        started=True,
        following=(next_line,) if next_line else (),
        prefixes=PREFIXES,
    )
    return role is LineRole.QUESTION_START, f"role={role.value}"


def main() -> int:
    ok = bad = 0
    width = max(len(c[0]) for c in CASES) + 2

    print(f"\n{'label line':{width}} {'source':18} {'':4} detail")
    print("─" * (width + 64))
    for label_line, next_line, source, _description in CASES:
        recognised, detail = verdict(label_line, next_line)
        # A description starting "!" marks a row that must be rejected.
        wanted = not _description.startswith("!") and "SECTION" not in label_line
        correct = recognised == wanted
        mark = " ok " if correct else "MISS"
        ok, bad = (ok + 1, bad) if correct else (ok, bad + 1)
        print(f"{label_line!r:{width}} {source:18} {mark} {detail}")

    print("─" * (width + 64))
    print(f"{ok} handled, {bad} dropped, of {len(CASES)} styles\n")

    if bad:
        print("Dropped styles, which is the fix list:")
        for label_line, next_line, source, description in CASES:
            recognised, _ = verdict(label_line, next_line)
            if recognised != (not description.startswith("!") and "SECTION" not in label_line):
                print(f"  · {description:34} e.g. {label_line!r}  [{source}]")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
