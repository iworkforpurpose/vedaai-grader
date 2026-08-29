"""Assembling a question paper from ordered, classified lines.

The structure being recovered is a tree, but a paper only ever shows it as
indentation and punctuation. Two rules reconstruct it.

**A label with several tokens is an absolute path.** ``2 (i) (a)`` names its own
position, so nothing needs inferring.

**A label with one non-numeric token is relative.** ``(a)`` on its own belongs
under whatever question precedes it, and how far under is decided by indentation
— which is the only signal a paper actually gives.

Indentation is measured against the other question labels on the same paper
rather than against any absolute figure, because margin width varies by board and
by rendering density.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vedaai_contracts import (
    Line,
    LineIndex,
    LineRole,
    PageBox,
    Question,
    QuestionPaper,
    Requirement,
    Section,
)

from . import furniture, optionality
from .numbering import (
    ParsedLabel,
    canonical_qid,
    detect_section_prefixes,
    extract_marks,
    parse_label,
)
from .validate import find_gaps

#: Two labels within this horizontal distance are at the same indent level.
#: Generous, because a proportional font makes ``1.`` and ``11.`` start at the
#: same place but end differently, and OCR adds a pixel or two of noise.
_INDENT_TOLERANCE = 0.012


@dataclass(frozen=True)
class _Ancestor:
    """An open question that a following relative label might attach to."""

    level: int
    """Indentation level of its label."""

    path: tuple[str, ...]
    """Its absolute token path."""

    self_describing: bool
    """Whether its printed label carried more than one token, e.g. ``11 (a)``."""


@dataclass
class _Building:
    """A question being accumulated as its lines are walked."""

    qid: str
    label: ParsedLabel
    path: tuple[str, ...]
    section: str | None
    indent: float
    print_order: int
    text_parts: list[str] = field(default_factory=list)
    line_ids: list[str] = field(default_factory=list)
    boxes: list[PageBox] = field(default_factory=list)
    marks: int | None = None

    def finish(self) -> Question:
        return Question(
            qid=self.qid,
            label_raw=self.label.raw,
            text=" ".join(part for part in self.text_parts if part).strip(),
            path=list(self.path),
            print_order=self.print_order,
            section_id=self.section,
            marks=self.marks,
            line_ids=list(self.line_ids),
            geometry=_union_per_page(self.boxes),
        )


def _union_per_page(boxes: list[PageBox]) -> list[PageBox]:
    from vedaai_contracts import BBox

    per_page: dict[int, list[BBox]] = {}
    for pb in boxes:
        per_page.setdefault(pb.page, []).append(pb.box)
    return [
        PageBox(page=page, box=BBox.union_all(bs)) for page, bs in sorted(per_page.items())
    ]


def _indent_levels(values: list[float]) -> list[float]:
    """Distinct indentation positions on this paper, left to right."""
    levels: list[float] = []
    for value in sorted(values):
        if not levels or value - levels[-1] > _INDENT_TOLERANCE:
            levels.append(value)
    return levels


def _level_of(value: float, levels: list[float]) -> int:
    for i, level in enumerate(levels):
        if abs(value - level) <= _INDENT_TOLERANCE:
            return i
    return len(levels)


def extract(index: LineIndex) -> QuestionPaper:
    """Recover the paper's questions, sections and choice rules."""
    # The index is already in reading order — that is where line IDs are
    # assigned, and spans depend on it. Ordering again here would create a second
    # source of truth for the same question.
    ordered = list(index.lines)
    roles = furniture.classify_all(ordered)

    label_indents = [
        line.box.x0 for line in ordered if roles[line.line_id] is LineRole.QUESTION_START
    ]
    levels = _indent_levels(label_indents)

    questions: list[Question] = []
    sections: list[Section] = []
    instructions: list[str] = []
    section_instructions: dict[str, list[str]] = {}

    current_section: str | None = None
    # Learned from the whole paper, because one line cannot tell a section
    # letter from a coincidence.
    prefixes = detect_section_prefixes([ln.text for ln in index.lines])
    building: _Building | None = None
    # One entry per open ancestor: (indent level, absolute path, was the label
    # multi-token). The last flag is what distinguishes a sibling from a child —
    # see _resolve_path.
    stack: list[_Ancestor] = []
    print_order = 0

    for line in ordered:
        role = roles[line.line_id]

        if role is LineRole.SECTION_HEADER:
            if building is not None:
                questions.append(building.finish())
                building = None
            label = furniture.section_label(line.text)
            if label is not None:
                current_section = label
                if all(s.section_id != label for s in sections):
                    sections.append(Section(section_id=label, label_raw=line.text.strip()))
                # Numbering may restart in a new section, so ancestry from the
                # previous one cannot apply here.
                stack = []
            continue

        if role is LineRole.INSTRUCTION:
            instructions.append(line.text)
            if current_section is not None:
                section_instructions.setdefault(current_section, []).append(line.text)
            continue

        if role is LineRole.MARKS:
            if building is not None and building.marks is None:
                _text, marks = extract_marks(line.text)
                building.marks = marks
            continue

        if role is LineRole.QUESTION_START:
            if building is not None:
                questions.append(building.finish())

            parsed = parse_label(line.text, prefixes=prefixes)
            if parsed is None:  # pragma: no cover - classify already checked
                continue

            # A label carrying its own section letter, as `T1` does. Without this
            # the letter was dropped and `T1` became question `1`, colliding with
            # `Q1`: a paper with four Q questions and five T questions extracted
            # four questions and five duplicates of them.
            if parsed.section_hint and parsed.section_hint != current_section:
                current_section = parsed.section_hint
                if all(s.section_id != current_section for s in sections):
                    sections.append(
                        Section(section_id=current_section, label_raw=current_section)
                    )
                # Numbering restarts with the section, so nothing above it can be
                # an ancestor of what follows.
                stack = []

            level = _level_of(line.box.x0, levels)
            path = _resolve_path(parsed, level, stack)

            # Drop ancestors at or deeper than this level, then record this one so
            # a following relative label can attach beneath it.
            stack = [entry for entry in stack if entry.level < level]
            stack.append(
                _Ancestor(level=level, path=path, self_describing=len(parsed.tokens) > 1)
            )

            body, marks = extract_marks(parsed.remainder)
            building = _Building(
                qid=canonical_qid(current_section, path),
                label=parsed,
                path=path,
                section=current_section,
                indent=line.box.x0,
                print_order=print_order,
                text_parts=[body],
                line_ids=[line.line_id],
                boxes=[PageBox(page=line.page, box=line.box)],
                marks=marks,
            )
            print_order += 1
            continue

        if role is LineRole.QUESTION_CONTINUATION and building is not None:
            body, marks = extract_marks(line.text)
            building.text_parts.append(body)
            building.line_ids.append(line.line_id)
            building.boxes.append(PageBox(page=line.page, box=line.box))
            if marks is not None and building.marks is None:
                building.marks = marks

    if building is not None:
        questions.append(building.finish())

    sections = _apply_requirements(sections, instructions, section_instructions)
    questions = mark_stems(questions)
    paper = QuestionPaper(
        questions=questions,
        sections=sections,
        gaps=find_gaps(questions),
        total_marks=_sum_marks(questions),
    )
    return paper


#: A heading's text points forward at the parts beneath it. Matched together with
#: a trailing colon, because that pairing is what separates a heading from a task
#: whose marks happen to be itemised below it.
#:
#: Structure alone is not enough, and a real paper showed why. "3. Write a program
#: that reads an array of 0s and 1s and prints the length of the longest run of
#: 1s." carries no marks — they sit on its (a) and (b) — but it is the task, and
#: calling it a heading would leave the question the student actually answered out
#: of the candidate list. "2. Answer the following about the program you wrote for
#: question 1:" is a heading, and the difference between them is not structural: it
#: is that one is self-contained and the other is meaningless without its parts.
_POINTS_AT_ITS_PARTS = re.compile(
    r"\b(?:the following|both parts?|all parts?|each part|the parts? below|"
    r"these questions?|the questions? below)\b",
    re.IGNORECASE,
)


def reads_as_a_heading(text: str) -> bool:
    """Whether a question's text introduces other questions rather than asking one.

    Both conditions are needed. The colon alone would catch "Balance the following
    equation:", which is a question. The phrase alone would catch "Answer any two
    of the following questions", which is rubric handled elsewhere and never a
    question in the first place.
    """
    stripped = text.strip()
    return stripped.endswith(":") and _POINTS_AT_ITS_PARTS.search(stripped) is not None


def mark_stems(questions: list[Question]) -> list[Question]:
    """Flag the questions that introduce others rather than asking anything.

    "2. Answer the following:" followed by (i), (a), (b), (ii) is a heading. It is
    kept as an extracted question, because the requirement is to preserve the
    paper's numbering and the teacher expects to see it — but it is not answerable,
    and treating it as though it were causes two distinct errors: it sits in the
    matching candidate list where it can absorb the answer to its own sub-part,
    and it is reported unanswered when nothing was ever asked.

    Three conditions, all necessary: another question's path extends this one, it
    printed no marks of its own, and its text reads as a pointer to those parts
    rather than as a question. Dropping the last of those was a mistake — a task
    statement whose marks are itemised on its sub-parts looks structurally
    identical to a heading, and treating it as one removes the question the student
    actually answered.
    """
    paths = {tuple(q.path) for q in questions}
    return [
        q.model_copy(update={"is_stem": True})
        if q.marks is None
        and reads_as_a_heading(q.text)
        and any(len(p) > len(q.path) and p[: len(q.path)] == tuple(q.path) for p in paths)
        else q
        for q in questions
    ]


def _resolve_path(
    parsed: ParsedLabel,
    level: int,
    stack: list[_Ancestor],
) -> tuple[str, ...]:
    """Work out a question's absolute token path.

    A lone non-numeric label such as ``(b)`` is relative, and the hard part is
    deciding whether it is a *child* of the preceding question or a *sibling* of
    its last part. Indentation cannot answer it, because both arrive indented
    further than the label above them:

        11 (a) What is meant by an echo?          <- indent 0
            (b) State two conditions.             <- indent 1, sibling of (a)

        2. Answer the following:                  <- indent 0
            (i) State the laws.                   <- indent 1, child of 2
                (a) Draw a diagram.               <- indent 2, child of (i)

    What separates them is whether the ancestor's own label stated its full path.
    ``11 (a)`` named both its parts, so the parent number was printed once and
    ``(b)`` continues that series — a sibling. ``(i)`` named only itself, so a
    deeper label below it is a new level — a child.

    Getting this wrong produced ``11/a/b`` for the first case: a question nested
    under its own sibling, which then has no correct place in the hierarchy.
    """
    if len(parsed.tokens) > 1 or parsed.is_top_level_candidate:
        # Either the label states its own full path, or it is a numbered
        # top-level question and therefore starts a new one.
        return parsed.tokens

    ancestors = [entry for entry in stack if entry.level < level]
    if not ancestors:
        return parsed.tokens

    nearest = ancestors[-1]
    if nearest.self_describing and len(nearest.path) > 1:
        # Sibling: replace the ancestor's final token rather than nesting under it.
        return nearest.path[:-1] + parsed.tokens
    return nearest.path + parsed.tokens


def _apply_requirements(
    sections: list[Section],
    instructions: list[str],
    section_instructions: dict[str, list[str]],
) -> list[Section]:
    """Attach choice rules to the sections they govern.

    A rubric naming a section wins over one that does not, because the naming is
    what makes it specific — "any four questions from Section B" on a cover page
    governs B, not whichever section the sentence was printed above.
    """
    global_rules = optionality.parse_all(instructions)

    out: list[Section] = []
    for section in sections:
        requirement: Requirement | None = global_rules.get(section.section_id)
        if requirement is None:
            local = optionality.parse_all(section_instructions.get(section.section_id, []))
            requirement = local.get(None) or local.get(section.section_id)
        out.append(
            section.model_copy(update={"requirement": requirement or Requirement()})
        )
    return out


def _sum_marks(questions: list[Question]) -> int | None:
    values = [q.marks for q in questions if q.marks is not None]
    return sum(values) if values else None


def question_lines(index: LineIndex, question: Question) -> list[Line]:
    """The lines a question was built from."""
    by_id = index.by_id()
    return [by_id[line_id] for line_id in question.line_ids if line_id in by_id]
