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
    per_question_marks,
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
            # Recorded under the open section, or under None while still on the
            # cover page. A cover-page rule governs the whole paper and a section's
            # own overrides it, which needs the two kept apart.
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
                building = None

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

            # A number is a name, and seeing it twice in one section means the
            # paper is still talking about the same question.
            #
            # An economics paper puts a table inside question 3 and repeats "Q3."
            # underneath it, so a reader picking up below the table knows where
            # they are. Both halves parsed as questions, both took the id A/3, and
            # the answer could not be placed on a question that existed twice.
            # Reopening the first is what the repetition means.
            qid = canonical_qid(current_section, path)
            existing = next((i for i, q in enumerate(questions) if q.qid == qid), None)
            if existing is not None:
                reopened = questions.pop(existing)
                building = _Building(
                    qid=reopened.qid,
                    label=parsed,
                    path=list(reopened.path),
                    section=current_section,
                    indent=reopened.geometry[0].box.x0 if reopened.geometry else line.box.x0,
                    print_order=reopened.print_order,
                    text_parts=[reopened.text, body],
                    line_ids=list(reopened.line_ids) + [line.line_id],
                    boxes=list(reopened.geometry) + [PageBox(page=line.page, box=line.box)],
                    marks=reopened.marks if reopened.marks is not None else marks,
                )
                continue

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
    # After mark_stems, so a stem — which introduces its parts and is worth
    # nothing itself — is not handed a share of its own question's marks.
    questions = _apply_section_marks(questions, section_instructions)
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
    r"\b(?:the following"
    r"|both parts?|all parts?|each part"
    r"|the (?:parts?|questions?)\s+(?:below|that follow|which follow|given below)"
    r"|these questions?"
    r")\b",
    re.IGNORECASE,
)

#: A heading tells the student to answer what comes next. A task merely mentions
#: it. This is what stands in for the colon when a paper does not use one.
_INVITES_ANSWERS = re.compile(r"\b(?:answer|attempt|respond to)\b", re.IGNORECASE)


def reads_as_a_heading(text: str) -> bool:
    """Whether a question's text introduces other questions rather than asking one.

    Pointing at the parts is necessary but never sufficient. "Balance the following
    equation" points at something and is still a question, so a second signal has
    to say that the parts are what gets answered.

    A colon is one such signal. Requiring it was the whole rule, and a geography
    paper ended the sentence instead: "Study the sketch of the river below and
    answer the parts that follow." stayed an answerable question, sat in the
    candidate list beside its own (i) and (ii), and took the answer to (ii) — which
    was then reported uncertain on a question the student had answered in full.

    The invitation is the other signal, and the more direct one: a heading asks the
    student to *answer* what follows. "Balance the following equation." does not,
    and stays a question whichever mark ends it.
    """
    stripped = text.strip()
    if _POINTS_AT_ITS_PARTS.search(stripped) is None:
        return False
    return stripped.endswith(":") or _INVITES_ANSWERS.search(stripped) is not None


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
    # Keyed by section as well as path. A path is not an identity on a paper whose
    # numbering restarts per section: both sections hold a ("2",), so keyed on the
    # path alone section B's "2 (a)" extends section A's "2." and makes it a
    # heading over parts that are not its own. The cost of that is not cosmetic —
    # a stem is NOT_REQUIRED, so a question the student was actually asked drops
    # out of absence reporting altogether.
    paths = {(_section_of(q), tuple(q.path)) for q in questions}

    def heads_some_parts(question: Question) -> bool:
        section, path = _section_of(question), tuple(question.path)
        return any(
            other_section == section and len(p) > len(path) and p[: len(path)] == path
            for other_section, p in paths
        )

    return [
        q.model_copy(update={"is_stem": True})
        if q.marks is None and reads_as_a_heading(q.text) and heads_some_parts(q)
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


def _section_of(question: Question) -> str | None:
    """The section a question sits in, read back off its canonical id."""
    if question.section_id is not None:
        return question.section_id
    suffix = "/".join(question.path)
    return question.qid[: -len(suffix)].rstrip("/") or None


def _apply_section_marks(
    questions: list[Question],
    section_instructions: dict[str | None, list[str]],
) -> list[Question]:
    """Spread a marks allocation stated once over the questions it governs.

    Most papers state marks for a group rather than beside every question:
    "(Each question carries 1 mark)" under SECTION A, and nothing at all against
    questions 1 to 6. Read only beside the question, that leaves six questions
    with no denominator — graded out of nothing, and shown to a teacher as "0/0".

    Three rules, in the order they matter.

    **A printed mark wins.** The section states a default; the question states a
    fact, and a fact about this question beats a rule about its neighbours.

    **The allocation is the whole question's, not each sub-part's.** "Each
    question carries 4 marks" beside 11(a) and 11(b) means four between them.
    Giving both four would double what the paper is out of. So sub-parts share,
    and a sibling that printed its own marks takes them off the top first — with
    5 for the question and "[2]" against part (i), part (ii) is worth 3.

    **An uneven share is declined.** Five marks across three parts could be
    2/2/1 or 1/2/2, and a denominator a teacher cannot check is worse than none.
    """
    rates: dict[str | None, int] = {}
    for section_id, texts in section_instructions.items():
        for text in texts:
            marks = per_question_marks(text)
            if marks is not None:
                rates.setdefault(section_id, marks)
                break
    if not rates:
        return questions

    groups: dict[tuple[str | None, str], list[int]] = {}
    for position, question in enumerate(questions):
        if question.is_stem:
            continue
        groups.setdefault((_section_of(question), question.path[0]), []).append(position)

    updated = list(questions)
    for (section_id, _root), members in groups.items():
        rate = rates.get(section_id)
        if rate is None:
            rate = rates.get(None)
        if rate is None:
            continue

        unknown = [i for i in members if updated[i].marks is None]
        if not unknown:
            continue
        spoken_for = sum(updated[i].marks or 0 for i in members)
        remaining = rate - spoken_for
        if remaining <= 0 or remaining % len(unknown):
            continue
        share = remaining // len(unknown)
        for i in unknown:
            updated[i] = updated[i].model_copy(update={"marks": share})

    return updated


def _sum_marks(questions: list[Question]) -> int | None:
    values = [q.marks for q in questions if q.marks is not None]
    return sum(values) if values else None


def question_lines(index: LineIndex, question: Question) -> list[Line]:
    """The lines a question was built from."""
    by_id = index.by_id()
    return [by_id[line_id] for line_id in question.line_ids if line_id in by_id]
