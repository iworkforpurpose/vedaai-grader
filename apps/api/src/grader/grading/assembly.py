"""Turning a model's judgement into a grade a teacher can check.

Nothing here trusts the judgement as given. A mark must cite a line that resolves
inside the answer, so a citation naming a line that does not exist loses that
point rather than the question; a check the panel could not settle becomes a
deferral with a name rather than a silent zero; and a question the marker never
reached is reported as unjudged rather than as nought out of five.

The two entry points share this file because they must not drift: `assemble`
handles a scalar rubric judgement and `assemble_checks` a binary check bank, and
both have to reach the same conclusions about citations, deferral and confidence.
"""

from __future__ import annotations

import json

from vedaai_contracts import LineIndex, Question, QuestionGrade, RubricPoint

from . import citations
from .rubric import Rubric


def _unjudged_points(rubric: Rubric, *, comment: str) -> list[RubricPoint]:
    return [
        RubricPoint(
            point_id=f"{rubric.qid}#{i + 1}",
            criterion=criterion.criterion,
            marks_available=criterion.marks,
            marks_awarded=0.0,
            satisfied=False,
            cited_line_ids=[],
            comment=comment,
        )
        for i, criterion in enumerate(rubric.criteria)
    ]


def _needs_a_person(rubric: Rubric, question: Question, *, graded_by: str) -> QuestionGrade:
    """A question whose answer is a drawing, left for someone who can see it."""
    return QuestionGrade(
        qid=question.qid,
        marks_available=rubric.marks_available,
        marks_awarded=0.0,
        rubric_points=_unjudged_points(
            rubric, comment="Answered by a drawing. Needs a person to look at the page."
        ),
        confidence=0.0,
        graded_by=graded_by,
        graded_on_partial_text=True,
    )


def _tool_input(message) -> dict:
    """The judgement from a tool-use response, whatever else the message holds."""
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            data = getattr(block, "input", None)
            if isinstance(data, str):
                return json.loads(data)
            if isinstance(data, dict):
                return data
    raise ValueError("the model returned no judgement")


def assemble(
    *,
    question: Question,
    rubric: Rubric,
    index: LineIndex,
    line_ids: list[str],
    judgement: dict,
    graded_by: str | None = None,
) -> QuestionGrade:
    """Turn a model judgement into a grade, or refuse it.

    Separated from the transport so the validation can be tested without a
    network call - it is the part that decides whether a grade is trustworthy,
    which makes it the part most worth testing.

    A judgement whose citations do not hold is not repaired. Every point reverts
    to unjudged and the reason is recorded, because a grade assembled from
    partially fabricated evidence is not a smaller grade, it is an unfounded one.
    """
    allowed = set(line_ids)
    points: list[RubricPoint] = []
    #: Points that claimed credit and cited nothing. They earn nothing, and they
    #: are counted because the student may have deserved them - see _confidence.
    unevidenced: list[str] = []

    for i, criterion in enumerate(rubric.criteria):
        raw = _point_for(judgement, i + 1)
        if raw is None:
            points.append(
                RubricPoint(
                    point_id=f"{rubric.qid}#{i + 1}",
                    criterion=criterion.criterion,
                    marks_available=criterion.marks,
                    marks_awarded=0.0,
                    satisfied=False,
                    cited_line_ids=[],
                    comment="The model did not judge this point.",
                )
            )
            continue

        satisfied = bool(raw.get("satisfied", False))
        awarded = max(0.0, min(float(raw.get("marks_awarded", 0.0)), criterion.marks))

        # A point the marker calls satisfied is worth what the paper allotted it.
        #
        # The two fields are returned independently and could disagree, and on a
        # real script they did: question 11(a), "Define atomic number and mass
        # number", worth 2. The student defined both. The marker set satisfied,
        # wrote "You provided clear definitions for both... Great job!" and awarded
        # 1 of 2. A teacher reading praise beside half marks cannot tell which half
        # of the grade to believe, which makes the whole grade useless.
        #
        # Partial credit is still expressible and is what an unsatisfied point with
        # a positive mark means: partly there, not met.
        #
        # Neither reading applies to a point that credited itself and cited
        # nothing. Question 16 of a real script, worth 5: two points cited four
        # lines each, and the third claimed credit with an empty citation list -
        # satisfied on one run, unsatisfied but carrying marks on the next. Either
        # way the citation check saw "marks awarded with no line cited", and it
        # refuses a question whole rather than in part, so a correct and fully
        # evidenced 3.5 became 0 and unjudged. Twice, by two different routes.
        #
        # A missing citation is an omission, not a fabrication. The model did not
        # invent evidence; it failed to give any, and the answer to an unevidenced
        # claim is to not credit that claim - not to throw away the claims that
        # were evidenced. An invented or out-of-scope line id is the other thing
        # entirely, and still refuses the question below: a model making evidence
        # up for one point has said nothing trustworthy about the others.
        cited = citations.resolve_all(raw.get("cited_line_ids") or [], index)
        comment = raw.get("comment")
        # The named fault, put in front of the comment. A teacher checking a
        # withheld mark wants "150/10 is 15, not 1.5" before the encouragement.
        named = str(raw.get("error") or "").strip()
        if named and awarded < criterion.marks:
            comment = f"{named} - {comment}" if comment else named
        if not cited and (satisfied or awarded > 0):
            # Not shown as met either. "Satisfied, nought marks" recreates in the
            # other direction the contradiction the promotion exists to remove.
            satisfied = False
            awarded = 0.0
            unevidenced.append(f"{rubric.qid}#{i + 1}")
            comment = (
                "The marker credited this point but cited no line for it, so it "
                "could not be checked and earned nothing. Worth reading yourself."
            )
        elif satisfied:
            awarded = criterion.marks
        points.append(
            RubricPoint(
                point_id=f"{rubric.qid}#{i + 1}",
                criterion=criterion.criterion,
                marks_available=criterion.marks,
                marks_awarded=awarded,
                satisfied=satisfied,
                cited_line_ids=cited,
                comment=comment,
            )
        )

    problems = citations.check(points, index, allowed_line_ids=allowed)
    if problems:
        reason = "; ".join(str(p) for p in problems[:3])
        return QuestionGrade(
            qid=question.qid,
            marks_available=rubric.marks_available,
            marks_awarded=0.0,
            rubric_points=_unjudged_points(
                rubric, comment=f"Marking was refused - evidence did not check out ({reason})."
            ),
            feedback=None,
            confidence=0.0,
            graded_by=graded_by,
            graded_on_partial_text=True,
        )

    uncertain = bool(judgement.get("uncertain", False))
    return QuestionGrade(
        qid=question.qid,
        marks_available=rubric.marks_available,
        marks_awarded=sum(p.marks_awarded for p in points),
        rubric_points=points,
        # The only place this is true. Every other constructor in this module
        # returns a grade nobody judged, and the difference is not otherwise
        # recoverable from the payload - see the field's own note.
        judged=True,
        feedback=judgement.get("feedback"),
        graded_by=graded_by,
        # Confidence is not asked of the model - a self-reported number is not
        # evidence. It is derived from whether the model flagged the transcription
        # as unreadable and from how much of the rubric it managed to cite.
        confidence=0.35 if uncertain else _confidence(points, unevidenced=len(unevidenced)),
        graded_on_partial_text=uncertain,
    )


def _point_for(judgement: dict, index: int) -> dict | None:
    for raw in judgement.get("points", []) or []:
        if isinstance(raw, dict) and raw.get("index") == index:
            return raw
    return None


def _confidence(points: list[RubricPoint], *, unevidenced: int) -> float:
    """How far a teacher can take this grade on trust.

    A point that claimed credit and cited nothing was dropped to zero. Whether
    the student had actually earned it is exactly what cannot be established
    here, so the grade goes to a person however clean the rest of it looks - a
    quiet zero on an answer nobody re-reads is how marking goes wrong without
    anyone noticing.
    """
    share = _cited_share(points)
    return min(share, 0.5) if unevidenced else share


def _cited_share(points: list[RubricPoint]) -> float:
    """Share of awarded points that pointed at specific lines.

    A grade whose marks are all traceable is one a teacher can check in seconds;
    one whose marks rest on unstated evidence needs re-reading. That difference is
    what the number is for.
    """
    awarded = [p for p in points if p.marks_awarded > 0]
    if not awarded:
        # Nothing awarded is a definite judgement, not an uncertain one: a zero
        # needs no citation to be checkable, since the whole answer is the
        # evidence.
        return 0.8
    cited = sum(1 for p in awarded if p.cited_line_ids)
    return round(0.5 + 0.5 * (cited / len(awarded)), 2)


def assemble_checks(
    *,
    question: Question,
    rubric: Rubric,
    bank,
    index: LineIndex,
    line_ids: list[str],
    judgement: dict,
    graded_by: str | None = None,
) -> QuestionGrade:
    """Turn binary check answers into a grade.

    One rule per state, and each exists because of a measured failure.

    **Yes earns the mark, and must cite a line.** Same as before: a mark with no
    resolvable evidence is unfounded rather than small.

    **No earns nothing, and must name the fault.** A refusal with no named fault is
    not a judgement, it is a shrug - so it is treated as *unsure* rather than as a
    zero. This is what stops the strictness that a model-written mark scheme
    introduced, where the false-zero rate tripled because the marker withheld marks
    it could not explain.

    **Unsure defers the mark.** Neither awarded nor refused. It is reported as a
    single named check for the teacher to settle, which is a smaller ask than
    re-reading the answer, and it is the honest state for damaged transcription.

    **An unverifiable check is credited.** Where the question refers to material the
    paper never supplied, nothing downstream can confirm a quotation or a position.
    The missing input is ours; the student does not lose a mark for it.
    """
    answers = {}
    for raw in judgement.get("checks", []) or []:
        if isinstance(raw, dict) and isinstance(raw.get("index"), int):
            answers[raw["index"]] = raw

    points: list[RubricPoint] = []
    deferred: list[str] = []
    credited_unverifiable: list[str] = []

    for i, check in enumerate(bank.checks, start=1):
        raw = answers.get(i) or {}
        met = raw.get("met")
        cited = citations.resolve_all(raw.get("cited_line_ids") or [], index)
        fault = str(raw.get("error") or "").strip()
        point_id = f"{rubric.qid}#{i}"

        if not check.verifiable and met is not True:
            # Cannot be checked against material nobody supplied. Credited, and
            # said out loud so the teacher knows which marks rest on it.
            credited_unverifiable.append(point_id)
            points.append(
                RubricPoint(
                    point_id=point_id,
                    criterion=check.ask,
                    marks_available=check.marks,
                    marks_awarded=check.marks,
                    satisfied=True,
                    cited_line_ids=cited,
                    comment="Given. This needs the passage or figure the question paper "
                    "does not contain, so it could not be checked - read it yourself.",
                )
            )
            continue

        if met is True:
            points.append(
                RubricPoint(
                    point_id=point_id,
                    criterion=check.ask,
                    marks_available=check.marks,
                    marks_awarded=check.marks,
                    satisfied=True,
                    cited_line_ids=cited,
                    comment=fault or None,
                )
            )
            continue

        if met is False and fault:
            points.append(
                RubricPoint(
                    point_id=point_id,
                    criterion=check.ask,
                    marks_available=check.marks,
                    marks_awarded=0.0,
                    satisfied=False,
                    cited_line_ids=cited,
                    comment=fault,
                )
            )
            continue

        # Unsure, or a refusal with no fault named. Deferred either way.
        deferred.append(point_id)
        points.append(
            RubricPoint(
                point_id=point_id,
                criterion=check.ask,
                marks_available=check.marks,
                marks_awarded=0.0,
                satisfied=False,
                cited_line_ids=cited,
                comment="Not decided - this one needs your eye."
                if met is None
                else "Refused without a stated reason, so it was not applied. Check this one.",
            )
        )

    # Validated over the points decided on evidence, not over all of them.
    #
    # A credited-unverifiable check has nothing to cite *by construction*: it was
    # given precisely because the material needed to check it is absent. Passing it
    # to a rule that requires a citation for any mark refused the whole question -
    # on the physics paper every question came back "evidence did not check out"
    # and `judged=False`, which read as the model failing when it was this function
    # contradicting itself.
    #
    # The invariant that matters is unchanged: a mark awarded *on the evidence of
    # the answer* must cite a line that resolves inside that answer. A mark given
    # because we could not look is a different thing, and it is labelled as such in
    # the comment rather than dressed up with evidence it does not have.
    evidenced = [p for p in points if p.point_id not in credited_unverifiable]
    problems = citations.check(evidenced, index, allowed_line_ids=set(line_ids))
    if problems:
        reason = "; ".join(str(p) for p in problems[:3])
        return QuestionGrade(
            qid=question.qid,
            marks_available=rubric.marks_available,
            marks_awarded=0.0,
            rubric_points=_unjudged_points(
                rubric, comment=f"Marking was refused - evidence did not check out ({reason})."
            ),
            confidence=0.0,
            graded_by=graded_by,
            graded_on_partial_text=True,
        )

    uncertain = bool(judgement.get("uncertain", False))
    awarded = sum(p.marks_awarded for p in points)
    settled = [p for p in points if p.point_id not in deferred]

    return QuestionGrade(
        qid=question.qid,
        marks_available=rubric.marks_available,
        marks_awarded=awarded,
        rubric_points=points,
        judged=True,
        feedback=judgement.get("feedback"),
        graded_by=graded_by,
        # Confidence is the share of the marks that were actually settled. A
        # question with a deferred check is not a confident grade however clean the
        # rest of it looks, and a teacher deciding where to spend their attention
        # needs that to be visible in the number.
        confidence=0.35
        if uncertain
        else round(
            (sum(p.marks_available for p in settled) / rubric.marks_available)
            if rubric.marks_available
            else 0.0,
            2,
        ),
        graded_on_partial_text=uncertain or bool(deferred) or bool(credited_unverifiable),
    )
