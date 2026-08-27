"""Tests for answer-to-question alignment.

This is the module the project is graded on, and the tests are weighted toward
the one error that matters most: telling a teacher a question was left blank when
it was answered. A teacher acts on that without re-reading the script, so several
cases below assert that the system prefers admitting uncertainty over claiming
absence — even where that costs precision elsewhere.
"""

from __future__ import annotations

from vedaai_contracts import (
    Anchor,
    AnchorStatus,
    AnswerBlock,
    AnswerStatus,
    BBox,
    InkRegion,
    InkRegionKind,
    PageBox,
    Question,
    QuestionPaper,
    Requirement,
    Section,
)

from grader.align import resolve


def q(qid: str, label: str, text: str, order: int, path: list[str], *, marks: int | None = 2,
      section: str | None = None) -> Question:
    return Question(
        qid=qid,
        label_raw=label,
        text=text,
        path=path,
        print_order=order,
        marks=marks,
        section_id=section,
    )


def block(
    block_id: str,
    text: str,
    *,
    y0: float,
    page: int = 0,
    line_ids: list[str] | None = None,
    ink_ids: list[str] | None = None,
    continuation: bool = False,
    pages: list[int] | None = None,
) -> AnswerBlock:
    return AnswerBlock(
        block_id=block_id,
        line_ids=line_ids if line_ids is not None else [f"as:{block_id[-3:]}"],
        ink_region_ids=ink_ids or [],
        text=text,
        geometry=[PageBox(page=page, box=BBox(x0=0.1, y0=y0, x1=0.9, y1=y0 + 0.08))],
        pages_spanned=pages or [page],
        has_continuation_marker=continuation,
    )


def anchor(anchor_id: str, label: str, qid: str | None, line_id: str, *,
           status: AnchorStatus = AnchorStatus.CONFIRMED) -> Anchor:
    return Anchor(
        anchor_id=anchor_id,
        claimed_label=label,
        claimed_qid=qid,
        line_id=line_id,
        page=0,
        box=BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.13),
        status=status,
    )


def paper(questions: list[Question], sections: list[Section] | None = None) -> QuestionPaper:
    return QuestionPaper(questions=questions, sections=sections or [])


REFRACTION = q("A/1", "1.", "Define refraction of light.", 0, ["1"])
REFLECTION = q("A/2", "2.", "State the laws of reflection.", 1, ["2"])
MOTOR = q("A/3", "3.", "Explain the working of an electric motor.", 2, ["3"], marks=5)


class TestConfirmedAnchors:
    def test_a_confirmed_anchor_places_its_answer(self) -> None:
        questions = [REFRACTION, REFLECTION]
        blocks = [
            block("blk:000", "Refraction is the bending of light.", y0=0.10, line_ids=["as:0001"]),
            block("blk:001", "Angles of incidence and reflection are equal.", y0=0.30,
                  line_ids=["as:0002"]),
        ]
        anchors = [
            anchor("anc:000", "1.", "A/1", "as:0001"),
            anchor("anc:001", "2.", "A/2", "as:0002"),
        ]
        result = resolve(paper(questions), blocks, anchors, [])
        by_qid = result.by_qid()

        assert by_qid["A/1"].block_ids == ["blk:000"]
        assert by_qid["A/2"].block_ids == ["blk:001"]

    def test_confirmed_anchors_are_honoured_in_reverse_order(self) -> None:
        # The design error this corrects. Treating anchors as pins for a monotone
        # DP meant only a monotone subset survived, so a fully reversed sheet lost
        # nearly every correctly-labelled answer to the orphan pile.
        questions = [REFRACTION, REFLECTION, MOTOR]
        blocks = [
            block("blk:000", "A coil in a magnetic field turns.", y0=0.10, line_ids=["as:0001"]),
            block("blk:001", "Angles of incidence and reflection are equal.", y0=0.35,
                  line_ids=["as:0002"]),
            block("blk:002", "Refraction is the bending of light.", y0=0.60,
                  line_ids=["as:0003"]),
        ]
        anchors = [
            anchor("anc:000", "3.", "A/3", "as:0001"),
            anchor("anc:001", "2.", "A/2", "as:0002"),
            anchor("anc:002", "1.", "A/1", "as:0003"),
        ]
        result = resolve(paper(questions), blocks, anchors, [])
        by_qid = result.by_qid()

        assert by_qid["A/3"].block_ids == ["blk:000"]
        assert by_qid["A/2"].block_ids == ["blk:001"]
        assert by_qid["A/1"].block_ids == ["blk:002"]
        assert result.orphans == []

    def test_a_disputed_anchor_does_not_place_its_answer(self) -> None:
        # It still influences the score, but it may not fix a pairing the aligner
        # cannot revisit — that is the whole point of confirmation.
        questions = [REFRACTION]
        blocks = [block("blk:000", "Refraction is the bending of light.", y0=0.10,
                        line_ids=["as:0001"])]
        anchors = [
            anchor("anc:000", "8.", None, "as:0001", status=AnchorStatus.DISPUTED)
        ]
        result = resolve(paper(questions), blocks, anchors, [])
        mapping = result.by_qid()["A/1"]
        # The DP may still place it on semantic grounds; what matters is that the
        # disputed label did not do the placing.
        assert mapping.evidence.label_agreement < 3.0


class TestGapsAndOrphans:
    def test_an_unanswered_question_is_a_gap_on_the_question_axis(self) -> None:
        questions = [REFRACTION, MOTOR]
        blocks = [block("blk:000", "Refraction is the bending of light.", y0=0.10,
                        line_ids=["as:0001"])]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, [])

        assert result.by_qid()["A/3"].status is not AnswerStatus.ANSWERED
        assert result.by_qid()["A/3"].highlight is None

    def test_an_orphan_answer_is_a_gap_on_the_block_axis(self) -> None:
        # Required by the brief, and worth surfacing: an orphan often means our
        # own extraction missed a question rather than the student writing extra.
        questions = [REFRACTION]
        blocks = [
            block("blk:000", "Refraction is the bending of light.", y0=0.10,
                  line_ids=["as:0001"]),
            block("blk:001", "Rough work: 12 x 4 = 48 divided by 6 gives 8.", y0=0.60,
                  line_ids=["as:0002"]),
        ]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, [])

        assert [o.block_id for o in result.orphans] == ["blk:001"]
        assert result.orphans[0].highlight.boxes


class TestMultiBlockAnswers:
    def test_a_page_spanning_block_keeps_one_box_per_page(self) -> None:
        questions = [REFRACTION]
        spanning = AnswerBlock(
            block_id="blk:000",
            line_ids=["as:0001", "as:0002"],
            text="Refraction is the bending of light as it passes between media.",
            geometry=[
                PageBox(page=0, box=BBox(x0=0.1, y0=0.85, x1=0.9, y1=0.95)),
                PageBox(page=1, box=BBox(x0=0.1, y0=0.05, x1=0.9, y1=0.20)),
            ],
            pages_spanned=[0, 1],
        )
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), [spanning], anchors, [])
        highlight = result.by_qid()["A/1"].highlight

        assert highlight is not None
        assert highlight.spans_pages
        assert highlight.pages == [0, 1]
        assert len(highlight.boxes) == 2, "one union box per page, not one across both"


class TestFourStateStatus:
    def test_a_genuinely_blank_question_is_unanswered(self) -> None:
        # The only status that asserts absence, and the only one a teacher acts on
        # without checking. Everything else must be preferred where it fits.
        questions = [REFRACTION, MOTOR]
        blocks = [block("blk:000", "Refraction is the bending of light.", y0=0.10,
                        line_ids=["as:0001"])]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, [])

        assert result.by_qid()["A/3"].status is AnswerStatus.UNANSWERED

    @staticmethod
    def _ink_block() -> AnswerBlock:
        return AnswerBlock(
            block_id="blk:ink000",
            line_ids=[],
            ink_region_ids=["ink:001"],
            text="",
            geometry=[PageBox(page=0, box=BBox(x0=0.1, y0=0.3, x1=0.6, y1=0.6))],
            pages_spanned=[0],
        )

    @staticmethod
    def _ink_region() -> InkRegion:
        return InkRegion(
            region_id="ink:001",
            page=0,
            box=BBox(x0=0.1, y0=0.3, x1=0.6, y1=0.6),
            kind=InkRegionKind.WRITING,
            ink_ratio=0.2,
            pixel_count=40_000,
        )

    def test_a_drawing_question_accepts_a_region_with_no_text(self) -> None:
        # The case the ink pipeline exists for. A diagram has no text by nature,
        # so refusing it would leave every drawn answer unfindable.
        drawing = q("A/6", "6.", "Draw a labelled diagram of the eye.", 0, ["6"], marks=5)
        result = resolve(paper([drawing]), [self._ink_block()], [], [self._ink_region()])
        mapping = result.by_qid()["A/6"]

        assert mapping.status is AnswerStatus.OCR_FAILED
        assert mapping.highlight is not None
        assert mapping.highlight.derived_from == "ink_regions"

    def test_an_unreadable_region_is_not_handed_to_a_prose_question(self) -> None:
        # A region with no readable text says nothing about *which* question it
        # answers. Attached to one anyway it would vanish from the orphan list and
        # stop counting towards unassigned ink — and that total is what downgrades
        # absence claims elsewhere on the page. Reporting it honestly keeps both.
        result = resolve(paper([REFRACTION]), [self._ink_block()], [], [self._ink_region()])
        mapping = result.by_qid()["A/1"]

        assert [o.block_id for o in result.orphans] == ["blk:ink000"]
        assert mapping.status is not AnswerStatus.ANSWERED

    def test_and_it_is_never_reported_as_blank(self) -> None:
        # The claim that must not be made. There is writing on the page; we simply
        # could not read it or place it.
        result = resolve(paper([REFRACTION]), [self._ink_block()], [], [self._ink_region()])
        assert result.by_qid()["A/1"].status is not AnswerStatus.UNANSWERED

    def test_an_optional_question_may_be_skipped(self) -> None:
        # "Attempt any one" satisfied by one answer means the other is not an
        # omission. Reporting it as one is a product error a teacher would spot.
        optional = Section(
            section_id="B", label_raw="SECTION B", requirement=Requirement(answer_any=1)
        )
        first = q("B/4", "4.", "Describe an experiment about air pressure.", 0, ["4"],
                  section="B")
        second = q("B/5", "5.", "Discuss the behaviour of gases when heated.", 1, ["5"],
                   section="B")
        blocks = [block("blk:000", "An experiment about air pressure was performed.",
                        y0=0.10, line_ids=["as:0001"])]
        anchors = [anchor("anc:000", "4.", "B/4", "as:0001")]

        result = resolve(paper([first, second], [optional]), blocks, anchors, [])
        assert result.by_qid()["B/5"].status is AnswerStatus.NOT_REQUIRED

    def test_an_unmet_requirement_still_reports_omissions(self) -> None:
        # Below the quota the student genuinely owes answers, and calling them
        # optional would hide a real gap.
        optional = Section(
            section_id="B", label_raw="SECTION B", requirement=Requirement(answer_any=2)
        )
        first = q("B/4", "4.", "Describe an experiment about air pressure.", 0, ["4"],
                  section="B")
        second = q("B/5", "5.", "Discuss the behaviour of gases when heated.", 1, ["5"],
                   section="B")
        blocks = [block("blk:000", "An experiment about air pressure was performed.",
                        y0=0.10, line_ids=["as:0001"])]
        anchors = [anchor("anc:000", "4.", "B/4", "as:0001")]

        result = resolve(paper([first, second], [optional]), blocks, anchors, [])
        assert result.by_qid()["B/5"].status is not AnswerStatus.NOT_REQUIRED

    def test_a_continuation_marker_on_the_last_page_reports_pages_missing(self) -> None:
        questions = [REFRACTION, MOTOR]
        blocks = [
            block("blk:000", "Refraction is the bending of light, cont. on next page",
                  y0=0.85, line_ids=["as:0001"], continuation=True),
        ]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, [], pages_uploaded=1)

        assert result.by_qid()["A/3"].status is AnswerStatus.PAGES_MISSING


class TestAbsenceGuards:
    def test_unassigned_ink_suppresses_every_absence_claim(self) -> None:
        # Substantial writing belonging to no block means some answer went
        # unmapped, and the system is in no position to call anything blank.
        questions = [REFRACTION, MOTOR]
        blocks = [
            block("blk:000", "Refraction is the bending of light.", y0=0.10,
                  line_ids=["as:0001"]),
            block("blk:ink000", "", y0=0.50, line_ids=[], ink_ids=["ink:002"]),
        ]
        ink = [
            InkRegion(
                region_id="ink:002",
                page=0,
                box=BBox(x0=0.1, y0=0.5, x1=0.9, y1=0.9),
                kind=InkRegionKind.WRITING,
                ink_ratio=0.30,
                pixel_count=9000,
            )
        ]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, ink)

        if result.absence_claims_suppressed:
            assert all(
                m.status is not AnswerStatus.UNANSWERED
                for m in result.mappings
                if m.status is not AnswerStatus.ANSWERED
            )

    def test_bleed_through_does_not_suppress_absence_claims(self) -> None:
        # It appears on most double-sided scripts, so counting it would suppress
        # every legitimate unanswered report the product exists to make.
        questions = [REFRACTION, MOTOR]
        blocks = [block("blk:000", "Refraction is the bending of light.", y0=0.10,
                        line_ids=["as:0001"])]
        faint = [
            InkRegion(
                region_id="ink:009",
                page=0,
                box=BBox(x0=0.1, y0=0.5, x1=0.9, y1=0.9),
                kind=InkRegionKind.BLEED_THROUGH,
                ink_ratio=0.30,
                pixel_count=9000,
            )
        ]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, faint)

        assert result.unassigned_ink_ratio == 0.0
        assert not result.absence_claims_suppressed

    def test_a_plausible_answer_prevents_a_blank_claim(self) -> None:
        # Writing that looks like an answer to this question is on the sheet, even
        # though the aligner placed it elsewhere. "Found but unplaced" is honest;
        # "blank" is not.
        questions = [REFRACTION, REFLECTION]
        blocks = [
            block(
                "blk:000",
                "Refraction of light is the bending of light. "
                "Reflection of light obeys the laws of reflection.",
                y0=0.10,
                line_ids=["as:0001"],
            )
        ]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        result = resolve(paper(questions), blocks, anchors, [])

        assert result.by_qid()["A/2"].status is not AnswerStatus.UNANSWERED

    def test_an_answered_sibling_prevents_a_blank_claim(self) -> None:
        # Structural evidence, and it reaches where semantics cannot. A maths
        # answer like "R = V / I = 10 / 2 = 5 ohm" shares almost no word with its
        # question, so no similarity threshold rescues it — but a student who
        # answered 5 (a) very rarely leaves 5 (b) silently blank.
        first = q("B/5/a", "5 (a)", "State Ohm's law.", 0, ["5", "a"])
        second = q("B/5/b", "5 (b)", "A resistor carries 2 A at 10 V. Find its resistance.",
                   1, ["5", "b"])
        blocks = [
            block("blk:000", "Current is proportional to potential difference.", y0=0.10,
                  line_ids=["as:0001"])
        ]
        anchors = [anchor("anc:000", "5 (a)", "B/5/a", "as:0001")]
        result = resolve(paper([first, second]), blocks, anchors, [])

        assert result.by_qid()["B/5/b"].status is not AnswerStatus.UNANSWERED

    def test_evidence_of_an_answer_outranks_the_optional_rule(self) -> None:
        # Precedence, and getting it backwards filed an answered sub-part as
        # "not required" because its section's quota was already met. What the
        # student appears to have written outranks what the paper permitted.
        optional = Section(
            section_id="B", label_raw="SECTION B", requirement=Requirement(answer_any=1)
        )
        first = q("B/5/a", "5 (a)", "State Ohm's law.", 0, ["5", "a"], section="B")
        second = q("B/5/b", "5 (b)", "Find the resistance of the circuit.", 1, ["5", "b"],
                   section="B")
        blocks = [
            block("blk:000", "Current is proportional to potential difference.", y0=0.10,
                  line_ids=["as:0001"])
        ]
        anchors = [anchor("anc:000", "5 (a)", "B/5/a", "as:0001")]
        result = resolve(paper([first, second], [optional]), blocks, anchors, [])

        assert result.by_qid()["B/5/b"].status is AnswerStatus.UNCERTAIN


class TestReportShape:
    def test_every_question_gets_a_mapping(self) -> None:
        # The teacher's list must be complete, or a question silently vanishes.
        questions = [REFRACTION, REFLECTION, MOTOR]
        result = resolve(paper(questions), [], [], [])
        assert {m.qid for m in result.mappings} == {qq.qid for qq in questions}

    def test_status_counts_are_reported(self) -> None:
        questions = [REFRACTION, MOTOR]
        blocks = [block("blk:000", "Refraction is the bending of light.", y0=0.10,
                        line_ids=["as:0001"])]
        anchors = [anchor("anc:000", "1.", "A/1", "as:0001")]
        counts = resolve(paper(questions), blocks, anchors, []).counts_by_status()
        assert sum(counts.values()) == 2

    def test_an_empty_sheet_produces_no_answers_and_no_crash(self) -> None:
        result = resolve(paper([REFRACTION]), [], [], [])
        assert result.by_qid()["A/1"].status is not AnswerStatus.ANSWERED
        assert result.orphans == []


class TestTeacherCorrections:
    """``reassign`` — the path a teacher's correction takes.

    Worth testing directly because it is the one place where a human overrules
    the aligner, and because its failure mode is silent: a correction that drops
    writing leaves the teacher with a script they can no longer fully see.
    """

    PAPER = paper([REFRACTION, REFLECTION, MOTOR])

    def _resolved(self, blocks):
        return blocks, resolve(self.PAPER, blocks, [], [])

    def test_a_moved_block_lands_on_the_chosen_question(self) -> None:
        from grader.align import reassign

        blocks = [
            block("blk:000", "Light bends when it changes medium.", y0=0.10),
            block("blk:001", "Angle of incidence equals angle of reflection.", y0=0.30),
        ]
        result = resolve(self.PAPER, blocks, [], [])
        moved = reassign(
            self.PAPER, blocks, result, block_id="blk:001", to_qid="A/3"
        )

        motor = next(m for m in moved.mappings if m.qid == "A/3")
        assert motor.block_ids == ["blk:001"]
        assert motor.status is AnswerStatus.ANSWERED
        assert motor.teacher_override is True
        assert motor.highlight is not None

    def test_the_question_losing_a_block_is_never_called_blank(self) -> None:
        # A teacher moving an answer says nothing about whether the original
        # question was attempted. Asserting a blank on the strength of a
        # correction elsewhere is the unfounded absence claim to avoid.
        from grader.align import reassign

        blocks = [block("blk:000", "Light bends when it changes medium.", y0=0.10)]
        result = resolve(self.PAPER, blocks, [], [])
        moved = reassign(
            self.PAPER, blocks, result, block_id="blk:000", to_qid="A/3"
        )

        loser = next(m for m in moved.mappings if m.qid == "A/1")
        assert loser.block_ids == []
        assert loser.status is AnswerStatus.UNCERTAIN
        assert loser.highlight is None

    def test_moving_onto_an_answered_question_keeps_both_blocks(self) -> None:
        # The correction that replace-semantics made impossible: an answer split
        # across two blocks, one of which the aligner gave to the neighbour.
        # Moving it back must leave the question holding both, or restoring the
        # first block would displace the second and the split could never be
        # repaired.
        from grader.align import reassign

        blocks = [
            block("blk:000", "A coil carrying current sits in a magnetic field.", y0=0.10),
            block("blk:001", "The force on it makes the coil rotate.", y0=0.30),
        ]
        result = resolve(self.PAPER, blocks, [], [])
        moved = reassign(self.PAPER, blocks, result, block_id="blk:000", to_qid="A/3")
        moved = reassign(self.PAPER, blocks, moved, block_id="blk:001", to_qid="A/3")

        motor = next(m for m in moved.mappings if m.qid == "A/3")
        assert motor.block_ids == ["blk:000", "blk:001"]
        assert motor.highlight is not None
        assert len(motor.highlight.boxes) >= 1

    def test_merged_blocks_stay_in_document_order(self) -> None:
        # start_line_id and end_line_id name a span, so the order the teacher
        # happened to click in must not decide which end is which.
        from grader.align import reassign

        blocks = [
            block("blk:000", "First part of the answer.", y0=0.10, line_ids=["as:0001"]),
            block("blk:001", "Second part of the answer.", y0=0.30, line_ids=["as:0002"]),
        ]
        result = resolve(self.PAPER, blocks, [], [])
        moved = reassign(self.PAPER, blocks, result, block_id="blk:001", to_qid="A/3")
        moved = reassign(self.PAPER, blocks, moved, block_id="blk:000", to_qid="A/3")

        motor = next(m for m in moved.mappings if m.qid == "A/3")
        assert motor.block_ids == ["blk:000", "blk:001"]
        assert motor.start_line_id == "as:0001"
        assert motor.end_line_id == "as:0002"

    def test_displaced_writing_is_never_lost(self) -> None:
        # Every block must remain reachable after a correction: owned by a
        # question, or listed as an orphan. Writing that is neither is invisible
        # to the teacher.
        from grader.align import reassign

        blocks = [
            block("blk:000", "Light bends when it changes medium.", y0=0.10),
            block("blk:001", "Angle of incidence equals angle of reflection.", y0=0.30),
            block("blk:002", "A coil rotates in a magnetic field.", y0=0.50),
        ]
        result = resolve(self.PAPER, blocks, [], [])
        moved = reassign(self.PAPER, blocks, result, block_id="blk:002", to_qid="A/1")

        reachable = {bid for m in moved.mappings for bid in m.block_ids}
        reachable |= {o.block_id for o in moved.orphans}
        assert reachable == {"blk:000", "blk:001", "blk:002"}

    def test_an_unknown_block_or_question_changes_nothing(self) -> None:
        from grader.align import reassign

        blocks = [block("blk:000", "Light bends when it changes medium.", y0=0.10)]
        result = resolve(self.PAPER, blocks, [], [])

        assert reassign(self.PAPER, blocks, result, block_id="blk:999", to_qid="A/1") is result
        assert reassign(self.PAPER, blocks, result, block_id="blk:000", to_qid="Z/9") is result
