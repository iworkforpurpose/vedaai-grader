"""Reading damaged handwriting again, without moving anything.

The value of this pass is entirely in the text. Its risk is entirely in
everything else: a repaired line that lost its id, its box or its place would
break a citation, a highlight or the mapping, and all three are things the model
is deliberately not allowed to decide. So most of what is asserted here is that
nothing moved.
"""

from __future__ import annotations

import pytest
from vedaai_contracts import BBox, DocumentKind, Line, LineIndex, OcrEngine

from grader import reread
from grader.questions.expects import EvidenceKind


def line(line_id: str, text: str, *, y: float = 0.2, page: int = 0, conf: float = 0.9) -> Line:
    return Line(
        line_id=line_id,
        kind=DocumentKind.ANSWER_SHEET,
        page=page,
        box=BBox(x0=0.1, y0=y, x1=0.6, y1=y + 0.02),
        text=text,
        confidence=conf,
        engine=OcrEngine.AWS_TEXTRACT,
    )


class TestWhichAnswersAreWorthTheCall:
    def test_a_calculation_is(self) -> None:
        """Where the recognizer is measured worst and the marks are near zero."""
        assert reread.worth_rereading(EvidenceKind.WORKING, [line("as:0001", "x = 2")])

    def test_a_drawing_is(self) -> None:
        assert reread.worth_rereading(EvidenceKind.DRAWING, [line("as:0001", "N")])

    def test_prose_is_not(self) -> None:
        """Character error on real handwritten prose is 0.027. There is nothing to buy."""
        assert not reread.worth_rereading(
            EvidenceKind.REASONING, [line("as:0001", "Refraction is the bending of light.")]
        )

    def test_prose_the_recognizer_doubted_is(self) -> None:
        """The command verb predicts well and not completely.

        A paper can ask "state the relationship" and be answered in algebra.
        """
        doubted = [line("as:0001", "x", conf=0.2), line("as:0002", "y", conf=0.3)]

        assert reread.worth_rereading(EvidenceKind.RECALL, doubted)

    def test_an_answer_with_no_lines_is_not(self) -> None:
        assert not reread.worth_rereading(EvidenceKind.WORKING, [])


class TestTheCrop:
    def test_pads_beyond_the_recognised_glyphs(self) -> None:
        """A box drawn round what was read clips what was not.

        The premise of the whole pass is that the first read missed something, so
        cropping to exactly what it found would hide the exponent and the minus
        sign that are the reason for looking again.
        """
        page, box = reread.crop_box([line("as:0001", "x^2", y=0.40)])

        assert page == 0
        assert box.y0 < 0.40
        assert box.x0 < 0.1

    def test_stays_inside_the_page(self) -> None:
        """Padding at the edge must not produce a box the contract refuses."""
        _, box = reread.crop_box([line("as:0001", "top of the page", y=0.0)])

        assert box.y0 == 0.0
        assert box.x1 <= 1.0

    def test_takes_the_page_holding_most_of_the_answer(self) -> None:
        """A crop is one image, so a page-spanning answer reads its larger half.

        The rest keeps the transcription it had, which is the same outcome as not
        re-reading rather than a worse one.
        """
        page, _ = reread.crop_box(
            [
                line("as:0001", "a", y=0.90, page=0),
                line("as:0002", "b", y=0.10, page=1),
                line("as:0003", "c", y=0.14, page=1),
            ]
        )

        assert page == 1

    def test_no_lines_is_no_crop(self) -> None:
        assert reread.crop_box([]) is None


class FakeClient:
    """Returns a fixed payload and records the request."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[dict] = []
        self.chat = self

    @property
    def completions(self):
        return self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        import json

        content = json.dumps(self.payload)
        return type(
            "R",
            (),
            {"choices": [type("C", (), {"message": type("M", (), {"content": content})()})()]},
        )()


def png() -> bytes:
    import cv2
    import numpy as np

    ok, buffer = cv2.imencode(".png", np.full((400, 300, 3), 255, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


class TestWhatComesBack:
    @pytest.mark.asyncio
    async def test_repairs_are_keyed_by_the_line_they_belong_to(self) -> None:
        lines = [line("as:0001", "A 1 Orange = 0"), line("as:0002", "x", y=0.24)]
        client = FakeClient(
            {"lines": [{"index": 1, "text": "A + 1 Orange = 40"}, {"index": 2, "text": "x = 12"}]}
        )

        repairs = await reread.repair(lines, png(), client=client, model="m")

        assert repairs == {"as:0001": "A + 1 Orange = 40", "as:0002": "x = 12"}

    @pytest.mark.asyncio
    async def test_an_index_outside_the_range_is_dropped_not_clamped(self) -> None:
        """Clamping would attach one line's reading to another line's box.

        That is a wrong highlight and a wrong citation at once, which is worse
        than the damaged text this exists to improve on.
        """
        client = FakeClient({"lines": [{"index": 7, "text": "invented"}]})

        assert await reread.repair([line("as:0001", "x")], png(), client=client, model="m") == {}

    @pytest.mark.asyncio
    async def test_an_empty_reading_leaves_the_line_alone(self) -> None:
        """No reading is not a reading of nothing."""
        client = FakeClient({"lines": [{"index": 1, "text": "   "}]})

        assert await reread.repair([line("as:0001", "x")], png(), client=client, model="m") == {}

    @pytest.mark.asyncio
    async def test_a_failure_returns_nothing_rather_than_raising(self) -> None:
        """A re-read improves a transcription that already exists.

        Failing it must leave the answer as it was: the damaged text still places
        the answer and still carries its highlight.
        """

        class Broken(FakeClient):
            async def create(self, **kwargs):
                raise RuntimeError("the provider is down")

        got = await reread.repair([line("as:0001", "x")], png(), client=Broken({}), model="m")

        assert got == {}

    @pytest.mark.asyncio
    async def test_the_image_is_sent_inline_with_the_text(self) -> None:
        client = FakeClient({"lines": []})
        await reread.repair([line("as:0001", "x")], png(), client=client, model="m")

        content = client.calls[0]["messages"][1]["content"]
        kinds = [part["type"] for part in content]
        assert kinds == ["text", "image_url"]
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


class TestApplying:
    def test_replaces_only_the_text_and_only_where_told(self) -> None:
        index = LineIndex(
            kind=DocumentKind.ANSWER_SHEET,
            lines=[line("as:0001", "damaged"), line("as:0002", "fine", y=0.24)],
            engine=OcrEngine.AWS_TEXTRACT,
        )

        out = reread.applied(index, {"as:0001": "repaired"})

        assert [ln.text for ln in out.lines] == ["repaired", "fine"]

    def test_every_box_and_id_survives_untouched(self) -> None:
        """The invariant. Geometry is code's, and a re-read is not code."""
        index = LineIndex(
            kind=DocumentKind.ANSWER_SHEET,
            lines=[line("as:0001", "damaged"), line("as:0002", "fine", y=0.24)],
            engine=OcrEngine.AWS_TEXTRACT,
        )

        out = reread.applied(index, {"as:0001": "repaired"})

        assert [ln.line_id for ln in out.lines] == [ln.line_id for ln in index.lines]
        assert [ln.box for ln in out.lines] == [ln.box for ln in index.lines]
        assert [ln.page for ln in out.lines] == [ln.page for ln in index.lines]
        assert out.reading_order_confidence == index.reading_order_confidence

    def test_nothing_to_apply_returns_the_same_index(self) -> None:
        index = LineIndex(
            kind=DocumentKind.ANSWER_SHEET,
            lines=[line("as:0001", "fine")],
            engine=OcrEngine.AWS_TEXTRACT,
        )

        assert reread.applied(index, {}) is index

    def test_a_repaired_line_says_which_engine_read_it(self) -> None:
        """Provenance is per line so that a second reading can be told apart.

        A mark resting on a re-read line rests on a different reading of the same
        ink, and the contract keeps `engine` per line for exactly this.
        """
        index = LineIndex(
            kind=DocumentKind.ANSWER_SHEET,
            lines=[line("as:0001", "damaged"), line("as:0002", "fine", y=0.24)],
            engine=OcrEngine.AWS_TEXTRACT,
        )

        out = reread.applied(index, {"as:0001": "repaired"})

        assert out.lines[0].engine is OcrEngine.VLM_CROP_REREAD
        assert out.lines[1].engine is OcrEngine.AWS_TEXTRACT
