"""Tests for the Textract engine.

No account and no network. The parts worth testing are the coordinate conversion
and the failure translation, and both are pure functions over a response or an
exception — so recorded shapes exercise them exactly, and a real call would only
add flakiness and cost.

The conversion is where a convention could be misread, which makes it the highest
value test in this file: Textract reports boxes as ratios of the page with the
origin top-left, and so does this project, so the test's real job is to prove that
nobody "helpfully" flipped an axis.
"""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from vedaai_contracts import OcrEngine

from grader.ocr.base import EngineUnavailable, PageInput
from grader.ocr.textract import TextractEngine, parse


def block(
    block_id: str,
    kind: str,
    text: str,
    box: tuple[float, float, float, float],
    *,
    confidence: float = 99.0,
    children: list[str] | None = None,
) -> dict:
    left, top, width, height = box
    out: dict = {
        "Id": block_id,
        "BlockType": kind,
        "Text": text,
        "Confidence": confidence,
        "Geometry": {"BoundingBox": {"Left": left, "Top": top, "Width": width, "Height": height}},
    }
    if children:
        out["Relationships"] = [{"Type": "CHILD", "Ids": children}]
    return out


def response(*blocks: dict) -> dict:
    return {"Blocks": [{"BlockType": "PAGE", "Id": "page"}, *blocks]}


class FakeClient:
    """Stands in for the boto3 Textract client."""

    def __init__(self, result: dict | Exception) -> None:
        self.result = result
        self.calls: list[dict] = []

    def detect_document_text(self, **kwargs: object) -> dict:
        self.calls.append(dict(kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def aws_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "DetectDocumentText")


def png_page(size: int = 64) -> PageInput:
    return PageInput(index=0, width=1000, height=1400, png=b"x" * size, filename="sheet.pdf")


class TestGeometry:
    def test_a_box_is_carried_across_unchanged(self) -> None:
        # The whole reason this engine was chosen. Textract's ratios and this
        # project's contract are the same convention, so the conversion is an
        # identity — and the test exists to prove nobody flipped an axis.
        lines = parse(response(block("l1", "LINE", "Refraction", (0.1, 0.2, 0.5, 0.04))))
        assert len(lines) == 1
        assert lines[0].box.x0 == pytest.approx(0.1)
        assert lines[0].box.y0 == pytest.approx(0.2)
        assert lines[0].box.x1 == pytest.approx(0.6)
        assert lines[0].box.y1 == pytest.approx(0.24)

    def test_the_origin_is_the_top_left(self) -> None:
        # A line near the top of the page must have a small y, not a large one.
        # An inverted axis is the failure that looks plausible in every other test.
        top = parse(response(block("l1", "LINE", "first line", (0.1, 0.02, 0.6, 0.03))))[0]
        bottom = parse(response(block("l2", "LINE", "last line", (0.1, 0.94, 0.6, 0.03))))[0]
        assert top.box.y0 < bottom.box.y0

    def test_a_box_running_off_the_page_is_clamped(self) -> None:
        # Textract occasionally reports fractionally outside the page on writing
        # that runs to the edge. The contract rejects that, correctly — but a
        # stray hundredth is not worth discarding a line over.
        lines = parse(response(block("l1", "LINE", "to the edge", (0.9, 0.5, 0.2, 0.04))))
        assert lines[0].box.x1 == pytest.approx(1.0)

    def test_a_zero_area_box_is_dropped_rather_than_repaired(self) -> None:
        # It carries no geometry, and inventing some would put a highlight
        # somewhere arbitrary.
        assert parse(response(block("l1", "LINE", "ghost", (0.4, 0.4, 0.0, 0.0)))) == []

    def test_geometry_that_is_missing_or_malformed_is_dropped(self) -> None:
        assert parse({"Blocks": [{"BlockType": "LINE", "Text": "no geometry", "Id": "l1"}]}) == []
        assert parse(
            {
                "Blocks": [
                    {
                        "BlockType": "LINE",
                        "Text": "bad geometry",
                        "Id": "l1",
                        "Geometry": {
                            "BoundingBox": {"Left": "x", "Top": 0, "Width": 1, "Height": 1}
                        },
                    }
                ]
            }
        ) == []


class TestLinesAndWords:
    def test_confidence_is_converted_from_a_percentage(self) -> None:
        lines = parse(
            response(block("l1", "LINE", "maybe", (0.1, 0.1, 0.4, 0.03), confidence=82.5))
        )
        assert lines[0].confidence == pytest.approx(0.825)

    def test_words_come_from_textract_s_own_relationships(self) -> None:
        # Not from comparing boxes. Two lines of a cramped hand overlap
        # vertically, and geometric re-association would let whichever line's box
        # contained a word's centre claim it.
        lines = parse(
            response(
                block("l1", "LINE", "angle of", (0.1, 0.1, 0.4, 0.03), children=["w1", "w2"]),
                block("w1", "WORD", "angle", (0.1, 0.1, 0.18, 0.03)),
                block("w2", "WORD", "of", (0.3, 0.1, 0.08, 0.03)),
                block("w3", "WORD", "elsewhere", (0.1, 0.9, 0.3, 0.03)),
            )
        )
        assert [w.text for w in lines[0].words] == ["angle", "of"]

    def test_a_word_child_that_is_not_a_word_is_ignored(self) -> None:
        lines = parse(
            response(
                block("l1", "LINE", "text", (0.1, 0.1, 0.4, 0.03), children=["s1"]),
                block("s1", "SELECTION_ELEMENT", "", (0.1, 0.1, 0.02, 0.02)),
            )
        )
        assert lines[0].words == []

    def test_non_line_blocks_never_become_lines(self) -> None:
        lines = parse(
            response(
                block("t1", "TABLE", "", (0.1, 0.1, 0.8, 0.4)),
                block("w1", "WORD", "orphan word", (0.1, 0.1, 0.2, 0.03)),
            )
        )
        assert lines == []

    def test_an_empty_response_is_not_an_error(self) -> None:
        # A blank page is a real thing a student hands in.
        assert parse({}) == []
        assert parse({"Blocks": []}) == []


class TestFailures:
    """Every one of these reaches a teacher as a warning beside an answer sheet.

    So the message has to name what to change. "InvalidClientTokenId" does not.
    """

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("InvalidClientTokenId", "expired key pair"),
            ("ExpiredToken", "session token has expired"),
            ("AccessDeniedException", "textract:DetectDocumentText"),
            ("ThrottlingException", "throttling"),
            ("DocumentTooLargeException", "lower DPI"),
            ("UnrecognizedClientException", "same account"),
        ],
    )
    def test_an_aws_error_is_translated_into_an_action(self, code: str, expected: str) -> None:
        engine = TextractEngine(region="ap-south-1", client=FakeClient(aws_error(code)))
        with pytest.raises(EngineUnavailable) as caught:
            engine.transcribe(png_page())
        assert expected in str(caught.value)
        # The region is always named, since calling the wrong one is a common and
        # otherwise invisible mistake.
        assert "ap-south-1" in str(caught.value)

    def test_a_network_failure_says_to_check_the_region(self) -> None:
        failure = EndpointConnectionError(endpoint_url="https://textract.ap-south-1.amazonaws.com")
        engine = TextractEngine(region="ap-south-1", client=FakeClient(failure))
        with pytest.raises(EngineUnavailable) as caught:
            engine.transcribe(png_page())
        assert "region is correct" in str(caught.value)

    def test_an_unknown_failure_still_reports_something_usable(self) -> None:
        engine = TextractEngine(client=FakeClient(RuntimeError("something odd")))
        with pytest.raises(EngineUnavailable) as caught:
            engine.transcribe(png_page())
        assert "something odd" in str(caught.value)

    def test_a_page_with_no_pixels_explains_the_cache(self) -> None:
        # The failure mode that once produced silence indistinguishable from a
        # blank page.
        engine = TextractEngine(client=FakeClient(response()))
        with pytest.raises(EngineUnavailable) as caught:
            engine.transcribe(PageInput(index=0, width=100, height=100, png=None))
        assert "render cache" in str(caught.value)

    def test_an_oversized_page_is_refused_before_the_call(self) -> None:
        # Paying for a request AWS will reject is pointless, and the local check
        # can name the fix.
        client = FakeClient(response())
        engine = TextractEngine(client=client)
        with pytest.raises(EngineUnavailable) as caught:
            engine.transcribe(png_page(size=11 * 1024 * 1024))
        assert "lower DPI" in str(caught.value)
        assert client.calls == [], "no request should have been sent"


class TestEngineIdentity:
    def test_it_reports_its_own_provenance(self) -> None:
        assert TextractEngine().engine is OcrEngine.AWS_TEXTRACT

    def test_an_injected_client_counts_as_available(self) -> None:
        assert TextractEngine(client=FakeClient(response())).available() is True

    def test_the_page_is_sent_as_bytes(self) -> None:
        # Synchronous, from bytes, so S3 stays out of the recognition path.
        client = FakeClient(response(block("l1", "LINE", "hello", (0.1, 0.1, 0.2, 0.03))))
        TextractEngine(client=client).transcribe(png_page())
        assert client.calls == [{"Document": {"Bytes": b"x" * 64}}]

    def test_the_region_defaults_to_the_configured_one(self) -> None:
        assert TextractEngine(region="eu-west-2").region == "eu-west-2"
