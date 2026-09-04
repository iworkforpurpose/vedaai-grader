"""Handwriting transcription with Amazon Textract.

The engine used by the deployed service, chosen over the alternatives for a
reason that is specific rather than incidental: Textract reports every bounding
box as ratios of the page with the origin at the top-left, which is this
project's coordinate contract exactly. The conversion at the adapter boundary is
an identity, so there is no arithmetic in which a convention can drift — and
coordinate drift across a boundary is the failure this codebase is organised to
prevent.

Two further properties matter for a service running on AWS. Authentication is IAM
rather than an API key, so nothing secret has to be shipped or rotated: the ECS
task role supplies credentials and the container holds none. And the recognizer
runs in the same region as the service, so page images never leave the VPC.

**Synchronous, one page at a time, from bytes.** The asynchronous API exists for
multi-page documents and requires staging them in S3 first. This pipeline already
renders one page bitmap at a time — deliberately, to bound memory — so the
synchronous call is the natural fit and removes S3 from the recognition path
altogether.
"""

from __future__ import annotations

import os
from typing import Any

from vedaai_contracts import BBox, OcrEngine, Word

from .base import EngineUnavailable, PageInput, TranscribedLine

#: Region the client talks to. Textract is regional, and the deployed service
#: keeps recognition beside itself so images stay in the VPC.
DEFAULT_REGION = os.getenv("AWS_REGION") or "ap-south-1"

#: Textract's own ceiling for a synchronous image: 10 MB, 10000 px on a side. The
#: renderer already caps at 4000 px, so only the byte size can realistically bind.
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class TextractEngine:
    """Line-level handwriting transcription via Textract's synchronous API."""

    def __init__(self, *, region: str | None = None, client: Any | None = None) -> None:
        self.region = region or DEFAULT_REGION
        self._client = client

    @property
    def engine(self) -> OcrEngine:
        return OcrEngine.AWS_TEXTRACT

    def available(self) -> bool:
        """Whether a client can be built and credentials are resolvable.

        Deliberately does not call Textract. Availability is checked before every
        document, and a network round trip per check would cost more than it tells
        us — a credential that resolves now can still be refused later, and that
        case has to be handled at the call site regardless.
        """
        if self._client is not None:
            return True
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError
        except ModuleNotFoundError:
            return False

        try:
            session = boto3.session.Session(region_name=self.region)
            return session.get_credentials() is not None
        except NoCredentialsError:
            return False

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
                raise EngineUnavailable(
                    "boto3 is not installed; install the 'aws' extra to use Textract"
                ) from exc
            from ..clients import TEXTRACT_READ_TIMEOUT, aws_config

            # Adaptive retries are what turn a throttle into a pause rather
            # than a lost document: every failure here becomes a terminal
            # `EngineUnavailable`, and that ends the transcription of every
            # page, including the fifty-seven already paid for.
            self._client = boto3.client(
                "textract",
                region_name=self.region,
                config=aws_config(TEXTRACT_READ_TIMEOUT),
            )
        return self._client

    def transcribe(self, page: PageInput) -> list[TranscribedLine]:
        if page.png is None:
            raise EngineUnavailable(
                "TextractEngine needs rendered page pixels, but PageInput.png was empty. "
                "This happens when a page was served from the render cache; re-render it "
                "before transcribing."
            )
        if len(page.png) > MAX_IMAGE_BYTES:
            raise EngineUnavailable(
                f"page {page.index + 1} of {page.filename!r} is "
                f"{len(page.png) / 1_048_576:.1f} MB, above Textract's 10 MB limit for a "
                "synchronous call. Render at a lower DPI."
            )

        client = self._ensure_client()
        try:
            response = client.detect_document_text(Document={"Bytes": page.png})
        except Exception as exc:  # noqa: BLE001 - translated below, then re-raised
            raise EngineUnavailable(_explain(exc, self.region)) from exc
        return parse(response)


#: AWS error codes translated into the thing the operator has to change. Written
#: out because the raw codes are opaque at the point of failure: "InvalidClient
#: TokenId" appears in a warning beside a student's answer sheet, where nobody is
#: thinking about credential chains.
_EXPLANATIONS = {
    "InvalidClientTokenId": (
        "the AWS credentials are not valid — most often an expired key pair. Check "
        "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or the task role if this is "
        "running on ECS."
    ),
    "ExpiredToken": "the AWS session token has expired; refresh it and retry.",
    "ExpiredTokenException": "the AWS session token has expired; refresh it and retry.",
    "UnrecognizedClientException": (
        "AWS did not recognize the credentials. Check they belong to the same account "
        "as the region being called."
    ),
    "AccessDeniedException": (
        "the credentials are valid but not allowed to call Textract. Grant "
        "textract:DetectDocumentText to the user or task role."
    ),
    "UnsupportedDocumentException": (
        "Textract refused the image format. Pages are rendered as PNG, so this "
        "usually means the page came back empty."
    ),
    "InvalidParameterException": "Textract rejected the request parameters.",
    "ProvisionedThroughputExceededException": (
        "Textract is throttling this account. Retry, or request a higher limit."
    ),
    "ThrottlingException": "Textract is throttling this account. Retry in a moment.",
    "DocumentTooLargeException": (
        "the page is larger than Textract accepts synchronously. Render at a lower DPI."
    ),
}


def _explain(error: Exception, region: str) -> str:
    """A failure message naming what to change, not just what went wrong."""
    code = ""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = str((response.get("Error") or {}).get("Code") or "")

    detail = _EXPLANATIONS.get(code)
    if detail is None:
        endpoint = getattr(error, "__class__", type(error)).__name__
        if endpoint in {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError"}:
            detail = (
                f"could not reach Textract in {region}. Check the region is correct and "
                "that the network allows outbound HTTPS."
            )
        else:
            detail = f"{type(error).__name__}: {error}"

    prefix = f"Textract ({region})"
    return f"{prefix} could not read this page: {detail}"


def parse(response: dict) -> list[TranscribedLine]:
    """Turn a Textract response into lines with normalized geometry.

    Separated from the call so the whole conversion can be tested against recorded
    responses without a network or an account — which is the part worth testing,
    since it is where a coordinate convention could be misread.

    Words are attached to their line through Textract's own ``Relationships``
    rather than by comparing boxes. Geometric re-association looks equivalent and
    is not: two lines of a cramped hand overlap vertically, and a word would then
    be claimed by whichever line's box happened to contain its centre.
    """
    blocks = response.get("Blocks") or []
    by_id = {block["Id"]: block for block in blocks if "Id" in block}

    lines: list[TranscribedLine] = []
    for block in blocks:
        if block.get("BlockType") != "LINE":
            continue

        text = (block.get("Text") or "").strip()
        box = _bbox(block)
        if not text or box is None:
            continue

        lines.append(
            TranscribedLine(
                text=text,
                box=box,
                # Textract reports confidence as a percentage.
                confidence=float(block.get("Confidence", 0.0)) / 100.0,
                words=_words(block, by_id),
            )
        )
    return lines


def _words(line_block: dict, by_id: dict[str, dict]) -> list[Word]:
    """The words Textract itself assigned to this line."""
    out: list[Word] = []
    for relationship in line_block.get("Relationships") or []:
        if relationship.get("Type") != "CHILD":
            continue
        for child_id in relationship.get("Ids") or []:
            child = by_id.get(child_id)
            if child is None or child.get("BlockType") != "WORD":
                continue
            text = (child.get("Text") or "").strip()
            box = _bbox(child)
            if not text or box is None:
                continue
            out.append(
                Word(
                    text=text,
                    box=box,
                    confidence=float(child.get("Confidence", 0.0)) / 100.0,
                )
            )
    return out


def _bbox(block: dict) -> BBox | None:
    """Textract geometry as a ``BBox``, or None if it is unusable.

    Already normalized to the page with the origin top-left, so this is a rename
    rather than a conversion. Clamped because Textract occasionally reports a box
    fractionally outside the page on writing that runs to the edge, and the
    contract rejects that — correctly, but a stray hundredth of a percent is not
    worth discarding a line over.

    A zero-area box is dropped rather than nudged into validity. It carries no
    geometry, and inventing some would put a highlight somewhere arbitrary.
    """
    geometry = block.get("Geometry") or {}
    raw = geometry.get("BoundingBox") or {}
    try:
        left = float(raw["Left"])
        top = float(raw["Top"])
        width = float(raw["Width"])
        height = float(raw["Height"])
    except (KeyError, TypeError, ValueError):
        return None

    x0 = min(max(left, 0.0), 1.0)
    y0 = min(max(top, 0.0), 1.0)
    x1 = min(max(left + width, 0.0), 1.0)
    y1 = min(max(top + height, 0.0), 1.0)

    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)
