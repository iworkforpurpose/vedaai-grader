"""Handwriting transcription with a locally-hosted PaddleOCR model.

Self-hosted rather than a cloud API, which removes three problems at once: no
monthly page cap, no per-minute rate limit, and student work never leaves the
machine. Apache-2.0 licensed, and it runs on CPU.

**Measured on a real handwritten answer script** (photographed, shadowed, finger
in frame, bleed-through from the reverse side, struck-through work, rough working
in the right margin) using PP-OCRv6_medium on an M-series CPU:

  * 50 text regions detected, boxes tight and correctly placed
  * the margin question number "1." detected at confidence 1.00 — the anchor
    mechanism works on real input
  * transcription quality poor: ``#include <stdio.h>`` came back as
    ``Hinclude (stdio.h7``
  * **recall is not complete** — a long variable-declaration line was missed
    entirely, so roughly one line in ten produced no box
  * every struck-through line landed below 0.7 confidence
  * 14 s per page for detection plus recognition; model init is 46 s and must
    happen once per process, never per page

Three design consequences follow from those numbers, and they matter more than
the numbers themselves.

First, the architecture holds: detection is good while recognition is poor, and
because highlight geometry comes from boxes rather than from text, highlights are
correct even where the transcription is garbage. That was the bet, and this is
the evidence for it.

Second, incomplete recall is exactly why the ink mask is a second, independent
geometry source rather than a refinement. A missed line is a missing highlight
unless something that does not depend on recognition can still find the ink.

Third, low confidence on struck-through work is a usable signal. Combined with
high ink density it distinguishes deleted work from an answer, which no amount
of reading the text would achieve.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from vedaai_contracts import BBox, OcrEngine, Word

from .base import EngineUnavailable, PageInput, TranscribedLine

#: Model pair. The medium detector is what produced the numbers above; the
#: mobile variants trade some recall for roughly a third of the latency, which
#: is the knob to reach for if per-page time becomes the binding constraint.
DET_MODEL = os.getenv("PADDLE_DET_MODEL") or None
REC_MODEL = os.getenv("PADDLE_REC_MODEL") or None

#: Regions below this are kept but flagged. They are not discarded, because a
#: struck-through line still occupies space that a highlight may need to cover,
#: and because dropping low-confidence text is how a real answer becomes an
#: apparently blank one.
_KEEP_ALL_THRESHOLD = 0.0

_instance: Any = None
_lock = threading.Lock()


def _get_ocr() -> Any:
    """Build the recognizer once per process.

    Initialization costs about 46 seconds, so doing this per page would dominate
    every other cost in the pipeline. Guarded by a lock because pages are
    transcribed from a thread pool.
    """
    global _instance
    if _instance is not None:
        return _instance

    with _lock:
        if _instance is not None:
            return _instance
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise EngineUnavailable(
                "PaddleOCR is not installed. Install the local OCR extra with "
                "`uv sync --extra ocr-local` in apps/api."
            ) from exc

        kwargs: dict[str, Any] = {
            "lang": "en",
            # All three are document-level preprocessing steps that we either do
            # ourselves or do not want. Orientation and unwarping in particular
            # would silently transform the page, and every box we produce has to
            # remain in the coordinate space of the image we rendered and will
            # display — otherwise highlights land on a page the user never sees.
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if DET_MODEL:
            kwargs["text_detection_model_name"] = DET_MODEL
        if REC_MODEL:
            kwargs["text_recognition_model_name"] = REC_MODEL

        _instance = PaddleOCR(**kwargs)
        return _instance


def reset() -> None:
    """Drop the cached recognizer. Used by tests."""
    global _instance
    with _lock:
        _instance = None


class PaddleOcrEngine:
    """Local handwriting transcription producing line-level geometry."""

    @property
    def engine(self) -> OcrEngine:
        return OcrEngine.PADDLE_OCR_VL

    def available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return False
        return True

    def transcribe(self, page: PageInput) -> list[TranscribedLine]:
        if page.png is None:
            raise EngineUnavailable(
                "PaddleOcrEngine needs rendered page pixels, but PageInput.png was empty. "
                "This happens when a page was served from the render cache; re-render it "
                "before transcribing."
            )

        import io

        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(page.png)).convert("RGB")
        array = np.asarray(image)

        # Normalize against the actual decoded image rather than the declared
        # page size. If they ever disagree, trusting the declared size would
        # scale every box by the ratio between them — a failure that produces
        # plausible-looking but uniformly shifted highlights.
        height, width = array.shape[0], array.shape[1]

        result = _get_ocr().predict(array)
        if not result:
            return []

        first = result[0]
        payload = first if isinstance(first, dict) else dict(first)

        polys = payload.get("rec_polys")
        if polys is None:
            polys = payload.get("dt_polys") or []
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []

        lines: list[TranscribedLine] = []
        for poly, text, score in zip(polys, texts, scores, strict=False):
            cleaned = str(text).strip()
            if not cleaned:
                continue

            points = [(float(p[0]), float(p[1])) for p in poly]
            try:
                box = BBox.from_polygon(points, width=width, height=height)
            except ValueError:
                # A degenerate polygon would fail the geometry contract. Skipping
                # it loses one line; letting it through would produce an
                # invisible highlight that reads as a mapping failure.
                continue

            confidence = max(0.0, min(1.0, float(score)))
            if confidence < _KEEP_ALL_THRESHOLD:
                continue

            lines.append(
                TranscribedLine(
                    text=cleaned,
                    box=box,
                    confidence=confidence,
                    # Line-level only. Word geometry is available from this
                    # engine via `return_word_box`, but it costs time and
                    # highlights are drawn per region, not per word. It becomes
                    # worth enabling when rubric points need to cite a single
                    # sentence.
                    words=[],
                )
            )

        return lines


def as_word(text: str, box: BBox, confidence: float) -> Word:
    """Helper for when word-level output is enabled."""
    return Word(text=text, box=box, confidence=confidence)
