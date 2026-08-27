"""Ink extraction: geometry that does not depend on recognition.

This is the pipeline's second geometry source, and it answers questions the
transcription layer structurally cannot.

*Where is a hand-drawn diagram?* It has no text, so no OCR box, but it is still
the answer to a question and still has to be highlightable.

*Did the recognizer miss something?* Measured detection recall on real
handwriting is about 90% — a long declaration line in the test script produced no
box at all. Ink found where no line was reported is how that stays survivable.

*Is this space blank, or did recognition just fail?* The difference decides
whether a question is reported as unanswered, which is the most consequential
claim this product makes.

Three CV decisions carry most of the weight here, each for a reason visible in
real scans rather than in theory:

**Illumination flattening comes first.** Photographed pages have shadow
gradients, and the test script has a hand shadow across one corner. A global
threshold on the raw image classifies that shadow as ink. Dividing by a heavily
blurred copy of the page flattens the background so a threshold means the same
thing everywhere on it.

**Strokes are dilated before components are found.** Connected components on raw
handwriting yields one component per pen stroke — hundreds of fragments, none of
them a meaningful region. A horizontally-biased dilation merges strokes into
lines without merging separate lines into each other.

**Two thresholds, not one.** Bleed-through from the reverse side is faint;
pen on the near side is dark. Density cannot tell them apart — both are "ink" to
a single threshold — but darkness can, and getting this wrong means treating the
back of the page as answers on the front.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from vedaai_contracts import BBox, InkRegion, InkRegionKind

#: Kernel for estimating the page background, as a fraction of the shorter side.
#: Must be substantially larger than any pen stroke, or the flattening removes
#: the writing along with the shadow.
_BACKGROUND_KERNEL_FRACTION = 0.05

#: Dilation used to merge strokes into lines, expressed as multiples of the
#: page's own measured stroke height.
#:
#: Scaling these off the page dimensions was the obvious approach and was wrong.
#: The right merge distance is set by how big the *writing* is, not how big the
#: image is: word gaps scale with text size, and so does line spacing. Two real
#: scripts made the failure concrete — a page-relative kernel large enough to
#: join words on one of them merged the entire second page into a single region
#: covering 68% of it.
#:
#: Measuring stroke height first makes both bounds adapt on their own. Wider than
#: tall for the usual reason: horizontally it must bridge word gaps, vertically it
#: must not bridge the gap between lines.
_MERGE_WIDTH_STROKES = 1.6
_MERGE_HEIGHT_STROKES = 0.32

#: Ceiling on any single region, as a share of the page. A merge that produces
#: something larger has bridged unrelated content, so the merge is retried with a
#: narrower kernel rather than trusted.
_MAX_REGION_AREA_FRACTION = 0.35

#: Intensity below which a flattened pixel counts as *any* marking, and below
#: which it counts as genuinely dark pen on this side of the paper.
#:
#: Absolute rather than Otsu-derived, and that is the whole reason flattening
#: runs first: it normalizes the page background to near-white everywhere, so a
#: fixed number means the same thing on every scan.
#:
#: Otsu would be the reflexive choice and is actively wrong here. It is adaptive,
#: which is normally its virtue — but on a page whose only marking is faint
#: bleed-through it splits white from faint and declares the faint side "dark",
#: destroying the one distinction this module exists to make. On a page mixing
#: dark pen with bleed-through it does the opposite, drawing the line between
#: them so the bleed-through is not detected at all. Adaptiveness is the enemy
#: of an absolute judgement about how dark ink is.
_LOOSE_MAX_INTENSITY = 225
_STRICT_MAX_INTENSITY = 150

#: Below this fraction of genuinely dark pixels, a region is faint throughout and
#: reads as showing through from the reverse side rather than written on this one.
_BLEED_THROUGH_STRICT_FRACTION = 0.35

#: Regions smaller than this fraction of page area are speckle.
_NOISE_AREA_FRACTION = 2e-5

#: Rejecting things in the photograph that are not the answer sheet.
#:
#: A real scan contains a thumb holding the page down, the dark gap beyond the
#: paper's edge, and a binding shadow. All threshold as ink, all pass the size
#: test, and on the test script the thumb was being classified as crossed-out
#: work, because a solid mass trivially contains long horizontal runs.
#:
#: Fill ratio alone cannot separate them from writing, which was the first
#: attempt and was wrong: a line obliterated by scribbling is *also* nearly
#: solid, and rejecting it discards exactly the deletion the classifier exists to
#: find. Two better signals, and they cover different intruders:
#:
#: **Border contact.** A thumb, a page edge and a binding shadow all reach the
#: frame edge, because they are the frame rather than content within it. Writing
#: does not — a student writes inside the margins.
#:
#: **Shape.** A line of text, even a scribbled-out one, is much wider than it is
#: tall. A thumb is blobby. So a filled region is only dismissed when it is not
#: line-shaped, which leaves obliterated lines intact.
_BORDER_TOUCH_TOLERANCE = 3
_BORDER_REJECT_AREA_FRACTION = 0.0015
_SOLID_FILL_RATIO = 0.62
_SOLID_MIN_AREA_FRACTION = 0.004
_LINE_SHAPED_ASPECT = 2.5

#: Length of continuous horizontal run, relative to region width, that reads as
#: a crossing-out. Deliberately demanding — see the note on _has_horizontal_strike.
_STRIKE_WIDTH_FRACTION = 0.60

#: Vertical slack allowed in that run, in pixels. Small on purpose: enough for a
#: slightly sloped ruled line, not enough to smear cursive letters into a band.
_STRIKE_WAVE_TOLERANCE_PX = 3


@dataclass(frozen=True)
class InkMasks:
    """Intermediate CV products, exposed so tests and the debug overlay can see them."""

    flattened: np.ndarray
    """Grayscale with the background illumination divided out."""

    strict: np.ndarray
    """Genuinely dark pixels — pen on this side of the paper."""

    loose: np.ndarray
    """Any pixel darker than the page, including faint bleed-through."""

    otsu_threshold: float
    """Where Otsu would have split this page. Recorded for diagnostics only; it
    is deliberately not used for classification. See the threshold constants."""


def build_masks(image: np.ndarray) -> InkMasks:
    """Flatten illumination and produce the strict and loose ink masks."""
    # BGR, because OpenCV's imdecode is what supplies these pages. The channel
    # order matters here only through the luminance weights, but assuming the
    # wrong one would shift every threshold slightly and invisibly.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Estimate the page background by closing over a kernel far larger than any
    # stroke, then divide it out. This is what makes a single threshold valid
    # across a page with a shadow gradient.
    short_side = min(gray.shape[:2])
    k = max(3, int(short_side * _BACKGROUND_KERNEL_FRACTION) | 1)
    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    flattened = cv2.divide(gray, background, scale=255)

    _, strict = cv2.threshold(flattened, _STRICT_MAX_INTENSITY, 255, cv2.THRESH_BINARY_INV)
    _, loose = cv2.threshold(flattened, _LOOSE_MAX_INTENSITY, 255, cv2.THRESH_BINARY_INV)

    # Computed but not used for any decision, so that a scan whose behaviour
    # surprises us can be compared against what the adaptive choice would have
    # been.
    otsu_threshold, _ = cv2.threshold(
        flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove single-pixel speckle before anything downstream reasons about shape.
    speckle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, speckle)
    loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN, speckle)

    return InkMasks(
        flattened=flattened,
        strict=strict,
        loose=loose,
        otsu_threshold=float(otsu_threshold),
    )


#: Percentile of raw component height taken as the character height.
#:
#: The median is the intuitive choice and is too low. Handwriting decomposes into
#: many small fragments — dots, accents, the separated halves of a letter — which
#: drag the median well below the height of an actual character. Measured across
#: three real scripts the median was 12-16px while the upper quartile was 21-32px,
#: and the larger figure is the one that predicts word spacing.
_STROKE_HEIGHT_PERCENTILE = 0.75


def estimate_stroke_height(mask: np.ndarray) -> float:
    """Height of a character on this page, from the raw ink components.

    Taken before any merging, so it reflects individual strokes rather than
    whatever a dilation happened to join. This is the scale every merge decision
    is expressed in, which is what lets the same code handle a phone photo and a
    flatbed scan without retuning.
    """
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = [
        int(stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, count)
        # Ignore specks and full-page artefacts; neither is a character.
        if stats[i, cv2.CC_STAT_AREA] >= 8 and stats[i, cv2.CC_STAT_HEIGHT] < mask.shape[0] * 0.1
    ]
    if not heights:
        return 0.0
    heights.sort()
    index = min(len(heights) - 1, int(len(heights) * _STROKE_HEIGHT_PERCENTILE))
    return float(heights[index])


def _merge_kernel(stroke_height: float, *, width_scale: float = 1.0) -> np.ndarray:
    kw = max(3, int(stroke_height * _MERGE_WIDTH_STROKES * width_scale))
    kh = max(2, int(stroke_height * _MERGE_HEIGHT_STROKES))
    return cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))


def _has_horizontal_strike(strict_crop: np.ndarray) -> bool:
    """Whether a long, near-straight horizontal run crosses the region's middle.

    Detects a ruled single-line deletion, and deliberately nothing more. The
    scope is narrow because of what two attempts on real handwriting showed.

    A strict erosion — requiring a level run — found none of the four visible
    deletions on the test script, because students cross out with a wavy stroke.
    Loosening it to a row-coverage test with generous vertical dilation then
    flagged ordinary writing instead, including the student's name and a line
    transcribed at 0.96 confidence. The reason is that cursive letters are
    already joined, so high horizontal coverage is normal for legible text, not
    evidence of anything.

    The costs are not symmetric, and that decides the setting. A false positive
    removes a real answer from grading, so a student scores zero on a question
    they answered, with nothing in the score to hint why. A false negative means
    an abandoned attempt reaches the grader — bad, but the rubric citations show
    a teacher which lines were marked, so it is visible rather than silent.

    Given that, this test is tuned to find clean deletions and miss ambiguous
    ones. Heavy obliteration is caught separately by ink density in
    ``grader.regions``. A wavy single-line crossing-out over cursive text is a
    known gap, documented rather than papered over.
    """
    h, w = strict_crop.shape[:2]
    if h < 6 or w < 12:
        return False

    # A few pixels of slack for a sloped line, and no more: dilating by a
    # fraction of the region height smears cursive text into a solid band.
    slack = cv2.dilate(
        strict_crop,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, _STRIKE_WAVE_TOLERANCE_PX)),
    )

    # Only a genuinely continuous run survives this erosion.
    run_length = max(6, int(w * _STRIKE_WIDTH_FRACTION))
    survivors = cv2.erode(slack, cv2.getStructuringElement(cv2.MORPH_RECT, (run_length, 1)))

    # Middle band only: an underline sits below the text and is emphasis.
    top = int(h * 0.25)
    bottom = max(top + 1, int(h * 0.75))
    return bool(survivors[top:bottom, :].any())


def find_regions(
    image: np.ndarray,
    page: int,
    *,
    masks: InkMasks | None = None,
) -> list[InkRegion]:
    """Find and classify regions of ink on one page.

    Classification here uses only what the pixels show: size, density, darkness
    and strike geometry. Whether a region was *transcribed* is unknown at this
    point and is reconciled separately, which keeps this module free of any
    dependency on the recognizer.
    """
    masks = masks or build_masks(image)
    height, width = masks.loose.shape[:2]
    page_area = float(height * width)

    stroke_height = estimate_stroke_height(masks.loose)
    if stroke_height <= 0:
        return []

    # Merge, then check the result did not bridge unrelated content. Retrying
    # with a narrower kernel is better than accepting a region that swallows the
    # page, which destroys per-line geometry entirely.
    merged = cv2.dilate(masks.loose, _merge_kernel(stroke_height))
    if _largest_area_fraction(merged) > _MAX_REGION_AREA_FRACTION:
        merged = cv2.dilate(masks.loose, _merge_kernel(stroke_height, width_scale=0.5))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    regions: list[InkRegion] = []
    for label in range(1, count):  # 0 is the background
        x, y, w, h, _area = (
            stats[label, cv2.CC_STAT_LEFT],
            stats[label, cv2.CC_STAT_TOP],
            stats[label, cv2.CC_STAT_WIDTH],
            stats[label, cv2.CC_STAT_HEIGHT],
            stats[label, cv2.CC_STAT_AREA],
        )
        if w <= 1 or h <= 1:
            continue

        component = labels[y : y + h, x : x + w] == label
        loose_crop = (masks.loose[y : y + h, x : x + w] > 0) & component
        strict_crop = (masks.strict[y : y + h, x : x + w] > 0) & component

        loose_pixels = int(loose_crop.sum())
        if loose_pixels == 0:
            continue
        strict_pixels = int(strict_crop.sum())

        box_area = float(w * h)
        ink_ratio = min(1.0, loose_pixels / box_area)
        strict_fraction = strict_pixels / loose_pixels

        flat_values = masks.flattened[y : y + h, x : x + w][loose_crop]
        mean_darkness = float(flat_values.mean() / 255.0) if flat_values.size else 1.0

        kind = _classify(
            loose_pixels=loose_pixels,
            box_area=box_area,
            page_area=page_area,
            ink_ratio=ink_ratio,
            strict_fraction=strict_fraction,
            touches_border=_touches_border(x, y, w, h, width, height),
            aspect=w / h,
        )

        has_strike = False
        if kind is InkRegionKind.WRITING:
            has_strike = _has_horizontal_strike(strict_crop.astype(np.uint8) * 255)
            if has_strike:
                kind = InkRegionKind.STRUCK_THROUGH

        try:
            box = BBox.from_pixels(x, y, x + w, y + h, width=width, height=height)
        except ValueError:
            continue

        regions.append(
            InkRegion(
                region_id=f"ink:{page}:{label:04d}",
                page=page,
                box=box,
                kind=kind,
                ink_ratio=ink_ratio,
                mean_darkness=mean_darkness,
                has_horizontal_strike=has_strike,
                pixel_count=loose_pixels,
            )
        )

    return regions


def _largest_area_fraction(mask: np.ndarray) -> float:
    """Share of the page occupied by the biggest component's bounding box."""
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0.0
    page_area = float(mask.shape[0] * mask.shape[1])
    return max(
        (stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]) / page_area
        for i in range(1, count)
    )


def _touches_border(x: int, y: int, w: int, h: int, width: int, height: int) -> bool:
    t = _BORDER_TOUCH_TOLERANCE
    return x <= t or y <= t or (x + w) >= (width - t) or (y + h) >= (height - t)


def _classify(
    *,
    loose_pixels: int,
    box_area: float,
    page_area: float,
    ink_ratio: float,
    strict_fraction: float,
    touches_border: bool,
    aspect: float,
) -> InkRegionKind:
    """Decide what a region is from its pixels alone."""
    area_fraction = box_area / page_area
    if area_fraction < _NOISE_AREA_FRACTION or loose_pixels < 40:
        return InkRegionKind.NOISE

    # Reaches the frame edge and is not small: this is the photograph rather than
    # the page — a thumb, the gap past the paper, a binding shadow. Students
    # write inside the margins, so writing does not reach the edge.
    if touches_border and area_fraction >= _BORDER_REJECT_AREA_FRACTION:
        return InkRegionKind.NOISE

    # Filled and blobby. Checked before the strike test, since a solid mass
    # contains long horizontal runs and would otherwise read as crossed out.
    # The shape condition is what spares an obliterated line of text, which is
    # equally dense but far wider than it is tall.
    if (
        ink_ratio >= _SOLID_FILL_RATIO
        and area_fraction >= _SOLID_MIN_AREA_FRACTION
        and aspect < _LINE_SHAPED_ASPECT
    ):
        return InkRegionKind.NOISE

    # Faint throughout: visible against the page but never actually dark. Pen on
    # this side of the paper produces dark cores; the reverse side does not.
    if strict_fraction < _BLEED_THROUGH_STRICT_FRACTION:
        return InkRegionKind.BLEED_THROUGH

    if ink_ratio < 0.01:
        return InkRegionKind.NOISE

    return InkRegionKind.WRITING


def page_ink_ratio(regions: list[InkRegion]) -> float:
    """Share of the page covered by ink that counts as marking.

    Bleed-through and noise are excluded. That exclusion matters: this figure
    feeds the check that suppresses absence claims when unexplained ink exists,
    and bleed-through appears on every double-sided script. Counting it would
    suppress every legitimate "unanswered" the product exists to report.
    """
    return sum(r.box.area * r.ink_ratio for r in regions if r.kind.counts_as_page_ink)
