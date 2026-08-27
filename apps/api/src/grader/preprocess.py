"""Straightening a photographed page before anything reads it.

A phone photo of an answer sheet is the likeliest input this product gets, and it
arrives rotated a few degrees, keystoned from an off-axis camera, and lit unevenly
by whoever was standing over it. Recognition degrades on all three.

**Where this runs is the load-bearing decision.** Correcting the image changes
where everything on it is, so a correction applied after transcription would
invalidate every box already computed, and a correction applied only for
recognition would produce coordinates in a space the displayed page does not
share — highlights drawn confidently in the wrong place, which is precisely the
class of bug this codebase is organised to prevent. So correction happens at
render time, and the corrected bitmap is the one that is stored, shown, *and*
read. One image, one coordinate space, no mapping back.

**Every step can decline.** A wrong perspective warp does not degrade a page, it
destroys it, and a deskew that mistakes a table rule for a text baseline can tilt
a straight page. Each correction below measures its own evidence and returns the
image untouched when that evidence is weak. The asymmetry is deliberate and
matches the rest of the pipeline: the cost of leaving a page slightly crooked is
a few percent of recognition accuracy, and the cost of a bad warp is the whole
page.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: Widest rotation worth searching for, in degrees. A page held by hand is within
#: a few degrees; a larger apparent angle usually means the estimator has locked
#: onto something that is not text, so refusing to act on it is the safer error.
MAX_SKEW_DEGREES = 8.0

#: Angular resolution of the skew search.
SKEW_STEP_DEGREES = 0.25

#: Below this, rotating costs more in resampling blur than it recovers.
MIN_SKEW_DEGREES = 0.4

#: A page quadrilateral must cover at least this share of the frame. Anything
#: smaller is a photograph of something else, or of one corner of the page.
MIN_PAGE_AREA_SHARE = 0.40

#: And at most this much, since a page filling the frame edge-to-edge has no
#: visible border to find corners against — the detector would be fitting the
#: frame, not the paper.
MAX_PAGE_AREA_SHARE = 0.97

#: How far a detected quadrilateral may depart from a right-angled shape before
#: it is rejected, as the cosine of its sharpest corner. A real page photographed
#: off-axis stays roughly rectangular; a contour traced around a shadow does not.
MAX_CORNER_COSINE = 0.35

#: Kernel for estimating background illumination. Must be far larger than any
#: glyph, or the estimate absorbs the text it is meant to isolate.
_BACKGROUND_KERNEL = 61

#: Percentiles the contrast stretch maps to black and white. Not the extremes,
#: because a single non-ink dark pixel — a finger, a punch hole, the desk — would
#: otherwise set the black point and the stretch would barely reach the writing.
_STRETCH_LOW_PERCENTILE = 1.0
_STRETCH_HIGH_PERCENTILE = 99.5

#: Darkest background estimate the flattening will divide by. Below this the
#: division is amplifying noise rather than removing a shadow.
_MIN_BACKGROUND = 40


@dataclass(frozen=True)
class Corrected:
    """A corrected image, and an honest account of what was done to it."""

    image: np.ndarray
    dewarped: bool
    skew_degrees: float
    shadow_flattened: bool

    @property
    def changed(self) -> bool:
        return self.dewarped or self.skew_degrees != 0.0 or self.shadow_flattened

    def describe(self) -> str:
        """One line for the submission's warnings, or empty if nothing was done.

        Reported rather than silent: a teacher comparing the displayed page with
        the paper on their desk should not have to wonder why it looks different.
        """
        parts = []
        if self.dewarped:
            parts.append("corrected the camera angle")
        if self.skew_degrees:
            parts.append(f"rotated {self.skew_degrees:+.1f}°")
        if self.shadow_flattened:
            parts.append("evened out the lighting")
        return ", ".join(parts)


def correct(image: np.ndarray, *, is_photograph: bool = True) -> Corrected:
    """Straighten and even out a page image.

    ``is_photograph`` is False for a page rasterized from a PDF, which is already
    square, evenly lit and axis-aligned. Running a corner detector over one would
    be looking for a distortion that cannot be there, and any quadrilateral it
    found would be a false positive.

    Order is perspective, then rotation, then lighting. Rotation is measured
    against text baselines, and baselines are only straight once the camera angle
    is out; lighting is corrected last because both earlier steps resample the
    image and would smear a flattened background.
    """
    if image.size == 0:
        return Corrected(image=image, dewarped=False, skew_degrees=0.0, shadow_flattened=False)

    working = image
    dewarped = False

    if is_photograph:
        quad = _page_quadrilateral(working)
        if quad is not None:
            working = _warp_to_quad(working, quad)
            dewarped = True

    angle = _skew_angle(working)
    if abs(angle) >= MIN_SKEW_DEGREES:
        working = _rotate(working, angle)
    else:
        angle = 0.0

    flattened = False
    if is_photograph:
        working = _flatten_illumination(working)
        flattened = True

    return Corrected(
        image=working,
        dewarped=dewarped,
        skew_degrees=round(angle, 2),
        shadow_flattened=flattened,
    )


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _page_quadrilateral(image: np.ndarray) -> np.ndarray | None:
    """The paper's four corners, or None if they are not convincingly there.

    Deliberately hard to satisfy. Four conditions must hold together — four
    vertices, convex, a plausible share of the frame, and corners near square —
    because the failure this guards against is not a missed correction but a
    confident warp of the wrong region, which leaves the page unreadable and the
    geometry meaningless.
    """
    gray = _grayscale(image)
    height, width = gray.shape[:2]
    frame_area = float(height * width)

    # Blur before edge detection so paper texture and handwriting do not each
    # contribute their own contours.
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)
    # Close small gaps along the paper's edge, which shadows and low contrast
    # against the desk tend to break.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best: np.ndarray | None = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * MIN_PAGE_AREA_SHARE or area > frame_area * MAX_PAGE_AREA_SHARE:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if _sharpest_corner_cosine(approx.reshape(4, 2)) > MAX_CORNER_COSINE:
            continue
        if area > best_area:
            best, best_area = approx.reshape(4, 2).astype(np.float32), area

    return best


def _sharpest_corner_cosine(quad: np.ndarray) -> float:
    """How far the shape's worst corner is from a right angle, as |cos|.

    Zero is a perfect rectangle. A page photographed from an angle stays low; a
    contour traced around a shadow or a desk edge does not.
    """
    worst = 0.0
    for i in range(4):
        a = quad[(i - 1) % 4] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm == 0:
            return 1.0
        worst = max(worst, abs(float(np.dot(a, b)) / norm))
    return worst


def _order_corners(quad: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left.

    By geometry rather than by the order the contour tracer happened to produce:
    the sums of the coordinates pick out the two extreme corners, and their
    differences separate the other two.
    """
    totals = quad.sum(axis=1)
    diffs = np.diff(quad, axis=1).ravel()
    return np.array(
        [
            quad[int(np.argmin(totals))],
            quad[int(np.argmin(diffs))],
            quad[int(np.argmax(totals))],
            quad[int(np.argmax(diffs))],
        ],
        dtype=np.float32,
    )


def _warp_to_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Flatten the detected page to a rectangle.

    Sized from the longer of each pair of opposite edges, so the correction never
    discards resolution by shrinking the page to its foreshortened side.
    """
    ordered = _order_corners(quad)
    top_left, top_right, bottom_right, bottom_left = ordered

    width = int(
        max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    )
    height = int(
        max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    )
    if width < 32 or height < 32:
        return image

    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(ordered, target)
    return cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )


def _skew_angle(image: np.ndarray) -> float:
    """The page's rotation, from the angle at which its text rows line up best.

    Measured by projection-profile variance rather than by detecting lines. Text
    rows are not lines — they are ragged runs of glyphs with gaps — and a line
    detector on a school paper locks onto ruled margins and table borders, which
    are frequently not parallel to the writing. Row variance peaks when the rows
    are horizontal because that is when ink concentrates into few rows and the
    gaps between them empty out, and it needs no line to exist at all.
    """
    gray = _grayscale(image)

    # Work small: the angle of a page is a global property, and a downscaled copy
    # makes searching the whole range cheap.
    scale = 800.0 / max(gray.shape)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    background = cv2.medianBlur(gray, 31)
    flattened = cv2.divide(gray, background, scale=255)
    _, ink = cv2.threshold(flattened, 200, 255, cv2.THRESH_BINARY_INV)

    if float(ink.mean()) < 0.5:
        # Effectively blank. There is no text to be crooked, and any angle the
        # search returned would be fitting noise.
        return 0.0

    best_angle, best_score = 0.0, -1.0
    steps = int(MAX_SKEW_DEGREES / SKEW_STEP_DEGREES)
    for step in range(-steps, steps + 1):
        angle = step * SKEW_STEP_DEGREES
        rotated = _rotate(ink, angle, border=0)
        rows = rotated.sum(axis=1, dtype=np.float64)
        score = float(np.var(np.diff(rows)))
        if score > best_score:
            best_angle, best_score = angle, score

    return best_angle


def _rotate(image: np.ndarray, degrees: float, *, border: int = 255) -> np.ndarray:
    """Rotate about the centre, growing the canvas so no content is cropped."""
    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, degrees, 1.0)

    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    grown_width = int(height * sin + width * cos)
    grown_height = int(height * cos + width * sin)
    matrix[0, 2] += grown_width / 2.0 - centre[0]
    matrix[1, 2] += grown_height / 2.0 - centre[1]

    return cv2.warpAffine(
        image,
        matrix,
        (grown_width, grown_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border if image.ndim == 2 else (border, border, border),
    )


def _flatten_illumination(image: np.ndarray) -> np.ndarray:
    """Remove the shadow gradient while keeping the strokes' grey levels.

    Divides by a heavily blurred copy, which estimates how bright the paper
    *would* be at each point and cancels it. Deliberately not a binarization:
    recognition uses stroke thickness and antialiasing, and throwing those away to
    get a cleaner-looking page costs accuracy. The ink mask binarizes separately
    for its own purposes.
    """
    gray = _grayscale(image)
    background = cv2.medianBlur(gray, _BACKGROUND_KERNEL)

    # Floor the background before dividing by it. Where the estimate is nearly
    # black — the desk showing past the paper, a shadowed border, a finger in
    # frame — the division amplifies sensor noise into dense speckle, which is
    # both ugly on screen and a source of ink regions that look like writing and
    # are not. Flooring leaves those areas dark instead of turning them to
    # confetti, and has no effect anywhere the paper is actually visible.
    background = np.maximum(background, _MIN_BACKGROUND)
    flattened = cv2.divide(gray, background, scale=255)

    # Restore the contrast the division costs, which leaves the page near-white
    # and the strokes lighter than they were.
    #
    # Stretched between percentiles rather than between the extremes. A min-max
    # stretch is set by its darkest single pixel, and these photographs reliably
    # contain one that is not ink: a finger at the edge of frame, a punch hole, the
    # desk showing past the paper. One such pixel pins the low end at zero and the
    # stretch then does almost nothing for the writing it was meant to help.
    low, high = np.percentile(flattened, (_STRETCH_LOW_PERCENTILE, _STRETCH_HIGH_PERCENTILE))
    if high - low < 1.0:
        return flattened
    scaled = (flattened.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)
