"""Rectangle feature classifier.

Mirrors ``MachiningRectanglePath`` from the SmartLaser hierarchy.

Detection gates (mirroring ``MachiningRectanglePath.RECT_STANDARD_STR``):
1. n ≈ 4  (the wire is nearly a 4-sided polygon)
2. Each interior corner angle ≈ 90° ± 22°
3. Opposite sides are similar in length (each ≥ 20% of the other)

No assumption is made about axis alignment — the classifier works on
arbitrarily rotated rectangles.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from .base import ContourMetrics, ShapeClassifier


# ── Thresholds ─────────────────────────────────────────────────────────────

_RECT_ANGLE_TOL_DEG = 22.0      # interior corner tolerance
_RECT_SIDE_RATIO_MIN = 0.20     # opposite sides must be ≥ 20% of each other
_RECT_SIDE_RATIO_MAX = 1.25     # also reject very elongated squares (>1.25)
_RECT_CIRCULARITY_MAX = 0.83    # must be less round than a near-circle


class RectangleClassifier(ShapeClassifier):
    """Detects an axis-independent rectangle and extracts L / W / CR.

    Corresponds to ``MachiningRectanglePath``.
    """

    shape_name: ClassVar[str] = "rectangle"
    priority: ClassVar[int] = 80    # after slot; rectangles are rarer than slots
    min_points: ClassVar[int] = 4

    def matches(self, m: ContourMetrics) -> bool:
        if m.n < self.min_points:
            return False
        # Reject near-circles early.
        if m.circularity > _RECT_CIRCULARITY_MAX:
            return False
        # Reject shapes that are too close to a square (aspect ≈ 1)
        # But allow low-point-count shapes (n < 10) to pass since they may be
        # coarse tessellations of actual rectangles
        if m.n >= 10 and m.aspect < 1.3:
            return False
        return self._rect_geometry_ok(m)

    def _rect_geometry_ok(self, m: ContourMetrics) -> bool:
        pts2d = m.pts2d
        n = m.n
        xmin, ymin, xmax, ymax = m.bbox
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5

        # 1) Extract corners using perpendicular corner detection.
        corners = _extract_corners(pts2d, cx, cy)
        if len(corners) < 4:
            return False

        # 2) Check interior corner angles (each ≈ 90°).
        angles = _corner_angles(corners)
        for ang in angles:
            # Reject if any angle is outside the tolerance OR if any angle
            # exceeds 100° (a rectangle cannot have corners > 90°).
            if abs(ang - 90.0) > _RECT_ANGLE_TOL_DEG or ang > 100.0:
                return False

        # 3) Opposite sides must be within the ratio tolerance.
        sides = _side_lengths(corners)
        if not sides:
            return False
        # Sort side lengths; the two smallest are adjacent pairs.
        s_sorted = sorted(sides)
        if len(s_sorted) >= 2:
            ratio_a = s_sorted[0] / max(s_sorted[2], 1e-9)
            ratio_b = s_sorted[1] / max(s_sorted[3] if len(s_sorted) > 3 else s_sorted[2], 1e-9)
            if ratio_a < _RECT_SIDE_RATIO_MIN or ratio_b < _RECT_SIDE_RATIO_MIN:
                return False
            if ratio_a > _RECT_SIDE_RATIO_MAX and ratio_b > _RECT_SIDE_RATIO_MAX:
                return False

        return True

    def classify(
        self,
        m: ContourMetrics,
        *,
        face_normal: tuple[float, float, float] | None,
        pts_world: list[tuple[float, float, float]] | None = None,
    ) -> dict[str, Any]:
        xmin, ymin, xmax, ymax = m.bbox
        length = round(max(xmax - xmin, ymax - ymin), 4)
        width = round(min(xmax - xmin, ymax - ymin), 4)
        # Rotation angle from bbox X axis.
        ra = self._rotation_angle(m)
        confidence = self._confidence(m)
        return {
            "diameter": None,
            "length": length,
            "width": width,
            "across_flats": None,
            "_confidence": round(confidence, 3),
        }

    def _rotation_angle(self, m: ContourMetrics) -> float:
        xmin, ymin, xmax, ymax = m.bbox
        w = xmax - xmin
        h = ymax - ymin
        # The "length" side is the longer bbox dimension.
        if w >= h:
            return 0.0
        return 90.0

    def _confidence(self, m: ContourMetrics) -> float:
        return min(1.0, 0.5 + m.circularity * 0.5)


# ── Corner helpers ─────────────────────────────────────────────────────────


def _extract_corners(
    pts2d: list[tuple[float, float]], cx: float, cy: float
) -> list[tuple[float, float]]:
    """Return up to 4 dominant corner points sorted by polar angle around (cx, cy).

    For low-tessellation STEP data the polygon may have many collinear
    points per side, so we cluster by turning-angle maxima rather than
    blindly taking every 4th vertex.
    """
    n = len(pts2d)
    if n < 4:
        return []

    angles: list[float] = []
    for i in range(n):
        prev = pts2d[(i - 1 + n) % n]
        curr = pts2d[i]
        next_pt = pts2d[(i + 1) % n]
        # Vectors from prev→curr and curr→next
        v1x, v1y = curr[0] - prev[0], curr[1] - prev[1]
        v2x, v2y = next_pt[0] - curr[0], next_pt[1] - curr[1]
        dot = v1x * v2x + v1y * v2y
        mag = math.hypot(v1x, v1y) * math.hypot(v2x, v2y) + 1e-12
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))
        angles.append(angle)

    threshold = 30.0   # angle must exceed this to be considered a corner
    candidates = [(angles[i], pts2d[i]) for i in range(n) if angles[i] > threshold]

    if len(candidates) >= 4:
        # ≥ 4 real turning-angle peaks — use the top-4.
        candidates.sort(key=lambda x: -x[0])
        raw = [c[1] for c in candidates[:4]]
    else:
        # < 4 turning-angle peaks → axis-aligned polygon with collinear
        # corners (low-poly step). Fall back to bbox extremes so a
        # 40×12 rectangle with only 3 real turning angles still works.
        xs = [p[0] for p in pts2d]
        ys = [p[1] for p in pts2d]
        raw = [
            (min(xs), min(ys)),
            (max(xs), min(ys)),
            (max(xs), max(ys)),
            (min(xs), max(ys)),
        ]

    # Deduplicate corners that are essentially the same point
    def _unique_by_dist(corners: list[tuple[float, float]], tol: float = 1e-4) -> list[tuple[float, float]]:
        unique = []
        for c in corners:
            if not unique:
                unique.append(c)
            else:
                is_dup = False
                for u in unique:
                    if abs(c[0] - u[0]) < tol and abs(c[1] - u[1]) < tol:
                        is_dup = True
                        break
                if not is_dup:
                    unique.append(c)
        return unique

    raw = _unique_by_dist(raw)

    # If we have exactly 4 bbox corners but they're not unique enough,
    # check if the shape looks like a rectangle based on aspect ratio
    # and circularity alone
    if len(raw) < 4:
        # Fallback: check if bbox aspect is reasonable for a rectangle
        # and circularity is in the right range
        xmin = min(p[0] for p in pts2d)
        xmax = max(p[0] for p in pts2d)
        ymin = min(p[1] for p in pts2d)
        ymax = max(p[1] for p in pts2d)
        w = xmax - xmin
        h = ymax - ymin
        aspect = max(w, h) / min(w, h) if min(w, h) > 1e-9 else float('inf')
        # A square has aspect=1, a 2:1 rectangle has aspect=2
        if 1.0 <= aspect <= 2.5:
            # This looks like a square or modest rectangle - use bbox corners
            raw = [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
            ]

    def polar(p: tuple[float, float]) -> float:
        return math.atan2(p[1] - cy, p[0] - cx)
    return sorted(raw, key=polar)


def _corner_angles(
    corners: list[tuple[float, float]]
) -> list[float]:
    """Return the interior corner angles (in degrees) for a list of 4 corners."""
    n = len(corners)
    if n < 3:
        return []
    angles: list[float] = []
    for i in range(n):
        prev = corners[(i - 1 + n) % n]
        curr = corners[i]
        next_pt = corners[(i + 1) % n]
        v1x, v1y = prev[0] - curr[0], prev[1] - curr[1]
        v2x, v2y = next_pt[0] - curr[0], next_pt[1] - curr[1]
        dot = v1x * v2x + v1y * v2y
        mag = math.hypot(v1x, v1y) * math.hypot(v2x, v2y) + 1e-12
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))
        angles.append(angle)
    return angles


def _side_lengths(
    corners: list[tuple[float, float]]
) -> list[float]:
    """Return the lengths of the 4 sides between consecutive corners."""
    n = len(corners)
    if n < 2:
        return []
    lengths: list[float] = []
    for i in range(n):
        p1 = corners[i]
        p2 = corners[(i + 1) % n]
        lengths.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return lengths
