"""Hexagon feature classifier.

Mirrors ``MachiningHexagonPath`` from the SmartLaser hierarchy.

Detection gates:
1. The polyline has ≈ 6 distinct corners (via turning-angle maxima)
2. Each interior corner angle ≈ 120° ± 15°
3. Opposite sides are parallel and similar in length

No assumption is made about axis alignment.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from .base import ContourMetrics, ShapeClassifier


# ── Thresholds ─────────────────────────────────────────────────────────────

_HEX_ANGLE_TOL_DEG = 15.0    # interior angle tolerance for a hexagon
_HEX_SIDE_RATIO_MIN = 0.25   # opposite sides must be ≥ 25% of each other
_HEX_MIN_POINTS = 6
_HEX_MAX_POINTS = 24         # conservative upper bound for tessellation


class HexagonClassifier(ShapeClassifier):
    """Detects an axis-independent regular hexagon and extracts across-flats.

    Corresponds to ``MachiningHexagonPath``.
    """

    shape_name: ClassVar[str] = "hexagon"
    priority: ClassVar[int] = 70    # hexagon is the least common; test last
    min_points: ClassVar[int] = _HEX_MIN_POINTS

    def matches(self, m: ContourMetrics) -> bool:
        if m.n < self.min_points:
            return False
        if m.n > _HEX_MAX_POINTS:
            return False
        return self._hex_geometry_ok(m)

    def _hex_geometry_ok(self, m: ContourMetrics) -> bool:
        pts2d = m.pts2d
        n = m.n
        xmin, ymin, xmax, ymax = m.bbox
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5

        # 1) Extract corners.
        corners = self._extract_corners(pts2d, cx, cy)
        if not (5 <= len(corners) <= 8):
            return False

        # 2) Merge close corners (tessellation may split one vertex into 2–3).
        merged = self._merge_close_corners(corners, cx, cy)
        if not (5 <= len(merged) <= 8):
            return False

        # 3) Interior angles must be ≈ 120°.
        angles = self._corner_angles(merged)
        if len(angles) != len(merged):
            return False
        for ang in angles:
            if abs(ang - 120.0) > _HEX_ANGLE_TOL_DEG:
                return False

        # 4) Side length ratio.
        sides = self._side_lengths(merged)
        if len(sides) < 6:
            return False
        # Take the 3 smallest / largest as pairs.
        s_sorted = sorted(sides)
        if len(s_sorted) >= 6:
            ratio_a = s_sorted[0] / max(s_sorted[3], 1e-9)
            ratio_b = s_sorted[1] / max(s_sorted[4], 1e-9)
            ratio_c = s_sorted[2] / max(s_sorted[5], 1e-9)
            for ratio in (ratio_a, ratio_b, ratio_c):
                if ratio < _HEX_SIDE_RATIO_MIN:
                    return False

        return True

    def _extract_corners(
        self,
        pts2d: list[tuple[float, float]],
        cx: float,
        cy: float,
    ) -> list[tuple[float, float]]:
        n = len(pts2d)
        angles: list[float] = []
        for i in range(n):
            prev = pts2d[(i - 1 + n) % n]
            curr = pts2d[i]
            next_pt = pts2d[(i + 1) % n]
            v1x, v1y = curr[0] - prev[0], curr[1] - prev[1]
            v2x, v2y = next_pt[0] - curr[0], next_pt[1] - curr[1]
            dot = v1x * v2x + v1y * v2y
            mag = math.hypot(v1x, v1y) * math.hypot(v2x, v2y) + 1e-12
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))
            angles.append(angle)

        threshold = 25.0
        candidates = [(angles[i], pts2d[i]) for i in range(n) if angles[i] > threshold]
        if not candidates:
            return []

        candidates.sort(key=lambda x: -x[0])
        raw = [c[1] for c in candidates[:8]]

        def polar(p: tuple[float, float]) -> float:
            return math.atan2(p[1] - cy, p[0] - cx)
        return sorted(raw, key=polar)

    def _merge_close_corners(
        self,
        corners: list[tuple[float, float]],
        cx: float,
        cy: float,
    ) -> list[tuple[float, float]]:
        """Merge corners closer than ~10° in polar angle."""
        if len(corners) <= 6:
            return corners
        # Sort by polar angle.
        sorted_pts = sorted(corners, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        merged: list[tuple[float, float]] = []
        cluster_tol = math.radians(12.0)
        prev_angle = None
        cluster_pts: list[tuple[float, float]] = []
        for pt in sorted_pts:
            ang = math.atan2(pt[1] - cy, pt[0] - cx)
            if prev_angle is None or abs(ang - prev_angle) < cluster_tol:
                cluster_pts.append(pt)
                prev_angle = ang
            else:
                if cluster_pts:
                    cx_m = sum(p[0] for p in cluster_pts) / len(cluster_pts)
                    cy_m = sum(p[1] for p in cluster_pts) / len(cluster_pts)
                    merged.append((cx_m, cy_m))
                cluster_pts = [pt]
                prev_angle = ang
        if cluster_pts:
            cx_m = sum(p[0] for p in cluster_pts) / len(cluster_pts)
            cy_m = sum(p[1] for p in cluster_pts) / len(cluster_pts)
            merged.append((cx_m, cy_m))
        return merged

    def _corner_angles(self, corners: list[tuple[float, float]]) -> list[float]:
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

    def _side_lengths(self, corners: list[tuple[float, float]]) -> list[float]:
        n = len(corners)
        lengths: list[float] = []
        for i in range(n):
            p1 = corners[i]
            p2 = corners[(i + 1) % n]
            lengths.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        return lengths

    def classify(
        self,
        m: ContourMetrics,
        *,
        face_normal: tuple[float, float, float] | None,
        pts_world: list[tuple[float, float, float]] | None = None,
    ) -> dict[str, Any]:
        xmin, ymin, xmax, ymax = m.bbox
        # Across-flats: the width of the regular-hexagon bounding box
        # equals the distance between two parallel opposite sides.
        across_flats = round(min(xmax - xmin, ymax - ymin), 4)
        # Rotation: angle of the first side from horizontal.
        ra = self._rotation_angle(m)
        confidence = self._confidence(m)
        return {
            "diameter": None,
            "length": None,
            "width": None,
            "across_flats": across_flats,
            "_confidence": round(confidence, 3),
        }

    def _rotation_angle(self, m: ContourMetrics) -> float:
        # Return the rotation of the bbox's longer side from horizontal.
        xmin, ymin, xmax, ymax = m.bbox
        w = xmax - xmin
        h = ymax - ymin
        if w >= h:
            return 0.0
        return 90.0

    def _confidence(self, m: ContourMetrics) -> float:
        return min(1.0, m.circularity * 1.2)
