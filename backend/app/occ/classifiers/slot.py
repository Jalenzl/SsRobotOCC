"""Slot (obround) feature classifier.

Mirrors ``MachiningSlotPath`` from the SmartLaser hierarchy:
two semi-circular caps connected by straight parallel sides.

Three-gate detection:
1. aspect >= 2.2  (long enough to be a slot, not a circle)
2. end-cap arc residual < 12% of short-radius²
3. middle section deviation from the long axis ≈ 0
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from .base import ContourMetrics, ShapeClassifier


# ── Thresholds ─────────────────────────────────────────────────────────────

_SLOT_ASPECT_MIN = 2.2
_ARC_FIT_REL_TOL = 0.12
_SLOT_CIRCULARITY_REJECT = 0.96   # obround / capsule is NOT a slot


class SlotClassifier(ShapeClassifier):
    """Detects an obround / slot and extracts L / W / RA.

    Corresponds to ``MachiningSlotPath``.
    """

    shape_name: ClassVar[str] = "slot"
    priority: ClassVar[int] = 90    # after circle, before rectangle
    min_points: ClassVar[int] = 8

    def matches(self, m: ContourMetrics) -> bool:
        if m.n < self.min_points:
            return False
        if m.aspect < _SLOT_ASPECT_MIN:
            return False
        # A very round silhouette (circularity > 0.96) is a near-circle,
        # not a slot.
        if m.circularity > _SLOT_CIRCULARITY_REJECT:
            return False
        return self._slot_geometry_ok(m)

    def _slot_geometry_ok(self, m: ContourMetrics) -> bool:
        """Run the three geometric gates."""
        pts2d = m.pts2d
        n = m.n
        xmin, ymin, xmax, ymax = m.bbox
        long_axis_len = m.length
        short_axis_len = max(m.short_side, 1e-9)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        horizontal = (xmax - xmin) >= (ymax - ymin)

        # Project each point onto the long axis (parameter t ∈ [0,1])
        # and record its perpendicular distance from the axis.
        if horizontal:
            ts = [(p[0] - xmin) / long_axis_len for p in pts2d]
            ds = [p[1] - cy for p in pts2d]
        else:
            ts = [(p[1] - ymin) / long_axis_len for p in pts2d]
            ds = [p[0] - cx for p in pts2d]

        edge_frac = (short_axis_len * 0.5) / long_axis_len
        near_left = [(t, d) for t, d in zip(ts, ds) if t <= edge_frac + 0.02]
        near_right = [(t, d) for t, d in zip(ts, ds) if t >= 1.0 - edge_frac - 0.02]
        middle = [(t, d) for t, d in zip(ts, ds)
                   if edge_frac < t < 1.0 - edge_frac]

        if len(near_left) < 4 or len(near_right) < 4 or len(middle) < 4:
            return False

        # End-cap arc centres.
        if horizontal:
            lcx, lcy = xmin + short_axis_len * 0.5, cy
            rcx, rcy = xmax - short_axis_len * 0.5, cy
        else:
            lcx, lcy = cx, ymin + short_axis_len * 0.5
            rcx, rcy = cx, ymax - short_axis_len * 0.5

        # Radial residual for each cap.
        r_cap = short_axis_len * 0.5
        rad_l = [
            abs((p[0] - lcx) ** 2 + (p[1] - lcy) ** 2 - r_cap ** 2)
            for t, p in zip(ts, pts2d) if t <= edge_frac + 0.02
        ]
        rad_r = [
            abs((p[0] - rcx) ** 2 + (p[1] - rcy) ** 2 - r_cap ** 2)
            for t, p in zip(ts, pts2d) if t >= 1.0 - edge_frac - 0.02
        ]
        if not rad_l or not rad_r:
            return False
        err_l = sum(rad_l) / len(rad_l)
        err_r = sum(rad_r) / len(rad_r)
        err = (err_l + err_r) * 0.5
        tol = (short_axis_len * 0.5) ** 2 * _ARC_FIT_REL_TOL
        if err > tol:
            return False

        # Middle section must be close to the long axis.
        mid_ds = [abs(d) for _, d in middle]
        if not mid_ds:
            return False
        if max(mid_ds) > short_axis_len * 0.5 * 0.06:
            return False

        return True

    def classify(
        self,
        m: ContourMetrics,
        *,
        face_normal: tuple[float, float, float] | None,
        pts_world: list[tuple[float, float, float]] | None = None,
    ) -> dict[str, Any]:
        length = round(m.length, 4)
        width = round(m.short_side, 4)
        # Rotation angle: angle from the longest bbox axis to the X axis.
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
        """Return the angle (0–90°) of the longest axis from the X axis."""
        xmin, ymin, xmax, ymax = m.bbox
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        max_dist = 0.0
        farthest_angle = 0.0
        for p in m.pts2d:
            dx = p[0] - cx
            dy = p[1] - cy
            dist = math.hypot(dx, dy)
            if dist > max_dist:
                max_dist = dist
                farthest_angle = math.degrees(math.atan2(dy, dx))
        angle = abs(farthest_angle % 90)
        if angle > 45:
            angle = 90 - angle
        return angle

    def _confidence(self, m: ContourMetrics) -> float:
        # Aspect ratio is the strongest signal for a slot.
        aspect_score = min(1.0, m.aspect / 4.0)
        # Circularity complement helps distinguish from circles.
        circ_score = 1.0 - m.circularity
        return 0.3 + 0.35 * aspect_score + 0.35 * circ_score
