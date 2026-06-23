"""Circle feature classifier.

Mirrors ``MachiningCirclePath`` from the SmartLaser feature hierarchy:

* automatic diameter (``D``) extraction from the polyline area;
* two-pass detection (arc-fit residual + circularity fallback) so
  low-poly tessellation of STEP circles still classifies correctly;
* ellipse disambiguation kept as a *negative* gate: a high-aspect
  shape whose bbox-relative radius variation exceeds 10% is **not** a
  circle and falls through to the next classifier in the registry.

The thresholds below were tuned against the
``tests/fixtures/cad/plate_with_hole_100.step`` fixture (10mm circle
sampled at ``linear_deflection=0.1`` → ≈36–72 segments).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from .base import ContourMetrics, ShapeClassifier


# ── Thresholds ───────────────────────────────────────────────────────────────

_ARC_FIT_REL_TOL = 0.02  # very tight — only true circles pass


_CIRCULARITY_CIRCLE = 0.90      # require near-perfect circularity
_CIRCULARITY_CIRCLE_FALLBACK = 0.70


_ELLIPSE_CV_THRESHOLD = 0.10
_ELLIPSE_MIN_ASPECT = 1.05
_ELLIPSE_MAX_ASPECT = 8.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fit_circle_2d(pts2d: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Least-squares-style circle fit via circumcentre candidate.

    Same approach as the legacy ``_fit_circle_2d`` (kept verbatim so the
    unit-tested tolerance numbers don't shift). Returns ``(cx, cy, r)``.
    """
    n = len(pts2d)
    if n < 3:
        return None

    def _circumcenter(ax, ay, bx, by, cx, cy):
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            return None
        ax2, ay2 = ax * ax, ay * ay
        bx2, by2 = bx * bx, by * by
        cx2, cy2 = cx * cx, cy * cy
        ux = ((ax2 + ay2) * (by - cy) + (bx2 + by2) * (cy - ay) + (cx2 + cy2) * (ay - by)) / d
        uy = ((ax2 + ay2) * (cx - bx) + (bx2 + by2) * (ax - cx) + (cx2 + cy2) * (bx - ay)) / d
        return ux, uy

    candidates: list[tuple[float, float]] = []
    for i in [0, n // 3, 2 * n // 3]:
        a = pts2d[i]
        b = pts2d[(i + 1) % n]
        c = pts2d[(i + 2) % n]
        cc = _circumcenter(a[0], a[1], b[0], b[1], c[0], c[1])
        if cc is not None:
            candidates.append(cc)
    if not candidates:
        return None

    best: tuple[float, float, float] | None = None
    best_var = float("inf")
    for cx, cy in candidates:
        radii = [math.hypot(p[0] - cx, p[1] - cy) for p in pts2d]
        r_mean = sum(radii) / len(radii)
        var = sum((r - r_mean) ** 2 for r in radii) / (r_mean * r_mean + 1e-12)

        # Validate: radius should be in a reasonable range
        # Use bbox to determine expected size
        xs = [p[0] for p in pts2d]
        ys = [p[1] for p in pts2d]
        bbox_diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        expected_size = bbox_diag / 2

        # Radius should be within 0.1x to 5x of expected size
        if r_mean < expected_size * 0.1 or r_mean > expected_size * 5:
            continue
        # Center should be within reasonable distance of bbox center
        bbox_cx = (min(xs) + max(xs)) / 2
        bbox_cy = (min(ys) + max(ys)) / 2
        center_dist = math.hypot(cx - bbox_cx, cy - bbox_cy)
        if center_dist > expected_size * 2:
            continue

        if var < best_var:
            best_var = var
            best = (cx, cy, r_mean)
    return best


def _is_ellipse(m: ContourMetrics) -> bool:
    """Heuristic check: a high-circularity polyline that's actually an ellipse.

    A circle has constant distance from its centre to the boundary; an
    ellipse doesn't. We use the coefficient of variation of
    point-to-bbox-centre distances, which is < 0.05 for circles and
    > 0.10 for ellipses with aspect > 1.05.
    """
    if m.n < 8 or m.aspect < _ELLIPSE_MIN_ASPECT:
        return False
    if m.aspect > _ELLIPSE_MAX_ASPECT:
        return False
    xmin, ymin, xmax, ymax = m.bbox
    bx = (xmin + xmax) * 0.5
    by = (ymin + ymax) * 0.5
    dists = [math.hypot(p[0] - bx, p[1] - by) for p in m.pts2d]
    mean_d = sum(dists) / m.n
    if mean_d < 1e-9:
        return False
    var = sum((d - mean_d) ** 2 for d in dists) / m.n
    std = math.sqrt(var)
    cv = std / mean_d
    return cv > _ELLIPSE_CV_THRESHOLD


# ── Classifier ───────────────────────────────────────────────────────────────


class CircleClassifier(ShapeClassifier):
    """Detects a circle hole / boss and extracts its diameter.

    Equivalent in role to ``MachiningCirclePath`` — owns the single
    shape-specific parameter ``D`` (and inherits the LS ``*_STANDARD``
    family: V / CWCCW / VD / VC1..4 / Lead / OA / Power / Duty / TNT,
    which are left to the LS planner downstream).
    """

    shape_name: ClassVar[str] = "circle"
    priority: ClassVar[int] = 100   # circle is the most "round" shape; test first
    min_points: ClassVar[int] = 12  # must have enough points for meaningful arc-fit

    def matches(self, m: ContourMetrics) -> bool:
        if m.n < self.min_points or m.area_2d <= 0:
            return False
        # Reject axis-stretched shapes: a circle has bbox aspect ≈ 1.0.
        # Shapes with bbox aspect > 1.2 or < 0.83 are NOT circles
        # regardless of circularity — the CircleClassifier must not
        # cannibalise slot/rectangle holes.
        if m.aspect > 1.2 or m.aspect < 0.83:
            return False
        # 1) arc-fit pass
        fit = _fit_circle_2d(m.pts2d)
        if fit is not None:
            cx, cy, radius = fit
            if radius > 1e-9:
                radii = [math.hypot(p[0] - cx, p[1] - cy) for p in m.pts2d]
                mean_r = sum(radii) / len(radii)
                rel = sum(abs(r - mean_r) for r in radii) / (len(radii) * mean_r)
                if rel < _ARC_FIT_REL_TOL:
                    return True
        # 2) circularity fallback
        return m.circularity >= _CIRCULARITY_CIRCLE

    def classify(
        self,
        m: ContourMetrics,
        *,
        face_normal: tuple[float, float, float] | None,
        pts_world: list[tuple[float, float, float]] | None = None,
    ) -> dict[str, Any]:
        # Diameter: arc-fit (preferred) or area-as-circle (robust fallback).
        diameter: float
        area_diameter = 2.0 * math.sqrt(m.area_2d / math.pi) if m.area_2d > 0 else 0.0
        fit = _fit_circle_2d(m.pts2d)
        if fit is not None:
            fit_diameter = 2.0 * fit[2]
            # Sanity guard: if the arc-fit diameter is more than 10x larger
            # than the area diameter, the tessellation is degenerate and
            # _fit_circle_2d produced a nonsensical minimum-enclosing-circle result.
            # Fall back to the area-based diameter (stable regardless of
            # how many polygon facets the STEP file used to approximate the circle).
            if area_diameter > 0 and fit_diameter / area_diameter > 10.0:
                diameter = area_diameter
            else:
                diameter = fit_diameter
        else:
            diameter = area_diameter

        # Confidence: combine arc-fit residual and circularity.
        confidence = 0.5
        if fit is not None:
            cx, cy, radius = fit
            if radius > 1e-9:
                radii = [math.hypot(p[0] - cx, p[1] - cy) for p in m.pts2d]
                mean_r = sum(radii) / len(radii)
                if mean_r > 1e-9:
                    rel = sum(abs(r - mean_r) for r in radii) / (len(radii) * mean_r)
                    rel_conf = max(0.0, min(1.0, 1.0 - rel / 0.10))
                    confidence = rel_conf * 0.5 + m.circularity * 0.5
        else:
            confidence = m.circularity

        return {
            "diameter": round(diameter, 4),
            "length": None,
            "width": None,
            "across_flats": None,
            "_confidence": round(confidence, 3),
        }
