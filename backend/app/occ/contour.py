"""Face-level feature recognition (wires → polylines → contours → holes).

Pure algorithm module. All primitives are reused from existing OCC helpers:

- ``app.occ.geometry_utils``  →  face_surface_info / face_outward_normal /
                                  face_wires / face_area / work_plane_normal
- ``app.occ.discretize``       →  wire_to_polyline / wire_length /
                                  wire_area_if_planar / wire_location_on_face
- ``app.occ.topology``         →  face_point_and_outward_normal (sampling)
- ``app.occ.loader``           →  read_step_bytes

No HTTP / Pydantic coupling. Inputs are pythonOCC ``TopoDS_Face`` instances;
outputs are plain ``dict`` (JSON-serialisable) so they can flow through
``app.services.feature_service`` and ``app.routers.feature`` unchanged.
"""

from __future__ import annotations

import math
from typing import Any

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape

from app.occ.discretize import (
    wire_area_if_planar,
    wire_length,
    wire_location_on_face,
    wire_to_polyline,
)
from app.occ.geometry_utils import (
    face_area,
    face_outward_normal,
    face_surface_info,
    face_wires,
    work_plane_normal,
)

# Lazy import for enhanced features (laser-software-style parameters)
_contour_enhanced = None


def _get_enhanced_module():
    """Lazy load enhanced module to avoid circular imports."""
    global _contour_enhanced
    if _contour_enhanced is None:
        try:
            from app.occ import contour_enhanced

            _contour_enhanced = contour_enhanced
        except ImportError:
            _contour_enhanced = False
    return _contour_enhanced


# Debug mode: set to True to include classification diagnosis
_DEBUG_CLASSIFICATION = True


# ── Contour-type thresholds; tuned against plate-with-hole / slotted-plate
# STEP fixtures (see tests/test_feature.py).

# Circle detection: circularity >= this → circle.
# Lowered from 0.90 to 0.75 to handle low-poly tessellation of STEP files
# (linear_deflection=0.1 produces ~36-72 segments for a 10mm circle,
# circularity ≈ 0.78-0.85). The arc-based fallback handles true circles
# that fail this test due to tessellation artifacts.
_CIRCULARITY_CIRCLE = 0.75       # >= 0.75  → try circle
_CIRCULARITY_CIRCLE_FALLBACK = 0.70  # fallback: also try circle if >= this
_CIRCULARITY_SLOT_MAX = 0.88    # <= 0.88 才允许判 slot（放宽以减少漏检）
_SLOT_ASPECT_MIN = 2.2          # 长宽比 ≥ 2.2 才考虑 slot（降低以捕获更多槽型）
_CLOSURE_TOL_DEFAULT = 0.5      # mm, default; overridable per call
# Width/length ratio threshold for treating a non-square contour as slot/obround
# (very thin long shapes look more like slots than rectangles).
_SLOT_WLR_MIN = 0.04            # min width / length
_SLOT_WLR_MAX = 0.55            # max width / length
_HEX_ANGLE_TOL_DEG = 6.0
# 轮廓最小面积阈值（mm²），小于此值的视为表面刻痕/点状噪声，不参与特征识别
_MIN_CONTOUR_AREA_MM2 = 1.0     # ≈ 1 mm² 的圆直径 ≈ 1.1mm
_MIN_IRREGULAR_AREA_MM2 = 10.0  # 异形孔最小面积，过小则为噪声
# Arc fitting tolerance relative to short-axis (used in slot cap-arc detection)
_ARC_FIT_REL_TOL = 0.12        # 12% of short radius² for radial residual mean


# ── Helpers ────────────────────────────────────────────────────────────────
def _vec(p: Any) -> list[float] | None:
    if p is None:
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def _pt(p: Any) -> list[float] | None:
    if p is None:
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def _dist(a, b) -> float:
    """Euclidean distance; accepts either 2D (x, y) or 3D (x, y, z) tuples.
    Some legacy callers (e.g. ``_looks_like_rectangle`` working on
    projected 2D points) feed 2D tuples into helpers that were originally
    written for 3D — pad with 0.0 to avoid IndexError.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if len(a) >= 3 and len(b) >= 3:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_length(pts: list[tuple[float, float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(pts)):
        total += _dist(pts[i - 1], pts[i])
    return total


def _polyline_area_2d(
    pts: list[tuple[float, float, float]],
    normal: tuple[float, float, float] | None,
) -> float:
    """Shoelace area of a closed polyline in its dominant 2D plane.

    BRepGProp.SurfaceProperties on a wire often returns 0, so we fall back
    to 2D shoelace using the same projection as ``_project_to_2d``.
    """
    if len(pts) < 3:
        return 0.0
    pts2d = _project_to_2d(pts, normal or (0.0, 0.0, 1.0))
    n = len(pts2d)
    s = 0.0
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _closed(pts, tol: float) -> bool:
    return len(pts) >= 3 and _dist(pts[0], pts[-1]) <= tol


def _project_to_2d(
    pts: list[tuple[float, float, float]],
    normal: tuple[float, float, float],
) -> list[tuple[float, float]]:
    """Drop the dominant-axis coordinate. Matches geometry_utils.project_point
    (the legacy helper) so contour 2D math is consistent with work-plane code.
    """
    nx, ny, nz = normal
    if abs(nz) >= max(abs(nx), abs(ny)):
        return [(p[0], p[1]) for p in pts]
    if abs(ny) >= abs(nx):
        return [(p[0], p[2]) for p in pts]
    return [(p[1], p[2]) for p in pts]


# ── Arc-fitting helpers for robust shape classification ──────────────────
def _arc_residual(
    pts: list[tuple[float, float]],
    cx: float,
    cy: float,
    radius: float,
) -> float:
    """Mean squared radial residual of points from a fitted arc."""
    if radius <= 0 or not pts:
        return float("inf")
    return sum(
        ((p[0] - cx) ** 2 + (p[1] - cy) ** 2 - radius ** 2) ** 2
        for p in pts
    ) / len(pts)


def _fit_circle_2d(
    pts: list[tuple[float, float]],
) -> tuple[float, float, float] | None:
    """Least-squares circle fit (algebraic, using circumcenter of 3-point pairs).

    Returns (cx, cy, radius) or None if fitting fails.
    """
    n = len(pts)
    if n < 3:
        return None

    # Use the circumcenter of 3 well-separated points as initial guess.
    # For roughly circular data this is close to the true centre.
    def _circumcenter(
        ax: float, ay: float, bx: float, by: float, cx: float, cy: float
    ) -> tuple[float, float] | None:
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            return None
        ax2, ay2 = ax * ax, ay * ay
        bx2, by2 = bx * bx, by * by
        cx2, cy2 = cx * cx, cy * cy
        ux = ((ax2 + ay2) * (by - cy) + (bx2 + by2) * (cy - ay) + (cx2 + cy2) * (ay - by)) / d
        uy = ((ax2 + ay2) * (cx - bx) + (bx2 + by2) * (ax - cx) + (cx2 + cy2) * (bx - ay)) / d
        return ux, uy

    # Try 3 evenly-spaced points as candidate circumcenters
    candidates: list[tuple[float, float]] = []
    for i in [0, n // 3, 2 * n // 3]:
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cc = _circumcenter(a[0], a[1], b[0], b[1], c[0], c[1])
        if cc:
            candidates.append(cc)

    if not candidates:
        return None

    # Pick the candidate whose radius is most consistent across all points
    best: tuple[float, float, float] | None = None
    best_var = float("inf")
    for cx, cy in candidates:
        radii = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
        r_mean = sum(radii) / len(radii)
        # Radius variance normalised by mean² — lower = more circular
        var = sum((r - r_mean) ** 2 for r in radii) / (r_mean ** 2 + 1e-12)
        if var < best_var:
            best_var = var
            best = (cx, cy, r_mean)
    return best


def _classification_confidence(
    pts2d: list[tuple[float, float]],
    circularity: float,
    ctype: str,
    radius: float | None,
) -> float:
    """Return a confidence score [0.0, 1.0] for the contour type.

    Uses arc-fit residual for circles and aspect/wl ratio for slots/rectangles.
    """
    if ctype == "outer":
        return 1.0
    if ctype == "unknown":
        return 0.0
    if ctype == "circle":
        if radius is None or radius <= 0:
            return 0.3
        # Check arc-fit residual: how well do points fit a circle?
        fit = _fit_circle_2d(pts2d)
        if fit is None:
            return 0.4
        cx, cy, r = fit
        residuals = [abs(math.hypot(p[0] - cx, p[1] - cy) - r) / max(r, 1e-9) for p in pts2d]
        max_rel_err = max(residuals)
        # relative error < 2% → very confident; > 10% → low confidence
        confidence = max(0.0, min(1.0, 1.0 - max_rel_err / 0.10))
        # boost by circularity (the harder the tessellation, the lower circ)
        confidence = confidence * 0.5 + circularity * 0.5
        return round(confidence, 3)
    if ctype == "slot":
        # Based on how well the cap arcs and mid-section conform
        # Heuristic: use aspect and circularity as proxy
        aspect = (max(p[0] for p in pts2d) - min(p[0] for p in pts2d)) / max(
            max(p[1] for p in pts2d) - min(p[1] for p in pts2d), 1e-9
        )
        # Very high aspect (>3) is more confidently a slot
        conf = min(1.0, aspect / 4.0)
        conf = conf * 0.6 + (1.0 - circularity) * 0.4
        return round(conf, 3)
    if ctype == "rectangle":
        # Based on how close to 90° corners and square-ish aspect
        xs = [p[0] for p in pts2d]
        ys = [p[1] for p in pts2d]
        aspect = max(xs) - min(xs) / max(max(ys) - min(ys), 1e-9)
        # More square-like → higher confidence
        conf = 1.0 - min(1.0, abs(1.0 - aspect) / 2.0)
        return round(max(0.3, min(0.95, conf)), 3)
    if ctype == "hexagon":
        return 0.75
    if ctype == "irregular":
        return 0.4
    return 0.3


# ── Public API: contour classification ────────────────────────────────────
def classify_wire_contour(
    pts_world: list[tuple[float, float, float]],
    face_normal: tuple[float, float, float] | None,
    is_outer: bool,
    wire_id: str,
    polyline_id: str,
    face_id: str,
    contour_index: int,
    prefer_pca_plane: bool = False,
    enhanced_params: bool = False,
) -> dict:
    """Classify one closed polyline as outer / circle / slot / rectangle /
    hexagon / unknown. Returns a plain dict that matches the
    ``ContourFeature`` pydantic schema (subset of fields).

    Args:
        enhanced_params: If True, extract laser-software-style parameters
            (rotation_angle, corner_radius, compensation_length, etc.)
    """
    cid = f"contour_{contour_index}"

    if not pts_world or len(pts_world) < 4:
        return _unknown_contour(cid, wire_id, polyline_id, face_id, is_outer)
    pts2d = _project_to_2d(pts_world, face_normal or (0.0, 0.0, 1.0))
    pts2d = _unique_ordered(pts2d)
    n = len(pts2d)
    if n < 4:
        return _unknown_contour(cid, wire_id, polyline_id, face_id, is_outer)

    # bounding box
    xs = [p[0] for p in pts2d]
    ys = [p[1] for p in pts2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = xmax - xmin
    height = ymax - ymin
    length = max(width, height)
    width_ = min(width, height)
    aspect = length / max(width_, 1e-9)
    span = max(length, 1e-9)

    # shoelace area (signed); use abs for circularity
    s = 0.0
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    area_2d = abs(s) * 0.5

    # Filter tiny surface marks (etching / dot patterns) — these are noise,
    # not real holes or slots. A 1mm-diameter circle ≈ 0.79mm²; we use 1.0mm².
    if area_2d < _MIN_CONTOUR_AREA_MM2:
        # Debug log: surface-mark filter rejection
        if _DEBUG_CLASSIFICATION and not is_outer:
            import logging
            logging.getLogger("contour").debug(
                "area_filter: cid=%s area=%.4f mm² < %.2f mm² (skipped as noise)",
                cid, area_2d, _MIN_CONTOUR_AREA_MM2,
            )
        return _unknown_contour(cid, wire_id, polyline_id, face_id, is_outer)
    # closed-loop perimeter: sum every segment INCLUDING the wrap from
    # the last point back to the first. (_polyline_length was originally
    # open-path only — that underestimated perim and overestimated
    # circularity, which made irregular shapes look like circles.)
    if n >= 2:
        perimeter = sum(
            _dist(pts_world[i], pts_world[(i + 1) % n]) for i in range(n)
        )
    else:
        perimeter = 0.0
    circularity = (4 * math.pi * area_2d) / (perimeter * perimeter) if perimeter > 0 else 0.0

    # Centroid via 2D shoelace (polygon centroid, robust for any n)
    if n >= 3 and s != 0:
        cx_c = 0.0
        cy_c = 0.0
        for i in range(n):
            x1, y1 = pts2d[i]
            x2, y2 = pts2d[i + 1] if i + 1 < n else pts2d[0]
            cx_c += (x1 + x2) * (x1 * y2 - x2 * y1)
            cy_c += (y1 + y2) * (x1 * y2 - x2 * y1)
        cx_c /= (3 * s)
        cy_c /= (3 * s)
    else:
        cx_c, cy_c = 0.0, 0.0

    # 3D centroid: use the **3D centroid of the wire points**, which
    # is guaranteed to lie on the face plane (the wire is on the
    # plane). Mapping the 2D shoelace centroid back to 3D with
    # ``axis_mean`` shortcuts breaks for faces whose normal isn't
    # axis-aligned (e.g. tilted / rotated planar faces) — the result
    # floats off the surface. Using 3D centroid avoids that entirely.
    if pts_world:
        cx_3d = sum(p[0] for p in pts_world) / len(pts_world)
        cy_3d = sum(p[1] for p in pts_world) / len(pts_world)
        cz_3d = sum(p[2] for p in pts_world) / len(pts_world)
        center_3d: list[float] | None = [float(cx_3d), float(cy_3d), float(cz_3d)]
    else:
        center_3d = None
    normal_3d = _vec(face_normal) if face_normal else None

    params = {
        "diameter": None,
        "length": None,
        "width": None,
        "across_flats": None,
    }

    # 1) outer wins — the face boundary always wins the "outer" label
    if is_outer:
        ctype = "outer"
        confidence = 1.0
    # 2) obround / slot: thin & long with cap arcs at both ends
    elif _looks_like_slot(pts2d, aspect, circularity):
        ctype = "slot"
        params["length"] = round(length, 4)
        params["width"] = round(width_, 4)
        confidence = _classification_confidence(pts2d, circularity, ctype, None)
    # 3) circle — try arc fitting first (handles low-poly tessellation)
    elif _try_circle(pts2d, circularity):
        # 3a) ellipse check: the arc-fit residual can be < 12% for a
        #     nearly-closed polyline of an ellipse too (since the
        #     sample points all sit near some "best fit circle"). The
        #     disambiguator is the bounding-box aspect ratio. A true
        #     circle has aspect ≈ 1.0; an ellipse with the same
        #     circularity as a polyline already excludes itself if
        #     aspect > 1.25 (or so).
        if 1.05 < aspect < 8.0 and _is_ellipse_not_circle(pts2d, aspect):
            ctype = "ellipse"
            params["length"] = round(length, 4)
            params["width"] = round(width_, 4)
            confidence = _classification_confidence(pts2d, circularity, ctype, None)
        else:
            ctype = "circle"
            diameter = 2.0 * math.sqrt(area_2d / math.pi)
            params["diameter"] = round(diameter, 4)
            # Also compute from arc-fit if available
            fit = _fit_circle_2d(pts2d)
            if fit:
                params["diameter"] = round(2.0 * fit[2], 4)
            confidence = _classification_confidence(pts2d, circularity, ctype, params["diameter"])
    # 4) hexagon
    elif _looks_like_hexagon(pts2d):
        ctype = "hexagon"
        params["across_flats"] = round(2.0 * area_2d / max(_hex_side(pts2d) * 1.5, 1e-9), 4)
        confidence = 0.75
    # 5) rectangle: 4 corners, edges aligned to bbox axes, widths close to bbox edges
    elif _looks_like_rectangle(pts2d, span):
        ctype = "rectangle"
        params["length"] = round(length, 4)
        params["width"] = round(width_, 4)
        confidence = _classification_confidence(pts2d, circularity, ctype, None)
    # 6) irregular: any other closed polyline with >= 4 vertices that
    #    isn't noise. This is the explicit "I don't know what this is but
    #    it's a real closed feature" bucket — distinct from the
    #    "couldn't classify" `unknown` bucket used for degenerate inputs
    #    (n < 4, NaN, etc.). Frontend renders it as a generic hole with
    #    a star / question-mark icon.
    #    Require minimum area to suppress tiny surface marks that failed the
    #    initial gate (e.g. complex compound faces with many small loops).
    elif n >= 4 and area_2d >= _MIN_IRREGULAR_AREA_MM2 and circularity < 0.98 and aspect < 12.0:
        ctype = "irregular"
        params["length"] = round(length, 4)
        params["width"] = round(width_, 4)
        confidence = 0.4
    # 7) fall-back: low-circularity non-rectangular blob → unknown
    else:
        ctype = "unknown"
        confidence = 0.0

    # contour_role: clear semantic role for the frontend
    contour_role = "outer_boundary" if is_outer else "inner_hole"

    # Build result
    result = {
        "id": cid,
        "contour_type": ctype,
        "contour_role": contour_role,
        "center": center_3d,
        "normal": normal_3d,
        "polyline_id": polyline_id,
        "wire_id": wire_id,
        "face_id": face_id,
        "is_outer": is_outer,
        "parameters": params,
        "area": round(area_2d, 4),
        "perimeter": round(perimeter, 4),
        "confidence": confidence,
    }

    # Enhanced parameters (laser-software-style)
    if enhanced_params:
        enhanced = _get_enhanced_module()
        if enhanced:
            # Extract enhanced parameters
            enhanced_params_dict = enhanced.extract_contour_parameters(
                pts2d, ctype, circularity, perimeter
            )
            # Merge into existing parameters
            for key, value in enhanced_params_dict.items():
                if value is not None:
                    result["parameters"][key] = value

            # Recalculate confidence with enhanced scoring
            result["confidence"] = enhanced.calculate_classification_confidence(
                pts2d, circularity, ctype, result["parameters"]
            )

            # Add validation status
            is_valid, error_msg = enhanced.validate_contour_parameters(
                result["parameters"], ctype
            )
            result["validation"] = {
                "is_valid": is_valid,
                "error": error_msg,
            }

            # Estimate lead length
            lead_length = enhanced.estimate_lead_length(ctype, result["parameters"])
            result["lead_length"] = round(lead_length, 4)

            # Add diagnosis for classification issues
            if _DEBUG_CLASSIFICATION or ctype in ("unknown", "irregular"):
                diagnosis = enhanced.diagnose_classification(pts2d, is_outer)
                result["_diagnosis"] = diagnosis

    return result


# ── Circle classification with arc fitting ─────────────────────────────────
def _try_circle(
    pts2d: list[tuple[float, float]],
    circularity: float,
) -> bool:
    """Determine if pts2d represent a circle, using arc fitting as primary.

    Handles low-poly tessellation by fitting a circle to the points and
    checking the relative residual. Also accepts high-circularity shapes
    as a secondary signal.

    Two-pass strategy:
    1. Arc-fit: fit a circle, check relative radial residual < 12%.
    2. Fallback: if residual fails, accept based on circularity threshold.
    """
    fit = _fit_circle_2d(pts2d)
    if fit:
        cx, cy, radius = fit
        # Relative radial residual: mean |r_i - r_mean| / r_mean
        radii = [math.hypot(p[0] - cx, p[1] - cy) for p in pts2d]
        r_mean = sum(radii) / len(radii)
        if r_mean > 1e-9:
            rel_residual = sum(abs(r - r_mean) for r in radii) / (len(radii) * r_mean)
        else:
            rel_residual = float("inf")
        # 12% relative residual is the pass threshold
        if rel_residual < 0.12:
            return True
        # Also pass if circularity is high enough even with bad arc-fit
        # (this handles cases where tessellation creates near-perfect
        # circularity but the circumcenter method is off)
        if circularity >= _CIRCULARITY_CIRCLE:
            return True
        return False
    # No fit → fall back to raw circularity
    return circularity >= _CIRCULARITY_CIRCLE_FALLBACK


def _is_ellipse_not_circle(
    pts2d: list[tuple[float, float]],
    aspect: float,
) -> bool:
    """Disambiguate a high-circularity polyline: is it a circle or an ellipse?

    An ellipse can produce a polyline with circularity close to 1 (the
    shape's 4π·A / P² is the same regardless of aspect, but the
    "circularity" computed here is for the 2D bounding-box approximation
    and can be similar). The key signature: an ellipse's points have
    **varying distance to the centroid** between the major and minor
    axes, while a circle's points are equidistant.

    Algorithm:
    1. Compute the 2D bounding box and its centre (bx, by).
    2. For each polyline point, compute distance to bbox centre.
    3. The coefficient of variation (std/mean) of these distances:
       - circle: < 0.05
       - ellipse: > 0.10
    """
    n = len(pts2d)
    if n < 8 or aspect < 1.05:
        return False
    xs = [p[0] for p in pts2d]
    ys = [p[1] for p in pts2d]
    bx = 0.5 * (min(xs) + max(xs))
    by = 0.5 * (min(ys) + max(ys))
    dists = [math.hypot(p[0] - bx, p[1] - by) for p in pts2d]
    mean_d = sum(dists) / n
    if mean_d < 1e-9:
        return False
    var = sum((d - mean_d) ** 2 for d in dists) / n
    std_d = math.sqrt(var)
    cv = std_d / mean_d
    # 0.10 threshold: ellipses of aspect > 1.05 produce cv > 0.10,
    # true circles produce cv < 0.05.
    return cv > 0.10


# ── Slot / hexagon heuristics ─────────────────────────────────────────────
def _looks_like_slot(
    pts2d: list[tuple[float, float]],
    aspect: float,
    circularity: float,
) -> bool:
    """Obround / slot shape: 两端半圆 + 中间直线；不要求圆度极高。

    三道关卡：
    1. 长宽比足够细长
    2. 端点附近"圆弧"段径向残差小
    3. 中段近似直线
    """
    if aspect < _SLOT_ASPECT_MIN:
        return False
    # 高圆度的细长形仍可能是非常宽头短尾的 obround → 允许通过
    if circularity > 0.96:
        # 圆度过高反而不是槽（短粗 = 圆孔）
        return False

    n = len(pts2d)
    xs = [p[0] for p in pts2d]
    ys = [p[1] for p in pts2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    long_axis_len = max(xmax - xmin, ymax - ymin)
    short_axis_len = max(min(xmax - xmin, ymax - ymin), 1e-9)

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    horizontal = (xmax - xmin) >= (ymax - ymin)

    # 把点列沿长轴投影为 (t, d) (参数 t∈[0,1], d 离长轴距离)
    if horizontal:
        ts = [(p[0] - xmin) / long_axis_len for p in pts2d]
        ds = [p[1] - cy for p in pts2d]
        # 两端半圆区域：t ∈ [0, edge_frac] ∪ [1-edge_frac, 1]
        # 中心点：(cx, cy)
        # 端部圆心：(xmin + short_axis_len/2, cy) / (xmax - short_axis_len/2, cy)
    else:
        ts = [(p[1] - ymin) / long_axis_len for p in pts2d]
        ds = [p[0] - cx for p in pts2d]

    edge_frac = (short_axis_len * 0.5) / long_axis_len
    near_left = [(t, d) for t, d in zip(ts, ds) if t <= edge_frac + 0.02]
    near_right = [(t, d) for t, d in zip(ts, ds) if t >= 1.0 - edge_frac - 0.02]
    middle = [(t, d) for t, d in zip(ts, ds) if edge_frac < t < 1.0 - edge_frac]

    if len(near_left) < 4 or len(near_right) < 4 or len(middle) < 4:
        return False

    # 端部圆心
    if horizontal:
        lcx, lcy = xmin + short_axis_len * 0.5, cy
        rcx, rcy = xmax - short_axis_len * 0.5, cy
        # d = p[1] - cy
        radial_l = [abs((p[0] - lcx) ** 2 + (p[1] - lcy) ** 2 - (short_axis_len * 0.5) ** 2)
                    for t, p in zip(ts, [(p[0], p[1]) for p in pts2d]) if t <= edge_frac + 0.02]
        radial_r = [abs((p[0] - rcx) ** 2 + (p[1] - rcy) ** 2 - (short_axis_len * 0.5) ** 2)
                    for t, p in zip(ts, [(p[0], p[1]) for p in pts2d]) if t >= 1.0 - edge_frac - 0.02]
    else:
        lcx, lcy = cx, ymin + short_axis_len * 0.5
        rcx, rcy = cx, ymax - short_axis_len * 0.5
        radial_l = [abs((p[1] - lcy) ** 2 + (p[0] - lcx) ** 2 - (short_axis_len * 0.5) ** 2)
                    for t, p in zip(ts, [(p[0], p[1]) for p in pts2d]) if t <= edge_frac + 0.02]
        radial_r = [abs((p[1] - rcy) ** 2 + (p[0] - rcx) ** 2 - (short_axis_len * 0.5) ** 2)
                    for t, p in zip(ts, [(p[0], p[1]) for p in pts2d]) if t >= 1.0 - edge_frac - 0.02]

    if not radial_l or not radial_r:
        return False

    # 端部径向残差 / 短半轴平方 平均值
    err_l = sum(radial_l) / len(radial_l)
    err_r = sum(radial_r) / len(radial_r)
    err = (err_l + err_r) * 0.5
    # 容差相对短轴: < _ARC_FIT_REL_TOL (短半轴)²
    tol = (short_axis_len * 0.5) ** 2 * _ARC_FIT_REL_TOL
    if err > tol:
        return False

    # 中段 d 应当几乎全为 0（直线）
    mid_d = [abs(d) for _, d in middle]
    if not mid_d:
        return False
    mid_max = max(mid_d)
    if mid_max > short_axis_len * 0.5 * 0.06:  # 6% 短半轴
        return False

    return True


def _looks_like_rectangle(pts2d: list[tuple[float, float]], span: float) -> bool:
    """4-corner, axis-aligned rectangle: dominant corners have ~90° interior
    angles and the four side lengths are within tolerance of the bbox.
    """
    if span <= 0:
        return False
    pts2d = _unique_ordered(pts2d)
    n = len(pts2d)
    if n < 4 or n > 200:
        return False
    # For n==4 (axis-aligned) we trust the bbox ordering
    if n == 4:
        xs = sorted({p[0] for p in pts2d})
        ys = sorted({p[1] for p in pts2d})
        if len(xs) == 2 and len(ys) == 2:
            return True
    corners = _dominant_corners(pts2d, k=4)
    if len(corners) != 4:
        return False
    # Interior angles at corners should be ~90°
    for ci in corners:
        prev_i = (ci - 1) % n
        next_i = (ci + 1) % n
        a = pts2d[prev_i]
        b = pts2d[ci]
        c = pts2d[next_i]
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1) or 1e-9
        n2 = math.hypot(*v2) or 1e-9
        cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        ang = math.degrees(math.acos(cos))
        if abs(ang - 90.0) > 22.0:
            return False
    # All 4 sides should be close to bbox extents (not too short)
    for i in range(4):
        a = pts2d[corners[i]]
        b = pts2d[corners[(i + 1) % 4]]
        side = _dist(a, b)
        if side < span * 0.20:
            return False
    return True


def _unique_ordered(pts: list[tuple[float, float]], tol: float = 1e-3) -> list[tuple[float, float]]:
    """Drop consecutive duplicates and the trailing closing-point (if equal to first)."""
    out: list[tuple[float, float]] = []
    for p in pts:
        if out and math.hypot(out[-1][0] - p[0], out[-1][1] - p[1]) <= tol:
            continue
        out.append(p)
    if len(out) >= 2 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def _looks_like_hexagon(pts2d: list[tuple[float, float]]) -> bool:
    """Approx regular hexagon: 6 dominant corners, internal angles ~120°."""
    corners = _dominant_corners(pts2d, k=6)
    if len(corners) != 6:
        return False
    # 角点处内部角度应在 120°±tolerance
    n = len(pts2d)
    for ci in corners:
        prev_i = (ci - 4) % n
        next_i = (ci + 4) % n
        a = pts2d[prev_i]
        b = pts2d[ci]
        c = pts2d[next_i]
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1) or 1e-9
        n2 = math.hypot(*v2) or 1e-9
        cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        ang = math.degrees(math.acos(cos))
        if abs(ang - 120.0) > _HEX_ANGLE_TOL_DEG * 2.5:
            return False
    return True


def _dominant_corners(pts2d: list[tuple[float, float]], k: int) -> list[int]:
    """Indices of ``k`` points with largest turning angle.

    Falls back to evenly-spaced indices when the polyline is too short for
    ``k * 2`` (e.g. a rectangle discretised as 4 corners + closing point).
    """
    n = len(pts2d)
    if n < 2:
        return []
    if n < k * 2:
        # Use the ``k`` points most distant from the centroid as corners
        cx = sum(p[0] for p in pts2d) / n
        cy = sum(p[1] for p in pts2d) / n
        scored = sorted(
            range(n),
            key=lambda i: -((pts2d[i][0] - cx) ** 2 + (pts2d[i][1] - cy) ** 2),
        )
        return sorted(scored[:k])
    turnings: list[tuple[float, int]] = []
    for i in range(n):
        a = pts2d[(i - 1) % n]
        b = pts2d[i]
        c = pts2d[(i + 1) % n]
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1) or 1e-9
        n2 = math.hypot(*v2) or 1e-9
        cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        ang = math.acos(cos)
        turnings.append((ang, i))
    turnings.sort(reverse=True)
    picked = sorted(i for _, i in turnings[:k])
    return picked


def _hex_side(pts2d: list[tuple[float, float]]) -> float:
    corners = _dominant_corners(pts2d, k=6)
    if len(corners) < 6:
        return 0.0
    sides: list[float] = []
    for i in range(6):
        a = pts2d[corners[i]]
        b = pts2d[corners[(i + 1) % 6]]
        sides.append(_dist(a, b))
    return sum(sides) / 6.0 if sides else 0.0


def _unknown_contour(cid, wire_id, polyline_id, face_id, is_outer) -> dict:
    return {
        "id": cid,
        "contour_type": "unknown",
        "contour_role": "outer_boundary" if is_outer else "inner_hole",
        "center": None,
        "normal": None,
        "polyline_id": polyline_id,
        "wire_id": wire_id,
        "face_id": face_id,
        "is_outer": is_outer,
        "parameters": {"diameter": None, "length": None, "width": None, "across_flats": None},
        "area": None,
        "perimeter": None,
        "confidence": 0.0,
    }


# ── Wire deduplication ────────────────────────────────────────────────────
def _wire_signature(
    pts2d: list[tuple[float, float]],
    area: float,
    bbox: tuple[float, float, float, float],
) -> tuple:
    """Quantised signature. Currently unused — see ``_dedupe_wires``
    which uses a stronger 3D-point hash. Kept for future heuristics
    (e.g. matching a 2D drawing to a 3D loop).
    """
    q = 0.1
    xmin, ymin, xmax, ymax = bbox
    return (
        round(xmin / q), round(ymin / q),
        round(xmax / q), round(ymax / q),
        round(area / (q * q)),
    )


def _dedupe_wires(
    wire_infos: list[dict],
    tol: float = 0.5,
) -> list[dict]:
    """Drop wire entries that are coincident copies of an earlier wire.

    Why this exists: in STEP assemblies (and in many native B-Rep exports)
    a single closed loop on a face can appear multiple times in
    ``TopExp_Explorer(wire)`` because coincident edges get duplicated
    during the geometry translation. Without dedup the wire count
    doubles (or worse) and the user sees "the same hole listed 4 times"
    in the feature table.

    Signature: a 3D-DoppleGanger hash of the polyline vertices. We sample
    up to 16 evenly-spaced points around the loop, quantise to 0.05 mm,
    and compare as a multiset. Two loops are "the same" if the
    corresponding sampled points match within ``tol``. This is
    translation- and rotation-invariant (per face, after the 3D points
    have been transformed to a common origin via the wire centroid),
    and it correctly rejects two holes that happen to share the same
    bbox+area but sit in different 3D positions.
    """
    if len(wire_infos) < 2:
        return wire_infos

    def _loop_fingerprint(pts3d: list[tuple[float, float, float]]) -> tuple | None:
        if len(pts3d) < 3:
            return None
        # Sample up to 16 points evenly around the loop (closed).
        step = max(1, len(pts3d) // 16)
        sampled = [pts3d[i] for i in range(0, len(pts3d), step)]
        if len(sampled) < 4:
            sampled = pts3d[:]
        # Translate to centroid so the fingerprint is position-invariant.
        cx = sum(p[0] for p in sampled) / len(sampled)
        cy = sum(p[1] for p in sampled) / len(sampled)
        cz = sum(p[2] for p in sampled) / len(sampled)
        q = 0.05  # 50 µm — well below linear_deflection but finer than noise
        return tuple(
            sorted(
                (
                    round((p[0] - cx) / q),
                    round((p[1] - cy) / q),
                    round((p[2] - cz) / q),
                )
                for p in sampled
            )
        )

    kept: list[dict] = []
    kept_fps: list[tuple] = []
    for w in wire_infos:
        pts = w.get("pts") or []
        if len(pts) < 3:
            # Open polylines (degenerate, noise) — never keep duplicates,
            # just take the first one and stop.
            if not any(True for k in kept if not (k.get("pts") or [])):
                kept.append(w)
            continue
        fp = _loop_fingerprint(pts)
        if fp is None:
            kept.append(w)
            kept_fps.append(())
            continue
        # Check for a matching prior fingerprint (set comparison so
        # direction-of-traversal and starting-point don't matter).
        fp_set = set(fp)
        dup = False
        for k_fp in kept_fps:
            if not k_fp:
                continue
            if fp_set == set(k_fp):
                dup = True
                break
        if not dup:
            kept.append(w)
            kept_fps.append(fp)
    return kept


def _project_wire_points_2d(
    pts_world: list[tuple[float, float, float]],
    face_normal: tuple[float, float, float] | None,
) -> list[tuple[float, float]]:
    """Project 3D wire points onto the dominant plane of the face normal.
    Used both by the classifier and by the dedup helper to keep their
    coordinate systems consistent.
    """
    nx, ny, nz = face_normal or (0.0, 0.0, 1.0)
    if abs(nz) >= max(abs(nx), abs(ny)):
        return [(p[0], p[1]) for p in pts_world]
    if abs(ny) >= abs(nx):
        return [(p[0], p[2]) for p in pts_world]
    return [(p[1], p[2]) for p in pts_world]


def _mark_concentric_rings(
    contours: list[dict],
    face_id: str,
    centre_tol_mm: float = 0.5,
) -> None:
    """Tag every ``circle`` contour that is a *larger concentric ring* of
    a smaller one with ``contour_role="concentric_ring"``.

    Multiple closed wires on the same face that classify as
    ``circle`` (or ``ellipse``) are usually the same physical hole
    seen at different depths / radii — the through-hole's wire, the
    bottom of a counterbore, the chamfer ring at the top edge, etc.
    They share a 2D centre within ``centre_tol_mm`` and have
    increasing diameter. Tagging all-but-the-smallest as
    ``concentric_ring`` lets the hole-derivation step keep the
    through-hole only.

    Modifies the contours list in-place by setting
    ``contour["contour_role"] = "concentric_ring"`` on the outer
    rings. The smallest circle in each cluster is left as
    ``"inner_hole"`` so the existing pipeline treats it as the
    primary hole.

    Diagnostic: emits a debug log per cluster so you can confirm the
    grouping matches what's on the model.
    """
    circles = [
        c for c in contours
        if c.get("contour_type") in ("circle", "ellipse")
        and not c.get("is_outer")
        and c.get("center")
    ]
    if len(circles) < 2:
        return

    def _diameter(c: dict) -> float:
        params = c.get("parameters") or {}
        if c.get("contour_type") == "ellipse":
            return float(params.get("length") or 0.0)
        return float(params.get("diameter") or 0.0)

    # Build clusters: greedy union-find by 2D centre distance.
    # 2D centre is the 2D polygon centroid already stored in c["center"]
    # (the 3D version is a list [x, y, z]; we just take the first two).
    n = len(circles)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            ci = circles[i]["center"]
            cj = circles[j]["center"]
            dx = ci[0] - cj[0]
            dy = ci[1] - cj[1]
            if (dx * dx + dy * dy) ** 0.5 < centre_tol_mm:
                _union(i, j)

    # Group by cluster root.
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        r = _find(i)
        clusters.setdefault(r, []).append(i)

    for r, members in clusters.items():
        if len(members) < 2:
            continue
        # Sort by diameter ascending: smallest is the through-hole,
        # the rest are counterbore / chamfer / boss rings.
        members.sort(key=lambda i: _diameter(circles[i]))
        kept = members[0]
        for idx in members[1:]:
            circles[idx]["contour_role"] = "concentric_ring"
            if _DEBUG_CLASSIFICATION:
                import logging
                logging.getLogger("contour").debug(
                    "concentric_ring: face=%s cid=%s diameter=%.4f tagged as ring (kept cid=%s d=%.4f)",
                    face_id,
                    circles[idx].get("id"),
                    _diameter(circles[idx]),
                    circles[kept].get("id"),
                    _diameter(circles[kept]),
                )


def _looks_like_irregular(pts2d: list[tuple[float, float]], aspect: float) -> bool:
    """Conservative pre-check used in the classifier before falling through
    to the 'irregular' bucket. Returns True for any closed polyline with
    >= 4 vertices that has a sensible aspect ratio and isn't a perfect
    circle. This is just the gate — the caller still records the
    detailed length/width.
    """
    if len(pts2d) < 4:
        return False
    if aspect >= 12.0:
        # Very long thin shape — probably a degenerate outer boundary,
        # not a real hole. Falls through to `unknown`.
        return False
    return True


def _dedupe_holes_across_faces(
    holes: list[dict],
    dominant: dict | None,
    centre_tol_mm: float = 0.5,
    size_tol_ratio: float = 0.05,
) -> list[dict]:
    """Drop holes that are coincident copies of an earlier one.

    A real through-hole on a plate is captured twice: once on the
    top face's wire and once on the bottom face's wire — same
    3D centre, same kind, same diameter, just mirrored. We keep
    the one whose axis is most aligned with the dominant face
    normal (the visible side the user is looking at).

    Algorithm: union-find over holes whose 3D centres are within
    ``centre_tol_mm`` and whose primary size (diameter for circles,
    max(length,width) for the rest) is within ``size_tol_ratio``.
    The keep-rule is:
      - prefer the hole whose ``axis`` dot ``dominant.normal`` is
        highest (i.e. facing the camera),
      - tie-break on smaller face_id (deterministic),
      - drop the others.
    """
    if len(holes) < 2:
        return list(holes)

    # We can't dedup holes whose centres are 3D coincident but
    # actually represent *different* features that happen to be on
    # the same XY location but different Z — those are valid
    # features at different heights. We *do* want to dedup when
    # centres are within ~0.5mm in 3D AND the two holes are on
    # opposing sides of the same plate (centres are basically
    # coincident because the plate is thin).
    n = len(holes)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    def _size(h: dict) -> float:
        # For circles: diameter. For others: max(length, width).
        params = h.get("parameters") or {}
        d = float(params.get("diameter") or 0.0)
        if d > 0:
            return d
        return max(float(params.get("length") or 0.0), float(params.get("width") or 0.0))

    def _centre(h: dict) -> tuple[float, float, float]:
        c = h.get("center") or [0.0, 0.0, 0.0]
        return float(c[0]), float(c[1]), float(c[2])

    dom_n = dominant.get("normal") if dominant else None

    for i in range(n):
        ci = _centre(holes[i])
        si = _size(holes[i])
        ki = holes[i].get("kind") or holes[i].get("contour_type")
        for j in range(i + 1, n):
            cj = _centre(holes[j])
            sj = _size(holes[j])
            kj = holes[j].get("kind") or holes[j].get("contour_type")
            if ki != kj:
                continue
            dx = ci[0] - cj[0]
            dy = ci[1] - cj[1]
            dz = ci[2] - cj[2]
            if (dx * dx + dy * dy + dz * dz) ** 0.5 > centre_tol_mm:
                continue
            if si <= 0 or sj <= 0:
                continue
            rel = abs(si - sj) / max(si, sj)
            if rel > size_tol_ratio:
                continue
            _union(i, j)

    # For each cluster, pick the survivor.
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(_find(i), []).append(i)
    keep_idx: set[int] = set()
    survivors: list[int] = []
    for r, members in clusters.items():
        if len(members) == 1:
            survivors.append(members[0])
            continue
        # Pick the one whose axis is most aligned with the dominant
        # face normal. If a hole has no axis info, fall back to the
        # smallest face_id (deterministic).
        def _score(idx: int) -> tuple[float, str]:
            axis = holes[idx].get("axis")
            if dom_n and axis and len(axis) == 3:
                try:
                    dot = abs(axis[0] * dom_n[0] + axis[1] * dom_n[1] + axis[2] * dom_n[2])
                except (TypeError, IndexError):
                    dot = 0.0
            else:
                dot = 0.0
            return (dot, str(holes[idx].get("face_id") or ""))

        members.sort(key=_score, reverse=True)
        survivors.append(members[0])
    return [holes[i] for i in sorted(survivors)]


# ── Per-face pipeline ─────────────────────────────────────────────────────
def analyze_face(
    face: TopoDS_Face,
    face_id: str,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    work_plane: str = "auto",
    work_plane_normal: tuple[float, float, float] | None = None,
    hole_diameter_min: float = 0.5,
    hole_diameter_max: float = 500.0,
    include_cylinder_holes: bool = False,
    closure_tol: float = _CLOSURE_TOL_DEFAULT,
    enhanced_params: bool = False,
) -> dict:
    """Run the full wire → polyline → contour → hole pipeline on a single
    TopoDS_Face. Returns plain dict that matches ``CadFaceAnalyzeResult``.

    Args:
        enhanced_params: If True, extract laser-software-style parameters
            (rotation_angle, corner_radius, compensation_length, etc.)
    """
    surf = face_surface_info(face)
    f_normal = face_outward_normal(face) or surf.get("normal")
    f_axis = surf.get("axis")
    f_center = surf.get("center")
    f_radius = surf.get("radius")

    # effective work-plane normal (geometry_utils handles auto)
    if work_plane_normal is None:
        # face.normal 在没有显式 bbox 上下文时，直接用 face 法向
        if f_normal and work_plane == "auto":
            wp_normal = f_normal
        else:
            wp_normal = work_plane_normal_for_mode(work_plane, f_normal)
    else:
        wp_normal = work_plane_normal

    wires = face_wires(face)
    polylines: list[dict] = []
    contours: list[dict] = []
    holes: list[dict] = []
    ref_points: list[dict] = []

    # 1) collect per-wire polyline + classify
    wire_infos: list[dict] = []
    for wi, wire in enumerate(wires):
        wid = f"wire_{face_id}_{wi}"
        loc = wire_location_on_face(face, wire)
        pts = wire_to_polyline(
            wire, linear_deflection, angular_deflection, location=loc
        )
        pid = f"poly_{wid}"
        # NB: we used to push the polyline into the response list here,
        # but the list is now built from the deduped wire set below so
        # coincident wires never produce a polyline entry. Building
        # early also meant a dedup at step 1b couldn't shrink the
        # payload.
        wlen = wire_length(wire) if pts else 0.0
        # BRepGProp.SurfaceProperties on a wire often returns 0; fall back
        # to 2D shoelace for a reliable per-wire area used by outer/inner split.
        warea = wire_area_if_planar(wire) if pts else None
        if not warea:
            warea = _polyline_area_2d(pts, tuple(f_normal) if f_normal else None)

        wire_infos.append({
            "id": wid,
            "is_outer": False,   # 暂时标记，后续重写
            "closed": _closed(pts, tol=closure_tol),
            "length": wlen,
            "area": warea,
            "polyline_id": pid,
            "pts": pts,
            # pre-projected 2D points, used both for dedup (bbox+area hash)
            # and for the classifier. Keeping them on the wire_info
            # avoids a second projection pass.
            "pts2d": _project_wire_points_2d(pts, tuple(f_normal) if f_normal else None),
        })

    # 1b) drop coincident wires. STEP assemblies frequently expose the
    # same closed loop multiple times (e.g. the face is the union of
    # two coincident sub-faces with identical trim curves). Without
    # dedup the user sees "the same hole listed 4 times" in the
    # feature table. We only dedup closed wires with positive area —
    # open/zero-area wires are not features, they're trimming
    # artefacts and we just throw them away a few lines later.
    wire_infos = _dedupe_wires(wire_infos)

    # 1c) build the per-face polyline list from the *deduped* wire
    # set. Previously polylines were appended in step 1, before
    # dedup, which meant the response payload still carried one
    # polyline per original wire (429 of them on the user's STEP
    # assembly) even after the dedup trimmed ``wire_infos``. Now
    # the polylines list is rebuilt from the post-dedup set so the
    # payload size matches what the classifier actually saw.
    polylines = [
        {
            "id": w.get("polyline_id") or f"poly_{w['id']}",
            "closed": w.get("closed", False),
            "points": [_pt(p) for p in (w.get("pts") or [])],
        }
        for w in wire_infos
    ]

    # 2) outer / inner split (正确的几何定义：outer = 不被任何其它
    # 闭环包含的 wire；inner = 至少被一个 outer 包含)。原本
    # 的"面积最大 = outer"启发式在 face 的外环被拆成多个 wire
    # （比如带椭圆挖洞的平板，椭圆的左右半圆或上下两条边被表达
    # 成独立的小 wire）时会被骗到——单看面积椭圆的"长边"折合
    # 出来的面积可能更大，于是被错标为 outer，把真正的 face 边
    # 界降级为 inner。
    #
    # 新算法：对每个闭合 wire，用 shoelace 面积 + 2D 重心，若 wire
    # A 的 bbox 被 wire B 的 bbox 完整包含（且中心点在 B 的多边形内）
    # 且 B 面积 > A 面积，则 A 是 inner。剩下的"最外层"就是 outer。
    closed_wires = [w for w in wire_infos if w["closed"] and (w["area"] or 0) > 0]
    if closed_wires:
        # 预计算每个 wire 的 2D bbox（来自 pts2d，已经在 face 平面内）
        bbox_by_id: dict[str, tuple[float, float, float, float]] = {}
        for w in closed_wires:
            xs = [p[0] for p in w.get("pts2d") or []]
            ys = [p[1] for p in w.get("pts2d") or []]
            if not xs:
                bbox_by_id[w["id"]] = (0.0, 0.0, 0.0, 0.0)
            else:
                bbox_by_id[w["id"]] = (min(xs), min(ys), max(xs), max(ys))
        # Containment check: A is inner if there is some other wire B
        # whose bbox strictly contains A's bbox AND B's area is at
        # least 1.05x A's. (We require a small area margin so two
        # coincident wires don't ping-pong each other.)
        contained_by: dict[str, str | None] = {w["id"]: None for w in closed_wires}
        for wA in closed_wires:
            ax1, ay1, ax2, ay2 = bbox_by_id[wA["id"]]
            for wB in closed_wires:
                if wB["id"] == wA["id"]:
                    continue
                bx1, by1, bx2, by2 = bbox_by_id[wB["id"]]
                # strict bbox containment with a 0.1mm margin
                if (bx1 <= ax1 + 0.1 and by1 <= ay1 + 0.1
                        and bx2 >= ax2 - 0.1 and by2 >= ay2 - 0.1):
                    if wB["area"] >= wA["area"] * 1.05:
                        # pick the smallest enclosing wire
                        if contained_by[wA["id"]] is None or wB["area"] < (closed_wires[
                            next(i for i, w in enumerate(closed_wires) if w["id"] == contained_by[wA["id"]])
                        ]["area"]):
                            contained_by[wA["id"]] = wB["id"]
        # Mark: contained -> inner; not contained -> outer
        # A face must have exactly one outer (the face boundary). If
        # containment analysis finds >1 "not contained" wire (e.g. two
        # sibling wires that touch but don't overlap), fall back to
        # "largest area" for that face so the feature is still
        # represented.
        not_contained = [w for w in closed_wires if contained_by[w["id"]] is None]
        if len(not_contained) == 1:
            outer_id = not_contained[0]["id"]
        elif not_contained:
            not_contained.sort(key=lambda w: w["area"] or 0.0, reverse=True)
            outer_id = not_contained[0]["id"]
        else:
            # All wires are contained by something — pathological. Fall
            # back to the absolute largest wire as outer.
            closed_wires_sorted = sorted(closed_wires, key=lambda w: w["area"] or 0.0, reverse=True)
            outer_id = closed_wires_sorted[0]["id"]
        for w in wire_infos:
            w["is_outer"] = (w["id"] == outer_id)

    # 3) classify each wire
    for ci, w in enumerate(wire_infos):
        contour = classify_wire_contour(
            w.get("pts") or [],
            face_normal=tuple(f_normal) if f_normal else None,
            is_outer=w["is_outer"],
            wire_id=w["id"],
            polyline_id=w["polyline_id"],
            face_id=face_id,
            contour_index=ci,
            enhanced_params=enhanced_params,
        )
        if w.get("area"):
            contour["area"] = w["area"]
        contours.append(contour)

    # 3b) concentric-circle grouping: a face may have multiple
    # coincident closed loops that all classify as ``circle`` (e.g.
    # the wall of a counterbore is a ring, the bottom of the bore is
    # another ring, the through-hole is yet another). Without
    # grouping the user sees the same hole listed 3 times and the
    # laser planner tries to cut each ring as an independent hole.
    # The fix: cluster ``circle`` contours by their 2D centre
    # distance, sort each cluster by diameter, and tag the outer
    # rings as ``ring`` (kept for visualisation, but skipped by hole
    # derivation). The smallest circle in a cluster is the through
    # hole; everything larger is a counterbore / chamfer ring.
    _mark_concentric_rings(contours, face_id)

    # 4) wires (post-classification)
    wires_out: list[dict] = []
    for w, contour in zip(wire_infos, contours):
        wires_out.append({
            "id": w["id"],
            "face_id": face_id,
            "is_outer": w["is_outer"],
            "length": w["length"],
            "area": w["area"],
            "polyline_id": w["polyline_id"],
            "contour_id": contour["id"],
            "contour_type": contour["contour_type"],
        })

    # 5) hole derivation (inner contours of supported types)
    is_planar = (surf.get("surface_type") == "plane")
    for contour in contours:
        if not is_planar:
            continue
        if contour["is_outer"]:
            continue
        if contour.get("contour_role") == "concentric_ring":
            # Concentric outer ring (counterbore / chamfer / boss
            # edge) — kept in the contour list for visualisation but
            # never reported as an independent hole. The through-hole
            # it surrounds already covers the actual cut.
            continue
        if contour["contour_type"] in (
            "circle",
            "ellipse",
            "slot",
            "rectangle",
            "hexagon",
            "irregular",
        ):
            _contour_to_hole(contour, face_id, holes, ref_points, hole_diameter_min, hole_diameter_max)

    # 6) reference points: contour centers + face center
    for contour in contours:
        if contour.get("center") is None:
            continue
        ref_points.append({
            "id": f"pt_{contour['id']}_center",
            "kind": "contour_center",
            "position": contour["center"],
            "meta": {
                "contour_id": contour["id"],
                "contour_type": contour["contour_type"],
                "face_id": face_id,
            },
        })

    # 7) outer_contours (global best)
    outer_contours = _select_global_outer_contours(contours)

    # 8) face record
    outer_wire_id = next((w["id"] for w in wire_infos if w["is_outer"]), None)
    inner_wire_ids = [w["id"] for w in wire_infos if not w["is_outer"]]
    face_record = {
        "id": face_id,
        "surface_type": surf.get("surface_type", "other"),
        "area": face_area(face) if surf.get("surface_type") == "plane" else None,
        "normal": _vec(f_normal) if f_normal else None,
        "axis": _vec(f_axis) if f_axis else None,
        "center": _pt(f_center) if f_center else None,
        "radius": f_radius,
        "bbox": None,
        "outer_wire_id": outer_wire_id,
        "inner_wire_ids": inner_wire_ids,
    }

    return {
        "schema_version": "1.1",
        "unit": "mm",
        "target_face_id": face_id,
        "face": face_record,
        "reference_points": ref_points,
        "polylines": polylines,
        "wires": wires_out,
        "contours": contours,
        "outer_contours": outer_contours,
        "outer_contour_ids": [c["id"] for c in outer_contours],
        "holes": holes,
        "pockets": [],
        "feature_groups": _build_feature_groups(contours=contours, holes=holes, wires=wires_out),
        "work_plane": work_plane,
        "work_plane_normal": _vec(wp_normal) if wp_normal else None,
    }


def _contour_to_hole(
    contour: dict,
    face_id: str,
    holes: list,
    ref_points: list,
    hole_diameter_min: float,
    hole_diameter_max: float,
) -> None:
    ctype = contour["contour_type"]
    params = contour.get("parameters") or {}
    diam = params.get("diameter")
    # Apply the size filter uniformly:
    # - circle: filter on diameter
    # - slot/rectangle/hexagon/irregular: filter on the longest bbox edge
    #   so a 0.1mm speck of noise doesn't get reported as a "slot".
    bbox_max = max(
        float(params.get("length") or 0.0),
        float(params.get("width") or 0.0),
        float(diam or 0.0),
    )
    if bbox_max > 0 and not (hole_diameter_min <= bbox_max <= hole_diameter_max):
        if _DEBUG_CLASSIFICATION:
            import logging
            logging.getLogger("contour").debug(
                "hole_size_filter: cid=%s type=%s bbox_max=%.4f (min=%.2f max=%.2f) - rejected",
                contour.get("id"), ctype, bbox_max, hole_diameter_min, hole_diameter_max,
            )
        return
    if _DEBUG_CLASSIFICATION:
        import logging
        logging.getLogger("contour").debug(
            "hole_accepted: cid=%s type=%s bbox_max=%.4f", contour.get("id"), ctype, bbox_max,
        )
    holes.append({
        "id": f"hole_{contour['id']}",
        "kind": ctype,
        "contour_type": ctype,
        "center": contour["center"],
        "axis": contour["normal"],
        "diameter": diam,
        "depth": None,
        "face_id": face_id,
        "wire_id": contour.get("wire_id"),
        "cylindrical_face_ids": [],
        "parameters": {
            "diameter": params.get("diameter"),
            "length": params.get("length"),
            "width": params.get("width"),
            "across_flats": params.get("across_flats"),
        },
    })
    if ref_points is not None and contour.get("center"):
        ref_points.append({
            "id": f"pt_hole_{contour['id']}",
            "kind": "hole_center",
            "position": contour["center"],
            "meta": {
                "diameter": diam,
                "contour_type": ctype,
                "contour_id": contour["id"],
                "face_id": face_id,
            },
        })


def _select_global_outer_contours(contours: list[dict]) -> list[dict]:
    """Return full contour dicts for outer boundaries, sorted by area desc."""
    outer = [c for c in contours if c.get("is_outer")]
    outer.sort(key=lambda c: c.get("area") or 0.0, reverse=True)
    return outer


def _build_feature_groups(*, contours, holes, wires) -> dict:
    contours_by_type: dict[str, list] = {}
    for c in contours:
        contours_by_type.setdefault(c.get("contour_type", "unknown"), []).append(c)
    holes_by_type: dict[str, list] = {}
    for h in holes:
        holes_by_type.setdefault(h.get("contour_type") or h.get("kind") or "unknown", []).append(h)
    return {
        "contours_by_type": contours_by_type,
        "holes_by_type": holes_by_type,
        "wires_by_role": {
            "outer": [w for w in wires if w.get("is_outer")],
            "inner": [w for w in wires if not w.get("is_outer")],
        },
    }


def work_plane_normal_for_mode(
    mode: str,
    fallback: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    """Wrapper around ``geometry_utils.work_plane_normal`` that accepts a
    degenerate fallback for non-bbox cases (e.g. work_plane='xy' before bbox
    is computed).
    """
    if mode in ("xy", "yz", "xz"):
        return work_plane_normal(mode, (0, 0, 0, 1, 1, 1))  # equal extents → first match
    return fallback


# ── Shape-level entry: enumerate faces & find by id ───────────────────────
def list_faces(shape: TopoDS_Shape) -> list[TopoDS_Face]:
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    out: list[TopoDS_Face] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(topods.Face(exp.Current()))
        exp.Next()
    return out


def find_face_by_id(shape: TopoDS_Shape, face_id: str) -> tuple[TopoDS_Face, str]:
    """Return (face, canonical_face_id).

    Accepts:
    - ``face_<index>`` (canonical, e.g. ``face_12``) — direct face lookup
    - bare integer string (``"12"``) — same as ``face_12``
    - ``part_<index>`` or ``Part_<index>`` — selects a part (Solid/Shell) and
      returns its first face; useful when the frontend selected an entire
      part node in ``pick_level=part`` mode
    """
    faces = list_faces(shape)
    if not faces:
        raise ValueError("model has no faces")

    fid = (face_id or "").strip()
    if not fid:
        raise ValueError("face_id 不能为空")

    target_idx: int | None = None
    lower = fid.lower()
    if lower.startswith("face_"):
        try:
            target_idx = int(fid[5:])
        except ValueError:
            target_idx = None
    elif lower.startswith("part_"):
        # pick the first face belonging to the requested solid (1-based part index)
        from app.occ.topology import iter_solids

        try:
            part_idx = int(fid[5:])
        except ValueError:
            raise ValueError(f"face_id 解析失败: {face_id!r}")
        solids = iter_solids(shape)
        if part_idx < 0 or part_idx >= len(solids):
            raise ValueError(
                f"part_id {face_id!r} out of range (0..{len(solids) - 1})"
            )
        target_solid = solids[part_idx]
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopoDS import topods

        exp = TopExp_Explorer(target_solid, TopAbs_FACE)
        if not exp.More():
            raise ValueError(f"part_id {face_id!r} 不含任何 face")
        # return the first face that exists in the global face index list
        first_face = topods.Face(exp.Current())
        try:
            local_idx = faces.index(first_face)
        except ValueError:
            local_idx = 0
        return faces[local_idx], f"face_{local_idx}"
    elif fid.isdigit():
        target_idx = int(fid)

    if target_idx is None or target_idx < 0 or target_idx >= len(faces):
        # Fall back: if the model has exactly one face, return it; otherwise
        # surface a clear error. This handles the ``Part_<timestamp>`` case
        # (frontend mesh.name fallback for unnamed nodes) and similar glitches.
        if len(faces) == 1:
            return faces[0], "face_0"
        raise ValueError(
            f"face_id {face_id!r} out of range (0..{len(faces) - 1})"
        )

    return faces[target_idx], f"face_{target_idx}"


def list_solids(shape: TopoDS_Shape) -> list[TopoDS_Solid]:
    """Return all Solids in ``shape``; empty list if shape is a single
    Solid/Shell without subdivision."""
    from app.occ.topology import iter_solids
    from OCC.Core.TopoDS import topods

    solids = iter_solids(shape)
    if solids:
        return solids
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer

    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    out: list[TopoDS_Solid] = []
    while exp.More():
        out.append(topods.Solid(exp.Current()))
        exp.Next()
    return out


def _face_count_in_solid(solid) -> int:
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer

    exp = TopExp_Explorer(solid, TopAbs_FACE)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def find_solid_by_id(shape: TopoDS_Shape, part_id: str) -> tuple[TopoDS_Solid | None, int | str]:
    """Resolve a part selector to a single Solid (and its global face offset),
    or signal "analyse every solid" via the sentinel ``(None, "ALL")``.

    Accepts:
    - ``part_<idx>`` / ``Part_<idx>`` — direct index lookup
    - ``*_part_<idx>`` (e.g. ``Assembly_part_0``, ``user_sample_root_part_1``) —
      prefix-aware index lookup; the index is always the last numeric suffix
      after the last ``_part_`` separator.
    - bare integer string — same as ``part_<n>``
    - ``part_<non_numeric>`` (e.g. timestamp fallback) — single-solid models
      resolve to ``(solid, 0)``; multi-solid models raise.
    - bare non-numeric (e.g. the assembly root mesh name ``"Assembly"``) —
      single-solid models resolve to ``(solid, 0)``; multi-solid models
      resolve to ``(None, "ALL")`` so the caller can run the analysis over
      every solid and aggregate.

    The two return shapes are intentionally distinct: callers that pick a
    specific part expect a single Solid; callers that pick "the whole
    assembly" expect a model-wide aggregate.
    """
    import re

    fid = (part_id or "").strip()
    if not fid:
        raise ValueError("part_id 不能为空")
    lower = fid.lower()
    idx: int | None = None

    if lower.startswith("part_"):
        suffix = fid[5:]
        if suffix.isdigit():
            idx = int(suffix)
        else:
            # Non-numeric suffix (e.g. timestamp from frontend fallback
            # ``Part_${Date.now()}``). Single-solid models fall back to that
            # one solid; multi-solid models raise.
            solids = list_solids(shape)
            if len(solids) == 1:
                return solids[0], 0
            raise ValueError(
                f"part_id {fid!r} 不含数字索引，且模型包含 {len(solids)} 个 solid，"
                f"请用 part_0..part_{len(solids) - 1} 指定"
            )
    elif fid.isdigit():
        idx = int(fid)
    else:
        # Try to extract a trailing numeric index from names like
        # "Assembly_part_0" or "user_sample_root_part_1". The index is always
        # the last component after splitting on "_part_".
        # We search from the end of the string for the last "_part_" pattern.
        # This handles: Assembly_part_0, user_sample_root_part_1, Part_12, etc.
        solids = list_solids(shape)

        # Match the last occurrence of _part_<number> (case-insensitive)
        m = re.search(r"_part_(\d+)$", lower, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
        else:
            # No recognisable part suffix — bare non-numeric part_id.
            # Single-solid models fall back to that one solid;
            # multi-solid models fan out to a full-assembly aggregate.
            if len(solids) == 1:
                return solids[0], 0
            return None, "ALL"

    solids = list_solids(shape)
    # If the numeric index is out of range, fall back to the only solid
    # if there is exactly one. This handles the common case where the
    # frontend sends ``Part_<timestamp>`` (its ``mesh.name`` fallback when
    # a GLB node has no name) but the model only has a single part.
    if idx < 0 or idx >= len(solids):
        if len(solids) == 1:
            return solids[0], 0
        raise ValueError(
            f"part_id {part_id!r} out of range (0..{len(solids) - 1})"
        )
    return solids[idx], idx


def analyze_part(
    shape: TopoDS_Shape,
    part_id: str,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    work_plane: str = "auto",
    work_plane_normal: tuple[float, float, float] | None = None,
    hole_diameter_min: float = 0.5,
    hole_diameter_max: float = 500.0,
    include_cylinder_holes: bool = False,
    closure_tol: float = _CLOSURE_TOL_DEFAULT,
    enhanced_params: bool = False,
    target_face_id: str | None = None,
) -> dict:
    """Analyse a whole Solid (part) and aggregate features of all its
    plane faces. Non-planar faces contribute only their metadata to the
    ``faces`` summary; the ``contours`` / ``holes`` lists are unions of
    every plane-face feature, de-duplicated by id.

    Returns a dict with the same schema as ``analyze_face`` but:
    - ``target_face_id`` is ``part_<idx>``
    - ``face`` is a synthetic "part" record
    - ``per_face`` lists each analysed face with its own result subset

    Args:
        target_face_id: Optional face selector (``face_<n>``) within the
            same part. When provided, the analysis keeps **only** the
            inner features (contours, holes, polylines) of faces whose
            outward normal is on the same side as this face
            (|dot(normal, target_normal)| > 0.5). This is the
            "one-sided" feature extraction the user asked for: a
            plate has two planar sides, and without filtering the
            back-side features are also picked up, which manifests as
            duplicate circle / ellipse contours and bogus "outer
            boundaries" mirrored from the back. The dominant face
            (largest plane area) is still used to define the
            synthetic part record and the dedup tie-breaker, but
            *content* is restricted to the target side.
    """
    # Ensure debug logs are visible when classification debug is on
    if _DEBUG_CLASSIFICATION:
        import logging
        contour_logger = logging.getLogger("contour")
        if not contour_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter("[CONTOUR] %(message)s"))
            contour_logger.addHandler(handler)
        contour_logger.setLevel(logging.DEBUG)

    from app.occ.geometry_utils import (
        face_area,
        face_surface_info,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods

    solid_or_all, solid_idx = find_solid_by_id(shape, part_id)
    bbox = shape_bbox_dict(shape)

    # Enumerate all faces of this shape in *global* face indices, so that the
    # face_ids returned to the client line up with the GLB node names.
    # We match by walking the full face list and using IsSame; this is O(n*m)
    # but the number of faces per part is small (< 1000).
    global_faces = list_faces(shape)

    def _face_index(target) -> int | None:
        for i, f in enumerate(global_faces):
            if f.IsSame(target):
                return i
        return None

    def _analyze_one_solid(solid, idx_label: int | str) -> dict:
        """Run analyse_face over every plane face of one Solid and aggregate
        contours / holes / polylines / wires. ``idx_label`` becomes the
        ``part_<idx>`` tag in the synthetic part record; for the
        whole-assembly aggregate we use the sentinel string ``"all"``.
        """
        solid_faces: list[tuple[int, TopoDS_Face]] = []
        exp = TopExp_Explorer(solid, TopAbs_FACE)
        while exp.More():
            f = topods.Face(exp.Current())
            gi = _face_index(f)
            if gi is not None:
                solid_faces.append((gi, f))
            exp.Next()
        solid_faces.sort(key=lambda x: x[0])

        # If a target face id was supplied, look up its outward normal
        # first so we can drop every contribution from the back side
        # of the plate (the same physical hole/edge that wraps around
        # the plate would otherwise show up twice).
        target_normal: tuple[float, float, float] | None = None
        if target_face_id:
            for gi, f in solid_faces:
                if f"face_{gi}" == target_face_id:
                    n = face_outward_normal(f)
                    if n:
                        target_normal = n
                    break

        per_face: list[dict] = []
        agg_contours: list[dict] = []
        agg_holes: list[dict] = []
        agg_ref: list[dict] = []
        agg_wires: list[dict] = []
        agg_polylines: list[dict] = []
        agg_outer: list[str] = []

        plane_face_count = 0
        for gi, f in solid_faces:
            info = face_surface_info(f)
            if info.get("surface_type") != "plane":
                per_face.append({
                    "id": f"face_{gi}",
                    "surface_type": info.get("surface_type"),
                    "area": None,
                    "normal": _vec(info.get("normal")),
                    "axis": _vec(info.get("axis")),
                    "center": _pt(info.get("center")),
                    "radius": info.get("radius"),
                    "analysed": False,
                    "contour_count": 0,
                    "hole_count": 0,
                })
                continue
            plane_face_count += 1
            # Side filter: when the caller picked a target face, only
            # contribute this face's features if it is the same side
            # (|dot| > 0.5). The 0.5 threshold is generous so that
            # very-shallow faces (e.g. ~60° chamfers that share the
            # same hole as the main face) are still considered the
            # same side; the back side of a plate typically gives
            # dot ≈ -1 and is dropped.
            if target_normal is not None:
                face_n = face_outward_normal(f)
                if face_n is None:
                    face_n = info.get("normal")
                if face_n is not None:
                    dot = abs(
                        face_n[0] * target_normal[0]
                        + face_n[1] * target_normal[1]
                        + face_n[2] * target_normal[2]
                    )
                    if dot < 0.5:
                        per_face.append({
                            "id": f"face_{gi}",
                            "surface_type": "plane",
                            "area": face_area(f),
                            "normal": _vec(info.get("normal")),
                            "axis": _vec(info.get("axis")),
                            "center": _pt(info.get("center")),
                            "radius": info.get("radius"),
                            "analysed": False,
                            "contour_count": 0,
                            "hole_count": 0,
                            "skipped_reason": "back_side",
                        })
                        continue
            sub = analyze_face(
                f,
                f"face_{gi}",
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
                work_plane=work_plane,
                work_plane_normal=work_plane_normal,
                hole_diameter_min=hole_diameter_min,
                hole_diameter_max=hole_diameter_max,
                include_cylinder_holes=include_cylinder_holes,
                closure_tol=closure_tol,
                enhanced_params=enhanced_params,
            )
            per_face.append({
                "id": f"face_{gi}",
                "surface_type": "plane",
                "area": face_area(f),
                "normal": _vec(info.get("normal")),
                "axis": _vec(info.get("axis")),
                "center": _pt(info.get("center")),
                "radius": info.get("radius"),
                "analysed": True,
                "contour_count": len(sub["contours"]),
                "hole_count": len(sub["holes"]),
            })
            for c in sub["contours"]:
                # IMPORTANT: do not dedup by raw c["id"] because every
                # face's analyse_face() call generates ids from a
                # per-face counter (``contour_0``…), so without
                # prefixing multiple faces' contours collide and only
                # the first face's set survives the aggregation.
                c2 = dict(c)
                c2["id"] = f"face_{gi}__{c['id']}"
                c2["wire_id"] = f"face_{gi}__{c.get('wire_id','')}"
                c2["polyline_id"] = f"face_{gi}__{c.get('polyline_id','')}"
                c2["face_id"] = f"face_{gi}"
                agg_contours.append(c2)
            for h in sub["holes"]:
                # Same prefixing rule as contours — without it the
                # second face's hole gets dropped because
                # ``hole_contour_0`` was already in seen_h.
                h2 = dict(h)
                h2["id"] = f"face_{gi}__{h['id']}"
                h2["face_id"] = f"face_{gi}"
                h2["wire_id"] = f"face_{gi}__{h.get('wire_id','')}"
                agg_holes.append(h2)
            for r in sub.get("reference_points") or []:
                r2 = dict(r)
                r2["id"] = f"face_{gi}__{r['id']}"
                r2["face_id"] = f"face_{gi}"
                agg_ref.append(r2)
            for w in sub.get("wires") or []:
                w2 = dict(w)
                w2["id"] = f"face_{gi}__{w['id']}"
                w2["face_id"] = f"face_{gi}"
                w2["polyline_id"] = f"face_{gi}__{w.get('polyline_id','')}"
                w2["contour_id"] = f"face_{gi}__{w.get('contour_id','')}"
                agg_wires.append(w2)
            for p in sub.get("polylines") or []:
                p2 = dict(p)
                p2["id"] = f"face_{gi}__{p['id']}"
                agg_polylines.append(p2)
            for oid in sub.get("outer_contours") or []:
                # Use the same per-face prefix as the contour id so the
                # outer ids line up with the rewritten agg_contours.
                prefixed = f"face_{gi}__{oid}"
                if prefixed not in agg_outer:
                    agg_outer.append(prefixed)

        # Pick the largest plane face as the synthetic "part" record
        plane_faces = [pf for pf in per_face if pf.get("analysed")]
        dominant = max(plane_faces, key=lambda x: x.get("area") or 0.0) if plane_faces else None
        if isinstance(idx_label, int):
            part_id_str = f"part_{idx_label}"
        else:
            part_id_str = f"part_{idx_label}"
        part_record = {
            "id": part_id_str,
            "surface_type": "compound",
            "area": dominant["area"] if dominant else None,
            "normal": dominant["normal"] if dominant else None,
            "axis": None,
            "center": None,
            "radius": None,
            "bbox": bbox,
            "face_count": len(solid_faces),
            "plane_face_count": plane_face_count,
            "dominant_face_id": dominant["id"] if dominant else None,
        }
        # Cross-face hole dedup: a single physical hole shows up on
        # *both* faces of the plate (the top face's wire + the
        # bottom face's wire). Same kind, same 3D centre, same
        # diameter. Keep the one whose face normal is most aligned
        # with the dominant face's normal (the visible side).
        agg_holes = _dedupe_holes_across_faces(agg_holes, dominant)
        return {
            "target_face_id": part_id_str,
            "part": part_record,
            "per_face": per_face,
            "reference_points": agg_ref,
            "polylines": agg_polylines,
            "wires": agg_wires,
            "contours": agg_contours,
            "outer_contours": agg_outer,
            "holes": agg_holes,
        }

    # Whole-assembly aggregate path: fan out to every solid and merge the
    # results, de-duplicating by 3D geometry. We don't dedup by id because
    # every solid's ``analyze_face`` call generates ids like ``contour_0``
    # from a per-face counter — without a solid prefix those collide
    # across solids and we'd drop legitimate features.
    if solid_or_all is None:
        solids = list_solids(shape)
        merged: dict = {
            "target_face_id": "part_all",
            "part": None,
            "per_face": [],
            "reference_points": [],
            "polylines": [],
            "wires": [],
            "contours": [],
            "outer_contours": [],
            "holes": [],
        }
        seen_f: set[str] = set()
        total_face_count = 0
        total_plane_count = 0
        best_dominant = None
        for s_idx, solid in enumerate(solids):
            sub = _analyze_one_solid(solid, s_idx)
            total_face_count += sub["part"]["face_count"]
            total_plane_count += sub["part"]["plane_face_count"]
            if sub["part"].get("dominant_face_id") and (
                best_dominant is None
                or (sub["part"].get("area") or 0) > (best_dominant.get("area") or 0)
            ):
                best_dominant = {
                    "id": sub["part"]["dominant_face_id"],
                    "area": sub["part"].get("area"),
                    "normal": sub["part"].get("normal"),
                }
            # Prefix all ids with ``s{s_idx}__`` so they don't collide
            # across solids. Without this the user sees only the
            # features of the first solid; with it they get the
            # full multi-solid feature set.
            for f in sub["per_face"]:
                if f["id"] in seen_f:
                    continue
                seen_f.add(f["id"])
                merged["per_face"].append(f)
            for c in sub["contours"]:
                c2 = dict(c)
                c2["id"] = f"s{s_idx}__{c['id']}"
                c2["face_id"] = f"s{s_idx}__{c.get('face_id','')}"
                c2["wire_id"] = f"s{s_idx}__{c.get('wire_id','')}"
                c2["polyline_id"] = f"s{s_idx}__{c.get('polyline_id','')}"
                if c2.get("center") and c2.get("face_id"):
                    c2["center"] = c2["center"]  # 3D coords are unique
                merged["contours"].append(c2)
            for h in sub["holes"]:
                h2 = dict(h)
                h2["id"] = f"s{s_idx}__{h['id']}"
                h2["face_id"] = f"s{s_idx}__{h.get('face_id','')}"
                h2["wire_id"] = f"s{s_idx}__{h.get('wire_id','')}"
                merged["holes"].append(h2)
            for p in sub["polylines"]:
                p2 = dict(p)
                p2["id"] = f"s{s_idx}__{p['id']}"
                merged["polylines"].append(p2)
            for w in sub["wires"]:
                w2 = dict(w)
                w2["id"] = f"s{s_idx}__{w['id']}"
                w2["face_id"] = f"s{s_idx}__{w.get('face_id','')}"
                w2["polyline_id"] = f"s{s_idx}__{w.get('polyline_id','')}"
                w2["contour_id"] = f"s{s_idx}__{w.get('contour_id','')}"
                merged["wires"].append(w2)
            for r in sub["reference_points"]:
                r2 = dict(r)
                r2["id"] = f"s{s_idx}__{r['id']}"
                r2["meta"] = {**r.get("meta", {}), "solid_index": s_idx}
                merged["reference_points"].append(r2)
            merged["outer_contours"].extend(
                [f"s{s_idx}__{oid}" for oid in sub.get("outer_contours", [])]
            )
        # Synthetic compound record for the whole assembly
        merged["part"] = {
            "id": "part_all",
            "surface_type": "assembly",
            "area": best_dominant["area"] if best_dominant else None,
            "normal": best_dominant["normal"] if best_dominant else None,
            "axis": None,
            "center": None,
            "radius": None,
            "bbox": bbox,
            "face_count": total_face_count,
            "plane_face_count": total_plane_count,
            "dominant_face_id": best_dominant["id"] if best_dominant else None,
            "solid_count": len(solids),
        }
        return {
            "schema_version": "1.1",
            "unit": "mm",
            "target_face_id": merged["target_face_id"],
            "part": merged["part"],
            "per_face": merged["per_face"],
            "reference_points": merged["reference_points"],
            "polylines": merged["polylines"],
            "wires": merged["wires"],
            "contours": merged["contours"],
            "outer_contours": merged["outer_contours"],
            "outer_contour_ids": [c["id"] for c in merged["outer_contours"]],
            "holes": merged["holes"],
            "pockets": [],
            "feature_groups": _build_feature_groups(
                contours=merged["contours"],
                holes=merged["holes"],
                wires=merged["wires"],
            ),
            "work_plane": work_plane,
            "work_plane_normal": list(work_plane_normal) if work_plane_normal else None,
            "model_bbox": bbox,
        }

    # Single-solid path
    solid = solid_or_all
    sub = _analyze_one_solid(solid, solid_idx)
    return {
        "schema_version": "1.1",
        "unit": "mm",
        "target_face_id": sub["target_face_id"],
        "part": sub["part"],
        "per_face": sub["per_face"],
        "reference_points": sub["reference_points"],
        "polylines": sub["polylines"],
        "wires": sub["wires"],
        "contours": sub["contours"],
        "outer_contours": sub["outer_contours"],
        "outer_contour_ids": list(sub["outer_contours"]),
        "holes": sub["holes"],
        "pockets": [],
        "feature_groups": _build_feature_groups(
            contours=sub["contours"],
            holes=sub["holes"],
            wires=sub["wires"],
        ),
        "work_plane": work_plane,
        "work_plane_normal": list(work_plane_normal) if work_plane_normal else None,
        "model_bbox": bbox,
    }


def shape_bbox_dict(shape: TopoDS_Shape) -> dict:
    from app.occ.geometry_utils import shape_bbox

    xmin, ymin, zmin, xmax, ymax, zmax = shape_bbox(shape)
    return {
        "xmin": xmin, "ymin": ymin, "zmin": zmin,
        "xmax": xmax, "ymax": ymax, "zmax": zmax,
    }
