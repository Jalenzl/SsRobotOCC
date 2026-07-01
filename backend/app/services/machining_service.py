"""CAM path generation service — converts feature extraction results to machining paths.

This service implements the path planning layer inspired by the SmartLaser architecture:

    Feature Result (contour.py) → MachiningPath → CAMLine

Key responsibilities:
- Convert ContourFeature to MachiningPath with CAMLines
- Generate lead-in / lead-out lines (LeadLine / LeadOutLine)
- Classify path segments (long line, short line, arc, corner, etc.)
- Apply craft parameters based on contour type
- Handle both outer and inner (hole) contours
- Support multi-hole selection with click-order preservation
- Generate idle (transit) lines between adjacent paths
"""

from __future__ import annotations

import math
import uuid
from typing import Any

from app.models.feature import (
    CAMLine,
    CAMLineTypeLiteral,
    ContourFeature,
    CraftParameters,
    HoleFeature,
    InnerPathTypeLiteral,
    MachiningGroup,
    MachiningPath,
    MachiningResult,
    OutPathTypeLiteral,
    PathTypeLiteral,
    Point3D,
    Vector3D,
)


# ── Default craft parameters by contour type ─────────────────────────────────

_DEFAULT_CRAFT_PARAMS: dict[str, CraftParameters] = {
    "circle": CraftParameters(velocity=100.0, power=80, duty=50, frequency=5000),
    "slot": CraftParameters(velocity=80.0, power=75, duty=50, frequency=5000),
    "rectangle": CraftParameters(velocity=90.0, power=80, duty=50, frequency=5000),
    "hexagon": CraftParameters(velocity=85.0, power=80, duty=50, frequency=5000),
    "outer": CraftParameters(velocity=100.0, power=85, duty=50, frequency=5000),
    "unknown": CraftParameters(velocity=80.0, power=70, duty=50, frequency=5000),
}


# ── Per-segment velocity table (mm/s) keyed by OutPathType ──────────────────
# Mirrors the SmartLaser "走形工艺" table; segments with sharper curvature
# or shorter length run slower to maintain cut quality.

_SEGMENT_VELOCITY: dict[str, float] = {
    "long_line":     1.00,   # 长直线 → 全速
    "shorter_line":  0.80,   # 短直线 → 80%
    "shortest_line": 0.60,   # 最短直线 → 60%
    "big_arc":       0.70,   # 大圆弧 → 70%
    "small_arc":     0.50,   # 小圆弧 → 50%
    "three_d_corner":0.40,   # 三维拐角 → 40%
    "point":         0.30,   # 退化点 → 30%
}


# ── Segment classification thresholds ─────────────────────────────────────────

_LONG_LINE_RATIO       = 0.4   # 长直线: 段长 > 周长 × 0.4
_SHORT_LINE_RATIO      = 0.1   # 短直线: 段长 > 周长 × 0.1
_ARC_ANGLE_THRESHOLD   = 30.0  # 圆弧角度阈值 (度) — 3 点圆心角 > 该值视为圆弧
_BIG_ARC_ANGLE         = 60.0  # 大圆弧 / 小圆弧分界 (度)
_CORNER_ANGLE_THRESHOLD= 60.0  # 段间夹角阈值 (度) — 大于该值视为三维拐角
_POINT_LENGTH_TOL      = 0.05  # 长度小于该值 (mm) 视为退化点
_LEAD_IN_LENGTH        = 5.0   # 默认引线长度 (mm)
_LEAD_IN_RADIUS        = 2.0   # 默认引线圆弧半径 (mm)
_IDLE_VELOCITY         = 200.0 # 跨 path 空驶速度 (mm/s)
_RAPID_VELOCITY        = 300.0 # 跨 path 快进速度 (mm/s) — 无抬刀直跳


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _dist(a: list[float] | tuple[float, float, float],
          b: list[float] | tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def _sub(a: list[float], b: list[float]) -> list[float]:
    """3D vector subtraction."""
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _add(a: list[float], b: list[float]) -> list[float]:
    """3D vector addition."""
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _scale(v: list[float], s: float) -> list[float]:
    """3D vector scalar multiplication."""
    return [v[0] * s, v[1] * s, v[2] * s]


def _normalize(v: list[float]) -> list[float]:
    """Unit vector. Returns [0,0,0] if input is zero-length."""
    m = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if m < 1e-9:
        return [0.0, 0.0, 0.0]
    return [v[0] / m, v[1] / m, v[2] / m]


def _total_perimeter(points: list[list[float]]) -> float:
    """Total polyline perimeter (assumes points form a closed loop)."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        total += _dist(points[i - 1], points[i])
    # Close the loop
    total += _dist(points[-1], points[0])
    return total


def _is_collinear_3d(p0: list[float], p1: list[float], p2: list[float], tol: float = 1e-4) -> bool:
    """True if p0, p1, p2 are (almost) collinear (3D cross-product magnitude test)."""
    v01 = _sub(p1, p0)
    v02 = _sub(p2, p0)
    cross_mag = math.sqrt(
        (v01[1] * v02[2] - v01[2] * v02[1]) ** 2
        + (v01[2] * v02[0] - v01[0] * v02[2]) ** 2
        + (v01[0] * v02[1] - v01[1] * v02[0]) ** 2
    )
    v01_mag = math.sqrt(v01[0] ** 2 + v01[1] ** 2 + v01[2] ** 2)
    v02_mag = math.sqrt(v02[0] ** 2 + v02[1] ** 2 + v02[2] ** 2)
    if v01_mag < 1e-9 or v02_mag < 1e-9:
        return True
    # Perpendicular distance from p1 to line p0-p2, normalised by v02 length
    return (cross_mag / v02_mag) < tol


def _bend_angle_deg(p_prev: list[float], p_cur: list[float], p_next: list[float]) -> float:
    """Angle (in degrees) between vectors (p_cur-p_prev) and (p_next-p_cur).

    Returns 0 if either segment is degenerate.  Result is in [0, 180].
    """
    v1 = _sub(p_cur, p_prev)
    v2 = _sub(p_next, p_cur)
    m1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    m2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if m1 < 1e-6 or m2 < 1e-6:
        return 0.0
    u1 = _scale(v1, 1.0 / m1)
    u2 = _scale(v2, 1.0 / m2)
    dot = u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2]
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _arc_angle_deg(p0: list[float], p1: list[float], p2: list[float]) -> float:
    """Estimate the central angle (in degrees) of a circular arc through 3 points.

    Uses the inscribed-angle theorem.  If the three points are (nearly)
    collinear, returns 0 (i.e. "no detectable curvature").

    For a smooth arc the central angle is in (0, 360).  For a 90° corner
    (two segments meeting at right angles) the central angle is 180° — i.e.
    the *bend*, not curvature.
    """
    if _is_collinear_3d(p0, p1, p2, tol=1e-3):
        return 0.0
    v01 = _sub(p1, p0)
    v12 = _sub(p2, p1)
    v02 = _sub(p2, p0)
    len01 = math.sqrt(v01[0] ** 2 + v01[1] ** 2 + v01[2] ** 2)
    len12 = math.sqrt(v12[0] ** 2 + v12[1] ** 2 + v12[2] ** 2)
    if len01 < 1e-6 or len12 < 1e-6:
        return 0.0
    u01 = _scale(v01, 1.0 / len01)
    u21 = _scale(_sub(p1, p2), 1.0 / len12)
    dot = u01[0] * u21[0] + u01[1] * u21[1] + u01[2] * u21[2]
    dot = max(-1.0, min(1.0, dot))
    inscribed = math.degrees(math.acos(dot))
    central = inscribed * 2.0
    # The 3-point inscribed-angle test gives 180° (central) both for
    # collinear points (straight line) and for a 90° sharp corner where
    # the apex is the middle point.  In both cases there is **no**
    # curvature here, so we treat it as a non-arc.
    if central >= 179.5 or central <= 0.5:
        return 0.0
    return central


def _classify_segment(
    p_prev: list[float] | None,
    p_cur: list[float],
    p_next: list[float] | None,
    seg_length: float,
    total_length: float,
) -> OutPathTypeLiteral:
    """Classify a single polyline segment into an OutPathTypeLiteral.

    Rules (applied in order):
    1. Length below `_POINT_LENGTH_TOL` → 'point'
    2. Curvature check: 3-point arc angle in `(0, 360)` →
       'big_arc' (>= `_BIG_ARC_ANGLE`) or 'small_arc' (>= `_ARC_ANGLE_THRESHOLD`).
       A 180° arc (i.e. 3 collinear points) is **not** curvature — it's a
       straight line, so we fall through to rule 4.
    3. Sharp bend: angle between prev-segment and next-segment
       >= `_CORNER_ANGLE_THRESHOLD` (e.g. 60°) → 'three_d_corner'.
       This catches sharp 90° / 120° turns that the inscribed-angle test
       would otherwise label as a half-circle arc.
    4. Length vs total_length ratio:
       - > `_LONG_LINE_RATIO`  → 'long_line'
       - > `_SHORT_LINE_RATIO` → 'shorter_line'
       - else                  → 'shortest_line'

    Falls back to 'shorter_line' when neighbours are unavailable.
    """
    # Rule 1: degenerate point
    if seg_length < _POINT_LENGTH_TOL:
        return "point"

    # Rule 2: arc detection (true curvature only)
    if p_prev is not None and p_next is not None:
        angle = _arc_angle_deg(p_prev, p_cur, p_next)
        # Exclude the "full circle" / "180° straight-line" degenerate case
        if 0.0 < angle < 359.0:
            if angle >= _BIG_ARC_ANGLE:
                return "big_arc"
            if angle >= _ARC_ANGLE_THRESHOLD:
                return "small_arc"

        # Rule 3: 3-D corner — sharp bend between prev and next segments.
        # Only flag a corner when the 3 points are not collinear (otherwise
        # the bend is exactly 180° for a straight line).
        bend = _bend_angle_deg(p_prev, p_cur, p_next)
        if not _is_collinear_3d(p_prev, p_cur, p_next, tol=1e-3):
            # bend=0 means continue forward, bend=180 means turn back.
            # We treat large bend as corner (>= threshold).
            if bend >= _CORNER_ANGLE_THRESHOLD:
                return "three_d_corner"

    # Rule 4: length-based classification
    if total_length <= 0:
        return "shorter_line"
    ratio = seg_length / total_length
    if ratio >= _LONG_LINE_RATIO:
        return "long_line"
    if ratio >= _SHORT_LINE_RATIO:
        return "shorter_line"
    return "shortest_line"


# ── Core conversion functions ────────────────────────────────────────────────

def _contour_inner_type(contour_type: str) -> InnerPathTypeLiteral | None:
    """Map a contour_type string to InnerPathTypeLiteral; fall back to 'irregular'."""
    mapping: dict[str, InnerPathTypeLiteral] = {
        "circle":    "circle",
        "slot":      "slot",
        "rectangle": "rectangle",
        "hexagon":   "hexagon",
    }
    return mapping.get(contour_type, "irregular")


def _make_cam_line(
    *,
    line_id: str,
    line_type: CAMLineTypeLiteral,
    path_type: PathTypeLiteral,
    start: list[float],
    end: list[float],
    normal: Vector3D | None,
    velocity: float,
    power: int,
    duty: int,
    inner_type: InnerPathTypeLiteral | None = None,
    out_type: OutPathTypeLiteral | None = None,
    is_clockwise: bool = True,
    order_index: int = 0,
) -> CAMLine:
    """Construct a CAMLine with consistent field population."""
    return CAMLine(
        id=line_id,
        line_type=line_type,
        path_type=path_type,
        inner_type=inner_type,
        out_type=out_type,
        start_point=Point3D(root=list(start)),
        end_point=Point3D(root=list(end)),
        normal=normal,
        velocity=velocity,
        power=power,
        duty=duty,
        is_clockwise=is_clockwise,
        order_index=order_index,
        robot_joints=[],
    )


def _generate_cam_lines_from_points(
    points: list[list[float]],
    path_id: str,
    path_type: PathTypeLiteral,
    contour_type: str,
    params: CraftParameters,
    center: Point3D | None,
    normal: Vector3D | None,
    inner_type: InnerPathTypeLiteral | None = None,
) -> list[CAMLine]:
    """Generate CAMLines from a list of 3D points (a closed polyline).

    Each point pair → one CAMLine.  Each line is classified with
    ``_classify_segment`` and its velocity is scaled by the per-segment
    velocity table.
    """
    if len(points) < 2:
        return []

    lines: list[CAMLine] = []
    total_length = _total_perimeter(points)

    n = len(points)
    for i in range(1, n):
        p_prev = points[i - 1] if i > 0 else None
        p_cur  = points[i]
        p_next = points[i + 1] if i + 1 < n else points[0]  # close the loop
        seg_length = _dist(p_prev or p_cur, p_cur)

        out_type = _classify_segment(p_prev, p_cur, p_next, seg_length, total_length)
        velocity = params.velocity * _SEGMENT_VELOCITY.get(out_type, 1.0)

        lines.append(_make_cam_line(
            line_id=f"{path_id}_line_{i}",
            line_type="machining",
            path_type=path_type,
            start=list(p_prev) if p_prev is not None else list(p_cur),
            end=list(p_cur),
            normal=normal,
            velocity=velocity,
            power=params.power,
            duty=params.duty,
            inner_type=inner_type,
            out_type=out_type,
            is_clockwise=True,
            order_index=i,
        ))

    return lines


def _generate_lead_line(
    points: list[list[float]],
    path_id: str,
    path_type: PathTypeLiteral,
    params: CraftParameters,
    lead_length: float,
    lead_radius: float,
    center: Point3D | None,
    normal: Vector3D | None,
    reverse_direction: bool = False,
) -> CAMLine | None:
    """Generate a lead-in line for a contour.

    Direction: tangent to the contour at the entry point, **extended outward**
    by `lead_length`.  For holes (`reverse_direction=True`) we approach from
    the *outside* of the hole, so the start point is set *beyond* the first
    sample point along the reversed tangent.
    """
    if len(points) < 2:
        return None

    if reverse_direction:
        # Holes: approach from outside; entry point is the *last* sample
        entry_idx = len(points) - 1
        if entry_idx <= 0:
            return None
        entry_pt = points[entry_idx]
        prev_pt  = points[entry_idx - 1]
        # Tangent at entry points from prev → entry; outward is the reverse
        tangent = _normalize(_sub(entry_pt, prev_pt))
    else:
        entry_pt = points[0]
        next_pt  = points[1] if len(points) > 1 else entry_pt
        tangent  = _normalize(_sub(next_pt, entry_pt))

    if tangent == [0.0, 0.0, 0.0]:
        return None

    # Lead-in start: extend `lead_length` opposite to the cutting direction
    sign = -1.0 if reverse_direction else -1.0
    lead_start = _add(entry_pt, _scale(tangent, sign * lead_length))

    return _make_cam_line(
        line_id=f"{path_id}_lead",
        line_type="lead",
        path_type=path_type,
        start=lead_start,
        end=entry_pt,
        normal=normal,
        velocity=params.velocity * 0.5,  # Slow down for lead-in
        power=params.power,
        duty=params.duty,
        is_clockwise=True,
        order_index=0,
    )


def _generate_lead_out_line(
    points: list[list[float]],
    path_id: str,
    path_type: PathTypeLiteral,
    params: CraftParameters,
    lead_length: float,
    normal: Vector3D | None,
) -> CAMLine | None:
    """Generate a lead-out line (退刀线) at the contour's exit point.

    The exit is the last sample point; the line extends along the cutting
    tangent beyond it.  Mirrors `_generate_lead_line`.
    """
    if len(points) < 2:
        return None
    exit_pt = points[-1]
    prev_pt = points[-2] if len(points) > 1 else exit_pt
    tangent = _normalize(_sub(exit_pt, prev_pt))
    if tangent == [0.0, 0.0, 0.0]:
        return None
    lead_end = _add(exit_pt, _scale(tangent, lead_length))
    return _make_cam_line(
        line_id=f"{path_id}_lead_out",
        line_type="lead",
        path_type=path_type,
        start=exit_pt,
        end=lead_end,
        normal=normal,
        velocity=params.velocity * 0.5,
        power=params.power,
        duty=params.duty,
        is_clockwise=True,
        order_index=len(points) + 1,
    )


def _generate_idle_line(
    *,
    line_id: str,
    from_point: list[float],
    to_point: list[float],
    path_type: PathTypeLiteral,
    normal: Vector3D | None,
    rapid: bool = False,
) -> CAMLine:
    """Generate a transit/idle CAMLine between two non-adjacent points.

    These represent the "rapid traverse" between the end of one path and
    the start of the next — the robot lifts the tool, traverses, and
    re-enters.  No power / duty is applied (laser is OFF during transit).
    """
    return _make_cam_line(
        line_id=line_id,
        line_type="idle",
        path_type=path_type,
        start=from_point,
        end=to_point,
        normal=normal,
        velocity=_RAPID_VELOCITY if rapid else _IDLE_VELOCITY,
        power=0,
        duty=0,
        inner_type=None,
        out_type=None,
        is_clockwise=True,
        order_index=0,
    )


# ── High-level: single contour / hole → MachiningPath ────────────────────────

def _wrap_point3d(raw: Any) -> Point3D | None:
    """Coerce a raw 3-vector into a ``Point3D``.

    The OCC feature-extraction module serialises ``center`` / ``axis`` /
    ``normal`` as bare ``[x, y, z]`` lists, but the Pydantic models
    require ``Point3D`` / ``Vector3D`` instances.  Accept both.
    """
    if raw is None:
        return None
    if isinstance(raw, Point3D):
        return raw
    if isinstance(raw, Vector3D):
        return Point3D(root=list(raw.root))
    if isinstance(raw, dict):
        return Point3D(**raw)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return Point3D(root=[float(raw[0]), float(raw[1]), float(raw[2])])
    raise TypeError(f"Cannot coerce {type(raw).__name__} to Point3D")


def _wrap_vector3d(raw: Any) -> Vector3D | None:
    """Coerce a raw 3-vector into a ``Vector3D``."""
    if raw is None:
        return None
    if isinstance(raw, Vector3D):
        return raw
    if isinstance(raw, Point3D):
        return Vector3D(root=list(raw.root))
    if isinstance(raw, dict):
        return Vector3D(**raw)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return Vector3D(root=[float(raw[0]), float(raw[1]), float(raw[2])])
    raise TypeError(f"Cannot coerce {type(raw).__name__} to Vector3D")


def _coerce_hole_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a hole dict so it can be passed to ``HoleFeature(**)``.

    - Wraps ``center`` / ``axis`` (and any other nested point/vector
      fields) into ``Point3D`` / ``Vector3D``.
    - Coerces ``parameters`` dict into ``ContourParameters``.
    - Returns a new dict; the input is not mutated.
    """
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)  # shallow copy
    if "center" in out:
        out["center"] = _wrap_point3d(out["center"])
    if "axis" in out:
        out["axis"] = _wrap_vector3d(out["axis"])
    # Some legacy serialisations store `normal` instead of `axis`
    # Only fall back when `axis` is absent / None after coercion
    if out.get("axis") is None:
        if "normal" in out and out["normal"] is not None:
            out["axis"] = _wrap_vector3d(out["normal"])
    if "parameters" in out and isinstance(out["parameters"], dict):
        try:
            out["parameters"] = ContourParameters(**out["parameters"])
        except Exception:
            pass
    if "cylindrical_face_ids" in out and out["cylindrical_face_ids"] is None:
        out["cylindrical_face_ids"] = []
    return out


def _coerce_contour_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Same idea as ``_coerce_hole_dict`` but for contour dicts."""
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    if "center" in out:
        out["center"] = _wrap_point3d(out["center"])
    if "normal" in out:
        out["normal"] = _wrap_vector3d(out["normal"])
    if "parameters" in out and isinstance(out["parameters"], dict):
        try:
            out["parameters"] = ContourParameters(**out["parameters"])
        except Exception:
            pass
    return out


def _hole_from_raw(raw: Any) -> HoleFeature:
    """Build a ``HoleFeature`` from a dict (or pass through an existing instance)."""
    if isinstance(raw, HoleFeature):
        return raw
    if isinstance(raw, dict):
        return HoleFeature(**_coerce_hole_dict(raw))
    raise TypeError(f"Cannot build HoleFeature from {type(raw).__name__}")


def _contour_from_raw(raw: Any) -> ContourFeature:
    if isinstance(raw, ContourFeature):
        return raw
    if isinstance(raw, dict):
        return ContourFeature(**_coerce_contour_dict(raw))
    raise TypeError(f"Cannot build ContourFeature from {type(raw).__name__}")


def _resolve_points_for_feature(
    feature: HoleFeature | ContourFeature,
    polylines: dict[str, list[list[float]]] | None,
    wires_by_id: dict[str, str] | None,
) -> list[list[float] | None] | list[list[float]]:
    """Resolve the polyline points that belong to a single hole / contour.

    Returns a tuple of (points, None) when the feature has a known
    polyline_id; otherwise (None, None) — the caller can fall back to
    heuristics.

    Strategy (in priority order):
    1. feature.polyline_id (ContourFeature only)
    2. wires_by_id[feature.wire_id] → polyline_id
    """
    poly_id: str | None = None
    if isinstance(feature, ContourFeature) and feature.polyline_id:
        poly_id = feature.polyline_id
    elif feature.wire_id and wires_by_id:
        poly_id = wires_by_id.get(feature.wire_id)
    if poly_id and polylines and poly_id in polylines:
        return polylines[poly_id]
    return None


def contour_to_machining_path(
    contour: ContourFeature,
    polylines: dict[str, list[list[float]]] | None = None,
    wires_by_id: dict[str, str] | None = None,
    craft_params: CraftParameters | None = None,
    lead_in_length: float = _LEAD_IN_LENGTH,
    lead_in_radius: float = _LEAD_IN_RADIUS,
    source_contour_id: str | None = None,
    order_index: int = 0,
) -> MachiningPath:
    """Convert a ContourFeature to a MachiningPath with CAMLines.

    Args:
        source_contour_id: Override `contour.id` for the resulting path's
            back-reference (used when grouping multiple contours under a
            single parent).
        order_index: 0-based index in the parent's path list.
    """
    params = craft_params or _DEFAULT_CRAFT_PARAMS.get(
        contour.contour_type, _DEFAULT_CRAFT_PARAMS["unknown"]
    )

    path_id = f"path_{uuid.uuid4().hex[:8]}"
    path_type: PathTypeLiteral = "inner" if not contour.is_outer else "outer"

    cam_lines: list[CAMLine] = []
    lead_line: CAMLine | None = None
    lead_out_line: CAMLine | None = None

    points = _resolve_points_for_feature(contour, polylines, wires_by_id)
    if points and len(points) >= 2:
        cam_lines = _generate_cam_lines_from_points(
            points=points,
            path_id=path_id,
            path_type=path_type,
            contour_type=contour.contour_type,
            params=params,
            center=contour.center,
            normal=contour.normal,
        )
        if contour.center:
            lead_line = _generate_lead_line(
                points=points,
                path_id=path_id,
                path_type=path_type,
                params=params,
                lead_length=lead_in_length,
                lead_radius=lead_in_radius,
                center=contour.center,
                normal=contour.normal,
                reverse_direction=not contour.is_outer,
            )
            lead_out_line = _generate_lead_out_line(
                points=points,
                path_id=path_id,
                path_type=path_type,
                params=params,
                lead_length=lead_in_length,
                normal=contour.normal,
            )

    return MachiningPath(
        id=path_id,
        name=f"{contour.contour_type}_{'outer' if contour.is_outer else 'inner'}",
        path_type=path_type,
        contour_id=contour.id,
        contour_type=contour.contour_type,
        source_contour_id=source_contour_id or contour.id,
        cam_lines=cam_lines,
        lead_line=lead_line,
        lead_out_line=lead_out_line,
        idle_lines=[],
        thickness=1.0,
        normal_reversed=not contour.is_outer,
        is_removed=False,
        order_index=order_index,
    )


def hole_to_machining_path(
    hole: HoleFeature,
    polylines: dict[str, list[list[float]]] | None = None,
    wires_by_id: dict[str, str] | None = None,
    craft_params: CraftParameters | None = None,
    lead_in_length: float = _LEAD_IN_LENGTH,
    source_hole_id: str | None = None,
    order_index: int = 0,
) -> MachiningPath:
    """Convert a HoleFeature to a MachiningPath.

    Args:
        source_hole_id: Override `hole.id` for the resulting path's
            back-reference.
        order_index: 0-based index in the inner_paths list.
    """
    params = craft_params or _DEFAULT_CRAFT_PARAMS.get(
        hole.contour_type, _DEFAULT_CRAFT_PARAMS["unknown"]
    )

    path_id = f"path_{uuid.uuid4().hex[:8]}"
    inner_type: InnerPathTypeLiteral = _contour_inner_type(hole.contour_type) or "irregular"

    cam_lines: list[CAMLine] = []
    lead_line: CAMLine | None = None
    lead_out_line: CAMLine | None = None

    points = _resolve_points_for_feature(hole, polylines, wires_by_id)
    if points and len(points) >= 2:
        cam_lines = _generate_cam_lines_from_points(
            points=points,
            path_id=path_id,
            path_type="inner",
            contour_type=hole.contour_type,
            params=params,
            center=hole.center,
            normal=hole.axis,
            inner_type=inner_type,
        )
        if hole.center:
            lead_line = _generate_lead_line(
                points=points,
                path_id=path_id,
                path_type="inner",
                params=params,
                lead_length=lead_in_length,
                lead_radius=0,  # Holes use direct lead-in
                center=hole.center,
                normal=hole.axis,
                reverse_direction=True,
            )
            lead_out_line = _generate_lead_out_line(
                points=points,
                path_id=path_id,
                path_type="inner",
                params=params,
                lead_length=lead_in_length,
                normal=hole.axis,
            )

    return MachiningPath(
        id=path_id,
        name=f"hole_{hole.contour_type}",
        path_type="inner",
        contour_id=hole.id,
        contour_type=hole.contour_type,
        source_hole_id=source_hole_id or hole.id,
        cam_lines=cam_lines,
        lead_line=lead_line,
        lead_out_line=lead_out_line,
        idle_lines=[],
        thickness=1.0,
        normal_reversed=True,  # Holes typically need reversed normal
        is_removed=False,
        order_index=order_index,
    )


# ── Public orchestration ─────────────────────────────────────────────────────

def _index_polylines(feature_result: dict[str, Any]) -> dict[str, list[list[float]]]:
    """Build a polyline_id → point-list lookup from the feature extraction result."""
    out: dict[str, list[list[float]]] = {}
    for polyline in feature_result.get("polylines", []):
        pid = polyline.get("id")
        if not pid or "points" not in polyline:
            continue
        raw_points = polyline["points"]
        out[pid] = [
            (p.root if isinstance(p, dict) else p) for p in raw_points
        ]
    return out


def _index_wires(feature_result: dict[str, Any]) -> dict[str, str]:
    """Build a wire_id → polyline_id lookup from the feature extraction result."""
    out: dict[str, str] = {}
    for w in feature_result.get("wires", []) or []:
        wid = w.get("id")
        pid = w.get("polyline_id")
        if wid and pid:
            out[wid] = pid
    return out


def _select_holes_by_ids(
    feature_result: dict[str, Any],
    hole_ids: list[str] | None,
) -> list[HoleFeature]:
    """Return HoleFeature objects in the order specified by `hole_ids`.

    - If `hole_ids` is None / empty: return all holes in feature-extraction order.
    - If `hole_ids` is provided: preserve that order, skip unknown ids, but
      raise ValueError if **none** of the requested ids match (suggests a
      bad request rather than a benign empty result).

    Hole dicts from the OCC feature-extraction module often contain bare
    ``[x, y, z]`` lists for ``center`` / ``axis``; this function wraps them
    into ``Point3D`` / ``Vector3D`` so the ``HoleFeature`` Pydantic model
    accepts them.
    """
    raw_holes = feature_result.get("holes", []) or []
    all_holes = [_hole_from_raw(h) for h in raw_holes]
    by_id = {h.id: h for h in all_holes}

    if not hole_ids:
        return all_holes

    selected: list[HoleFeature] = []
    for hid in hole_ids:
        if hid in by_id:
            selected.append(by_id[hid])
    if not selected:
        raise ValueError(
            f"None of the requested hole_ids were found: {hole_ids!r}. "
            f"Available: {list(by_id.keys())[:10]}{'...' if len(by_id) > 10 else ''}"
        )
    return selected


def _select_outer_contours(feature_result: dict[str, Any]) -> list[ContourFeature]:
    """Return all outer contours in feature-extraction order."""
    out: list[ContourFeature] = []
    for c in feature_result.get("contours", []) or []:
        cf = _contour_from_raw(c)
        if cf.is_outer:
            out.append(cf)
    return out


def _path_entry_point(path: MachiningPath) -> list[float] | None:
    """Return the first 3D point of a MachiningPath (lead-in start, or first CAMLine start)."""
    if path.lead_line is not None:
        return list(path.lead_line.start_point.root)
    if path.cam_lines:
        return list(path.cam_lines[0].start_point.root)
    return None


def _path_exit_point(path: MachiningPath) -> list[float] | None:
    """Return the last 3D point of a MachiningPath (lead-out end, or last CAMLine end)."""
    if path.lead_out_line is not None:
        return list(path.lead_out_line.end_point.root)
    if path.cam_lines:
        return list(path.cam_lines[-1].end_point.root)
    return None


def _build_transition_lines(
    paths: list[MachiningPath],
    path_type: PathTypeLiteral,
    normal: Vector3D | None,
) -> list[CAMLine]:
    """Build idle/transition CAMLines connecting adjacent paths.

    The resulting list has length ``len(paths) - 1``: each transition
    connects the *exit* of ``paths[i]`` to the *entry* of ``paths[i+1]``.
    """
    if len(paths) < 2:
        return []
    transitions: list[CAMLine] = []
    for i in range(len(paths) - 1):
        from_pt = _path_exit_point(paths[i])
        to_pt   = _path_entry_point(paths[i + 1])
        if from_pt is None or to_pt is None:
            continue
        transitions.append(_generate_idle_line(
            line_id=f"idle_{paths[i].id}_to_{paths[i + 1].id}",
            from_point=from_pt,
            to_point=to_pt,
            path_type=path_type,
            normal=normal,
            rapid=True,
        ))
    return transitions


def generate_machining_paths(
    feature_result: dict[str, Any],
    *,
    apply_craft_params: bool = True,
    generate_lead_lines: bool = True,
) -> MachiningResult:
    """Convert a feature extraction result to machining paths.

    This is the legacy entry point — keeps the original behaviour:
    - includes **all** holes in feature-extraction order
    - includes **all** outer contours
    - no explicit cross-path idle lines

    For multi-hole click-order selection, use ``generate_machining_paths_multi``.
    """
    machining_groups: list[MachiningGroup] = []
    polylines = _index_polylines(feature_result)
    wires_by_id = _index_wires(feature_result)

    # Holes → inner paths
    inner_paths: list[MachiningPath] = []
    for hole in feature_result.get("holes", []):
        hole_feature = _hole_from_raw(hole)
        path = hole_to_machining_path(
            hole=hole_feature,
            polylines=polylines,
            wires_by_id=wires_by_id,
            craft_params=(
                _DEFAULT_CRAFT_PARAMS.get(hole_feature.contour_type)
                if apply_craft_params else None
            ),
            order_index=len(inner_paths),
        )
        inner_paths.append(path)

    # Outer contours → outer paths
    outer_paths: list[MachiningPath] = []
    for contour in feature_result.get("contours", []):
        contour_feature = _contour_from_raw(contour)
        if contour_feature.is_outer:
            path = contour_to_machining_path(
                contour=contour_feature,
                polylines=polylines,
                wires_by_id=wires_by_id,
                craft_params=(
                    _DEFAULT_CRAFT_PARAMS.get(contour_feature.contour_type)
                    if apply_craft_params else None
                ),
                order_index=len(outer_paths),
            )
            outer_paths.append(path)

    group = MachiningGroup(
        id=f"group_{uuid.uuid4().hex[:8]}",
        name="Default Machining Group",
        inner_paths=inner_paths,
        outer_paths=outer_paths,
        process_face_ids=[feature_result.get("target_face_id", "unknown")],
        is_merged=False,
    )
    # path_order: outer (if any) first, then inner — front-end can override
    group.path_order = [p.id for p in outer_paths] + [p.id for p in inner_paths]
    machining_groups.append(group)

    total_paths = len(inner_paths) + len(outer_paths)
    total_lines = sum(len(p.cam_lines) for p in inner_paths + outer_paths)

    return MachiningResult(
        schema_version="2.0",
        unit="mm",
        model_id=feature_result.get("model_id", "unknown"),
        feature_result=feature_result,
        machining_groups=machining_groups,
        total_path_count=total_paths,
        total_line_count=total_lines,
    )


def generate_machining_paths_multi(
    feature_result: dict[str, Any],
    hole_ids: list[str],
    *,
    include_outer: bool = False,
    apply_craft_params: bool = True,
    generate_lead_lines: bool = True,
    idle_velocity: float | None = None,
) -> MachiningResult:
    """Generate machining paths for a user-selected set of holes in click order.

    Front-end flow:
        1. User clicks / multi-selects holes on the 3D viewer
        2. Front-end maintains an ordered list of `hole_id` (click order)
        3. Front-end POSTs that list to ``/api/v1/cad/machining/paths/multi``
        4. This function:
           - Filters the feature extraction result by `hole_ids`
           - Preserves `hole_ids` order in `inner_paths`
           - Fills `order_index` on each MachiningPath
           - Generates cross-path idle (transit) CAMLines
           - Optionally appends outer contours at the end

    Args:
        feature_result: Output from ``feature_service.analyze_face_spread`` /
            ``analyze_part_spread``.
        hole_ids: Hole IDs in the order they were clicked. Unknown IDs are
            silently skipped; if **none** match, a ValueError is raised.
        include_outer: If True, append all outer contours after the holes
            in the same group.  Default False (pure multi-hole mode).
        apply_craft_params: Apply default craft parameters by contour type.
        generate_lead_lines: Currently informational — lead lines are always
            generated in this service; flag is kept for API symmetry.
        idle_velocity: Override the default idle (transit) velocity (mm/s).

    Returns:
        MachiningResult with a single MachiningGroup whose ``inner_paths``
        are in click order and ``path_order`` encodes the full sequence
        (holes → optional outers) for the front-end 3D simulator.
    """
    polylines = _index_polylines(feature_result)
    wires_by_id = _index_wires(feature_result)
    selected_holes = _select_holes_by_ids(feature_result, hole_ids)

    # Override global idle velocity if requested (must come before
    # _build_transition_lines call).  Stored on the function-local default.
    global _RAPID_VELOCITY  # noqa: PLW0127  (re-assigned below only in scope)
    if idle_velocity is not None and idle_velocity > 0:
        _RAPID_VELOCITY_ORIG = _RAPID_VELOCITY  # noqa: F841
        # NOTE: we cannot easily rebind the module-level constant used by
        # _generate_idle_line; instead we let callers override via
        # `idle_velocity` parameter to _build_transition_lines directly.
        # See the explicit pass-through below.

    # Inner paths in click order
    inner_paths: list[MachiningPath] = []
    for idx, hole in enumerate(selected_holes):
        path = hole_to_machining_path(
            hole=hole,
            polylines=polylines,
            wires_by_id=wires_by_id,
            craft_params=(
                _DEFAULT_CRAFT_PARAMS.get(hole.contour_type)
                if apply_craft_params else None
            ),
            source_hole_id=hole.id,
            order_index=idx,
        )
        inner_paths.append(path)

    # Outer contours (optional)
    outer_paths: list[MachiningPath] = []
    if include_outer:
        for idx, contour in enumerate(_select_outer_contours(feature_result)):
            path = contour_to_machining_path(
                contour=contour,
                polylines=polylines,
                wires_by_id=wires_by_id,
                craft_params=(
                    _DEFAULT_CRAFT_PARAMS.get(contour.contour_type)
                    if apply_craft_params else None
                ),
                source_contour_id=contour.id,
                order_index=idx,
            )
            outer_paths.append(path)

    # Build the group's path_order: holes first (in click order),
    # then outer contours.  This is the canonical sequence for the
    # 3D simulator.
    path_order: list[str] = [p.id for p in inner_paths] + [p.id for p in outer_paths]

    # Cross-path idle lines (only between adjacent inner paths, since outers
    # are usually processed last as a separate stage).
    transition_lines = _build_transition_lines(
        inner_paths, path_type="inner", normal=None
    )

    # Propagate idle velocity override
    if idle_velocity is not None and idle_velocity > 0:
        for line in transition_lines:
            line.velocity = idle_velocity

    group = MachiningGroup(
        id=f"group_{uuid.uuid4().hex[:8]}",
        name=(
            f"MultiHole[{len(inner_paths)}]"
            if inner_paths and not outer_paths
            else f"MultiHole[{len(inner_paths)}]+Outer[{len(outer_paths)}]"
        ),
        inner_paths=inner_paths,
        outer_paths=outer_paths,
        process_face_ids=[feature_result.get("target_face_id", "unknown")],
        is_merged=False,
        path_order=path_order,
        transition_lines=transition_lines,
    )
    machining_groups = [group]

    total_paths = len(inner_paths) + len(outer_paths)
    total_lines = (
        sum(len(p.cam_lines) for p in inner_paths + outer_paths)
        + len(transition_lines)
    )

    return MachiningResult(
        schema_version="2.0",
        unit="mm",
        model_id=feature_result.get("model_id", "unknown"),
        feature_result=feature_result,
        machining_groups=machining_groups,
        total_path_count=total_paths,
        total_line_count=total_lines,
    )


# ── Public lookup helpers ─────────────────────────────────────────────────────

def get_craft_parameters_by_contour_type(
    contour_type: str,
    thickness: float | None = None,
) -> CraftParameters:
    """Get craft parameters for a specific contour type.

    In a full implementation, this would load from a recipe database
    based on contour type, material thickness, and other factors.

    Args:
        contour_type: The contour type (circle, slot, rectangle, etc.)
        thickness: Optional material thickness for parameter scaling

    Returns:
        CraftParameters for the given contour type
    """
    params = _DEFAULT_CRAFT_PARAMS.get(
        contour_type, _DEFAULT_CRAFT_PARAMS["unknown"]
    ).model_copy()

    # Scale parameters based on thickness if provided
    if thickness and thickness > 0:
        thickness_factor = min(2.0, thickness / 1.0)  # Cap at 2x for 2mm
        params.velocity = params.velocity / math.sqrt(thickness_factor)
        params.power = min(100, int(params.power * math.sqrt(thickness_factor)))

    return params
