"""CAM path generation service — converts feature extraction results to machining paths.

This service implements the path planning layer inspired by the SmartLaser architecture:

    Feature Result (contour.py) → MachiningPath → CAMLine

Key responsibilities:
- Convert ContourFeature to MachiningPath with CAMLines
- Generate lead-in / lead-out lines (LeadLine)
- Classify path segments (long line, short line, arc, corner, etc.)
- Apply craft parameters based on contour type
- Handle both outer and inner (hole) contours
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


# ── Segment classification thresholds ─────────────────────────────────────────

_LONG_LINE_RATIO = 0.4      # 长直线: 长度 > 周长 * 0.4  
_SHORT_LINE_RATIO = 0.1     # 短直线: 长度 > 周长 * 0.1  
_ARC_ANGLE_THRESHOLD = 30.0 # 平面圆弧角度阈值 (度) — 用于把「3 点偏差」换算成角度, 大于该值视为大/小圆弧
_CORNER_ANGLE_THRESHOLD = 60.0  # 段间夹角阈值 (度) — 大于该值视为三维拐角
_POINT_LENGTH_TOL = 0.05     # 长度小于该值(mm)视为退化点 
_LEAD_IN_LENGTH = 5.0        # 默认引线长度 (mm)
_LEAD_IN_RADIUS = 2.0        # 默认引线圆弧半径 (mm)


# ── Core conversion functions ───────────────────────────────────────────────────
"""轮廓->加工路径"""
def contour_to_machining_path(
    contour: ContourFeature,
    polylines: dict[str, list[list[float]]] | None = None,
    craft_params: CraftParameters | None = None,
    lead_in_length: float = _LEAD_IN_LENGTH,
    lead_in_radius: float = _LEAD_IN_RADIUS,
) -> MachiningPath:
    """Convert a ContourFeature to a MachiningPath with CAMLines.

    Args:
        contour: The contour feature to convert
        polylines: Optional dict mapping polyline_id to point lists
        craft_params: Override craft parameters
        lead_in_length: Lead-in line length (mm)
        lead_in_radius: Lead-in arc radius (mm)

    Returns:
        MachiningPath with CAMLines, lead_line, and idle_lines
    """
    params = craft_params or _DEFAULT_CRAFT_PARAMS.get(
        contour.contour_type, _DEFAULT_CRAFT_PARAMS["unknown"]
    )

    path_id = f"path_{uuid.uuid4().hex[:8]}"

    # Determine path type
    path_type: PathTypeLiteral = "inner" if not contour.is_outer else "outer"

    # Get center and normal
    center = contour.center
    normal = contour.normal

    # Generate CAMLines from polyline if available
    cam_lines: list[CAMLine] = []
    lead_line: CAMLine | None = None
    idle_lines: list[CAMLine] = []

    if polylines and contour.polyline_id and contour.polyline_id in polylines:
        points = polylines[contour.polyline_id]
        cam_lines = _generate_cam_lines_from_points(
            points=points,
            path_id=path_id,
            path_type=path_type,
            contour_type=contour.contour_type,
            params=params,
            center=center,
            normal=normal,
        )

        # Generate lead-in line
        if len(points) >= 2 and center:
            lead_line = _generate_lead_line(
                points=points,
                path_id=path_id,
                path_type=path_type,
                params=params,
                lead_length=lead_in_length,
                lead_radius=lead_in_radius,
                center=center,
                normal=normal,
            )

    return MachiningPath(
        id=path_id,
        name=f"{contour.contour_type}_{'outer' if contour.is_outer else 'inner'}",
        path_type=path_type,
        contour_id=contour.id,
        contour_type=contour.contour_type,
        cam_lines=cam_lines,
        lead_line=lead_line,
        idle_lines=idle_lines,
        thickness=1.0,
        normal_reversed=False,
        is_removed=False,
    )

"""孔洞->加工路径"""
def hole_to_machining_path(
    hole: HoleFeature,
    polylines: dict[str, list[list[float]]] | None = None,
    craft_params: CraftParameters | None = None,
    lead_in_length: float = _LEAD_IN_LENGTH,
) -> MachiningPath:
    """Convert a HoleFeature to a MachiningPath.

    Args:
        hole: The hole feature to convert
        polylines: Optional dict mapping polyline_id to point lists
        craft_params: Override craft parameters
        lead_in_length: Lead-in line length (mm)

    Returns:
        MachiningPath for the hole
    """
    params = craft_params or _DEFAULT_CRAFT_PARAMS.get(
        hole.contour_type, _DEFAULT_CRAFT_PARAMS["unknown"]
    )

    path_id = f"path_{uuid.uuid4().hex[:8]}"

    # Map hole contour type to InnerPathTypeLiteral
    inner_type_map: dict[str, InnerPathTypeLiteral] = {
        "circle": "circle",
        "slot": "slot",
        "rectangle": "rectangle",
        "hexagon": "hexagon",
    }
    inner_type: InnerPathTypeLiteral | None = inner_type_map.get(hole.contour_type, "irregular")

    cam_lines: list[CAMLine] = []
    lead_line: CAMLine | None = None

    if polylines:
        # Find the first available polyline
        for polyline_id, points in polylines.items():
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

                # Lead-in line enters from outside (opposite direction for holes)
                if hole.center:
                    lead_line = _generate_lead_line(
                        points=points,
                        path_id=path_id,
                        path_type="inner",
                        params=params,
                        lead_length=lead_in_length,
                        lead_radius=0,  # Holes typically use direct lead-in
                        center=hole.center,
                        normal=hole.axis,
                        reverse_direction=True,
                    )
                break

    return MachiningPath(
        id=path_id,
        name=f"hole_{hole.contour_type}",
        path_type="inner",
        contour_id=hole.id,
        contour_type=hole.contour_type,
        cam_lines=cam_lines,
        lead_line=lead_line,
        idle_lines=[],
        thickness=1.0,
        normal_reversed=True,  # Holes typically need reversed normal
        is_removed=False,
    )

"""点列->CAM线"""
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
    """Generate CAMLines from a list of 3D points.

    每个段 (p[i-1] → p[i]) 按以下优先级判定 outType(对齐 C# ``OutPathType`` 枚举
    的判定顺序,见 ``激光/MODELCORE/Products/Laser/CAMLine.cs`` 与
    ``CraftRecipe.cs``):

      1. ``point``           — 段长退化(近似重合点)
      2. ``three_d_corner``  — 段间夹角 > 60°(三维拐角)
      3. ``big_arc`` / ``small_arc`` — 段内 3 点垂直偏差换算成角度后 > 30°
      4. ``long_line`` / ``shorter_line`` / ``shortest_line`` — 段长占周长比例分档
    """
    if len(points) < 2:
        return []

    # Pre-compute segment lengths and total perimeter so the classifier
    # only walks the polyline once.
    seg_count = len(points) - 1
    seg_lengths: list[float] = [
        math.sqrt(
            (points[i][0] - points[i - 1][0]) ** 2
            + (points[i][1] - points[i - 1][1]) ** 2
            + (points[i][2] - points[i - 1][2]) ** 2
        )
        for i in range(1, len(points))
    ]
    total_length = sum(seg_lengths) or 1e-9

    lines: list[CAMLine] = []

    for i in range(1, len(points)):
        p1 = points[i - 1]
        p2 = points[i]
        seg_length = seg_lengths[i - 1]

        out_type, velocity = _classify_segment(
            i=i,
            seg_count=seg_count,
            p1=p1,
            p2=p2,
            seg_length=seg_length,
            total_length=total_length,
            points=points,
            params=params,
        )

        lines.append(
            CAMLine(
                id=f"{path_id}_line_{i}",
                line_type="machining",
                path_type=path_type,
                inner_type=inner_type,
                out_type=out_type,
                start_point=Point3D(root=p1),
                end_point=Point3D(root=p2),
                normal=normal,
                velocity=velocity,
                power=params.power,
                duty=params.duty,
                is_clockwise=True,  # Default, should be determined by contour direction
                order_index=i,
                robot_joints=[],
            )
        )

    return lines


def _classify_segment(
    *,
    i: int,
    seg_count: int,
    p1: list[float],
    p2: list[float],
    seg_length: float,
    total_length: float,
    points: list[list[float]],
    params: CraftParameters,
) -> tuple[OutPathTypeLiteral, float]:
    """Classify a single polyline segment into (out_type, velocity).

    Mirrors the C# ``OutPathType`` decision tree (see
    ``CAMLine.cs`` ``InitRobotPara`` dispatch and ``CraftRecipe.cs`` mapping).
    Velocity stays as a ratio over ``params.velocity`` so the upstream
    Pydantic schema is unchanged; the absolute mm/s values used by the
    C# CraftRecipe layer are noted inline for reference.
    """
    # 1) Degenerate point — a segment shorter than the tolerance collapses
    #    onto a single machining point (C# OutPathType.Point).
    if seg_length < _POINT_LENGTH_TOL:
        return "point", params.velocity * 0.5

    # 2) Three-dimensional corner — the segment-to-segment angle exceeds
    #    the corner threshold. This corresponds to OutPathType.ThreeDCorner
    #    in C#; ``SmallThreeDCorner`` / ``PDLine`` (small step / slope
    #    segments) require upstream face-edge topology which we don't
    #    have here, so they remain unmapped.
    if 0 < i < seg_count:
        prev_seg = _vec_sub(points[i - 1], points[i - 2]) if i >= 2 else None
        next_seg = _vec_sub(points[i + 1], points[i]) if i + 1 < len(points) else None
        if prev_seg is not None and next_seg is not None:
            angle = _angle_between(prev_seg, next_seg)
            if angle > _CORNER_ANGLE_THRESHOLD:
                # C# uses deg/s units here; we keep the velocity ratio
                # convention by halving the baseline.
                return "three_d_corner", params.velocity * 0.5

    # 3) Planar arc — convert the existing 3-point deviation into an
    #    equivalent angle (deviation ≈ seg_length * sin(theta)) and split
    #    into big / small using _ARC_ANGLE_THRESHOLD. This is what the
    #    C# WLineAnalyze uses to dispatch BigArc / SmallArc.
    if 0 < i < seg_count:
        prev_pt = points[i - 1]
        next_pt = points[i + 1]
        deviation = _point_line_deviation(p2, prev_pt, next_pt)
        arc_angle = _deviation_to_angle(deviation, seg_length)
        if arc_angle > _ARC_ANGLE_THRESHOLD:
            # C# BigArc = 180 mm/s, SmallArc = 100 mm/s; we keep the
            # ratio form (0.7 vs 0.5) so the upstream schema is stable.
            return ("big_arc" if arc_angle > 2 * _ARC_ANGLE_THRESHOLD else "small_arc"), params.velocity * 0.7

    # 4) Straight segment — bucket by length as a fraction of the
    #    perimeter, matching C# LongLine / ShorterLine / ShortestLine.
    if seg_length > total_length * _LONG_LINE_RATIO:
        return "long_line", params.velocity
    if seg_length > total_length * _SHORT_LINE_RATIO:
        return "shorter_line", params.velocity * 0.8
    return "shortest_line", params.velocity * 0.6


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _angle_between(u: list[float], v: list[float]) -> float:
    """Interior angle (degrees) between two vectors; 0..180."""
    dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    nu = math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    nv = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (nu * nv)))
    return math.degrees(math.acos(cos))


def _deviation_to_angle(deviation: float, seg_length: float) -> float:
    """Approximate the arc angle from a 3-point deviation.

    deviation ≈ |seg_length| · sin(theta)  →  theta = asin(dev / seg_length)
    Clamps to a sane upper bound so a degenerate (collinear) segment
    doesn't report a 90° angle due to numerical noise.
    """
    if seg_length < 1e-9:
        return 0.0
    ratio = max(-1.0, min(1.0, deviation / seg_length))
    return math.degrees(math.asin(ratio))

"""生成引线"""
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

    The lead-in line approaches the first contour point from outside,
    typically at a tangent to the contour.
    """
    if len(points) < 2:
        return None

    # Get first point and tangent direction
    if reverse_direction:
        # For holes: approach from outside toward center
        start_pt = points[-1]
        end_pt = points[0]
    else:
        start_pt = points[0]
        end_pt = points[1]

    # Calculate approach direction (tangent to contour at start)
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    dz = end_pt[2] - start_pt[2]
    seg_len = math.sqrt(dx*dx + dy*dy + dz*dz)

    if seg_len < 1e-6:
        return None

    # Lead-in starts from a point outside the contour
    lead_start = [
        start_pt[0] - dx / seg_len * lead_length,
        start_pt[1] - dy / seg_len * lead_length,
        start_pt[2] - dz / seg_len * lead_length,
    ]

    return CAMLine(
        id=f"{path_id}_lead",
        line_type=CAMLineType.LEAD,
        path_type=path_type,
        inner_type=None,
        out_type=None,
        start_point=Point3D(root=lead_start),
        end_point=Point3D(root=start_pt),
        normal=normal,
        velocity=params.velocity * 0.5,  # Slow down for lead-in
        power=params.power,
        duty=params.duty,
        is_clockwise=True,
        order_index=0,
        robot_joints=[],
    )


def _point_line_deviation(
    p: list[float],
    a: list[float],
    b: list[float],
) -> float:
    """Perpendicular distance from point ``p`` to the line through ``a`` and ``b``.

    Used by the arc branch of ``_classify_segment`` to detect curved
    segments; ``_deviation_to_angle`` converts the result into an angle
    which is then compared against ``_ARC_ANGLE_THRESHOLD``.
    """
    # Vector from a to b
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    abz = b[2] - a[2]

    # Vector from a to p
    apx = p[0] - a[0]
    apy = p[1] - a[1]
    apz = p[2] - a[2]

    # Length of ab
    len_ab = math.sqrt(abx*abx + aby*aby + abz*abz)
    if len_ab < 1e-9:
        return 0.0

    # Cross product magnitude |ab x ap| / |ab|
    cross_x = aby * apz - abz * apy
    cross_y = abz * apx - abx * apz
    cross_z = abx * apy - aby * apx
    cross_mag = math.sqrt(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z)

    return cross_mag / len_ab


# ── High-level orchestration ───────────────────────────────────────────────────
"""高层编排函数"""
def generate_machining_paths(
    feature_result: dict[str, Any],
    *,
    apply_craft_params: bool = True,
    generate_lead_lines: bool = True,
) -> MachiningResult:
    """Convert a feature extraction result to machining paths.

    This is the main entry point for CAM path generation.

    Args:
        feature_result: The result from feature_service.analyze_face_spread()
        apply_craft_params: Whether to apply default craft parameters
        generate_lead_lines: Whether to generate lead-in/lead-out lines

    Returns:
        MachiningResult containing machining groups with paths and lines
    """
    machining_groups: list[MachiningGroup] = []
    total_paths = 0
    total_lines = 0

    # Extract polylines for path generation
    polylines: dict[str, list[list[float]]] = {}
    for polyline in feature_result.get("polylines", []):
        polyline_id = polyline.get("id")
        if polyline_id and "points" in polyline:
            polylines[polyline_id] = [p.root if isinstance(p, dict) else p for p in polyline["points"]]

    # Create inner machining paths (holes)
    inner_paths: list[MachiningPath] = []
    for hole in feature_result.get("holes", []):
        hole_feature = HoleFeature(**hole) if isinstance(hole, dict) else hole
        path = hole_to_machining_path(
            hole=hole_feature,
            polylines=polylines,
            craft_params=_DEFAULT_CRAFT_PARAMS.get(hole_feature.contour_type) if apply_craft_params else None,
        )
        inner_paths.append(path)
        total_paths += 1
        total_lines += len(path.cam_lines)

    # Create outer machining paths (contours)
    outer_paths: list[MachiningPath] = []
    for contour in feature_result.get("contours", []):
        contour_feature = ContourFeature(**contour) if isinstance(contour, dict) else contour
        if contour_feature.is_outer:
            path = contour_to_machining_path(
                contour=contour_feature,
                polylines=polylines,
                craft_params=_DEFAULT_CRAFT_PARAMS.get(contour_feature.contour_type) if apply_craft_params else None,
            )
            outer_paths.append(path)
            total_paths += 1
            total_lines += len(path.cam_lines)

    # Group paths by process face
    group = MachiningGroup(
        id=f"group_{uuid.uuid4().hex[:8]}",
        name="Default Machining Group",
        inner_paths=inner_paths,
        outer_paths=outer_paths,
        process_face_ids=[feature_result.get("target_face_id", "unknown")],
        is_merged=False,
    )
    machining_groups.append(group)

    return MachiningResult(
        schema_version="2.0",
        unit="mm",
        model_id=feature_result.get("model_id", "unknown"),
        feature_result=feature_result,
        machining_groups=machining_groups,
        total_path_count=total_paths,
        total_line_count=total_lines,
    )

"""获取工艺参数"""
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
    params = _DEFAULT_CRAFT_PARAMS.get(contour_type, _DEFAULT_CRAFT_PARAMS["unknown"]).model_copy()

    # Scale parameters based on thickness if provided
    if thickness and thickness > 0:
        # Thicker materials may need lower speed, higher power
        thickness_factor = min(2.0, thickness / 1.0)  # Cap at 2x for 2mm
        params.velocity = params.velocity / math.sqrt(thickness_factor)
        params.power = min(100, int(params.power * math.sqrt(thickness_factor)))

    return params
