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
_SHORT_LINE_RATIO = 0.1      # 短直线: 长度 > 周长 * 0.1
_ARC_ANGLE_THRESHOLD = 30.0  # 圆弧角度阈值 (度)
_CORNER_ANGLE_THRESHOLD = 60.0  # 拐角阈值 (度)
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

    Classifies each segment as:
    - Long line
    - Short line
    - Arc (large or small)
    - Corner
    """
    if len(points) < 2:
        return []

    lines: list[CAMLine] = []

    # Calculate total perimeter for segment classification
    total_length = sum(
        math.sqrt(
            (points[i][0] - points[i-1][0]) ** 2 +
            (points[i][1] - points[i-1][1]) ** 2 +
            (points[i][2] - points[i-1][2]) ** 2
        )
        for i in range(1, len(points))
    )

    out_type_map: dict[str, OutPathTypeLiteral] = {
        "long_line": "long_line",
        "shorter_line": "shorter_line",
        "shortest_line": "shortest_line",
        "big_arc": "big_arc",
        "small_arc": "small_arc",
        "three_d_corner": "three_d_corner",
        "point": "point",
    }

    for i in range(1, len(points)):
        p1 = points[i - 1]
        p2 = points[i]
        seg_length = math.sqrt(
            (p2[0] - p1[0]) ** 2 +
            (p2[1] - p1[1]) ** 2 +
            (p2[2] - p1[2]) ** 2
        )

        # Classify segment type
        if seg_length > total_length * _LONG_LINE_RATIO:
            out_type = OutPathType.LONG_LINE
            velocity = params.velocity
        elif seg_length > total_length * _SHORT_LINE_RATIO:
            out_type = OutPathType.SHORTER_LINE
            velocity = params.velocity * 0.8
        else:
            out_type = OutPathType.SHORTEST_LINE
            velocity = params.velocity * 0.6

        # Determine if segment is an arc (simplified: check if 3 consecutive points deviate from straight line)
        if i > 0 and i < len(points) - 1:
            p0 = points[i - 1]
            p3 = points[i + 1]
            deviation = _point_line_deviation(p2, p1, p3)
            if deviation > 0.5:  # Significant deviation suggests arc
                if deviation > 2.0:
                    out_type = "big_arc"
                else:
                    out_type = "small_arc"
                velocity = params.velocity * 0.7

        line = CAMLine(
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
        lines.append(line)

    return lines

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
    """Calculate perpendicular distance from point p to line segment ab.

    This is used to detect if a point deviates from a straight line,
    indicating a curved segment (arc).
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
