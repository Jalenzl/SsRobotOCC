"""CAD feature extraction & toolpath API schemas (frontend-friendly JSON)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WorkPlane(str, Enum):
    AUTO = "auto"
    XY = "xy"
    YZ = "yz"
    XZ = "xz"


class PathStrategy(str, Enum):
    OUTER_CONTOUR = "outer_contour"
    HOLE_CIRCLE = "hole_circle"
    ZIGZAG = "zigzag"
    COMBINED = "combined"


class ContourType(str, Enum):
    OUTER = "outer"
    CIRCLE = "circle"
    SLOT = "slot"
    RECTANGLE = "rectangle"
    HEXAGON = "hexagon"
    UNKNOWN = "unknown"


class Point3D(BaseModel):
    x: float
    y: float
    z: float


class Vector3D(BaseModel):
    x: float
    y: float
    z: float


class BoundingBox3D(BaseModel):
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    center: Point3D


class Polyline3D(BaseModel):
    """Discretized edge or contour for Three.js/Babylon Line geometry."""

    id: str
    closed: bool = False
    points: list[Point3D] = Field(default_factory=list)


class ReferencePoint(BaseModel):
    id: str
    kind: str = Field(
        description="datum | face_center | hole_center | bbox_corner | contour_vertex"
    )
    position: Point3D
    meta: dict[str, Any] = Field(default_factory=dict)


class FaceFeature(BaseModel):
    id: str
    surface_type: str = Field(description="plane | cylinder | cone | sphere | torus | other")
    area: float
    normal: Vector3D | None = None
    axis: Vector3D | None = None
    center: Point3D | None = None
    radius: float | None = None
    bbox: BoundingBox3D | None = None
    outer_wire_id: str | None = None
    inner_wire_ids: list[str] = Field(default_factory=list)
    side: str | None = Field(
        None,
        description="内/外表面判定：outer 外表面 | inner 内表面 | unknown（法向不可定义）",
    )
    side_score: float | None = Field(
        None,
        description="内外判定得分 = 外法向·(面上点-所属实体质心)，>0 外、<0 内",
    )


class ContourParameters(BaseModel):
    """特征参数（按 contour_type 填对应字段）。"""

    diameter: float | None = Field(None, description="圆形：Φ直径 (mm)")
    length: float | None = Field(None, description="槽/矩形：L 长 (mm)")
    width: float | None = Field(None, description="槽/矩形：W 宽 (mm)")
    across_flats: float | None = Field(None, description="六边形：对边长 L (mm)")


class ContourFeature(BaseModel):
    """轮廓：线 + 类型 + 中心 + 法向 + 特征参数。"""

    id: str
    contour_type: str = Field(
        description="outer | circle | slot | rectangle | hexagon | unknown"
    )
    center: Point3D
    normal: Vector3D
    polyline_id: str
    wire_id: str | None = None
    face_id: str | None = None
    is_outer: bool = False
    parameters: ContourParameters
    area: float | None = None
    perimeter: float | None = None


class WireFeature(BaseModel):
    id: str
    face_id: str | None = None
    is_outer: bool = True
    length: float
    area: float | None = None
    polyline_id: str | None = None
    contour_id: str | None = None
    contour_type: str | None = None


class HoleFeature(BaseModel):
    id: str
    kind: str = Field(
        description=(
            "through 通孔 | blind 盲孔 | pocket 型腔 | boss 凸台 | counterbore 沉头 | "
            "slot | rectangle | hexagon | circle | unknown"
        )
    )
    contour_type: str | None = Field(None, description="与 contours.contour_type 一致")
    direction: str | None = Field(
        None,
        description="recess 凹陷(孔/型腔，材料向内) | protrusion 凸出(凸台，材料向外)",
    )
    through: bool | None = Field(None, description="凹陷是否贯通（通孔 True，盲孔/型腔 False）")
    center: Point3D
    axis: Vector3D
    diameter: float | None = None
    depth: float | None = Field(None, description="凹陷深度或凸台高度 (mm)，3D 识别后填充")
    face_id: str | None = None
    wire_id: str | None = None
    cylindrical_face_ids: list[str] = Field(default_factory=list)
    parameters: ContourParameters | None = Field(
        None, description="与 contours.parameters 相同结构"
    )


class PocketFeature(BaseModel):
    id: str
    bottom_face_id: str | None = None
    depth: float
    through: bool | None = Field(None, description="型腔是否贯通")
    center: Point3D | None = None
    axis: Vector3D | None = None
    contour_type: str | None = Field(None, description="型腔口部轮廓：rectangle | slot | unknown")
    face_id: str | None = Field(None, description="型腔口部所在面 id")
    wire_ids: list[str] = Field(default_factory=list)
    parameters: ContourParameters | None = None


class ShapeSummary(BaseModel):
    volume: float | None = None
    surface_area: float | None = None
    bbox: BoundingBox3D
    face_count: int
    edge_count: int
    solid_count: int


class CadAnalyzeOptions(BaseModel):
    linear_deflection: float = Field(0.1, gt=0, description="edge discretization (mm)")
    angular_deflection: float = Field(0.5, gt=0, description="edge discretization (rad)")
    work_plane: WorkPlane = WorkPlane.AUTO
    hole_diameter_min: float = Field(0.5, gt=0, description="min hole diameter (mm)")
    hole_diameter_max: float = Field(500.0, gt=0, description="max hole diameter (mm)")
    include_cylinder_holes: bool = Field(
        False,
        description=(
            "Whether to synthesize hole contours from cylindrical faces. "
            "Disabled by default to avoid treating outer cylinders/fillets as holes."
        ),
    )
    enable_depth: bool = Field(
        True,
        description=(
            "是否启用 3D 深度识别（通孔/盲孔/型腔/凸台 + 深度）。"
            "需要模型含实体(Solid)；纯 Shell 模型自动回退到 2D。"
        ),
    )
    model_config = {"use_enum_values": True}


class CadAnalyzeResult(BaseModel):
    schema_version: str = "1.1"
    unit: str = "mm"
    summary: ShapeSummary
    reference_points: list[ReferencePoint]
    polylines: list[Polyline3D]
    faces: list[FaceFeature]
    wires: list[WireFeature]
    contours: list[ContourFeature] = Field(
        default_factory=list,
        description="轮廓列表：线、类型、中心、法向、特征参数",
    )
    outer_contours: list[str] = Field(
        default_factory=list, description="外轮廓 contour id 列表"
    )
    holes: list[HoleFeature]
    pockets: list[PocketFeature]
    work_plane: str
    work_plane_normal: Vector3D


class FaceFeatureGroups(BaseModel):
    """按类型索引的特征存储，便于路径规划快速过滤。"""

    contours_by_type: dict[str, list[ContourFeature]] = Field(default_factory=dict)
    holes_by_type: dict[str, list[HoleFeature]] = Field(default_factory=dict)
    wires_by_role: dict[str, list[WireFeature]] = Field(default_factory=dict)


class CadFaceAnalyzeResult(BaseModel):
    """单面提取结果：只包含一个面及其轮廓/孔特征。"""

    schema_version: str = "1.0"
    unit: str = "mm"
    target_face_id: str
    model_bbox: BoundingBox3D
    face: FaceFeature
    reference_points: list[ReferencePoint]
    polylines: list[Polyline3D]
    wires: list[WireFeature]
    contours: list[ContourFeature]
    outer_contours: list[str] = Field(default_factory=list)
    holes: list[HoleFeature]
    pockets: list[PocketFeature]
    feature_groups: FaceFeatureGroups
    work_plane: str
    work_plane_normal: Vector3D


class CadFaceSpreadResult(BaseModel):
    """选面 → 内/外表面扩散分析结果。

    流程：选中一个面 → 用外法向正负判定该面是外表面还是内表面 →
    扩散到整个装配体的全部同侧表面 → 复用单面算法逐面提取轮廓/孔，
    并叠加 3D 深度识别（通孔/盲孔/型腔/凸台 + 深度）。
    """

    schema_version: str = "1.0"
    unit: str = "mm"
    target_face_id: str = Field(description="前端选中的种子面 id")
    side: str = Field(description="种子面判定的侧别：outer 外表面 | inner 内表面")
    side_score: float = Field(description="种子面内外判定得分")
    model_bbox: BoundingBox3D
    solid_count: int = Field(1, description="装配体实体数")
    face_ids: list[str] = Field(default_factory=list, description="本次扩散覆盖的同侧面 id")
    faces: list[FaceFeature] = Field(default_factory=list)
    reference_points: list[ReferencePoint] = Field(default_factory=list)
    polylines: list[Polyline3D] = Field(default_factory=list)
    wires: list[WireFeature] = Field(default_factory=list)
    contours: list[ContourFeature] = Field(default_factory=list)
    outer_contours: list[str] = Field(default_factory=list)
    holes: list[HoleFeature] = Field(default_factory=list)
    pockets: list[PocketFeature] = Field(default_factory=list)
    feature_groups: FaceFeatureGroups
    work_plane: str
    work_plane_normal: Vector3D


class PathSegment(BaseModel):
    id: str
    strategy: str
    feed: float | None = None
    points: list[Point3D]


class PathPlanOptions(BaseModel):
    strategy: PathStrategy = PathStrategy.COMBINED
    tool_diameter: float = Field(6.0, gt=0)
    step_over: float = Field(3.0, gt=0, description="stepover for zigzag (mm)")
    safe_z: float | None = Field(None, description="rapid plane; default bbox.zmax + clearance")
    clearance_z: float = Field(5.0, ge=0)
    feed_rapid: float = 5000.0
    feed_cut: float = 800.0
    hole_lead_in: bool = True


class PathPlanResult(BaseModel):
    schema_version: str = "1.0"
    unit: str = "mm"
    strategy: str
    segments: list[PathSegment]
    total_length: float
    estimated_time_s: float | None = None


class CadAnalyzeAndPathResponse(BaseModel):
    analyze: CadAnalyzeResult
    path: PathPlanResult


class CadFaceAnalyzeAndPathResponse(BaseModel):
    analyze: CadFaceAnalyzeResult
    path: PathPlanResult
