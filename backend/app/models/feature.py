"""Feature-extraction data models — mirrors the frontend consumed schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Point3D(BaseModel):
    """3D point encoded as a flat ``[x, y, z]`` list to keep the
    response compact and to match the frontend's ``Vector3`` adapter
    (which accepts either ``{x,y,z}`` or a flat array)."""

    root: list[float] = Field(..., min_length=3, max_length=3)

    @property
    def x(self) -> float: return self.root[0]
    @property
    def y(self) -> float: return self.root[1]
    @property
    def z(self) -> float: return self.root[2]

    def __iter__(self):  # allows tuple / unpacking consumers
        return iter(self.root)

    def model_dump(self, **kw):  # type: ignore[override]
        return self.root


class Vector3D(BaseModel):
    root: list[float] = Field(..., min_length=3, max_length=3)

    @property
    def x(self) -> float: return self.root[0]
    @property
    def y(self) -> float: return self.root[1]
    @property
    def z(self) -> float: return self.root[2]

    def __iter__(self):
        return iter(self.root)

    def model_dump(self, **kw):  # type: ignore[override]
        return self.root


class FaceInfo(BaseModel):
    id: str
    surface_type: str = "other"
    area: float | None = None
    normal: Vector3D | None = None
    axis: Vector3D | None = None
    center: Point3D | None = None
    radius: float | None = None
    bbox: Any = None
    outer_wire_id: str | None = None
    inner_wire_ids: list[str] = Field(default_factory=list)


class ContourParameters(BaseModel):
    diameter: float | None = None
    length: float | None = None
    width: float | None = None
    across_flats: float | None = None


class ContourFeature(BaseModel):
    id: str
    contour_type: Literal["outer", "circle", "slot", "rectangle", "hexagon", "irregular", "unknown"] = "unknown"
    center: Point3D | None = None
    normal: Vector3D | None = None
    polyline_id: str | None = None
    wire_id: str | None = None
    face_id: str | None = None
    is_outer: bool = False
    parameters: ContourParameters = Field(default_factory=ContourParameters)
    area: float | None = None
    perimeter: float | None = None


class Polyline3D(BaseModel):
    id: str
    closed: bool = False
    points: list[Point3D] = Field(default_factory=list)


class WireInfo(BaseModel):
    id: str
    face_id: str | None = None
    is_outer: bool = False
    length: float | None = None
    area: float | None = None
    polyline_id: str | None = None
    contour_id: str | None = None
    contour_type: str | None = None


class HoleFeature(BaseModel):
    id: str
    kind: str = "unknown"
    contour_type: str = "unknown"
    center: Point3D | None = None
    axis: Vector3D | None = None
    diameter: float | None = None
    depth: float | None = None
    face_id: str | None = None
    wire_id: str | None = None
    cylindrical_face_ids: list[str] = Field(default_factory=list)
    parameters: ContourParameters = Field(default_factory=ContourParameters)


class ReferencePoint(BaseModel):
    id: str
    kind: str
    position: Point3D
    meta: dict = Field(default_factory=dict)


class ModelBBox(BaseModel):
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float


class CadFaceAnalyzeResult(BaseModel):
    schema_version: str = "1.0"
    unit: str = "mm"
    target_face_id: str
    model_bbox: ModelBBox | None = None
    face: FaceInfo | None = None
    reference_points: list[ReferencePoint] = Field(default_factory=list)
    polylines: list[Polyline3D] = Field(default_factory=list)
    wires: list[WireInfo] = Field(default_factory=list)
    contours: list[ContourFeature] = Field(default_factory=list)
    outer_contours: list[str] = Field(default_factory=list)
    holes: list[HoleFeature] = Field(default_factory=list)
    pockets: list[dict] = Field(default_factory=list)
    feature_groups: dict = Field(default_factory=dict)
    work_plane: str = "auto"
    work_plane_normal: Vector3D | None = None


# ── CAM Path Models ────────────────────────────

# Path type literals for Pydantic field types
PathTypeLiteral = Literal["outer", "inner"]
InnerPathTypeLiteral = Literal["circle", "slot", "rectangle", "hexagon", "irregular"]
OutPathTypeLiteral = Literal[
    "long_line", "shorter_line", "shortest_line",
    "big_arc", "small_arc", "three_d_corner", "point"
]
CAMLineTypeLiteral = Literal[
    "machining", "cut_in", "cut_out", "fast", "lead", "idle", "location"
]


class CraftParameters(BaseModel):
    """工艺参数"""
    velocity: float = 100.0
    power: int = 0
    duty: int = 50
    frequency: int = 0
    acc: int = 100
    cnt: int = 100
    lead_in: float = 5.0
    lead_out: float = 5.0
    direction: Literal["CW", "CCW"] = "CW"


class CAMLine(BaseModel):
    """CAM加工线"""
    id: str
    line_type: CAMLineTypeLiteral = "machining"
    path_type: PathTypeLiteral = "outer"
    inner_type: InnerPathTypeLiteral | None = None
    out_type: OutPathTypeLiteral | None = None
    start_point: Point3D
    end_point: Point3D
    normal: Vector3D | None = None
    velocity: float = 100.0
    power: int = 0
    duty: int = 50
    is_clockwise: bool = True
    order_index: int = 0
    robot_joints: list[float] = Field(default_factory=list)


class MachiningPath(BaseModel):
    """加工路径

    语义：
    - 单一 MachiningPath 对应一个**封闭加工单元**（一个外轮廓 / 一个孔 / 一段异形孔）
    - `cam_lines` 列表顺序 = 加工顺序（每条 CAMLine = 起点→终点的一段加工动作）
    - `lead_line` / `lead_out_line` = 引线 / 退刀线（单条 CAMLine，可选）
    - `idle_lines` = 路径内不同 CAMLine 之间的过渡线（与跨 path 的空驶区别开）
    - `source_hole_id` / `source_contour_id` = 该 path 关联的特征 ID（前端高亮用）
    - `order_index` = 在所属 inner_paths / outer_paths 列表中的位置（=前端点击顺序）
    """
    id: str
    name: str
    path_type: PathTypeLiteral = "outer"
    contour_id: str | None = None
    contour_type: str = "unknown"
    source_hole_id: str | None = None
    source_contour_id: str | None = None
    cam_lines: list[CAMLine] = Field(default_factory=list)
    lead_line: CAMLine | None = None
    lead_out_line: CAMLine | None = None
    idle_lines: list[CAMLine] = Field(default_factory=list)
    thickness: float = 1.0
    normal_reversed: bool = False
    is_removed: bool = False
    # 在所属 inner_paths / outer_paths 列表中的点击顺序
    # 0-based，由后端在生成时按前端传入 hole_ids 顺序填入
    order_index: int = 0


class MachiningGroup(BaseModel):
    """加工组

    `inner_paths` / `outer_paths` 列表顺序即为加工顺序；
    `path_order` 是所有 path_id 按加工顺序的扁平序列，便于前端一次性还原。
    """
    id: str
    name: str = "Default Group"
    inner_paths: list[MachiningPath] = Field(default_factory=list)
    outer_paths: list[MachiningPath] = Field(default_factory=list)
    process_face_ids: list[str] = Field(default_factory=list)
    is_merged: bool = False
    # 按加工顺序排列的所有 path_id；前端用此序列驱动 3D 仿真播放
    path_order: list[str] = Field(default_factory=list)
    # 跨 path 之间的空驶线（idle/transition）；第一段的起点处无前置空驶
    transition_lines: list[CAMLine] = Field(default_factory=list)


class MachiningResult(BaseModel):
    """加工分析结果 (包含特征识别 + CAM路径规划)

    ``feature_result`` 允许是 ``dict``，以接受来自特征识别模块的原始
    序列化结果（其内嵌的 ``points: list[Point3D]`` 可能尚未实例化）。
    当前端直接消费 ``MachiningResult.model_dump()`` 输出时，这能避免
    嵌套模型的强校验带来的额外开销。
    """
    schema_version: str = "2.0"
    unit: str = "mm"
    model_id: str
    feature_result: Any | None = None
    machining_groups: list[MachiningGroup] = Field(default_factory=list)
    total_path_count: int = 0
    total_line_count: int = 0
