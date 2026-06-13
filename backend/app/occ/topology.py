"""拓扑邻接 / 实体归属 / 外法向工具（3D 特征识别基础设施）。

本模块为 3D 特征识别（深度孔、盲孔、型腔、凸台、内外表面判定）提供与
``geometry_utils`` 互补的拓扑能力：

- 遍历装配体内的每个 ``TopAbs_SOLID``（多零件 STEP 常见）；
- 构建「边 → 相邻面」「面 → 所属实体」邻接映射（特征沿壁面扩展所需）；
- 计算面在世界坐标系下的「面上一点 + 外法向」（外法向 = 几何法向 × 面朝向）。

设计要点：
- 全部以 ``shape``（loader 规范化后的 Solid / Shell / Compound）为遍历根，
  与 ``geometry_utils.iterate_faces`` 的顺序保持一致，从而 face 全局索引
  ``face_{idx}`` 与可视化 GLB / analyze API 完全对齐。
- 不在主进程执行；与现有 OCC 代码一样，仅在隔离的 occ 子进程内被导入。
"""

from __future__ import annotations

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepTools import breptools
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopAbs_SOLID,
)
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_ListIteratorOfListOfShape,
    TopTools_MapOfShape,
)


Vec3 = tuple[float, float, float]


# --------------------------------------------------------------------------- #
# 实体 / 面遍历
# --------------------------------------------------------------------------- #


def iter_solids(shape) -> list:
    """返回 shape 内的所有 ``TopAbs_SOLID``（装配体多零件场景）。

    若 shape 本身不含 Solid（例如仅 Shell），返回空列表，调用方应回退到
    无实体分类器的 2D 路径。
    """
    out = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        out.append(topods.Solid(exp.Current()))
        exp.Next()
    return out


def face_list(shape) -> list:
    """与 ``geometry_utils.iterate_faces`` 同序的面列表（含全局索引语义）。"""
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(topods.Face(exp.Current()))
        exp.Next()
    return out


# --------------------------------------------------------------------------- #
# 邻接映射
# --------------------------------------------------------------------------- #


def build_edge_face_map(shape) -> TopTools_IndexedDataMapOfShapeListOfShape:
    """边 → 相邻面 映射；用于从特征口部 wire 的边扩展到侧壁面。"""
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    return edge_face_map


def build_face_solid_map(shape) -> TopTools_IndexedDataMapOfShapeListOfShape:
    """面 → 所属实体 映射；装配体中每个面定位其 owning solid。"""
    face_solid_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_FACE, TopAbs_SOLID, face_solid_map)
    return face_solid_map


def _iter_list_of_shape(list_of_shape):
    it = TopTools_ListIteratorOfListOfShape(list_of_shape)
    while it.More():
        yield it.Value()
        it.Next()


def faces_adjacent_to_face(
    face,
    edge_face_map: TopTools_IndexedDataMapOfShapeListOfShape,
) -> list:
    """返回与 ``face`` 共享至少一条边的所有相邻面（不含自身）。

    用于特征沿壁面区域生长：孔口面 → 圆柱/侧壁面 → 底面。
    """
    # OCCT 7.8+ 移除了 TopoDS_Shape.HashCode，改用基于 IsSame 的 MapOfShape 去重。
    neighbors: list = []
    seen = TopTools_MapOfShape()
    seen.Add(face)
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        if edge_face_map.Contains(edge):
            for other in _iter_list_of_shape(edge_face_map.FindFromKey(edge)):
                # Add 返回 True 表示首次加入（即此前未见过的面）。
                if seen.Add(other):
                    neighbors.append(topods.Face(other))
        exp.Next()
    return neighbors


def owning_solid(
    face,
    face_solid_map: TopTools_IndexedDataMapOfShapeListOfShape,
):
    """返回包含 ``face`` 的实体（找不到时返回 None，例如纯 Shell 模型）。"""
    if not face_solid_map.Contains(face):
        return None
    lst = face_solid_map.FindFromKey(face)
    for solid in _iter_list_of_shape(lst):
        return topods.Solid(solid)
    return None


# --------------------------------------------------------------------------- #
# 几何量：质心 / 外法向 / 面上点
# --------------------------------------------------------------------------- #


def solid_centroid(solid) -> Vec3:
    """实体体积质心（世界坐标），用于内/外表面的法向正负判定基准。"""
    props = GProp_GProps()
    brepgprop.VolumeProperties(solid, props)
    c = props.CentreOfMass()
    return (c.X(), c.Y(), c.Z())


def face_point_and_outward_normal(face) -> tuple[Vec3, Vec3] | None:
    """返回 (面上一点 P, 外法向 N)，均为世界坐标；失败返回 None。

    外法向 = 曲面在 (u,v) 处几何法向，按面朝向 ``TopAbs_REVERSED`` 取反，
    使其指向实体外部（远离材料）。取参数域中点作为采样点：
    - 平面：法向为常量，中点采样稳定；
    - 圆柱/自由曲面：中点位于壁面上，法向沿径向 / 局部外法向，足以支撑
      「法向 · (P − 质心)」的内外判定。
    """
    try:
        umin, umax, vmin, vmax = breptools.UVBounds(face)
    except Exception:
        return None
    u = 0.5 * (umin + umax)
    v = 0.5 * (vmin + vmax)
    try:
        surf = BRepAdaptor_Surface(face)
        props = BRepLProp_SLProps(surf, u, v, 1, 1e-6)
        if not props.IsNormalDefined():
            return None
        n = props.Normal()
        p = props.Value()
    except Exception:
        return None
    normal = (n.X(), n.Y(), n.Z())
    if face.Orientation() == TopAbs_REVERSED:
        normal = (-normal[0], -normal[1], -normal[2])
    return (p.X(), p.Y(), p.Z()), normal


def bbox_diagonal(bbox: tuple[float, float, float, float, float, float]) -> float:
    """包围盒对角线长度，用作射线步进的最大行程上界。"""
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    return (dx * dx + dy * dy + dz * dz) ** 0.5
