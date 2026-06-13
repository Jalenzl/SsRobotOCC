"""3D 特征识别：通孔 / 盲孔 / 型腔(pocket) / 凸台(boss) + 深度。

现有 ``extractor`` 只做 2.5D 边界识别（口部 wire → 2D 轮廓），孔的 ``depth``
恒为 None。本模块在此基础上补齐 **深度方向** 的语义：

输入一个「特征口部」（host 面上的一条内环 wire + 其 2D 轮廓中心/法向），输出：

- ``direction``：``recess``（凹陷：孔/型腔，材料向内）或 ``protrusion``（凸出：凸台）；
- ``kind``：``through`` / ``blind`` / ``pocket`` / ``boss``；
- ``depth``：凹陷深度或凸台高度（mm，正值）；
- ``through``：凹陷是否贯通。

核心方法（不依赖曲面是平面还是自由曲面，故自由曲面天然兼容）：

1. **方向判定（射线两侧探针）**：在口部中心沿外法向 ±ε 各取一点，用
   ``BRepClass3d_SolidClassifier`` 判定其在所属实体内/外：
   - 外侧点在实体内（IN） → 凸台（protrusion）；
   - 内侧点在实体外（OUT，即空腔）→ 凹陷（recess）。
2. **通孔/盲孔判定（轴向步进）**：沿 -N 向材料内步进，若在行程内重新进入材料
   （IN）则为盲孔，命中处即孔底；若一直在空腔（OUT）直到穿出包围盒则为通孔。
3. **深度量化（壁面轴向跨度）**：取与口部 wire 共享边的侧壁面顶点，投影到轴向
   N，相对口部的下方跨度即凹陷深度、上方跨度即凸台高度。壁面几何比射线步进
   更精确，作为深度首选；射线步进只负责类型判定与兜底深度。
"""

from __future__ import annotations

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_IN, TopAbs_OUT, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import TopTools_MapOfShape

from app.occ.topology import (
    bbox_diagonal,
    build_edge_face_map,
    build_face_solid_map,
    face_point_and_outward_normal,
    owning_solid,
)

Vec3 = tuple[float, float, float]

_CLASSIFY_TOL = 1e-6


def _normalize(v: Vec3) -> Vec3:
    L = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    if L < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / L, v[1] / L, v[2] / L)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _shift(c: Vec3, axis: Vec3, dist: float) -> Vec3:
    return (c[0] + axis[0] * dist, c[1] + axis[1] * dist, c[2] + axis[2] * dist)


class Feature3DContext:
    """承载一次 3D 识别所需的拓扑映射与实体分类器缓存。

    在一次 analyze 调用内复用：邻接映射只建一次，每个实体的
    ``BRepClass3d_SolidClassifier`` 按需懒构建并按 ``IsSame`` 缓存。
    """

    def __init__(self, shape, bbox: tuple[float, float, float, float, float, float]) -> None:
        self._shape = shape
        self.edge_face_map = build_edge_face_map(shape)
        self.face_solid_map = build_face_solid_map(shape)
        self.bbox_diag = bbox_diagonal(bbox)
        # 探针偏移与步距：相对包围盒尺度自适应，并做合理上下限钳制。
        self.eps = min(0.5, max(0.05, self.bbox_diag * 0.002))
        self.step = min(2.0, max(0.3, self.bbox_diag / 150.0))
        self._classifiers: list[tuple[object, BRepClass3d_SolidClassifier]] = []

    def _classifier_for(self, solid):
        if solid is None:
            return None
        for s, clf in self._classifiers:
            if s.IsSame(solid):
                return clf
        clf = BRepClass3d_SolidClassifier(solid)
        self._classifiers.append((solid, clf))
        return clf

    def _state(self, clf, p: Vec3):
        clf.Perform(gp_Pnt(p[0], p[1], p[2]), _CLASSIFY_TOL)
        return clf.State()

    # ----------------------------------------------------------------- #
    # 对外主入口
    # ----------------------------------------------------------------- #

    def analyze_loop(
        self,
        *,
        host_face,
        mouth_wire,
        center: Vec3,
        normal: Vec3,
        contour_type: str,
    ) -> dict | None:
        """识别一个口部环对应的 3D 特征；无法判定时返回 None。

        返回 dict::

            {"direction": "recess"|"protrusion",
             "kind": "through"|"blind"|"pocket"|"boss",
             "depth": float,          # 深度/高度，正值
             "through": bool,
             "axis": (x,y,z),         # 指向材料外的单位外法向
             "wall_face_count": int}
        """
        solid = owning_solid(host_face, self.face_solid_map)
        clf = self._classifier_for(solid)
        if clf is None:
            return None  # 无实体（纯 Shell）→ 回退 2D，depth 留空

        axis = _normalize(normal)
        direction = self._probe_direction(clf, center, axis)
        if direction is None:
            return None

        wall_faces = self._wall_faces(mouth_wire, host_face)
        up_ext, down_ext = self._wall_axial_extent(wall_faces, center, axis)

        if direction == "protrusion":
            height = up_ext if up_ext > 1e-6 else self._march_outward(clf, center, axis)
            return {
                "direction": "protrusion",
                "kind": "boss",
                "depth": round(height, 4),
                "through": False,
                "axis": axis,
                "wall_face_count": len(wall_faces),
            }

        # recess：通孔/盲孔判定
        kind_through, march_depth = self._probe_inward(clf, center, axis, hint_depth=down_ext)
        depth = down_ext if down_ext > 1e-6 else (march_depth or 0.0)
        is_through = kind_through == "through"
        if contour_type == "circle":
            kind = "through" if is_through else "blind"
        else:
            kind = "pocket"
        return {
            "direction": "recess",
            "kind": kind,
            "depth": round(depth, 4),
            "through": is_through,
            "axis": axis,
            "wall_face_count": len(wall_faces),
        }

    # ----------------------------------------------------------------- #
    # 方向 / 通盲 / 高度 子过程
    # ----------------------------------------------------------------- #

    def _probe_direction(self, clf, center: Vec3, axis: Vec3) -> str | None:
        """口部两侧探针：外侧在材料内→凸台；内侧为空腔→凹陷。"""
        eps = self.eps
        s_out = self._state(clf, _shift(center, axis, eps))
        s_in = self._state(clf, _shift(center, axis, -eps))
        if s_out == TopAbs_IN:
            return "protrusion"
        if s_in == TopAbs_OUT:
            return "recess"
        return None

    def _probe_inward(self, clf, center: Vec3, axis: Vec3, *, hint_depth: float) -> tuple[str, float | None]:
        """沿 -N 步进：命中材料(IN)=盲孔(返回孔底距离)；穿出=通孔。

        若已有壁面深度 hint_depth，则优先用「孔底略深处单点探针」快速判定，
        省去逐步步进；hint 不可用时退化为粗步进。
        """
        if hint_depth > 1e-6:
            probe_d = hint_depth + max(self.eps, self.step * 0.5)
            state = self._state(clf, _shift(center, axis, -probe_d))
            if state == TopAbs_IN:
                return "blind", hint_depth
            return "through", None

        d = self.eps
        max_dist = self.bbox_diag + self.step
        while d <= max_dist:
            if self._state(clf, _shift(center, axis, -d)) == TopAbs_IN:
                return "blind", d
            d += self.step
        return "through", None

    def _march_outward(self, clf, center: Vec3, axis: Vec3) -> float:
        """沿 +N 步进到凸台顶部（首个 OUT），返回高度。"""
        d = self.eps
        max_dist = self.bbox_diag + self.step
        while d <= max_dist:
            if self._state(clf, _shift(center, axis, d)) == TopAbs_OUT:
                return d
            d += self.step
        return max_dist

    # ----------------------------------------------------------------- #
    # 壁面几何
    # ----------------------------------------------------------------- #

    def _wall_faces(self, mouth_wire, host_face) -> list:
        """与口部 wire 共享边的侧壁面（排除 host 面本身）。"""
        if mouth_wire is None:
            return []
        walls: list = []
        seen = TopTools_MapOfShape()
        seen.Add(host_face)
        exp = TopExp_Explorer(mouth_wire, TopAbs_EDGE)
        while exp.More():
            edge = exp.Current()
            if self.edge_face_map.Contains(edge):
                for other in self._list_iter(self.edge_face_map.FindFromKey(edge)):
                    if seen.Add(other):
                        walls.append(topods.Face(other))
            exp.Next()
        return walls

    def _wall_axial_extent(self, wall_faces: list, center: Vec3, axis: Vec3) -> tuple[float, float]:
        """侧壁顶点投影到轴向，返回 (相对口部上方跨度, 下方跨度)。"""
        ref = _dot(center, axis)
        up = 0.0
        down = 0.0
        for face in wall_faces:
            exp = TopExp_Explorer(face, TopAbs_VERTEX)
            while exp.More():
                v = topods.Vertex(exp.Current())
                p = BRep_Tool.Pnt(v)
                proj = p.X() * axis[0] + p.Y() * axis[1] + p.Z() * axis[2]
                up = max(up, proj - ref)
                down = max(down, ref - proj)
                exp.Next()
        return up, down

    @staticmethod
    def _list_iter(list_of_shape):
        from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape

        it = TopTools_ListIteratorOfListOfShape(list_of_shape)
        while it.More():
            yield it.Value()
            it.Next()
