"""内表面 / 外表面分类（face side）。

需求背景：前端选中一个面后，希望「外表面 → 扩散到整个装配体的全部外表面；
内表面 → 扩散到整个装配体的全部内表面」。本模块负责把每个面判定为：

- ``outer``：外表面（零件外轮廓 / 可从外部接触的蒙皮，如顶面、侧面、凸台外壁）；
- ``inner``：内表面（孔壁、镗孔、内腔、装配贴合面等朝向材料内部的面）。

判定方法（按用户建议「通过法向的正负」）：

    score = n_hat · (P - C)

其中 ``P`` 为面上一点、``n_hat`` 为该点处指向实体外部的单位外法向、``C`` 为该面
**所属实体的体积质心**（装配体逐实体计算，避免被其它零件干扰）。

- ``score >= 0``：外法向背离质心 → 外表面；
- ``score <  0``：外法向指向质心 → 内表面（典型：孔壁外法向指向孔轴=朝内）。

该方法快速、稳定、对自由曲面同样适用；对极端凹形（如偏置很大的型腔底面）为
近似启发式，``score`` 越接近 0 越不确定，调用方可据 ``score`` 做阈值过滤。
"""

from __future__ import annotations

from app.occ.topology import (
    build_face_solid_map,
    face_list,
    face_point_and_outward_normal,
    iter_solids,
    owning_solid,
    solid_centroid,
)

Vec3 = tuple[float, float, float]

# score 绝对值低于该比例（相对包围盒尺度）时视为「弱判定」，仍给出 side 但置信度低。
_WEAK_SCORE_RATIO = 1e-3


def _normalize(v: Vec3) -> Vec3:
    L = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    if L < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / L, v[1] / L, v[2] / L)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class FaceSideClassifier:
    """把 shape 中每个面判定为内/外表面（装配体逐实体计算质心）。

    用法::

        clf = FaceSideClassifier(shape)
        info = clf.classify(face)          # {"side": "outer"|"inner"|"unknown", ...}
        same = clf.faces_on_side("outer")  # [(global_index, face), ...]
    """

    def __init__(self, shape) -> None:
        self._shape = shape
        self._faces = face_list(shape)
        self._face_solid_map = build_face_solid_map(shape)
        # 预计算每个实体的质心；按 IsSame 线性匹配 owning solid（实体数通常很少）。
        self._solids = iter_solids(shape)
        self._centroids: list[Vec3] = [solid_centroid(s) for s in self._solids]
        self._fallback_centroid = self._global_centroid()

    def _global_centroid(self) -> Vec3:
        """无 owning solid（如纯 Shell）时的兜底基准：所有实体质心均值或原点。"""
        if not self._centroids:
            return (0.0, 0.0, 0.0)
        n = len(self._centroids)
        return (
            sum(c[0] for c in self._centroids) / n,
            sum(c[1] for c in self._centroids) / n,
            sum(c[2] for c in self._centroids) / n,
        )

    def _centroid_for(self, solid) -> Vec3:
        if solid is None:
            return self._fallback_centroid
        for s, c in zip(self._solids, self._centroids):
            if s.IsSame(solid):
                return c
        return self._fallback_centroid

    def classify(self, face) -> dict:
        """返回 {"side", "score", "point", "normal", "solid"}。

        side ∈ {"outer", "inner", "unknown"}；unknown 表示法向无法定义。
        """
        pn = face_point_and_outward_normal(face)
        if pn is None:
            return {"side": "unknown", "score": 0.0, "point": None, "normal": None, "solid": None}
        point, normal = pn
        solid = owning_solid(face, self._face_solid_map)
        centroid = self._centroid_for(solid)
        n_hat = _normalize(normal)
        score = _dot(n_hat, _sub(point, centroid))
        side = "outer" if score >= 0.0 else "inner"
        return {
            "side": side,
            "score": score,
            "point": point,
            "normal": n_hat,
            "solid": solid,
        }

    def classify_index(self, face_index: int) -> dict:
        if not (0 <= face_index < len(self._faces)):
            raise ValueError(f"face 索引越界: {face_index} / {len(self._faces)}")
        return self.classify(self._faces[face_index])

    def faces_on_side(self, side: str) -> list[tuple[int, object]]:
        """返回与 ``side`` 同侧（outer / inner）的所有 (全局面索引, face)。"""
        out: list[tuple[int, object]] = []
        for idx, face in enumerate(self._faces):
            info = self.classify(face)
            if info["side"] == side:
                out.append((idx, face))
        return out

    @property
    def faces(self) -> list:
        return self._faces

    @property
    def solid_count(self) -> int:
        return len(self._solids)
