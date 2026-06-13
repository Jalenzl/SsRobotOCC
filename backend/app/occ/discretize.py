"""Discretize TopoDS_Edge / Wire to 3D polylines."""

from __future__ import annotations

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Pnt


def apply_location(
    pts: list[tuple[float, float, float]],
    loc: TopLoc_Location,
) -> list[tuple[float, float, float]]:
    """Map wire/edge points into world coords (match BRep mesh / GLB)."""
    if loc.IsIdentity() or not pts:
        return pts
    trsf = loc.Transformation()
    out: list[tuple[float, float, float]] = []
    for x, y, z in pts:
        p = gp_Pnt(x, y, z)
        p.Transform(trsf)
        out.append((p.X(), p.Y(), p.Z()))
    return out


def wire_location_on_face(face, wire) -> TopLoc_Location:
    """Cumulative placement of a wire on its face (STEP 装配/实例化常见)."""
    wl = wire.Location()
    fl = face.Location()
    if wl.IsIdentity():
        return fl
    if fl.IsIdentity():
        return wl
    return fl.Multiplied(wl)


def discretize_edge(edge, linear_deflection: float, angular_deflection: float = 0.5) -> list[tuple[float, float, float]]:
    """Sample edge polyline. `angular_deflection` reserved; OCC GCPnts uses linear deflection only."""
    del angular_deflection  # not used by GCPnts_QuasiUniformDeflection
    curve = BRepAdaptor_Curve(edge)
    deflection = GCPnts_QuasiUniformDeflection(curve, linear_deflection)
    if not deflection.IsDone():
        u0, u1 = curve.FirstParameter(), curve.LastParameter()
        p0 = curve.Value(u0)
        p1 = curve.Value(u1)
        return [(p0.X(), p0.Y(), p0.Z()), (p1.X(), p1.Y(), p1.Z())]
    pts: list[tuple[float, float, float]] = []
    for i in range(1, deflection.NbPoints() + 1):
        p = deflection.Value(i)
        pts.append((p.X(), p.Y(), p.Z()))
    return pts


def _wire_edges_ordered(wire) -> list:
    """沿 wire 拓扑顺序返回边（BRepTools_WireExplorer，比 TopExp 无序遍历可靠）。"""
    edges: list = []
    exp = BRepTools_WireExplorer(wire)
    while exp.More():
        edges.append(topods.Edge(exp.Current()))
        exp.Next()
    return edges


def _append_segment(
    chain: list[tuple[float, float, float]],
    seg: list[tuple[float, float, float]],
    join_tol: float,
) -> bool:
    """把一条边折线接到 chain 末尾；成功返回 True。"""
    if not seg:
        return True
    if not chain:
        chain.extend(seg)
        return True
    if _dist(chain[-1], seg[0]) <= join_tol:
        chain.extend(seg[1:])
        return True
    if _dist(chain[-1], seg[-1]) <= join_tol:
        chain.extend(list(reversed(seg[:-1])))
        return True
    return False


def _discretize_edge_along_wire(
    edge,
    linear_deflection: float,
    angular_deflection: float,
) -> list[tuple[float, float, float]]:
    """按 wire 中的边朝向离散（REVERSED 边需翻转点序）。"""
    pts = discretize_edge(edge, linear_deflection, angular_deflection)
    if edge.Orientation() == TopAbs_REVERSED:
        pts = list(reversed(pts))
    return pts


def wire_to_polyline(
    wire,
    linear_deflection: float,
    angular_deflection: float,
    *,
    location: TopLoc_Location | None = None,
) -> list[tuple[float, float, float]]:
    """把 TopoDS_Wire 离散为有序 3D 折线。

    主路径：``BRepTools_WireExplorer`` 按 wire 拓扑顺序遍历每条边，保留 **全部**
    边段（不再只取最长链），避免圆孔/圆弧被截成一段。

    若有序拼接因容差失败（少见），回退到旧的贪心拼链；仍失败则按顺序硬拼，
    宁可出现短间隙也不丢弃边。
    """
    ordered_edges = _wire_edges_ordered(wire)
    if not ordered_edges:
        return []

    join_tol = _join_tolerance(linear_deflection)
    chain: list[tuple[float, float, float]] = []

    for edge in ordered_edges:
        seg = _discretize_edge_along_wire(edge, linear_deflection, angular_deflection)
        if not _append_segment(chain, seg, join_tol):
            # 容差内接不上：仍保留几何，避免丢边导致「圆变弧」
            if chain and seg:
                chain.extend(seg)
            elif seg:
                chain.extend(seg)

    if not chain and ordered_edges:
        chain = _wire_to_polyline_greedy_fallback(ordered_edges, linear_deflection, angular_deflection)

    if chain and len(chain) >= 3:
        close_tol = max(join_tol * 2.0, linear_deflection * 2.0, 1e-3)
        if _dist(chain[0], chain[-1]) <= close_tol:
            chain.append(chain[0])

    if location is not None:
        return apply_location(chain, location)
    return chain


def _wire_to_polyline_greedy_fallback(
    edges: list,
    linear_deflection: float,
    angular_deflection: float,
) -> list[tuple[float, float, float]]:
    """有序拼接失败时的兜底：贪心拼链，但保留所有边（按链顺序串联，不截断）。"""
    if not edges:
        return []

    segs = [_discretize_edge_along_wire(e, linear_deflection, angular_deflection) for e in edges]
    join_tol = _join_tolerance(linear_deflection)
    used = [False] * len(segs)
    chains: list[list[tuple[float, float, float]]] = []
    cur: list[tuple[float, float, float]] = []

    start_idx = next((i for i, s in enumerate(segs) if s), None)
    if start_idx is None:
        return []
    cur.extend(segs[start_idx])
    used[start_idx] = True

    while True:
        found = False
        endp = cur[-1] if cur else None
        for i, seg in enumerate(segs):
            if used[i] or not seg or endp is None:
                continue
            if _dist(endp, seg[0]) <= join_tol or _dist(endp, seg[-1]) <= join_tol:
                _append_segment(cur, seg, join_tol)
                used[i] = True
                found = True
                break
        if found:
            continue
        next_idx = next((i for i, s in enumerate(segs) if (not used[i]) and s), None)
        if next_idx is None:
            break
        chains.append(cur.copy())
        cur = list(segs[next_idx])
        used[next_idx] = True

    if cur:
        chains.append(cur)

    if not chains:
        return []

    # 串联所有链，避免只保留最长一段
    merged: list[tuple[float, float, float]] = []
    for part in chains:
        if not part:
            continue
        if not merged:
            merged.extend(part)
            continue
        if _append_segment(merged, part, join_tol):
            continue
        merged.extend(part)
    return merged


def wire_area_if_planar(wire) -> float | None:
    """Signed area in wire plane via BRepTools; None if not planar."""
    try:
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        props = GProp_GProps()
        brepgprop.SurfaceProperties(wire, props)
        return abs(props.Mass())
    except Exception:
        return None


def wire_length(wire) -> float:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.LinearProperties(wire, props)
    return props.Mass()


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _join_tolerance(linear_deflection: float) -> float:
    """Adaptive wire endpoint tolerance in model units."""
    return max(1e-6, linear_deflection * 0.5)


def _polyline_length(pts: list[tuple[float, float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(pts)):
        total += _dist(pts[i - 1], pts[i])
    return total
