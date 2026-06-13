"""Bounding box, surface classification, work-plane helpers."""

from __future__ import annotations

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Vec


SURFACE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
}


def shape_bbox(shape) -> tuple[float, float, float, float, float, float]:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    return box.Get()


def face_surface_info(face) -> dict:
    """Surface metadata in world coordinates (applies face TopLoc like mesh export).

    ``normal`` 字段统一为 **外法向**（指向实体外部），与 GLB / 内外表面判定一致。
    解析曲面（B-spline 等）无解析 normal 时，由 ``topology.face_point_and_outward_normal`` 采样。
    """
    adaptor = BRepAdaptor_Surface(face)
    stype = adaptor.GetType()
    name = SURFACE_NAMES.get(stype, "other")
    info: dict = {"surface_type": name}
    if stype == GeomAbs_Plane:
        pln = adaptor.Plane()
        ax = pln.Axis()
        info["center"] = _pnt(ax.Location())
        info["normal"] = _dir(ax.Direction())
        if face.Orientation() == TopAbs_REVERSED:
            n = info["normal"]
            info["normal"] = (-n[0], -n[1], -n[2])
    elif stype == GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        ax = cyl.Axis()
        info["center"] = _pnt(ax.Location())
        info["axis"] = _dir(ax.Direction())
        info["radius"] = cyl.Radius()
    _apply_outward_normal(face, info)
    return info


def face_outward_normal(face) -> tuple[float, float, float] | None:
    """宿主面外法向（世界坐标）；失败时返回 None。"""
    from app.occ.topology import face_point_and_outward_normal

    pn = face_point_and_outward_normal(face)
    return pn[1] if pn else None


def _apply_outward_normal(face, info: dict) -> None:
    """用 B-Rep 采样外法向覆盖/补全 ``info['normal']``。"""
    sampled = face_outward_normal(face)
    if sampled is not None:
        info["normal"] = sampled


def face_area(face) -> float:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()


def iterate_faces(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        yield topods.Face(exp.Current())
        exp.Next()


def face_wires(face) -> list:
    wires = []
    exp = TopExp_Explorer(face, TopAbs_WIRE)
    while exp.More():
        wires.append(topods.Wire(exp.Current()))
        exp.Next()
    return wires


def work_plane_normal(mode: str, bbox: tuple[float, float, float, float, float, float]) -> tuple[float, float, float]:
    if mode == "xy":
        return (0.0, 0.0, 1.0)
    if mode == "yz":
        return (1.0, 0.0, 0.0)
    if mode == "xz":
        return (0.0, 1.0, 0.0)
    # auto: smallest bbox extent => setup normal
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    if dz <= dx and dz <= dy:
        return (0.0, 0.0, 1.0)
    if dy <= dx:
        return (0.0, 1.0, 0.0)
    return (1.0, 0.0, 0.0)


def project_point(p: tuple[float, float, float], normal: tuple[float, float, float]) -> tuple[float, float]:
    nx, ny, nz = normal
    if abs(nz) >= max(abs(nx), abs(ny)):
        return (p[0], p[1])
    if abs(ny) >= abs(nx):
        return (p[0], p[2])
    return (p[1], p[2])


def _pnt(p: gp_Pnt) -> tuple[float, float, float]:
    return (p.X(), p.Y(), p.Z())


def _dir(d: gp_Dir) -> tuple[float, float, float]:
    return (d.X(), d.Y(), d.Z())
