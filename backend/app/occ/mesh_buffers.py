"""OCC shape → plain Python mesh buffers (no numpy / trimesh)."""

from __future__ import annotations

import math
from typing import Any

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLib import BRepLib_ToolTriangulatedShape
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Pnt, gp_Vec

from app.occ.discretize import wire_to_polyline
from app.occ.geometry_utils import face_surface_info, face_wires, iterate_faces
from app.occ.mesh_export import _mesh_deflection, _mesh_shape
from app.utils.raw_glb import CadGlbDrawable


def _winding_indices(
    face,
    verts: list[tuple[float, float, float]],
    i1: int,
    i2: int,
    i3: int,
) -> list[int]:
    face_n = _face_outward_normal_tuple(face)
    if face_n is None:
        return [i1, i3, i2] if face.Orientation() == TopAbs_REVERSED else [i1, i2, i3]
    v0, v1, v2 = verts[i1], verts[i2], verts[i3]
    e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    cx = e1[1] * e2[2] - e1[2] * e2[1]
    cy = e1[2] * e2[0] - e1[0] * e2[2]
    cz = e1[0] * e2[1] - e1[1] * e2[0]
    if cx * face_n[0] + cy * face_n[1] + cz * face_n[2] < 0.0:
        return [i1, i3, i2]
    return [i1, i2, i3]


def _face_outward_normal_tuple(face) -> tuple[float, float, float] | None:
    try:
        adaptor = BRepAdaptor_Surface(face)
        u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
        um, vm = (u0 + u1) * 0.5, (v0 + v1) * 0.5
        p = gp_Pnt()
        du = gp_Vec()
        dv = gp_Vec()
        adaptor.D1(um, vm, p, du, dv)
        n = du.Crossed(dv)
        if n.Magnitude() < 1e-12:
            return None
        if face.Orientation() == TopAbs_REVERSED:
            n.Reverse()
        return (n.X(), n.Y(), n.Z())
    except Exception:
        return None


def _tri_normal(
    v0: tuple[float, float, float],
    v1: tuple[float, float, float],
    v2: tuple[float, float, float],
) -> tuple[float, float, float]:
    e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    cx = e1[1] * e2[2] - e1[2] * e2[1]
    cy = e1[2] * e2[0] - e1[0] * e2[2]
    cz = e1[0] * e2[1] - e1[1] * e2[0]
    L = math.sqrt(cx * cx + cy * cy + cz * cz) or 1.0
    return (cx / L, cy / L, cz / L)


def _iter_solids(shape):
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        yield topods.Solid(exp.Current())
        exp.Next()


def _solid_index_for_face(shape, face) -> int | None:
    solids = list(_iter_solids(shape))
    if not solids:
        return None
    for si, solid in enumerate(solids):
        exp = TopExp_Explorer(solid, TopAbs_FACE)
        while exp.More():
            if topods.Face(exp.Current()).IsSame(face):
                return si
            exp.Next()
    return None


def _face_mesh_buffers(face) -> tuple[list[float], list[int], list[float]] | None:
    """Flat-shaded triangle soup for one B-Rep face (independent vertices per corner).

    Vertices are written in the SAME world-space frame that
    ``_wire_line_positions`` writes its wire vertices in. We transform the
    triangulation nodes by ``BRep_Tool.Triangulation(face, loc).Location()``,
    which OCCT sets to ``face.Location()`` — for STEP assemblies this
    inherits all parent placements (root → compound → solid → shell →
    face), so after Transformed the node is in the same world frame as
    the wire polyline. Without this Transformed, the triangulation nodes
    stay in the face-local frame and the wire frame drifts hundreds of
    millimetres away.
    """
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, loc)
    if triangulation is None:
        return None

    try:
        BRepLib_ToolTriangulatedShape.ComputeNormals(face, triangulation)
    except Exception:
        pass

    has_normals = triangulation.HasNormals()
    n_triangles = triangulation.NbTriangles()
    if n_triangles == 0:
        return None

    face_verts: list[tuple[float, float, float]] = []
    face_nrms: list[tuple[float, float, float]] = []

    # triangulation.Node(i) is always in face-local coords.
    # BRep_Tool.Triangulation sets |loc| = face.Location().
    # We must always Transformed so the mesh ends up in world coords
    # (same frame as wire vertices from BRepAdaptor_Curve.Value(u)).
    tri_trsf = loc.Transformation()

    for i in range(1, triangulation.NbNodes() + 1):
        p = triangulation.Node(i)
        p = p.Transformed(tri_trsf)
        face_verts.append((p.X(), p.Y(), p.Z()))
        if has_normals:
            nn = triangulation.Normal(i)
            if not loc.IsIdentity():
                nn = nn.Transformed(tri_trsf)
            face_nrms.append((nn.X(), nn.Y(), nn.Z()))
        else:
            face_nrms.append((0.0, 0.0, 0.0))

    face_n = _face_outward_normal_tuple(face)
    positions: list[float] = []
    normals: list[float] = []
    indices: list[int] = []
    vertex_offset = 0

    for i in range(1, n_triangles + 1):
        n1, n2, n3 = triangulation.Triangle(i).Get()
        i1, i2, i3 = n1 - 1, n2 - 1, n3 - 1
        tri = _winding_indices(face, face_verts, i1, i2, i3)
        v0, v1, v2 = face_verts[tri[0]], face_verts[tri[1]], face_verts[tri[2]]
        tn = _tri_normal(v0, v1, v2)
        if face_n:
            e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
            e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
            cx = e1[1] * e2[2] - e1[2] * e2[1]
            cy = e1[2] * e2[0] - e1[0] * e2[2]
            cz = e1[0] * e2[1] - e1[1] * e2[0]
            dot = cx * face_n[0] + cy * face_n[1] + cz * face_n[2]
            if dot < 0:
                tn = (-tn[0], -tn[1], -tn[2])

        for vi in tri:
            if has_normals and face_nrms[vi] != (0.0, 0.0, 0.0):
                nn = face_nrms[vi]
            else:
                nn = tn
            positions.extend(face_verts[vi])
            normals.extend(nn)
            indices.append(vertex_offset)
            vertex_offset += 1

    if not indices:
        return None
    return positions, indices, normals


def _wire_line_positions(
    face,
    wire,
    *,
    linear_deflection: float,
    angular_deflection: float,
) -> list[float]:
    # Wire vertices must be in the SAME coordinate frame as the face mesh
    # vertices. _face_mesh_buffers writes triangulation.Node(i).Transformed(face.Location())
    # so both face mesh and wire vertices end up in world space.
    # We must use the SAME location here: tri_loc (= face.Location() from
    # BRep_Tool.Triangulation). Do NOT use wire_location_on_face() which
    # additionally multiplies wire.Location() and causes divergence in
    # assembly STEPs.
    tri_loc = TopLoc_Location()
    BRep_Tool.Triangulation(face, tri_loc)
    pts = wire_to_polyline(
        wire,
        linear_deflection,
        angular_deflection,
        location=tri_loc,
    )
    if len(pts) < 2:
        return []
    flat: list[float] = []
    for x, y, z in pts:
        flat.extend((x, y, z))
    return flat


def shape_to_cad_drawables(
    shape,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    filename: str = "model",
    *,
    start_face_idx: int = 0,
) -> tuple[list[CadGlbDrawable], list[dict[str, Any]], dict[str, list[str]]]:
    """
    Build hierarchical CAD drawables: per-face meshes + wire outlines + part grouping.

    Face index order matches ``iterate_faces`` / ``/cad/analyze`` (``face_0`` …).
    ``filename`` is used as the single-part group name when there is only one Solid.

    坐标约定：face mesh 顶点 + wire 顶点都在「shape 所在的世界坐标系」下
    （_face_mesh_buffers 会用 ``face.Location()`` 把 face-local 三角形顶点
    转到世界；wire 顶点来自 ``BRepAdaptor_Curve.Value(u)``，调用方传
    入 ``tri_loc = face.Location()``）。因此每个 part group 的 matrix 必须
    **是 identity**——再乘一次 ``solid.Location()`` 会把面片整体二次平移。
    solid 的真实位置仍然通过 ``extras.cad.matrix`` 暴露给前端。
    """
    _mesh_shape(shape, linear_deflection, angular_deflection)

    solids = list(_iter_solids(shape))
    multi_solid = len(solids) > 1
    part_nodes: dict[int, str] = {}
    # Pre-build part_nodes BEFORE the face loop so mesh/wire drawables can
    # attach to the correct part group from the start (previously this map
    # was filled only after the loop, leaving every face parented to "model"
    # and producing an empty solid_X group in the GLB).
    if multi_solid:
        for si in range(len(solids)):
            part_nodes[si] = f"solid_{si}"

    drawables: list[CadGlbDrawable] = []
    face_manifest: list[dict[str, Any]] = []
    parts_manifest: dict[str, list[str]] = {}

    for face_idx, face in enumerate(iterate_faces(shape)):
        fid = f"face_{start_face_idx + face_idx}"
        buffers = _face_mesh_buffers(face)
        if buffers is None:
            continue
        positions, indices, normals = buffers
        surf = face_surface_info(face)

        solid_idx = _solid_index_for_face(shape, face) if multi_solid else None
        part_id = part_nodes.get(solid_idx) if solid_idx is not None else None
        parent = part_id if part_id else "model"

        if part_id:
            parts_manifest.setdefault(part_id, []).append(fid)

        face_extras = {
            "cad": {
                "role": "face",
                "face_id": fid,
                "face_index": face_idx,
                "surface_type": surf.get("surface_type", "other"),
                "part_id": part_id,
            }
        }
        drawables.append(
            {
                "kind": "mesh",
                "name": fid,
                "parent": parent,
                "positions": positions,
                "indices": indices,
                "normals": normals,
                "extras": face_extras,
            }
        )

        wires = face_wires(face)
        wire_ids: list[str] = []
        for wi, wire in enumerate(wires):
            wid = f"wire_{fid}_{wi}"
            wire_ids.append(wid)
            line_pos = _wire_line_positions(
                face,
                wire,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
            )
            if len(line_pos) < 6:
                continue
            drawables.append(
                {
                    "kind": "line_strip",
                    "name": wid,
                    "parent": fid,
                    "positions": line_pos,
                    "extras": {
                        "cad": {
                            "role": "wire",
                            "wire_id": wid,
                            "face_id": fid,
                            "face_index": face_idx,
                            "wire_index": wi,
                            "part_id": part_id,
                        }
                    },
                }
            )

        face_manifest.append(
            {
                "face_id": fid,
                "face_index": face_idx,
                "surface_type": surf.get("surface_type", "other"),
                "part_id": part_id,
                "wire_ids": wire_ids,
            }
        )

    if not drawables:
        raise ValueError("模型无三角面片，请减小 linear_deflection 或检查 STEP 文件")

    # Always create a part group node so the GLB always has a meaningful
    # part-level hierarchy: single-solid → one group named by filename;
    # multi-solid → one group per solid named by STEP product name.
    #
    # 关键：face mesh 顶点已经在世界坐标系下，part group matrix 必须
    # 是 identity（不能再乘 solid.Location()，否则装配体各零件会
    # 二次平移、看起来相互分离）。solid 的真实位置仍通过 extras 暴露
    # 给前端（matrix 字段）。
    if multi_solid:
        part_id_to_name: dict[str, str] = {}
        for si in range(len(solids)):
            part_id = f"solid_{si}"
            solid = solids[si]
            part_name = _solid_name(solid, part_id)
            part_id_to_name[part_id] = part_name
            drawables.insert(
                0,
                {
                    "kind": "group",
                    "name": part_name,
                    "parent": "model",
                    "matrix": _identity_matrix(),
                    "extras": {
                        "cad": {
                            "role": "part",
                            "part_id": part_id,
                            "part_name": part_name,
                            "solid_index": si,
                            "face_ids": parts_manifest.get(part_id, []),
                            "matrix": _location_to_matrix(solid.Location()),
                        }
                    },
                },
            )
        # Rewrite every face/wire drawable whose parent was a part_id
        # ("solid_X") to the resolved part_name used by the group node.
        # The previous implementation only checked drawables written with
        # parent="model" before the multi-solid block filled part_nodes,
        # so this branch never matched and meshes stayed under "model".
        for d in drawables:
            parent = d.get("parent")
            if parent in part_id_to_name:
                d["parent"] = part_id_to_name[parent]
    else:
        base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
        part_name = base_name or "Part_1"
        drawables.insert(
            0,
            {
                "kind": "group",
                "name": part_name,
                "parent": "model",
                "matrix": _identity_matrix(),
                "extras": {
                    "cad": {
                        "role": "part",
                        "part_id": "model",
                        "part_name": part_name,
                        "solid_index": 0,
                        "face_ids": list(m["face_id"] for m in face_manifest),
                    }
                },
            },
        )
        for d in drawables:
            if d.get("parent") == "model":
                d["parent"] = part_name
        parts_manifest["model"] = [m["face_id"] for m in face_manifest]

    return drawables, face_manifest, parts_manifest


def shape_to_mesh_buffers(
    shape,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> tuple[list[float], list[int], list[float]]:
    """Return merged (positions flat, indices, normals flat per-vertex)."""
    _mesh_shape(shape, linear_deflection, angular_deflection)
    positions: list[float] = []
    normals: list[float] = []
    indices: list[int] = []
    offset = 0

    for face in iterate_faces(shape):
        buffers = _face_mesh_buffers(face)
        if buffers is None:
            continue
        fpos, fidx, fnrm = buffers
        positions.extend(fpos)
        normals.extend(fnrm)
        indices.extend(i + offset for i in fidx)
        offset += len(fpos) // 3

    if not indices:
        raise ValueError("模型无三角面片，请减小 linear_deflection 或检查 STEP 文件")
    return positions, indices, normals


def shape_to_glb_bytes(
    shape,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    filename: str = "model",
    *,
    start_face_idx: int = 0,
) -> bytes:
    """Export hierarchical GLB: per-face meshes + wire outlines + STEP part grouping."""
    from app.utils.raw_glb import cad_scene_to_glb_bytes

    drawables, face_manifest, parts_manifest = shape_to_cad_drawables(
        shape,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        filename=filename,
        start_face_idx=start_face_idx,
    )

    scene_extras = {
        "cad": {
            "schema": "robotlaser.glb.cad/v1",
            "unit": "mm",
            "face_ids": [m["face_id"] for m in face_manifest],
            "faces": face_manifest,
            "parts": [
                {"part_id": pid, "face_ids": fids}
                for pid, fids in sorted(parts_manifest.items())
            ],
        }
    }

    return cad_scene_to_glb_bytes(drawables, scene_extras=scene_extras)


def _solid_name(shape: Any, fallback: str) -> str:
    name = getattr(shape, "Name", None)
    if callable(name):
        try:
            value = name()
            if value:
                return str(value)
        except Exception:
            pass
    return fallback


def _identity_matrix() -> list[float]:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _location_to_matrix(loc: TopLoc_Location) -> list[float]:
    try:
        trsf = loc.Transformation()
        values = [
            trsf.Value(1, 1), trsf.Value(1, 2), trsf.Value(1, 3), trsf.Value(1, 4),
            trsf.Value(2, 1), trsf.Value(2, 2), trsf.Value(2, 3), trsf.Value(2, 4),
            trsf.Value(3, 1), trsf.Value(3, 2), trsf.Value(3, 3), trsf.Value(3, 4),
        ]
        return [float(v) for v in values] + [0.0, 0.0, 0.0, 1.0]
    except Exception:
        return _identity_matrix()


def _merged_mesh_buffers_for_shape(
    shape,
    linear_deflection: float,
    angular_deflection: float,
    *,
    remesh: bool = True,
) -> tuple[list[float], list[int], list[float]]:
    """Merge all faces of a sub-shape into one mesh buffer."""
    if remesh:
        deflection = _mesh_deflection(shape, linear_deflection)
        mesher = BRepMesh_IncrementalMesh(shape, deflection, False, angular_deflection, True)
        if hasattr(mesher, "IsDone") and not mesher.IsDone():
            raise ValueError("BRepMesh 三角化未完成")

    positions: list[float] = []
    normals: list[float] = []
    indices: list[int] = []
    offset = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        exp.Next()
        buffers = _face_mesh_buffers(face)
        if buffers is None:
            continue
        fpos, fidx, fnrm = buffers
        positions.extend(fpos)
        normals.extend(fnrm)
        indices.extend(i + offset for i in fidx)
        offset += len(fpos) // 3

    if not indices:
        raise ValueError("模型无三角面片，请减小 linear_deflection 或检查 STEP 文件")
    return positions, indices, normals


def shape_to_component_meshes(
    shape,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> list[dict[str, Any]]:
    """Tessellate each solid separately so the GLB can keep per-part node hierarchy."""
    _mesh_shape(shape, linear_deflection, angular_deflection)

    components: list[dict[str, Any]] = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_index = 0
    while exp.More():
        # ``shape`` 在这个路径下可能来自 XDE ``GetShape(label)``（已经
        # 把 label 的累积 Location 应用到几何里了）。在装配体里，每个
        # solid 自带 ``Location()`` 描述它相对 compound 的位移；如果
        # 直接拿去 tessellate，得到的顶点已经处于 world 坐标系，再
        # 把 ``solid.Location()`` 写到节点 ``matrix`` 就会双倍叠加，
        # 装配体表现为「零件相互分离」。
        # 正确做法：先 ``Located(TopLoc_Location())`` 把 solid 自身的
        # Location 剥掉再 tessellate（顶点退到 solid-local 坐标系），
        # 节点 matrix 仍保留 ``solid.Location()``，渲染时
        # ``vertex × matrix`` = world。
        solid = topods.Solid(exp.Current())
        local_solid = solid.Located(TopLoc_Location())
        loc = solid.Location()
        pos, idx, nrm = _merged_mesh_buffers_for_shape(
            local_solid, linear_deflection, angular_deflection, remesh=False
        )
        components.append(
            {
                "name": _solid_name(solid, f"Part_{solid_index + 1}"),
                "positions": pos,
                "indices": idx,
                "normals": nrm,
                "matrix": _location_to_matrix(loc),
            }
        )
        solid_index += 1
        exp.Next()

    if components:
        return components

    shell_exp = TopExp_Explorer(shape, TopAbs_SHELL)
    shell_index = 0
    while shell_exp.More():
        shell = topods.Shell(shell_exp.Current())
        local_shell = shell.Located(TopLoc_Location())
        pos, idx, nrm = _merged_mesh_buffers_for_shape(
            local_shell, linear_deflection, angular_deflection, remesh=False
        )
        components.append(
            {
                "name": f"Shell_{shell_index + 1}",
                "positions": pos,
                "indices": idx,
                "normals": nrm,
                "matrix": _location_to_matrix(shell.Location()),
            }
        )
        shell_index += 1
        shell_exp.Next()

    if components:
        return components

    local_shape = shape.Located(TopLoc_Location()) if shape is not None else None
    pos, idx, nrm = _merged_mesh_buffers_for_shape(
        local_shape, linear_deflection, angular_deflection, remesh=False
    )
    loc = shape.Location() if shape is not None else None
    return [
        {
            "name": _solid_name(shape, "Part_1"),
            "positions": pos,
            "indices": idx,
            "normals": nrm,
            "matrix": _location_to_matrix(loc) if loc is not None else _identity_matrix(),
        }
    ]


def shape_to_hierarchical_glb_bytes(
    shape,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> bytes:
    """Export per-part GLB (each Solid/Shell = selectable mesh node)."""
    from app.utils.raw_glb import meshes_to_glb_bytes

    meshes = shape_to_component_meshes(shape, linear_deflection, angular_deflection)
    return meshes_to_glb_bytes(meshes)


def shape_to_cad_hierarchical_glb_bytes(
    shape,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    filename: str = "model",
) -> bytes:
    """Export full CAD hierarchy: parts + per-face meshes + wire outlines."""
    return shape_to_glb_bytes(
        shape,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        filename=filename,
    )
