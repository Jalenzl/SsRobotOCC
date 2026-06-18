"""Minimal GLB writer (no trimesh / numpy — safe after pythonOCC on Windows)."""

from __future__ import annotations

import json
import math
import struct
from typing import Any, Literal, TypedDict


class CadGlbDrawable(TypedDict, total=False):
    """One mesh or polyline drawable in a CAD hierarchy GLB."""

    kind: Literal["mesh", "line_strip", "group"]
    name: str
    parent: str | None
    positions: list[float]
    indices: list[int]
    normals: list[float]
    extras: dict[str, Any]
    matrix: list[float]


def _pad_json_chunk(data: bytes) -> bytes:
    """glTF 2.0: JSON chunk padding must be Space (0x20), not NUL."""
    pad = (4 - len(data) % 4) % 4
    return data + b" " * pad


def _pad_bin_chunk(data: bytes) -> bytes:
    pad = (4 - len(data) % 4) % 4
    return data + b"\x00" * pad


def _finite_float(x: float) -> float:
    return float(x) if math.isfinite(x) else 0.0


def _bounds(positions: list[float]) -> tuple[list[float], list[float]]:
    xs = [_finite_float(positions[i]) for i in range(0, len(positions), 3)]
    ys = [_finite_float(positions[i]) for i in range(1, len(positions), 3)]
    zs = [_finite_float(positions[i]) for i in range(2, len(positions), 3)]
    return (
        [min(xs), min(ys), min(zs)],
        [max(xs), max(ys), max(zs)],
    )


def validate_glb_bytes(glb: bytes) -> None:
    """Raise ValueError if GLB JSON chunk length does not match spec."""
    if len(glb) < 20 or glb[:4] != b"glTF":
        raise ValueError("not a GLB file")
    total = struct.unpack("<I", glb[8:12])[0]
    if total != len(glb):
        raise ValueError(f"GLB length mismatch: header={total} file={len(glb)}")
    json_len = struct.unpack("<I", glb[12:16])[0]
    if glb[16:20] != b"JSON":
        raise ValueError("missing JSON chunk")
    json_data = glb[20 : 20 + json_len]
    if len(json_data) != json_len:
        raise ValueError(f"JSON chunk truncated: declared={json_len} got={len(json_data)}")
    text = json_data.rstrip(b" ")
    json.loads(text.decode("utf-8"))
    bin_off = 20 + json_len
    if bin_off + 8 > len(glb):
        return
    if glb[bin_off + 4 : bin_off + 8] != b"BIN\x00":
        raise ValueError("missing BIN chunk")


def pack_glb(json_data: bytes, bin_data: bytes) -> bytes:
    """Assemble a valid GLB from JSON + BIN payloads."""
    json_padded = _pad_json_chunk(json_data)
    bin_padded = _pad_bin_chunk(bin_data)
    json_chunk = struct.pack("<I4s", len(json_padded), b"JSON") + json_padded
    bin_chunk = struct.pack("<I4s", len(bin_padded), b"BIN\x00") + bin_padded
    total = 12 + len(json_chunk) + len(bin_chunk)
    header = struct.pack("<4sII", b"glTF", 2, total)
    glb = header + json_chunk + bin_chunk
    validate_glb_bytes(glb)
    return glb


_CAD_FACE_MATERIAL = {
    "pbrMetallicRoughness": {
        "baseColorFactor": [0.75, 0.78, 0.82, 1.0],
        "metallicFactor": 0.12,
        "roughnessFactor": 0.45,
    },
    "doubleSided": True,
}

_CAD_WIRE_MATERIAL = {
    "pbrMetallicRoughness": {
        "baseColorFactor": [0.12, 0.14, 0.18, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0,
    },
}


def cad_scene_to_glb_bytes(
    drawables: list[CadGlbDrawable],
    *,
    scene_extras: dict[str, Any] | None = None,
    root_name: str = "model",
) -> bytes:
    """
    Export a CAD scene: grouping nodes + per-face triangle meshes + wire line strips.

    Each drawable must have a unique ``name``. ``parent`` defaults to ``root_name``.
    """
    if not drawables:
        raise ValueError("empty CAD scene")

    blob = bytearray()
    accessors: list[dict[str, Any]] = []
    buffer_views: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    materials = [_CAD_FACE_MATERIAL, _CAD_WIRE_MATERIAL]

    node_names: set[str] = {root_name}
    children_of: dict[str, list[str]] = {root_name: []}
    node_mesh: dict[str, int] = {}
    node_extras: dict[str, dict[str, Any]] = {}
    node_matrix: dict[str, list[float]] = {}

    def _append_f32(data: list[float]) -> tuple[int, int]:
        packed = struct.pack(f"<{len(data)}f", *[_finite_float(v) for v in data])
        offset = len(blob)
        blob.extend(packed)
        return offset, len(packed)

    def _append_u32(data: list[int]) -> tuple[int, int]:
        packed = struct.pack(f"<{len(data)}I", *data)
        offset = len(blob)
        blob.extend(packed)
        return offset, len(packed)

    def _add_accessor(
        offset: int,
        length: int,
        *,
        component_type: int,
        count: int,
        type_name: str,
        min_v: list[float] | None = None,
        max_v: list[float] | None = None,
    ) -> int:
        view_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        acc: dict[str, Any] = {
            "bufferView": view_idx,
            "componentType": component_type,
            "count": count,
            "type": type_name,
        }
        if min_v is not None:
            acc["min"] = min_v
        if max_v is not None:
            acc["max"] = max_v
        accessors.append(acc)
        return len(accessors) - 1

    for d in drawables:
        name = d["name"]
        if name in node_names:
            # Auto-rename: append _1, _2, ... until unique.
            base = name
            n = 1
            while name in node_names:
                name = f"{base}_{n}"
                n += 1
        parent = d.get("parent")
        node_names.add(name)
        if parent:
            node_names.add(parent)
            children_of.setdefault(parent, []).append(name)
        children_of.setdefault(name, [])
        extras = d.get("extras")
        if extras:
            node_extras[name] = extras
        matrix = d.get("matrix")
        if matrix is not None:
            node_matrix[name] = [float(v) for v in matrix]

        kind = d.get("kind", "mesh")
        # group nodes are recorded (name+parent+matrix+extras) so they appear
        # in the final glTF nodes list as child containers; only skip geometry.
        if kind == "group":
            continue

        positions = d.get("positions") or []
        if len(positions) < 6:
            continue

        if kind == "line_strip":
            pos_off, pos_len = _append_f32(positions)
            n_verts = len(positions) // 3
            vmin, vmax = _bounds(positions)
            pos_acc = _add_accessor(
                pos_off,
                pos_len,
                component_type=5126,
                count=n_verts,
                type_name="VEC3",
                min_v=vmin,
                max_v=vmax,
            )
            mesh_idx = len(meshes)
            meshes.append(
                {
                    "name": name,
                    "primitives": [
                        {
                            "attributes": {"POSITION": pos_acc},
                            "material": 1,
                            "mode": 3,
                        }
                    ],
                }
            )
            node_mesh[name] = mesh_idx
            continue

        indices = d.get("indices") or []
        if len(indices) < 3:
            continue
        normals = d.get("normals")
        n_verts = len(positions) // 3
        if normals is None or len(normals) != len(positions):
            normals = [0.0, 0.0, 1.0] * n_verts

        pos_off, pos_len = _append_f32(positions)
        nrm_off, nrm_len = _append_f32(normals)
        idx_off, idx_len = _append_u32(indices)
        vmin, vmax = _bounds(positions)

        pos_acc = _add_accessor(
            pos_off,
            pos_len,
            component_type=5126,
            count=n_verts,
            type_name="VEC3",
            min_v=vmin,
            max_v=vmax,
        )
        nrm_acc = _add_accessor(
            nrm_off,
            nrm_len,
            component_type=5126,
            count=n_verts,
            type_name="VEC3",
        )
        idx_acc = _add_accessor(
            idx_off,
            idx_len,
            component_type=5125,
            count=len(indices),
            type_name="SCALAR",
        )
        mesh_idx = len(meshes)
        meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc},
                        "indices": idx_acc,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        )
        node_mesh[name] = mesh_idx

    for parent in list(node_names):
        if parent not in children_of:
            children_of[parent] = []

    # ── cycle / duplicate detection ──────────────────────────────────────────
    # Any node encountered twice during BFS is a cycle (glTF node hierarchy = DAG)
    ordered: list[str] = [root_name]
    seen: set[str] = {root_name}
    queue: list[str] = list(children_of.get(root_name, []))
    while queue:
        current = queue.pop(0)
        if current in seen:
            raise ValueError(
                f"GLB node hierarchy contains a cycle at node '{current}'. "
                f"Check the 'parent' field of all drawables — a node cannot "
                f"appear as its own ancestor."
            )
        seen.add(current)
        ordered.append(current)
        for child in sorted(children_of.get(current, [])):
            if child not in seen:
                queue.append(child)

    # Orphan nodes that were never reachable from root
    for name in sorted(node_names - seen):
        ordered.append(name)

    name_to_idx = {name: idx for idx, name in enumerate(ordered)}
    nodes: list[dict[str, Any]] = []
    for name in ordered:
        nd: dict[str, Any] = {"name": name}
        if name in node_mesh:
            nd["mesh"] = node_mesh[name]
        kids = children_of.get(name) or []
        if kids:
            nd["children"] = [name_to_idx[c] for c in kids]
        if name in node_extras:
            nd["extras"] = node_extras[name]
        if name in node_matrix:
            nd["matrix"] = node_matrix[name]
        nodes.append(nd)

    if not meshes:
        raise ValueError("empty CAD scene (no mesh primitives)")

    scene: dict[str, Any] = {"nodes": [name_to_idx[root_name]]}
    if scene_extras:
        scene["extras"] = scene_extras

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "cad-backend-hierarchical-glb"},
        "scene": 0,
        "scenes": [scene],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
        "materials": materials,
    }

    json_data = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return pack_glb(json_data, bytes(blob))


def _identity_matrix() -> list[float]:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def meshes_to_glb_bytes(meshes: list[dict[str, Any]]) -> bytes:
    """Export one or more meshes as a glTF scene (per-part selectable nodes).

    Each item may contain:
    - positions, indices, normals
    - name: node name (e.g. Part_1)
    - matrix: optional 4x4 column-major transform
    """
    if not meshes:
        raise ValueError("empty mesh list")

    buffers: list[bytes] = []
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    gltf_meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    scenes_nodes: list[int] = []

    for mesh in meshes:
        positions = list(mesh.get("positions") or [])
        indices = list(mesh.get("indices") or [])
        normals = mesh.get("normals")
        if normals is None or len(normals) != len(positions):
            normals = [0.0, 0.0, 1.0] * (len(positions) // 3)

        n_verts = len(positions) // 3
        if n_verts == 0 or len(indices) < 3:
            continue

        pos_bin = struct.pack(f"<{len(positions)}f", *[_finite_float(v) for v in positions])
        nrm_bin = struct.pack(f"<{len(normals)}f", *[_finite_float(v) for v in normals])
        idx_bin = struct.pack(f"<{len(indices)}I", *indices)

        current_blob_len = len(b"".join(buffers))
        pos_off = current_blob_len
        nrm_off = current_blob_len + len(pos_bin)
        idx_off = nrm_off + len(nrm_bin)

        vmin, vmax = _bounds(positions)
        pos_bv = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_bin)})
        pos_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": pos_bv,
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC3",
                "min": vmin,
                "max": vmax,
            }
        )
        nrm_bv = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": nrm_off, "byteLength": len(nrm_bin)})
        nrm_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": nrm_bv,
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC3",
            }
        )
        idx_bv = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_bin)})
        idx_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": idx_bv,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
            }
        )

        buffers.append(pos_bin + nrm_bin + idx_bin)

        mesh_index = len(gltf_meshes)
        gltf_meshes.append(
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_accessor, "NORMAL": nrm_accessor},
                        "indices": idx_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        )

        node: dict[str, Any] = {"mesh": mesh_index}
        name = mesh.get("name")
        if name:
            node["name"] = str(name)
        matrix = mesh.get("matrix")
        if matrix is not None:
            node["matrix"] = [float(v) for v in matrix]
        children = mesh.get("children")
        if children:
            node["children"] = [int(idx) for idx in children]
        nodes.append(node)
        scenes_nodes.append(len(nodes) - 1)

    if not nodes:
        raise ValueError("empty mesh")

    blob = b"".join(buffers)
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "cad-backend-raw-glb"},
        "scene": 0,
        "scenes": [{"nodes": scenes_nodes}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.75, 0.78, 0.82, 1.0],
                    "metallicFactor": 0.12,
                    "roughnessFactor": 0.45,
                },
                "doubleSided": True,
            }
        ],
    }

    json_data = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return pack_glb(json_data, blob)


def mesh_to_glb_bytes(
    positions: list[float],
    indices: list[int],
    normals: list[float] | None = None,
) -> bytes:
    """
    positions: flat [x,y,z, ...]
    indices: triangle corner indices
    normals: optional flat [nx,ny,nz, ...] per vertex
    """
    return meshes_to_glb_bytes(
        [
            {
                "positions": positions,
                "indices": indices,
                "normals": normals,
            }
        ]
    )


# ----------------------------------------------------------------------
# 层级 GLB 导出：支持完整的 STEP 产品层级结构
# ----------------------------------------------------------------------

class HierarchicalMesh(TypedDict, total=False):
    """层级化网格数据."""
    name: str
    part_id: str
    positions: list[float]
    indices: list[int]
    normals: list[float]
    matrix: list[float]
    color: tuple[float, float, float] | None
    extras: dict[str, Any]


def hierarchical_scene_to_glb_bytes(
    nodes: list[dict[str, Any]],
    *,
    scene_extras: dict[str, Any] | None = None,
    root_name: str = "model",
) -> bytes:
    """将层级化场景导出为 GLB.

    Args:
        nodes: 节点列表，每个节点结构:
            {
                "name": str,              # 节点名称
                "part_id": str,           # 唯一标识 (用于前端 metadata)
                "parent": str | None,      # 父节点名称
                "is_assembly": bool,       # 是否为装配体
                "positions": list[float] | None,  # 顶点数据 (仅叶子节点)
                "indices": list[int] | None,      # 索引数据 (仅叶子节点)
                "normals": list[float] | None,    # 法线数据
                "matrix": list[float] | None,     # 4x4 变换矩阵
                "color": tuple | None,    # RGB 颜色
                "extras": dict | None,   # 额外数据
            }
        scene_extras: 场景级别的额外数据 (如层级树信息)
        root_name: 根节点名称

    Returns:
        GLB 字节数据
    """
    if not nodes:
        raise ValueError("empty node list")

    blob = bytearray()
    accessors: list[dict[str, Any]] = []
    buffer_views: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []

    materials = [_CAD_FACE_MATERIAL]

    def _append_f32(data: list[float]) -> tuple[int, int]:
        packed = struct.pack(f"<{len(data)}f", *[_finite_float(v) for v in data])
        offset = len(blob)
        blob.extend(packed)
        return offset, len(packed)

    def _append_u32(data: list[int]) -> tuple[int, int]:
        packed = struct.pack(f"<{len(data)}I", *data)
        offset = len(blob)
        blob.extend(packed)
        return offset, len(packed)

    def _add_accessor(
        offset: int,
        length: int,
        *,
        component_type: int,
        count: int,
        type_name: str,
        min_v: list[float] | None = None,
        max_v: list[float] | None = None,
    ) -> int:
        view_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        acc: dict[str, Any] = {
            "bufferView": view_idx,
            "componentType": component_type,
            "count": count,
            "type": type_name,
        }
        if min_v is not None:
            acc["min"] = min_v
        if max_v is not None:
            acc["max"] = max_v
        accessors.append(acc)
        return len(accessors) - 1

    # 收集所有节点名称并建立父子关系
    node_names: set[str] = {root_name}
    children_of: dict[str, list[str]] = {root_name: []}
    node_mesh: dict[str, int] = {}
    node_extras: dict[str, dict[str, Any]] = {}
    node_matrix: dict[str, list[float]] = {}

    for node in nodes:
        name = node.get("name", f"node_{len(node_names)}")
        parent = node.get("parent")

        if name not in node_names:
            node_names.add(name)
        if parent and parent not in node_names:
            node_names.add(parent)

        if parent and parent != name:
            children_of.setdefault(parent, []).append(name)
        children_of.setdefault(name, [])

        extras = node.get("extras")
        if extras:
            node_extras[name] = extras

        matrix = node.get("matrix")
        if matrix:
            node_matrix[name] = [float(v) for v in matrix]

        # 如果是叶子节点 (有几何数据)，创建 mesh
        positions = node.get("positions") or []
        indices = node.get("indices") or []
        if len(positions) >= 6 and len(indices) >= 3:
            normals = node.get("normals") or []
            n_verts = len(positions) // 3
            if normals is None or len(normals) != len(positions):
                normals = [0.0, 0.0, 1.0] * n_verts

            pos_off, pos_len = _append_f32(positions)
            nrm_off, nrm_len = _append_f32(normals)
            idx_off, idx_len = _append_u32(indices)
            vmin, vmax = _bounds(positions)

            pos_acc = _add_accessor(
                pos_off, pos_len,
                component_type=5126, count=n_verts, type_name="VEC3",
                min_v=vmin, max_v=vmax,
            )
            nrm_acc = _add_accessor(
                nrm_off, nrm_len,
                component_type=5126, count=n_verts, type_name="VEC3",
            )
            idx_acc = _add_accessor(
                idx_off, idx_len,
                component_type=5125, count=len(indices), type_name="SCALAR",
            )

            mesh_idx = len(meshes)
            mesh_entry: dict[str, Any] = {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc},
                        "indices": idx_acc,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
            meshes.append(mesh_entry)
            node_mesh[name] = mesh_idx

    # 确保所有节点都在 children_of 中
    for name in node_names:
        children_of.setdefault(name, [])

    # BFS 排序节点 (保证父节点在子节点之前) + 循环引用检测
    # 任何节点在遍历中第二次出现即为循环（glTF node hierarchy 必须是 DAG）
    ordered: list[str] = [root_name]
    seen: set[str] = {root_name}
    queue: list[str] = list(children_of.get(root_name, []))
    while queue:
        current = queue.pop(0)
        if current in seen:
            raise ValueError(
                f"GLB node hierarchy contains a cycle at node '{current}'. "
                f"Check the 'parent' field of all nodes — a node cannot "
                f"appear as its own ancestor."
            )
        seen.add(current)
        ordered.append(current)
        for child in sorted(children_of.get(current, [])):
            if child not in seen:
                queue.append(child)
    # 处理孤儿节点
    for name in sorted(node_names - seen):
        ordered.append(name)

    # 构建 glTF nodes
    name_to_idx = {name: idx for idx, name in enumerate(ordered)}
    gltf_nodes: list[dict[str, Any]] = []
    for name in ordered:
        nd: dict[str, Any] = {"name": name}
        if name in node_mesh:
            nd["mesh"] = node_mesh[name]
        kids = children_of.get(name) or []
        if kids:
            nd["children"] = [name_to_idx[c] for c in kids if c in name_to_idx]
        if name in node_extras:
            nd["extras"] = node_extras[name]
        if name in node_matrix:
            nd["matrix"] = node_matrix[name]
        gltf_nodes.append(nd)

    if not gltf_nodes:
        raise ValueError("no nodes in scene")

    # 确保根节点在 scenes 中
    if root_name not in name_to_idx:
        root_name = ordered[0]

    scene: dict[str, Any] = {"nodes": [name_to_idx[root_name]]}
    if scene_extras:
        scene["extras"] = scene_extras

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "cad-backend-hierarchical-glb"},
        "scene": 0,
        "scenes": [scene],
        "nodes": gltf_nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
        "materials": materials,
    }

    json_data = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return pack_glb(json_data, bytes(blob))


def hierarchical_meshes_to_glb_bytes(
    meshes: list[HierarchicalMesh],
    *,
    scene_extras: dict[str, Any] | None = None,
) -> bytes:
    """将层级化网格列表导出为 GLB.

    简化版接口，直接从网格数据构建层级。

    Args:
        meshes: 网格列表，每项包含 name, part_id, positions, indices, normals, matrix 等
        scene_extras: 场景额外数据

    Returns:
        GLB 字节数据
    """
    nodes: list[dict[str, Any]] = []
    name_to_parent: dict[str, str] = {}

    for mesh in meshes:
        name = mesh.get("name", f"mesh_{len(nodes)}")
        part_id = mesh.get("part_id", name)
        parent = mesh.get("parent")

        node: dict[str, Any] = {
            "name": name,
            "part_id": part_id,
            "parent": parent,
            "positions": mesh.get("positions"),
            "indices": mesh.get("indices"),
            "normals": mesh.get("normals"),
            "matrix": mesh.get("matrix"),
            "extras": {
                "cad": {
                    "role": "part",
                    "part_id": part_id,
                }
            },
        }

        if mesh.get("color"):
            node["color"] = mesh["color"]

        nodes.append(node)

    return hierarchical_scene_to_glb_bytes(nodes, scene_extras=scene_extras)
