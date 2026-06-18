"""STEP/STP -> GLB (保留层级结构) API.

结合前端 insofworks2026 的逻辑，提供两种输出模式：
1. hierarchy: 完整层级树 (需要 pythonOCC XDE 模块)
2. flat: 扁平化输出 (仅零件层级，无需 XDE)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.occ.step_hierarchy import StepHierarchyReader, XDE_AVAILABLE, read_step_hierarchy
from app.services import cad_cache
from app.utils.cascadio_guard import cascadio_installed, cascadio_usable_with_occ
from app.utils.occ_guard import occ_installed
from app.utils.file_handler import (
    content_disposition_attachment,
    read_upload_file,
)
from app.utils.step_bytes import is_step_bytes, step_filename_hint

_BACKEND_ROOT = Path(__file__).parent.parent.parent.resolve()


router = APIRouter(prefix="/stp", tags=["stp-hierarchy"])


@router.get("/hierarchy/status")
def hierarchy_status() -> dict:
    """检查层级转换引擎可用性."""
    return {
        "xde_available": XDE_AVAILABLE,
        "pythonocc_available": occ_installed(),
        "convert_ready": XDE_AVAILABLE,
        "engine": "pythonocc-xde" if XDE_AVAILABLE else "none",
        "hint": (
            "pythonOCC XDE 可用：支持完整 STEP 产品层级"
            if XDE_AVAILABLE
            else "需要完整版 pythonOCC 来读取 STEP 层级结构"
        ),
    }


@router.post("/hierarchy/tree")
async def get_step_hierarchy(
    file: UploadFile = File(..., description="STEP/STP 零件或装配文件"),
) -> dict:
    """获取 STEP 文件的层级结构 (不生成 GLB).

    返回层级树 JSON，可用于前端构建树形 UI。
    """
    raw, name = await read_upload_file(file)
    if not name:
        name = "model.stp"
    elif not (name.lower().endswith(".stp") or name.lower().endswith(".step")):
        if not is_step_bytes(raw):
            raise HTTPException(status_code=400, detail="需要 .stp/.step 文件")

    if not XDE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="需要完整版 pythonOCC (XDE 模块) 来读取层级信息"
        )

    try:
        hierarchy = read_step_hierarchy(raw, name)
        return hierarchy
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"读取层级失败: {e}") from e


@router.post("/hierarchy/convert")
async def convert_stp_to_hierarchical_glb(
    file: UploadFile = File(..., description="STEP/STP 零件或装配文件 (.stp / .step)"),
    linear_deflection: float = Query(
        0.1,
        gt=0,
        description="线性偏差 (mm)，越小网格越细",
    ),
    angular_deflection: float = Query(
        0.5,
        gt=0,
        le=1.5,
        description="角度偏差 (rad)，默认 0.5",
    ),
    mode: str = Query(
        "hierarchy",
        description="转换模式: hierarchy=保留完整层级(需XDE), flat=扁平化零件层级",
    ),
    per_face: bool = Query(
        False,
        description=(
            "每个面生成独立 mesh（用于选中面做特征识别）。"
            "注意：面数很多时 GLB 会变大，建议对小模型或需单面分析时开启。"
        ),
    ),
    merge_faces: bool | None = Query(
        None,
        description=(
            "每个 Solid 内的所有面合并为单一 mesh（巨大 STEP 的关键优化："
            "减少 mesh 数量与顶点重复）。None=按文件大小自动决定（>= 5MB 自动合并）。"
        ),
    ),
    bypass_cache: bool = Query(
        False,
        description="跳过 GLB 缓存，强制重新三角化。",
    ),
) -> Response:
    """将 STEP 转换为带层级结构的 GLB.

    - **mode=hierarchy**: 保留完整 STEP 产品树，每个产品对应一个 GLTF Node
    - **mode=flat**: 扁平化输出，每个 Solid/Shell 对应一个 mesh 节点

    前端可通过 gltfJson.nodes 和 scene.extras.cad.hierarchy 来解析层级结构。
    """
    raw, name = await read_upload_file(file)

    lower_name = name.lower() if name else ""
    if name:
        if not (lower_name.endswith(".stp") or lower_name.endswith(".step")):
            if not is_step_bytes(raw):
                raise HTTPException(
                    status_code=400,
                    detail=f"需要 .stp/.step 扩展名，当前文件名: {name!r}",
                )
    elif not is_step_bytes(raw):
        raise HTTPException(status_code=400, detail="文件不是有效的 STEP 内容")

    hint = step_filename_hint(name or "model.stp")

    if mode not in ("hierarchy", "flat"):
        raise HTTPException(status_code=400, detail="mode 须为 hierarchy 或 flat")

    # Smart default: large STEP files (>5 MB raw, or any per-face-mode file)
    # automatically merge all faces of each Solid into a single mesh. This
    # avoids the per-face flat-shading blowup that turns a 100k-triangle
    # model into 300k vertices across hundreds of Babylon Meshes.
    if merge_faces is None:
        merge_faces = len(raw) >= 5 * 1024 * 1024 or mode == "flat"

        # GLB cache: keyed by STEP content hash + conversion params. Re-uploading
        # the same file (or re-converting with the same params) becomes instant.
        model_id = cad_cache.compute_model_id(raw)
        cached = None
        if not bypass_cache:
            cad_cache.store_step(raw, hint or "model.stp")
            from app.services import glb_cache

            cached = glb_cache.get_cached(
                model_id,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
                merge_faces=merge_faces,
                mode=mode,
                per_face=per_face,  # per_face changes GLB structure, must be part of cache key
            )
        if cached is not None:
            return _glb_response(cached, hint)

    try:
        if mode == "hierarchy":
            if not XDE_AVAILABLE:
                raise HTTPException(
                    status_code=501,
                    detail="层级模式需要 pythonOCC XDE 模块。请使用 mode=flat 或安装完整版 pythonOCC"
                )
            glb = _convert_with_hierarchy(
                raw, hint,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
                merge_faces=merge_faces,
                per_face=per_face,
            )
        else:
            glb = _convert_flat(
                raw, hint,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"STEP 转 GLB 失败: {e}") from e

    # Populate cache for next time.
    if not bypass_cache:
        try:
            from app.services import glb_cache

            glb_cache.put_cached(
                model_id,
                glb,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
                merge_faces=merge_faces,
                mode=mode,
                per_face=per_face,
            )
        except Exception:
            pass  # cache failures must never block the response

    return _glb_response(glb, hint)


def _glb_response(glb: bytes, hint: str) -> Response:
    return Response(
        content=glb,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": content_disposition_attachment(hint or "model.stp")},
    )


def _convert_with_hierarchy(
    data: bytes,
    filename: str,
    *,
    linear_deflection: float,
    angular_deflection: float,
    merge_faces: bool = False,
    per_face: bool = False,
) -> bytes:
    """使用 XDE 读取层级并生成层级化 GLB.

    ``merge_faces``: 将同一 Solid 内的所有 B-Rep 面合并为单一 mesh
    （在 worker 脚本里通过 ``merge_faces`` 命令行参数传递）。
    """
    import subprocess
    import sys

    from app.occ.step_hierarchy import StepHierarchyReader, XDE_AVAILABLE

    if not XDE_AVAILABLE:
        raise ImportError("需要 pythonOCC XDE 模块")

    # 在子进程中执行（避免 DLL 冲突）
    suffix = ".step" if filename.lower().endswith(".step") else ".stp"
    with tempfile.TemporaryDirectory() as td:
        step_path = Path(td) / f"input{suffix}"
        glb_path = Path(td) / "output.glb"
        meta_path = Path(td) / "meta.json"
        script_path = Path(td) / "hierarchy_worker.py"
        step_path.write_bytes(data)

        # Write worker script to a temp .py file (avoids encoding issues with
        # `python -c "..."` on Windows command-line argument encoding).
        script_path.write_text(_HIERARCHY_WORKER_SCRIPT, encoding="utf-8")

        env = __import__("os").environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUTF8"] = "1"

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(step_path),
                str(glb_path),
                str(meta_path),
                str(linear_deflection),
                str(angular_deflection),
                str(_BACKEND_ROOT),
                filename,
                "1" if merge_faces else "0",
                "1" if per_face else "0",
            ],
            capture_output=True,
            env=env,
            timeout=600,
            creationflags=creationflags,
        )

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
            raise ValueError(err or f"层级转换失败 (code {proc.returncode})")

        if not glb_path.is_file():
            raise ValueError("层级转换未生成输出文件")

        glb = glb_path.read_bytes()

        # 如果有 metadata，附加到 GLB
        if meta_path.is_file():
            import json
            meta = json.loads(meta_path.read_text())
            glb = _attach_hierarchy_metadata(glb, meta)

        return glb


def _attach_hierarchy_metadata(glb: bytes, meta: dict) -> bytes:
    """将层级元数据附加到 GLB 的 scene.extras 中."""
    from app.utils.raw_glb import validate_glb_bytes
    import json
    import struct

    validate_glb_bytes(glb)

    # 解析现有 GLB
    header = glb[:12]
    json_len = struct.unpack("<I", glb[12:16])[0]
    json_start = 20
    json_data = glb[json_start:json_start + json_len].rstrip(b" ")
    gltf = json.loads(json_data.decode("utf-8"))

    # 添加层级元数据到 scene.extras
    if "scenes" in gltf and gltf["scenes"]:
        scene = gltf["scenes"][0]
        if "extras" not in scene:
            scene["extras"] = {}
        scene["extras"]["cad"] = {
            "schema": "robotlaser.step.hierarchy/v1",
            "hierarchy": meta,
        }

    # 重新打包
    new_json = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8")
    from app.utils.raw_glb import pack_glb

    # GLB spec: JSON chunk 后是 4 字节对齐的 padding，然后是 BIN chunk。
    # BIN chunk 头 = [uint32 length][4-byte type "BIN\0"]，所以 BIN 数据从
    # json_start + json_len + padding + 8 开始；其 length 字段在 type 之前。
    json_padded_len = json_len + ((4 - (json_len % 4)) % 4)
    bin_offset = json_start + json_padded_len
    if (
        bin_offset + 8 <= len(glb)
        and glb[bin_offset + 4 : bin_offset + 8] == b"BIN\x00"
    ):
        bin_len = struct.unpack("<I", glb[bin_offset : bin_offset + 4])[0]
        bin_data = glb[bin_offset + 8 : bin_offset + 8 + bin_len]
    else:
        bin_data = b""

    return pack_glb(new_json, bin_data)


def _convert_flat(
    data: bytes,
    filename: str,
    *,
    linear_deflection: float,
    angular_deflection: float,
) -> bytes:
    """扁平化转换：每个 Solid 对应一个 mesh 节点."""
    from app.services.stp_converter import stp_bytes_to_glb

    return stp_bytes_to_glb(
        data,
        filename,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        pick_level="part",
    )


# 子进程脚本：用于 XDE 层级转换
_HIERARCHY_WORKER_SCRIPT = r"""
import sys
import json
from pathlib import Path

def main():
    step_path = Path(sys.argv[1])
    glb_path = Path(sys.argv[2])
    meta_path = Path(sys.argv[3])
    linear_deflection = float(sys.argv[4])
    angular_deflection = float(sys.argv[5])
    backend_root = Path(sys.argv[6])
    filename = sys.argv[7] if len(sys.argv) > 7 else step_path.name
    merge_faces = (len(sys.argv) > 8 and sys.argv[8] == "1")
    per_face = (len(sys.argv) > 9 and sys.argv[9] == "1")

    # Add backend to path
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from app.occ.step_hierarchy import StepHierarchyReader
    from app.occ.mesh_buffers import (
        _merged_mesh_buffers_for_shape,
        shape_to_cad_drawables,
    )
    from app.utils.raw_glb import (
        hierarchical_scene_to_glb_bytes,
        cad_scene_to_glb_bytes,
        validate_glb_bytes,
    )

    data = step_path.read_bytes()

    reader = StepHierarchyReader()
    root, shapes = reader.read(data, filename)

    if per_face:
        # Per-face mode: each face becomes an independent mesh (for single-face feature analysis).
        # Each Part's faces are expanded into individual drawables.
        # Face mesh parent = "Part_<n>" group (consistent with backend face_id = "face_<n>").
        from collections import deque

        nodes: list[dict] = []
        used_names: dict[str, int] = {}
        queue: deque[tuple] = deque()

        def unique_name(name: str) -> str:
            count = used_names.get(name, 0)
            used_names[name] = count + 1
            return name if count == 0 else f"{name}_{count}"

        def process_node(node, parent_unique):
            result = {
                "name": unique_name(node.name),
                "parent": parent_unique,
                "matrix": node.location_matrix,
                "kind": "group",  # container node, no geometry; glTF writer skips it
                "extras": {
                    "cad": {
                        "role": "assembly" if node.is_assembly else "part",
                        "part_id": node.part_id,
                    }
                },
            }
            if node.color:
                result["color"] = list(node.color)
            return result

        def add_shape_faces(shape, parent_group_name):
            # Expand shape faces into individual mesh drawables (parent = part group).
            if shape is None:
                return
            try:
                drawables, _, _ = shape_to_cad_drawables(
                    shape,
                    linear_deflection=linear_deflection,
                    angular_deflection=angular_deflection,
                    filename=filename,
                )
                for d in drawables:
                    nodes.append(d)
            except Exception as e:
                print(f"Warning: failed to add shape faces for {parent_group_name}: {e}", file=sys.stderr)

        # root node (kind="group" avoids collision with same-named drawable mesh)
        root_data = process_node(root, None)
        nodes.append(root_data)

        # root itself may have a shape (e.g. plate_with_slot_100 where root.is_assembly=False)
        if root.shape is not None:
            add_shape_faces(root.shape, root_data["name"])

        # BFS traverse all child nodes
        for child in root.children:
            queue.append((child, root_data["name"]))

        while queue:
            node, parent_name = queue.popleft()
            node_data = process_node(node, parent_name)
            nodes.append(node_data)

            if node.shape is not None:
                add_shape_faces(node.shape, node_data["name"])

            for child in node.children:
                queue.append((child, node_data["name"]))

        hierarchy_meta = {
            "schema": "robotlaser.step.hierarchy/v1",
            "filename": filename,
            "total_parts": len([n for n in nodes if n.get("positions")]),
            "total_nodes": len(nodes),
        }

        glb = cad_scene_to_glb_bytes(
            nodes,
            scene_extras={"cad": hierarchy_meta},
            root_name=root.name,
        )
    else:
        # Merged-face mode (default): each Part = one merged mesh.
        from collections import deque

        nodes: list[dict] = []
        node_map: dict[str, dict] = {}
        used_names: dict[str, int] = {}
        queue: deque[tuple] = deque()

        def unique_name(name: str) -> str:
            count = used_names.get(name, 0)
            used_names[name] = count + 1
            return name if count == 0 else f"{name}_{count}"

        def process_node(node, parent_unique):
            result = {
                "name": unique_name(node.name),
                "parent": parent_unique,
                "matrix": node.location_matrix,
                "extras": {
                    "cad": {
                        "role": "assembly" if node.is_assembly else "part",
                        "part_id": node.part_id,
                    }
                },
            }
            if node.color:
                result["color"] = list(node.color)
            return result

        def mesh_node(node_data, shape):
            if shape is None:
                return
            try:
                pos, idx, nrm = _merged_mesh_buffers_for_shape(
                    shape,
                    linear_deflection,
                    angular_deflection,
                    remesh=True,
                )
                node_data["positions"] = pos
                node_data["indices"] = idx
                node_data["normals"] = nrm
            except Exception as e:
                print(f"Warning: failed to mesh {node_data['name']}: {e}", file=sys.stderr)

        root_data = process_node(root, None)
        node_map[root.part_id] = root_data
        nodes.append(root_data)
        mesh_node(root_data, root.shape)

        for child in root.children:
            queue.append((child, root_data["name"]))

        while queue:
            node, parent_name = queue.popleft()
            node_data = process_node(node, parent_name)
            node_map[node.part_id] = node_data
            mesh_node(node_data, node.shape)
            nodes.append(node_data)
            current_unique = node_data["name"]
            for child in node.children:
                queue.append((child, current_unique))

        hierarchy_meta = {
            "schema": "robotlaser.step.hierarchy/v1",
            "filename": filename,
            "total_parts": len([n for n in nodes if n.get("positions")]),
            "total_nodes": len(nodes),
        }

        glb = hierarchical_scene_to_glb_bytes(
            nodes,
            scene_extras={"cad": hierarchy_meta},
            root_name=root.name,
        )

    # Validate generated GLB
    try:
        validate_glb_bytes(glb)
    except Exception as e:
        print(f"ERROR: generated GLB is invalid: {e}", file=sys.stderr)
        raise

    glb_path.write_bytes(glb)
    meta_path.write_text(json.dumps(hierarchy_meta, indent=2))
    print("OK", file=sys.stdout)

if __name__ == "__main__":
    main()
"""
