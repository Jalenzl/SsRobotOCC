"""Feature-extraction HTTP API (analyze / path / face_spread).

All endpoints take a ``model_id`` (from ``POST /api/v1/cad/upload``) and
optionally a ``face_id``; the heavy lifting is done by
``app.services.feature_service`` which wraps the OCC algorithm modules.

The router accepts both ``application/json`` and ``multipart/form-data``
bodies, since the frontend ``RobotClass`` posts a FormData with
``options_json`` stringified. JSON body remains the recommended
machine-friendly path.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.services import cad_cache, feature_service
from app.utils.occ_guard import occ_installed

router = APIRouter(prefix="/cad", tags=["feature"])


@router.get("/health")
def feature_health() -> dict:
    """Health check (separate path from ``/cad/status`` to avoid clobbering
    the upload-only status endpoint)."""
    return {
        "pythonocc_available": occ_installed(),
        "module": "feature",
    }


@router.get("/feature/status")
def feature_status() -> dict:
    return {
        "pythonocc_available": occ_installed(),
        "endpoints": [
            "POST /api/v1/cad/analyze/face_spread",
            "POST /api/v1/cad/analyze/part_spread",
            "POST /api/v1/cad/analyze/path",
            "POST /api/v1/cad/machining/paths",
            "POST /api/v1/cad/machining/paths/multi",
            "GET  /api/v1/cad/machining/craft_params",
            "GET  /api/v1/cad/machining/path_types",
            "GET  /api/v1/cad/faces",
            "GET  /api/v1/cad/parts",
        ],
    }


def _load_step(model_id: str) -> tuple[bytes, str]:
    if not occ_installed():
        raise HTTPException(status_code=503, detail="pythonOCC 未安装")
    try:
        path = cad_cache.get_step_path(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"model_id 未找到: {model_id}")
    try:
        meta = cad_cache.get_meta(model_id)
    except KeyError:
        meta = {"filename": path.name}
    return path.read_bytes(), meta.get("filename", path.name)


@router.get("/faces")
def list_faces(model_id: str = Query(..., description="model_id from /cad/upload")):
    raw, name = _load_step(model_id)
    faces = feature_service.list_faces(raw, name)
    return JSONResponse(content={"model_id": model_id, "faces": faces, "count": len(faces)})


@router.get("/parts")
def list_parts(model_id: str = Query(..., description="model_id from /cad/upload")):
    raw, name = _load_step(model_id)
    parts = feature_service.list_parts(raw, name)
    return JSONResponse(content={"model_id": model_id, "parts": parts, "count": len(parts)})


def _parse_options(options_json: str | None, options_dict: dict | None) -> dict:
    """Pull options from a JSON string, a dict, or both (form fields)."""
    out: dict[str, Any] = {}
    if options_json:
        try:
            out.update(json.loads(options_json))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"options_json 无效: {exc}")
    if options_dict:
        out.update(options_dict)
    return out


def _resolve_selector(face_id: str) -> str:
    """Normalise a face_id from frontend (face_<n>, face_<n>_<n>, part_<n>, bare int, ...)."""
    fid = (face_id or "").strip()
    if not fid:
        return fid
    if fid.isdigit():
        return f"face_{fid}"
    # Handle compound IDs like face_78_4 — extract first numeric segment as global face index
    lower = fid.lower()
    if lower.startswith("face_"):
        suffix = fid[5:]
        first_num = suffix.split("_")[0]
        if first_num.isdigit():
            return f"face_{first_num}"
    return fid


@router.post("/analyze/face_spread")
async def analyze_face_spread(
    request: Request,
    model_id: str | None = Form(default=None),
    face_id: str | None = Form(default=None),
    part_id: str | None = Form(default=None),
    analyze_mode: str | None = Form(default=None),
    options_json: str | None = Form(default=None),
    linear_deflection: float | None = Form(default=None),
    angular_deflection: float | None = Form(default=None),
    work_plane: str | None = Form(default=None),
    hole_diameter_min: float | None = Form(default=None),
    hole_diameter_max: float | None = Form(default=None),
    include_cylinder_holes: bool | None = Form(default=None),
    closure_tol: float | None = Form(default=None),
    enhanced_params: bool | None = Form(default=None),
    target_face_id: str | None = Form(default=None),
) -> JSONResponse:
    """Analyze the wires/contours/holes of a single face or whole part on a cached model.

    Accepts ``multipart/form-data`` (frontend FormData) or ``application/json``
    (machine clients). Frontend posts::

        model_id = ...
        face_id  = "face_12"  # 单面分析时传入 face ID
        part_id  = "part_0"   # 整零件分析时传入 part ID
        analyze_mode = "face" | "part"
        options_json = '{"linear_deflection":0.1,"work_plane":"auto",...}'
        enhanced_params = true  # Enable laser-software-style parameters

    ``analyze_mode`` determines the analysis scope:
    - "face" (default if face_id looks like a face): analyze one face
    - "part" (default for part IDs / mesh names): analyze whole solid

    ``target_face_id`` (part 模式): 只返回与该面法向对齐的同侧特征。
    """
    if not (model_id and (face_id or part_id)):
        # Try JSON body if no form fields were supplied.
        try:
            payload = await request.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            model_id = model_id or payload.get("model_id")
            face_id = face_id or payload.get("face_id")
            part_id = part_id or payload.get("part_id")
            analyze_mode = analyze_mode or payload.get("analyze_mode")
            target_face_id = target_face_id or payload.get("target_face_id")
            opts = _parse_options(payload.pop("options_json", None), payload)
        else:
            opts = {}
    else:
        opts = _parse_options(options_json, None)

    if not (model_id and (face_id or part_id)):
        raise HTTPException(status_code=400, detail="model_id and (face_id or part_id) are required")

    raw, name = _load_step(model_id)

    lin = float(opts.get("linear_deflection", linear_deflection if linear_deflection is not None else 0.1))
    ang = float(opts.get("angular_deflection", angular_deflection if angular_deflection is not None else 0.5))
    wp = str(opts.get("work_plane", work_plane or "auto"))
    wp_normal = opts.get("work_plane_normal")
    dmin = float(opts.get("hole_diameter_min", hole_diameter_min if hole_diameter_min is not None else 0.5))
    dmax = float(opts.get("hole_diameter_max", hole_diameter_max if hole_diameter_max is not None else 500.0))
    cyl = bool(opts.get("include_cylinder_holes", include_cylinder_holes if include_cylinder_holes is not None else False))
    ctl = float(opts.get("closure_tol", closure_tol if closure_tol is not None else 0.5))
    enh = bool(opts.get("enhanced_params", enhanced_params if enhanced_params is not None else False))
    tfid = target_face_id or opts.get("target_face_id")

    # ── 模式决策 ───────────────────────────────────────────────────────
    # 优先级：analyze_mode 显式参数 > face_id 前缀判断
    mode = (analyze_mode or "").strip().lower()
    if mode == "face":
        sel = _resolve_selector(str(face_id or part_id))
        is_part = False
    elif mode == "part":
        raw_sel = part_id or face_id
        sel = _resolve_selector(str(raw_sel))
        if not sel.lower().startswith("part_"):
            sel = f"part_{sel}" if sel.isdigit() else sel
        is_part = True
    else:
        # 兼容旧行为：face_id 优先走单面分析
        raw_sel = face_id or part_id
        sel = _resolve_selector(str(raw_sel))
        sel_lower = sel.lower()
        is_part = sel_lower.startswith("part_")
        if not is_part and not (sel_lower.startswith("face_") or sel.isdigit()):
            sel = f"part_{sel}" if sel.isdigit() else sel
            is_part = True

    try:
        if is_part:
            result = feature_service.analyze_part_spread(
                step_bytes=raw,
                filename_hint=name,
                part_id=sel,
                linear_deflection=lin,
                angular_deflection=ang,
                work_plane=wp,
                work_plane_normal=wp_normal,
                hole_diameter_min=dmin,
                hole_diameter_max=dmax,
                include_cylinder_holes=cyl,
                closure_tol=ctl,
                enhanced_params=enh,
                target_face_id=tfid,
            )
        else:
            result = feature_service.analyze_face_spread(
                step_bytes=raw,
                filename_hint=name,
                face_id=sel,
                linear_deflection=lin,
                angular_deflection=ang,
                work_plane=wp,
                work_plane_normal=wp_normal,
                hole_diameter_min=dmin,
                hole_diameter_max=dmax,
                include_cylinder_holes=cyl,
                closure_tol=ctl,
                enhanced_params=enh,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return JSONResponse(content=result)


@router.post("/analyze/part_spread")
async def analyze_part_spread(
    request: Request,
    model_id: str | None = Form(default=None),
    part_id: str | None = Form(default=None),
    face_id: str | None = Form(default=None),  # alias for part_id (frontend compat)
    options_json: str | None = Form(default=None),
    linear_deflection: float | None = Form(default=None),
    angular_deflection: float | None = Form(default=None),
    work_plane: str | None = Form(default=None),
    hole_diameter_min: float | None = Form(default=None),
    hole_diameter_max: float | None = Form(default=None),
    include_cylinder_holes: bool | None = Form(default=None),
    closure_tol: float | None = Form(default=None),
    enhanced_params: bool | None = Form(default=None),
    target_face_id: str | None = Form(default=None),
) -> JSONResponse:
    """Analyse a whole part (Solid). Same body shape as ``face_spread`` but
    ``part_id`` selects a Solid. ``face_id`` is accepted as an alias for
    backward compatibility with the frontend (which always uses ``face_id``).

    ``target_face_id`` optionally narrows the result to features of faces
    on the same side as the given face. Use this to suppress duplicate
    back-side features on a plate.
    """
    if not (model_id and (part_id or face_id)):
        try:
            payload = await request.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            model_id = model_id or payload.get("model_id")
            part_id = part_id or payload.get("part_id") or payload.get("face_id")
            target_face_id = target_face_id or payload.get("target_face_id")
            opts = _parse_options(payload.pop("options_json", None), payload)
        else:
            opts = {}
    else:
        opts = _parse_options(options_json, None)

    sel_raw = part_id or face_id
    if not (model_id and sel_raw):
        raise HTTPException(status_code=400, detail="model_id and part_id are required")

    sel = _resolve_selector(str(sel_raw))
    if not sel.lower().startswith("part_"):
        sel = f"part_{sel}" if sel.isdigit() else sel

    raw, name = _load_step(model_id)

    lin = float(opts.get("linear_deflection", linear_deflection if linear_deflection is not None else 0.1))
    ang = float(opts.get("angular_deflection", angular_deflection if angular_deflection is not None else 0.5))
    wp = str(opts.get("work_plane", work_plane or "auto"))
    wp_normal = opts.get("work_plane_normal")
    dmin = float(opts.get("hole_diameter_min", hole_diameter_min if hole_diameter_min is not None else 0.5))
    dmax = float(opts.get("hole_diameter_max", hole_diameter_max if hole_diameter_max is not None else 500.0))
    cyl = bool(opts.get("include_cylinder_holes", include_cylinder_holes if include_cylinder_holes is not None else False))
    ctl = float(opts.get("closure_tol", closure_tol if closure_tol is not None else 0.5))
    enh = bool(opts.get("enhanced_params", enhanced_params if enhanced_params is not None else False))
    tfid = target_face_id or opts.get("target_face_id")

    try:
        result = feature_service.analyze_part_spread(
            step_bytes=raw,
            filename_hint=name,
            part_id=sel,
            linear_deflection=lin,
            angular_deflection=ang,
            work_plane=wp,
            work_plane_normal=wp_normal,
            hole_diameter_min=dmin,
            hole_diameter_max=dmax,
            include_cylinder_holes=cyl,
            closure_tol=ctl,
            enhanced_params=enh,
            target_face_id=tfid,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return JSONResponse(content=result)


@router.post("/analyze/path")
async def analyze_path(
    request: Request,
    model_id: str | None = Form(default=None),
    face_id: str | None = Form(default=None),
    options_json: str | None = Form(default=None),
    linear_deflection: float | None = Form(default=None),
    work_plane: str | None = Form(default=None),
    hole_diameter_min: float | None = Form(default=None),
    hole_diameter_max: float | None = Form(default=None),
    closure_tol: float | None = Form(default=None),
) -> JSONResponse:
    """Path-plan placeholder.

    Returns the same contour/hole data as ``face_spread`` plus a minimal
    toolpath suggestion (one entry per contour/hole). The real toolpath
    generator (multi-pass / stepover / lead-in) lands in a follow-up PR
    once the new contour schema is validated end-to-end.
    """
    if not (model_id and face_id):
        try:
            payload = await request.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            model_id = model_id or payload.get("model_id")
            face_id = face_id or payload.get("face_id")
            opts = _parse_options(payload.pop("options_json", None), payload)
        else:
            opts = {}
    else:
        opts = _parse_options(options_json, None)

    if not (model_id and face_id):
        raise HTTPException(status_code=400, detail="model_id and face_id are required")

    raw, name = _load_step(model_id)

    lin = float(opts.get("linear_deflection", linear_deflection if linear_deflection is not None else 0.5))
    wp = str(opts.get("work_plane", work_plane or "auto"))
    wp_normal = opts.get("work_plane_normal")
    dmin = float(opts.get("hole_diameter_min", hole_diameter_min if hole_diameter_min is not None else 0.5))
    dmax = float(opts.get("hole_diameter_max", hole_diameter_max if hole_diameter_max is not None else 500.0))
    ctl = float(opts.get("closure_tol", closure_tol if closure_tol is not None else 0.5))

    try:
        result = feature_service.analyze_face_spread(
            step_bytes=raw,
            filename_hint=name,
            face_id=str(face_id),
            linear_deflection=lin,
            angular_deflection=0.5,
            work_plane=wp,
            work_plane_normal=wp_normal,
            hole_diameter_min=dmin,
            hole_diameter_max=dmax,
            include_cylinder_holes=False,
            closure_tol=ctl,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    paths: list[dict] = []
    for c in result.get("contours", []):
        if c.get("is_outer"):
            paths.append({
                "kind": "boundary",
                "contour_id": c["id"],
                "points": c.get("center") or [],
            })
    for h in result.get("holes", []):
        paths.append({
            "kind": "hole",
            "hole_id": h["id"],
            "points": h.get("center") or [],
            "diameter": h.get("diameter"),
        })
    result["toolpath_suggestion"] = paths
    return JSONResponse(content=result)


# ── CAM Path Generation Endpoints (参考激光软件架构) ──────────────────────────────

@router.post("/machining/paths")
async def generate_machining_paths(
    request: Request,
    model_id: str | None = Form(default=None),
    face_id: str | None = Form(default=None),
    part_id: str | None = Form(default=None),
    options_json: str | None = Form(default=None),
    linear_deflection: float | None = Form(default=None),
    angular_deflection: float | None = Form(default=None),
    work_plane: str | None = Form(default=None),
    apply_craft_params: bool | None = Form(default=None),
    generate_lead_lines: bool | None = Form(default=None),
) -> JSONResponse:
    """Generate CAM machining paths from feature extraction results.

    This endpoint combines feature analysis with path planning to produce
    robot-ready machining data, inspired by the SmartLaser architecture:

    1. Analyze face/part for features (contours, holes)
    2. Generate machining paths (MachiningPath with CAMLines)
    3. Apply craft parameters based on contour type
    4. Generate lead-in / lead-out lines

    Returns MachiningResult with machining groups containing paths and lines.
    """
    from app.services.machining_service import generate_machining_paths as _gen_paths

    # Parse request (reuse existing logic)
    if not (model_id and (face_id or part_id)):
        try:
            payload = await request.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            model_id = model_id or payload.get("model_id")
            face_id = face_id or payload.get("face_id")
            part_id = part_id or payload.get("part_id")
            opts = _parse_options(payload.pop("options_json", None), payload)
        else:
            opts = {}
    else:
        opts = _parse_options(options_json, None)

    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    sel_raw = face_id or part_id
    if not sel_raw:
        raise HTTPException(status_code=400, detail="face_id or part_id is required")

    # Parse options
    lin = float(opts.get("linear_deflection", linear_deflection if linear_deflection is not None else 0.1))
    ang = float(opts.get("angular_deflection", angular_deflection if angular_deflection is not None else 0.5))
    wp = str(opts.get("work_plane", work_plane or "auto"))
    apply_craft = bool(opts.get("apply_craft_params", apply_craft_params if apply_craft_params is not None else True))
    gen_leads = bool(opts.get("generate_lead_lines", generate_lead_lines if generate_lead_lines is not None else True))

    # Normalize selector
    sel = _resolve_selector(str(sel_raw))
    if not sel.lower().startswith("part_") and not sel.lower().startswith("face_"):
        sel = f"part_{sel}" if sel.isdigit() else sel

    raw, name = _load_step(model_id)

    try:
        # First get feature analysis result
        if sel.lower().startswith("part_"):
            feature_result = feature_service.analyze_part_spread(
                step_bytes=raw,
                filename_hint=name,
                part_id=sel,
                linear_deflection=lin,
                angular_deflection=ang,
                work_plane=wp,
                hole_diameter_min=0.5,
                hole_diameter_max=500.0,
                include_cylinder_holes=False,
                closure_tol=0.5,
            )
        else:
            feature_result = feature_service.analyze_face_spread(
                step_bytes=raw,
                filename_hint=name,
                face_id=sel,
                linear_deflection=lin,
                angular_deflection=ang,
                work_plane=wp,
                hole_diameter_min=0.5,
                hole_diameter_max=500.0,
                include_cylinder_holes=False,
                closure_tol=0.5,
            )

        # Generate machining paths
        machining_result = _gen_paths(
            feature_result=feature_result,
            apply_craft_params=apply_craft,
            generate_lead_lines=gen_leads,
        )

        return JSONResponse(content=machining_result.model_dump())

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/machining/craft_params")
def get_craft_params(
    contour_type: str = Query(..., description="Contour type: circle, slot, rectangle, hexagon, outer"),
    thickness: float | None = Query(default=None, description="Material thickness in mm"),
) -> JSONResponse:
    """Get default craft parameters for a specific contour type.

    This endpoint provides the craft parameters that would be applied
    during machining path generation. Parameters include:
    - velocity: Cutting speed (mm/s)
    - power: Laser power (0-100%)
    - duty: Duty cycle (0-100%)
    - frequency: Pulse frequency (Hz)
    - acc: Acceleration
    - lead_in: Lead-in length (mm)

    Args:
        contour_type: The type of contour feature
        thickness: Optional material thickness for parameter scaling

    Returns:
        CraftParameters for the given contour type
    """
    from app.services.machining_service import get_craft_parameters_by_contour_type

    valid_types = ["circle", "slot", "rectangle", "hexagon", "outer", "unknown"]
    if contour_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid contour_type. Must be one of: {valid_types}"
        )

    params = get_craft_parameters_by_contour_type(contour_type, thickness)
    return JSONResponse(content=params.model_dump())


# ── Multi-Hole Path Generation (with click-order preservation) ──────────────
#
# Front-end flow:
#   1. User multi-selects holes on the 3D viewer (click / shift-click / drag)
#   2. Front-end keeps an *ordered* list of hole_id in click sequence
#   3. Front-end POSTs that list to this endpoint as JSON:
#        { "model_id": "...", "face_id": "face_12", "hole_ids": ["hole_3","hole_1","hole_7"] }
#   4. Backend generates one MachiningPath per hole in that order, plus
#      cross-path idle (transit) lines; outer contours are optional.


def _normalize_hole_ids(raw: Any) -> list[str]:
    """Coerce a hole_ids payload (list / JSON string / None) to list[str]."""
    if raw is None:
        return []
    if isinstance(raw, str):
        # Accept JSON-array string or comma-separated fallback
        s = raw.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400, detail=f"hole_ids JSON 解析失败: {exc}"
                ) from exc
            if not isinstance(parsed, list):
                raise HTTPException(status_code=400, detail="hole_ids 必须是列表")
            return [str(x) for x in parsed]
        return [tok.strip() for tok in s.split(",") if tok.strip()]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise HTTPException(status_code=400, detail=f"hole_ids 类型不支持: {type(raw).__name__}")


@router.post("/machining/paths/multi")
async def generate_machining_paths_multi(
    request: Request,
    model_id: str | None = Form(default=None),
    face_id: str | None = Form(default=None),
    part_id: str | None = Form(default=None),
    hole_ids_json: str | None = Form(default=None),
    include_outer: str | None = Form(default=None),
    apply_craft_params: str | None = Form(default=None),
    idle_velocity: str | None = Form(default=None),
) -> JSONResponse:
    """Generate CAM machining paths for a **user-selected set of holes** in
    click order.

    Differences from ``/machining/paths``:
    - Accepts a JSON-serialised ``hole_ids`` list (click order is preserved)
    - Optionally includes outer contours at the end (``include_outer=true``)
    - Emits cross-path idle (transit) lines between adjacent holes
    - Each ``MachiningPath`` carries ``order_index`` and ``source_hole_id``
    - The ``MachiningGroup`` carries ``path_order`` (flat sequence) and
      ``transition_lines`` (idle CAMLines)
    """
    from app.services.machining_service import generate_machining_paths_multi as _gen_paths_multi

    # ── Parse request body (support both form and JSON) ────────────────────
    payload: dict | None = None
    if not (model_id and (face_id or part_id)):
        try:
            payload = await request.json()
        except Exception:
            payload = None

    if payload is not None:
        model_id = model_id or payload.get("model_id")
        face_id = face_id or payload.get("face_id")
        part_id = part_id or payload.get("part_id")
        # hole_ids: prefer direct list, else JSON string
        if "hole_ids" in payload:
            raw_ids = payload["hole_ids"]
            if isinstance(raw_ids, str):
                hole_ids = _normalize_hole_ids(raw_ids)
            else:
                hole_ids = _normalize_hole_ids(raw_ids)
        else:
            hole_ids = []
        if "include_outer" in payload:
            include_outer_flag = bool(payload["include_outer"])
        else:
            include_outer_flag = include_outer in ("1", "true", "True", "yes")
        if "apply_craft_params" in payload:
            apply_craft = bool(payload["apply_craft_params"])
        else:
            apply_craft = apply_craft_params not in ("0", "false", "False", "no")
        if "idle_velocity" in payload:
            try:
                idle_v = float(payload["idle_velocity"])
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"idle_velocity 解析失败: {exc}"
                ) from exc
        else:
            idle_v = None
    else:
        hole_ids = _normalize_hole_ids(hole_ids_json)
        include_outer_flag = include_outer in ("1", "true", "True", "yes")
        apply_craft = apply_craft_params not in ("0", "false", "False", "no")
        try:
            idle_v = float(idle_velocity) if idle_velocity is not None and idle_velocity != "" else None
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"idle_velocity 解析失败: {exc}"
            ) from exc

    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if not (face_id or part_id):
        raise HTTPException(status_code=400, detail="face_id or part_id is required")
    if not hole_ids:
        raise HTTPException(
            status_code=400,
            detail="hole_ids is required and must be a non-empty list (click order)",
        )

    raw, name = _load_step(model_id)

    try:
        if part_id and (not face_id or part_id.lower().startswith("part_")):
            feature_result = feature_service.analyze_part_spread(
                step_bytes=raw, filename_hint=name, part_id=str(part_id),
                hole_diameter_min=0.5, hole_diameter_max=500.0,
                include_cylinder_holes=False, closure_tol=0.5,
            )
        else:
            feature_result = feature_service.analyze_face_spread(
                step_bytes=raw, filename_hint=name, face_id=str(face_id or part_id),
                hole_diameter_min=0.5, hole_diameter_max=500.0,
                include_cylinder_holes=False, closure_tol=0.5,
            )

        machining_result = _gen_paths_multi(
            feature_result=feature_result,
            hole_ids=hole_ids,
            include_outer=include_outer_flag,
            apply_craft_params=apply_craft,
            idle_velocity=idle_v,
        )
        return JSONResponse(content=machining_result.model_dump())

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/machining/path_types")
def get_path_types() -> JSONResponse:
    """Get available path types, inner path types, and out path types.

    Returns the enumeration values used for classifying machining paths,
    inspired by the SmartLaser InnerPathType and OutPathType enums.
    """
    return JSONResponse(content={
        "path_types": {
            "outer": "外轮廓 (Outer contour boundary)",
            "inner": "内轮廓 (Hole/internal contour)",
        },
        "inner_path_types": {
            "circle": "圆形孔",
            "slot": "槽形孔",
            "rectangle": "矩形孔",
            "hexagon": "六边形孔",
            "irregular": "异形孔",
        },
        "out_path_types": {
            "long_line": "长直线 (段长 ≥ 周长 × 40%)",
            "shorter_line": "短直线 (周长 × 10% < 段长 < 周长 × 40%)",
            "shortest_line": "最短直线 (段长 ≤ 周长 × 10%)",
            "big_arc": "平面大圆弧 (段内换算角度 > 60°)",
            "small_arc": "平面小圆弧 (段内换算角度 30°–60°)",
            "three_d_corner": "三维拐角 (段间夹角 > 60°)",
            "point": "过渡点 (段长 < 0.05mm)",
        },
        "cam_line_types": {
            "machining": "加工线 (Main cutting path)",
            "lead": "引线 (Lead-in/lead-out)",
            "cut_in": "切入线 (Cut-in)",
            "cut_out": "切出线 (Cut-out)",
            "fast": "快速移动",
            "idle": "过渡线 (Idle/transit)",
            "location": "定位点",
        },
    })
