"""CAD feature extraction & toolpath API for frontend."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from app.models.cad import (
    CadAnalyzeAndPathResponse,
    CadAnalyzeOptions,
    CadFaceAnalyzeAndPathResponse,
    CadFaceAnalyzeResult,
    CadFaceSpreadResult,
    CadAnalyzeResult,
    PathPlanOptions,
    PathPlanResult,
    WorkPlane,
)
from app.services import cad_cache
from app.services.cad_service import (
    OccNotInstalledError,
    analyze_and_plan,
    analyze_and_plan_path,
    analyze_face_and_plan,
    analyze_face_and_plan_path,
    analyze_step,
    analyze_step_face,
    analyze_step_face_path,
    analyze_step_face_spread,
    analyze_step_face_spread_path,
    analyze_step_path,
    plan_path_from_analyze,
    plan_path_from_face_analyze,
)
from app.utils.file_handler import ensure_step_filename, read_upload_file, require_extension
from app.utils.form_json import parse_optional_json_form, parse_required_json_form

router = APIRouter(prefix="/cad", tags=["cad"])


def _occ_http(e: OccNotInstalledError) -> HTTPException:
    return HTTPException(status_code=501, detail=str(e))


def _cache_path(model_id: str) -> Path:
    try:
        return cad_cache.get_step_path(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"model_id 不存在或已过期，请重新上传: {model_id}",
        ) from e


def _persist_step_upload(raw: bytes, filename: str) -> dict:
    """校验文件名并写入 STEP 缓存，返回 model_id 等元数据。"""
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    name = ensure_step_filename(filename)
    require_extension(name.lower(), (".stp", ".step"))
    cad_cache.prune_expired()
    return cad_cache.store_step(raw, name)


async def _resolve_step(
    file: UploadFile | None, model_id: str | None
) -> tuple[bytes | None, str | None, Path | None]:
    """Return (raw_bytes, filename, cached_path). Exactly one source is used."""
    mid = (model_id or "").strip()
    if mid and mid.lower() not in ("string", "null", "none"):
        return None, None, _cache_path(mid)
    if file is None:
        raise HTTPException(status_code=400, detail="需要提供 file 或 model_id 其中之一")
    raw, name = await read_upload_file(file)
    name = ensure_step_filename(name)
    require_extension(name.lower(), (".stp", ".step"))
    return raw, name, None


@router.get("/status")
def cad_status() -> dict:
    from app.utils.occ_guard import occ_installed

    return {
        "pythonocc_available": occ_installed(),
        "api_version": "1.1",
        "endpoints": [
            "POST /api/v1/cad/upload",
            "POST /api/v1/cad/upload/binary",
            "POST /api/v1/cad/analyze",
            "POST /api/v1/cad/analyze/face",
            "POST /api/v1/cad/analyze/face_spread",
            "POST /api/v1/cad/analyze/face_and_path",
            "POST /api/v1/cad/path/generate",
            "POST /api/v1/cad/path/generate/face",
            "POST /api/v1/cad/analyze_and_path",
        ],
    }


@router.post("/upload")
async def cad_upload(
    file: UploadFile = File(..., description="STEP/STP 零件文件，上传一次后用 model_id 复用"),
) -> JSONResponse:
    """上传 STEP 一次，返回 model_id；后续选面分析只需传 model_id，无需重传文件。

    要求 ``multipart/form-data``，字段名 **file**。
    不要手动设置 ``Content-Type``（须由浏览器自动生成 boundary）。
    若 multipart 解析失败，可改用 ``POST /cad/upload/binary`` 直接传文件字节。
    """
    raw, name = await read_upload_file(file)
    meta = _persist_step_upload(raw, name)
    return JSONResponse(content=meta)


@router.post("/upload/binary")
async def cad_upload_binary(
    request: Request,
    filename: str | None = Query(
        None,
        description="文件名，建议带 .stp/.step；也可用请求头 X-Filename",
    ),
    x_filename: str | None = Header(default=None, alias="X-Filename"),
) -> JSONResponse:
    """直接上传 STEP 原始字节（``application/octet-stream``），绕过 multipart 解析。

    适用于 axios/fetch 手动设置了 ``Content-Type: multipart/form-data`` 导致
    ``There was an error parsing the body`` 的前端场景。
    """
    raw = await request.body()
    hint = filename or x_filename or "model.stp"
    meta = _persist_step_upload(raw, hint)
    return JSONResponse(content=meta)


@router.post("/analyze", response_model=CadAnalyzeResult)
async def cad_analyze(
    file: UploadFile | None = File(None, description="STEP/STP 零件文件（或改用 model_id）"),
    model_id: str | None = Form(None, description="来自 /cad/upload 的 model_id（替代 file）"),
    options_json: str | None = Form(
        None,
        description='可选 JSON，留空即可。示例: {"linear_deflection":0.1,"work_plane":"auto"}',
    ),
) -> JSONResponse:
    raw, name, cached = await _resolve_step(file, model_id)
    opts = parse_optional_json_form(options_json, CadAnalyzeOptions, field_name="options_json")
    try:
        if cached is not None:
            result = analyze_step_path(cached, opts)
        else:
            result = analyze_step(raw, name, opts)
    except OccNotInstalledError as e:
        raise _occ_http(e) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"CAD 分析失败: {e}") from e
    return JSONResponse(content=result.model_dump())


@router.post("/analyze/face", response_model=CadFaceAnalyzeResult)
async def cad_analyze_face(
    file: UploadFile | None = File(None, description="STEP/STP 零件文件（或改用 model_id）"),
    face_id: str = Form(..., description='前端选中的面 ID，例如 "face_12"'),
    model_id: str | None = Form(None, description="来自 /cad/upload 的 model_id（替代 file）"),
    options_json: str | None = Form(
        None,
        description='可选 JSON，留空即可。示例: {"linear_deflection":0.1,"work_plane":"auto"}',
    ),
) -> JSONResponse:
    raw, name, cached = await _resolve_step(file, model_id)
    opts = parse_optional_json_form(options_json, CadAnalyzeOptions, field_name="options_json")
    try:
        if cached is not None:
            result = analyze_step_face_path(cached, face_id=face_id, options=opts)
        else:
            result = analyze_step_face(raw, name, face_id=face_id, options=opts)
    except OccNotInstalledError as e:
        raise _occ_http(e) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"单面 CAD 分析失败: {e}") from e
    return JSONResponse(content=result.model_dump())


@router.post("/analyze/face_spread", response_model=CadFaceSpreadResult)
async def cad_analyze_face_spread(
    file: UploadFile | None = File(None, description="STEP/STP 零件文件（或改用 model_id）"),
    face_id: str = Form(..., description='前端选中的种子面 ID，例如 "face_12"'),
    model_id: str | None = Form(None, description="来自 /cad/upload 的 model_id（替代 file）"),
    options_json: str | None = Form(
        None,
        description='可选 JSON，留空即可。示例: {"enable_depth":true,"work_plane":"auto"}',
    ),
) -> JSONResponse:
    """选中一个面 → 判定内/外表面 → 扩散到整个装配体同侧表面，识别孔/型腔/凸台及深度。"""
    raw, name, cached = await _resolve_step(file, model_id)
    opts = parse_optional_json_form(options_json, CadAnalyzeOptions, field_name="options_json")
    try:
        if cached is not None:
            result = analyze_step_face_spread_path(cached, face_id=face_id, options=opts)
        else:
            result = analyze_step_face_spread(raw, name, face_id=face_id, options=opts)
    except OccNotInstalledError as e:
        raise _occ_http(e) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"面扩散分析失败: {e}") from e
    return JSONResponse(content=result.model_dump())


@router.post("/path/generate/face", response_model=PathPlanResult)
async def cad_path_generate_face(
    analyze_json: str = Form(..., description="CadFaceAnalyzeResult JSON（来自 /analyze/face）"),
    path_options_json: str | None = Form(
        None,
        description='JSON: {"strategy":"combined","tool_diameter":6,"step_over":3}',
    ),
) -> JSONResponse:
    try:
        analyze = parse_required_json_form(analyze_json, CadFaceAnalyzeResult, field_name="analyze_json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"analyze_json 无效: {e}") from e
    popts = parse_optional_json_form(path_options_json, PathPlanOptions, field_name="path_options_json")
    try:
        result = plan_path_from_face_analyze(analyze, popts)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"单面路径生成失败: {e}") from e
    return JSONResponse(content=result.model_dump())


@router.post("/path/generate", response_model=PathPlanResult)
async def cad_path_generate(
    analyze_json: str = Form(..., description="CadAnalyzeResult JSON（来自 /analyze）"),
    path_options_json: str | None = Form(
        None,
        description='JSON: {"strategy":"combined","tool_diameter":6,"step_over":3}',
    ),
) -> JSONResponse:
    try:
        analyze = parse_required_json_form(analyze_json, CadAnalyzeResult, field_name="analyze_json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"analyze_json 无效: {e}") from e
    popts = parse_optional_json_form(path_options_json, PathPlanOptions, field_name="path_options_json")
    try:
        result = plan_path_from_analyze(analyze, popts)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"路径生成失败: {e}") from e
    return JSONResponse(content=result.model_dump())


@router.post("/analyze/face_and_path", response_model=CadFaceAnalyzeAndPathResponse)
async def cad_analyze_face_and_path(
    file: UploadFile | None = File(None, description="STEP/STP 零件文件（或改用 model_id）"),
    face_id: str = Form(..., description='前端选中的面 ID，例如 "face_12"'),
    model_id: str | None = Form(None, description="来自 /cad/upload 的 model_id（替代 file）"),
    analyze_options_json: str | None = Form(None),
    path_options_json: str | None = Form(None),
) -> JSONResponse:
    raw, name, cached = await _resolve_step(file, model_id)
    aopts = parse_optional_json_form(analyze_options_json, CadAnalyzeOptions, field_name="analyze_options_json")
    popts = parse_optional_json_form(path_options_json, PathPlanOptions, field_name="path_options_json")
    try:
        if cached is not None:
            analyze, path = analyze_face_and_plan_path(
                cached, face_id=face_id, analyze_options=aopts, path_options=popts
            )
        else:
            analyze, path = analyze_face_and_plan(
                raw, name, face_id=face_id, analyze_options=aopts, path_options=popts
            )
    except OccNotInstalledError as e:
        raise _occ_http(e) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"单面分析或路径失败: {e}") from e
    return JSONResponse(content=CadFaceAnalyzeAndPathResponse(analyze=analyze, path=path).model_dump())


@router.post("/analyze_and_path", response_model=CadAnalyzeAndPathResponse)
async def cad_analyze_and_path(
    file: UploadFile | None = File(None, description="STEP/STP 零件文件（或改用 model_id）"),
    model_id: str | None = Form(None, description="来自 /cad/upload 的 model_id（替代 file）"),
    analyze_options_json: str | None = Form(None),
    path_options_json: str | None = Form(None),
) -> JSONResponse:
    raw, name, cached = await _resolve_step(file, model_id)
    aopts = parse_optional_json_form(analyze_options_json, CadAnalyzeOptions, field_name="analyze_options_json")
    popts = parse_optional_json_form(path_options_json, PathPlanOptions, field_name="path_options_json")
    try:
        if cached is not None:
            result = analyze_and_plan_path(cached, aopts, popts)
        else:
            result = analyze_and_plan(raw, name, aopts, popts)
    except OccNotInstalledError as e:
        raise _occ_http(e) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"分析或路径失败: {e}") from e
    return JSONResponse(content=result.model_dump())


@router.post("/analyze/json", response_model=CadAnalyzeResult)
async def cad_analyze_json_body(
    file: UploadFile = File(...),
    linear_deflection: float = 0.1,
    work_plane: WorkPlane = WorkPlane.AUTO,
    hole_diameter_min: float = 0.5,
    include_cylinder_holes: bool = False,
) -> JSONResponse:
    """Query-style params for clients that prefer simple form fields."""
    opts = CadAnalyzeOptions(
        linear_deflection=linear_deflection,
        work_plane=work_plane,
        hole_diameter_min=hole_diameter_min,
        include_cylinder_holes=include_cylinder_holes,
    )
    raw, name = await read_upload_file(file)
    require_extension(name.lower(), (".stp", ".step"))
    try:
        result = analyze_step(raw, name, opts)
    except OccNotInstalledError as e:
        raise _occ_http(e) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(e)) from e
    return JSONResponse(content=result.model_dump())
