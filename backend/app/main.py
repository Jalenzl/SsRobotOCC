"""FastAPI application entry."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import cors_allow_credentials, cors_allow_origins_raw, ensure_runtime_dirs
from app.routers import cad, convert, feature, health, ikfast, stp, stp_hierarchy

ensure_runtime_dirs()

app = FastAPI(
    title="URDF / CAD conversion backend",
    version="1.0.0",
    description=(
        "URDF（或含 mesh 的 zip）转 Babylon `babylon_robot_scene` JSON；"
        "STEP/STP 转 GLB（需安装 cascadio）；"
        "CAD 特征提取与刀路规划（需安装 pythonOCC）；"
        "FANUC M-20iA/35M 解析逆解（需编译 ikfast 本地库）。"
    ),
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

_cors_raw = cors_allow_origins_raw()
if _cors_raw:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    _cors_credentials = cors_allow_credentials()
else:
    _cors_origins = ["*"]
    _cors_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# GZip large responses (GLB / scene JSON) for clients that send
# `Accept-Encoding: gzip` (axios + browsers do this by default and
# transparently decompress). 2-3× transfer reduction for typical CAD
# GLBs because the binary stream contains long runs of zeros (empty
# attribute slots, padding) and floats with similar mantissas.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.exception_handler(StarletteHTTPException)
async def _starlette_http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """multipart 解析失败时给出可操作的提示（前端常误设 Content-Type）。"""
    if exc.status_code == 400 and exc.detail == "There was an error parsing the body":
        return JSONResponse(
            status_code=400,
            content={
                "detail": (
                    "请求体解析失败。POST /api/v1/cad/upload 必须使用 multipart/form-data，"
                    "字段名 file，且不要手动设置 Content-Type（让浏览器自动生成 boundary）。"
                    "若使用 axios，请用 FormData 且不要设置 headers['Content-Type']。"
                    "备选：POST /api/v1/cad/upload/binary，Content-Type: application/octet-stream，"
                    "body 为 STEP 文件字节，可选 Header X-Filename: model.stp 或 ?filename=model.stp"
                )
            },
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(HTTPException)
async def _fastapi_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app.include_router(health.router)
app.include_router(convert.router, prefix="/api/v1")
app.include_router(stp.router, prefix="/api/v1")
app.include_router(stp_hierarchy.router, prefix="/api/v1")
app.include_router(cad.router, prefix="/api/v1")
app.include_router(feature.router, prefix="/api/v1")
app.include_router(ikfast.router, prefix="/api/v1")
