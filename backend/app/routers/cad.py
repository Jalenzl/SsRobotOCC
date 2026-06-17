"""CAD HTTP API — file intake and pythonOCC availability only.

Feature recognition endpoints (analyze / path / face_spread / ...) have been
removed along with the feature subsystem. Re-add them once the new feature
module is available; see ``app/occ/`` for the underlying pythonOCC algorithms.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from app.services import cad_cache
from app.utils.file_handler import ensure_step_filename, read_upload_file, require_extension
from app.utils.occ_guard import occ_installed

router = APIRouter(prefix="/cad", tags=["cad"])


@router.get("/status")
def cad_status() -> dict:
    return {
        "pythonocc_available": occ_installed(),
        "api_version": "1.1",
        "endpoints": [
            "POST /api/v1/cad/upload",
            "POST /api/v1/cad/upload/binary",
        ],
    }


def _persist_step_upload(raw: bytes, filename: str) -> dict:
    """Validate filename and write STEP bytes into the cache; return metadata."""
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    name = ensure_step_filename(filename)
    require_extension(name.lower(), (".stp", ".step"))
    cad_cache.prune_expired()
    return cad_cache.store_step(raw, name)


@router.post("/upload")
async def cad_upload(
    file: UploadFile = File(..., description="STEP/STP part file; cached for reuse via model_id"),
) -> JSONResponse:
    """Upload a STEP file once and receive a ``model_id`` for later reuse."""
    raw, name = await read_upload_file(file)
    meta = _persist_step_upload(raw, name)
    return JSONResponse(content=meta)


@router.post("/upload/binary")
async def cad_upload_binary(
    request: Request,
    filename: str | None = Query(
        None,
        description="Filename, ideally ending in .stp/.step; X-Filename header also accepted.",
    ),
    x_filename: str | None = Header(default=None, alias="X-Filename"),
) -> JSONResponse:
    """Upload raw STEP bytes (Content-Type: application/octet-stream)."""
    raw = await request.body()
    hint = filename or x_filename or "model.stp"
    meta = _persist_step_upload(raw, hint)
    return JSONResponse(content=meta)
