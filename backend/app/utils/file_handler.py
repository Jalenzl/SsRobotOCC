"""Upload helpers: read body, normalize extensions."""

from __future__ import annotations

import re
from urllib.parse import quote

from fastapi import HTTPException, UploadFile


async def read_upload_file(file: UploadFile) -> tuple[bytes, str]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    name = (file.filename or "").strip()
    return raw, name


def ensure_step_filename(filename: str, *, default_stem: str = "model") -> str:
    """Normalize upload filename for STEP cache (浏览器 Blob / 无扩展名文件常见).

    - ``part.stp`` / ``part.step`` → 原样保留
    - 空文件名 → ``model.stp``
    - 无扩展名 ``assembly`` → ``assembly.stp``
    """
    name = (filename or "").strip()
    lower = name.lower()
    if lower.endswith(".stp") or lower.endswith(".step"):
        return name
    if not name:
        return f"{default_stem}.stp"
    if "." in name:
        return name
    return f"{name}.stp"


def require_extension(filename: str, allowed: tuple[str, ...]) -> None:
    lower = filename.lower()
    if not any(lower.endswith(ext) for ext in allowed):
        raise HTTPException(
            status_code=400,
            detail=f"expected one of {allowed}, got: {filename!r}",
        )


def glb_output_name(filename: str, *, fallback: str = "model.glb") -> str:
    """Derive ASCII-safe GLB download name from uploaded STEP filename."""
    name = (filename or "").strip()
    if not name:
        return fallback
    stem = name.rsplit(".", 1)[0].strip() if "." in name else name
    if not stem:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9.\-_]", "_", stem).strip("._") or "model"
    return f"{safe[:180]}.glb"


def content_disposition_attachment(filename: str, *, fallback: str = "model.glb") -> str:
    """
    RFC 5987 Content-Disposition safe for non-ASCII filenames.
    Starlette encodes headers as latin-1; bare Chinese in filename= causes 500.
    """
    display = (filename or "").strip() or fallback
    ascii_name = glb_output_name(display, fallback=fallback)
    stem = display.rsplit(".", 1)[0].strip() if "." in display else display
    utf8_name = f"{stem}.glb" if stem else fallback
    encoded = quote(utf8_name, safe="")
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded}'
