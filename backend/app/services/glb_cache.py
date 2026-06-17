"""Cache for generated GLBs so re-importing the same STEP (or re-converting
with the same deflection params) does not pay the OCC tessellation cost twice.

Key = ``{model_id}_{linear_deflection}_{angular_deflection}_{merge_faces}``.

Stored alongside the STEP cache in ``uploads/cad_cache/``. Entries are
invalidated automatically when the source STEP file's mtime changes (the
STEP cache writes a single ``{model_id}.json`` meta file, and we use the
matching ``.stp`` / ``.step`` mtime as the invalidation token).
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

from app.services import cad_cache
from app.config import BACKEND_ROOT

CACHE_DIR = Path(BACKEND_ROOT, "uploads", "cad_cache").resolve()
DEFAULT_TTL_SECONDS = 6 * 3600  # 6h: matches typical CAD session length

_KEY_RE = re.compile(r"^[0-9a-f]{8,128}$")


def _glb_key(
    model_id: str,
    *,
    linear_deflection: float,
    angular_deflection: float,
    merge_faces: bool,
    mode: str,
) -> str:
    """Stable GLB cache key independent of the model_id format."""
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(f"|ld={linear_deflection:.6f}".encode())
    h.update(f"|ad={angular_deflection:.6f}".encode())
    h.update(f"|merge={int(bool(merge_faces))}".encode())
    h.update(f"|mode={mode}".encode())
    return h.hexdigest()[:32]


def _glb_path(key: str) -> Path:
    if not _KEY_RE.match(key):
        raise ValueError(f"invalid GLB cache key: {key!r}")
    return CACHE_DIR / f"glb_{key}.glb"


def get_cached(
    model_id: str,
    *,
    linear_deflection: float,
    angular_deflection: float,
    merge_faces: bool,
    mode: str,
) -> bytes | None:
    """Return the cached GLB bytes if fresh, else ``None``."""
    try:
        step_path = cad_cache.get_step_path(model_id)
    except KeyError:
        return None
    if not step_path.is_file():
        return None

    key = _glb_key(
        model_id,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        merge_faces=merge_faces,
        mode=mode,
    )
    glb_path = _glb_path(key)
    if not glb_path.is_file():
        return None
    # Invalidate if STEP is newer than cached GLB (user re-uploaded a new file
    # with the same content hash — extremely rare but correct).
    if glb_path.stat().st_mtime < step_path.stat().st_mtime:
        return None
    return glb_path.read_bytes()


def put_cached(
    model_id: str,
    glb: bytes,
    *,
    linear_deflection: float,
    angular_deflection: float,
    merge_faces: bool,
    mode: str,
) -> None:
    """Store the generated GLB bytes; overwrites any prior entry."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _glb_key(
        model_id,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        merge_faces=merge_faces,
        mode=mode,
    )
    _glb_path(key).write_bytes(glb)


def prune_expired(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> int:
    """Remove GLB cache entries older than ``ttl_seconds``."""
    if not CACHE_DIR.is_dir():
        return 0
    cutoff = time.time() - ttl_seconds
    removed = 0
    for p in list(CACHE_DIR.glob("glb_*.glb")):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed
