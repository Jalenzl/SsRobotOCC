"""CAD analyze & toolpath orchestration.

Feature extraction runs in an isolated subprocess (``app.services.cad_worker``)
so the main uvicorn process never imports OCCT DLLs alongside trimesh/cascadio
(that clash crashes the process with ``0xC06D007F`` on Windows). The subprocess
also contains any native OCC crash, returning a clean error instead of killing
the server. Path planning is pure-Python and stays in-process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from app.models.cad import (
    CadAnalyzeAndPathResponse,
    CadFaceAnalyzeResult,
    CadFaceSpreadResult,
    CadAnalyzeOptions,
    CadAnalyzeResult,
    PathPlanOptions,
    PathPlanResult,
)
from app.occ.path.planner import face_analyze_to_path_payload, generate_toolpath
from app.utils.occ_guard import occ_installed

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OCC_PYTHON = Path.home() / "miniconda3" / "envs" / "occ" / "python.exe"


class OccNotInstalledError(RuntimeError):
    pass


def require_occ() -> None:
    # Use find_spec (no import) so the main process never loads OCCT DLLs.
    if not occ_installed():
        raise OccNotInstalledError(
            "pythonOCC 未安装。请执行: conda install -c conda-forge pythonocc-core "
            "或 pip install pythonocc-core（Windows 建议 conda）"
        )


def _occ_python() -> str:
    """Interpreter that has pythonOCC. Prefer OCC_PYTHON env, then conda occ."""
    env_py = os.environ.get("OCC_PYTHON", "").strip()
    if env_py:
        return env_py
    if _DEFAULT_OCC_PYTHON.is_file():
        return str(_DEFAULT_OCC_PYTHON)
    return sys.executable


def _use_subprocess() -> bool:
    """Isolate OCC on Windows (DLL clash) or when explicitly forced."""
    if os.environ.get("CAD_FORCE_SUBPROCESS", "").strip() in ("1", "true", "yes"):
        return True
    if os.environ.get("CAD_FORCE_INPROCESS", "").strip() in ("1", "true", "yes"):
        return False
    return sys.platform == "win32"


def _analyze_via_subprocess(
    step_path: Path,
    *,
    mode: str,
    face_id: str | None,
    options: CadAnalyzeOptions,
) -> dict:
    with tempfile.TemporaryDirectory() as td:
        req_path = Path(td) / "request.json"
        resp_path = Path(td) / "response.json"
        req_path.write_text(
            json.dumps(
                {
                    "step_path": str(step_path),
                    "mode": mode,
                    "face_id": face_id,
                    "options": options.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = str(_BACKEND_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.run(
            [
                _occ_python(),
                "-m",
                "app.services.cad_worker",
                str(req_path),
                str(resp_path),
            ],
            cwd=str(_BACKEND_ROOT),
            capture_output=True,
            env=env,
            timeout=6000,
            creationflags=creationflags,
        )
        if not resp_path.is_file():
            err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
            # No response file → likely a native crash (the very thing we isolate).
            raise RuntimeError(
                err or f"CAD 分析子进程异常退出 (code {proc.returncode})，可能是 OCC 原生崩溃"
            )
        payload = json.loads(resp_path.read_text(encoding="utf-8"))

    if not payload.get("ok"):
        msg = payload.get("error") or "CAD 分析失败"
        if payload.get("error_type") == "ValueError":
            raise ValueError(msg)
        raise RuntimeError(msg)
    return payload["result"]


def _analyze_inprocess(
    step_path: Path,
    *,
    mode: str,
    face_id: str | None,
    options: CadAnalyzeOptions,
) -> dict:
    from app.occ.features.extractor import (
        extract_all_features,
        extract_face_features,
        extract_face_spread_features,
    )
    from app.occ.loader import read_step_file

    shape = read_step_file(step_path)
    if mode in ("face", "face_spread"):
        if not face_id:
            raise ValueError("face_id 不能为空")
        if mode == "face_spread":
            return extract_face_spread_features(shape, options, face_id=face_id)
        return extract_face_features(shape, options, face_id=face_id)
    return extract_all_features(shape, options)


def _analyze_path(
    step_path: Path,
    *,
    mode: str,
    face_id: str | None,
    options: CadAnalyzeOptions,
) -> dict:
    if _use_subprocess():
        return _analyze_via_subprocess(step_path, mode=mode, face_id=face_id, options=options)
    return _analyze_inprocess(step_path, mode=mode, face_id=face_id, options=options)


def _materialize_step(data: bytes, filename: str) -> Path:
    suffix = ".step" if (filename or "").lower().endswith(".step") else ".stp"
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(name)
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- #
# Path-based analyze (no re-upload: caller passes a cached STEP file path)
# --------------------------------------------------------------------------- #


def analyze_step_path(step_path: str | Path, options: CadAnalyzeOptions | None = None) -> CadAnalyzeResult:
    require_occ()
    opts = options or CadAnalyzeOptions()
    raw = _analyze_path(Path(step_path), mode="full", face_id=None, options=opts)
    return CadAnalyzeResult(**raw)


def analyze_step_face_path(
    step_path: str | Path,
    face_id: str,
    options: CadAnalyzeOptions | None = None,
) -> CadFaceAnalyzeResult:
    require_occ()
    opts = options or CadAnalyzeOptions()
    raw = _analyze_path(Path(step_path), mode="face", face_id=face_id, options=opts)
    return CadFaceAnalyzeResult(**raw)


def analyze_step_face_spread_path(
    step_path: str | Path,
    face_id: str,
    options: CadAnalyzeOptions | None = None,
) -> CadFaceSpreadResult:
    """选面 → 内/外表面扩散分析（缓存路径版，无需重传文件）。"""
    require_occ()
    opts = options or CadAnalyzeOptions()
    raw = _analyze_path(Path(step_path), mode="face_spread", face_id=face_id, options=opts)
    return CadFaceSpreadResult(**raw)


# --------------------------------------------------------------------------- #
# Bytes-based analyze (compatibility: caller uploads the STEP file each call)
# --------------------------------------------------------------------------- #


def analyze_step(data: bytes, filename: str, options: CadAnalyzeOptions | None = None) -> CadAnalyzeResult:
    require_occ()
    path = _materialize_step(data, filename)
    try:
        return analyze_step_path(path, options)
    finally:
        path.unlink(missing_ok=True)


def analyze_step_face(
    data: bytes,
    filename: str,
    face_id: str,
    options: CadAnalyzeOptions | None = None,
) -> CadFaceAnalyzeResult:
    require_occ()
    path = _materialize_step(data, filename)
    try:
        return analyze_step_face_path(path, face_id=face_id, options=options)
    finally:
        path.unlink(missing_ok=True)


def analyze_step_face_spread(
    data: bytes,
    filename: str,
    face_id: str,
    options: CadAnalyzeOptions | None = None,
) -> CadFaceSpreadResult:
    """选面 → 内/外表面扩散分析（字节流版，每次上传文件）。"""
    require_occ()
    path = _materialize_step(data, filename)
    try:
        return analyze_step_face_spread_path(path, face_id=face_id, options=options)
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Path planning (pure-Python, in-process)
# --------------------------------------------------------------------------- #


def plan_path_from_analyze(analyze: CadAnalyzeResult, options: PathPlanOptions | None = None) -> PathPlanResult:
    opts = options or PathPlanOptions()
    raw = generate_toolpath(analyze.model_dump(), opts)
    return PathPlanResult(**raw)


def plan_path_from_face_analyze(
    analyze: CadFaceAnalyzeResult,
    options: PathPlanOptions | None = None,
) -> PathPlanResult:
    opts = options or PathPlanOptions()
    raw = generate_toolpath(face_analyze_to_path_payload(analyze.model_dump()), opts)
    return PathPlanResult(**raw)


# --------------------------------------------------------------------------- #
# Combined analyze + plan
# --------------------------------------------------------------------------- #


def analyze_and_plan(
    data: bytes,
    filename: str,
    analyze_options: CadAnalyzeOptions | None = None,
    path_options: PathPlanOptions | None = None,
) -> CadAnalyzeAndPathResponse:
    analyze = analyze_step(data, filename, analyze_options)
    path = plan_path_from_analyze(analyze, path_options)
    return CadAnalyzeAndPathResponse(analyze=analyze, path=path)


def analyze_face_and_plan(
    data: bytes,
    filename: str,
    face_id: str,
    analyze_options: CadAnalyzeOptions | None = None,
    path_options: PathPlanOptions | None = None,
) -> tuple[CadFaceAnalyzeResult, PathPlanResult]:
    analyze = analyze_step_face(data, filename, face_id=face_id, options=analyze_options)
    path = plan_path_from_face_analyze(analyze, path_options)
    return analyze, path


def analyze_and_plan_path(
    step_path: str | Path,
    analyze_options: CadAnalyzeOptions | None = None,
    path_options: PathPlanOptions | None = None,
) -> CadAnalyzeAndPathResponse:
    analyze = analyze_step_path(step_path, analyze_options)
    path = plan_path_from_analyze(analyze, path_options)
    return CadAnalyzeAndPathResponse(analyze=analyze, path=path)


def analyze_face_and_plan_path(
    step_path: str | Path,
    face_id: str,
    analyze_options: CadAnalyzeOptions | None = None,
    path_options: PathPlanOptions | None = None,
) -> tuple[CadFaceAnalyzeResult, PathPlanResult]:
    analyze = analyze_step_face_path(step_path, face_id=face_id, options=analyze_options)
    path = plan_path_from_face_analyze(analyze, path_options)
    return analyze, path
