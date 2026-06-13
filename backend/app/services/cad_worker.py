"""Isolated CAD feature-extraction worker (fresh process, only pythonOCC).

Runs in a clean subprocess so the parent (uvicorn) never loads OCCT DLLs in the
same process as trimesh / cascadio. That DLL clash crashes the whole process on
Windows with ``0xC06D007F: Procedure not found``. Running here also contains any
native OCC crash to this child process instead of taking down the server.

Protocol:
    python -m app.services.cad_worker <request.json> <response.json>

request.json:
    {"step_path": "...", "mode": "full"|"face", "face_id": "face_3"|null,
     "options": { ...CadAnalyzeOptions... }}

response.json:
    {"ok": true, "result": {...}}              # success
    {"ok": false, "error_type": "...", "error": "..."}   # handled failure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _run(req: dict) -> dict:
    from app.models.cad import CadAnalyzeOptions
    from app.occ.features.extractor import (
        extract_all_features,
        extract_face_features,
        extract_face_spread_features,
    )
    from app.occ.loader import read_step_file

    options = CadAnalyzeOptions(**(req.get("options") or {}))
    shape = read_step_file(req["step_path"])

    mode = req.get("mode", "full")
    if mode in ("face", "face_spread"):
        face_id = req.get("face_id")
        if not face_id:
            raise ValueError("face_id 不能为空")
        if mode == "face_spread":
            return extract_face_spread_features(shape, options, face_id=face_id)
        return extract_face_features(shape, options, face_id=face_id)
    return extract_all_features(shape, options)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print(
            "usage: python -m app.services.cad_worker <request.json> <response.json>",
            file=sys.stderr,
        )
        return 2

    req_path = Path(args[0])
    resp_path = Path(args[1])
    try:
        req = json.loads(req_path.read_text(encoding="utf-8"))
        out = {"ok": True, "result": _run(req)}
    except Exception as e:  # noqa: BLE001 - surface as structured error to parent
        out = {"ok": False, "error_type": type(e).__name__, "error": str(e)}
    resp_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
