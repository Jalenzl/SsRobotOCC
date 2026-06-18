"""Load STEP/STP into TopoDS_Shape via pythonOCC."""

from __future__ import annotations

import tempfile
from pathlib import Path

from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods


def read_step_bytes(data: bytes, filename_hint: str = "model.stp") -> "TopoDS_Shape":
    suffix = ".step" if filename_hint.lower().endswith(".step") else ".stp"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return read_step_file(path)
    finally:
        Path(path).unlink(missing_ok=True)


def read_step_file(path: str | Path) -> "TopoDS_Shape":
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise ValueError(f"STEP read failed: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError("STEP produced null shape")
    return _normalize_shape(shape)


def write_step_bytes(shape: "TopoDS_Shape", filename_hint: str = "model.step") -> bytes:
    """Serialize a TopoDS_Shape to STEP bytes. Inverse of ``read_step_bytes``.

    Used by tests that need to inject a custom shape (e.g. a multi-solid
    compound) into the upload pipeline.
    """
    from OCC.Core.STEPControl import STEPControl_Writer
    from OCC.Core.IFSelect import IFSelect_RetDone

    suffix = ".step" if filename_hint.lower().endswith(".step") else ".stp"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = tmp.name
    try:
        writer = STEPControl_Writer()
        writer.Transfer(shape, 0)
        status = writer.Write(path)
        if status != IFSelect_RetDone:
            raise ValueError(f"STEP write failed: {path}")
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


def _count_subshapes(shape: "TopoDS_Shape", kind) -> int:
    n = 0
    exp = TopExp_Explorer(shape, kind)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _normalize_shape(shape: "TopoDS_Shape") -> "TopoDS_Shape":
    """单实体取 Solid；多零件装配保留完整 Compound（避免只网格化第一个零件）。"""
    solid_n = _count_subshapes(shape, TopAbs_SOLID)
    if solid_n > 1:
        return shape
    if solid_n == 1:
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        exp.More()
        return topods.Solid(exp.Current())
    shell_n = _count_subshapes(shape, TopAbs_SHELL)
    if shell_n > 1:
        return shape
    if shell_n == 1:
        exp = TopExp_Explorer(shape, TopAbs_SHELL)
        exp.More()
        return topods.Shell(exp.Current())
    return shape
