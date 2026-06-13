"""
生成 CAD 特征识别测试用 STEP 文件。

用法（conda occ 环境）:
    cd backend
    python tests/fixtures/cad/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.IFSelect import IFSelect_RetDone


OUT_DIR = Path(__file__).resolve().parent


def _write_step(shape, path: Path) -> None:
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {path}")


def make_box_shape(w: float = 100.0, h: float = 60.0, d: float = 20.0):
    return BRepPrimAPI_MakeBox(w, h, d).Shape()


def make_plate_with_slot_shape(
    size: float = 100.0,
    thickness: float = 10.0,
    slot_len: float = 40.0,
    slot_w: float = 12.0,
):
    """顶面细长槽（矩形切口，分类为 slot / rectangle）。"""
    box = BRepPrimAPI_MakeBox(size, size, thickness).Shape()
    ax = gp_Ax2(gp_Pnt(size / 2, size / 2, 0), gp_Dir(0, 0, 1))
    cutter = BRepPrimAPI_MakeBox(ax, slot_len, slot_w, thickness).Shape()
    return BRepAlgoAPI_Cut(box, cutter).Shape()


def make_plate_with_rect_pocket_shape(
    size: float = 100.0,
    thickness: float = 10.0,
    rect_l: float = 30.0,
    rect_w: float = 20.0,
):
    box = BRepPrimAPI_MakeBox(size, size, thickness).Shape()
    ax = gp_Ax2(gp_Pnt(size / 2, size / 2, 0), gp_Dir(0, 0, 1))
    cutter = BRepPrimAPI_MakeBox(ax, rect_l, rect_w, thickness).Shape()
    return BRepAlgoAPI_Cut(box, cutter).Shape()


def make_plate_with_hole_shape(
    size: float = 100.0,
    thickness: float = 10.0,
    hole_r: float = 15.0,
):
    """通孔板：圆柱贯穿整块板（through hole）。"""
    box = BRepPrimAPI_MakeBox(size, size, thickness).Shape()
    ax = gp_Ax2(gp_Pnt(size / 2, size / 2, 0), gp_Dir(0, 0, 1))
    cyl = BRepPrimAPI_MakeCylinder(ax, hole_r, thickness).Shape()
    return BRepAlgoAPI_Cut(box, cyl).Shape()


def make_plate_with_blind_hole_shape(
    size: float = 100.0,
    thickness: float = 20.0,
    hole_r: float = 12.0,
    depth: float = 8.0,
):
    """盲孔板：从顶面向下钻 ``depth`` 深，不贯穿（blind hole）。

    顶面 z=thickness；圆柱从 z=thickness-depth 向上钻到顶面外，布尔减后
    形成一个深度为 ``depth`` 的盲孔（孔底为平面）。
    """
    box = BRepPrimAPI_MakeBox(size, size, thickness).Shape()
    ax = gp_Ax2(gp_Pnt(size / 2, size / 2, thickness - depth), gp_Dir(0, 0, 1))
    cyl = BRepPrimAPI_MakeCylinder(ax, hole_r, depth + 1.0).Shape()
    return BRepAlgoAPI_Cut(box, cyl).Shape()


def make_plate_with_boss_shape(
    size: float = 100.0,
    thickness: float = 10.0,
    boss_r: float = 15.0,
    boss_h: float = 12.0,
):
    """凸台板：底板顶面中央熔合一个圆柱凸台（protrusion / boss）。"""
    box = BRepPrimAPI_MakeBox(size, size, thickness).Shape()
    ax = gp_Ax2(gp_Pnt(size / 2, size / 2, thickness), gp_Dir(0, 0, 1))
    boss = BRepPrimAPI_MakeCylinder(ax, boss_r, boss_h).Shape()
    return BRepAlgoAPI_Fuse(box, boss).Shape()


def make_assembly_two_solids_shape(
    size: float = 60.0,
    thickness: float = 10.0,
    gap: float = 40.0,
    hole_r: float = 10.0,
):
    """双实体装配体：两块独立板（其一带通孔），合入一个 Compound。

    用于验证多实体（装配体）场景下：逐实体质心的内/外判定、面全局索引、
    跨实体的同侧扩散。
    """
    plate_a = make_plate_with_hole_shape(size, thickness, hole_r)

    plate_b = BRepPrimAPI_MakeBox(size, size, thickness).Shape()
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(size + gap, 0.0, 0.0))
    plate_b = BRepBuilderAPI_Transform(plate_b, trsf, True).Shape()

    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    builder.Add(comp, plate_a)
    builder.Add(comp, plate_b)
    return comp


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = {
        "box_100x60x20.step": make_box_shape(100, 60, 20),
        "plate_with_hole_100.step": make_plate_with_hole_shape(100, 10, 15),
        "plate_with_slot_100.step": make_plate_with_slot_shape(100, 10, 40, 12),
        "plate_with_rect_100.step": make_plate_with_rect_pocket_shape(100, 10, 30, 20),
        "plate_with_blind_hole.step": make_plate_with_blind_hole_shape(100, 20, 12, 8),
        "plate_with_boss.step": make_plate_with_boss_shape(100, 10, 15, 12),
        "assembly_two_solids.step": make_assembly_two_solids_shape(60, 10, 40, 10),
    }
    for name, shape in cases.items():
        path = OUT_DIR / name
        _write_step(shape, path)
        print(f"written {path}")


if __name__ == "__main__":
    main()
