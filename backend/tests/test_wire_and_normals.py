"""wire 有序离散与轮廓法向测试。"""

from __future__ import annotations

import math

import pytest

from app.occ import occ_available

pytestmark = pytest.mark.skipif(
    not occ_available(),
    reason="pythonOCC not installed (use conda env occ)",
)

from app.occ.discretize import wire_length, wire_to_polyline  # noqa: E402
from app.occ.geometry_utils import face_outward_normal, face_wires  # noqa: E402
from tests.fixtures.cad.generate_fixtures import (  # noqa: E402
    make_plate_with_hole_shape,
)


def _polyline_length(pts: list[tuple[float, float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        total += math.sqrt(sum((a[j] - b[j]) ** 2 for j in range(3)))
    return total


class TestWireDiscretize:
    def test_hole_wire_is_closed_and_near_full_perimeter(self):
        """内环圆孔 wire 应有序离散为闭合折线，周长接近 2πr。"""
        shape = make_plate_with_hole_shape(100, 10, 15)
        from app.occ.geometry_utils import iterate_faces

        top_face = None
        for face in iterate_faces(shape):
            wires = face_wires(face)
            if len(wires) >= 2:
                top_face = face
                break
        assert top_face is not None

        # 找内环（较短 wire）
        wires = sorted(face_wires(top_face), key=wire_length)
        inner_wire = wires[0]
        expected_len = wire_length(inner_wire)
        assert expected_len == pytest.approx(2 * math.pi * 15, rel=0.05)

        pts = wire_to_polyline(inner_wire, 0.1, 0.5)
        poly_len = _polyline_length(pts)
        assert poly_len == pytest.approx(expected_len, rel=0.08)
        assert len(pts) >= 12
        # 首尾闭合（wire_to_polyline 会重复首点）
        assert math.sqrt(sum((pts[0][j] - pts[-1][j]) ** 2 for j in range(3))) < 0.05


class TestContourOutwardNormal:
    def test_cylinder_face_outward_normal_is_radial(self):
        """圆柱面的外法向应为径向（水平分量），而非加工平面 (0,0,1)。"""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder

        from app.occ.geometry_utils import face_surface_info, iterate_faces

        cyl_shape = BRepPrimAPI_MakeCylinder(10.0, 20.0).Shape()
        found = False
        for face in iterate_faces(cyl_shape):
            if face_surface_info(face).get("surface_type") != "cylinder":
                continue
            n = face_outward_normal(face)
            assert n is not None
            horiz = math.sqrt(n[0] ** 2 + n[1] ** 2)
            assert horiz > 0.9, f"圆柱面外法向应近似径向，实际 {n}"
            assert abs(n[2]) < 0.2
            found = True
            break
        assert found

    def test_face_outward_normal_defined_for_all_box_faces(self):
        from app.occ.geometry_utils import iterate_faces
        from tests.fixtures.cad.generate_fixtures import make_box_shape

        for face in iterate_faces(make_box_shape(10, 10, 10)):
            outward = face_outward_normal(face)
            assert outward is not None
            L = math.sqrt(sum(c * c for c in outward))
            assert L == pytest.approx(1.0, abs=0.01)
