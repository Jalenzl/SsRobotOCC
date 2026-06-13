"""Unit tests for 2D contour classification."""

from __future__ import annotations

import math

import pytest

from app.occ.features.contour_classifier import classify_wire_contour


def _circle_pts_3d(r: float, n: int = 48, z: float = 0.0):
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        pts.append((r * math.cos(t), r * math.sin(t), z))
    pts.append(pts[0])
    return pts


def _rect_pts_3d(lx: float, ly: float, z: float = 0.0):
    x0, y0 = -lx / 2, -ly / 2
    return [
        (x0, y0, z),
        (x0 + lx, y0, z),
        (x0 + lx, y0 + ly, z),
        (x0, y0 + ly, z),
        (x0, y0, z),
    ]


def _rounded_rect_pts_3d(
    lx: float,
    ly: float,
    radius: float,
    *,
    z: float = 0.0,
    n_arc: int = 12,
):
    """圆角矩形（模拟 STEP 离散后的高圆度内环，易误判为圆）。"""
    r = min(radius, lx / 2 - 1e-6, ly / 2 - 1e-6)
    x0, y0 = -lx / 2, -ly / 2
    x1, y1 = lx / 2, ly / 2
    pts: list[tuple[float, float, float]] = []

    def arc(cx: float, cy: float, t0: float, t1: float) -> None:
        for i in range(n_arc + 1):
            t = t0 + (t1 - t0) * i / n_arc
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t), z))

    arc(x1 - r, y0 + r, -math.pi / 2, 0.0)
    arc(x1 - r, y1 - r, 0.0, math.pi / 2)
    arc(x0 + r, y1 - r, math.pi / 2, math.pi)
    arc(x0 + r, y0 + r, math.pi, 3 * math.pi / 2)
    pts.append(pts[0])
    return pts


def _slot_pts_3d(length: float, width: float, z: float = 0.0, n_arc: int = 16):
    half_len = length / 2
    half_w = width / 2
    radius = half_w
    straight_half = max(0.0, half_len - radius)
    pts = []

    for i in range(n_arc + 1):
        t = -math.pi / 2 + math.pi * i / n_arc
        pts.append((straight_half + radius * math.cos(t), radius * math.sin(t), z))

    for i in range(1, n_arc + 1):
        t = math.pi / 2 + math.pi * i / n_arc
        pts.append((-straight_half + radius * math.cos(t), radius * math.sin(t), z))

    pts.append(pts[0])
    return pts


def test_classify_circle():
    pts = _circle_pts_3d(15)
    c = classify_wire_contour(
        pts,
        face_normal=(0, 0, 1),
        is_outer=False,
        wire_id="w1",
        polyline_id="p1",
        face_id="f1",
        contour_index=0,
    )
    assert c["contour_type"] == "circle"
    assert c["parameters"]["diameter"] == pytest.approx(30, rel=0.15)


def test_classify_rounded_rectangle_not_circle():
    """圆角矩形圆度约 0.65–0.85，不应落入 circle 兜底。"""
    pts = _rounded_rect_pts_3d(55.0, 48.0, 8.0, n_arc=16)
    c = classify_wire_contour(
        pts,
        face_normal=(0, 0, 1),
        is_outer=False,
        wire_id="w1",
        polyline_id="p1",
        face_id="f1",
        contour_index=0,
    )
    assert c["contour_type"] == "rectangle", c
    assert c["parameters"]["diameter"] is None
    assert c["parameters"]["length"] == pytest.approx(55, rel=0.15)
    assert c["parameters"]["width"] == pytest.approx(48, rel=0.15)


def test_classify_rectangle():
    pts = _rect_pts_3d(30, 20)
    c = classify_wire_contour(
        pts,
        face_normal=(0, 0, 1),
        is_outer=False,
        wire_id="w1",
        polyline_id="p1",
        face_id="f1",
        contour_index=0,
    )
    assert c["contour_type"] == "rectangle"
    assert c["parameters"]["length"] == pytest.approx(30, rel=0.2)
    assert c["parameters"]["width"] == pytest.approx(20, rel=0.2)


def test_classify_slot():
    pts = _slot_pts_3d(50, 10)
    c = classify_wire_contour(
        pts,
        face_normal=(0, 0, 1),
        is_outer=False,
        wire_id="w1",
        polyline_id="p1",
        face_id="f1",
        contour_index=0,
    )
    assert c["contour_type"] == "slot"
    assert c["parameters"]["length"] >= c["parameters"]["width"]


def test_classify_outer():
    pts = _rect_pts_3d(100, 60)
    c = classify_wire_contour(
        pts,
        face_normal=(0, 0, 1),
        is_outer=True,
        wire_id="w0",
        polyline_id="p0",
        face_id="f0",
        contour_index=0,
    )
    assert c["contour_type"] == "outer"
    assert c["is_outer"] is True
