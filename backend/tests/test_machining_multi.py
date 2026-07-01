"""Mock-data test for the multi-hole CAM path generation pipeline.

This script does NOT require pythonOCC — it builds a synthetic
``feature_result`` dict (matching the schema returned by
``feature_service.analyze_face_spread``) and exercises
``machining_service.generate_machining_paths_multi`` end-to-end.

Run:
    cd E:/SsRobotOCC/backend
    python -m tests.test_machining_multi   (or python tests/test_machining_multi.py)
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

# Make `app.*` importable when run from anywhere
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# ── Synthetic feature_result factory ────────────────────────────────────────

def _circle_polyline(cx: float, cy: float, r: float, n: int = 32) -> list[list[float]]:
    """Generate a closed circle polyline (CCW) on the XY plane, Z=0."""
    pts: list[list[float]] = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append([cx + r * math.cos(a), cy + r * math.sin(a), 0.0])
    return pts


def _slot_polyline(cx: float, cy: float, length: float, width: float, n: int = 32) -> list[list[float]]:
    """Generate a rounded-end slot polyline on the XY plane, Z=0."""
    pts: list[list[float]] = []
    half = length / 2
    rw = width / 2
    for i in range(n // 2):
        a = math.pi * i / (n // 2 - 1) + math.pi / 2
        pts.append([cx - half + rw * math.cos(a), cy + rw * math.sin(a), 0.0])
    for i in range(n // 2):
        a = math.pi * i / (n // 2 - 1) - math.pi / 2
        pts.append([cx + half + rw * math.cos(a), cy + rw * math.sin(a), 0.0])
    return pts


def build_feature_result() -> dict:
    """Build a fake analyze_face_spread result with 3 holes + 1 outer contour."""
    # Three holes of different shapes
    h1_pts = _circle_polyline(cx=10.0, cy=20.0, r=5.0)
    h2_pts = _circle_polyline(cx=50.0, cy=20.0, r=8.0)
    h3_pts = _slot_polyline (cx=90.0, cy=20.0, length=20.0, width=10.0)

    # Outer boundary of the part (large rectangle)
    outer_pts: list[list[float]] = [
        [0.0,  0.0, 0.0],
        [120.0, 0.0, 0.0],
        [120.0, 40.0, 0.0],
        [0.0,   40.0, 0.0],
    ]

    polylines = [
        {"id": "poly_outer",  "closed": True, "points": outer_pts},
        {"id": "poly_hole_1", "closed": True, "points": h1_pts},
        {"id": "poly_hole_2", "closed": True, "points": h2_pts},
        {"id": "poly_hole_3", "closed": True, "points": h3_pts},
    ]

    return {
        "schema_version": "2.0",
        "model_id": "test_model_001",
        "target_face_id": "face_0",
        "polylines": polylines,
        # Each wire has: { id, face_id, is_outer, polyline_id, contour_id, contour_type }
        "wires": [
            {"id": "wire_outer", "face_id": "face_0", "is_outer": True,
             "polyline_id": "poly_outer", "contour_id": "contour_0",
             "contour_type": "outer"},
            {"id": "wire_1", "face_id": "face_0", "is_outer": False,
             "polyline_id": "poly_hole_1", "contour_id": "contour_1",
             "contour_type": "circle"},
            {"id": "wire_2", "face_id": "face_0", "is_outer": False,
             "polyline_id": "poly_hole_2", "contour_id": "contour_2",
             "contour_type": "circle"},
            {"id": "wire_3", "face_id": "face_0", "is_outer": False,
             "polyline_id": "poly_hole_3", "contour_id": "contour_3",
             "contour_type": "slot"},
        ],
        "contours": [
            {
                "id": "contour_0",
                "contour_type": "outer",
                "center": {"root": [60.0, 20.0, 0.0]},
                "normal": {"root": [0.0, 0.0, 1.0]},
                "polyline_id": "poly_outer",
                "wire_id":   "wire_outer",
                "face_id":   "face_0",
                "is_outer":  True,
                "parameters": {},
                "area": 4800.0,
                "perimeter": 320.0,
            },
        ],
        "holes": [
            {
                "id": "hole_circle_A",     # ← user clicks this 1st
                "kind": "circle",
                "contour_type": "circle",
                "center": {"root": [10.0, 20.0, 0.0]},
                "axis":   {"root": [0.0, 0.0, 1.0]},
                "diameter": 10.0,
                "depth": 1.0,
                "face_id": "face_0",
                "wire_id": "wire_1",
                "cylindrical_face_ids": [],
                "parameters": {"diameter": 10.0},
            },
            {
                "id": "hole_circle_B",     # ← user clicks this 2nd
                "kind": "circle",
                "contour_type": "circle",
                "center": {"root": [50.0, 20.0, 0.0]},
                "axis":   {"root": [0.0, 0.0, 1.0]},
                "diameter": 16.0,
                "depth": 1.0,
                "face_id": "face_0",
                "wire_id": "wire_2",
                "cylindrical_face_ids": [],
                "parameters": {"diameter": 16.0},
            },
            {
                "id": "hole_slot_C",       # ← user clicks this 3rd
                "kind": "slot",
                "contour_type": "slot",
                "center": {"root": [90.0, 20.0, 0.0]},
                "axis":   {"root": [0.0, 0.0, 1.0]},
                "diameter": None,
                "depth": 1.0,
                "face_id": "face_0",
                "wire_id": "wire_3",
                "cylindrical_face_ids": [],
                "parameters": {"length": 20.0, "width": 10.0},
            },
        ],
    }


# ── Tests ────────────────────────────────────────────────────────────────────

def test_click_order_preserved():
    """inner_paths list order must match the input hole_ids order."""
    from app.services.machining_service import generate_machining_paths_multi

    feature_result = build_feature_result()
    # User clicked: slot first, then circle B, then circle A
    click_order = ["hole_slot_C", "hole_circle_B", "hole_circle_A"]

    result = generate_machining_paths_multi(
        feature_result=feature_result,
        hole_ids=click_order,
    )

    group = result.machining_groups[0]
    inner_ids = [p.source_hole_id for p in group.inner_paths]
    assert inner_ids == click_order, (
        f"Click order lost! got={inner_ids} expected={click_order}"
    )

    # Each path's order_index must also be 0/1/2 in sequence
    indices = [p.order_index for p in group.inner_paths]
    assert indices == [0, 1, 2], f"order_index wrong: {indices}"

    # group.path_order must mirror the inner_paths order
    assert group.path_order == [p.id for p in group.inner_paths], (
        f"path_order mismatch: {group.path_order}"
    )

    print("[PASS] test_click_order_preserved")
    print(f"       inner_ids = {inner_ids}")
    print(f"       path_order = {group.path_order}")


def test_idle_lines_between_holes():
    """For N holes there must be N-1 cross-path idle lines."""
    from app.services.machining_service import generate_machining_paths_multi

    feature_result = build_feature_result()
    click_order = ["hole_circle_A", "hole_circle_B", "hole_slot_C"]

    result = generate_machining_paths_multi(
        feature_result=feature_result,
        hole_ids=click_order,
    )
    group = result.machining_groups[0]
    assert len(group.transition_lines) == 2, (
        f"Expected 2 transition lines, got {len(group.transition_lines)}"
    )
    # Idle lines must all be of line_type='idle' and have power=0
    for line in group.transition_lines:
        assert line.line_type == "idle", f"Wrong line_type: {line.line_type}"
        assert line.power == 0, f"Idle should have power=0, got {line.power}"
        assert line.duty == 0, f"Idle should have duty=0, got {line.duty}"
    print("[PASS] test_idle_lines_between_holes")


def test_unknown_hole_id_raises():
    """Passing an unknown id should raise ValueError when *none* match."""
    from app.services.machining_service import generate_machining_paths_multi

    feature_result = build_feature_result()
    try:
        generate_machining_paths_multi(
            feature_result=feature_result,
            hole_ids=["nonexistent_1", "nonexistent_2"],
        )
    except ValueError as exc:
        assert "None of the requested hole_ids" in str(exc)
        print(f"[PASS] test_unknown_hole_id_raises (msg: {exc})")
        return
    raise AssertionError("Expected ValueError for unknown hole_ids")


def test_partial_unknown_silently_skipped():
    """If some hole_ids are valid and some are not, the valid ones still process."""
    from app.services.machining_service import generate_machining_paths_multi

    feature_result = build_feature_result()
    click_order = ["hole_circle_A", "ghost", "hole_circle_B", "phantom"]
    result = generate_machining_paths_multi(
        feature_result=feature_result,
        hole_ids=click_order,
    )
    inner_ids = [p.source_hole_id for p in result.machining_groups[0].inner_paths]
    assert inner_ids == ["hole_circle_A", "hole_circle_B"], inner_ids
    print(f"[PASS] test_partial_unknown_silently_skipped -> {inner_ids}")


def test_segment_velocity_uses_table():
    """CAMLines with out_type='long_line' should have higher velocity than 'three_d_corner'."""
    from app.services.machining_service import generate_machining_paths_multi, _SEGMENT_VELOCITY

    feature_result = build_feature_result()
    result = generate_machining_paths_multi(
        feature_result=feature_result,
        hole_ids=["hole_circle_A"],
    )
    cam = result.machining_groups[0].inner_paths[0].cam_lines
    by_type: dict[str, list[float]] = {}
    for line in cam:
        by_type.setdefault(line.out_type or "unknown", []).append(line.velocity)
    if "long_line" in by_type and "shortest_line" in by_type:
        long_v   = by_type["long_line"][0]
        short_v  = by_type["shortest_line"][0]
        assert long_v > short_v, (
            f"long_line ({long_v}) should be faster than shortest_line ({short_v})"
        )
        print(f"[PASS] test_segment_velocity_uses_table  "
              f"long_line={long_v:.1f} > shortest_line={short_v:.1f}")
    else:
        print(f"[SKIP] test_segment_velocity_uses_table (no contrasting segments in circle)")


def test_segment_velocity_with_sharp_corner():
    """A polyline with a sharp corner must produce a 'three_d_corner' line
    with reduced velocity (<= 0.5× base)."""
    from app.services.machining_service import generate_machining_paths_multi

    # Hand-crafted polyline forming a sharp corner at point #2
    # Long straight line, then a sharp 90° turn, then another long line.
    # Using many points on each straight segment keeps the inscribed-angle
    # of the straight segments at 0°, leaving only the corner classified.
    pts: list[list[float]] = []
    # Segment 1: 0 → 5 along Y (11 points)
    for i in range(11):
        pts.append([0.0, float(i) * 0.5, 0.0])
    # Segment 2: 5 → 5 along X (corner at index 10)
    for i in range(1, 11):
        pts.append([float(i) * 1.0, 5.0, 0.0])
    feature_result = {
        "schema_version": "2.0",
        "model_id": "test_corner",
        "target_face_id": "face_0",
        "polylines": [
            {"id": "poly_h1", "closed": False, "points": pts},
        ],
        "wires": [
            {"id": "wire_1", "face_id": "face_0", "is_outer": False,
             "polyline_id": "poly_h1", "contour_id": "contour_1",
             "contour_type": "rectangle"},
        ],
        "contours": [],
        "holes": [
            {
                "id": "hole_path_1",
                "kind": "rectangle",
                "contour_type": "rectangle",
                "center": {"root": [5.0, 2.5, 0.0]},
                "axis":   {"root": [0.0, 0.0, 1.0]},
                "diameter": None,
                "depth": 1.0,
                "face_id": "face_0",
                "wire_id": "wire_1",
                "cylindrical_face_ids": [],
                "parameters": {},
            },
        ],
    }
    result = generate_machining_paths_multi(
        feature_result=feature_result, hole_ids=["hole_path_1"]
    )
    cam = result.machining_groups[0].inner_paths[0].cam_lines
    corner_lines = [ln for ln in cam if ln.out_type == "three_d_corner"]
    assert corner_lines, (
        f"Expected at least one three_d_corner segment in sharp-turn polyline, "
        f"got types: {[ln.out_type for ln in cam]}"
    )
    for ln in corner_lines:
        assert ln.velocity <= 100.0 * 0.5, (
            f"three_d_corner velocity should be <= 50, got {ln.velocity}"
        )
    print(f"[PASS] test_segment_velocity_with_sharp_corner  "
          f"corner_vel={corner_lines[0].velocity:.1f} mm/s")


def test_legacy_entry_point_still_works():
    """generate_machining_paths (no hole_ids) must still process every hole."""
    from app.services.machining_service import generate_machining_paths

    feature_result = build_feature_result()
    result = generate_machining_paths(feature_result=feature_result)
    group = result.machining_groups[0]
    assert len(group.inner_paths) == 3
    assert len(group.transition_lines) == 0   # legacy doesn't add idle lines
    print(f"[PASS] test_legacy_entry_point_still_works  ({len(group.inner_paths)} holes)")


def test_response_serializable_to_json():
    """Full round-trip: model_dump() → JSON → reload (lossy on Point3D but that's fine)."""
    from app.services.machining_service import generate_machining_paths_multi

    feature_result = build_feature_result()
    result = generate_machining_paths_multi(
        feature_result=feature_result,
        hole_ids=["hole_circle_B", "hole_circle_A", "hole_slot_C"],
        include_outer=True,
        idle_velocity=350.0,
    )
    dumped = result.model_dump()
    text = json.dumps(dumped, ensure_ascii=False, indent=2)
    parsed = json.loads(text)

    # ── Assertions on the JSON shape ──
    assert "machining_groups" in parsed
    g = parsed["machining_groups"][0]
    assert g["path_order"] == [p["id"] for p in g["inner_paths"] + g["outer_paths"]]
    assert g["name"] == "MultiHole[3]+Outer[1]"
    assert len(g["transition_lines"]) == 2
    # idle_velocity override
    for line in g["transition_lines"]:
        assert line["velocity"] == 350.0, f"idle_velocity override failed: {line['velocity']}"

    print(f"[PASS] test_response_serializable_to_json  "
          f"({len(text)} bytes, {g['total_path_count'] if 'total_path_count' in g else '?'} paths)")


def test_hole_resolves_to_correct_polyline():
    """Each hole's CAMLines must be sampled from *its own* polyline, not
    the outer boundary or another hole's polyline (regression test for
    the previous 'first polyline wins' bug)."""
    from app.services.machining_service import generate_machining_paths_multi

    feature_result = build_feature_result()
    result = generate_machining_paths_multi(
        feature_result=feature_result,
        hole_ids=["hole_circle_A", "hole_circle_B", "hole_slot_C"],
    )
    # hole_circle_A: center (10,20,0), radius 5 — cam_lines should be
    # all within roughly [5..15] in X and [15..25] in Y.
    paths = {p.source_hole_id: p for p in result.machining_groups[0].inner_paths}
    for hole_id, x_range, y_range in [
        ("hole_circle_A", (3, 17),  (13, 27)),
        ("hole_circle_B", (40, 60), (10, 30)),
        ("hole_slot_C",   (75, 105),(13, 27)),
    ]:
        cam = paths[hole_id].cam_lines
        assert cam, f"{hole_id} produced no cam_lines"
        for line in cam:
            x, y, _ = line.end_point.root
            assert x_range[0] <= x <= x_range[1], (
                f"{hole_id} has end_point.x={x} outside {x_range}"
            )
            assert y_range[0] <= y <= y_range[1], (
                f"{hole_id} has end_point.y={y} outside {y_range}"
            )
    print("[PASS] test_hole_resolves_to_correct_polyline  (no polyline bleed)")


def test_hole_from_raw_accepts_bare_list():
    """Regression test: feature_result from OCC module sends bare [x,y,z]
    lists for center/axis.  The _coerce_hole_dict wrapper must handle this
    without Pydantic ValidationError (the exact error the user hit)."""
    from app.services.machining_service import _coerce_hole_dict, _hole_from_raw

    raw_hole = {
        "id": "hole_1",
        "kind": "circle",
        "contour_type": "circle",
        # Bare list — this is what OCC serialisation produces
        "center": [1538.798, -929.361, -7.358],
        "axis":   [0.0, -1.0, 0.0],
        "diameter": 10.0,
        "depth": 1.0,
        "face_id": "face_0",
        "wire_id": "wire_1",
        "cylindrical_face_ids": [],
        "parameters": {"diameter": 10.0},
    }
    coerced = _coerce_hole_dict(raw_hole)
    # After coercion: center/axis must be Point3D / Vector3D objects
    from app.models.feature import Point3D, Vector3D
    assert isinstance(coerced["center"], Point3D), (
        f"center should be Point3D, got {type(coerced['center'])}"
    )
    assert isinstance(coerced["axis"], Vector3D), (
        f"axis should be Vector3D, got {type(coerced['axis'])}"
    )
    # HoleFeature(**coerced) should not raise
    hole = _hole_from_raw(raw_hole)
    assert hole.id == "hole_1"
    assert hole.contour_type == "circle"
    print("[PASS] test_hole_from_raw_accepts_bare_list")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_click_order_preserved,
        test_idle_lines_between_holes,
        test_unknown_hole_id_raises,
        test_partial_unknown_silently_skipped,
        test_segment_velocity_uses_table,
        test_segment_velocity_with_sharp_corner,
        test_hole_resolves_to_correct_polyline,
        test_hole_from_raw_accepts_bare_list,
        test_legacy_entry_point_still_works,
        test_response_serializable_to_json,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {t.__name__}")
            traceback.print_exc()
    print(f"\n=== {passed} passed, {failed} failed ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
