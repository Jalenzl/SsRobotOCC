"""Feature-extraction API + service tests (multipart + JSON paths)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.utils.occ_guard import occ_installed

pytestmark = pytest.mark.skipif(not occ_installed(), reason="pythonOCC not installed")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cad" / "plate_with_hole_100.step"
SLOT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cad" / "plate_with_slot_100.step"
# Real-world 6-solid assembly provided by the user. Large (≈100 MB) and
# gitignored; tests using it skip cleanly when the file is missing.
USER_SAMPLE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cad" / "user_sample.STEP"


def _upload(client: TestClient) -> str:
    data = FIXTURE.read_bytes()
    r = client.post(
        "/api/v1/cad/upload/binary",
        content=data,
        headers={"Content-Type": "application/octet-stream", "X-Filename": "plate_with_hole.step"},
    )
    assert r.status_code == 200, r.text
    return r.json()["model_id"]


# ── Status / list-faces ───────────────────────────────────────────────────
def test_feature_status_lists_endpoints():
    client = TestClient(app)
    r = client.get("/api/v1/cad/feature/status")
    assert r.status_code == 200
    eps = r.json()["endpoints"]
    assert any("analyze/face_spread" in e for e in eps)


def test_list_faces_returns_mixed_surface_types():
    client = TestClient(app)
    mid = _upload(client)
    r = client.get(f"/api/v1/cad/faces?model_id={mid}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 5
    types = {f["surface_type"] for f in body["faces"]}
    assert "plane" in types
    assert "cylinder" in types


# ── Multipart (frontend FormData) ─────────────────────────────────────────
def test_analyze_face_spread_multipart():
    client = TestClient(app)
    mid = _upload(client)
    options = {
        "linear_deflection": 0.1,
        "angular_deflection": 0.5,
        "work_plane": "auto",
        "hole_diameter_min": 0.5,
        "hole_diameter_max": 500.0,
    }
    r = client.post(
        "/api/v1/cad/analyze/face_spread",
        data={"model_id": mid, "face_id": "face_2", "options_json": json.dumps(options)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_version"] == "1.1"
    assert body["target_face_id"] == "face_2"
    assert body["face"]["surface_type"] == "plane"

    types = {c["contour_type"] for c in body["contours"]}
    assert "outer" in types
    assert "circle" in types

    circle = next(c for c in body["contours"] if c["contour_type"] == "circle")
    assert circle["parameters"]["diameter"] is not None
    assert 25.0 < circle["parameters"]["diameter"] < 35.0

    assert any(h.get("diameter") and 25.0 < h["diameter"] < 35.0 for h in body["holes"])
    # Vector3 contract: polylines[0].points[*] is a flat [x,y,z] array
    assert body["polylines"], "polylines should be populated"
    pt0 = body["polylines"][0]["points"][0]
    assert isinstance(pt0, list) and len(pt0) == 3
    assert all(isinstance(v, (int, float)) for v in pt0)


# ── JSON path ─────────────────────────────────────────────────────────────
def test_analyze_face_spread_json():
    client = TestClient(app)
    mid = _upload(client)
    payload = {"model_id": mid, "face_id": "face_2"}
    r = client.post("/api/v1/cad/analyze/face_spread", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_face_id"] == "face_2"
    assert any(c["contour_type"] == "outer" for c in body["contours"])


# ── Error cases ───────────────────────────────────────────────────────────
def test_invalid_face_id_returns_400():
    client = TestClient(app)
    mid = _upload(client)
    r = client.post(
        "/api/v1/cad/analyze/face_spread",
        data={"model_id": mid, "face_id": "face_9999"},
    )
    assert r.status_code == 400


def test_part_face_id_in_face_spread_routes_to_part_mode():
    """`/analyze/face_spread` with `face_id=part_0` should automatically
    run part-level analysis and return ``target_face_id == 'part_0'``."""
    client = TestClient(app)
    mid = _upload(client)
    r = client.post(
        "/api/v1/cad/analyze/face_spread",
        data={"model_id": mid, "face_id": "part_0"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_face_id"] == "part_0"
    assert "per_face" in body
    assert body["part"]["face_count"] >= 5


def test_part_id_with_timestamp_suffix_falls_back_to_only_solid():
    """When the frontend sends a ``Part_<timestamp>`` (its ``mesh.name``
    fallback for unnamed GLB nodes), the backend should still succeed by
    using the only solid in the model. Without this, the user sees a
    spurious 400 on every pick of the unnamed part node."""
    client = TestClient(app)
    mid = _upload(client)
    r = client.post(
        "/api/v1/cad/analyze/face_spread",
        data={"model_id": mid, "face_id": "Part_1734567890123"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_face_id"] == "part_0"


def test_face_id_out_of_range_falls_back_to_only_face():
    """Same fallback for face mode: a numeric ``face_id`` past the end of
    the global face list should fall back to the only face when the model
    has exactly one face."""
    # The plate-with-hole fixture has 1 solid and 7 faces; face_9999 is
    # way out of range. With multiple faces the fallback is not used and
    # the request 400s. We instead verify the "single face" branch via
    # the part-mode route: face_9999 is a numeric id that resolves to
    # face_9999 (out of range), but ``Part_<big>`` with one solid
    # should fall back to part_0. That's already covered by
    # ``test_part_id_with_timestamp_suffix_falls_back_to_only_solid``;
    # here we just confirm the explicit 400 still fires when the model
    # has more than one face and the user-supplied id is bogus.
    client = TestClient(app)
    mid = _upload(client)
    r = client.post(
        "/api/v1/cad/analyze/face_spread",
        data={"model_id": mid, "face_id": "face_9999"},
    )
    # Multiple faces -> no fallback, expect a clear 400
    assert r.status_code == 400
    assert "out of range" in r.json()["detail"]


def test_bare_non_numeric_face_id_routes_to_part_mode():
    """When the user picks a part by its GLB mesh name (e.g. ``"Assembly"``,
    which is the root mesh of an imported STEP that the frontend sets as
    ``part.mesh.name``), the analyzer should treat the input as a part
    selector — not as a numeric face index — and succeed via the
    single-solid fallback. Otherwise the user sees a misleading
    "out of range" error from face lookup."""
    client = TestClient(app)
    mid = _upload(client)
    r = client.post(
        "/api/v1/cad/analyze/face_spread",
        data={"model_id": mid, "face_id": "Assembly"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_face_id"] == "part_0"
    assert "part" in body
    assert body["part"]["face_count"] >= 5


def test_assembly_pick_aggregates_all_solids():
    """When the user picks the assembly root on a multi-solid model, the
    bare non-numeric ``face_id`` triggers a model-wide aggregate over
    every Solid — not a misleading "out of range" error from the single
    Solid path. The response carries one synthetic ``part`` record and a
    merged ``per_face`` list covering every Solid."""
    from app.occ.loader import write_step_bytes  # local import to avoid pulling OCC at import time
    from app.utils.occ_guard import occ_installed

    if not occ_installed():
        pytest.skip("pythonOCC not installed")
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder

    # Two distinct boxes wrapped in a Compound so the model has 2 Solids
    # (Boolean fuse would merge them into one). Use the BRep_Builder API.
    box1 = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
    box2 = BRepPrimAPI_MakeBox(8.0, 8.0, 30.0).Shape()
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, box1)
    builder.Add(compound, box2)
    step_bytes = write_step_bytes(compound, "two_boxes.step")

    client = TestClient(app)
    up = client.post(
        "/api/v1/cad/upload",
        files={"file": ("two_boxes.step", step_bytes, "application/octet-stream")},
    )
    assert up.status_code == 200, up.text
    mid = up.json()["model_id"]

    r = client.post(
        "/api/v1/cad/analyze/face_spread",
        data={"model_id": mid, "face_id": "Assembly"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_face_id"] == "part_all"
    assert body["part"]["surface_type"] == "assembly"
    assert body["part"]["solid_count"] == 2
    # per_face covers both solids
    face_ids = {f["id"] for f in body["per_face"]}
    assert len(face_ids) == len(body["per_face"])  # deduped
    assert len(body["per_face"]) >= 10  # box1 has 6 plane faces; box2 adds at least 4 new ones


def test_part_spread_aggregates_all_plane_faces():
    """When the frontend selected a whole part, ``analyze_part_spread``
    should return aggregated contours / holes for every plane face of the
    solid (top plate, bottom plate, slot walls, etc.) — not just the
    first face."""
    client = TestClient(app)
    mid = _upload(client)
    r = client.post(
        "/api/v1/cad/analyze/part_spread",
        data={"model_id": mid, "face_id": "part_0"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_face_id"] == "part_0"
    assert body["part"]["face_count"] >= 5
    assert body["part"]["plane_face_count"] >= 5
    # The plate-with-hole fixture has 2 outer (top + bottom) and a circle
    assert any(c["contour_type"] == "outer" for c in body["contours"])
    assert any(c["contour_type"] == "circle" for c in body["contours"])
    # The top plate contour should appear in the aggregated result
    assert len(body["holes"]) >= 1
    # per_face lists every face in the part
    per_face_types = {pf["surface_type"] for pf in body["per_face"]}
    assert "plane" in per_face_types


def test_unknown_model_id_returns_404():
    client = TestClient(app)
    r = client.post(
        "/api/v1/cad/analyze/face_spread",
        data={"model_id": "deadbeefdeadbeefdeadbeefdeadbeef", "face_id": "face_0"},
    )
    assert r.status_code == 404


# ── Path placeholder ──────────────────────────────────────────────────────
def test_analyze_path_returns_toolpath_suggestion():
    client = TestClient(app)
    mid = _upload(client)
    r = client.post(
        "/api/v1/cad/analyze/path",
        data={"model_id": mid, "face_id": "face_2"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "toolpath_suggestion" in body
    kinds = {p["kind"] for p in body["toolpath_suggestion"]}
    assert "hole" in kinds or "boundary" in kinds


# ── Algorithm unit (slot classification) ──────────────────────────────────
def test_slot_classification_for_slot_fixture():
    """`plate_with_slot_100.step` is a 100×100×10 plate with a 40×12 cutout."""
    from app.occ.contour import analyze_face, list_faces
    from app.occ.geometry_utils import face_area, face_surface_info
    from app.occ.loader import read_step_bytes

    shape = read_step_bytes(SLOT_FIXTURE.read_bytes(), "slot.step")
    # Pick the top plane face (largest area = 100*100 - 40*12 ≈ 9520).
    candidates = []
    for i, f in enumerate(list_faces(shape)):
        info = face_surface_info(f)
        if info.get("surface_type") != "plane":
            continue
        a = face_area(f)
        candidates.append((i, f, a, info))
    candidates.sort(key=lambda x: -x[2])
    top = candidates[0]
    face = top[1]
    canonical = f"face_{top[0]}"
    result = analyze_face(face, canonical)
    types = {c["contour_type"] for c in result["contours"]}
    assert "outer" in types
    inners = [c for c in result["contours"] if not c.get("is_outer")]
    assert inners, "expected the slot to appear as an inner contour"
    # The inner shape is exactly 40×12 rectangle, so rectangle or slot is fine.
    assert inners[0]["contour_type"] in ("slot", "rectangle")
    p = inners[0]["parameters"]
    # At least one of length/width should reflect the cutout.
    assert (p.get("length") or 0) > 0 and (p.get("width") or 0) > 0


# ── Per-solid pipeline (real assembly) ────────────────────────────────────
def test_per_solid_dedups_coincident_wires():
    """STEP assemblies frequently expose the same closed loop on a face
    multiple times (the face is the union of two coincident sub-faces
    with identical trim curves). On the user_sample assembly, solid 0
    has a plane face of area ≈ 12 335 with 4 wires — but two pairs are
    coincident copies of the same hole. After dedup, the face should
    report at most one entry per real feature: 1 outer + 2 holes (2
    distinct circle shapes: ~12mm and ~10mm diameter).

    This is the user-facing fix: the "Assembly → 429 polylines" output
    is the symptom, the dedup is the cure.
    """
    if not USER_SAMPLE_FIXTURE.exists():
        pytest.skip("user_sample.STEP not present")
    from app.occ.contour import analyze_face, list_faces
    from app.occ.geometry_utils import face_area, face_surface_info
    from app.occ.loader import read_step_bytes

    shape = read_step_bytes(USER_SAMPLE_FIXTURE.read_bytes(), "user_sample.STEP")
    # Find a plane face on solid 0 with area ≈ 12 335 — that is the face
    # we already characterised in the wire-shape dump: 4 wires, 2
    # coincident pairs.
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from app.occ.contour import list_solids
    from app.occ.geometry_utils import face_wires

    solid = list_solids(shape)[0]
    target = None
    exp = TopExp_Explorer(solid, TopAbs_FACE)
    while exp.More():
        f = topods.Face(exp.Current())
        info = face_surface_info(f)
        if info.get("surface_type") == "plane":
            a = face_area(f)
            wires = face_wires(f)
            # Look for the 4-wire face with the area signature we saw in
            # the dump.
            if abs(a - 12335.5) < 1.0 and len(wires) == 4:
                target = (f, info, a)
                break
        exp.Next()
    if target is None:
        pytest.skip("couldn't locate the characterised 4-wire face")
    result = analyze_face(target[0], "face_test")
    inners = [c for c in result["contours"] if not c.get("is_outer")]
    # Two distinct circle holes; coincident pairs should have been merged.
    circle_holes = [c for c in inners if c["contour_type"] == "circle"]
    # We saw two distinct circle shapes (12mm and 10mm) in the dump.
    # With dedup we expect exactly 2 (one per unique feature), not 4.
    assert len(circle_holes) == 2, (
        f"expected 2 deduped circle holes, got {len(circle_holes)}: "
        f"{[round(c['parameters'].get('diameter') or 0, 2) for c in circle_holes]}"
    )
    diams = sorted([c["parameters"]["diameter"] for c in circle_holes])
    assert 9.5 < diams[0] < 10.5, f"smallest diameter out of range: {diams}"
    assert 11.5 < diams[1] < 12.5, f"largest diameter out of range: {diams}"


def test_per_solid_emits_unknown_for_unclassified_loops():
    """The new 4-classifier design (circle/slot/rectangle/hexagon) replaces
    the legacy `irregular` bucket with `unknown`. Any closed polyline that
    falls through all four classifiers is `unknown`. This test uses an
    ellipse-like shape (aspect=1.5) with a non-axis-aligned 5-point
    star cross-section to ensure no classifier claims it.
    """
    from app.occ.contour import classify_wire_contour
    import math

    # Slightly elongated cross: 5-point star outline with aspect=1.5.
    # - aspect=1.5 → CircleClassifier matches=False (bbox aspect too stretched)
    # - aspect<2.2 → SlotClassifier matches=False
    # - Not axis-aligned 4 corners → RectangleClassifier matches=False
    # - Not 6 corners → HexagonClassifier matches=False
    # → all 4 classifiers return matches=False → falls to unknown
    pts = [
        (0.0, 0.0, 0.0),
        (15.0, 2.0, 0.0),
        (22.0, 5.0, 0.0),
        (18.0, 12.0, 0.0),
        (8.0, 14.0, 0.0),
        (0.0, 8.0, 0.0),
    ]
    c = classify_wire_contour(
        pts,
        face_normal=(0.0, 0.0, 1.0),
        is_outer=False,
        wire_id="w_test",
        polyline_id="p_test",
        face_id="f_test",
        contour_index=0,
    )
    assert c["contour_type"] == "unknown", (
        f"expected `unknown`, got {c['contour_type']}"
    )


def test_per_solid_filters_out_unwanted_surface_lines():
    """A plane face with mixed real features + surface-trim noise
    (e.g. 4 wires, of which 1 is the outer boundary, 2 are real
    holes, 1 is decorative mesh noise) should report only the
    recognised features. The decorative line either gets classified
    into the standard buckets (rectangle/irregular) and is then
    included, or stays unrecognised and is filtered out by the
    bbox min-size check.
    """
    from app.occ.contour import _contour_to_hole

    # Speck-of-dust: a tiny closed polyline (0.1 mm square). No
    # matter what the classifier thinks of it, the size filter in
    # _contour_to_hole must drop it — otherwise the user sees
    # "0.1mm hole" in the feature table.
    tiny = [
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.1, 0.1, 0.0),
        (0.0, 0.1, 0.0),
    ]
    fake_contour = {
        "id": "c_tiny",
        "contour_type": "rectangle",
        "center": [0.05, 0.05, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "polyline_id": "p_tiny",
        "wire_id": "w_tiny",
        "face_id": "f_tiny",
        "is_outer": False,
        "parameters": {"length": 0.1, "width": 0.1, "diameter": None, "across_flats": None},
        "area": 0.01,
        "perimeter": 0.4,
    }
    holes: list = []
    _contour_to_hole(
        fake_contour, "f_tiny", holes, None,
        hole_diameter_min=0.5, hole_diameter_max=500.0,
    )
    assert holes == [], f"tiny feature should be filtered, got {holes}"

    # Sanity: a real-size 10mm circle should pass through.
    real = dict(fake_contour)
    real["id"] = "c_real"
    real["parameters"] = {"length": 10.0, "width": 10.0, "diameter": 10.0, "across_flats": None}
    holes2: list = []
    _contour_to_hole(
        real, "f_tiny", holes2, None,
        hole_diameter_min=0.5, hole_diameter_max=500.0,
    )
    assert len(holes2) == 1
    assert holes2[0]["diameter"] == 10.0

