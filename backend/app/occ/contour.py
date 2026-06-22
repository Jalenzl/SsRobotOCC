"""Face-level feature recognition (wires → polylines → contours → holes).

Pure algorithm module. All primitives are reused from existing OCC helpers:

- ``app.occ.geometry_utils``  →  face_surface_info / face_outward_normal /
                                  face_wires / face_area / work_plane_normal
- ``app.occ.discretize``       →  wire_to_polyline / wire_length /
                                  wire_area_if_planar / wire_location_on_face
- ``app.occ.topology``         →  face_point_and_outward_normal (sampling)
- ``app.occ.loader``           →  read_step_bytes

No HTTP / Pydantic coupling. Inputs are pythonOCC ``TopoDS_Face`` instances;
outputs are plain ``dict`` (JSON-serialisable) so they can flow through
``app.services.feature_service`` and ``app.routers.feature`` unchanged.

Classification is delegated to ``classifiers.registry.ClassifierRegistry``,
which tries Circle → Slot → Rectangle → Hexagon in priority order.
Any input that falls through all four classifiers returns the ``unknown``
bucket. This mirrors the SmartLaser ``MachiningPath`` abstract-base +
concrete-subclass pattern.
"""

from __future__ import annotations

import math
from typing import Any

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape

from app.occ.discretize import (
    wire_area_if_planar,
    wire_length,
    wire_location_on_face,
    wire_to_polyline,
)
from app.occ.geometry_utils import (
    face_area,
    face_outward_normal,
    face_surface_info,
    face_wires,
    work_plane_normal,
)

# ── Classifier registry (import lazily to avoid circular import) ──────────────
_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        from app.occ.classifiers.registry import ClassifierRegistry

        _registry = ClassifierRegistry()
    return _registry


# Lazy import for enhanced features (laser-software-style parameters)
_enhanced_module = None


def _get_enhanced_module():
    """Lazy load enhanced module to avoid circular imports."""
    global _enhanced_module
    if _enhanced_module is None:
        try:
            from app.occ import contour_enhanced

            _enhanced_module = contour_enhanced
        except ImportError:
            _enhanced_module = False
    return _enhanced_module


# Debug mode: set to True to include classification diagnosis
_DEBUG_CLASSIFICATION = True


# ── Shared thresholds ────────────────────────────────────────────────────────
_CLOSURE_TOL_DEFAULT = 0.5       # mm, default; overridable per call
# 轮廓最小面积阈值（mm²），小于此值的视为表面刻痕/点状噪声，不参与特征识别
_MIN_CONTOUR_AREA_MM2 = 1.0       # ≈ 1 mm² 的圆直径 ≈ 1.1mm
_MIN_IRREGULAR_AREA_MM2 = 10.0    # 异形孔最小面积，过小则为噪声


# ── Geometric helpers ─────────────────────────────────────────────────────────


def _vec(p: Any) -> list[float] | None:
    if p is None:
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def _pt(p: Any) -> list[float] | None:
    if p is None:
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def _dist(a, b) -> float:
    """Euclidean distance; accepts either 2D (x, y) or 3D (x, y, z) tuples.
    Some legacy callers (e.g. ``_looks_like_rectangle`` working on
    projected 2D points) feed 2D tuples into helpers that were originally
    written for 3D — pad with 0.0 to avoid IndexError.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if len(a) >= 3 and len(b) >= 3:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_length(pts: list[tuple[float, float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(pts)):
        total += _dist(pts[i - 1], pts[i])
    return total


def _polyline_area_2d(
    pts: list[tuple[float, float, float]],
    normal: tuple[float, float, float] | None,
) -> float:
    """Shoelace area of a closed polyline in its dominant 2D plane.

    BRepGProp.SurfaceProperties on a wire often returns 0, so we fall back
    to 2D shoelace using the same projection as ``_project_to_2d``.
    """
    if len(pts) < 3:
        return 0.0
    pts2d = _project_to_2d(pts, normal or (0.0, 0.0, 1.0))
    n = len(pts2d)
    s = 0.0
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _closed(pts, tol: float) -> bool:
    return len(pts) >= 3 and _dist(pts[0], pts[-1]) <= tol


def _project_to_2d(
    pts: list[tuple[float, float, float]],
    normal: tuple[float, float, float],
) -> list[tuple[float, float]]:
    """Drop the dominant-axis coordinate. Matches geometry_utils.project_point
    (the legacy helper) so contour 2D math is consistent with work-plane code.
    """
    nx, ny, nz = normal
    if abs(nz) >= max(abs(nx), abs(ny)):
        return [(p[0], p[1]) for p in pts]
    if abs(ny) >= abs(nx):
        return [(p[0], p[2]) for p in pts]
    return [(p[1], p[2]) for p in pts]


def _unique_ordered(pts: list[tuple[float, float]], tol: float = 1e-3) -> list[tuple[float, float]]:
    """Drop consecutive duplicates and the trailing closing-point (if equal to first)."""
    out: list[tuple[float, float]] = []
    for p in pts:
        if out and math.hypot(out[-1][0] - p[0], out[-1][1] - p[1]) <= tol:
            continue
        out.append(p)
    if len(out) >= 2 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


# ── Wire deduplication ───────────────────────────────────────────────────────


def _dedupe_wires(
    wire_infos: list[dict],
    tol: float = 0.5,
) -> list[dict]:
    """Drop wire entries that are coincident copies of an earlier wire.

    Why this exists: in STEP assemblies (and in many native B-Rep exports)
    a single closed loop on a face can appear multiple times in
    ``TopExp_Explorer(wire)`` because coincident edges get duplicated
    during the geometry translation. Without dedup the wire count
    doubles (or worse) and the user sees "the same hole listed 4 times"
    in the feature table.

    Signature: a 3D-DoppleGanger hash of the polyline vertices. We sample
    up to 16 evenly-spaced points around the loop, quantise to 0.05 mm,
    and compare as a multiset. Two loops are "the same" if the
    corresponding sampled points match within ``tol``. This is
    translation- and rotation-invariant (per face, after the 3D points
    have been transformed to a common origin via the wire centroid),
    and it correctly rejects two holes that happen to share the same
    bbox+area but sit in different 3D positions.
    """
    if len(wire_infos) < 2:
        return wire_infos

    def _loop_fingerprint(pts3d: list[tuple[float, float, float]]) -> tuple | None:
        if len(pts3d) < 3:
            return None
        step = max(1, len(pts3d) // 16)
        sampled = [pts3d[i] for i in range(0, len(pts3d), step)]
        if len(sampled) < 4:
            sampled = pts3d[:]
        cx = sum(p[0] for p in sampled) / len(sampled)
        cy = sum(p[1] for p in sampled) / len(sampled)
        cz = sum(p[2] for p in sampled) / len(sampled)
        q = 0.05
        return tuple(
            sorted(
                (
                    round((p[0] - cx) / q),
                    round((p[1] - cy) / q),
                    round((p[2] - cz) / q),
                )
                for p in sampled
            )
        )

    kept: list[dict] = []
    kept_fps: list[tuple] = []
    for w in wire_infos:
        pts = w.get("pts") or []
        if len(pts) < 3:
            if not any(True for k in kept if not (k.get("pts") or [])):
                kept.append(w)
            continue
        fp = _loop_fingerprint(pts)
        if fp is None:
            kept.append(w)
            kept_fps.append(())
            continue
        fp_set = set(fp)
        dup = False
        for k_fp in kept_fps:
            if not k_fp:
                continue
            if fp_set == set(k_fp):
                dup = True
                break
        if not dup:
            kept.append(w)
            kept_fps.append(fp)
    return kept


def _project_wire_points_2d(
    pts_world: list[tuple[float, float, float]],
    face_normal: tuple[float, float, float] | None,
) -> list[tuple[float, float]]:
    """Project 3D wire points onto the dominant plane of the face normal.
    Used both by the classifier and by the dedup helper to keep their
    coordinate systems consistent.
    """
    nx, ny, nz = face_normal or (0.0, 0.0, 1.0)
    if abs(nz) >= max(abs(nx), abs(ny)):
        return [(p[0], p[1]) for p in pts_world]
    if abs(ny) >= abs(nx):
        return [(p[0], p[2]) for p in pts_world]
    return [(p[1], p[2]) for p in pts_world]


def _mark_concentric_rings(
    contours: list[dict],
    face_id: str,
    centre_tol_mm: float = 0.5,
) -> None:
    """Tag every ``circle`` contour that is a *larger concentric ring* of
    a smaller one with ``contour_role="concentric_ring"``.

    Multiple closed wires on the same face that classify as
    ``circle`` (or ``ellipse``) are usually the same physical hole
    seen at different depths / radii — the through-hole's wire, the
    bottom of a counterbore, the chamfer ring at the top edge, etc.
    They share a 2D centre within ``centre_tol_mm`` and have
    increasing diameter. Tagging all-but-the-smallest as
    ``concentric_ring`` lets the hole-derivation step keep the
    through-hole only.

    Modifies the contours list in-place by setting
    ``contour["contour_role"] = "concentric_ring"`` on the outer
    rings. The smallest circle in each cluster is left as
    ``"inner_hole"`` so the existing pipeline treats it as the
    primary hole.
    """
    circles = [
        c for c in contours
        if c.get("contour_type") in ("circle", "ellipse")
        and not c.get("is_outer")
        and c.get("center")
    ]
    if len(circles) < 2:
        return

    def _diameter(c: dict) -> float:
        params = c.get("parameters") or {}
        if c.get("contour_type") == "ellipse":
            return float(params.get("length") or 0.0)
        return float(params.get("diameter") or 0.0)

    n = len(circles)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            ci = circles[i]["center"]
            cj = circles[j]["center"]
            dx = ci[0] - cj[0]
            dy = ci[1] - cj[1]
            if (dx * dx + dy * dy) ** 0.5 < centre_tol_mm:
                _union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        r = _find(i)
        clusters.setdefault(r, []).append(i)

    for r, members in clusters.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda i: _diameter(circles[i]))
        kept = members[0]
        for idx in members[1:]:
            circles[idx]["contour_role"] = "concentric_ring"
            if _DEBUG_CLASSIFICATION:
                import logging

                logging.getLogger("contour").debug(
                    "concentric_ring: face=%s cid=%s diameter=%.4f tagged as ring (kept cid=%s d=%.4f)",
                    face_id,
                    circles[idx].get("id"),
                    _diameter(circles[idx]),
                    circles[kept].get("id"),
                    _diameter(circles[kept]),
                )


def _dedupe_holes_across_faces(
    holes: list[dict],
    dominant: dict | None,
    centre_tol_mm: float = 0.5,
    size_tol_ratio: float = 0.05,
) -> list[dict]:
    """Drop holes that are coincident copies of an earlier one.

    A real through-hole on a plate is captured twice: once on the
    top face's wire and once on the bottom face's wire — same
    3D centre, same kind, same diameter, just mirrored. We keep
    the one whose axis is most aligned with the dominant face
    normal (the visible side the user is looking at).
    """
    if len(holes) < 2:
        return list(holes)

    n = len(holes)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    def _size(h: dict) -> float:
        params = h.get("parameters") or {}
        d = float(params.get("diameter") or 0.0)
        if d > 0:
            return d
        return max(float(params.get("length") or 0.0), float(params.get("width") or 0.0))

    def _centre(h: dict) -> tuple[float, float, float]:
        c = h.get("center") or [0.0, 0.0, 0.0]
        return float(c[0]), float(c[1]), float(c[2])

    dom_n = dominant.get("normal") if dominant else None

    for i in range(n):
        ci = _centre(holes[i])
        si = _size(holes[i])
        ki = holes[i].get("kind") or holes[i].get("contour_type")
        for j in range(i + 1, n):
            cj = _centre(holes[j])
            sj = _size(holes[j])
            kj = holes[j].get("kind") or holes[j].get("contour_type")
            if ki != kj:
                continue
            dx = ci[0] - cj[0]
            dy = ci[1] - cj[1]
            dz = ci[2] - cj[2]
            if (dx * dx + dy * dy + dz * dz) ** 0.5 > centre_tol_mm:
                continue
            if si <= 0 or sj <= 0:
                continue
            rel = abs(si - sj) / max(si, sj)
            if rel > size_tol_ratio:
                continue
            _union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(_find(i), []).append(i)
    survivors: list[int] = []
    for r, members in clusters.items():
        if len(members) == 1:
            survivors.append(members[0])
            continue

        def _score(idx: int) -> tuple[float, str]:
            axis = holes[idx].get("axis")
            if dom_n and axis and len(axis) == 3:
                try:
                    dot = abs(axis[0] * dom_n[0] + axis[1] * dom_n[1] + axis[2] * dom_n[2])
                except (TypeError, IndexError):
                    dot = 0.0
            else:
                dot = 0.0
            return (dot, str(holes[idx].get("face_id") or ""))

        members.sort(key=_score, reverse=True)
        survivors.append(members[0])
    return [holes[i] for i in sorted(survivors)]


# ── Unknown / degenerate bucket ──────────────────────────────────────────────


def _unknown_contour(cid, wire_id, polyline_id, face_id, is_outer) -> dict:
    return {
        "id": cid,
        "contour_type": "unknown",
        "contour_role": "outer_boundary" if is_outer else "inner_hole",
        "center": None,
        "normal": None,
        "polyline_id": polyline_id,
        "wire_id": wire_id,
        "face_id": face_id,
        "is_outer": is_outer,
        "parameters": {"diameter": None, "length": None, "width": None, "across_flats": None},
        "area": None,
        "perimeter": None,
        "confidence": 0.0,
    }


# ── Public API: contour classification ────────────────────────────────────────


def classify_wire_contour(
    pts_world: list[tuple[float, float, float]],
    face_normal: tuple[float, float, float] | None,
    is_outer: bool,
    wire_id: str,
    polyline_id: str,
    face_id: str,
    contour_index: int,
    prefer_pca_plane: bool = False,
    enhanced_params: bool = False,
) -> dict:
    """Classify one closed polyline as outer / circle / slot / rectangle /
    hexagon / unknown. Returns a plain dict that matches the
    ``ContourFeature`` pydantic schema (subset of fields).

    Classification is delegated to ``ClassifierRegistry`` which tries
    Circle → Slot → Rectangle → Hexagon in priority order. Any input
    that falls through all four classifiers returns the ``unknown`` bucket.
    This mirrors the SmartLaser ``MachiningPath`` abstract-base +
    concrete-subclass pattern.

    Args:
        enhanced_params: If True, extract laser-software-style parameters
            (rotation_angle, corner_radius, compensation_length, etc.)
    """
    cid = f"contour_{contour_index}"

    # Degenerate / noise gate — must have at least 4 real points
    if not pts_world or len(pts_world) < 4:
        return _unknown_contour(cid, wire_id, polyline_id, face_id, is_outer)

    face_n = face_normal or (0.0, 0.0, 1.0)
    pts2d_raw = _project_to_2d(pts_world, face_n)
    n_raw = len(pts2d_raw)
    if n_raw < 4:
        return _unknown_contour(cid, wire_id, polyline_id, face_id, is_outer)

    # ── Deduplicate — mirrors the legacy code which called
    #    `_unique_ordered` before any metric computation.
    pts2d = _unique_ordered(pts2d_raw)
    n = len(pts2d)
    if n < 4:
        return _unknown_contour(cid, wire_id, polyline_id, face_id, is_outer)

    # ── Bbox + shoelace area from the deduplicated polyline.
    xs = [p[0] for p in pts2d]
    ys = [p[1] for p in pts2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    s = 0.0
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    area_2d = abs(s) * 0.5

    # ── Tiny-noise gate
    if area_2d < _MIN_CONTOUR_AREA_MM2:
        if _DEBUG_CLASSIFICATION and not is_outer:
            import logging

            logging.getLogger("contour").debug(
                "area_filter: cid=%s area=%.4f mm² < %.2f mm² (skipped as noise)",
                cid, area_2d, _MIN_CONTOUR_AREA_MM2,
            )
        return _unknown_contour(cid, wire_id, polyline_id, face_id, is_outer)

    # ── Perimeter from deduplicated polyline (no wrap-back duplicate).
    #    Legacy `_polyline_length` summed segments 0→1, 1→2, …, (n-2)→(n-1)
    #    without the (n-1)→0 wrap. We match that exactly.
    perimeter = 0.0
    if n >= 2:
        perimeter = sum(
            _dist(pts2d[i], pts2d[i + 1]) for i in range(n - 1)
        )

    # ── Tessellated 2D points for the classifier.
    #    The registry needs full tessellation for slot edge detection and
    #    circle arc-fitting. Pass the raw 2D projected points (before
    #    deduplication of consecutive duplicates at the end).
    pts2d_for_classifier = pts2d_raw

    # 3D centroid: guaranteed to lie on the face plane (unlike an
    # axis-projected 2D centroid mapped back to 3D, which floats off
    # for tilted planar faces).
    if pts_world:
        cx_3d = sum(p[0] for p in pts_world) / len(pts_world)
        cy_3d = sum(p[1] for p in pts_world) / len(pts_world)
        cz_3d = sum(p[2] for p in pts_world) / len(pts_world)
        center_3d: list[float] | None = [float(cx_3d), float(cy_3d), float(cz_3d)]
    else:
        center_3d = None
    normal_3d = _vec(face_normal) if face_normal else None

    # contour_role: clear semantic role for the frontend
    contour_role = "outer_boundary" if is_outer else "inner_hole"

    # ── 1) outer always wins — must be preserved from the registry
    #    (the registry returns a contour_type but cannot know whether
    #    this wire is the face boundary, so we override it here)
    if is_outer:
        result = {
            "id": cid,
            "contour_type": "outer",
            "contour_role": contour_role,
            "center": center_3d,
            "normal": normal_3d,
            "polyline_id": polyline_id,
            "wire_id": wire_id,
            "face_id": face_id,
            "is_outer": True,
            "parameters": {"diameter": None, "length": None, "width": None, "across_flats": None},
            "area": round(area_2d, 4),
            "perimeter": round(perimeter, 4),
            "confidence": 1.0,
        }
    else:
        # ── 2) delegate to classifier registry ─────────────────────────────
        registry = _get_registry()
        classifier_result = registry.classify(
            pts2d_for_classifier,
            face_normal=tuple(face_n),
            pts_world=pts_world,
        )
        result = {
            "id": cid,
            "contour_type": classifier_result["contour_type"],
            "contour_role": contour_role,
            "center": classifier_result.get("center") or center_3d,
            "normal": classifier_result.get("normal") or normal_3d,
            "polyline_id": polyline_id,
            "wire_id": wire_id,
            "face_id": face_id,
            "is_outer": False,
            "parameters": classifier_result.get("parameters", {}),
            "area": round(classifier_result.get("area") or area_2d, 4),
            "perimeter": round(classifier_result.get("perimeter") or perimeter, 4),
            "confidence": round(classifier_result.get("_confidence", 0.0), 3),
        }

    # ── 3) enhanced parameters (laser-software-style) ───────────────────────
    if enhanced_params:
        enhanced = _get_enhanced_module()
        if enhanced:
            ctype = result["contour_type"]
            # Use the same deduplicated polyline for both area and perimeter
            # (consistent with the perimeter computed above).
            n_pts = len(pts2d)
            xs = [p[0] for p in pts2d]
            ys = [p[1] for p in pts2d]
            s_enh = 0.0
            for i in range(n_pts):
                x1, y1 = pts2d[i]
                x2, y2 = pts2d[(i + 1) % n_pts]
                s_enh += x1 * y2 - x2 * y1
            area_enh = abs(s_enh) * 0.5
            circularity_enh = (4 * math.pi * area_enh) / (perimeter * perimeter) if perimeter > 0 else 0.0

            enhanced_params_dict = enhanced.extract_contour_parameters(
                pts2d, ctype, circularity_enh, perimeter
            )
            for key, value in enhanced_params_dict.items():
                if value is not None:
                    result["parameters"][key] = value

            result["confidence"] = enhanced.calculate_classification_confidence(
                pts2d_raw, circularity_enh, ctype, result["parameters"]
            )

            is_valid, error_msg = enhanced.validate_contour_parameters(
                result["parameters"], ctype
            )
            result["validation"] = {"is_valid": is_valid, "error": error_msg}

            lead_length = enhanced.estimate_lead_length(ctype, result["parameters"])
            result["lead_length"] = round(lead_length, 4)

            if _DEBUG_CLASSIFICATION or ctype == "unknown":
                diagnosis = enhanced.diagnose_classification(pts2d_raw, is_outer)
                result["_diagnosis"] = diagnosis

    return result


# ── Per-face pipeline ─────────────────────────────────────────────────────────


def analyze_face(
    face: TopoDS_Face,
    face_id: str,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    work_plane: str = "auto",
    work_plane_normal: tuple[float, float, float] | None = None,
    hole_diameter_min: float = 0.5,
    hole_diameter_max: float = 500.0,
    include_cylinder_holes: bool = False,
    closure_tol: float = _CLOSURE_TOL_DEFAULT,
    enhanced_params: bool = False,
) -> dict:
    """Run the full wire → polyline → contour → hole pipeline on a single
    TopoDS_Face. Returns plain dict that matches ``CadFaceAnalyzeResult``.
    """
    if _DEBUG_CLASSIFICATION:
        import logging
        _cl = logging.getLogger("contour")
        if not _cl.handlers:
            _h = logging.StreamHandler()
            _h.setLevel(logging.DEBUG)
            _h.setFormatter(logging.Formatter("[CONTOUR] %(message)s"))
            _cl.addHandler(_h)
        _cl.setLevel(logging.DEBUG)
    surf = face_surface_info(face)
    f_normal = face_outward_normal(face) or surf.get("normal")
    f_axis = surf.get("axis")
    f_center = surf.get("center")
    f_radius = surf.get("radius")

    if work_plane_normal is None:
        if f_normal and work_plane == "auto":
            wp_normal = f_normal
        else:
            wp_normal = work_plane_normal_for_mode(work_plane, f_normal)
    else:
        wp_normal = work_plane_normal

    wires = face_wires(face)
    polylines: list[dict] = []
    contours: list[dict] = []
    holes: list[dict] = []
    ref_points: list[dict] = []

    # 1) collect per-wire polyline + classify
    wire_infos: list[dict] = []
    for wi, wire in enumerate(wires):
        wid = f"wire_{face_id}_{wi}"
        loc = wire_location_on_face(face, wire)
        pts = wire_to_polyline(
            wire, linear_deflection, angular_deflection, location=loc
        )
        pid = f"poly_{wid}"
        wlen = wire_length(wire) if pts else 0.0
        warea = wire_area_if_planar(wire) if pts else None
        if not warea:
            warea = _polyline_area_2d(pts, tuple(f_normal) if f_normal else None)

        wire_infos.append({
            "id": wid,
            "is_outer": False,
            "closed": _closed(pts, tol=closure_tol),
            "length": wlen,
            "area": warea,
            "polyline_id": pid,
            "pts": pts,
            "pts2d": _project_wire_points_2d(pts, tuple(f_normal) if f_normal else None),
        })

    # 1b) keep ALL wires (no dedup) — coincident wires on the same face
    # may represent distinct design features (e.g. a stepped hole whose
    # inner and outer rims share the same 3D loop in some STEP exports);
    # deduping them silently drops features the user is expecting to see.

    # 1c) build polyline list from deduped wire set
    polylines = [
        {
            "id": w.get("polyline_id") or f"poly_{w['id']}",
            "closed": w.get("closed", False),
            "points": [_pt(p) for p in (w.get("pts") or [])],
        }
        for w in wire_infos
    ]

    # 2) outer / inner split using bbox containment
    closed_wires = [w for w in wire_infos if w["closed"] and (w["area"] or 0) > 0]
    # ── DEBUG: 完整 wire 列表 ─────────────────────────────────────
    import logging
    logging.getLogger("contour").debug(
        "[WIRES] face=%s all_wires=%s",
        face_id,
        [(w["id"], w.get("closed"), round(w.get("area") or 0.0, 4),
          len(w.get("pts2d") or []))
         for w in wire_infos],
    )
    if closed_wires:
        bbox_by_id: dict[str, tuple[float, float, float, float]] = {}
        for w in closed_wires:
            xs = [p[0] for p in w.get("pts2d") or []]
            ys = [p[1] for p in w.get("pts2d") or []]
            if not xs:
                bbox_by_id[w["id"]] = (0.0, 0.0, 0.0, 0.0)
            else:
                bbox_by_id[w["id"]] = (min(xs), min(ys), max(xs), max(ys))

        contained_by: dict[str, str | None] = {w["id"]: None for w in closed_wires}
        for wA in closed_wires:
            ax1, ay1, ax2, ay2 = bbox_by_id[wA["id"]]
            for wB in closed_wires:
                if wB["id"] == wA["id"]:
                    continue
                bx1, by1, bx2, by2 = bbox_by_id[wB["id"]]
                if (bx1 <= ax1 + 0.1 and by1 <= ay1 + 0.1
                        and bx2 >= ax2 - 0.1 and by2 >= ay2 - 0.1):
                    if wB["area"] >= wA["area"] * 1.05:
                        if contained_by[wA["id"]] is None or wB["area"] < (
                            closed_wires[next(
                                i for i, w in enumerate(closed_wires)
                                if w["id"] == contained_by[wA["id"]]
                            )]["area"]
                        ):
                            contained_by[wA["id"]] = wB["id"]

        not_contained = [w for w in closed_wires if contained_by[w["id"]] is None]
        if len(not_contained) == 1:
            outer_id = not_contained[0]["id"]
        elif not_contained:
            not_contained.sort(key=lambda w: w["area"] or 0.0, reverse=True)
            outer_id = not_contained[0]["id"]
        else:
            closed_wires_sorted = sorted(closed_wires, key=lambda w: w["area"] or 0.0, reverse=True)
            outer_id = closed_wires_sorted[0]["id"]

        for w in wire_infos:
            w["is_outer"] = (w["id"] == outer_id)

        # ── DEBUG: outer 判定详细输出 ─────────────────────────────
        import logging
        logging.getLogger("contour").debug(
            "[OUTER] face=%s closed_wires=%s outer_id=%s",
            face_id,
            [(w["id"], round(w.get("area") or 0.0, 4)) for w in closed_wires],
            outer_id,
        )

    # 3) classify each wire
    for ci, w in enumerate(wire_infos):
        contour = classify_wire_contour(
            w.get("pts") or [],
            face_normal=tuple(f_normal) if f_normal else None,
            is_outer=w["is_outer"],
            wire_id=w["id"],
            polyline_id=w["polyline_id"],
            face_id=face_id,
            contour_index=ci,
            enhanced_params=enhanced_params,
        )
        if w.get("area"):
            contour["area"] = w["area"]
        contours.append(contour)

    # 3b) concentric-circle grouping
    _mark_concentric_rings(contours, face_id)

    # ── DEBUG: 每条 wire 的分类结果 ──────────────────────────────
    if _DEBUG_CLASSIFICATION:
        import logging
        for w, contour in zip(wire_infos, contours):
            logging.getLogger("contour").debug(
                "[CLASS] wire=%s -> contour=%s type=%s is_outer=%s area=%.4f",
                w["id"], contour.get("id"), contour.get("contour_type"),
                w.get("is_outer"), w.get("area") or 0.0,
            )

    # 4) wires (post-classification)
    wires_out: list[dict] = []
    for w, contour in zip(wire_infos, contours):
        wires_out.append({
            "id": w["id"],
            "face_id": face_id,
            "is_outer": w["is_outer"],
            "length": w["length"],
            "area": w["area"],
            "polyline_id": w["polyline_id"],
            "contour_id": contour["id"],
            "contour_type": contour["contour_type"],
        })

    # 5) hole derivation
    is_planar = (surf.get("surface_type") == "plane")
    for contour in contours:
        if not is_planar:
            continue
        if contour["is_outer"]:
            continue
        if contour.get("contour_role") == "concentric_ring":
            continue
        if contour["contour_type"] in (
            "circle",
            "ellipse",
            "slot",
            "rectangle",
            "hexagon",
            "unknown",
        ):
            _contour_to_hole(
                contour, face_id, holes, ref_points,
                hole_diameter_min, hole_diameter_max
            )

    # 6) reference points
    for contour in contours:
        if contour.get("center") is None:
            continue
        ref_points.append({
            "id": f"pt_{contour['id']}_center",
            "kind": "contour_center",
            "position": contour["center"],
            "meta": {
                "contour_id": contour["id"],
                "contour_type": contour["contour_type"],
                "face_id": face_id,
            },
        })

    # 7) outer_contours
    outer_contours = _select_global_outer_contours(contours)

    # 8) face record
    outer_wire_id = next((w["id"] for w in wire_infos if w["is_outer"]), None)
    inner_wire_ids = [w["id"] for w in wire_infos if not w["is_outer"]]
    face_record = {
        "id": face_id,
        "surface_type": surf.get("surface_type", "other"),
        "area": face_area(face) if surf.get("surface_type") == "plane" else None,
        "normal": _vec(f_normal) if f_normal else None,
        "axis": _vec(f_axis) if f_axis else None,
        "center": _pt(f_center) if f_center else None,
        "radius": f_radius,
        "bbox": None,
        "outer_wire_id": outer_wire_id,
        "inner_wire_ids": inner_wire_ids,
    }

    return {
        "schema_version": "1.1",
        "unit": "mm",
        "target_face_id": face_id,
        "face": face_record,
        "reference_points": ref_points,
        "polylines": polylines,
        "wires": wires_out,
        "contours": contours,
        "outer_contours": outer_contours,
        "outer_contour_ids": [c["id"] for c in outer_contours],
        "holes": holes,
        "pockets": [],
        "feature_groups": _build_feature_groups(contours=contours, holes=holes, wires=wires_out),
        "work_plane": work_plane,
        "work_plane_normal": _vec(wp_normal) if wp_normal else None,
    }


def _contour_to_hole(
    contour: dict,
    face_id: str,
    holes: list,
    ref_points: list,
    hole_diameter_min: float,
    hole_diameter_max: float,
) -> None:
    ctype = contour["contour_type"]
    params = contour.get("parameters") or {}
    diam = params.get("diameter")
    # Primary size estimate from explicit parameters (diameter > length > width)
    bbox_max = max(
        float(diam or 0.0),
        float(params.get("length") or 0.0),
        float(params.get("width") or 0.0),
    )
    # If no explicit size parameter is available, back-compute from the
    # 2D shoelace area stored in contour["area"]. This is consistent with
    # how the classifier computed the diameter and avoids mismatches where
    # raw BRepGProp wire area is wrong on non-planar faces.
    contour_area = float(contour.get("area") or 0.0)
    if bbox_max <= 0 and contour_area > 0:
        bbox_max = 2.0 * math.sqrt(contour_area / math.pi)
    if bbox_max > 0 and not (hole_diameter_min <= bbox_max <= hole_diameter_max):
        if _DEBUG_CLASSIFICATION:
            import logging

            logging.getLogger("contour").debug(
                "hole_size_filter: cid=%s type=%s bbox_max=%.4f (min=%.2f max=%.2f) - rejected",
                contour.get("id"), ctype, bbox_max, hole_diameter_min, hole_diameter_max,
            )
        return
    if _DEBUG_CLASSIFICATION:
        import logging

        logging.getLogger("contour").debug(
            "hole_accepted: cid=%s type=%s bbox_max=%.4f", contour.get("id"), ctype, bbox_max,
        )
    holes.append({
        "id": f"hole_{contour['id']}",
        "kind": ctype,
        "contour_type": ctype,
        "center": contour["center"],
        "axis": contour["normal"],
        "diameter": diam,
        "depth": None,
        "face_id": face_id,
        "wire_id": contour.get("wire_id"),
        "cylindrical_face_ids": [],
        "parameters": {
            "diameter": params.get("diameter"),
            "length": params.get("length"),
            "width": params.get("width"),
            "across_flats": params.get("across_flats"),
        },
    })
    if ref_points is not None and contour.get("center"):
        ref_points.append({
            "id": f"pt_hole_{contour['id']}",
            "kind": "hole_center",
            "position": contour["center"],
            "meta": {
                "diameter": diam,
                "contour_type": ctype,
                "contour_id": contour["id"],
                "face_id": face_id,
            },
        })


def _select_global_outer_contours(contours: list[dict]) -> list[dict]:
    outer = [c for c in contours if c.get("is_outer")]
    outer.sort(key=lambda c: c.get("area") or 0.0, reverse=True)
    return outer


def _build_feature_groups(*, contours, holes, wires) -> dict:
    contours_by_type: dict[str, list] = {}
    for c in contours:
        contours_by_type.setdefault(c.get("contour_type", "unknown"), []).append(c)
    holes_by_type: dict[str, list] = {}
    for h in holes:
        holes_by_type.setdefault(h.get("contour_type") or h.get("kind") or "unknown", []).append(h)
    return {
        "contours_by_type": contours_by_type,
        "holes_by_type": holes_by_type,
        "wires_by_role": {
            "outer": [w for w in wires if w.get("is_outer")],
            "inner": [w for w in wires if not w.get("is_outer")],
        },
    }


def work_plane_normal_for_mode(
    mode: str,
    fallback: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    """Wrapper around ``geometry_utils.work_plane_normal`` that accepts a
    degenerate fallback for non-bbox cases.
    """
    if mode in ("xy", "yz", "xz"):
        return work_plane_normal(mode, (0, 0, 0, 1, 1, 1))
    return fallback


# ── Shape-level entry: enumerate faces & find by id ──────────────────────────


def list_faces(shape: TopoDS_Shape) -> list[TopoDS_Face]:
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    out: list[TopoDS_Face] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(topods.Face(exp.Current()))
        exp.Next()
    return out


def find_face_by_id(shape: TopoDS_Shape, face_id: str) -> tuple[TopoDS_Face, str]:
    faces = list_faces(shape)
    if not faces:
        raise ValueError("model has no faces")

    fid = (face_id or "").strip()
    if not fid:
        raise ValueError("face_id 不能为空")

    target_idx: int | None = None
    lower = fid.lower()
    if lower.startswith("face_"):
        try:
            target_idx = int(fid[5:])
        except ValueError:
            target_idx = None
    elif lower.startswith("part_"):
        from app.occ.topology import iter_solids

        try:
            part_idx = int(fid[5:])
        except ValueError:
            raise ValueError(f"face_id 解析失败: {face_id!r}")
        solids = iter_solids(shape)
        if part_idx < 0 or part_idx >= len(solids):
            raise ValueError(
                f"part_id {face_id!r} out of range (0..{len(solids) - 1})"
            )
        target_solid = solids[part_idx]
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopoDS import topods

        exp = TopExp_Explorer(target_solid, TopAbs_FACE)
        if not exp.More():
            raise ValueError(f"part_id {face_id!r} 不含任何 face")
        first_face = topods.Face(exp.Current())
        try:
            local_idx = faces.index(first_face)
        except ValueError:
            local_idx = 0
        return faces[local_idx], f"face_{local_idx}"
    elif fid.isdigit():
        target_idx = int(fid)

    if target_idx is None or target_idx < 0 or target_idx >= len(faces):
        if len(faces) == 1:
            return faces[0], "face_0"
        raise ValueError(
            f"face_id {face_id!r} out of range (0..{len(faces) - 1})"
        )
    return faces[target_idx], f"face_{target_idx}"


def list_solids(shape: TopoDS_Shape) -> list:
    from app.occ.topology import iter_solids
    from OCC.Core.TopoDS import topods

    solids = iter_solids(shape)
    if solids:
        return solids
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer

    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    out: list = []
    while exp.More():
        out.append(topods.Solid(exp.Current()))
        exp.Next()
    return out


def _face_count_in_solid(solid) -> int:
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer

    exp = TopExp_Explorer(solid, TopAbs_FACE)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def find_solid_by_id(shape: TopoDS_Shape, part_id: str) -> tuple:
    import re

    fid = (part_id or "").strip()
    if not fid:
        raise ValueError("part_id 不能为空")
    lower = fid.lower()
    idx: int | None = None

    if lower.startswith("part_"):
        suffix = fid[5:]
        if suffix.isdigit():
            idx = int(suffix)
        else:
            solids = list_solids(shape)
            if len(solids) == 1:
                return solids[0], 0
            raise ValueError(
                f"part_id {fid!r} 不含数字索引，且模型包含 {len(solids)} 个 solid，"
                f"请用 part_0..part_{len(solids) - 1} 指定"
            )
    elif fid.isdigit():
        idx = int(fid)
    else:
        solids = list_solids(shape)
        m = re.search(r"_part_(\d+)$", lower, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
        else:
            if len(solids) == 1:
                return solids[0], 0
            return None, "ALL"

    solids = list_solids(shape)
    if idx < 0 or idx >= len(solids):
        if len(solids) == 1:
            return solids[0], 0
        raise ValueError(
            f"part_id {part_id!r} out of range (0..{len(solids) - 1})"
        )
    return solids[idx], idx


def analyze_part(
    shape: TopoDS_Shape,
    part_id: str,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    work_plane: str = "auto",
    work_plane_normal: tuple[float, float, float] | None = None,
    hole_diameter_min: float = 0.5,
    hole_diameter_max: float = 500.0,
    include_cylinder_holes: bool = False,
    closure_tol: float = _CLOSURE_TOL_DEFAULT,
    enhanced_params: bool = False,
    target_face_id: str | None = None,
) -> dict:
    """Analyse a whole Solid (part) and aggregate features of all its
    plane faces. Non-planar faces contribute only their metadata.
    """
    if _DEBUG_CLASSIFICATION:
        import logging

        contour_logger = logging.getLogger("contour")
        if not contour_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter("[CONTOUR] %(message)s"))
            contour_logger.addHandler(handler)
        contour_logger.setLevel(logging.DEBUG)

    from app.occ.geometry_utils import (
        face_area,
        face_surface_info,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods

    solid_or_all, solid_idx = find_solid_by_id(shape, part_id)
    bbox = shape_bbox_dict(shape)

    global_faces = list_faces(shape)

    def _face_index(target) -> int | None:
        for i, f in enumerate(global_faces):
            if f.IsSame(target):
                return i
        return None

    def _analyze_one_solid(solid, idx_label: int | str) -> dict:
        solid_faces: list[tuple[int, TopoDS_Face]] = []
        exp = TopExp_Explorer(solid, TopAbs_FACE)
        while exp.More():
            f = topods.Face(exp.Current())
            gi = _face_index(f)
            if gi is not None:
                solid_faces.append((gi, f))
            exp.Next()
        solid_faces.sort(key=lambda x: x[0])

        target_normal: tuple[float, float, float] | None = None
        if target_face_id:
            for gi, f in solid_faces:
                if f"face_{gi}" == target_face_id:
                    n = face_outward_normal(f)
                    if n:
                        target_normal = n
                    break

        per_face: list[dict] = []
        agg_contours: list[dict] = []
        agg_holes: list[dict] = []
        agg_ref: list[dict] = []
        agg_wires: list[dict] = []
        agg_polylines: list[dict] = []
        agg_outer: list[str] = []

        plane_face_count = 0
        for gi, f in solid_faces:
            info = face_surface_info(f)
            if info.get("surface_type") != "plane":
                per_face.append({
                    "id": f"face_{gi}",
                    "surface_type": info.get("surface_type"),
                    "area": None,
                    "normal": _vec(info.get("normal")),
                    "axis": _vec(info.get("axis")),
                    "center": _pt(info.get("center")),
                    "radius": info.get("radius"),
                    "analysed": False,
                    "contour_count": 0,
                    "hole_count": 0,
                })
                continue
            plane_face_count += 1

            if target_normal is not None:
                face_n = face_outward_normal(f)
                if face_n is None:
                    face_n = info.get("normal")
                if face_n is not None:
                    dot = abs(
                        face_n[0] * target_normal[0]
                        + face_n[1] * target_normal[1]
                        + face_n[2] * target_normal[2]
                    )
                    if dot < 0.5:
                        per_face.append({
                            "id": f"face_{gi}",
                            "surface_type": "plane",
                            "area": face_area(f),
                            "normal": _vec(info.get("normal")),
                            "axis": _vec(info.get("axis")),
                            "center": _pt(info.get("center")),
                            "radius": info.get("radius"),
                            "analysed": False,
                            "contour_count": 0,
                            "hole_count": 0,
                            "skipped_reason": "back_side",
                        })
                        continue

            sub = analyze_face(
                f,
                f"face_{gi}",
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
                work_plane=work_plane,
                work_plane_normal=work_plane_normal,
                hole_diameter_min=hole_diameter_min,
                hole_diameter_max=hole_diameter_max,
                include_cylinder_holes=include_cylinder_holes,
                closure_tol=closure_tol,
                enhanced_params=enhanced_params,
            )
            per_face.append({
                "id": f"face_{gi}",
                "surface_type": "plane",
                "area": face_area(f),
                "normal": _vec(info.get("normal")),
                "axis": _vec(info.get("axis")),
                "center": _pt(info.get("center")),
                "radius": info.get("radius"),
                "analysed": True,
                "contour_count": len(sub["contours"]),
                "hole_count": len(sub["holes"]),
            })
            for c in sub["contours"]:
                c2 = dict(c)
                c2["id"] = f"face_{gi}__{c['id']}"
                c2["wire_id"] = f"face_{gi}__{c.get('wire_id', '')}"
                c2["polyline_id"] = f"face_{gi}__{c.get('polyline_id', '')}"
                c2["face_id"] = f"face_{gi}"
                agg_contours.append(c2)
            for h in sub["holes"]:
                h2 = dict(h)
                h2["id"] = f"face_{gi}__{h['id']}"
                h2["face_id"] = f"face_{gi}"
                h2["wire_id"] = f"face_{gi}__{h.get('wire_id', '')}"
                agg_holes.append(h2)
            for r in sub.get("reference_points") or []:
                r2 = dict(r)
                r2["id"] = f"face_{gi}__{r['id']}"
                r2["face_id"] = f"face_{gi}"
                agg_ref.append(r2)
            for w in sub.get("wires") or []:
                w2 = dict(w)
                w2["id"] = f"face_{gi}__{w['id']}"
                w2["face_id"] = f"face_{gi}"
                w2["polyline_id"] = f"face_{gi}__{w.get('polyline_id', '')}"
                w2["contour_id"] = f"face_{gi}__{w.get('contour_id', '')}"
                agg_wires.append(w2)
            for p in sub.get("polylines") or []:
                p2 = dict(p)
                p2["id"] = f"face_{gi}__{p['id']}"
                agg_polylines.append(p2)
            for oid in sub.get("outer_contours") or []:
                prefixed = f"face_{gi}__{oid}"
                if prefixed not in agg_outer:
                    agg_outer.append(prefixed)

        plane_faces = [pf for pf in per_face if pf.get("analysed")]
        dominant = max(plane_faces, key=lambda x: x.get("area") or 0.0) if plane_faces else None
        if isinstance(idx_label, int):
            part_id_str = f"part_{idx_label}"
        else:
            part_id_str = f"part_{idx_label}"
        part_record = {
            "id": part_id_str,
            "surface_type": "compound",
            "area": dominant["area"] if dominant else None,
            "normal": dominant["normal"] if dominant else None,
            "axis": None,
            "center": None,
            "radius": None,
            "bbox": bbox,
            "face_count": len(solid_faces),
            "plane_face_count": plane_face_count,
            "dominant_face_id": dominant["id"] if dominant else None,
        }
        agg_holes = _dedupe_holes_across_faces(agg_holes, dominant)
        return {
            "target_face_id": part_id_str,
            "part": part_record,
            "per_face": per_face,
            "reference_points": agg_ref,
            "polylines": agg_polylines,
            "wires": agg_wires,
            "contours": agg_contours,
            "outer_contours": agg_outer,
            "holes": agg_holes,
        }

    # Whole-assembly aggregate path
    if solid_or_all is None:
        solids = list_solids(shape)
        merged: dict = {
            "target_face_id": "part_all",
            "part": None,
            "per_face": [],
            "reference_points": [],
            "polylines": [],
            "wires": [],
            "contours": [],
            "outer_contours": [],
            "holes": [],
        }
        seen_f: set[str] = set()
        total_face_count = 0
        total_plane_count = 0
        best_dominant = None
        for s_idx, solid in enumerate(solids):
            sub = _analyze_one_solid(solid, s_idx)
            total_face_count += sub["part"]["face_count"]
            total_plane_count += sub["part"]["plane_face_count"]
            if sub["part"].get("dominant_face_id") and (
                best_dominant is None
                or (sub["part"].get("area") or 0) > (best_dominant.get("area") or 0)
            ):
                best_dominant = {
                    "id": sub["part"]["dominant_face_id"],
                    "area": sub["part"].get("area"),
                    "normal": sub["part"].get("normal"),
                }
            for f in sub["per_face"]:
                if f["id"] in seen_f:
                    continue
                seen_f.add(f["id"])
                merged["per_face"].append(f)
            for c in sub["contours"]:
                c2 = dict(c)
                c2["id"] = f"s{s_idx}__{c['id']}"
                c2["face_id"] = f"s{s_idx}__{c.get('face_id', '')}"
                c2["wire_id"] = f"s{s_idx}__{c.get('wire_id', '')}"
                c2["polyline_id"] = f"s{s_idx}__{c.get('polyline_id', '')}"
                merged["contours"].append(c2)
            for h in sub["holes"]:
                h2 = dict(h)
                h2["id"] = f"s{s_idx}__{h['id']}"
                h2["face_id"] = f"s{s_idx}__{h.get('face_id', '')}"
                h2["wire_id"] = f"s{s_idx}__{h.get('wire_id', '')}"
                merged["holes"].append(h2)
            for p in sub["polylines"]:
                p2 = dict(p)
                p2["id"] = f"s{s_idx}__{p['id']}"
                merged["polylines"].append(p2)
            for w in sub["wires"]:
                w2 = dict(w)
                w2["id"] = f"s{s_idx}__{w['id']}"
                w2["face_id"] = f"s{s_idx}__{w.get('face_id', '')}"
                w2["polyline_id"] = f"s{s_idx}__{w.get('polyline_id', '')}"
                w2["contour_id"] = f"s{s_idx}__{w.get('contour_id', '')}"
                merged["wires"].append(w2)
            for r in sub["reference_points"]:
                r2 = dict(r)
                r2["id"] = f"s{s_idx}__{r['id']}"
                r2["meta"] = {**r.get("meta", {}), "solid_index": s_idx}
                merged["reference_points"].append(r2)
            merged["outer_contours"].extend(
                [f"s{s_idx}__{oid}" for oid in sub.get("outer_contours", [])]
            )
        merged["part"] = {
            "id": "part_all",
            "surface_type": "assembly",
            "area": best_dominant["area"] if best_dominant else None,
            "normal": best_dominant["normal"] if best_dominant else None,
            "axis": None,
            "center": None,
            "radius": None,
            "bbox": bbox,
            "face_count": total_face_count,
            "plane_face_count": total_plane_count,
            "dominant_face_id": best_dominant["id"] if best_dominant else None,
            "solid_count": len(solids),
        }
        return {
            "schema_version": "1.1",
            "unit": "mm",
            "target_face_id": merged["target_face_id"],
            "part": merged["part"],
            "per_face": merged["per_face"],
            "reference_points": merged["reference_points"],
            "polylines": merged["polylines"],
            "wires": merged["wires"],
            "contours": merged["contours"],
            "outer_contours": merged["outer_contours"],
            "outer_contour_ids": [c["id"] for c in merged["contours"] if c["id"] in merged["outer_contours"]],
            "holes": merged["holes"],
            "pockets": [],
            "feature_groups": _build_feature_groups(
                contours=merged["contours"],
                holes=merged["holes"],
                wires=merged["wires"],
            ),
            "work_plane": work_plane,
            "work_plane_normal": list(work_plane_normal) if work_plane_normal else None,
            "model_bbox": bbox,
        }

    # Single-solid path
    solid = solid_or_all
    sub = _analyze_one_solid(solid, solid_idx)
    return {
        "schema_version": "1.1",
        "unit": "mm",
        "target_face_id": sub["target_face_id"],
        "part": sub["part"],
        "per_face": sub["per_face"],
        "reference_points": sub["reference_points"],
        "polylines": sub["polylines"],
        "wires": sub["wires"],
        "contours": sub["contours"],
        "outer_contours": sub["outer_contours"],
        "outer_contour_ids": list(sub["outer_contours"]),
        "holes": sub["holes"],
        "pockets": [],
        "feature_groups": _build_feature_groups(
            contours=sub["contours"],
            holes=sub["holes"],
            wires=sub["wires"],
        ),
        "work_plane": work_plane,
        "work_plane_normal": list(work_plane_normal) if work_plane_normal else None,
        "model_bbox": bbox,
    }


def shape_bbox_dict(shape: TopoDS_Shape) -> dict:
    from app.occ.geometry_utils import shape_bbox

    xmin, ymin, zmin, xmax, ymax, zmax = shape_bbox(shape)
    return {
        "xmin": xmin, "ymin": ymin, "zmin": zmin,
        "xmax": xmax, "ymax": ymax, "zmax": zmax,
    }
