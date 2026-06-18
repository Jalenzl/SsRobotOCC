"""Feature service: orchestrates OCC analysis and returns JSON-ready dicts."""

from __future__ import annotations

from typing import Any

from app.occ import contour
from app.occ.loader import read_step_bytes


def _ensure_occ() -> None:
    try:
        from app.utils.occ_guard import occ_installed  # noqa: WPS433
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"pythonOCC 不可用: {exc}") from exc
    if not occ_installed():
        raise RuntimeError("pythonOCC 未安装，无法进行特征识别")


def analyze_face_spread(
    step_bytes: bytes,
    filename_hint: str,
    face_id: str,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    work_plane: str = "auto",
    work_plane_normal: list[float] | None = None,
    hole_diameter_min: float = 0.5,
    hole_diameter_max: float = 500.0,
    include_cylinder_holes: bool = False,
    closure_tol: float = 0.5,
    enhanced_params: bool = False,
) -> dict[str, Any]:
    """Read STEP bytes, run face-level analysis, return JSON-ready dict.

    Args:
        enhanced_params: If True, extract laser-software-style parameters
            (rotation_angle, corner_radius, compensation_length, etc.)
    """
    _ensure_occ()

    wp_normal = tuple(work_plane_normal) if work_plane_normal else None
    if work_plane in ("xy", "yz", "xz") and wp_normal is None:
        # bbox not known until shape is loaded; populate wp_normal after
        # read so callers can rely on the resulting key.
        pass

    shape = read_step_bytes(step_bytes, filename_hint)
    face, canonical_id = contour.find_face_by_id(shape, face_id)

    bbox = contour.shape_bbox_dict(shape)
    if work_plane == "auto" and wp_normal is None:
        wp_normal = contour.work_plane_normal_for_mode("auto", _auto_normal(bbox, face))

    result = contour.analyze_face(
        face,
        canonical_id,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        work_plane=work_plane,
        work_plane_normal=wp_normal,
        hole_diameter_min=hole_diameter_min,
        hole_diameter_max=hole_diameter_max,
        include_cylinder_holes=include_cylinder_holes,
        closure_tol=closure_tol,
        enhanced_params=enhanced_params,
    )
    result["model_bbox"] = bbox
    return result


def analyze_part_spread(
    step_bytes: bytes,
    filename_hint: str,
    part_id: str,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    work_plane: str = "auto",
    work_plane_normal: list[float] | None = None,
    hole_diameter_min: float = 0.5,
    hole_diameter_max: float = 500.0,
    include_cylinder_holes: bool = False,
    closure_tol: float = 0.5,
    enhanced_params: bool = False,
    target_face_id: str | None = None,
) -> dict[str, Any]:
    """Analyse a whole Solid (when the user selected an entire part instead
    of a single face). Returns a dict with ``per_face`` plus an aggregated
    ``contours`` / ``holes`` list.

    Args:
        enhanced_params: If True, extract laser-software-style parameters
            (rotation_angle, corner_radius, compensation_length, etc.)
        target_face_id: Optional face selector ``face_<n>`` within the
            same part. When provided, only features from faces whose
            outward normal is on the same side as the given face are
            returned. This is the "one-sided" mode the user asked for:
            it suppresses duplicate back-side features (mirrored
            circle / ellipse contours, fake outer boundary) that
            otherwise show up on a plate.
    """
    _ensure_occ()
    shape = read_step_bytes(step_bytes, filename_hint)

    bbox = contour.shape_bbox_dict(shape)
    if work_plane == "auto" and work_plane_normal is None:
        wp_normal = _auto_normal(bbox, None)
    else:
        wp_normal = tuple(work_plane_normal) if work_plane_normal else None

    result = contour.analyze_part(
        shape,
        str(part_id),
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        work_plane=work_plane,
        work_plane_normal=wp_normal,
        hole_diameter_min=hole_diameter_min,
        hole_diameter_max=hole_diameter_max,
        include_cylinder_holes=include_cylinder_holes,
        closure_tol=closure_tol,
        enhanced_params=enhanced_params,
        target_face_id=target_face_id,
    )
    return result


def _auto_normal(bbox: dict, face) -> tuple[float, float, float] | None:
    xmin, ymin, zmin = bbox["xmin"], bbox["ymin"], bbox["zmin"]
    xmax, ymax, zmax = bbox["xmax"], bbox["ymax"], bbox["zmax"]
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    # bbox-derived work plane first; fall back to face normal.
    if dz <= dx and dz <= dy:
        return (0.0, 0.0, 1.0)
    if dy <= dx:
        return (0.0, 1.0, 0.0)
    return (1.0, 0.0, 0.0)


def list_faces(step_bytes: bytes, filename_hint: str) -> list[dict]:
    """Return a lightweight face catalogue (id, surface_type, area, normal)."""
    _ensure_occ()
    shape = read_step_bytes(step_bytes, filename_hint)
    from app.occ.geometry_utils import face_surface_info, face_area as _face_area

    out: list[dict] = []
    for i, face in enumerate(contour.list_faces(shape)):
        info = face_surface_info(face)
        out.append({
            "id": f"face_{i}",
            "surface_type": info.get("surface_type", "other"),
            "area": _face_area(face) if info.get("surface_type") == "plane" else None,
            "normal": info.get("normal"),
            "axis": info.get("axis"),
            "center": info.get("center"),
            "radius": info.get("radius"),
        })
    return out


def list_parts(step_bytes: bytes, filename_hint: str) -> list[dict]:
    """Return a catalogue of solids (parts) in the model with their
    face-count and bounding box. Useful for letting the user pick which
    part to recognise."""
    _ensure_occ()
    from app.occ.geometry_utils import shape_bbox

    shape = read_step_bytes(step_bytes, filename_hint)
    solids = contour.list_solids(shape)
    if not solids:
        return []
    bbox = shape_bbox(shape)
    out: list[dict] = []
    for idx, solid in enumerate(solids):
        sb = shape_bbox(solid)
        out.append({
            "id": f"part_{idx}",
            "index": idx,
            "face_count": contour._face_count_in_solid(solid),
            "bbox": {
                "xmin": sb[0], "ymin": sb[1], "zmin": sb[2],
                "xmax": sb[3], "ymax": sb[4], "zmax": sb[5],
            },
        })
    return out
