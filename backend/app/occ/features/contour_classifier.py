"""2D contour shape classification: circle, slot, rectangle, hexagon, outer."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# 圆度阈值：越接近 1 越像圆
_CIRCULARITY_CIRCLE = 0.88
_CIRCULARITY_SLOT_MAX = 0.9
_SLOT_ASPECT_MIN = 1.75
_SLOT_CIRCULARITY_MIN = 0.72
_RECT_ASPECT_MAX = 1.6
# 圆角矩形离散后圆度常落在 0.65–0.85；低于此值则过扁/不规则，不强行标矩形
_RECT_CIRCULARITY_MIN = 0.55
_HEX_CORNER_TARGET = 6
_HEX_CORNER_TOL = 1
_MAX_NONPLANARITY_RATIO = 0.03


def classify_wire_contour(
    pts_3d: list[tuple[float, float, float]],
    *,
    face_normal: tuple[float, float, float],
    is_outer: bool,
    wire_id: str | None,
    polyline_id: str,
    face_id: str,
    contour_index: int,
    prefer_pca_plane: bool = False,
) -> dict[str, Any]:
    """Classify a wire loop and return ContourFeature-compatible dict."""
    cid = f"contour_{contour_index}"
    normal_hint = _normalize(face_normal)
    origin, u, v, proj_normal, nonplanarity = _projection_basis(
        pts_3d,
        normal_hint,
        prefer_pca_plane=prefer_pca_plane,
    )
    # API 输出的 contour.normal 始终为宿主面外法向（normal_hint），
    # 不用 PCA/投影基法向，避免圆弧/自由曲面上法向「贴着邻接平面」。
    del proj_normal  # 仅用于 2D 投影分类，不写入 JSON

    if is_outer:
        pts2d = np.array([_project2(p, origin, u, v) for p in pts_3d], dtype=np.float64)
        if len(pts2d) > 1 and _dist2d(pts2d[0], pts2d[-1]) < _loop_close_tol_2d(pts2d):
            pts2d = pts2d[:-1]
        obb_l, obb_w, _ = _obb_dimensions(pts2d) if len(pts2d) >= 2 else (0.0, 0.0, 0.0)
        return _build_contour(
            cid,
            "outer",
            pts_3d,
            normal_hint,
            wire_id,
            polyline_id,
            face_id,
            is_outer=True,
            parameters={"length": max(obb_l, obb_w), "width": min(obb_l, obb_w)},
            area=abs(_polygon_area(pts2d)) if len(pts2d) >= 3 else None,
            perimeter=_perimeter(pts2d) if len(pts2d) >= 2 else None,
        )

    if len(pts_3d) < 3:
        return _build_contour(
            cid,
            "unknown",
            pts_3d,
            normal_hint,
            wire_id,
            polyline_id,
            face_id,
            is_outer=False,
            parameters={},
        )

    pts2d = np.array([_project2(p, origin, u, v) for p in pts_3d], dtype=np.float64)
    if len(pts2d) > 1 and _dist2d(pts2d[0], pts2d[-1]) < _loop_close_tol_2d(pts2d):
        pts2d = pts2d[:-1]

    analysis = _analyze_loop_2d(pts2d)
    ctype = analysis["contour_type"]
    params = analysis["parameters"]
    if nonplanarity > _MAX_NONPLANARITY_RATIO:
        obb_l, obb_w, _ = _obb_dimensions(pts2d) if len(pts2d) >= 2 else (0.0, 0.0, 0.0)
        ctype = "unknown"
        params = {
            "diameter": None,
            "length": max(obb_l, obb_w),
            "width": min(obb_l, obb_w),
            "across_flats": None,
        }
    c2d = analysis.get("center_2d", (float(np.mean(pts2d[:, 0])), float(np.mean(pts2d[:, 1]))))
    center_3d = _lift2to3(c2d[0], c2d[1], origin, u, v)

    return _build_contour(
        cid,
        ctype,
        pts_3d,
        normal_hint,
        wire_id,
        polyline_id,
        face_id,
        is_outer=False,
        parameters=params,
        center_3d=center_3d,
        area=abs(_polygon_area(pts2d)),
        perimeter=_perimeter(pts2d),
    )


def _analyze_loop_2d(pts2d: np.ndarray) -> dict[str, Any]:
    area = abs(_polygon_area(pts2d))
    perim = _perimeter(pts2d)
    if area < 1e-8 or perim < 1e-8:
        return {"contour_type": "unknown", "parameters": {}}

    circularity = min(1.0, 4.0 * math.pi * area / (perim * perim))
    corners = _corner_count(pts2d)
    obb_l, obb_w, obb_angle = _obb_dimensions(pts2d)
    length = max(obb_l, obb_w)
    width = min(obb_l, obb_w) if min(obb_l, obb_w) > 1e-9 else max(obb_l, obb_w)
    aspect = length / width if width > 1e-9 else 1.0

    cx = float(np.mean(pts2d[:, 0]))
    cy = float(np.mean(pts2d[:, 1]))

    # 槽：细长 + 近似跑道形（避免被高圆度提前误判成圆）
    if aspect >= _SLOT_ASPECT_MIN and _SLOT_CIRCULARITY_MIN <= circularity <= _CIRCULARITY_SLOT_MAX:
        return {
            "contour_type": "slot",
            "parameters": {
                "diameter": None,
                "length": length,
                "width": width,
                "across_flats": None,
            },
            "center_2d": (cx, cy),
        }

    # 圆
    if circularity >= _CIRCULARITY_CIRCLE:
        diameter = 2.0 * math.sqrt(area / math.pi)
        return {
            "contour_type": "circle",
            "parameters": {"diameter": diameter, "length": None, "width": None, "across_flats": None},
            "center_2d": (cx, cy),
        }

    # 六边形：约 6 个拐角，且边长较均匀
    if _is_hexagon(pts2d, corners):
        af = _hex_across_flats(pts2d)
        return {
            "contour_type": "hexagon",
            "parameters": {
                "diameter": None,
                "length": None,
                "width": None,
                "across_flats": af,
            },
            "center_2d": (cx, cy),
        }

    # 矩形：直角或圆角矩形（OBB 非细长、圆度未达圆孔阈值）
    if _is_rectangle_like(corners, aspect, circularity):
        return {
            "contour_type": "rectangle",
            "parameters": {
                "diameter": None,
                "length": length,
                "width": width,
                "across_flats": None,
            },
            "center_2d": (cx, cy),
        }

    # 兜底：圆度较高但未达 strict 圆阈值的小圆孔
    if circularity >= 0.82:
        diameter = 2.0 * math.sqrt(area / math.pi)
        return {
            "contour_type": "circle",
            "parameters": {"diameter": diameter, "length": None, "width": None, "across_flats": None},
            "center_2d": (cx, cy),
        }

    return {
        "contour_type": "unknown",
        "parameters": {"diameter": None, "length": length, "width": width, "across_flats": None},
        "center_2d": (cx, cy),
    }


def _is_rectangle_like(corners: int, aspect: float, circularity: float) -> bool:
    """True when loop is a rectangle or rounded-rect (not slot/circle)."""
    if aspect > _RECT_ASPECT_MAX:
        return False
    if circularity >= _CIRCULARITY_CIRCLE:
        return False
    if circularity < _RECT_CIRCULARITY_MIN:
        return False
    if corners in (4, 5):
        return True
    # 圆角矩形：离散后拐角数常 > 6，但 OBB 仍近方形/矩形
    return True


def _lift2to3(
    x: float,
    y: float,
    origin: tuple[float, float, float],
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    p = np.array(origin) + x * np.array(u) + y * np.array(v)
    return (float(p[0]), float(p[1]), float(p[2]))


def _build_contour(
    cid: str,
    ctype: str,
    pts_3d: list,
    normal: tuple[float, float, float],
    wire_id: str | None,
    polyline_id: str,
    face_id: str,
    *,
    is_outer: bool,
    parameters: dict,
    center_3d: tuple[float, float, float] | None = None,
    area: float | None = None,
    perimeter: float | None = None,
) -> dict[str, Any]:
    if center_3d:
        center = center_3d
    elif pts_3d:
        center = (
            sum(p[0] for p in pts_3d) / len(pts_3d),
            sum(p[1] for p in pts_3d) / len(pts_3d),
            sum(p[2] for p in pts_3d) / len(pts_3d),
        )
    else:
        center = (0.0, 0.0, 0.0)

    return {
        "id": cid,
        "contour_type": ctype,
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "normal": {"x": normal[0], "y": normal[1], "z": normal[2]},
        "polyline_id": polyline_id,
        "wire_id": wire_id,
        "face_id": face_id,
        "is_outer": is_outer,
        "parameters": {
            "diameter": parameters.get("diameter"),
            "length": parameters.get("length"),
            "width": parameters.get("width"),
            "across_flats": parameters.get("across_flats"),
        },
        "area": area,
        "perimeter": perimeter,
    }


# NOTE: This module deliberately avoids numpy BLAS/LAPACK routines
# (``np.dot`` / ``@`` / ``np.linalg.*``). On some Windows + conda OCC
# environments the bundled BLAS/LAPACK DLL is mismatched and any such call
# crashes the whole process with ``0xC06D007F: Procedure not found`` (the same
# error that takes down feature extraction). Element-wise ops (``*``, ``np.sum``,
# ``np.cross``, ``np.sqrt``) do not hit BLAS and are safe.


def _vdot(a, b) -> float:
    """BLAS-free dot product of two vectors."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.sum(aa * bb))


def _vnorm(a) -> float:
    """BLAS-free Euclidean norm."""
    aa = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.sum(aa * aa)))


def _jacobi_eigh_3x3(
    a: list[list[float]],
) -> tuple[list[float], list[list[float]]]:
    """Symmetric 3x3 eigendecomposition via Jacobi rotations (no LAPACK).

    Returns (eigenvalues, eigenvectors) where ``eigenvectors[i][k]`` is the
    i-th component of the k-th eigenvector (eigenvectors stored as columns).
    """
    A = [list(map(float, row)) for row in a]
    V = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(50):
        # Largest off-diagonal magnitude (upper triangle).
        p, q, off = 0, 1, abs(A[0][1])
        if abs(A[0][2]) > off:
            p, q, off = 0, 2, abs(A[0][2])
        if abs(A[1][2]) > off:
            p, q, off = 1, 2, abs(A[1][2])
        if off < 1e-18:
            break
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        theta = (aqq - app) / (2.0 * apq)
        sign = 1.0 if theta >= 0.0 else -1.0
        t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c
        for i in range(3):
            aip, aiq = A[i][p], A[i][q]
            A[i][p] = c * aip - s * aiq
            A[i][q] = s * aip + c * aiq
        for i in range(3):
            api, aqi = A[p][i], A[q][i]
            A[p][i] = c * api - s * aqi
            A[q][i] = s * api + c * aqi
        for i in range(3):
            vip, viq = V[i][p], V[i][q]
            V[i][p] = c * vip - s * viq
            V[i][q] = s * vip + c * viq
    eigvals = [A[0][0], A[1][1], A[2][2]]
    return eigvals, V


def _plane_basis(
    pts_3d: list[tuple[float, float, float]],
    normal: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    n = np.array(normal, dtype=np.float64)
    n = n / (_vnorm(n) + 1e-15)
    origin = np.array(
        [
            sum(p[0] for p in pts_3d) / len(pts_3d),
            sum(p[1] for p in pts_3d) / len(pts_3d),
            sum(p[2] for p in pts_3d) / len(pts_3d),
        ],
        dtype=np.float64,
    )
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u = u / (_vnorm(u) + 1e-15)
    v = np.cross(n, u)
    v = v / (_vnorm(v) + 1e-15)
    return tuple(origin), tuple(u), tuple(v)


def _projection_basis(
    pts_3d: list[tuple[float, float, float]],
    face_normal: tuple[float, float, float],
    *,
    prefer_pca_plane: bool,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
]:
    """Choose a stable 2D projection basis and return non-planarity ratio."""
    if len(pts_3d) < 3:
        origin, u, v = _plane_basis(pts_3d or [(0.0, 0.0, 0.0)], face_normal)
        return origin, u, v, face_normal, 0.0

    pca = _pca_plane_basis(pts_3d, face_normal)
    if pca is None:
        origin, u, v = _plane_basis(pts_3d, face_normal)
        return origin, u, v, face_normal, 0.0

    origin, u, v, pca_normal, nonplanarity = pca
    if prefer_pca_plane:
        return origin, u, v, pca_normal, nonplanarity

    # For near-planar loops, keep face normal for consistency across faces.
    if nonplanarity <= _MAX_NONPLANARITY_RATIO:
        origin_face, u_face, v_face = _plane_basis(pts_3d, face_normal)
        return origin_face, u_face, v_face, face_normal, nonplanarity

    return origin, u, v, pca_normal, nonplanarity


def _pca_plane_basis(
    pts_3d: list[tuple[float, float, float]],
    face_normal: tuple[float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
] | None:
    coords = np.asarray(pts_3d, dtype=np.float64)
    if coords.shape[0] < 3:
        return None

    origin = np.mean(coords, axis=0)
    centered = coords - origin
    cx, cy, cz = centered[:, 0], centered[:, 1], centered[:, 2]
    denom = max(1, centered.shape[0] - 1)
    # 3x3 covariance via element-wise reductions (no BLAS / np.dot).
    sxx = float(np.sum(cx * cx)) / denom
    syy = float(np.sum(cy * cy)) / denom
    szz = float(np.sum(cz * cz)) / denom
    sxy = float(np.sum(cx * cy)) / denom
    sxz = float(np.sum(cx * cz)) / denom
    syz = float(np.sum(cy * cz)) / denom
    cov = [[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]]

    eigvals_list, eigvecs = _jacobi_eigh_3x3(cov)
    order = sorted(range(3), key=lambda k: eigvals_list[k])
    smallest, largest = order[0], order[-1]
    normal = np.array([eigvecs[i][smallest] for i in range(3)], dtype=np.float64)
    tangent = np.array([eigvecs[i][largest] for i in range(3)], dtype=np.float64)

    if _vnorm(normal) < 1e-12 or _vnorm(tangent) < 1e-12:
        return None

    face_n = np.asarray(face_normal, dtype=np.float64)
    if _vdot(normal, face_n) < 0.0:
        normal = -normal

    normal = normal / (_vnorm(normal) + 1e-15)
    tangent = tangent / (_vnorm(tangent) + 1e-15)
    bitangent = np.cross(normal, tangent)
    bitangent = bitangent / (_vnorm(bitangent) + 1e-15)

    signed_dist = np.sum(centered * normal, axis=1)
    rms = float(np.sqrt(np.mean(np.square(signed_dist))))
    span = float(np.sqrt(max(eigvals_list[largest], 1e-15)))
    nonplanarity = rms / (span + 1e-15)

    return (
        (float(origin[0]), float(origin[1]), float(origin[2])),
        (float(tangent[0]), float(tangent[1]), float(tangent[2])),
        (float(bitangent[0]), float(bitangent[1]), float(bitangent[2])),
        (float(normal[0]), float(normal[1]), float(normal[2])),
        nonplanarity,
    )


def _project2(
    p: tuple[float, float, float],
    origin: tuple[float, float, float],
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> tuple[float, float]:
    d = np.array(p) - np.array(origin)
    return (_vdot(d, u), _vdot(d, v))


def _normalize(n: tuple[float, float, float]) -> tuple[float, float, float]:
    L = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    if L < 1e-12:
        return (0.0, 0.0, 1.0)
    return (n[0] / L, n[1] / L, n[2] / L)


def _polygon_area(pts: np.ndarray) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    # Shoelace via element-wise products (no BLAS / np.dot).
    cross = float(np.sum(x * np.roll(y, -1)) - np.sum(y * np.roll(x, -1)))
    return 0.5 * abs(cross)


def _perimeter(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        total += _dist2d(pts[i], pts[j])
    return total


def _corner_count(pts: np.ndarray, angle_thresh_deg: float = 28.0) -> int:
    simp = _simplify_collinear(pts)
    n = len(simp)
    if n < 3:
        return n
    thresh = math.radians(angle_thresh_deg)
    count = 0
    for i in range(n):
        a = simp[(i - 1) % n]
        b = simp[i]
        c = simp[(i + 1) % n]
        v1 = a - b
        v2 = c - b
        n1 = math.hypot(float(v1[0]), float(v1[1]))
        n2 = math.hypot(float(v2[0]), float(v2[1]))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = max(-1.0, min(1.0, _vdot(v1, v2) / (n1 * n2)))
        angle = math.acos(cos_a)
        if angle > thresh:
            count += 1
    return count


def _simplify_collinear(pts: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if cross > tol:
            out.append(b)
    out.append(pts[-1])
    return np.array(out)


def _obb_dimensions(pts: np.ndarray) -> tuple[float, float, float]:
    """最小外接矩形长宽（纯 Python，避免 OCC 进程内调用 numpy.linalg）。"""
    n = len(pts)
    if n < 2:
        return 0.0, 0.0, 0.0
    coords = [(float(pts[i, 0]), float(pts[i, 1])) for i in range(n)]
    best_l, best_w, best_angle = 0.0, 0.0, 0.0
    best_area = float("inf")
    for k in range(18):
        angle = math.pi * k / 18.0
        ca, sa = math.cos(angle), math.sin(angle)
        xs: list[float] = []
        ys: list[float] = []
        for x, y in coords:
            xs.append(x * ca + y * sa)
            ys.append(-x * sa + y * ca)
        length = max(xs) - min(xs)
        width = max(ys) - min(ys)
        area = length * width
        if area < best_area:
            best_area = area
            best_l, best_w = length, width
            best_angle = angle
    return best_l, best_w, best_angle


def _is_hexagon(pts: np.ndarray, corners: int) -> bool:
    if not (6 - _HEX_CORNER_TOL <= corners <= 6 + _HEX_CORNER_TOL):
        return False
    # 边长均匀性
    simp = _simplify_collinear(pts)
    n = len(simp)
    if n < 6:
        return False
    lengths = []
    for i in range(n):
        d = simp[(i + 1) % n] - simp[i]
        lengths.append(math.hypot(float(d[0]), float(d[1])))
    if not lengths:
        return False
    mean_l = sum(lengths) / len(lengths)
    if mean_l < 1e-9:
        return False
    var = sum((L - mean_l) ** 2 for L in lengths) / len(lengths)
    return (var**0.5) / mean_l < 0.25


def _hex_across_flats(pts: np.ndarray) -> float:
    _, w, _ = _obb_dimensions(pts)
    return float(w)


def _dist2d(a, b) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.hypot(dx, dy)


def _loop_close_tol_2d(pts2d: np.ndarray) -> float:
    if len(pts2d) == 0:
        return 1e-6
    x_span = float(np.max(pts2d[:, 0]) - np.min(pts2d[:, 0]))
    y_span = float(np.max(pts2d[:, 1]) - np.min(pts2d[:, 1]))
    scale = max(x_span, y_span, 1.0)
    return max(1e-6, scale * 1e-6)
