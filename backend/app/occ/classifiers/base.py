"""Shape classifier base class.

Architecture mirrors the SmartLaser ``MachiningPath`` design:
``MachiningPath`` is the abstract base, and
``MachiningCirclePath`` / ``MachiningSlotPath`` /
``MachiningRectanglePath`` / ``MachiningHexagonPath`` are concrete
sub-classes, each with its own shape-specific parameters and
serialisation (``*_STANDARD_STR``).

Here each shape classifier follows the same idea:

* declare a stable ``shape_name`` (the ``contour_type`` token);
* expose a fast ``matches`` gate (cheap geometric test);
* implement a thorough ``classify`` that extracts shape-specific
  parameters (D / L / W / RA / CR ...) and a confidence score.

The registry composes them in priority order so that ambiguous inputs
fall through to the first classifier that claims them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


# ── Shared geometry / metrics bundle ─────────────────────────────────────────


@dataclass(slots=True)
class ContourMetrics:
    """Pre-computed geometric measurements of a closed polyline.

    The classifier should consume this rather than re-deriving the same
    values; this is the equivalent of ``MachiningPath.pathLength``,
    ``pathTopodsShape``, etc. that every sub-class can rely on.
    """

    pts2d: list[tuple[float, float]]
    n: int
    width: float
    height: float
    length: float
    short_side: float
    aspect: float
    area_2d: float
    perimeter: float
    circularity: float
    centroid_2d: tuple[float, float]            # 2D shoelace centroid
    bbox: tuple[float, float, float, float]     # xmin, ymin, xmax, ymax


# ── Helpers used by the base class ───────────────────────────────────────────


def _shoelace_area(pts2d: list[tuple[float, float]]) -> float:
    """Signed shoelace area of a closed polyline (returns ``abs`` value)."""
    n = len(pts2d)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _shoelace_signed_area(pts2d: list[tuple[float, float]]) -> float:
    """Signed shoelace area (sign = polygon orientation)."""
    n = len(pts2d)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s * 0.5


def _shoelace_centroid(
    pts2d: list[tuple[float, float]],
    signed_area: float,
) -> tuple[float, float]:
    """Shoelace centroid. Falls back to bbox centre on degenerate input."""
    n = len(pts2d)
    if n < 3 or abs(signed_area) < 1e-9:
        xs = [p[0] for p in pts2d]
        ys = [p[1] for p in pts2d]
        if not xs:
            return 0.0, 0.0
        return (min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5
    cx = cy = 0.0
    a6 = 6.0 * signed_area
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i + 1) % n]
        factor = x1 * y2 - x2 * y1
        cx += (x1 + x2) * factor
        cy += (y1 + y2) * factor
    return cx / a6, cy / a6


def compute_metrics(pts2d: list[tuple[float, float]]) -> ContourMetrics:
    """Build a :class:`ContourMetrics` from a closed 2D polyline.

    Cheap O(n) and shared by every classifier — equivalent to
    ``MachiningPath.UpdateCAMLine`` that every sub-class reuses.
    """
    n = len(pts2d)
    if n < 3:
        return ContourMetrics(
            pts2d=pts2d, n=n, width=0.0, height=0.0, length=0.0,
            short_side=0.0, aspect=1.0, area_2d=0.0, perimeter=0.0,
            circularity=0.0, centroid_2d=(0.0, 0.0),
            bbox=(0.0, 0.0, 0.0, 0.0),
        )
    xs = [p[0] for p in pts2d]
    ys = [p[1] for p in pts2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = xmax - xmin
    height = ymax - ymin
    length = max(width, height)
    short_side = min(width, height)
    aspect = length / max(short_side, 1e-9)
    area_2d = _shoelace_area(pts2d)
    signed = _shoelace_signed_area(pts2d)
    perimeter = 0.0
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i + 1) % n]
        perimeter += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    circularity = (4 * 3.141592653589793 * area_2d) / (perimeter * perimeter) if perimeter > 0 else 0.0
    cx, cy = _shoelace_centroid(pts2d, signed)
    return ContourMetrics(
        pts2d=pts2d, n=n, width=width, height=height, length=length,
        short_side=short_side, aspect=aspect, area_2d=area_2d,
        perimeter=perimeter, circularity=circularity, centroid_2d=(cx, cy),
        bbox=(xmin, ymin, xmax, ymax),
    )


def _3d_centroid(pts_world: list[tuple[float, float, float]]) -> list[float] | None:
    """3D centroid that always lies on the face plane (the wire is planar).

    Same trick used by the original ``analyze_face`` to avoid
    ``axis_mean``-style shortcuts that break for tilted planar faces.
    """
    if not pts_world:
        return None
    cx = sum(p[0] for p in pts_world) / len(pts_world)
    cy = sum(p[1] for p in pts_world) / len(pts_world)
    cz = sum(p[2] for p in pts_world) / len(pts_world)
    return [float(cx), float(cy), float(cz)]


# ── Abstract base ────────────────────────────────────────────────────────────


class ShapeClassifier(ABC):
    """Base class for the four shape classifiers (mirror of MachiningPath).

    Each concrete classifier owns its own parameters (D, L, W, RA, CR, ...).
    The base only owns the *common* output contract:

    * ``shape_name`` → ``contour_type`` token in the response
    * ``priority``  → ordering used by the registry
    * ``matches``   → cheap gate
    * ``classify``  → returns the dict to merge into the contour result
    """

    shape_name: ClassVar[str] = ""
    priority: ClassVar[int] = 0
    min_points: ClassVar[int] = 4

    # ── Public contract ───────────────────────────────────────────────────

    @abstractmethod
    def matches(self, m: ContourMetrics) -> bool:
        """Return True if ``m`` looks like this shape.

        Must be cheap (single pass over metrics is fine; no per-point
        iteration on the full polyline). The registry uses it to skip
        classifiers that don't apply.
        """

    @abstractmethod
    def classify(
        self,
        m: ContourMetrics,
        *,
        face_normal: tuple[float, float, float] | None,
    ) -> dict[str, Any]:
        """Return the ``parameters`` dict for this shape.

        Mirrors the parameter set on the SmartLaser ``*_STANDARD_STR``
        attributes (D / L / W / RA / CR / VL / VW / OD …) and includes a
        ``_confidence`` key the registry uses to break ties.
        """

    # ── Shared output envelope ───────────────────────────────────────────

    def build_result(
        self,
        m: ContourMetrics,
        *,
        face_normal: tuple[float, float, float] | None,
        params: dict[str, float | None],
        confidence: float,
        pts_world: list[tuple[float, float, float]],
    ) -> dict[str, Any]:
        """Build the ``contours[i]`` entry that gets returned to the client.

        The schema is **unchanged** so the existing API (FeaturePanel.vue,
        ``test_feature.py``, contour_enhanced.py) keeps working.
        """
        centroid_3d = _3d_centroid(pts_world)
        normal_3d: list[float] | None = None
        if face_normal is not None:
            normal_3d = [float(face_normal[0]), float(face_normal[1]), float(face_normal[2])]
        return {
            "contour_type": self.shape_name,
            "parameters": _with_defaults(params),
            "area": round(m.area_2d, 4),
            "perimeter": round(m.perimeter, 4),
            "center": centroid_3d,
            "normal": normal_3d,
            "_confidence": float(confidence),
        }


def _with_defaults(params: dict[str, float | None]) -> dict[str, float | None]:
    """Backfill the standard parameter keys with ``None``.

    This preserves the contract that every ``contours[i].parameters``
    object has the same set of well-known keys, which the frontend
    (FeaturePanel.vue) and contour_enhanced.py both rely on.
    """
    out = {
        "diameter": None,
        "length": None,
        "width": None,
        "across_flats": None,
    }
    for k, v in params.items():
        out[k] = v
    return out


def unknown_shape_result(
    m: ContourMetrics,
    *,
    face_normal: tuple[float, float, float] | None,
    pts_world: list[tuple[float, float, float]],
) -> dict[str, Any]:
    """Standard ``unknown`` bucket.

    Distinct from ``irregular`` in the legacy code: this is the
    explicit "I don't know what this is" return used when **no**
    classifier claims the input. Used for degenerate inputs
    (n < min_points, NaN, zero area) and shapes that fall through
    every classifier.
    """
    centroid_3d = _3d_centroid(pts_world)
    normal_3d: list[float] | None = None
    if face_normal is not None:
        normal_3d = [float(face_normal[0]), float(face_normal[1]), float(face_normal[2])]
    return {
        "contour_type": "unknown",
        "parameters": {
            "diameter": None,
            "length": None,
            "width": None,
            "across_flats": None,
        },
        "area": round(m.area_2d, 4) if m.n >= 3 else None,
        "perimeter": round(m.perimeter, 4) if m.n >= 3 else None,
        "center": centroid_3d,
        "normal": normal_3d,
        "_confidence": 0.0,
    }
