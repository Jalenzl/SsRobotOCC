"""Irregular hole classifier — fallback bucket.

Mirrors the legacy ``irregular`` return value: when no other classifier
(circle / slot / rectangle / hexagon) claims the input, this classifier
absorbs it and emits ``contour_type='irregular'`` so the frontend
displays a single "异形孔" row instead of "未识别".
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import ContourMetrics, ShapeClassifier


class IrregularClassifier(ShapeClassifier):
    """Fallback bucket. Always claims whatever fell through."""

    shape_name: ClassVar[str] = "irregular"
    priority: ClassVar[int] = 0  # lowest — last in the chain
    min_points: ClassVar[int] = 0

    def matches(self, m: ContourMetrics) -> bool:
        # The registry stops at the first match; we register last, so
        # reaching this classifier means none of the shape-specific ones
        # claimed the wire. Always return True.
        return True

    def classify(
        self,
        m: ContourMetrics,
        *,
        face_normal: tuple[float, float, float] | None,
        pts_world: list[tuple[float, float, float]] | None = None,
    ) -> dict[str, Any]:
        # No shape-specific parameters for an irregular hole.
        return {
            "diameter": None,
            "length": None,
            "width": None,
            "across_flats": None,
            "_confidence": 0.0,
        }
