"""Classifier registry — wires enter here and fall through to the first matching shape.

Mirrors the SmartLaser ``AnalyzeLS`` / ``BuildPathLS`` design:
the registry is the single dispatch point, and each concrete
classifier is tried in priority order until one claims the input
(``matches`` returns True). If none claim it the ``unknown`` bucket is
returned, which replaces the legacy ``irregular`` classification.

Priority order (mirrors the natural nesting of shapes):
  1. circle   (priority 100) — most round, most common
  2. slot     (priority 90)
  3. rectangle (priority 80)
  4. hexagon  (priority 70)
"""

from __future__ import annotations

from typing import Any

from .base import (
    ContourMetrics,
    compute_metrics,
    unknown_shape_result,
)
from .circle import CircleClassifier
from .hexagon import HexagonClassifier
from .rectangle import RectangleClassifier
from .slot import SlotClassifier


class ClassifierRegistry:
    """Single dispatch point for all feature recognition.

    Use :meth:`classify` as the public API (called by the rewritten
    ``classify_wire_contour`` in ``contour.py``).
    """

    __slots__ = ()

    _priority = ["circle", "slot", "rectangle", "hexagon"]

    _classifiers: dict[str, Any] = {
        "circle": CircleClassifier(),
        "slot": SlotClassifier(),
        "rectangle": RectangleClassifier(),
        "hexagon": HexagonClassifier(),
    }

    def classify(
        self,
        pts2d: list[tuple[float, float]],
        *,
        face_normal: tuple[float, float, float] | None = None,
        pts_world: list[tuple[float, float, float]] | None = None,
    ) -> dict[str, Any]:
        """Classify a closed 2D polyline and return the standard result dict.

        Returns a dict with the full contour entry schema
        (``contour_type``, ``parameters``, ``area``, ``perimeter``,
        ``center``, ``normal``, ``_confidence``).
        If no classifier claims the wire, returns the ``unknown`` bucket.
        """
        m = compute_metrics(pts2d)

        for name in self._priority:
            clf = self._classifiers[name]
            if clf.matches(m):
                params = clf.classify(
                    m,
                    face_normal=face_normal,
                    pts_world=pts_world,
                )
                confidence = params.pop("_confidence", 0.5)
                return clf.build_result(
                    m,
                    face_normal=face_normal,
                    params=params,
                    confidence=confidence,
                    pts_world=pts_world or [],
                )

        return unknown_shape_result(
            m,
            face_normal=face_normal,
            pts_world=pts_world or [],
        )
