"""Enhanced contour analysis with laser-software-inspired parameter extraction.

This module extends contour.py with:
- Geometric parameter extraction (RA, CR, VL, VW, etc.)
- Parameter validation rules
- Improved classification confidence scoring
- Compensation parameter models

Inspired by the SmartLaser CAM architecture (MachiningCirclePath,
MachiningRectanglePath, etc.).
"""

from __future__ import annotations

import math
from typing import Any


# ── Parameter extraction thresholds ─────────────────────────────────────────────

# Lead-in validation: lead_length < diameter/2 for circles
_LEAD_DIAMETER_RATIO = 0.4

# Rectangle validation: CR < width/2
_CORNER_RADIUS_MAX_RATIO = 0.45

# Minimum confidence to accept a classification
_MIN_CLASSIFICATION_CONFIDENCE = 0.6

# Rotation angle tolerance (degrees) for aligned rectangles
_ROTATION_ANGLE_TOL = 5.0


# ── Geometric parameter extraction ────────────────────────────────────────────────

def extract_contour_parameters(
    pts2d: list[tuple[float, float]],
    contour_type: str,
    circularity: float,
    perimeter: float,
) -> dict[str, float | None]:
    """Extract detailed geometric parameters based on contour type.

    Args:
        pts2d: 2D projected points
        contour_type: Classification result
        circularity: Circularity ratio (4πA/P²)
        perimeter: Total perimeter length

    Returns:
        Dictionary with extracted parameters matching SmartLaser convention:
        - diameter: For circles (mm)
        - length: Long side (mm)
        - width: Short side (mm)
        - across_flats: For hexagons (mm)
        - rotation_angle (RA): Rotation from axis-aligned (degrees)
        - corner_radius (CR): Fillet radius for rectangles (mm)
        - compensation_length (VL): Length compensation (mm)
        - compensation_width (VW): Width compensation (mm)
        - overlap_distance (OD): Cutting overlap for rectangles (mm)
    """
    n = len(pts2d)
    if n < 3:
        return _empty_params()

    # Bounding box
    xs = [p[0] for p in pts2d]
    ys = [p[1] for p in pts2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = xmax - xmin
    height = ymax - ymin
    length = max(width, height)
    short_side = min(width, height)

    # Centroid
    area = _shoelace_area(pts2d)
    cx, cy = _shoelace_centroid(pts2d, area)

    params: dict[str, float | None] = {
        "diameter": None,
        "length": None,
        "width": None,
        "across_flats": None,
        "rotation_angle": None,
        "corner_radius": None,
        "compensation_length": None,
        "compensation_width": None,
        "overlap_distance": None,
    }

    if contour_type == "circle":
        # Diameter from area: D = 2 * sqrt(A/π)
        diameter = 2.0 * math.sqrt(abs(area) / math.pi)
        params["diameter"] = round(diameter, 4)

    elif contour_type == "rectangle":
        params["length"] = round(length, 4)
        params["width"] = round(short_side, 4)

        # Extract rotation angle
        ra = _estimate_rotation_angle(pts2d, cx, cy)
        params["rotation_angle"] = round(ra, 4)

        # Extract corner radius (fillet)
        cr = _estimate_corner_radius(pts2d, length, short_side)
        params["corner_radius"] = round(cr, 4) if cr > 0.01 else None

        # Initialize compensation parameters
        params["compensation_length"] = 0.0
        params["compensation_width"] = 0.0
        params["overlap_distance"] = 1.0  # Default OD value

    elif contour_type == "slot":
        params["length"] = round(length, 4)
        params["width"] = round(short_side, 4)

        # Rotation angle for slots
        ra = _estimate_rotation_angle(pts2d, cx, cy)
        params["rotation_angle"] = round(ra, 4)

        # Compensation parameters
        params["compensation_length"] = 0.0
        params["compensation_width"] = 0.0

    elif contour_type == "hexagon":
        # Across-flats for hexagon
        side = _hex_side_length(pts2d)
        params["across_flats"] = round(side * math.sqrt(3), 4)  # Distance across flats
        params["length"] = round(side * 2, 4)  # Circum diameter
        params["rotation_angle"] = round(_estimate_rotation_angle(pts2d, cx, cy), 4)

    elif contour_type == "outer":
        params["length"] = round(length, 4)
        params["width"] = round(short_side, 4)
        params["rotation_angle"] = round(_estimate_rotation_angle(pts2d, cx, cy), 4)

    elif contour_type == "irregular":
        params["length"] = round(length, 4)
        params["width"] = round(short_side, 4)

    return params


def _shoelace_area(pts2d: list[tuple[float, float]]) -> float:
    """Calculate signed area using shoelace formula."""
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
    area: float,
) -> tuple[float, float]:
    """Calculate centroid using shoelace formula."""
    n = len(pts2d)
    if n < 3 or abs(area) < 1e-9:
        # Fallback to bounding box center
        xs = [p[0] for p in pts2d]
        ys = [p[1] for p in pts2d]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    cx = cy = 0.0
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i + 1) % n]
        factor = x1 * y2 - x2 * y1
        cx += (x1 + x2) * factor
        cy += (y1 + y2) * factor

    cx /= (6 * area)
    cy /= (6 * area)
    return cx, cy


def _estimate_rotation_angle(
    pts2d: list[tuple[float, float]],
    cx: float,
    cy: float,
) -> float:
    """Estimate the rotation angle of a shape relative to axis-aligned.

    Returns angle in degrees (0-90), where:
    - 0° = axis-aligned (length along X-axis)
    - 45° = rotated 45°
    """
    # Find the point farthest from centroid
    max_dist = 0.0
    farthest_angle = 0.0
    for p in pts2d:
        dx = p[0] - cx
        dy = p[1] - cy
        dist = math.hypot(dx, dy)
        if dist > max_dist:
            max_dist = dist
            # Angle from centroid to point
            farthest_angle = math.degrees(math.atan2(dy, dx))

    # Normalize to 0-90 range (rectangles are symmetric)
    angle = abs(farthest_angle % 90)
    if angle > 45:
        angle = 90 - angle

    return angle


def _estimate_corner_radius(
    pts2d: list[tuple[float, float]],
    length: float,
    width: float,
) -> float:
    """Estimate corner radius (fillet) for a rectangle-like shape.

    Uses the deviation from sharp corners to estimate fillet radius.
    """
    if len(pts2d) < 4:
        return 0.0

    # Find corners using turning angle
    corners = _find_corners(pts2d, k=4)
    if len(corners) != 4:
        return 0.0

    # Estimate corner radius from interior angle deviation
    corner_radii = []
    for corner_idx in corners:
        prev_idx = (corner_idx - 1) % len(pts2d)
        next_idx = (corner_idx + 1) % len(pts2d)

        # Vectors from corner
        v1x = pts2d[prev_idx][0] - pts2d[corner_idx][0]
        v1y = pts2d[prev_idx][1] - pts2d[corner_idx][1]
        v2x = pts2d[next_idx][0] - pts2d[corner_idx][0]
        v2y = pts2d[next_idx][1] - pts2d[corner_idx][1]

        len1 = math.hypot(v1x, v1y)
        len2 = math.hypot(v2x, v2y)
        if len1 < 1e-6 or len2 < 1e-6:
            continue

        # Interior angle
        cos_angle = (v1x * v2x + v1y * v2y) / (len1 * len2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        interior_angle = math.degrees(math.acos(cos_angle))

        # Corner radius approximation
        # For a rectangle, interior_angle = 90°, CR = 0
        # For a rounded rectangle, interior_angle > 90°
        if interior_angle > 91:  # Fillet detected
            # Approximate CR from angle difference
            # For small fillets, CR ≈ (angle_diff / 90) * min(width, length) / 2
            angle_diff = interior_angle - 90
            estimated_cr = (angle_diff / 90) * min(width, length) * 0.3
            corner_radii.append(estimated_cr)

    if corner_radii:
        return sum(corner_radii) / len(corner_radii)
    return 0.0


def _find_corners(
    pts2d: list[tuple[float, float]],
    k: int,
) -> list[int]:
    """Find k dominant corner indices using turning angle."""
    n = len(pts2d)
    if n < k * 2:
        return []

    # Calculate turning angle at each point
    angles = []
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        v1x = pts2d[i][0] - pts2d[prev_idx][0]
        v1y = pts2d[i][1] - pts2d[prev_idx][1]
        v2x = pts2d[next_idx][0] - pts2d[i][0]
        v2y = pts2d[next_idx][1] - pts2d[i][1]

        len1 = math.hypot(v1x, v1y)
        len2 = math.hypot(v2x, v2y)
        if len1 < 1e-6 or len2 < 1e-6:
            angles.append(0.0)
            continue

        cos_val = (v1x * v2x + v1y * v2y) / (len1 * len2)
        cos_val = max(-1.0, min(1.0, cos_val))
        angle = math.degrees(math.acos(cos_val))
        angles.append(angle)

    # Find k points with largest turning angles
    indexed_angles = [(angles[i], i) for i in range(n)]
    indexed_angles.sort(reverse=True)

    # Ensure corners are reasonably spread apart
    corners = []
    min_gap = n // (k * 2)
    for _, idx in indexed_angles:
        if not corners:
            corners.append(idx)
        else:
            # Check minimum gap from existing corners
            gaps = [abs(idx - c) for c in corners]
            gaps = [min(g, n - g) for g in gaps]  # Handle wrap-around
            if min(gaps) >= min_gap:
                corners.append(idx)
        if len(corners) >= k:
            break

    corners.sort()
    return corners


def _hex_side_length(pts2d: list[tuple[float, float]]) -> float:
    """Estimate side length of a hexagon from its vertices."""
    corners = _find_corners(pts2d, k=6)
    if len(corners) < 6:
        return 0.0

    # Calculate average edge length
    total_length = 0.0
    for i in range(6):
        p1 = pts2d[corners[i]]
        p2 = pts2d[corners[(i + 1) % 6]]
        total_length += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    return total_length / 6


def _empty_params() -> dict[str, float | None]:
    """Return empty parameter dictionary."""
    return {
        "diameter": None,
        "length": None,
        "width": None,
        "across_flats": None,
        "rotation_angle": None,
        "corner_radius": None,
        "compensation_length": None,
        "compensation_width": None,
        "overlap_distance": None,
    }


# ── Parameter validation ─────────────────────────────────────────────────────────

def validate_contour_parameters(
    params: dict[str, float | None],
    contour_type: str,
) -> tuple[bool, str | None]:
    """Validate extracted parameters against type-specific rules.

    Based on SmartLaser parameter validation rules:
    - Circle: D > 0, Lead < D
    - Slot: L > W, Lead < W
    - Rectangle: L >= W, Lead < W, CR < W/2
    - Hexagon: L > 0, Lead < L

    Args:
        params: Parameter dictionary from extract_contour_parameters
        contour_type: The contour type classification

    Returns:
        Tuple of (is_valid, error_message)
    """
    if contour_type == "circle":
        d = params.get("diameter") or 0
        if d <= 0:
            return False, "Circle diameter must be positive"
        # Lead validation would require lead parameter
        return True, None

    elif contour_type == "slot":
        l = params.get("length") or 0
        w = params.get("width") or 0
        if l <= w:
            return False, "Slot length must be greater than width"
        if w <= 0:
            return False, "Slot width must be positive"
        return True, None

    elif contour_type == "rectangle":
        l = params.get("length") or 0
        w = params.get("width") or 0
        if l < w:
            return False, "Rectangle length must be >= width"
        if w <= 0:
            return False, "Rectangle width must be positive"
        cr = params.get("corner_radius") or 0
        if cr >= w * _CORNER_RADIUS_MAX_RATIO:
            return False, f"Corner radius ({cr}) too large for width ({w})"
        return True, None

    elif contour_type == "hexagon":
        l = params.get("length") or 0
        if l <= 0:
            return False, "Hexagon size must be positive"
        return True, None

    elif contour_type == "outer":
        # Outer boundary validation is more lenient
        l = params.get("length") or 0
        w = params.get("width") or 0
        if l <= 0:
            return False, "Outer contour must have positive length"
        return True, None

    return True, None


# ── Classification confidence scoring ───────────────────────────────────────────

def calculate_classification_confidence(
    pts2d: list[tuple[float, float]],
    circularity: float,
    contour_type: str,
    params: dict[str, float | None],
) -> float:
    """Calculate confidence score for a classification result.

    Higher score = more confident classification.
    Based on geometric consistency checks.
    """
    n = len(pts2d)
    if n < 4:
        return 0.3

    confidence = 0.5  # Base confidence

    if contour_type == "circle":
        # High circularity = confident
        confidence = circularity

        # Check if points are uniformly distributed
        radii = _point_radii(pts2d)
        if radii:
            radius_std = _standard_deviation(radii)
            radius_mean = sum(radii) / len(radii)
            if radius_mean > 0:
                # Lower relative std = higher confidence
                rel_std = radius_std / radius_mean
                confidence = max(0.5, circularity - rel_std)

    elif contour_type == "rectangle":
        # Check for 4 sharp corners
        corners = _find_corners(pts2d, k=4)
        corner_score = len(corners) / 4.0

        # Check edge alignment
        edge_score = _edge_alignment_score(pts2d)

        confidence = 0.4 + 0.3 * corner_score + 0.3 * edge_score

        # Penalize if rotation is significant but corners are rounded
        ra = params.get("rotation_angle") or 0
        cr = params.get("corner_radius") or 0
        if ra > _ROTATION_ANGLE_TOL and cr < 0.1:
            confidence *= 0.8

    elif contour_type == "slot":
        # Slot: elongated with circular ends
        aspect = (params.get("length") or 1) / max(params.get("width") or 1, 1e-9)
        aspect_score = min(1.0, aspect / 3.0)  # Ideal at aspect=3

        # Check for two circular ends
        end_score = _circular_end_score(pts2d)

        confidence = 0.3 + 0.35 * aspect_score + 0.35 * end_score

    elif contour_type == "hexagon":
        # Check for 6 corners
        corners = _find_corners(pts2d, k=6)
        corner_score = len(corners) / 6.0

        # Check edge uniformity
        edge_uniformity = _edge_uniformity_score(pts2d, 6)

        confidence = 0.3 + 0.35 * corner_score + 0.35 * edge_uniformity

    elif contour_type == "outer":
        confidence = 0.7  # Outer is always valid

    elif contour_type == "irregular":
        confidence = 0.4  # Irregular is a fallback

    return min(1.0, max(0.0, confidence))


def _point_radii(pts2d: list[tuple[float, float]]) -> list[float]:
    """Get distances from centroid for all points."""
    cx, cy = _shoelace_centroid(pts2d, _shoelace_area(pts2d))
    return [math.hypot(p[0] - cx, p[1] - cy) for p in pts2d]


def _standard_deviation(values: list[float]) -> float:
    """Calculate standard deviation."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _edge_alignment_score(pts2d: list[tuple[float, float]]) -> float:
    """Score how well edges align with bounding box axes."""
    if len(pts2d) < 4:
        return 0.5

    # For each edge, check if it's axis-aligned
    aligned_count = 0
    total_count = 0
    for i in range(len(pts2d)):
        j = (i + 1) % len(pts2d)
        dx = abs(pts2d[j][0] - pts2d[i][0])
        dy = abs(pts2d[j][1] - pts2d[i][1])

        if dx > 1e-6 or dy > 1e-6:
            total_count += 1
            # Edge is aligned if one component dominates
            if dx > dy * 5 or dy > dx * 5:
                aligned_count += 1

    if total_count == 0:
        return 0.5
    return aligned_count / total_count


def _circular_end_score(pts2d: list[tuple[float, float]]) -> float:
    """Score how circular the ends of a shape are (for slot detection)."""
    if len(pts2d) < 6:
        return 0.3

    # Find the two ends of the shape (min/max in dominant axis)
    xs = [p[0] for p in pts2d]
    ys = [p[1] for p in pts2d]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    if width > height:
        # Horizontal orientation
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        left_end = [(p[0], p[1]) for p in pts2d if p[0] < (xmin + width * 0.2)]
        right_end = [(p[0], p[1]) for p in pts2d if p[0] > (xmax - width * 0.2)]
    else:
        # Vertical orientation
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        left_end = [(p[0], p[1]) for p in pts2d if p[1] < (ymin + height * 0.2)]
        right_end = [(p[0], p[1]) for p in pts2d if p[1] > (ymax - height * 0.2)]

    # Check circularity of each end
    scores = []
    for end_pts in [left_end, right_end]:
        if len(end_pts) >= 3:
            area = _shoelace_area(end_pts)
            perimeter = sum(
                math.hypot(end_pts[i][0] - end_pts[(i+1)%len(end_pts)][0],
                          end_pts[i][1] - end_pts[(i+1)%len(end_pts)][1])
                for i in range(len(end_pts))
            )
            if perimeter > 0:
                circ = (4 * math.pi * abs(area)) / (perimeter * perimeter)
                scores.append(min(1.0, circ))

    if not scores:
        return 0.3
    return sum(scores) / len(scores)


def _edge_uniformity_score(pts2d: list[tuple[float, float]], expected_edges: int) -> float:
    """Score how uniform the edge lengths are."""
    if len(pts2d) < expected_edges:
        return 0.3

    # Find corners
    corners = _find_corners(pts2d, k=expected_edges)
    if len(corners) < expected_edges:
        return 0.3

    # Calculate edge lengths
    edge_lengths = []
    for i in range(len(corners)):
        p1 = pts2d[corners[i]]
        p2 = pts2d[corners[(i + 1) % len(corners)]]
        edge_lengths.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    if not edge_lengths:
        return 0.3

    # Coefficient of variation (lower = more uniform)
    mean_length = sum(edge_lengths) / len(edge_lengths)
    if mean_length < 1e-9:
        return 0.3

    cv = _standard_deviation(edge_lengths) / mean_length

    # Perfect uniformity = CV of 0, score of 1.0
    # CV of 0.5+ = score of 0.3
    score = max(0.3, 1.0 - cv * 2)
    return score


# ── Lead-line length calculation ────────────────────────────────────────────────

def estimate_lead_length(
    contour_type: str,
    params: dict[str, float | None],
) -> float:
    """Estimate recommended lead-in line length based on contour size.

    Returns lead length in mm.
    """
    if contour_type == "circle":
        d = params.get("diameter") or 10
        return min(d * _LEAD_DIAMETER_RATIO, 5.0)
    elif contour_type == "rectangle":
        w = params.get("width") or 10
        return min(w * _LEAD_DIAMETER_RATIO, 5.0)
    elif contour_type == "slot":
        w = params.get("width") or 5
        return min(w * 0.3, 3.0)
    elif contour_type == "hexagon":
        af = params.get("across_flats") or 10
        return min(af * 0.25, 3.0)
    return 3.0  # Default


# ── Extended hole feature creation ─────────────────────────────────────────────

def create_extended_hole(
    contour: dict[str, Any],
    contour_type: str,
    lead_length: float | None = None,
) -> dict[str, Any]:
    """Create extended hole feature with laser-software parameters.

    Extends the standard hole with:
    - lead_length (引线长度)
    - compensation parameters
    - validation status
    """
    params = contour.get("parameters", {})
    extracted_params = extract_contour_parameters(
        pts2d=[],  # Would need original pts2d
        contour_type=contour_type,
        circularity=contour.get("circularity", 0),
        perimeter=contour.get("perimeter", 0),
    )

    # Merge parameters
    for key, value in extracted_params.items():
        if value is not None and params.get(key) is None:
            params[key] = value

    # Estimate lead length
    if lead_length is None:
        lead_length = estimate_lead_length(contour_type, params)

    hole = {
        **contour,
        "kind": contour_type,
        "lead_length": lead_length,
        "parameters": params,
        "validation": validate_contour_parameters(params, contour_type)[0],
    }

    return hole


# ── Classification diagnosis ─────────────────────────────────────────────────

def diagnose_classification(
    pts2d: list[tuple[float, float]],
    is_outer: bool,
) -> dict[str, Any]:
    """Diagnose why a contour may not be correctly classified.

    Returns detailed diagnostic info including:
    - Point count
    - Bounding box dimensions
    - Aspect ratio
    - Circularity
    - Corner count
    - Likely issues
    """
    import math

    n = len(pts2d)
    if n < 4:
        return {
            "point_count": n,
            "issue": "Too few points for classification",
            "recommendation": "Increase linear_deflection parameter",
        }

    # Bounding box
    xs = [p[0] for p in pts2d]
    ys = [p[1] for p in pts2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = xmax - xmin
    height = ymax - ymin
    length = max(width, height)
    width_ = min(width, height)
    aspect = length / max(width_, 1e-9)

    # Area and perimeter
    s = 0.0
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    area = abs(s) * 0.5

    perimeter = sum(
        math.hypot(pts2d[i][0] - pts2d[(i + 1) % n][0],
                   pts2d[i][1] - pts2d[(i + 1) % n][1])
        for i in range(n)
    )
    circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    # Corner detection
    corner_count = _count_corners(pts2d)

    # Diagnosis
    issues = []
    recommendations = []

    if is_outer:
        classification = "outer"
    else:
        # Check each type
        likely_types = []

        # Circle check
        if circularity >= 0.70:
            likely_types.append("circle")
        elif circularity >= 0.60:
            likely_types.append("circle (uncertain)")

        # Slot check
        if aspect >= 2.2 and circularity <= 0.88:
            likely_types.append("slot")
        if aspect >= 2.0:
            likely_types.append("slot (uncertain)")

        # Rectangle check
        if corner_count == 4 and circularity >= 0.55:
            likely_types.append("rectangle")
        elif corner_count >= 4 and circularity >= 0.50:
            likely_types.append("rectangle (uncertain)")

        # Hexagon check
        if corner_count == 6:
            likely_types.append("hexagon")

        # Irregular check
        if circularity < 0.55:
            likely_types.append("irregular")
        elif circularity < 0.70:
            likely_types.append("irregular (uncertain)")

        classification = likely_types[0] if likely_types else "unknown"

    # Common issues
    if circularity < 0.50 and aspect < 2.0:
        issues.append("Low circularity suggests irregular shape")
        recommendations.append("Shape may be intentionally non-standard")

    if aspect > 10:
        issues.append("Very high aspect ratio may cause classification issues")
        recommendations.append("Consider if this is a thin slot or a noise contour")

    if corner_count < 4 and circularity < 0.70:
        issues.append("Few corners but low circularity suggests tessellation noise")
        recommendations.append("Try reducing linear_deflection for smoother contour")

    if n < 8:
        issues.append("Very few points may cause poor classification")
        recommendations.append("Increase linear_deflection (try 0.05 for finer sampling)")

    return {
        "point_count": n,
        "bounding_box": {"width": round(width, 2), "height": round(height, 2)},
        "aspect_ratio": round(aspect, 2),
        "area": round(area, 2),
        "perimeter": round(perimeter, 2),
        "circularity": round(circularity, 3),
        "corner_count": corner_count,
        "likely_classification": classification,
        "issues": issues,
        "recommendations": recommendations,
    }


def _count_corners(pts2d: list[tuple[float, float]]) -> int:
    """Count dominant corners in the contour."""
    n = len(pts2d)
    if n < 3:
        return 0

    angles = []
    for i in range(n):
        p1 = pts2d[(i - 2) % n]
        p2 = pts2d[i]
        p3 = pts2d[(i + 2) % n]

        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue

        cos_val = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        angle = math.degrees(math.acos(cos_val))
        angles.append((i, angle))

    # Count corners: angles significantly different from 180°
    corner_threshold = 30  # degrees deviation from straight line
    corners = [i for i, a in angles if abs(a - 180) > corner_threshold]

    return len(corners)
