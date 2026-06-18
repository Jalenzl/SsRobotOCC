"""Test script for enhanced contour parameters.

Run with: python -m app.occ.test_enhanced_params
"""

import math
from app.occ.contour_enhanced import (
    extract_contour_parameters,
    validate_contour_parameters,
    calculate_classification_confidence,
    estimate_lead_length,
)


def create_circle_points(cx: float, cy: float, radius: float, n: int = 36):
    """Create n points on a circle."""
    pts = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return pts


def create_rectangle_points(x: float, y: float, w: float, h: float):
    """Create rectangle points (clockwise from bottom-left)."""
    return [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h),
    ]


def create_slot_points(length: float, width: float, n: int = 60):
    """Create slot (obround) points."""
    pts = []
    half_w = width / 2
    half_l = length / 2

    # Left semicircle
    for i in range(n // 4):
        angle = math.pi * i / (n // 4)
        pts.append((-half_l + half_w * (1 - math.cos(angle)),
                    half_w * math.sin(angle)))

    # Bottom line
    pts.append((-half_l + half_w, -half_w))

    # Right semicircle
    for i in range(n // 4):
        angle = math.pi + math.pi * i / (n // 4)
        pts.append((half_l + half_w * math.cos(angle),
                    half_w * math.sin(angle)))

    # Top line
    pts.append((half_l + half_w, half_w))
    pts.append((-half_l + half_w, half_w))

    return pts


def create_hexagon_points(cx: float, cy: float, side: float):
    """Create regular hexagon points."""
    pts = []
    for i in range(6):
        angle = math.pi * i / 3
        pts.append((cx + side * math.cos(angle),
                    cy + side * math.sin(angle)))
    return pts


def test_circle():
    """Test circle extraction."""
    print("\n" + "=" * 50)
    print("TEST: Circle")
    print("=" * 50)

    pts = create_circle_points(0, 0, 10)
    circularity = 0.95
    perimeter = 2 * math.pi * 10

    params = extract_contour_parameters(pts, "circle", circularity, perimeter)
    print(f"\nExtracted parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    is_valid, error = validate_contour_parameters(params, "circle")
    print(f"\nValidation: is_valid={is_valid}, error={error}")

    confidence = calculate_classification_confidence(pts, circularity, "circle", params)
    print(f"Confidence: {confidence:.3f}")

    lead = estimate_lead_length("circle", params)
    print(f"Estimated lead length: {lead:.2f} mm")


def test_rectangle():
    """Test rectangle extraction."""
    print("\n" + "=" * 50)
    print("TEST: Rectangle (50x30 mm)")
    print("=" * 50)

    pts = create_rectangle_points(0, 0, 50, 30)
    circularity = 0.75  # Rectangles have lower circularity
    perimeter = 2 * (50 + 30)

    params = extract_contour_parameters(pts, "rectangle", circularity, perimeter)
    print(f"\nExtracted parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    is_valid, error = validate_contour_parameters(params, "rectangle")
    print(f"\nValidation: is_valid={is_valid}, error={error}")

    confidence = calculate_classification_confidence(pts, circularity, "rectangle", params)
    print(f"Confidence: {confidence:.3f}")

    lead = estimate_lead_length("rectangle", params)
    print(f"Estimated lead length: {lead:.2f} mm")


def test_slot():
    """Test slot extraction."""
    print("\n" + "=" * 50)
    print("TEST: Slot (60x20 mm)")
    print("=" * 50)

    pts = create_slot_points(60, 20)
    circularity = 0.80  # Obround has moderate circularity
    perimeter = 2 * 60 + math.pi * 20

    params = extract_contour_parameters(pts, "slot", circularity, perimeter)
    print(f"\nExtracted parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    is_valid, error = validate_contour_parameters(params, "slot")
    print(f"\nValidation: is_valid={is_valid}, error={error}")

    confidence = calculate_classification_confidence(pts, circularity, "slot", params)
    print(f"Confidence: {confidence:.3f}")

    lead = estimate_lead_length("slot", params)
    print(f"Estimated lead length: {lead:.2f} mm")


def test_hexagon():
    """Test hexagon extraction."""
    print("\n" + "=" * 50)
    print("TEST: Hexagon (side=10 mm)")
    print("=" * 50)

    pts = create_hexagon_points(0, 0, 10)
    circularity = 0.80
    perimeter = 6 * 10

    params = extract_contour_parameters(pts, "hexagon", circularity, perimeter)
    print(f"\nExtracted parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    is_valid, error = validate_contour_parameters(params, "hexagon")
    print(f"\nValidation: is_valid={is_valid}, error={error}")

    confidence = calculate_classification_confidence(pts, circularity, "hexagon", params)
    print(f"Confidence: {confidence:.3f}")

    lead = estimate_lead_length("hexagon", params)
    print(f"Estimated lead length: {lead:.2f} mm")


def main():
    print("\n" + "#" * 60)
    print("# ENHANCED CONTOUR PARAMETERS TEST")
    print("#" * 60)

    test_circle()
    test_rectangle()
    test_slot()
    test_hexagon()

    print("\n" + "#" * 60)
    print("# TEST COMPLETE")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
