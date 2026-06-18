"""Example API response with enhanced_params=true.

This shows what the API returns when enhanced_params is enabled.
"""

# Example response structure when enhanced_params=true
EXAMPLE_RESPONSE = {
    "schema_version": "1.1",
    "unit": "mm",
    "target_face_id": "face_0",
    "contours": [
        {
            "id": "contour_0",
            "contour_type": "outer",
            "contour_role": "outer_boundary",
            "center": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "is_outer": True,
            "parameters": {
                "diameter": None,
                "length": 100.0,
                "width": 80.0,
                "across_flats": None,
                "rotation_angle": 15.5,           # NEW
                "corner_radius": 2.0,             # NEW
                "compensation_length": 0.0,        # NEW (VL)
                "compensation_width": 0.0,         # NEW (VW)
                "overlap_distance": 1.0,          # NEW (OD)
            },
            "area": 8000.0,
            "perimeter": 360.0,
            "confidence": 0.85,
        },
        {
            "id": "contour_1",
            "contour_type": "circle",
            "contour_role": "inner_hole",
            "center": [25.0, 20.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "is_outer": False,
            "parameters": {
                "diameter": 30.0,
                "length": None,
                "width": None,
                "across_flats": None,
                "rotation_angle": None,
                "corner_radius": None,
                "compensation_length": None,
                "compensation_width": None,
                "overlap_distance": None,
            },
            "area": 706.86,
            "perimeter": 94.25,
            "confidence": 0.95,
            "validation": {                     # NEW
                "is_valid": True,
                "error": None,
            },
            "lead_length": 5.0,                # NEW
        },
        {
            "id": "contour_2",
            "contour_type": "rectangle",
            "contour_role": "inner_hole",
            "center": [-30.0, 15.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "is_outer": False,
            "parameters": {
                "diameter": None,
                "length": 50.0,
                "width": 30.0,
                "across_flats": None,
                "rotation_angle": 0.0,          # NEW
                "corner_radius": 2.0,           # NEW
                "compensation_length": 0.0,      # NEW (VL)
                "compensation_width": 0.0,       # NEW (VW)
                "overlap_distance": 1.0,         # NEW (OD)
            },
            "area": 1500.0,
            "perimeter": 160.0,
            "confidence": 0.88,
            "validation": {                     # NEW
                "is_valid": True,
                "error": None,
            },
            "lead_length": 3.0,                # NEW
        },
        {
            "id": "contour_3",
            "contour_type": "slot",
            "contour_role": "inner_hole",
            "center": [0.0, -25.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "is_outer": False,
            "parameters": {
                "diameter": None,
                "length": 60.0,
                "width": 20.0,
                "across_flats": None,
                "rotation_angle": 45.0,          # NEW
                "corner_radius": None,
                "compensation_length": 0.0,      # NEW (VL)
                "compensation_width": 0.0,      # NEW (VW)
                "overlap_distance": None,
            },
            "area": 1042.7,
            "perimeter": 182.83,
            "confidence": 0.78,
            "validation": {                     # NEW
                "is_valid": True,
                "error": None,
            },
            "lead_length": 3.0,                # NEW
        },
    ],
    "holes": [
        # Holes include the same enhanced parameters
        {
            "id": "hole_contour_1",
            "kind": "circle",
            "center": [25.0, 20.0, 0.0],
            "parameters": {
                "diameter": 30.0,
                "lead_length": 5.0,             # NEW
            },
            "validation": {                     # NEW
                "is_valid": True,
            },
        },
    ],
}


def print_example():
    """Print the example response in a readable format."""
    import json
    print(json.dumps(EXAMPLE_RESPONSE, indent=2))


if __name__ == "__main__":
    print_example()
