import json
import os

import cv2
import numpy as np

from backend.getsequence import get_sequence_from_image

# Pass/fail tolerance in CIE Lab units (Delta E). 15 sits between same-wire drift (<=7 on
# the sample photos) and the closest different pair (~24, black vs gray). Tune on your own reference/test photos.
DEFAULT_MAX_DELTA_E = 15.0


def _bgr_to_hsv_components(bgr_tuple):
    """Kept for naming only. Returns (hue in degrees, saturation, value)."""
    pixel = np.array([[list(bgr_tuple)]], dtype=np.uint8)
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    return int(h) * 2, int(s), int(v)


def _classify_hue(h_deg, s, v):
    """Human-readable colour name (display only, never used for pass/fail)."""
    if v < 60:
        return "black"
    if s < 45:
        return "white" if v > 200 else "gray"
    if h_deg <= 12 or h_deg >= 348:
        return "red"
    if h_deg <= 20 and v < 130:
        return "brown"
    if h_deg <= 40:
        return "orange"
    if h_deg <= 70:
        return "yellow"
    if h_deg <= 165:
        return "green"
    if h_deg <= 200:
        return "cyan"
    if h_deg <= 265:
        return "blue"
    if h_deg <= 320:
        return "purple"
    return "pink"


def color_name(bgr):
    return _classify_hue(*_bgr_to_hsv_components(tuple(int(c) for c in bgr)))


def _bgr_to_lab(bgr):
    px = np.uint8([[[int(c) for c in bgr]]])
    L, a, b = cv2.cvtColor(px, cv2.COLOR_BGR2Lab)[0][0].astype(float)
    return np.array([L * 100.0 / 255.0, a - 128.0, b - 128.0])


def _wire_color_distance(bgr_a, bgr_b):
    """Delta E (Lab) between two BGR colours. 0 = identical."""
    return float(np.linalg.norm(_bgr_to_lab(bgr_a) - _bgr_to_lab(bgr_b)))


def store_reference(image, num_wires, save_path="reference.json"):
    """Detect and store a reference. Refuses to store if the count is wrong."""
    count, colors = get_sequence_from_image(image, expected_count=num_wires)
    if count != int(num_wires):
        raise ValueError(f"Reference has {count} wires detected, expected {num_wires}")
    data = {
        "num_wires": int(num_wires),
        "color_sequence": [[int(c) for c in col] for col in colors],  # BGR
        "color_names": [color_name(c) for c in colors],
    }
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    return count, colors


def compare_sequences(ref_colors, det_colors, max_delta_e=DEFAULT_MAX_DELTA_E):
    """Per-wire comparison. Lists must be the same length (checked by the caller)."""
    rows = []
    for i, (r, d) in enumerate(zip(ref_colors, det_colors), start=1):
        dist = _wire_color_distance(r, d)
        rows.append({
            "wire": i,
            "match": dist <= max_delta_e,
            "distance": round(dist, 2),
            "expected": color_name(r),
            "detected": color_name(d),
            "reference_bgr": [int(c) for c in r],
            "detected_bgr": [int(c) for c in d],
        })
    return rows


def verify_against_reference(new_image, reference_path="reference.json",
                             max_delta_e=DEFAULT_MAX_DELTA_E):
    if not os.path.exists(reference_path):
        return {"match": False, "details": "Reference file not found", "wire_comparisons": []}
    with open(reference_path) as f:
        ref = json.load(f)

    num_wires = int(ref["num_wires"])
    ref_colors = [tuple(c) for c in ref["color_sequence"]]
    count, new_colors = get_sequence_from_image(new_image, expected_count=num_wires)
    if count != num_wires:
        return {"match": False,
                "details": f"Expected {num_wires} wires, detected {count}",
                "wire_comparisons": []}

    rows = compare_sequences(ref_colors, new_colors, max_delta_e)
    ok = all(r["match"] for r in rows)
    return {"match": ok, "details": "reference verification", "wire_comparisons": rows}
