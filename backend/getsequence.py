import base64
import json
import sys

import cv2
import numpy as np

from backend.multi_pin_bfs import detect_wire_colors_bfs


def base64_to_cv2_image(base64_str):
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",", 1)[1]
    base64_str += "=" * (-len(base64_str) % 4)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(base64_str), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be decoded")
    return img


def _normalize_expected_count(expected_count):
    try:
        n = int(expected_count)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def get_sequence_from_image(image, expected_count=None):
    """Return (count, [(b, g, r), ...]) left to right. Colours are plain ints."""
    if image is None or image.size == 0:
        return 0, []
    return detect_wire_colors_bfs(image, _normalize_expected_count(expected_count))


def main():
    try:
        payload = json.loads(sys.stdin.read())
        images = payload.get("input", [])
        if payload.get("wireType") != "singlewire":
            raise ValueError("Only 'singlewire' is supported in this version")
        img = base64_to_cv2_image(images[0])
        count, colors = get_sequence_from_image(img, payload.get("expected_count"))
        print(json.dumps({"type": "singlewire", "count": count,
                          "sequence": [list(c) for c in colors]}))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
