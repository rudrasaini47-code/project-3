"""Show what the detector sees.

Usage (run from the project root, the folder that contains backend/):
  python examples/run_demo.py detect  harness.png 6
  python examples/run_demo.py compare reference.png test.png 6
Outputs are printed and saved as PNG files next to the input.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from backend.multi_pin_bfs import detect_wire_colors_bfs
from backend.reference import color_name
from backend.compare import compare_single


def annotate(img, n, colors, dbg, rows=None):
    """Draw band, cut lines and numbered wire swatches under the image."""
    vis = img.copy()
    y1, y2 = dbg["band"]
    x0, x1 = dbg["extent"]
    cv2.rectangle(vis, (x0, y1), (x1, y2), (0, 255, 255), 2)
    for c in dbg["cuts"]:
        cv2.line(vis, (c, y1 - 40), (c, y2 + 40), (255, 0, 255), 2)
    sw_h, w = 70, vis.shape[1]
    strip = np.full((sw_h, w, 3), 255, np.uint8)
    bounds = [x0] + dbg["cuts"] + [x1]
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        cv2.rectangle(strip, (a, 0), (b, sw_h), tuple(int(c) for c in colors[i]), -1)
        ok = rows[i]["match"] if rows else True
        txt = f"{i+1}" if ok else f"{i+1} X"
        cv2.putText(strip, txt, (a + 3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if ok else (0, 0, 255), 2)
        cv2.putText(strip, color_name(colors[i]), (a + 3, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
    return np.vstack([vis, strip])


def detect(path, n):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Cannot read image: {path}")
    count, colors, dbg = detect_wire_colors_bfs(img, n, return_debug=True)
    print("INPUT :", path, img.shape)
    print("COUNT :", count)
    for i, c in enumerate(colors, 1):
        print(f"  wire {i}: BGR={c}  name={color_name(c)}")
    print("JSON  :", json.dumps({"count": count, "sequence": [list(c) for c in colors]}))
    out = os.path.splitext(path)[0] + "_detected.png"
    cv2.imwrite(out, annotate(img, n, colors, dbg))
    print("SAVED :", out)
    return img, count, colors


def compare(ref_path, test_path, n):
    _, count, ref_colors = detect(ref_path, n)
    print("\n--- comparing ---")
    test = cv2.imread(test_path)
    res = compare_single([n, ref_colors], test)
    print("MATCH  :", res["match"])
    print("DETAILS:", res["details"])
    for r in res["wire_comparisons"]:
        print(f"  wire {r['wire']}: expected {r['expected']:>7} got {r['detected']:>7} "
              f"dE={r['distance']:>5}  {'OK' if r['match'] else 'FAIL'}")
    c2, col2, dbg2 = detect_wire_colors_bfs(test, n, return_debug=True)
    if res["wire_comparisons"]:
        out = os.path.splitext(test_path)[0] + "_compared.png"
        cv2.imwrite(out, annotate(test, n, col2, dbg2, res["wire_comparisons"]))
        print("SAVED  :", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "detect":
        detect(a[1], int(a[2]))
    elif len(a) == 4 and a[0] == "compare":
        compare(a[1], a[2], int(a[3]))
    else:
        print(__doc__)
