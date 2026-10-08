"""One-call check: reference image + test image + wire count -> RIGHT / WRONG + confidence.

Confidence is a heuristic score in [0, 1] (NOT a calibrated probability):
  RIGHT: how safely every wire sits inside the colour tolerance (weakest wire decides),
         reduced if the bundle width looks unusual for this wire count.
  WRONG: how clearly something is off (worst wire colour, or a wrong wire count).
Results with confidence below REVIEW_BELOW are flagged "needs_review".
"""
import json
import sys

import numpy as np

from backend.getsequence import base64_to_cv2_image
from backend.multi_pin_bfs import detect_wire_colors_bfs, NOMINAL_WIRE_PX
from backend.reference import (DEFAULT_MAX_DELTA_E, _wire_color_distance, color_name)

SOFTNESS = 3.0        # dE units: how gradually a wire goes from "match" to "mismatch"
REVIEW_BELOW = 0.75   # below this confidence, a human should look at it


def _wire_match_prob(de, threshold):
    """~1 for a clear match, 0.5 right at the tolerance, ~0 for a clear mismatch."""
    return float(1.0 / (1.0 + np.exp((de - threshold) / SOFTNESS)))


def _count_deviation(dbg, n):
    return abs(dbg["count_estimate"] - n)


def check_harness(reference_img, test_img, num_wires, max_delta_e=DEFAULT_MAX_DELTA_E,
                  band=None, wire_px=None):
    """band = optional (y1, y2) rows to read the wires from (None = automatic);
    wire_px = optional wire pitch in px for the wire-count check (None = default)."""
    n = int(num_wires)
    out = {"verdict": "WRONG", "confidence": 0.0, "needs_review": True, "reason": "",
           "num_wires": n, "wires": []}

    if reference_img is None or test_img is None:
        out["reason"] = "An image could not be read."
        return out

    rc, rcols, rdbg = detect_wire_colors_bfs(reference_img, n, return_debug=True,
                                           band=band, nominal_wire_px=wire_px)
    if rc != n or not rcols or rdbg.get("count_mismatch"):
        out["reason"] = (f"The reference photo does not look like {n} wires "
                         f"(it looks like {rdbg.get('count_estimate', 0):.1f}). "
                         "Check the wire count and the reference photo.")
        out["confidence"] = 0.0
        out["reference_error"] = True
        return out

    tc, tcols, tdbg = detect_wire_colors_bfs(test_img, n, return_debug=True,
                                           band=band, nominal_wire_px=wire_px)
    if not tcols:
        out["reason"] = "No wires found in the test image."
        out["confidence"] = 0.9
        return out

    # ---- wrong wire count ----
    if tdbg.get("count_mismatch"):
        dev = _count_deviation(tdbg, n)
        if dev > 2.0:   # far off: more likely a detection failure (lighting / framing) than a real fault
            out["confidence"] = 0.5
            out["reason"] = (f"Could not read the harness reliably (bundle width suggests "
                             f"{tdbg['count_estimate']:.1f} wires, expected {n}). Check lighting and framing.")
        else:
            out["confidence"] = round(float(min(0.9, 0.6 + 0.3 * (dev - 0.6) / 0.6)), 3)
            out["reason"] = (f"Wire count looks wrong: expected {n}, bundle width suggests "
                             f"about {tdbg['count_estimate']:.1f}.")
        out["needs_review"] = out["confidence"] < REVIEW_BELOW
        return out

    # ---- per-wire colour check ----
    probs = []
    for i, (r, t) in enumerate(zip(rcols, tcols), start=1):
        de = _wire_color_distance(r, t)
        p = _wire_match_prob(de, max_delta_e)
        probs.append(p)
        out["wires"].append({
            "wire": i, "expected": color_name(r), "detected": color_name(t),
            "delta_e": round(de, 2), "match": de <= max_delta_e,
            "match_probability": round(p, 3),
            "reference_bgr": [int(c) for c in r], "detected_bgr": [int(c) for c in t]})

    bad = [w for w in out["wires"] if not w["match"]]
    # width quality: 1.0 for a normal bundle width, down to 0.85 near the allowed limit
    limit = max(0.6, 0.1 * n)
    quality = 1.0 - 0.15 * min(1.0, max(0.0, _count_deviation(tdbg, n) - 0.3) / (limit - 0.3))

    if not bad:
        out["verdict"] = "RIGHT"
        out["confidence"] = round(float(min(probs) * quality), 3)
        out["reason"] = f"All {n} wires match the reference."
    else:
        out["verdict"] = "WRONG"
        out["confidence"] = round(float(max(1.0 - p for p in probs)), 3)
        out["reason"] = "; ".join(f"wire {w['wire']}: expected {w['expected']}, got {w['detected']} "
                                  f"(dE {w['delta_e']})" for w in bad)
    out["needs_review"] = out["confidence"] < REVIEW_BELOW
    out["_ref_dbg"], out["_test_dbg"] = rdbg, tdbg
    out["_ref_colors"], out["_test_colors"] = rcols, tcols
    return out


def public(result):
    """JSON-safe copy without the internal debug fields."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def main():
    """stdin JSON: {"reference": b64, "input": b64, "expected_count": 6}"""
    try:
        d = json.loads(sys.stdin.read())
        res = check_harness(base64_to_cv2_image(d["reference"]), base64_to_cv2_image(d["input"]),
                            d["expected_count"], d.get("max_delta_e", DEFAULT_MAX_DELTA_E))
        print(json.dumps(public(res)))
    except Exception as exc:
        print(json.dumps({"verdict": "ERROR", "confidence": 0.0, "reason": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
