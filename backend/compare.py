import json
import sys

from backend.getsequence import get_sequence_from_image, base64_to_cv2_image
from backend.reference import compare_sequences, DEFAULT_MAX_DELTA_E


def compare_single(original, input_image, max_delta_e=DEFAULT_MAX_DELTA_E):
    """original = [num_wires, [BGR, ...]]. Returns {"match", "details", "wire_comparisons"}."""
    desired_num, desired_colors = int(original[0]), original[1]
    if len(desired_colors) != desired_num:
        return {"match": False, "wire_comparisons": [],
                "details": f"Reference has {len(desired_colors)} colours but wire count is {desired_num}"}

    count, detected = get_sequence_from_image(input_image, expected_count=desired_num)
    if count != desired_num:
        return {"match": False, "wire_comparisons": [],
                "details": f"Expected {desired_num} wires, detected {count}"}

    rows = compare_sequences(desired_colors, detected, max_delta_e)
    bad = [r for r in rows if not r["match"]]
    if not bad:
        return {"match": True, "wire_comparisons": rows,
                "details": "SUCCESSFUL: Number of wires and colors match."}
    msg = "; ".join(f"Wire {r['wire']}: expected {r['expected']}, got {r['detected']} "
                    f"[dE={r['distance']}]" for r in bad)
    return {"match": False, "details": msg, "wire_comparisons": rows}


def main():
    try:
        data = json.loads(sys.stdin.read())
        if data.get("wireType") != "singlewire":
            raise ValueError("Only 'singlewire' is supported in this version")
        wire_count = data["wire_count"]
        seq = data["sequence"]
        if isinstance(seq, str):                 # old double-encoded format
            seq = json.loads(seq)
        seq = [json.loads(s) if isinstance(s, str) else s for s in seq]
        img = base64_to_cv2_image(data["input"][0])
        print(json.dumps(compare_single([wire_count[0], seq[0]], img)))
    except Exception as exc:
        print(json.dumps({"match": False, "details": f"Error: {exc}", "wire_comparisons": []}))
        sys.exit(1)


if __name__ == "__main__":
    main()
