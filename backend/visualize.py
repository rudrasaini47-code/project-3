import cv2
import numpy as np

from backend.reference import color_name


def annotate(img, colors, dbg, rows=None):
    """Draw band, cut lines and numbered colour swatches under the image."""
    vis = img.copy()
    y1, y2 = dbg["band"]
    x0, x1 = dbg["extent"]
    cv2.rectangle(vis, (x0, y1), (x1, y2), (0, 255, 255), 2)
    if not dbg.get("segments"):
        for c in dbg["cuts"]:
            cv2.line(vis, (c, max(0, y1 - 40)), (c, y2 + 40), (255, 0, 255), 2)
    sw_h, w = 70, vis.shape[1]
    strip = np.full((sw_h, w, 3), 255, np.uint8)
    segs = dbg.get("segments")            # separate (non-touching) wires: one box per wire
    if segs:
        for a, b in segs:
            cv2.rectangle(vis, (a, y1), (b, y2), (0, 255, 255), 1)
        pairs = [(a, b + 1) for a, b in segs]
    else:
        bounds = [x0] + list(dbg["cuts"]) + [x1]
        pairs = list(zip(bounds[:-1], bounds[1:]))
    for i, (a, b) in enumerate(pairs):
        cv2.rectangle(strip, (a, 0), (b, sw_h), tuple(int(c) for c in colors[i]), -1)
        ok = rows[i]["match"] if rows else True
        cv2.putText(strip, f"{i + 1}" if ok else f"{i + 1} X", (a + 3, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if ok else (0, 0, 255), 2)
        if b - a >= 40:                    # narrow wires: the table below shows the names
            cv2.putText(strip, color_name(colors[i]), (a + 3, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return np.vstack([vis, strip])
