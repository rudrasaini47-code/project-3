"""Single-strand wire colour detector (fixed camera, vertical wires, light background).

Public API is unchanged:  detect_wire_colors_bfs(image, expected_count) -> (count, [(b, g, r), ...])

Method (replaces the pixel-BFS):
  1. Work in Lab. Estimate the background per image row (row median), so a vignetted
     background does not matter.
  2. Locate the row where the wires enter the connector, and take a band of rows a
     little above it, where wires are parallel.
  3. Collapse the band to one Lab colour per column (median over rows).
  4. Find the bundle's left/right extent, then cut it into `expected_count` segments at the
     strongest colour edges (dynamic programming, with a sane min/max wire width).
  5. Each wire's colour = median of the centre of its segment.

Touching wires are no problem because we split by colour edges, not by connectivity.
"""
import cv2
import numpy as np

FG_DELTA_E = 30.0        # strong foreground: used to find where wires enter the connector
EXTENT_DELTA_E = 15.0    # weaker threshold: used for the bundle's left/right edge (pale wires)
END_ROW_MIN_PIXELS = 40  # a row with fewer strong pixels is not part of the wire
BAND_FROM_END = (75, 40) # band = rows [end-75, end-40]
MIN_WIDTH_FRAC = 0.70    # wires are the same gauge, so widths are similar
MAX_WIDTH_FRAC = 1.5
NOMINAL_WIRE_PX = 21.0   # typical wire width in pixels for your fixed camera (calibrate!)


def _to_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _lab_to_bgr(lab_vec):
    v = np.array(lab_vec, dtype=np.float32)
    v[0] *= 255.0 / 100.0
    v[1:] += 128.0
    px = np.clip(v, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_Lab2BGR)[0, 0]
    return int(b), int(g), int(r)


def _background_distance(lab):
    """Distance of every pixel from its own row's median colour (= the background)."""
    bg = np.median(lab, axis=1)
    return np.linalg.norm(lab - bg[:, None, :], axis=2)


def _find_band(dist):
    h, w = dist.shape
    d = dist.copy()
    d[:, : int(w * 0.10)] = 0           # ignore vignette at the left/right edges
    d[:, int(w * 0.90):] = 0
    counts = (d > FG_DELTA_E).sum(axis=1)
    rows = np.where(counts >= END_ROW_MIN_PIXELS)[0]
    if len(rows) == 0:
        return None
    end = int(rows.max())
    y1 = max(0, end - BAND_FROM_END[0])
    y2 = max(y1 + 5, end - BAND_FROM_END[1])
    return y1, min(h, y2)


def _bundle_extent(col_dist, gap=4):
    """Widest run of foreground columns, bridging gaps up to `gap` px."""
    fg = _mask_margins(col_dist) > EXTENT_DELTA_E
    best, start, last = (0, 0), None, None
    for x, v in enumerate(fg):
        if v:
            if start is None:
                start = x
            last = x
        elif start is not None and x - last > gap:
            if last - start > best[1] - best[0]:
                best = (start, last)
            start = None
    if start is not None and last - start > best[1] - best[0]:
        best = (start, last)
    return best if best[1] > best[0] else None


EDGE_MARGIN = 0.10       # ignore the outer 10 % of the frame on each side (lens vignette)
RUN_BRIDGE_PX = 3        # gaps up to this many px inside one wire are bridged (highlights)
RUN_MIN_PX = 4           # a foreground run narrower than this is noise


def _mask_margins(col_dist):
    d = col_dist.copy()
    w = len(d)
    d[: int(w * EDGE_MARGIN)] = 0
    d[int(w * (1 - EDGE_MARGIN)):] = 0
    return d


def _fg_runs(col_dist, thr=EXTENT_DELTA_E):
    """Separate foreground runs (one per wire when the wires do not touch)."""
    fg = _mask_margins(col_dist) > thr
    runs, start, last = [], None, None
    for x, v in enumerate(fg):
        if v:
            if start is None:
                start = x
            last = x
        elif start is not None and x - last > RUN_BRIDGE_PX:
            runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    return [r for r in runs if r[1] - r[0] + 1 >= RUN_MIN_PX]


def _edge_strength(col_lab, x0, x1):
    """Colour change between neighbouring columns, smoothed a little."""
    seg = col_lab[x0:x1 + 1]
    k = np.ones(3, np.float32) / 3.0
    sm = np.stack([np.convolve(seg[:, c], k, mode="same") for c in range(3)], axis=1)
    # compare 3 columns left of x with 3 columns right of x (robust to 1-px noise)
    e = np.zeros(len(seg), np.float32)
    for i in range(3, len(seg) - 3):
        e[i] = np.linalg.norm(sm[i - 3:i].mean(0) - sm[i:i + 3].mean(0))
    return e


def _split_dp(seg, n):
    """Optimal split of the bundle into n pieces of near-constant colour.

    Minimises the total within-segment squared Lab error (piecewise-constant fit) with
    min/max wire-width limits. More robust than cutting at edge peaks, because the dark
    shadow line between two wires produces two edges but only one boundary.
    seg: (L, 3) per-column Lab colours of the bundle. Returns n-1 cut indices.
    """
    length = len(seg)
    avg = length / n
    lo = max(3, int(avg * MIN_WIDTH_FRAC))
    hi = int(np.ceil(avg * MAX_WIDTH_FRAC))
    s1 = np.vstack([np.zeros((1, 3)), np.cumsum(seg, axis=0)])
    s2 = np.concatenate([[0.0], np.cumsum((seg ** 2).sum(axis=1))])

    def cost(a, b):
        m = b - a
        tot = s1[b] - s1[a]
        return (s2[b] - s2[a]) - float((tot ** 2).sum()) / m

    INF = 1e18
    best = np.full((n + 1, length + 1), INF)
    prev = np.zeros((n + 1, length + 1), dtype=np.int32)
    best[0][0] = 0.0
    for k in range(1, n + 1):
        for i in range(lo * k, min(length, hi * k) + 1):
            for j in range(max(0, i - hi), i - lo + 1):
                if best[k - 1][j] >= INF:
                    continue
                c = best[k - 1][j] + cost(j, i)
                if c < best[k][i]:
                    best[k][i], prev[k][i] = c, j
    if best[n][length] >= INF:
        return [int(round(avg * k)) for k in range(1, n)]   # equal-width fallback
    cuts, i = [], length
    for k in range(n, 1, -1):
        i = int(prev[k][i])
        cuts.append(i)
    return sorted(cuts)


def _auto_cuts(edge, avg_min_width):
    """No expected count given: cut at strong, well separated colour edges."""
    thr = max(12.0, float(np.percentile(edge, 90)) * 0.5)
    cuts = []
    for i in np.argsort(-edge):
        if edge[i] < thr:
            break
        if all(abs(i - c) >= avg_min_width for c in cuts):
            cuts.append(int(i))
    return sorted(cuts)


def _split_touching(runs, col_lab, n):
    """Two touching wires show up as one wide run. Split the widest run in two (by colour)
    until we have n runs, but only if it is clearly wider than a single wire."""
    runs = list(runs)
    while len(runs) < n:
        widths = [b - a + 1 for a, b in runs]
        i = int(np.argmax(widths))
        others = [w for j, w in enumerate(widths) if j != i]
        single = float(np.median(others)) if others else 0.0
        if widths[i] < max(8, 1.5 * single) or widths[i] < 8:
            break
        a, b = runs[i]
        cut = a + _split_dp(col_lab[a:b + 1], 2)[0]
        runs[i:i + 1] = [(a, cut - 1), (cut, b)]
    return runs


def _centre_colour(col_lab, a, b, frac=0.25):
    w = b - a + 1
    ca, cb = a + int(w * frac), b + 1 - int(w * frac)
    cb = max(cb, ca + 1)
    return _lab_to_bgr(np.median(col_lab[ca:cb], axis=0))


def detect_wire_colors_bfs(image, expected_count=None, return_debug=False,
                           band=None, nominal_wire_px=None):
    """Return (count, [(b, g, r), ...]) left to right. Plain Python ints (JSON safe).

    band            optional (y1, y2) rows to read the wires from. None = find automatically.
    nominal_wire_px optional wire pitch in px for the count check. None = NOMINAL_WIRE_PX.
    """
    nominal = float(nominal_wire_px) if nominal_wire_px else NOMINAL_WIRE_PX
    empty = (0, [], {}) if return_debug else (0, [])
    if image is None or getattr(image, "size", 0) == 0:
        return empty

    lab = _to_lab(image)
    dist = _background_distance(lab)
    if band and int(band[1]) > int(band[0]):
        h = lab.shape[0]
        y1, y2 = max(0, int(band[0])), min(h, int(band[1]))
    else:
        found = _find_band(dist)
        if found is None:
            return empty
        y1, y2 = found

    col_lab = np.median(lab[y1:y2], axis=0)             # (W, 3)
    col_dist = np.median(dist[y1:y2], axis=0)           # (W,)

    # ---- wires that do NOT touch: every wire is its own foreground run ----
    n_req = int(expected_count) if expected_count else 0
    if n_req >= 2:
        first_ok = None
        for thr in (EXTENT_DELTA_E, 25.0, 35.0):      # a stricter threshold separates wires whose
            runs = _fg_runs(col_dist, thr)            # soft shadows would otherwise merge them
            if len(runs) < 2 or max(b - a + 1 for a, b in runs) > 2.2 * nominal:
                continue
            runs = _split_touching(runs, col_lab, n_req)
            if len(runs) < 2 or max(b - a + 1 for a, b in runs) > 1.6 * nominal:
                continue
            if len(runs) == n_req:
                first_ok = runs
                break
            first_ok = first_ok or runs
        if first_ok:
            runs = first_ok
            colors = [_centre_colour(col_lab, a, b) for a, b in runs]
            dbg = {"band": (y1, y2), "extent": (runs[0][0], runs[-1][1]),
                   "cuts": [int((runs[i][1] + runs[i + 1][0]) // 2) for i in range(len(runs) - 1)],
                   "segments": [(int(a), int(b)) for a, b in runs], "mode": "separate",
                   "px_per_wire": round(float(np.mean([b - a + 1 for a, b in runs])), 1),
                   "bundle_px": int(runs[-1][1] - runs[0][0] + 1),
                   "count_estimate": float(len(runs)),
                   "count_mismatch": len(runs) != n_req}
            return (len(colors), colors, dbg) if return_debug else (len(colors), colors)

    ext = _bundle_extent(col_dist)
    if ext is None:
        return empty
    x0, x1 = ext
    length = x1 - x0 + 1

    edge = _edge_strength(col_lab, x0, x1)
    n = int(expected_count) if expected_count else 0

    # Sanity check: forcing N segments would hide a missing/extra wire. Wires have a fixed
    # width on a fixed camera, so bundle width / nominal wire width must be within ~0.6 wire of N.
    est_count = max(1, int(round(length / nominal)))
    count_mismatch = (n >= 1 and abs(length / nominal - n) > max(0.6, 0.1 * n))
    if count_mismatch:
        n = est_count          # split by the estimate, so the reported count reflects reality
    if n >= 2:
        cuts = _split_dp(col_lab[x0:x1 + 1], n)
    elif n == 1:
        cuts = []
    else:
        cuts = _auto_cuts(edge, max(6, length // 20))

    bounds = [0] + list(cuts) + [length]
    colors = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        w = b - a
        ca, cb = x0 + a + int(w * 0.30), x0 + b - int(w * 0.30)     # centre 40 %
        cb = max(cb, ca + 1)
        colors.append(_lab_to_bgr(np.median(col_lab[ca:cb], axis=0)))

    if return_debug:
        return len(colors), colors, {"band": (y1, y2), "extent": (x0, x1),
                                     "cuts": [int(x0 + c) for c in cuts],
                                     "px_per_wire": round(length / max(1, len(colors)), 1),
                                     "bundle_px": int(length),
                                     "count_estimate": round(length / nominal, 2), "mode": "bundle",
                                     "count_mismatch": bool(count_mismatch)}
    return len(colors), colors
