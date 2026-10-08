"""Check history: every check is stored with its images and result.

Layout (folder override: env WHC_HISTORY_DIR, default  history/  next to app.py):
  history/<id>/result.json            verdict, confidence, per-wire rows + settings used
  history/<id>/reference.jpg          the reference photo
  history/<id>/input.jpg              the photo that was checked
  history/<id>/reference_annotated.jpg, input_annotated.jpg   (when wires were read)

<id> is a hash of (reference pixels, input pixels, wire count, band), so checking the same
pair again (e.g. with another tolerance) updates that entry instead of adding duplicates.
"""
import csv
import hashlib
import io
import json
import os
import re
import shutil
import time
import zipfile

import cv2

from backend.library import APP_DIR
from backend.verdict import public
from backend.visualize import annotate

HISTORY_DIR = os.environ.get("WHC_HISTORY_DIR", os.path.join(APP_DIR, "history"))
MAX_ENTRIES = 500                      # oldest entries are removed beyond this
_ID_RE = re.compile(r"^[0-9a-f]{16}$")
IMAGE_NAMES = ("reference", "input", "reference_annotated", "input_annotated")


def run_key(ref_img, test_img, num_wires, band=None):
    h = hashlib.sha1()
    h.update(ref_img.tobytes())
    h.update(test_img.tobytes())
    h.update(f"{int(num_wires)}|{list(band) if band else None}".encode())
    return h.hexdigest()[:16]


def _dir(run_id):
    if not _ID_RE.match(str(run_id)):
        raise ValueError("bad history id")
    return os.path.join(HISTORY_DIR, run_id)


def _imwrite(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def save_run(run_id, ref_img, test_img, res, meta):
    """Store (or update) one check. `meta` = reference_name, reference_source, input_name,
    input_source, tolerance, band, wire_px."""
    d = _dir(run_id)
    os.makedirs(d, exist_ok=True)
    _imwrite(os.path.join(d, "reference.jpg"), ref_img)
    _imwrite(os.path.join(d, "input.jpg"), test_img)
    for stale in ("reference_annotated.jpg", "input_annotated.jpg"):
        p = os.path.join(d, stale)
        if os.path.exists(p):
            os.remove(p)
    if "_test_colors" in res:
        rows = res.get("wires") or None
        _imwrite(os.path.join(d, "reference_annotated.jpg"),
                 annotate(ref_img, res["_ref_colors"], res["_ref_dbg"]))
        _imwrite(os.path.join(d, "input_annotated.jpg"),
                 annotate(test_img, res["_test_colors"], res["_test_dbg"], rows))
    rec = dict(public(res))
    rec.update(meta)
    rec["id"] = run_id
    rec["created_at"] = time.time()
    tmp = os.path.join(d, "result.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, os.path.join(d, "result.json"))
    _prune()
    return rec


def _load(run_id):
    try:
        with open(os.path.join(_dir(run_id), "result.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def list_runs(verdict=None, text=None):
    """Newest first. verdict: 'RIGHT' | 'WRONG' | 'REVIEW' | None. text: substring filter."""
    if not os.path.isdir(HISTORY_DIR):
        return []
    runs = [r for r in (_load(n) for n in os.listdir(HISTORY_DIR) if _ID_RE.match(n)) if r]
    if verdict == "REVIEW":
        runs = [r for r in runs if r.get("needs_review")]
    elif verdict:
        runs = [r for r in runs if r.get("verdict") == verdict]
    if text:
        t = text.lower()
        runs = [r for r in runs if t in (str(r.get("reference_name", "")) + " "
                                         + str(r.get("input_name", "")) + " "
                                         + str(r.get("reason", ""))).lower()]
    return sorted(runs, key=lambda r: -float(r.get("created_at", 0)))


def image_bytes(run_id, name):
    """Raw JPEG bytes of one stored image, or None."""
    if name not in IMAGE_NAMES:
        return None
    try:
        with open(os.path.join(_dir(run_id), name + ".jpg"), "rb") as f:
            return f.read()
    except OSError:
        return None


def delete_run(run_id):
    d = _dir(run_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def clear_history():
    n = 0
    if os.path.isdir(HISTORY_DIR):
        for name in os.listdir(HISTORY_DIR):
            if _ID_RE.match(name):
                shutil.rmtree(os.path.join(HISTORY_DIR, name), ignore_errors=True)
                n += 1
    return n


def _prune():
    runs = list_runs()
    for r in runs[MAX_ENTRIES:]:
        delete_run(r["id"])


def _stamp(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))


def export_csv(runs):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "verdict", "confidence_pct", "needs_review", "reference", "input",
                "wires", "tolerance_dE", "reason", "id"])
    for r in runs:
        w.writerow([_stamp(r.get("created_at", 0)), r.get("verdict"),
                    round(float(r.get("confidence", 0)) * 100, 1), r.get("needs_review"),
                    r.get("reference_name"), r.get("input_name"), r.get("num_wires"),
                    r.get("tolerance"), r.get("reason"), r.get("id")])
    return buf.getvalue().encode("utf-8-sig")        # BOM so Excel opens it correctly


def export_zip(runs):
    """ZIP with history.csv plus one folder per check (images + result.json)."""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("history.csv", export_csv(runs))
        for r in runs:
            d = _dir(r["id"])
            for fn in os.listdir(d):
                if fn.endswith((".jpg", ".json")):
                    z.write(os.path.join(d, fn), f"{r['id']}/{fn}")
    return mem.getvalue()
