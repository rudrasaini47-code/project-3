"""Reference library: named reference photos + the settings needed to read them.

Files live in  reference_library/  (override with env WHC_LIBRARY_DIR):
  library.json   list of entries {id, name, file, num_wires, band, wire_px, builtin}
  <file>.png     the photo

band     [y1, y2] rows the wires are read from, or null for automatic.
         Needed for connectors where the wires do not end inside the picture (e.g. 3-wire
         connectors with a coloured body), because the automatic search looks for the wire end.
wire_px  wire pitch in px for the wire-count check, or null for the default.
"""
import json
import os
import re
import time
import uuid

import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_DIR = os.environ.get("WHC_LIBRARY_DIR", os.path.join(APP_DIR, "reference_library"))
MANIFEST = os.path.join(LIB_DIR, "library.json")


def _read():
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _write(entries):
    os.makedirs(LIB_DIR, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp, MANIFEST)


def list_references():
    """Entries whose image file still exists, built-ins first, then newest."""
    out = [e for e in _read() if os.path.isfile(os.path.join(LIB_DIR, e.get("file", "")))]
    return sorted(out, key=lambda e: (not e.get("builtin"), -float(e.get("created", 0))))


def get_reference(ref_id):
    """-> (entry, BGR image) or (None, None)."""
    for e in _read():
        if e.get("id") == ref_id:
            img = cv2.imread(os.path.join(LIB_DIR, e["file"]), cv2.IMREAD_COLOR)
            return (e, img) if img is not None else (None, None)
    return None, None


def _slug(text):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s or "ref")[:30]


def add_reference(name, img_bgr, num_wires, band=None, wire_px=None):
    """Save a new reference. Returns the entry. Raises ValueError on bad input."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Please give the reference a name.")
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        raise ValueError("The image could not be read.")
    if any(e.get("name", "").lower() == name.lower() for e in _read()):
        raise ValueError(f"A reference called '{name}' already exists.")
    ref_id = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
    fname = ref_id + ".png"
    os.makedirs(LIB_DIR, exist_ok=True)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("The image could not be saved.")
    with open(os.path.join(LIB_DIR, fname), "wb") as f:
        f.write(buf.tobytes())
    entry = {"id": ref_id, "name": name, "file": fname, "num_wires": int(num_wires),
             "band": [int(band[0]), int(band[1])] if band else None,
             "wire_px": float(wire_px) if wire_px else None,
             "builtin": False, "created": time.time()}
    _write(_read() + [entry])
    return entry


def delete_reference(ref_id):
    """Delete a user-added reference. Built-in ones are protected."""
    entries = _read()
    keep, gone = [], None
    for e in entries:
        if e.get("id") == ref_id and not e.get("builtin"):
            gone = e
        else:
            keep.append(e)
    if gone is None:
        return False
    _write(keep)
    try:
        os.remove(os.path.join(LIB_DIR, gone["file"]))
    except OSError:
        pass
    return True
