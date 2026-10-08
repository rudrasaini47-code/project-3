import os
import sys
import time
import traceback

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st

st.set_page_config(page_title="Wire Harness Checker", page_icon="🔌", layout="wide")

# Streamlit Cloud hides import errors. Catch them and show the real cause on the page.
try:
    import cv2
    import numpy as np
    import pandas as pd

    from backend import history, library
    from backend.multi_pin_bfs import detect_wire_colors_bfs
    from backend.reference import DEFAULT_MAX_DELTA_E
    from backend.verdict import check_harness, REVIEW_BELOW
    from backend.visualize import annotate
except Exception:
    st.error("The app could not start because of an import error. Details below.")
    st.code(traceback.format_exc())
    st.write("App folder:", APP_DIR)
    st.write("Files here:", sorted(os.listdir(APP_DIR)))
    bdir = os.path.join(APP_DIR, "backend")
    st.write("backend/ exists:", os.path.isdir(bdir),
             "| files:", sorted(os.listdir(bdir)) if os.path.isdir(bdir) else "-")
    st.stop()

SRC_LIB, SRC_UP = "📚 Pick from library", "⬆️ Upload my own"


# ----------------------------------------------------------------------------- helpers
def load_image(file):
    if file is None:
        return None
    return cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def show(img_bgr, caption):
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=caption, width="stretch")


def when(ts):
    return time.strftime("%d %b %Y, %H:%M:%S", time.localtime(float(ts)))


def band_text(entry_or_band):
    b = entry_or_band
    return f"rows {b[0]}–{b[1]}" if b else "automatic"


def library_picker(label, key):
    """Selectbox over the library. Returns (entry, image) or (None, None)."""
    refs = library.list_references()
    if not refs:
        st.info("The library is empty. Add a reference on the 'Reference library' page.")
        return None, None
    by_id = {e["id"]: e for e in refs}
    ref_id = st.selectbox(label, list(by_id), key=key,
                          format_func=lambda i: f"{by_id[i]['name']} ({by_id[i]['num_wires']} wires)")
    entry, img = library.get_reference(ref_id)
    if img is None:
        st.error("That image file is missing from the library folder.")
        return None, None
    return entry, img


def band_inputs(prefix):
    """Advanced detection settings. Returns (band or None, wire_px or None)."""
    with st.expander("Advanced: where to read the wires"):
        st.caption("Leave this off for connectors where the wires run into the bottom of the "
                   "picture. Turn it on when the wires are coloured/spaced apart or do not end "
                   "inside the photo (e.g. 3-wire connectors): pick rows that cross only wires. "
                   "The yellow box in the annotated image shows the rows used.")
        manual = st.checkbox("Choose rows manually", key=f"{prefix}_manual")
        c1, c2, c3 = st.columns(3)
        y1 = c1.number_input("Top row", 0, 4000, 20, 5, key=f"{prefix}_y1", disabled=not manual)
        y2 = c2.number_input("Bottom row", 0, 4000, 60, 5, key=f"{prefix}_y2", disabled=not manual)
        wpx = c3.number_input("Wire width px (0 = default)", 0.0, 200.0, 0.0, 1.0,
                              key=f"{prefix}_wpx", help="Only used for the wire-count check.")
    band = None
    if manual:
        if y2 <= y1:
            st.warning("Bottom row must be larger than top row. Using automatic rows instead.")
        else:
            band = (int(y1), int(y2))
    return band, (float(wpx) if wpx > 0 else None)


def save_reference_ui(img, n, band, wire_px, key):
    """Name box + button that stores `img` in the library (only if it reads correctly)."""
    name = st.text_input("Name", placeholder="e.g. 4-wire sensor cable, red/black/white/yellow",
                         key=f"{key}_name")
    if st.button("💾 Save to library", key=f"{key}_save", disabled=img is None):
        count, _ = detect_wire_colors_bfs(img, n, band=band, nominal_wire_px=wire_px)
        probe = check_harness(img, img, n, band=band, wire_px=wire_px)
        if probe.get("reference_error") or count != n:
            st.error(f"This image does not read as {n} wires, so it would be a bad reference. "
                     "Check the wire count or the rows under 'Advanced'.\n\n" + probe["reason"])
            return
        try:
            e = library.add_reference(name, img, n, band, wire_px)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success(f"Saved '{e['name']}'. You can now pick it from the library.")


# ----------------------------------------------------------------------------- page: check
def page_check():
    st.title("🔌 Wire Harness Checker")
    st.caption("Pick a reference (or upload one), give the photo to check, and get RIGHT or WRONG "
               "with a confidence score. Every check is saved to History.")

    max_de = st.session_state["max_de"]
    use_samples = st.session_state["use_samples"]

    ref_img = test_img = None
    ref_name = test_name = ""
    ref_source = test_source = ""
    band = wire_px = None
    n_wires = 6

    col_a, col_b, col_c = st.columns([2, 2, 1])

    with col_a:
        st.subheader("1 · Reference image")
        if use_samples:
            ref_img = cv2.imread(os.path.join(APP_DIR, "sample_images", "reference_sample.png"))
            ref_name, ref_source = "Sample reference", "sample"
            st.caption("Using the sample reference (untick 'Use sample images' in the sidebar to change).")
        else:
            src = st.radio("Reference source", [SRC_LIB, SRC_UP], horizontal=True,
                           key="ref_src", label_visibility="collapsed")
            if src == SRC_LIB:
                entry, ref_img = library_picker("Reference from library", "ref_pick")
                if entry:
                    ref_name, ref_source = entry["name"], "library"
                    n_wires = int(entry["num_wires"])
                    band = tuple(entry["band"]) if entry.get("band") else None
                    wire_px = entry.get("wire_px")
                    st.caption(f"{n_wires} wires · reading {band_text(entry.get('band'))}")
                    show(ref_img, "Selected reference")
            else:
                ref_file = st.file_uploader("Known-good harness", type=["png", "jpg", "jpeg"], key="ref")
                ref_img = load_image(ref_file)
                if ref_file:
                    ref_name, ref_source = ref_file.name, "upload"

    with col_c:
        st.subheader("3 · Wires")
        if use_samples:
            n_wires = int(st.number_input("Number of wires", 1, 40, 6, 1, key="n_sample"))
        elif st.session_state.get("ref_src", SRC_LIB) == SRC_LIB:
            st.metric("Number of wires", n_wires)
            st.caption("Set by the library entry.")
        else:
            n_wires = int(st.number_input("Number of wires", 1, 40, 6, 1, key="n_upload"))

    if not use_samples and st.session_state.get("ref_src", SRC_LIB) == SRC_UP:
        with col_a:
            band, wire_px = band_inputs("chk")
            if ref_img is not None:
                with st.expander("💾 Save this reference to the library"):
                    save_reference_ui(ref_img, n_wires, band, wire_px, "chk_ref")

    with col_b:
        st.subheader("2 · Input image")
        if use_samples:
            test_img = cv2.imread(os.path.join(APP_DIR, "sample_images", "test_sample.png"))
            test_name, test_source = "Sample test image", "sample"
        else:
            tsrc = st.radio("Input source", [SRC_UP, SRC_LIB], horizontal=True,
                            key="test_src", label_visibility="collapsed")
            if tsrc == SRC_UP:
                test_file = st.file_uploader("Harness to check", type=["png", "jpg", "jpeg"], key="test")
                test_img = load_image(test_file)
                if test_file:
                    test_name, test_source = test_file.name, "upload"
            else:
                st.caption("Handy for trying the checker without a camera "
                           "(e.g. pick a different library image to see a WRONG result).")
                entry, test_img = library_picker("Input from library", "test_pick")
                if entry:
                    test_name, test_source = entry["name"], "library"
                    show(test_img, "Selected input")

    st.markdown("---")
    if ref_img is None or test_img is None:
        st.info("Choose a reference and an input image (or tick 'Use sample images' in the sidebar).")
        return

    res = check_harness(ref_img, test_img, n_wires, max_delta_e=max_de, band=band, wire_px=wire_px)
    conf = res["confidence"]

    # ---- save to history (once per image pair + settings, not on every rerun)
    if st.session_state["auto_save"]:
        run_id = history.run_key(ref_img, test_img, n_wires, band)
        sig = (run_id, max_de, wire_px)
        if st.session_state.get("_saved_sig") != sig:
            try:
                history.save_run(run_id, ref_img, test_img, res, {
                    "reference_name": ref_name, "reference_source": ref_source,
                    "input_name": test_name, "input_source": test_source,
                    "tolerance": max_de, "band": list(band) if band else None, "wire_px": wire_px})
                st.session_state["_saved_sig"] = sig
            except OSError as exc:
                st.warning(f"Could not save to history: {exc}")

    head1, head2 = st.columns([1, 2])
    with head1:
        if res.get("reference_error"):
            st.warning("⚠️ REFERENCE PROBLEM")
        elif res["verdict"] == "RIGHT":
            st.success("## ✅ RIGHT")
        else:
            st.error("## ❌ WRONG")
    with head2:
        st.metric("Confidence", f"{conf * 100:.0f}%")
        st.progress(float(conf))
        if res["needs_review"]:
            st.caption(f"⚠️ Below {REVIEW_BELOW * 100:.0f}%: please review this one manually.")
        st.caption("Confidence is a heuristic score, not a calibrated probability.")
        if st.session_state["auto_save"] and st.session_state.get("_saved_sig"):
            st.page_link(PAGE_HISTORY, label="💾 Saved to history", icon="🕘")
    st.write(res["reason"])

    if "_test_colors" in res:
        rows = res["wires"]
        i1, i2 = st.columns(2)
        with i1:
            show(annotate(ref_img, res["_ref_colors"], res["_ref_dbg"]), "Reference (detected wires)")
        with i2:
            show(annotate(test_img, res["_test_colors"], res["_test_dbg"], rows or None),
                 "Input (detected wires, X = mismatch)")
        if rows:
            df = pd.DataFrame([{
                "Wire": r["wire"], "Expected": r["expected"], "Detected": r["detected"],
                "Delta E": r["delta_e"], "Match chance": f"{r['match_probability'] * 100:.0f}%",
                "Result": "OK" if r["match"] else "FAIL"} for r in rows])
            st.dataframe(df.style.apply(
                lambda row: ["background-color:#5c1f1f" if row["Result"] == "FAIL" else "" for _ in row],
                axis=1), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- page: history
PAGE_SIZE = 8


def page_history():
    st.title("🕘 History")
    everything = history.list_runs()
    if not everything:
        st.info("No checks saved yet. Run a check on the 'Check' page and it appears here "
                "(make sure 'Save every check to history' is ticked in the sidebar).")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Checks", len(everything))
    m2.metric("✅ RIGHT", sum(r["verdict"] == "RIGHT" for r in everything))
    m3.metric("❌ WRONG", sum(r["verdict"] == "WRONG" for r in everything))
    m4.metric("⚠️ Needs review", sum(bool(r.get("needs_review")) for r in everything))

    f1, f2 = st.columns([2, 3])
    choice = f1.segmented_control("Show", ["All", "RIGHT", "WRONG", "Needs review"],
                                  default="All", key="hist_filter") or "All"
    query = f2.text_input("Search reference / input name / reason", key="hist_q")
    runs = history.list_runs({"All": None, "Needs review": "REVIEW"}.get(choice, choice), query or None)

    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    b1.download_button("⬇️ CSV (table)", history.export_csv(runs), "wire_check_history.csv",
                       "text/csv", disabled=not runs, width="stretch")
    b2.download_button("⬇️ ZIP (with images)", history.export_zip(runs) if runs else b"",
                       "wire_check_history.zip", "application/zip", disabled=not runs,
                       width="stretch")
    with b3.popover("🗑️ Clear all", width="stretch"):
        st.write(f"Delete all {len(everything)} saved checks and their images?")
        if st.button("Yes, delete everything", type="primary", key="hist_clear"):
            history.clear_history()
            st.rerun()
    st.caption("History is stored in the app's `history/` folder. On Streamlit Community Cloud "
               "that disk is reset when the app restarts, so download the ZIP if you need a "
               "permanent copy. Times are in the server's time zone.")

    if not runs:
        st.warning("Nothing matches this filter.")
        return

    pages = max(1, -(-len(runs) // PAGE_SIZE))
    page = 1
    if pages > 1:
        page = st.number_input(f"Page (1–{pages})", 1, pages, 1, 1, key="hist_page")
    for r in runs[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
        render_run(r)


def render_run(r):
    icon = "⚠️" if r.get("reference_error") else ("✅" if r["verdict"] == "RIGHT" else "❌")
    with st.container(border=True):
        t1, t2, t3 = st.columns([1.2, 1.2, 3])
        for col, name, cap in ((t1, "reference", "Reference"), (t2, "input", "Input")):
            data = history.image_bytes(r["id"], name)
            if data:
                col.image(data, caption=cap, width="stretch")
        with t3:
            st.markdown(f"### {icon} {r['verdict']} · {r['confidence'] * 100:.0f}%"
                        + ("  ⚠️ review" if r.get("needs_review") else ""))
            st.caption(when(r.get("created_at", 0)))
            st.write(f"**Reference:** {r.get('reference_name') or '-'}  \n"
                     f"**Input:** {r.get('input_name') or '-'}  \n"
                     f"**Wires:** {r.get('num_wires')} · **Tolerance:** ΔE {r.get('tolerance')}")
            st.write(r.get("reason", ""))
        with st.expander("Details: annotated images and per-wire table"):
            a1, a2 = st.columns(2)
            for col, name, cap in ((a1, "reference_annotated", "Reference (detected wires)"),
                                   (a2, "input_annotated", "Input (detected wires)")):
                data = history.image_bytes(r["id"], name)
                if data:
                    col.image(data, caption=cap, width="stretch")
            if r.get("wires"):
                st.dataframe(pd.DataFrame([{
                    "Wire": w["wire"], "Expected": w["expected"], "Detected": w["detected"],
                    "Delta E": w["delta_e"], "Match chance": f"{w['match_probability'] * 100:.0f}%",
                    "Result": "OK" if w["match"] else "FAIL"} for w in r["wires"]]),
                    hide_index=True, width="stretch")
        if st.button("Delete this entry", key=f"del_{r['id']}"):
            history.delete_run(r["id"])
            st.rerun()


# ----------------------------------------------------------------------------- page: library
def page_library():
    st.title("📚 Reference library")
    st.caption("Known-good photos you can pick on the Check page. The wire count and reading "
               "settings are stored with each one. References you add are saved in the app's "
               "`reference_library/` folder (on Streamlit Community Cloud, commit them to GitHub "
               "to keep them after a restart).")
    refs = library.list_references()
    cols = st.columns(3)
    for i, e in enumerate(refs):
        _, img = library.get_reference(e["id"])
        with cols[i % 3].container(border=True):
            if img is not None:
                show(img, e["name"])
            st.write(f"**{e['num_wires']} wires** · reading {band_text(e.get('band'))}")
            if e.get("builtin"):
                st.caption("Built-in")
            elif st.button("Delete", key=f"libdel_{e['id']}"):
                library.delete_reference(e["id"])
                st.rerun()

    st.markdown("---")
    st.subheader("Add your own reference")
    f = st.file_uploader("Photo of a known-good harness", type=["png", "jpg", "jpeg"], key="lib_up")
    c1, c2 = st.columns([1, 3])
    n = int(c1.number_input("Number of wires", 1, 40, 6, 1, key="lib_n"))
    with c2:
        band, wire_px = band_inputs("lib")
    img = load_image(f)
    if img is not None:
        count, cols_found, dbg = detect_wire_colors_bfs(img, n, return_debug=True,
                                                        band=band, nominal_wire_px=wire_px)
        if cols_found:
            show(annotate(img, cols_found, dbg), f"How the checker reads this photo ({count} wires found)")
        else:
            st.warning("No wires could be found in this photo.")
    save_reference_ui(img, n, band, wire_px, "lib")


# ----------------------------------------------------------------------------- navigation
PAGE_CHECK = st.Page(page_check, title="Check", icon="🔍", url_path="check", default=True)
PAGE_HISTORY = st.Page(page_history, title="History", icon="🕘", url_path="history")
PAGE_LIBRARY = st.Page(page_library, title="Reference library", icon="📚", url_path="library")
nav = st.navigation([PAGE_CHECK, PAGE_HISTORY, PAGE_LIBRARY])

with st.sidebar:
    st.header("Settings")
    st.slider("Colour tolerance (Delta E)", 5.0, 40.0, float(DEFAULT_MAX_DELTA_E), 0.5, key="max_de",
              help="Lower = stricter. Same wire across photos is ~2-7; different wires are usually 24+.")
    st.checkbox("Use sample images", key="use_samples")
    st.checkbox("Save every check to history", value=True, key="auto_save")
    st.caption("Photograph reference and test from the same side with the same camera.")

nav.run()
