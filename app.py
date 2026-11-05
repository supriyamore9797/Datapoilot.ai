"""DataPilot AI - entry point and Home page.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import importlib
import streamlit as st

from src import config, state, ui

importlib.reload(config)
importlib.reload(ui)
from src.config import ALLOWED_EXTENSIONS, APP_NAME, APP_TAGLINE, MAX_FILE_MB, MAX_ROWS
from src.data_loader import DataLoadError, load_dataframe
from src.profiling import dataset_overview
from src.sample_data import SAMPLES, load_sample
from src.utils import human_bytes

ui.configure_page()
state.init_state()


def _load_upload(uploaded) -> None:
    """Load an uploaded file once per distinct upload (reruns must not wipe the user's work)."""
    upload_id = f"{uploaded.name}:{uploaded.size}:{getattr(uploaded, 'file_id', '')}"
    if st.session_state.get("last_upload_id") == upload_id:
        return
    st.session_state["last_upload_id"] = upload_id
    try:
        result = load_dataframe(uploaded.getvalue(), uploaded.name)
    except DataLoadError as exc:
        st.session_state["upload_error"] = str(exc)
        return
    st.session_state.pop("upload_error", None)
    state.load_dataset(result.df, result.filename, "upload", result.warnings)
    ui.set_flash("success", f"Loaded '{result.filename}' ({len(result.df):,} rows, {result.df.shape[1]} columns).")
    st.rerun()


def home() -> None:
    ui.setup_page(APP_NAME, f"{APP_TAGLINE}: upload a dataset, explore it, clean it, visualise it, train models and export a report - locally, without writing code.")
    ui.show_flash()

    left, right = st.columns([1, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("1. Upload a dataset")
        uploaded = st.file_uploader(
            f"CSV or Excel file (up to {MAX_FILE_MB} MB / {MAX_ROWS:,} rows)",
            type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
            help="The file is read in memory on this computer and is never sent anywhere.",
        )
        if uploaded is not None:
            _load_upload(uploaded)
        if st.session_state.get("upload_error"):
            st.error(st.session_state["upload_error"])
    with right, st.container(border=True):
        st.subheader("... or load a sample dataset")
        for key, info in SAMPLES.items():
            cols = st.columns([3, 1])
            cols[0].markdown(f"**{info.title}**  \n{info.description}")
            cols[0].caption(f"Suggested target: `{info.suggested_target}` ({info.suggested_task}). Defects: {info.defects}")
            if cols[1].button("Load", key=f"load_{key}"):
                state.load_dataset(load_sample(key), info.filename, "sample")
                st.session_state.pop("upload_error", None)
                ui.set_flash("success", f"Loaded sample dataset '{info.title}'.")
                st.rerun()

    if state.has_data():
        df = state.get_working()
        overview = dataset_overview(df)
        st.subheader("Current dataset")
        ui.data_badges()
        ui.metric_cards([
            ("Rows", f"{overview['rows']:,}", None),
            ("Columns", overview["columns"], f"{overview['numeric_columns']} numeric, {overview['categorical_columns']} categorical"),
            ("Missing cells", f"{overview['missing_cells']:,}", f"{overview['missing_pct']}% of all cells"),
            ("Duplicate rows", f"{overview['duplicate_rows']:,}", f"{overview['duplicate_pct']}% of rows"),
            ("Memory", human_bytes(overview["memory_bytes"]), None),
        ])
        for note in st.session_state.get("load_warnings", []):
            st.warning(note)
        st.dataframe(ui.arrow_safe(df.head(10)), hide_index=True)
        st.caption("Preview of the first 10 rows of the working data. Use the sidebar to open the other pages.")
    else:
        st.info("Start by uploading a file or loading a sample dataset above. Nothing is loaded yet.")

    st.subheader("Workflow")
    steps = ["Upload", "Explore", "Clean", "Visualise", "Student Analysis", "Train models", "Export report"]
    for column, (number, label) in zip(st.columns(len(steps)), enumerate(steps, start=1)):
        column.markdown(f'<div class="dp-step"><b>{number}.</b> {label}</div>', unsafe_allow_html=True)
    st.caption("DataPilot AI reports facts computed from your data and clearly labelled recommendations. It is not an autonomous AI scientist.")
    ui.show_footer()


home_page = st.Page(home, title="Home", icon="\U0001F3E0", default=True, url_path="home")
ui.set_home_page(home_page)

pages = [
    home_page,
    st.Page("pages/1_Explore_Data.py", title="Explore Data", icon="\U0001F50E"),
    st.Page("pages/2_Clean_Data.py", title="Clean Data", icon="\U0001F9F9"),
    st.Page("pages/3_Visual_Analytics.py", title="Visual Analytics", icon="\U0001F4C8"),
    st.Page("pages/6_Student_Analysis.py", title="Student Data Analysis", icon="\U0001F393"),
    st.Page("pages/4_ML_Studio.py", title="ML Studio", icon="\U0001F916"),
    st.Page("pages/5_Reports.py", title="Reports", icon="\U0001F4C4"),
]
navigation = st.navigation(pages)
ui.sidebar_status()
navigation.run()
