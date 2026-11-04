"""Student Data Analysis page: filter, search, inspect and export student cohorts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import state, student_filter, ui
from src.cleaning import to_csv_bytes, to_xlsx_bytes
from src.data_loader import DataLoadError, load_dataframe
from src.student_filter import (
    build_breakdown_chart,
    compute_column_breakdown,
    detect_student_columns,
    filter_students,
    generate_student_insights,
    get_unique_filter_values,
    make_college_roster_sample,
    summarize_filter_results,
)

ui.setup_page(
    "Student Data Analysis",
    "Filter, search, inspect and export student cohorts across districts, departments, and academic years without altering the underlying dataset.",
)
ui.show_flash()

# ---------------------------------------------------------------- dataset resolution
if not state.has_data():
    st.info("No dataset is loaded yet. Upload a college CSV/Excel file below or load the built-in demo college student roster.")
    col_demo, col_up = st.columns([1, 1], gap="medium")
    with col_demo, st.container(border=True):
        st.subheader("Demo College Dataset")
        st.markdown(
            "Load a realistic cohort of **500 Maharashtra engineering students** featuring districts (Pune, Nashik, Satara, Mumbai, etc.), "
            "departments (AIML, Computer, Mechanical, etc.), academic years, and admission details."
        )
        if st.button("Load Demo College Roster (500 Students)", type="primary", key="load_demo_roster_empty"):
            demo_df = make_college_roster_sample()
            state.load_dataset(demo_df, "maharashtra_college_students.csv", "sample")
            ui.set_flash("success", "Loaded Demo College Dataset (500 students across 8 Maharashtra districts).")
            st.rerun()

    with col_up, st.container(border=True):
        st.subheader("Upload Student File")
        uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], key="stu_empty_upload")
        if uploaded is not None:
            try:
                res = load_dataframe(uploaded.getvalue(), uploaded.name)
                state.load_dataset(res.df, res.filename, "upload", res.warnings)
                ui.set_flash("success", f"Loaded '{res.filename}' ({len(res.df):,} rows).")
                st.rerun()
            except DataLoadError as exc:
                st.error(str(exc))
    st.stop()

# Data is present in state
df = state.get_working()
ui.data_badges()

# Option to switch to demo dataset or upload a different file
with st.expander("Switch or Upload Dataset"):
    sw_col1, sw_col2 = st.columns(2)
    with sw_col1:
        st.caption("Want to demo Maharashtra student data?")
        if st.button("Load 500-Student College Demo Roster", key="load_demo_roster_exist"):
            demo_df = make_college_roster_sample()
            state.load_dataset(demo_df, "maharashtra_college_students.csv", "sample")
            ui.set_flash("success", "Loaded Demo College Dataset (Pune, Nashik, Satara, etc.).")
            st.rerun()
    with sw_col2:
        new_upload = st.file_uploader("Upload new student file", type=["csv", "xlsx"], key="stu_exist_upload")
        if new_upload is not None:
            try:
                res = load_dataframe(new_upload.getvalue(), new_upload.name)
                state.load_dataset(res.df, res.filename, "upload", res.warnings)
                ui.set_flash("success", f"Loaded '{res.filename}' ({len(res.df):,} rows).")
                st.rerun()
            except DataLoadError as exc:
                st.error(str(exc))

# ---------------------------------------------------------------- column detection
col_info = detect_student_columns(df)
rec_filters = col_info["recommended_filters"]
all_candidates = col_info["all_filter_candidates"]
search_candidates = col_info["searchable_columns"]

st.divider()
st.subheader("1. Interactive Filters & Search")

filter_col_left, filter_col_right = st.columns([3, 1])
with filter_col_left:
    active_filter_cols = st.multiselect(
        "Select columns to filter by",
        options=all_candidates,
        default=rec_filters if rec_filters else all_candidates[:3],
        help="Choose any demographic or academic field to filter students. Detection automatically recommends likely fields.",
        key="stu_filter_columns_selector",
    )

with filter_col_right:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("Clear All Filters", key="btn_clear_filters", help="Reset all filter selections and show all student records"):
        # Clear filter state keys
        for col in all_candidates:
            st.session_state.pop(f"stu_filter_val_{col}", None)
        st.session_state.pop("stu_search_query", None)
        st.rerun()

# Build filter inputs
filters_dict: dict[str, list[str]] = {}

if active_filter_cols:
    num_cols = min(len(active_filter_cols), 3)
    cols_layout = st.columns(num_cols)
    for idx, col in enumerate(active_filter_cols):
        target_layout = cols_layout[idx % num_cols]
        unique_vals = get_unique_filter_values(df[col])
        with target_layout:
            selected_vals = st.multiselect(
                f"Filter by **{col}**",
                options=unique_vals,
                default=st.session_state.get(f"stu_filter_val_{col}", []),
                key=f"stu_filter_val_{col}",
                help=f"Select one or more values. Leave blank to include all {col}s.",
            )
            if selected_vals:
                filters_dict[col] = selected_vals

# Search Bar
search_c1, search_c2 = st.columns([3, 2])
with search_c1:
    search_query = st.text_input(
        "Search records",
        placeholder="Type student name, roll number, PRN, or ID...",
        key="stu_search_query",
    )
with search_c2:
    selected_search_cols = st.multiselect(
        "Search within column(s)",
        options=search_candidates or list(df.columns),
        default=search_candidates[:3] if search_candidates else list(df.columns)[:2],
        key="stu_search_columns",
        help="Search will look for your text across these columns.",
    )

# ---------------------------------------------------------------- filtering execution
filtered_df = filter_students(
    df,
    filters=filters_dict,
    search_query=search_query,
    search_columns=selected_search_cols,
)

metrics = summarize_filter_results(len(df), len(filtered_df))

# ---------------------------------------------------------------- results presentation
st.divider()
st.subheader("2. Filtered Student Results")

ui.metric_cards([
    ("Total Records", f"{metrics['total_records']:,}", "All records in working dataset"),
    ("Matching Records", f"{metrics['matching_records']:,}", f"{metrics['match_percentage']}% of cohort"),
    ("Excluded Records", f"{metrics['excluded_records']:,}", f"{100.0 - metrics['match_percentage']:.1f}% excluded"),
    ("Active Filters", len(filters_dict) + (1 if search_query.strip() else 0), "AND logic applied"),
])

# Show active filter pills/badges
active_badges_html = []
for col, vals in filters_dict.items():
    active_badges_html.append(ui.badge(f"{col}: {', '.join(vals)}"))
if search_query.strip():
    active_badges_html.append(ui.badge(f"Search: '{search_query.strip()}'"))

if active_badges_html:
    st.markdown("<b>Active criteria:</b> " + " ".join(active_badges_html), unsafe_allow_html=True)
else:
    st.caption("No filters active — displaying the complete student dataset.")

if filtered_df.empty:
    st.warning(
        "No student records match the active criteria. "
        "Try removing one of your filter selections or clicking **Clear All Filters**."
    )
else:
    st.dataframe(ui.arrow_safe(filtered_df), hide_index=True)
    st.caption(f"Displaying {len(filtered_df):,} matching record(s) out of {len(df):,} total records.")

# ---------------------------------------------------------------- summary & charts
st.divider()
st.subheader("3. Demographic Breakdown & Analysis")

chart_tab, summary_tab = st.tabs(["Visual Breakdown Charts", "Summary Tables & Takeaways"])

with chart_tab:
    if filtered_df.empty:
        st.info("No matching records to display charts.")
    else:
        chart_cand = [c for c in active_filter_cols if c in filtered_df.columns]
        if not chart_cand:
            chart_cand = [c for c in all_candidates if c in filtered_df.columns]

        if chart_cand:
            chart_c1, chart_c2, chart_c3 = st.columns([2, 1, 1])
            with chart_c1:
                plot_col = st.selectbox("Select attribute to visualize", chart_cand, key="stu_plot_col")
            with chart_c2:
                plot_type = st.selectbox("Chart Type", ["Bar chart", "Pie chart"], key="stu_plot_type")
            with chart_c3:
                top_items = st.slider("Top categories", 5, 30, 10, key="stu_plot_top")

            type_key = "pie" if plot_type == "Pie chart" else "bar"
            fig = build_breakdown_chart(filtered_df, plot_col, chart_type=type_key, top_n=top_items)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns available for demographic breakdown.")

with summary_tab:
    if filtered_df.empty:
        st.info("No matching records to generate summary.")
    else:
        st.markdown("#### Key Cohort Observations")
        insights = generate_student_insights(filtered_df, key_columns=active_filter_cols or rec_filters)
        for insight in insights:
            st.markdown(f"- {insight}")

        st.markdown("#### Frequency Breakdown")
        if all_candidates:
            tbl_col = st.selectbox("View frequency table for", [c for c in all_candidates if c in filtered_df.columns], key="stu_tbl_col")
            breakdown_tbl = compute_column_breakdown(filtered_df, tbl_col)
            st.dataframe(breakdown_tbl, hide_index=True)

# ---------------------------------------------------------------- export section
st.divider()
st.subheader("4. Export Filtered Records")
st.write(
    f"Download the **{len(filtered_df):,} matching student record(s)**. "
    "The exported file contains exclusively the filtered subset."
)

raw_name = (st.session_state.get("dataset_name") or "students").rsplit(".", 1)[0]
export_c1, export_c2 = st.columns(2)

with export_c1:
    st.download_button(
        "Download Filtered CSV",
        data=to_csv_bytes(filtered_df),
        file_name=f"{raw_name}_filtered.csv",
        mime="text/csv",
        disabled=filtered_df.empty,
        type="primary",
        key="btn_download_filtered_csv",
    )

with export_c2:
    st.download_button(
        "Download Filtered Excel (.xlsx)",
        data=to_xlsx_bytes(filtered_df),
        file_name=f"{raw_name}_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=filtered_df.empty,
        key="btn_download_filtered_xlsx",
    )

ui.show_footer()

