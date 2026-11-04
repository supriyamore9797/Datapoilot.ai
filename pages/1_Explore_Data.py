"""Explore Data page: profile the current working dataset."""

import pandas as pd
import streamlit as st

from src import profiling, ui, visualization
from src.config import PREVIEW_ROWS
from src.utils import human_bytes

ui.setup_page("Explore Data", "Understand the structure, quality and distributions of your dataset. Every number is computed from the current data.")
df = ui.require_data()

overview = profiling.dataset_overview(df)
ui.metric_cards([
    ("Rows", f"{overview['rows']:,}", None),
    ("Columns", overview["columns"], None),
    ("Numeric", overview["numeric_columns"], None),
    ("Categorical", overview["categorical_columns"], None),
    ("Missing cells", f"{overview['missing_cells']:,}", f"{overview['missing_pct']}%"),
    ("Duplicate rows", f"{overview['duplicate_rows']:,}", f"{overview['duplicate_pct']}%"),
    ("Memory", human_bytes(overview["memory_bytes"]), None),
])

tabs = st.tabs(["Preview", "Columns", "Numeric statistics", "Categorical", "Missing values", "Quality issues", "Inspect a column"])

with tabs[0]:
    rows = st.slider("Rows to show", 5, min(PREVIEW_ROWS, max(len(df), 6)), min(20, max(len(df), 6)), key="explore_rows")
    st.dataframe(ui.arrow_safe(df.head(rows)))
    st.caption(f"Showing the first {min(rows, len(df))} of {len(df):,} rows.")

with tabs[1]:
    st.dataframe(profiling.column_profile(df), hide_index=True)

with tabs[2]:
    numeric = profiling.numeric_summary(df)
    if numeric.empty:
        st.info("This dataset has no numeric columns.")
    else:
        st.dataframe(numeric, hide_index=True)
        st.subheader("Potential outliers (IQR rule)")
        st.caption("Values outside Q1 - 1.5 x IQR and Q3 + 1.5 x IQR. They are flagged only; nothing is removed here.")
        st.dataframe(profiling.outlier_summary(df), hide_index=True)

with tabs[3]:
    summary = profiling.categorical_summary(df)
    if summary.empty:
        st.info("This dataset has no categorical columns.")
    else:
        st.dataframe(summary, hide_index=True)
        column = st.selectbox("Show value frequencies for", profiling.categorical_columns(df), key="explore_cat_col")
        st.dataframe(profiling.categorical_frequency(df[column], top=15), hide_index=True)

with tabs[4]:
    st.dataframe(profiling.missing_by_column(df), hide_index=True)
    try:
        result = visualization.missing_values_figure(df)
        st.plotly_chart(result.figure)
    except visualization.VisualizationError as exc:
        st.success(str(exc))

with tabs[5]:
    issues = profiling.quality_issues(df)
    if not issues:
        st.success("The automatic checks found no data-quality problems.")
    else:
        counts = {sev: sum(i.severity == sev for i in issues) for sev in ("high", "medium", "low")}
        st.write(f"**{counts['high']}** high, **{counts['medium']}** medium, **{counts['low']}** low severity flags.")
        st.dataframe(profiling.issues_to_frame(issues), hide_index=True)
        st.caption("These are automatic flags for you to review. They are not confirmed errors. Fix them on the Clean Data page.")

with tabs[6]:
    column = st.selectbox("Column", list(df.columns), key="explore_inspect_col")
    detail = profiling.column_detail(df, column)
    ui.metric_cards([
        ("Type", detail["dtype"], detail["kind"]),
        ("Missing", detail["missing"], f"{detail['missing_pct']}%"),
        ("Unique values", detail["unique"], None),
    ])
    if "stats" in detail:
        st.dataframe(pd.DataFrame([detail["stats"]]), hide_index=True)
        st.caption(f"{detail['outliers_iqr']} potential outlier(s) by the IQR rule.")
        try:
            chart = visualization.histogram(df, column)
            st.plotly_chart(chart.figure)
        except visualization.VisualizationError as exc:
            st.info(str(exc))
    elif "top_values" in detail:
        st.dataframe(detail["top_values"], hide_index=True)
    elif "range" in detail:
        st.write(f"Earliest: {detail['range'][0]}  \nLatest: {detail['range'][1]}")

ui.show_footer()
