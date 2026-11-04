"""Visual Analytics page: interactive Plotly charts driven by the current working data."""

import streamlit as st

from src import profiling, ui, visualization
from src.visualization import AGGREGATIONS, VisualizationError

ui.setup_page("Visual Analytics", "Build interactive charts from your data. Charts always use the current working data and the columns you pick.")
df = ui.require_data()

numeric = profiling.numeric_columns(df)
categorical = profiling.categorical_columns(df)
NONE = "(none)"

# ------------------------------------------------------------------------- filters
with st.expander("Optional filters (applied to every chart below)"):
    cat_filter_cols = st.multiselect("Filter by categorical column(s)", [c for c in categorical if df[c].nunique() <= 100], key="va_cat_filters")
    cat_filters = {}
    for col in cat_filter_cols:
        values = sorted(df[col].dropna().astype(str).unique())
        cat_filters[col] = st.multiselect(f"Keep values of '{col}'", values, default=values, key=f"va_cat_{col}")
    num_filter_cols = st.multiselect("Filter by numeric range", numeric, key="va_num_filters")
    num_filters = {}
    for col in num_filter_cols:
        low, high = float(df[col].min()), float(df[col].max())
        if low < high:
            num_filters[col] = st.slider(f"Range of '{col}'", low, high, (low, high), key=f"va_num_{col}")

try:
    work = visualization.apply_filters(df, cat_filters, num_filters)
except VisualizationError as exc:
    st.error(str(exc))
    st.stop()
if len(work) != len(df):
    st.caption(f"Filters keep {len(work):,} of {len(df):,} rows.")
if work.empty:
    st.warning("The filters remove every row. Relax them to draw a chart.")
    st.stop()

# ----------------------------------------------------------------------- controls
CHARTS = ["Histogram", "Bar chart", "Scatter plot", "Box plot", "Correlation heatmap", "Line chart", "Pie chart"]
chart_type = st.selectbox("Chart type", CHARTS, key="va_chart")
time_columns = visualization.detect_time_columns(work)
if chart_type == "Line chart" and not time_columns:
    st.info("A line chart needs a date column or an ordered numeric column, and this dataset has neither.")

result = None
try:
    if chart_type == "Histogram":
        if not numeric:
            raise VisualizationError("This dataset has no numeric columns for a histogram.")
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("Column (X)", numeric, key="va_h_x")
        color = c2.selectbox("Colour / group", [NONE] + categorical, key="va_h_c")
        bins = c3.slider("Bins", 5, 100, 30, key="va_h_bins")
        result = visualization.histogram(work, x, None if color == NONE else color, bins)

    elif chart_type == "Bar chart":
        c1, c2, c3, c4 = st.columns(4)
        x = c1.selectbox("Category (X)", list(work.columns), key="va_b_x")
        agg = c2.selectbox("Aggregation", AGGREGATIONS, key="va_b_agg")
        y = c3.selectbox("Value (Y)", numeric or [NONE], key="va_b_y", disabled=agg == "count")
        color = c4.selectbox("Colour / group", [NONE] + categorical, key="va_b_c")
        result = visualization.bar_chart(work, x, None if agg == "count" else y, agg, None if color == NONE else color)

    elif chart_type == "Scatter plot":
        if len(numeric) < 2:
            raise VisualizationError("A scatter plot needs at least two numeric columns.")
        c1, c2, c3, c4 = st.columns(4)
        x = c1.selectbox("X axis", numeric, key="va_s_x")
        y = c2.selectbox("Y axis", numeric, index=1, key="va_s_y")
        color = c3.selectbox("Colour", [NONE] + list(work.columns), key="va_s_c")
        trend = c4.checkbox("Show linear fit", key="va_s_t")
        result = visualization.scatter(work, x, y, None if color == NONE else color, trend)

    elif chart_type == "Box plot":
        if not numeric:
            raise VisualizationError("This dataset has no numeric columns for a box plot.")
        c1, c2 = st.columns(2)
        y = c1.selectbox("Values (Y)", numeric, key="va_x_y")
        group = c2.selectbox("Group by", [NONE] + categorical, key="va_x_g")
        result = visualization.box_plot(work, y, None if group == NONE else group)

    elif chart_type == "Correlation heatmap":
        c1, c2 = st.columns([3, 1])
        cols = c1.multiselect("Columns (empty = all numeric)", numeric, key="va_c_cols")
        method = c2.selectbox("Method", ["pearson", "spearman"], key="va_c_m")
        result = visualization.correlation_heatmap(work, cols or None, method)

    elif chart_type == "Line chart":
        if not time_columns or not numeric:
            raise VisualizationError("A line chart needs an ordered X column (date or number) and a numeric Y column.")
        c1, c2, c3, c4, c5 = st.columns(5)
        x = c1.selectbox("X (time / order)", time_columns, key="va_l_x")
        y = c2.selectbox("Y (numeric)", numeric, key="va_l_y")
        agg = c3.selectbox("Aggregation", [a for a in AGGREGATIONS if a != "count"], key="va_l_agg")
        freq = c4.selectbox("Group dates by", ["None", "Day", "Week", "Month", "Quarter", "Year"], index=3, key="va_l_f")
        color = c5.selectbox("Colour / group", [NONE] + categorical, key="va_l_c")
        result = visualization.line_chart(work, x, y, agg, None if color == NONE else color, freq)

    elif chart_type == "Pie chart":
        small = [c for c in categorical if work[c].nunique() <= visualization.MAX_PIE_CATEGORIES]
        if not small:
            raise VisualizationError("A pie chart needs a categorical column with 12 or fewer categories; this dataset has none. Use a bar chart.")
        c1, c2, c3 = st.columns(3)
        names = c1.selectbox("Categories", small, key="va_p_n")
        agg = c2.selectbox("Aggregation", AGGREGATIONS, key="va_p_agg")
        values = c3.selectbox("Value", numeric or [NONE], key="va_p_v", disabled=agg == "count")
        result = visualization.pie_chart(work, names, None if agg == "count" else values, agg)
except VisualizationError as exc:
    st.warning(str(exc))

if result is not None:
    st.plotly_chart(result.figure)
    for note in result.notes:
        st.caption(note)
    st.subheader("Computed summary")
    for line in result.summary:
        st.markdown(f"- {line}")
    if chart_type == "Correlation heatmap":
        st.dataframe(profiling.top_correlations(work, n=8), hide_index=True)
    with st.expander("Data behind this chart"):
        st.dataframe(ui.arrow_safe(result.data.head(200)))

ui.show_footer()
