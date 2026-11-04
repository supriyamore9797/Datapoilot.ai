"""Chart data preparation and Plotly figure builders.

Each builder returns a :class:`ChartResult` holding the figure, the exact data that was
plotted, notes (sampling, exclusions) and computed summary facts. Invalid combinations raise
:class:`VisualizationError` with an explanation the page can show to the user.

Plotly is imported lazily inside ``_figure_*`` helpers, so data preparation can be tested
without it. Colours: the categorical order, sequential ramp and diverging pair come from a
validated palette (colour-blind separation checked); a single series uses the app accent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import MAX_BAR_CATEGORIES, MAX_PLOT_POINTS, RANDOM_STATE
from src.profiling import (
    categorical_columns,
    column_kind,
    correlation_matrix,
    datetime_columns,
    numeric_columns,
    top_correlations,
)

__all__ = [
    "AGGREGATIONS", "ChartResult", "VisualizationError", "apply_filters", "bar_chart", "box_plot",
    "correlation_heatmap", "correlation_matrix", "detect_time_columns", "histogram", "line_chart",
    "pie_chart", "scatter", "top_correlations", "missing_values_figure", "confusion_matrix_figure",
    "actual_vs_predicted_figure", "residuals_figure", "importance_figure",
]

AGGREGATIONS = ("count", "sum", "mean", "median", "min", "max")
MAX_COLOR_CATEGORIES = 8
MAX_PIE_CATEGORIES = 12
MAX_BOX_GROUPS = 20

ACCENT = "#4F46E5"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = [[0.0, "#eaf2fc"], [1.0, "#1b4f93"]]
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f3f3f1"], [1.0, "#eb6834"]]
INK = "#1F2937"
GRID = "#E5E7EB"


class VisualizationError(ValueError):
    """The requested chart is not possible with the chosen columns / data."""


@dataclass
class ChartResult:
    figure: Any
    data: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------- helpers
def _need(df: pd.DataFrame, column: str | None, label: str, kind: str | None = None) -> str:
    if column is None or column not in df.columns:
        raise VisualizationError(f"Choose a column for {label}.")
    if kind == "numeric" and column_kind(df[column]) != "numeric":
        raise VisualizationError(f"'{column}' is not numeric, so it cannot be used for {label}. "
                                 f"Choose a numeric column ({', '.join(numeric_columns(df)[:5]) or 'none available'}).")
    return column


def _check_group_column(df: pd.DataFrame, column: str | None, label: str, limit: int = MAX_COLOR_CATEGORIES) -> str | None:
    if column in (None, "", "(none)"):
        return None
    _need(df, column, label)
    unique = int(df[column].nunique(dropna=True))
    if unique > limit:
        raise VisualizationError(f"'{column}' has {unique} distinct values; {label} supports at most {limit}. "
                                 "Filter the data or choose a column with fewer categories.")
    return column


def _sample(data: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    if len(data) > MAX_PLOT_POINTS:
        notes.append(f"Showing a random sample of {MAX_PLOT_POINTS:,} of {len(data):,} rows to keep the chart responsive "
                     "(summaries use all rows).")
        return data.sample(MAX_PLOT_POINTS, random_state=RANDOM_STATE).sort_index()
    return data


def _drop_na(data: pd.DataFrame, columns: list[str], notes: list[str]) -> pd.DataFrame:
    cleaned = data.dropna(subset=columns)
    dropped = len(data) - len(cleaned)
    if dropped:
        notes.append(f"{dropped:,} row(s) with missing values in {', '.join(columns)} were left out of this chart.")
    if cleaned.empty:
        raise VisualizationError("No rows are left to plot after removing missing values.")
    return cleaned


def _style(fig: Any, title: str, x_title: str | None = None, y_title: str | None = None) -> Any:
    fig.update_layout(
        template="plotly_white", title=dict(text=title, font=dict(size=16, color=INK)),
        colorway=CATEGORICAL, font=dict(color=INK), margin=dict(l=50, r=20, t=60, b=50),
        legend=dict(title_text=""), height=480, paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title_text=x_title)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_text=y_title)
    return fig


def apply_filters(df: pd.DataFrame, categorical: dict[str, list] | None = None,
                  numeric: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Return a filtered copy. Categorical filters keep listed values; numeric filters keep [lo, hi]."""
    out = df
    for col, allowed in (categorical or {}).items():
        if col not in out.columns:
            raise VisualizationError(f"Filter column '{col}' does not exist.")
        as_text = out[col].astype("object").map(lambda v: None if pd.isna(v) else str(v))
        out = out[as_text.isin([str(a) for a in allowed])]
    for col, (low, high) in (numeric or {}).items():
        if col not in out.columns or column_kind(out[col]) != "numeric":
            raise VisualizationError(f"Numeric filter column '{col}' is not available.")
        out = out[out[col].between(low, high)]
    return out.copy()


def _aggregate(df: pd.DataFrame, keys: list[str], y: str | None, agg: str) -> tuple[pd.DataFrame, str]:
    if agg not in AGGREGATIONS:
        raise VisualizationError(f"Unknown aggregation '{agg}'.")
    if agg == "count" or y is None:
        table = df.groupby(keys, observed=True, dropna=True).size().reset_index(name="count")
        return table, "count"
    name = f"{agg} of {y}"
    table = df.groupby(keys, observed=True, dropna=True)[y].agg(agg).reset_index(name=name)
    return table, name


# -------------------------------------------------------------------- histogram
def histogram(df: pd.DataFrame, x: str, color: str | None = None, bins: int = 30) -> ChartResult:
    import plotly.express as px

    _need(df, x, "the histogram", "numeric")
    color = _check_group_column(df, color, "colour grouping")
    notes: list[str] = []
    cols = [x] + ([color] if color else [])
    data = _drop_na(df[cols], [x], notes)
    values = data[x].astype("float64")
    plot_data = _sample(data, notes)
    fig = px.histogram(plot_data, x=x, color=color, nbins=int(bins), barmode="overlay" if color else "relative",
                       opacity=0.8 if color else 1.0, color_discrete_sequence=CATEGORICAL if color else [ACCENT])
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    _style(fig, f"Distribution of {x}", x, "Count")
    summary = [f"n = {len(values):,}; mean {values.mean():.3g}, median {values.median():.3g}, std {values.std():.3g}.",
               f"Range: {values.min():.3g} to {values.max():.3g}; skewness {values.skew():.2f}."]
    return ChartResult(fig, data, notes, summary)


# --------------------------------------------------------------------- bar chart
def bar_chart(df: pd.DataFrame, x: str, y: str | None = None, agg: str = "count", color: str | None = None,
              top_n: int = MAX_BAR_CATEGORIES) -> ChartResult:
    import plotly.express as px

    _need(df, x, "the X axis")
    if column_kind(df[x]) == "numeric" and df[x].nunique(dropna=True) > 20:
        raise VisualizationError(f"'{x}' is numeric with {df[x].nunique()} distinct values, which would make one bar each. "
                                 "Use a histogram, or choose a categorical column for the X axis.")
    if column_kind(df[x]) == "datetime":
        raise VisualizationError("Datetime columns are better shown with a line chart.")
    if agg != "count":
        _need(df, y, f"the '{agg}' aggregation", "numeric")
    else:
        y = None
    color = _check_group_column(df, color, "colour grouping")
    if color == x:
        color = None
    notes: list[str] = []
    cols = list(dict.fromkeys(c for c in (x, y, color) if c))
    data = df[cols].copy()
    data[x] = data[x].astype(str).where(data[x].notna())
    data = _drop_na(data, [x] + ([y] if y else []), notes)
    table, value_name = _aggregate(data, [x] + ([color] if color else []), y, agg)
    totals = table.groupby(x)[value_name].sum().sort_values(ascending=False)
    if len(totals) > top_n:
        notes.append(f"'{x}' has {len(totals)} categories; showing the top {top_n} by {value_name}.")
    keep = list(totals.head(top_n).index)
    table = table[table[x].isin(keep)].copy()
    table[x] = pd.Categorical(table[x], categories=keep, ordered=True)
    table = table.sort_values([x] + ([color] if color else []), ignore_index=True)
    fig = px.bar(table, x=x, y=value_name, color=color, barmode="group",
                 color_discrete_sequence=CATEGORICAL if color else [ACCENT])
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    _style(fig, f"{value_name.capitalize()} by {x}", x, value_name)
    ranked = table.groupby(x, observed=True)[value_name].sum().sort_values(ascending=False)
    summary = [f"Highest: '{ranked.index[0]}' ({ranked.iloc[0]:,.4g}); lowest: '{ranked.index[-1]}' ({ranked.iloc[-1]:,.4g})."]
    if agg == "count":
        share = 100 * ranked.iloc[0] / max(ranked.sum(), 1)
        summary.append(f"The most frequent category is '{ranked.index[0]}' with {share:.1f}% of the plotted rows.")
    return ChartResult(fig, table, notes, summary)


# -------------------------------------------------------------------- scatter plot
def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, trendline: bool = False) -> ChartResult:
    import plotly.express as px
    import plotly.graph_objects as go

    _need(df, x, "the X axis", "numeric")
    _need(df, y, "the Y axis", "numeric")
    if color in (None, "", "(none)"):
        color = None
    else:
        _need(df, color, "colour")
        if column_kind(df[color]) != "numeric":
            _check_group_column(df, color, "colour grouping")
    notes: list[str] = []
    cols = list(dict.fromkeys([x, y] + ([color] if color else [])))
    data = _drop_na(df[cols], [x, y], notes)
    plot_data = _sample(data, notes)
    numeric_color = bool(color) and column_kind(df[color]) == "numeric"
    fig = px.scatter(plot_data, x=x, y=y, color=color, opacity=0.7,
                     color_continuous_scale=SEQUENTIAL if numeric_color else None,
                     color_discrete_sequence=CATEGORICAL if color and not numeric_color else [ACCENT])
    fig.update_traces(marker=dict(size=7, line=dict(width=1, color="white")))
    summary = []
    if len(data) >= 3 and data[x].nunique() > 1 and data[y].nunique() > 1:
        r = float(data[x].corr(data[y]))
        summary.append(f"Pearson r = {r:.3f} (n = {len(data):,}). Correlation does not imply causation.")
        if trendline:
            slope, intercept = np.polyfit(data[x].astype("float64"), data[y].astype("float64"), 1)
            xs = np.array([data[x].min(), data[x].max()], dtype="float64")
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear fit",
                                     line=dict(color=INK, width=2, dash="dash")))
            summary.append(f"Linear fit: y = {slope:.4g} x + {intercept:.4g}.")
    else:
        summary.append("Not enough variation to compute a correlation.")
    _style(fig, f"{y} vs {x}", x, y)
    return ChartResult(fig, data, notes, summary)


# ----------------------------------------------------------------------- box plot
def box_plot(df: pd.DataFrame, y: str, x: str | None = None) -> ChartResult:
    import plotly.express as px

    _need(df, y, "the box plot values", "numeric")
    if x in ("", "(none)"):
        x = None
    notes: list[str] = []
    if x:
        _need(df, x, "grouping")
        unique = int(df[x].nunique(dropna=True))
        if unique > MAX_BOX_GROUPS:
            raise VisualizationError(f"'{x}' has {unique} groups; box plots support at most {MAX_BOX_GROUPS}. Filter the data first.")
    cols = [y] + ([x] if x else [])
    data = df[cols].copy()
    if x:
        data[x] = data[x].astype(str).where(data[x].notna())
    data = _drop_na(data, cols, notes)
    plot_data = _sample(data, notes)
    fig = px.box(plot_data, x=x, y=y, color=x, color_discrete_sequence=CATEGORICAL if x else [ACCENT], points="outliers")
    fig.update_layout(showlegend=False)
    _style(fig, f"{y}" + (f" by {x}" if x else ""), x, y)
    groups = data.groupby(x)[y] if x else [("all rows", data[y])]
    summary = []
    for name, values in groups:
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        fence_low, fence_high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outliers = int(((values < fence_low) | (values > fence_high)).sum())
        summary.append(f"{name}: median {values.median():.3g}, IQR {q1:.3g} to {q3:.3g}, {outliers} outlier(s).")
    return ChartResult(fig, data, notes, summary[:10])


# --------------------------------------------------------------------- correlation
def correlation_heatmap(df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson") -> ChartResult:
    import plotly.graph_objects as go

    candidates = columns if columns else numeric_columns(df)
    matrix = correlation_matrix(df, candidates, method)
    if matrix.empty:
        raise VisualizationError("A correlation heatmap needs at least two numeric columns that are not constant.")
    notes = []
    excluded = [c for c in candidates if c not in matrix.columns]
    if excluded:
        notes.append(f"Left out (constant, non-numeric or all missing): {', '.join(excluded)}.")
    show_text = len(matrix) <= 14
    fig = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index), zmin=-1, zmax=1,
                               colorscale=DIVERGING, colorbar=dict(title=method.capitalize()),
                               text=np.round(matrix.to_numpy(), 2) if show_text else None,
                               texttemplate="%{text}" if show_text else None,
                               hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>"))
    _style(fig, f"{method.capitalize()} correlation matrix")
    fig.update_yaxes(autorange="reversed")
    pairs = top_correlations(df[list(matrix.columns)], n=3, method=method)
    summary = [f"{r['Column A']} and {r['Column B']}: {r['Correlation']:+.2f}" for _, r in pairs.iterrows()]
    summary.append("Correlation measures association only; it does not show that one variable causes another.")
    return ChartResult(fig, matrix, notes, summary)


# ------------------------------------------------------------------------ line chart
def detect_time_columns(df: pd.DataFrame) -> list[str]:
    """Columns suitable for the X axis of a line chart: datetimes, date-like text, then numerics."""
    found = list(datetime_columns(df))
    for col in categorical_columns(df):
        if col in found or df[col].dropna().empty:
            continue
        sample = df[col].dropna().astype(str).head(200)
        parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
        if parsed.notna().mean() >= 0.9 and not sample.str.fullmatch(r"\d{1,4}").all():
            found.append(col)
    return found + [c for c in numeric_columns(df) if c not in found]


_FREQ = {"None": None, "Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}


def line_chart(df: pd.DataFrame, x: str, y: str, agg: str = "mean", color: str | None = None,
               freq: str = "None") -> ChartResult:
    import plotly.express as px

    _need(df, x, "the X axis")
    _need(df, y, "the Y axis", "numeric")
    if agg == "count":
        raise VisualizationError("Choose sum, mean, median, min or max for a line chart of a numeric column.")
    color = _check_group_column(df, color, "colour grouping")
    notes: list[str] = []
    cols = list(dict.fromkeys([x, y] + ([color] if color else [])))
    data = df[cols].copy()
    kind = column_kind(data[x])
    if kind == "categorical":
        parsed = pd.to_datetime(data[x].astype("object").where(data[x].notna(), None), format="mixed", errors="coerce")
        if data[x].notna().any() and parsed[data[x].notna()].notna().mean() < 0.9:
            raise VisualizationError(f"'{x}' is not a date or an ordered numeric column. Choose a date column or a number.")
        data[x] = parsed
        kind = "datetime"
    data = _drop_na(data, [x, y], notes)
    if kind == "datetime" and freq != "None":
        if freq not in _FREQ:
            raise VisualizationError(f"Unknown time grouping '{freq}'.")
        data[x] = data[x].dt.to_period(_FREQ[freq]).dt.to_timestamp()
    table, value_name = _aggregate(data, [x] + ([color] if color else []), y, agg)
    table = table.sort_values([x] + ([color] if color else []), ignore_index=True)
    if len(table) > MAX_PLOT_POINTS:
        raise VisualizationError(f"The chart would have {len(table):,} points. Group by a coarser time period.")
    fig = px.line(table, x=x, y=value_name, color=color, markers=len(table) <= 60,
                  color_discrete_sequence=CATEGORICAL if color else [ACCENT])
    fig.update_traces(line=dict(width=2))
    _style(fig, f"{value_name.capitalize()} over {x}", x, value_name)
    best, worst = table.loc[table[value_name].idxmax()], table.loc[table[value_name].idxmin()]
    summary = [f"Peak: {best[value_name]:,.4g} at {best[x]}; lowest: {worst[value_name]:,.4g} at {worst[x]}."]
    return ChartResult(fig, table, notes, summary)


# ------------------------------------------------------------------------ pie chart
def pie_chart(df: pd.DataFrame, names: str, values: str | None = None, agg: str = "count") -> ChartResult:
    import plotly.express as px

    _need(df, names, "the pie slices")
    unique = int(df[names].nunique(dropna=True))
    if unique > MAX_PIE_CATEGORIES:
        raise VisualizationError(f"'{names}' has {unique} categories; pie charts are only readable with {MAX_PIE_CATEGORIES} or fewer. "
                                 "Use a bar chart instead.")
    if agg != "count":
        _need(df, values, f"the '{agg}' aggregation", "numeric")
    else:
        values = None
    notes: list[str] = []
    data = df[[c for c in dict.fromkeys([names, values]) if c]].copy()
    data[names] = data[names].astype(str).where(data[names].notna())
    data = _drop_na(data, [names] + ([values] if values else []), notes)
    table, value_name = _aggregate(data, [names], values, agg)
    if (table[value_name] < 0).any():
        raise VisualizationError("A pie chart cannot show negative values. Use a bar chart instead.")
    table = table.sort_values(value_name, ascending=False, ignore_index=True)
    fig = px.pie(table, names=names, values=value_name, color_discrete_sequence=CATEGORICAL, hole=0.35)
    fig.update_traces(marker=dict(line=dict(color="white", width=2)), textinfo="percent+label")
    _style(fig, f"{value_name.capitalize()} by {names}")
    share = 100 * table[value_name].iloc[0] / table[value_name].sum() if table[value_name].sum() else 0.0
    return ChartResult(fig, table, notes, [f"Largest slice: '{table[names].iloc[0]}' ({share:.1f}%)."])


# ------------------------------------------------- figures for the Explore / ML pages
def missing_values_figure(df: pd.DataFrame) -> ChartResult:
    """Bar chart of the percentage of missing values per column that has any."""
    import plotly.express as px

    from src.profiling import missing_by_column

    table = missing_by_column(df)
    table = table[table["Missing"] > 0]
    if table.empty:
        raise VisualizationError("There are no missing values to display.")
    fig = px.bar(table, x="Missing %", y="Column", orientation="h", color_discrete_sequence=[ACCENT],
                 hover_data={"Missing": True})
    fig.update_yaxes(autorange="reversed")
    _style(fig, "Missing values by column", "Missing (% of rows)", None)
    fig.update_layout(height=max(300, 60 + 32 * len(table)))
    return ChartResult(fig, table, [], [f"{len(table)} of {df.shape[1]} columns contain missing values."])


def confusion_matrix_figure(confusion: pd.DataFrame) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure(go.Heatmap(z=confusion.to_numpy(), x=list(confusion.columns), y=list(confusion.index),
                               colorscale=SEQUENTIAL, text=confusion.to_numpy(), texttemplate="%{text}",
                               hovertemplate="%{y}, %{x}: %{z}<extra></extra>", showscale=False))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Confusion matrix (held-out test set)")


def actual_vs_predicted_figure(predictions: pd.DataFrame) -> Any:
    import plotly.express as px
    import plotly.graph_objects as go

    fig = px.scatter(predictions, x="Actual", y="Predicted", opacity=0.7, color_discrete_sequence=[ACCENT])
    low = float(min(predictions["Actual"].min(), predictions["Predicted"].min()))
    high = float(max(predictions["Actual"].max(), predictions["Predicted"].max()))
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect prediction",
                             line=dict(color=INK, dash="dash", width=2)))
    return _style(fig, "Actual vs predicted (held-out test set)", "Actual", "Predicted")


def residuals_figure(predictions: pd.DataFrame) -> Any:
    import plotly.express as px

    fig = px.histogram(predictions, x="Residual", nbins=30, color_discrete_sequence=[ACCENT])
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    return _style(fig, "Residuals (actual minus predicted)", "Residual", "Count")


def importance_figure(importance: pd.DataFrame, top: int = 15) -> Any:
    import plotly.express as px

    shown = importance.head(top)
    fig = px.bar(shown, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[ACCENT])
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Feature importance (impurity-based, training data)", "Importance", None)
