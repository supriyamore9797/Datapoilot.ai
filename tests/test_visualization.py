"""Chart data-preparation and validation tests.

Without Plotly installed, tests/conftest.py supplies a stand-in so the data logic is still tested;
tests marked ``needs_plotly`` render real figures and are skipped in that case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import visualization as viz
from src.config import MAX_PLOT_POINTS
from src.visualization import VisualizationError


@pytest.fixture
def sales() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2024-01-05", "2024-01-20", "2024-02-03", "2024-02-18", "2024-03-02", "2024-03-30"],
        "product": ["A", "B", "A", "B", "A", None],
        "region": ["N", "N", "S", "S", "N", "S"],
        "units": [10, 20, 30, 40, 50, 60],
        "price": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
    })


# ------------------------------------------------------------------- bar chart data
def test_bar_chart_count_aggregation_and_summary(sales):
    result = viz.bar_chart(sales, "product")
    counts = dict(zip(result.data["product"].astype(str), result.data["count"]))
    assert counts == {"A": 3, "B": 2}  # the missing product is left out, and reported
    assert any("left out" in n for n in result.notes)
    assert "'A'" in result.summary[0] and "60.0%" in result.summary[1]


def test_bar_chart_sum_and_mean_values_are_correct(sales):
    summed = viz.bar_chart(sales, "region", "units", "sum").data.set_index("region")["sum of units"]
    assert summed["N"] == 10 + 20 + 50 and summed["S"] == 40 + 60 + 30
    mean = viz.bar_chart(sales, "region", "units", "mean").data.set_index("region")["mean of units"]
    assert mean["N"] == pytest.approx(80 / 3)


def test_bar_chart_orders_categories_by_value_and_caps_categories():
    df = pd.DataFrame({"c": [f"k{i % 40}" for i in range(400)], "v": range(400)})
    result = viz.bar_chart(df, "c", "v", "sum", top_n=10)
    assert result.data["c"].nunique() == 10
    assert any("showing the top 10" in n for n in result.notes)
    totals = result.data.set_index("c")["sum of v"]
    assert totals.is_monotonic_decreasing


def test_bar_chart_rejects_unsuitable_columns(sales):
    with pytest.raises(VisualizationError, match="not numeric"):
        viz.bar_chart(sales, "region", "product", "sum")
    with pytest.raises(VisualizationError, match="Choose a column"):
        viz.bar_chart(sales, "missing_col")
    many = pd.DataFrame({"n": np.arange(100.0)})
    with pytest.raises(VisualizationError, match="histogram"):
        viz.bar_chart(many, "n")
    with pytest.raises(VisualizationError, match="Unknown aggregation"):
        viz.bar_chart(sales, "region", "units", "mode")


def test_color_grouping_rejects_high_cardinality():
    df = pd.DataFrame({"x": range(100), "g": [f"g{i}" for i in range(100)], "y": range(100)})
    with pytest.raises(VisualizationError, match="distinct values"):
        viz.scatter(df, "x", "y", color="g")


# ----------------------------------------------------------------------- histogram
def test_histogram_requires_numeric_and_reports_statistics(sales):
    with pytest.raises(VisualizationError, match="not numeric"):
        viz.histogram(sales, "region")
    result = viz.histogram(sales, "units")
    assert "mean 35" in result.summary[0] and "median 35" in result.summary[0]
    assert len(result.data) == 6


def test_histogram_drops_nulls_and_says_so(sales):
    result = viz.histogram(sales, "price")
    assert len(result.data) == 5
    assert any("1 row(s) with missing values" in n for n in result.notes)


def test_all_null_column_gives_a_helpful_error():
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    with pytest.raises(VisualizationError, match="No rows are left"):
        viz.histogram(df, "x")


def test_large_data_is_sampled_with_an_explanation_but_summary_uses_all_rows():
    big = pd.DataFrame({"x": np.arange(MAX_PLOT_POINTS + 5000, dtype=float)})
    result = viz.histogram(big, "x")
    assert any("random sample" in n for n in result.notes)
    assert f"n = {len(big):,}" in result.summary[0]


# ---------------------------------------------------------------- scatter and box
def test_scatter_reports_pearson_r_without_claiming_causation():
    x = np.arange(50.0)
    df = pd.DataFrame({"x": x, "y": 2 * x + 1, "z": np.sin(x)})
    result = viz.scatter(df, "x", "y", trendline=True)
    assert "r = 1.000" in result.summary[0] and "does not imply causation" in result.summary[0]
    assert "y = 2 x + 1" in result.summary[1]
    with pytest.raises(VisualizationError):
        viz.scatter(df.assign(t=list("ab") * 25), "x", "t")


def test_scatter_constant_column_has_no_correlation():
    df = pd.DataFrame({"x": [1.0] * 5, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert "Not enough variation" in viz.scatter(df, "x", "y").summary[0]


def test_box_plot_summary_counts_outliers_per_group():
    df = pd.DataFrame({"g": ["a"] * 8 + ["b"] * 8, "v": [10, 11, 12, 11, 10, 12, 11, 100] + [5, 6, 5, 6, 5, 6, 5, 6]})
    summary = viz.box_plot(df, "v", "g").summary
    assert any(s.startswith("a:") and "1 outlier" in s for s in summary)
    assert any(s.startswith("b:") and "0 outlier" in s for s in summary)
    with pytest.raises(VisualizationError, match="not numeric"):
        viz.box_plot(df, "g")
    many = pd.DataFrame({"g": [f"g{i}" for i in range(30)] * 2, "v": range(60)})
    with pytest.raises(VisualizationError, match="at most"):
        viz.box_plot(many, "v", "g")


# ---------------------------------------------------------------------- correlation
def test_correlation_heatmap_matrix_is_correct_and_drops_unusable_columns():
    x = np.arange(20.0)
    df = pd.DataFrame({"a": x, "b": -x, "c": np.ones(20), "d": ["t"] * 20})
    result = viz.correlation_heatmap(df)
    assert list(result.data.columns) == ["a", "b"]  # constant and text columns are not correlated
    assert result.data.loc["a", "b"] == pytest.approx(-1.0)
    assert "does not show that one variable causes another" in result.summary[-1]
    assert "a and b: -1.00" in result.summary[0]


def test_correlation_heatmap_spearman_differs_from_pearson_on_monotonic_data():
    x = np.arange(1.0, 21.0)
    df = pd.DataFrame({"a": x, "b": np.exp(x / 3)})
    pearson = viz.correlation_heatmap(df, method="pearson").data.loc["a", "b"]
    spearman = viz.correlation_heatmap(df, method="spearman").data.loc["a", "b"]
    assert spearman == pytest.approx(1.0) and pearson < 0.99


def test_correlation_heatmap_needs_two_numeric_columns():
    with pytest.raises(VisualizationError, match="at least two numeric"):
        viz.correlation_heatmap(pd.DataFrame({"a": [1, 2, 3], "t": list("abc")}))


# ------------------------------------------------------------- line chart / time
def test_detect_time_columns_finds_date_strings_and_ignores_plain_text(sales):
    found = viz.detect_time_columns(sales)
    assert found[0] == "date"
    assert "region" not in found and "product" not in found
    assert "units" in found  # numeric columns can be an ordered axis


def test_line_chart_groups_by_month_and_aggregates_correctly(sales):
    result = viz.line_chart(sales, "date", "units", "sum", freq="Month")
    table = result.data.set_index("date")["sum of units"]
    assert table.tolist() == [30, 70, 110]  # Jan, Feb, Mar
    assert pd.api.types.is_datetime64_any_dtype(result.data["date"])


def test_line_chart_rejects_non_date_text_and_count_aggregation(sales):
    with pytest.raises(VisualizationError, match="not a date"):
        viz.line_chart(sales, "region", "units")
    with pytest.raises(VisualizationError, match="Choose sum"):
        viz.line_chart(sales, "date", "units", "count")
    with pytest.raises(VisualizationError, match="not numeric"):
        viz.line_chart(sales, "date", "region")


# ------------------------------------------------------------------------- pie
def test_pie_chart_limits_and_values(sales):
    result = viz.pie_chart(sales, "region", "units", "sum")
    assert result.data.set_index("region")["sum of units"].to_dict() == {"N": 80, "S": 130}
    assert "'S'" in result.summary[0]
    many = pd.DataFrame({"c": [f"k{i}" for i in range(20)]})
    with pytest.raises(VisualizationError, match="bar chart"):
        viz.pie_chart(many, "c")
    with pytest.raises(VisualizationError, match="negative"):
        viz.pie_chart(pd.DataFrame({"c": ["a", "b"], "v": [-5, 3]}), "c", "v", "sum")


# ------------------------------------------------------------------------ filters
def test_apply_filters_categorical_and_numeric_and_does_not_mutate(sales):
    before = sales.copy()
    out = viz.apply_filters(sales, {"region": ["N"]}, {"units": (15, 55)})
    assert out["units"].tolist() == [20, 50]
    pd.testing.assert_frame_equal(sales, before)
    assert len(viz.apply_filters(sales, {"product": ["A", "B"]})) == 5  # missing product excluded by the filter
    with pytest.raises(VisualizationError):
        viz.apply_filters(sales, {"ghost": ["x"]})
    with pytest.raises(VisualizationError):
        viz.apply_filters(sales, numeric={"region": (0, 1)})


def test_filtered_chart_uses_filtered_rows_not_stale_data(sales):
    only_north = viz.apply_filters(sales, {"region": ["N"]})
    result = viz.bar_chart(only_north, "region")
    assert result.data["region"].astype(str).tolist() == ["N"] and result.data["count"].tolist() == [3]


def test_missing_values_figure_reports_columns_with_gaps(sales):
    result = viz.missing_values_figure(sales)
    assert set(result.data["Column"]) == {"product", "price"}
    with pytest.raises(VisualizationError, match="no missing values"):
        viz.missing_values_figure(sales.dropna())


# ---------------------------------------------------- real Plotly rendering (optional)
@pytest.mark.needs_plotly
def test_real_plotly_figures_render_with_the_requested_columns(sales):
    fig = viz.scatter(sales, "units", "price").figure
    assert fig.data[0].type == "scatter" and fig.layout.xaxis.title.text == "units"
    bar = viz.bar_chart(sales, "region", "units", "sum").figure
    assert list(bar.data[0].y) == [130, 80] or list(bar.data[0].y) == [80, 130]
    heat = viz.correlation_heatmap(sales[["units", "price"]].dropna().assign(z=lambda d: d.units * 2 + d.price)).figure
    assert heat.data[0].type == "heatmap"
    assert viz.confusion_matrix_figure(pd.DataFrame([[3, 1], [0, 4]], index=["Actual: a", "Actual: b"],
                                                    columns=["Predicted: a", "Predicted: b"])) is not None
