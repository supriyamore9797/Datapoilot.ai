import numpy as np
import pandas as pd
import pytest

from src import profiling as p


def test_overview_counts(small_df):
    o = p.dataset_overview(small_df)
    assert (o["rows"], o["columns"]) == (6, 3)
    assert o["numeric_columns"] == 2  # age and score
    assert o["categorical_columns"] == 1
    assert o["missing_cells"] == 2
    assert o["missing_pct"] == pytest.approx(100 * 2 / 18, abs=0.01)


def test_overview_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2, 2, 2], "b": ["x", "x", "y", "y", "z"]})
    o = p.dataset_overview(df)
    assert o["duplicate_rows"] == 2
    assert o["duplicate_pct"] == 40.0


def test_kinds_detected_for_each_type():
    df = pd.DataFrame({
        "n": [1.5, 2.5], "i": [1, 2], "b": [True, False], "s": ["a", "b"],
        "d": pd.to_datetime(["2024-01-01", "2024-01-02"]), "c": pd.Categorical(["u", "v"]),
    })
    kinds = {c: p.column_kind(df[c]) for c in df}
    assert kinds == {"n": "numeric", "i": "numeric", "b": "boolean", "s": "categorical", "d": "datetime", "c": "categorical"}
    assert p.numeric_columns(df) == ["n", "i"]
    assert p.categorical_columns(df) == ["b", "s", "c"]
    assert p.datetime_columns(df) == ["d"]


def test_empty_dataset_does_not_crash():
    df = pd.DataFrame()
    o = p.dataset_overview(df)
    assert o["rows"] == 0 and o["columns"] == 0 and o["missing_pct"] == 0.0 and o["memory_bytes"] == 0
    assert p.column_profile(df).empty
    assert p.numeric_summary(df).empty
    assert p.quality_issues(df)[0].category == "empty"
    assert p.generate_insights(df)[0].startswith("The dataset is empty")


def test_zero_row_dataframe_with_columns():
    df = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=object)})
    assert p.dataset_overview(df)["rows"] == 0
    assert p.quality_issues(df)[0].category == "empty"


def test_column_profile_missing_and_unique(small_df):
    prof = p.column_profile(small_df).set_index("Column")
    assert prof.loc["age", "Missing"] == 1
    assert prof.loc["age", "Missing %"] == pytest.approx(16.67, abs=0.01)
    assert prof.loc["city", "Unique"] == 2
    assert prof.loc["score", "Unique"] == 5


def test_missing_by_column_sorted_descending():
    df = pd.DataFrame({"a": [1, None, None], "b": [1, 2, None], "c": [1, 2, 3]})
    out = p.missing_by_column(df)
    assert out["Column"].tolist() == ["a", "b", "c"]
    assert out["Missing"].tolist() == [2, 1, 0]


def test_numeric_summary_values():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan], "t": list("abcde")})
    s = p.numeric_summary(df).set_index("Column")
    assert list(s.index) == ["x"]
    assert s.loc["x", "mean"] == 2.5 and s.loc["x", "min"] == 1.0 and s.loc["x", "max"] == 4.0
    assert s.loc["x", "missing"] == 1


def test_categorical_frequency_includes_missing_label():
    series = pd.Series(["a", "a", "b", None])
    out = p.categorical_frequency(series)
    assert out.loc[out["Value"] == "a", "Count"].iloc[0] == 2
    assert "(missing)" in out["Value"].tolist()
    assert out["Percent"].iloc[0] == 50.0


def test_iqr_outliers_flag_extreme_value_only():
    s = pd.Series([10, 11, 12, 11, 10, 12, 11, 500])
    mask = p.iqr_outlier_mask(s)
    assert mask.tolist() == [False] * 7 + [True]
    summary = p.outlier_summary(pd.DataFrame({"v": s}))
    assert summary.loc[0, "Outliers"] == 1


def test_iqr_needs_at_least_four_values():
    assert p.iqr_bounds(pd.Series([1, 2, 3])) is None
    assert not p.iqr_outlier_mask(pd.Series([1, 2, 3])).any()


def test_text_type_problems_detected():
    assert p.classify_text_column(pd.Series(["1", "2", "3"])) == "numeric_text"
    assert p.classify_text_column(pd.Series([1, "a", 2.5], dtype=object)) == "mixed"
    assert p.classify_text_column(pd.Series(["1"] * 19 + ["x"])) == "mostly_numeric_text"
    assert p.classify_text_column(pd.Series(["a", "b"])) is None
    assert p.classify_text_column(pd.Series([1.0, 2.0])) is None


def test_blank_strings_counted():
    assert p.count_blank_strings(pd.Series(["a", " ", "", None])) == 2


def test_quality_issues_cover_main_problems():
    df = pd.DataFrame({
        "all_missing": [np.nan] * 25,
        "constant": [1] * 25,
        "mostly_missing": [1.0] * 5 + [np.nan] * 20,
        "id": list(range(25)),
        "num": [1.0] * 24 + [1000.0],
    })
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    issues = p.quality_issues(df)
    by = {(i.column, i.category) for i in issues}
    assert ("all_missing", "missing") in by
    assert ("constant", "constant") in by
    assert ("mostly_missing", "missing") in by
    assert ("(dataset)", "duplicates") in by
    assert ("num", "outliers") in by
    assert issues[0].severity == "high"  # sorted most severe first


def test_clean_dataset_has_no_issues():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=40), "b": rng.choice(["x", "y"], 40)})
    assert [i for i in p.quality_issues(df) if i.severity != "low"] == []


def test_column_detail_numeric_and_categorical(small_df):
    num = p.column_detail(small_df, "score")
    assert num["stats"]["mean"] == pytest.approx(17 / 6)
    cat = p.column_detail(small_df, "city")
    assert cat["top_values"].iloc[0]["Value"] == "B"
    with pytest.raises(KeyError):
        p.column_detail(small_df, "nope")


def test_correlation_matrix_perfect_and_constant_dropped():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [1, 1, 1, 1], "t": list("abcd")})
    m = p.correlation_matrix(df)
    assert list(m.columns) == ["a", "b"]
    assert m.loc["a", "b"] == pytest.approx(1.0)


def test_correlation_needs_two_usable_columns():
    assert p.correlation_matrix(pd.DataFrame({"a": [1, 2, 3]})).empty
    with pytest.raises(ValueError):
        p.correlation_matrix(pd.DataFrame({"a": [1, 2], "b": [1, 2]}), method="bogus")


def test_top_correlations_orders_by_absolute_value():
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    df = pd.DataFrame({"a": a, "neg": -a + rng.normal(0, 0.1, 200), "noise": rng.normal(size=200)})
    top = p.top_correlations(df, n=2)
    assert {top.iloc[0]["Column A"], top.iloc[0]["Column B"]} == {"a", "neg"}
    assert top.iloc[0]["Correlation"] < -0.9


def test_insights_are_computed_from_data(small_df):
    text = " ".join(p.generate_insights(small_df))
    assert "6 rows and 3 columns" in text
    assert "'age'" in text  # age and city tie for most missing; one of them is named
