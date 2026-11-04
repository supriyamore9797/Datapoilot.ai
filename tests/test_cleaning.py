"""Behavioural tests for src.cleaning (every operation works on a copy and never mutates its input)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src import cleaning
from src.cleaning import CleaningError


def _snapshot(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


def test_fill_mean_uses_column_mean_and_does_not_mutate_input(small_df):
    before = _snapshot(small_df)
    result = cleaning.fill_missing_numeric(small_df, ["age"], "mean")
    expected = small_df["age"].mean()  # (20 + 25 + 40 + 35 + 25) / 5 = 29
    assert expected == pytest.approx(29.0)
    assert result.df["age"].isna().sum() == 0
    assert result.df.loc[2, "age"] == pytest.approx(29.0)
    pd.testing.assert_frame_equal(small_df, before)  # input untouched
    assert result.entry["cells_changed"] == 1
    assert result.entry["columns"] == ["age"]


def test_fill_median_uses_column_median(small_df):
    result = cleaning.fill_missing_numeric(small_df, ["age"], "median")
    assert result.df.loc[2, "age"] == pytest.approx(small_df["age"].median())
    assert small_df["age"].isna().sum() == 1


def test_fill_numeric_skips_text_and_complete_columns_with_reasons(small_df):
    result = cleaning.fill_missing_numeric(small_df, ["age", "city", "score"], "mean")
    reasons = " | ".join(result.skipped)
    assert "city: not a numeric column" in reasons
    assert "score: no missing values" in reasons
    assert result.entry["columns"] == ["age"]


def test_fill_numeric_all_missing_column_is_reported_not_filled():
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, np.nan, 3.0]})
    result = cleaning.fill_missing_numeric(df, ["a", "b"], "mean")
    assert result.df["a"].isna().all()  # no mean exists, so it stays missing
    assert result.df["b"].tolist() == [1.0, 2.0, 3.0]
    assert any("every value is missing" in s for s in result.skipped)


def test_fill_numeric_nothing_to_do_raises():
    df = pd.DataFrame({"a": [np.nan, np.nan]})
    with pytest.raises(CleaningError, match="Nothing was filled"):
        cleaning.fill_missing_numeric(df, ["a"], "median")


def test_fill_numeric_rejects_bad_method(small_df):
    with pytest.raises(CleaningError):
        cleaning.fill_missing_numeric(small_df, ["age"], "mode")


def test_fill_mode_for_categorical(small_df):
    result = cleaning.fill_missing_mode(small_df, ["city"])
    # B appears 3 times, A twice -> B
    assert result.df.loc[3, "city"] == "B"
    assert result.df["city"].isna().sum() == 0
    assert small_df["city"].isna().sum() == 1


def test_fill_mode_all_missing_column_has_no_mode():
    df = pd.DataFrame({"c": [None, None, None], "d": ["x", None, "x"]})
    result = cleaning.fill_missing_mode(df, ["c", "d"])
    assert result.df["c"].isna().all()
    assert result.df["d"].tolist() == ["x", "x", "x"]
    assert any("no mode" in s for s in result.skipped)


def test_fill_constant_converts_to_column_type(small_df):
    result = cleaning.fill_missing_constant(small_df, ["age", "city"], "0")
    assert result.df.loc[2, "age"] == 0.0
    assert result.df.loc[3, "city"] == "0"


def test_fill_constant_rejects_value_that_does_not_fit_numeric_column(small_df):
    with pytest.raises(CleaningError, match="cannot be used"):
        cleaning.fill_missing_constant(small_df, ["age"], "abc")


def test_fill_constant_requires_a_value(small_df):
    with pytest.raises(CleaningError, match="Enter a value"):
        cleaning.fill_missing_constant(small_df, ["age"], "  ")


def test_drop_missing_rows_any_and_all():
    df = pd.DataFrame({"a": [1, np.nan, np.nan, 4], "b": [1, 2, np.nan, 4]})
    any_result = cleaning.drop_missing_rows(df)
    assert any_result.df["a"].tolist() == [1.0, 4.0]
    assert any_result.entry["rows_before"] == 4 and any_result.entry["rows_after"] == 2
    assert any_result.entry["rows_affected"] == 2
    all_result = cleaning.drop_missing_rows(df, how="all")
    assert len(all_result.df) == 3
    assert list(all_result.df.index) == [0, 1, 2]  # index is reset


def test_drop_missing_rows_subset_only_looks_at_selected_columns():
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3]})
    result = cleaning.drop_missing_rows(df, ["a"])
    assert result.df["a"].tolist() == [1.0, 3.0]
    assert result.df["b"].isna().sum() == 1  # b's missing value is kept: it was not selected


def test_drop_missing_rows_refuses_to_empty_the_dataset():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]})
    with pytest.raises(CleaningError, match="every row"):
        cleaning.drop_missing_rows(df)


def test_drop_missing_rows_without_missing_values_raises(classification_df):
    clean = classification_df.dropna()
    with pytest.raises(CleaningError, match="nothing to drop"):
        cleaning.drop_missing_rows(clean)


def test_remove_duplicates_keeps_first_by_default():
    df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "x", "y", "z", "w"]})
    result = cleaning.remove_duplicates(df)
    assert len(result.df) == 4
    assert result.entry["rows_affected"] == 1
    assert result.df["b"].tolist() == ["x", "y", "z", "w"]


def test_remove_duplicates_on_subset_and_keep_last():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["first", "last", "only"]})
    result = cleaning.remove_duplicates(df, ["a"], keep="last")
    assert result.df["b"].tolist() == ["last", "only"]


def test_remove_duplicates_none_found_raises(small_df):
    with pytest.raises(CleaningError, match="No duplicate"):
        cleaning.remove_duplicates(small_df.drop_duplicates())


def test_convert_dtype_success_and_strict_failure():
    df = pd.DataFrame({"n": ["1", "2", "3"], "bad": ["1", "x", "3"]})
    ok = cleaning.convert_dtype(df, "n", "integer")
    assert str(ok.df["n"].dtype) == "Int64"
    assert ok.df["n"].tolist() == [1, 2, 3]
    with pytest.raises(CleaningError, match="cannot be converted"):
        cleaning.convert_dtype(df, "bad", "float")  # default errors='raise'


def test_convert_dtype_coerce_turns_failures_into_missing_and_logs_them():
    df = pd.DataFrame({"bad": ["1", "x", "3"]})
    result = cleaning.convert_dtype(df, "bad", "float", errors="coerce")
    assert result.df["bad"].isna().sum() == 1
    assert result.df["bad"].tolist()[0] == 1.0
    assert "1 unconvertible value(s) became missing" in result.entry["details"]
    assert df["bad"].tolist() == ["1", "x", "3"]  # source unchanged


def test_convert_to_datetime_and_boolean():
    df = pd.DataFrame({"d": ["2024-01-05", "2024-02-10"], "flag": ["yes", "no"]})
    assert pd.api.types.is_datetime64_any_dtype(cleaning.convert_dtype(df, "d", "datetime").df["d"])
    flags = cleaning.convert_dtype(df, "flag", "boolean").df["flag"]
    assert flags.tolist() == [True, False]


def test_integer_conversion_rejects_fractional_values():
    df = pd.DataFrame({"x": [1.0, 2.5]})
    with pytest.raises(CleaningError):
        cleaning.convert_dtype(df, "x", "integer")


def test_unknown_conversion_target():
    with pytest.raises(CleaningError, match="Unsupported target type"):
        cleaning.convert_dtype(pd.DataFrame({"x": [1]}), "x", "complex")


def test_rename_columns_and_uniqueness_rule():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = cleaning.rename_columns(df, {"a": "alpha"})
    assert list(result.df.columns) == ["alpha", "b"]
    assert list(df.columns) == ["a", "b"]
    with pytest.raises(CleaningError, match="unique"):
        cleaning.rename_columns(df, {"a": "b"})
    with pytest.raises(CleaningError, match="No column names were changed"):
        cleaning.rename_columns(df, {"a": "a"})


def test_drop_columns_cannot_remove_everything():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert list(cleaning.drop_columns(df, ["a"]).df.columns) == ["b"]
    with pytest.raises(CleaningError, match="every column"):
        cleaning.drop_columns(df, ["a", "b"])
    with pytest.raises(CleaningError, match="Unknown column"):
        cleaning.drop_columns(df, ["zzz"])


def test_blank_text_becomes_missing():
    df = pd.DataFrame({"t": ["a", "  ", "", "b"]})
    result = cleaning.blank_to_missing(df)
    assert result.df["t"].isna().sum() == 2


def test_outlier_removal_requires_confirmation_and_removes_only_outliers():
    df = pd.DataFrame({"v": [10, 11, 12, 11, 10, 12, 11, 500]})
    with pytest.raises(CleaningError, match="explicit confirmation"):
        cleaning.remove_outliers_iqr(df, ["v"])
    result = cleaning.remove_outliers_iqr(df, ["v"], confirm=True)
    assert 500 not in result.df["v"].tolist()
    assert len(result.df) == 7


def test_outlier_removal_rejects_text_columns():
    df = pd.DataFrame({"t": list("abcdefgh")})
    with pytest.raises(CleaningError, match="numeric"):
        cleaning.remove_outliers_iqr(df, ["t"], confirm=True)


def test_empty_dataframe_is_rejected_by_every_operation():
    empty = pd.DataFrame()
    with pytest.raises(CleaningError, match="empty"):
        cleaning.drop_missing_rows(empty)
    with pytest.raises(CleaningError, match="empty"):
        cleaning.remove_duplicates(empty)
    with pytest.raises(CleaningError, match="empty"):
        cleaning.fill_missing_numeric(empty, ["a"], "mean")


def test_summarize_change_reports_before_and_after(small_df):
    after = cleaning.drop_missing_rows(small_df).df
    table = cleaning.summarize_change(small_df, after).set_index("Metric")
    assert table.loc["Rows", "Before"] == 6 and table.loc["Rows", "After"] == 4
    assert table.loc["Missing cells", "After"] == 0
    assert table.loc["Rows", "Change"] == -2


def test_audit_entry_has_required_fields(small_df):
    entry = cleaning.fill_missing_numeric(small_df, ["age"], "mean").entry
    for key in ("timestamp", "operation", "columns", "rows_before", "rows_after", "rows_affected", "cells_changed", "details"):
        assert key in entry
    pd.Timestamp(entry["timestamp"])  # parseable timestamp


def test_csv_and_xlsx_export_round_trip(small_df):
    csv_bytes = cleaning.to_csv_bytes(small_df)
    assert csv_bytes.startswith(b"\xef\xbb\xbf")  # BOM for Excel on Windows
    back = pd.read_csv(io.BytesIO(csv_bytes))
    assert back.shape == small_df.shape
    xlsx = pd.read_excel(io.BytesIO(cleaning.to_xlsx_bytes(small_df)), engine="openpyxl")
    assert xlsx.shape == small_df.shape
    assert xlsx["score"].tolist() == small_df["score"].tolist()
