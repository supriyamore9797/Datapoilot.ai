"""Unit and integration tests for student data filtering and analysis."""

from __future__ import annotations

import io
import pandas as pd
import pytest

from src import student_filter
from src.student_filter import (
    MISSING_VALUE_LABEL,
    compute_column_breakdown,
    detect_student_columns,
    filter_students,
    generate_student_insights,
    make_college_roster_sample,
    summarize_filter_results,
)
from src.cleaning import to_csv_bytes, to_xlsx_bytes


@pytest.fixture
def college_df() -> pd.DataFrame:
    """Fixture providing a deterministic sample college student dataset."""
    return make_college_roster_sample(n=300, seed=123)


def test_filtering_pune_students_correctly(college_df):
    """Verify that selecting District=Pune only returns Pune students."""
    original_copy = college_df.copy(deep=True)
    res = filter_students(college_df, filters={"district": ["Pune"]})

    assert not res.empty
    assert (res["district"] == "Pune").all()
    # Confirm records from other districts are excluded
    assert "Nashik" not in res["district"].values
    assert "Satara" not in res["district"].values
    assert len(res) < len(college_df)
    # Confirm original dataframe is completely unmodified
    pd.testing.assert_frame_equal(college_df, original_copy)


def test_combining_district_department_year_filters(college_df):
    """Verify multi-criteria AND filtering (District=Pune AND Department=AIML AND Academic Year=2nd Year)."""
    res = filter_students(
        college_df,
        filters={
            "district": ["Pune"],
            "department": ["AIML"],
            "academic_year": ["2nd Year"],
        },
    )

    assert not res.empty
    assert (res["district"] == "Pune").all()
    assert (res["department"] == "AIML").all()
    assert (res["academic_year"] == "2nd Year").all()
    # Less records than Pune alone
    pune_only = filter_students(college_df, filters={"district": ["Pune"]})
    assert len(res) < len(pune_only)


def test_clear_filters_restoring_all_rows(college_df):
    """Verify that clearing filters (empty dict or empty value lists) returns all rows."""
    res1 = filter_students(college_df, filters={})
    assert len(res1) == len(college_df)

    res2 = filter_students(college_df, filters={"district": []})
    assert len(res2) == len(college_df)

    res3 = filter_students(college_df, filters=None, search_query="")
    assert len(res3) == len(college_df)


def test_missing_filter_columns_handled_gracefully(college_df):
    """Verify filtering against non-existent column does not raise an exception."""
    res = filter_students(college_df, filters={"non_existent_col": ["Pune"]})
    assert len(res) == len(college_df)


def test_missing_values_handled_properly(college_df):
    """Verify missing values can be filtered using MISSING_VALUE_LABEL or excluded when not selected."""
    has_missing = college_df["district"].isna().any()
    assert has_missing

    # Include only missing districts
    res_missing = filter_students(college_df, filters={"district": [MISSING_VALUE_LABEL]})
    assert not res_missing.empty
    assert res_missing["district"].isna().all()

    # Normal filter should not include missing rows
    res_pune = filter_students(college_df, filters={"district": ["Pune"]})
    assert res_pune["district"].isna().sum() == 0


def test_no_matching_results(college_df):
    """Verify filtering with contradictory criteria returns an empty DataFrame with original columns."""
    res = filter_students(
        college_df,
        filters={"district": ["NonExistentDistrict123"]},
    )
    assert res.empty
    assert list(res.columns) == list(college_df.columns)

    metrics = summarize_filter_results(len(college_df), len(res))
    assert metrics["matching_records"] == 0
    assert metrics["match_percentage"] == 0.0
    assert metrics["excluded_records"] == len(college_df)


def test_search_by_student_name_and_id(college_df):
    """Verify substring search by student name or roll number/ID."""
    first_student = college_df.iloc[0]
    first_name = first_student["student_name"].split()[0]
    roll_no = first_student["roll_no"]

    # Search by name substring (case-insensitive)
    res_name = filter_students(college_df, search_query=first_name.lower())
    assert not res_name.empty
    assert any(first_name.lower() in name.lower() for name in res_name["student_name"])

    # Search by Roll number
    res_roll = filter_students(college_df, search_query=roll_no)
    assert len(res_roll) >= 1
    assert (res_roll["roll_no"] == roll_no).any()


def test_csv_and_excel_exports_contain_only_filtered_rows(college_df):
    """Verify exported CSV and Excel bytes match exactly the filtered rows."""
    pune_df = filter_students(college_df, filters={"district": ["Pune"]})
    expected_count = len(pune_df)

    # Test CSV
    csv_bytes = to_csv_bytes(pune_df)
    assert len(csv_bytes) > 0
    df_from_csv = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(df_from_csv) == expected_count
    assert (df_from_csv["district"] == "Pune").all()

    # Test Excel
    xlsx_bytes = to_xlsx_bytes(pune_df)
    assert len(xlsx_bytes) > 0
    df_from_xlsx = pd.read_excel(io.BytesIO(xlsx_bytes))
    assert len(df_from_xlsx) == expected_count
    assert (df_from_xlsx["district"] == "Pune").all()


def test_original_dataset_remains_unchanged(college_df):
    """Verify input dataset is never modified in-place by filtering or search."""
    orig_rows = len(college_df)
    orig_cols = list(college_df.columns)
    orig_checksum = college_df["roll_no"].iloc[0]

    _ = filter_students(college_df, filters={"district": ["Pune"], "department": ["AIML"]})
    _ = filter_students(college_df, search_query="xyz")

    assert len(college_df) == orig_rows
    assert list(college_df.columns) == orig_cols
    assert college_df["roll_no"].iloc[0] == orig_checksum


def test_detect_student_columns(college_df):
    """Verify detection finds recommended filter columns and searchable columns."""
    detected = detect_student_columns(college_df)
    rec = detected["recommended_filters"]
    search = detected["searchable_columns"]

    assert "district" in rec
    assert "department" in rec
    assert "academic_year" in rec
    assert "student_name" in search or "roll_no" in search


def test_insights_and_breakdown_tables(college_df):
    """Verify demographic summary tables and insights generation."""
    pune_df = filter_students(college_df, filters={"district": ["Pune"]})

    insights = generate_student_insights(pune_df, key_columns=["district", "department"])
    assert len(insights) > 0
    assert any("pune" in ins.lower() for ins in insights)

    tbl = compute_column_breakdown(pune_df, "department")
    assert not tbl.empty
    assert "Count" in tbl.columns
    assert "Share %" in tbl.columns

