"""Sample datasets: deterministic, documented defects present, and usable end to end."""

from __future__ import annotations

import pandas as pd
import pytest

from src import cleaning, profiling, reporting, sample_data
from src.ml_pipeline import train_models
from src.sample_data import SAMPLES, load_sample


def test_generators_are_deterministic():
    pd.testing.assert_frame_equal(sample_data.make_student_performance(), sample_data.make_student_performance())
    pd.testing.assert_frame_equal(sample_data.make_sales_data(), sample_data.make_sales_data())


def test_shipped_csv_files_match_the_generators():
    for key, gen in (("student_performance", sample_data.make_student_performance),
                     ("student_regression", sample_data.make_student_regression),
                     ("sales_data", sample_data.make_sales_data)):
        pd.testing.assert_frame_equal(load_sample(key), pd.read_csv(sample_data.SAMPLE_DATA_DIR / SAMPLES[key].filename))
        assert load_sample(key).shape == gen().shape


def test_documented_defects_really_exist():
    sp = load_sample("student_performance")
    assert sp.duplicated().sum() >= 15 and sp.isna().sum().sum() > 0
    assert load_sample("student_regression").isna().sum().sum() > 0
    sales = load_sample("sales_data")
    assert sales.isna().sum().sum() == 0 and sales.duplicated().sum() == 0


def test_sales_revenue_is_consistent():
    s = load_sample("sales_data")
    assert ((s.units_sold * s.unit_price).round(2) - s.revenue).abs().max() < 0.011
    assert set(s.columns) == {"date", "product", "region", "units_sold", "unit_price", "revenue"}


def test_unknown_sample_key():
    with pytest.raises(KeyError):
        load_sample("nope")


def test_classification_sample_workflow_end_to_end():
    df = load_sample("student_performance")
    assert profiling.dataset_overview(df)["duplicate_rows"] >= 15
    step1 = cleaning.remove_duplicates(df)
    step2 = cleaning.fill_missing_numeric(step1.df, ["attendance_pct", "assignment_score", "sleep_hours"], "median")
    assert step2.df.isna().sum().sum() == 0 and step2.df.duplicated().sum() == 0
    res = train_models(step2.df, "result", "classification")
    assert res.comparison.iloc[0]["Accuracy"] > res.baseline["Accuracy"]
    pdf = reporting.build_pdf_report(step2.df, "student_performance.csv", "Cleaned data (2 operations applied)",
                                     [step1.entry, step2.entry], res)
    assert pdf.startswith(b"%PDF-")


def test_regression_sample_workflow_end_to_end():
    df = load_sample("student_regression")
    res = train_models(df, "final_score", "regression")  # missing values handled inside the pipeline
    assert res.outcomes["Linear Regression"].metrics["R2"] > 0.5
    assert "parental_support" in res.plan.features


def test_sales_sample_supports_charts_and_regression():
    from src import visualization as viz

    df = load_sample("sales_data")
    assert viz.line_chart(df, "date", "revenue", "sum", freq="Month").data["date"].nunique() == 12
    assert viz.bar_chart(df, "product", "revenue", "sum").data["product"].nunique() == 6
    res = train_models(df, "revenue", "regression", ["product", "region", "units_sold"], ["Linear Regression", "Random Forest"])
    assert res.best_model in res.outcomes
