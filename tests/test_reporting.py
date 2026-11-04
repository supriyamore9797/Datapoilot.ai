"""PDF report tests: valid file, real values, optional sections, awkward input."""

from __future__ import annotations

import pandas as pd
import pytest
from reportlab import rl_config

from src import cleaning, reporting
from src.ml_pipeline import train_models
from src.sample_data import make_student_performance


@pytest.fixture
def plain_text_pdfs(monkeypatch):
    """Disable stream compression so the PDF's text can be searched."""
    monkeypatch.setattr(rl_config, "pageCompression", 0)


def test_report_without_models_is_a_valid_pdf_and_says_ml_is_unavailable(plain_text_pdfs):
    df = make_student_performance()
    pdf = reporting.build_pdf_report(df, "student.csv", "Original data")
    assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF") and len(pdf) > 3000
    assert b"Not available: no model has been trained" in pdf
    assert b"No cleaning operations were applied" in pdf
    assert b"Data Analysis Report" in pdf and b"Generated on" in pdf


def test_report_contains_real_values_from_the_data(plain_text_pdfs):
    df = make_student_performance()
    pdf = reporting.build_pdf_report(df, "student.csv", "Original data")
    assert f"{len(df):,}".encode() in pdf  # row count
    assert b"study_hours" in pdf and b"attendance_pct" in pdf
    assert f"{df.duplicated().sum()} duplicate row".encode() in pdf


def test_report_with_cleaning_log_and_model_results(plain_text_pdfs, classification_df):
    cleaned = cleaning.drop_missing_rows(classification_df)
    results = train_models(cleaned.df, "label", "classification", ["x1", "x2", "group"])
    pdf = reporting.build_pdf_report(cleaned.df, "demo.csv", "Cleaned data (1 operation applied)", [cleaned.entry], results)
    assert b"No cleaning operations were applied" not in pdf and b"Drop" in pdf
    assert b"held-out test performance" in pdf
    assert results.best_model.encode() in pdf
    assert b"Not available: no model" not in pdf
    assert b"Baseline" in pdf


def test_regression_report_mentions_negative_r2(plain_text_pdfs, regression_df):
    results = train_models(regression_df, "y", "regression", ["a", "b", "kind"])
    pdf = reporting.build_pdf_report(regression_df, "reg.csv", "Original data", None, results)
    assert b"can be negative" in pdf


def test_long_names_special_characters_and_wide_tables_do_not_break_the_pdf():
    wide = pd.DataFrame({f"a_very_long_column_name_{i}_" + "x" * 40: [1.0, 2.0, None] for i in range(30)})
    wide["texte_é_日本"] = ["café", "<b>&</b>", None]
    pdf = reporting.build_pdf_report(wide, "wéird <name>.csv", "Original data")
    assert pdf.startswith(b"%PDF-") and len(pdf) > 3000


def test_save_report_writes_a_pdf_file(tmp_path):
    pdf = reporting.build_pdf_report(pd.DataFrame({"a": [1, 2, 3]}), "x.csv", "Original data")
    path = reporting.save_report(pdf, "my data?.csv", tmp_path / "reports")
    assert path.exists() and path.suffix == ".pdf"
    assert path.read_bytes() == pdf and path.read_bytes().startswith(b"%PDF-")
    assert "?" not in path.name


def test_report_requires_a_dataset():
    with pytest.raises(ValueError):
        reporting.build_pdf_report(pd.DataFrame(), "x", "Original data")
    with pytest.raises(ValueError):
        reporting.build_pdf_report(None, "x", "Original data")


def test_report_contains_authorship_attribution(plain_text_pdfs, small_df):
    pdf = reporting.build_pdf_report(small_df, "students.csv", "Original data")
    assert b"Generated using DataPilot AI" in pdf
    assert b"Developed by Supriya S More" in pdf
    assert b"All rights reserved" in pdf

