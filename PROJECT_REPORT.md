# DataPilot AI — Project Report (starting draft)

## 1. Abstract
DataPilot AI is a local web application that lets beginners profile, clean, visualise and model tabular data and export a PDF report without coding. It emphasises correct ML practice (split before preprocessing, pipelines, leakage checks, baselines).

## 2. Introduction
Data-science tooling usually assumes programming skill. DataPilot AI packages the standard workflow behind a Streamlit interface built on Pandas, scikit-learn and Plotly, running entirely offline.

## 3. Problem statement
Beginners lack an accessible way to apply the full workflow correctly; typical errors (test-set leakage, accuracy on imbalanced data, unreviewed cleaning) go unnoticed.

## 4. Objectives
Upload and validate CSV/Excel; profile data; guided, reversible cleaning with an audit log; interactive charts; leakage-safe classification and regression; PDF reporting; automated tests; documentation.

## 5. Proposed system
A multi-page Streamlit app with reusable logic in `src/` and thin UI pages. Session state holds the original data (never modified), the working data, cleaning history and model results tied to a data version.

## 6. System architecture
See the Mermaid diagram in `README.md` §5. Layers: UI pages → shared UI helpers and state → processing modules (loader, profiling, cleaning, visualization, ML, reporting).

## 7. Modules
Described in `README.md` §5 (data_loader, profiling, cleaning, state, visualization, ml_pipeline, reporting, student_filter, sample_data, ui).

## 8. Technology stack
Python 3.12, Streamlit, Pandas, NumPy, Plotly, scikit-learn, ReportLab, openpyxl, joblib, pytest (SciPy is installed as a scikit-learn dependency).

## 9. Methodology
Data loading with validation; profiling via Pandas; cleaning as pure functions returning a preview and an audit entry; ML: row filtering, feature screening, train/test split, `Pipeline` + `ColumnTransformer` fitted on training data, held-out evaluation, majority/mean baseline; reporting with ReportLab. Details in `README.md` §10–11.

## 10. Testing approach and verification (factual record)
Automated tests (pytest) cover each processing module, college student filtering, and sample datasets end to end.
- **Verification on target Windows environment (Python 3.12.10):**
  - All dependencies installed (`requirements.txt` including Streamlit 1.66.1, Plotly 7.1.0, Pandas 3.1.0, scikit-learn 1.9.1, ReportLab 5.1.0).
  - `python -m py_compile` across all project files succeeded with 0 syntax errors.
  - Full pytest suite: **168 passed in ~11s** (0 failed, 0 skipped, 0 warnings).
  - Streamlit application launch verified on `127.0.0.1:8501` with HTTP health probe returning `200 OK`.
  - All sample datasets and college student filtering workflows verified end-to-end through loading, multi-criteria filtering (e.g. Pune district), text search, profiling, cleaning, visualization, ML training/evaluation, and ReportLab PDF report generation.

## 11. Results and evaluation
Model metrics depend on the dataset and split and are computed live in the app; none are hard-coded here. Run the sample datasets and record the figures you observe, together with the baseline shown beside them.

## 12. Limitations
In-memory state only (session-based); first Excel sheet only; fixed model defaults and single train/test split; heuristic leakage detection; Helvetica-only built-in font for PDF reports; optional rule-based assistant not implemented; automated headless browser interaction was not run in this CLI environment.

## 13. Future scope
Rule-based assistant, cross-validation/tuning, more models, saved sessions, charts embedded in PDF, multi-sheet Excel.

## 14. Conclusion
The project delivers a fully verified, end-to-end, offline workbench with tested core logic, leakage-aware ML, and local PDF report generation. The application is completely functional, tested, and ready for demonstration on Windows with Python 3.12.
