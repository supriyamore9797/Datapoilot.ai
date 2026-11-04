# Viva Questions and Simple Answers

## Python / Pandas
- **Why a virtual environment?** It isolates this project's package versions from other projects.
- **DataFrame vs Series?** A DataFrame is a table; a Series is one column.
- **How do you find missing values?** `df.isna().sum()`; percent = count / rows × 100.
- **Why copy DataFrames (`df.copy()`)?** So an operation cannot change data used elsewhere; the original dataset stays untouched.
- **What does `groupby` do?** Splits rows by a key, aggregates each group (sum, mean, count).
- **Why type hints / docstrings?** Readability and fewer mistakes.

## Streamlit
- **How does Streamlit run?** The script re-runs top to bottom on every interaction.
- **How do you keep data between reruns?** `st.session_state`; `src/state.py` manages it centrally.
- **Why `st.navigation`/pages?** Separate screens for each workflow step.
- **Is data uploaded to a server?** No; it is processed in memory on the local machine.
- **What happens when the session ends?** State is lost (documented limitation).

## Data cleaning
- **Mean vs median imputation?** Median is robust to outliers; mean is fine for symmetric data.
- **Mode imputation?** Most frequent value, used for categories.
- **Why not delete outliers automatically?** They may be real; removal needs explicit confirmation.
- **IQR rule?** Outlier if below Q1 − 1.5·IQR or above Q3 + 1.5·IQR.
- **How is undo safe?** Each step stores the previous DataFrame; the original is kept separately and never modified.
- **Why an audit log?** Reproducibility: operation, columns, rows affected, timestamp.

## Visualisation
- **Which chart for what?** Histogram: one numeric distribution; bar: category totals; scatter: two numerics; box: spread and outliers; heatmap: correlations; line: trend over time.
- **Does correlation mean causation?** No; the app says so next to every correlation.
- **Why sample points above 20,000?** Keeps charts responsive; the app tells the user and summaries use all rows.

## Machine learning
- **Classification vs regression?** Predict a category vs a number.
- **Why split into train and test?** To measure performance on unseen data.
- **What is data leakage?** Information from the test set or the target leaking into training, giving falsely good results.
- **How do you avoid it?** Split first; fit imputers/encoders on training only (inside a `Pipeline`); drop columns derived from the target.
- **What is a Pipeline / ColumnTransformer?** A Pipeline chains steps so they run in order; ColumnTransformer applies different preprocessing to numeric and categorical columns.
- **Why one-hot encoding?** Models need numbers; one-hot makes one 0/1 column per category; `handle_unknown="ignore"` copes with new categories.
- **Why stratified split?** Keeps class proportions similar in train and test.
- **Why `random_state`?** Reproducible results.
- **Why scale features for logistic regression?** It is sensitive to feature scale; trees are not.
- **Why a baseline?** A model must beat "always predict the majority class / the mean" to be useful.
- **Did you tune on the test set?** No; fixed defaults, test set used only for final evaluation.

## Metrics
- **Accuracy?** Share of correct predictions; misleading on imbalanced data.
- **Precision / recall / F1?** Precision: of predicted positives, how many are right; recall: of real positives, how many found; F1: their harmonic mean.
- **Weighted vs binary averaging?** Binary scores one positive class; weighted averages classes by size for multiclass.
- **Confusion matrix?** Table of actual vs predicted classes.
- **MAE / RMSE / R²?** Average absolute error; root of mean squared error (penalises big errors); fraction of variance explained (can be negative on test data).

## Design decisions
- **Why local-first?** Privacy, no cost, works offline.
- **Why `src/` modules separate from pages?** Logic is testable without the UI.
- **Why results tied to a data version?** So old models/reports are never shown for changed data.
- **Why a PDF with ReportLab?** Pure Python, local, no external service.
- **Limitations?** In-memory state, fixed models, heuristic leakage checks, UI verified by tests of logic rather than a browser in development.
