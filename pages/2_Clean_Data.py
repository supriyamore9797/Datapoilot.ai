"""Clean Data page: guided, previewed, undoable cleaning with an audit log."""

import pandas as pd
import streamlit as st

from src import cleaning, profiling, state, ui
from src.cleaning import CleaningError
from src.config import IQR_FACTOR

ui.setup_page("Clean Data", "Choose an operation, review the proposed change, then apply it. Every step is logged and the original data is never modified.")
ui.show_flash()
df = ui.require_data()

OPERATIONS = [
    "Drop rows with missing values",
    "Fill missing numbers (mean)",
    "Fill missing numbers (median)",
    "Fill missing categories (mode)",
    "Fill missing with a constant",
    "Convert blank text to missing",
    "Remove duplicate rows",
    "Convert column type",
    "Rename columns",
    "Drop columns",
    "Remove outliers (IQR)",
]

numeric_cols = profiling.numeric_columns(df)
with_missing = [c for c in df.columns if df[c].isna().any()]

# ---------------------------------------------------------------- toolbar: undo / reset
bar = st.columns([1, 1, 2])
if bar[0].button("Undo last step", disabled=not state.can_undo(), help="Restore the data as it was before the most recent operation."):
    undone = state.undo_last()
    ui.set_flash("success", f"Undid: {undone['operation']}." if undone else "Nothing to undo.")
    st.rerun()
with bar[1].popover("Reset to original"):
    st.write("This removes every cleaning step and restores the data as it was loaded.")
    confirm_reset = st.checkbox("Yes, reset everything", key="confirm_reset")
    if st.button("Reset now", disabled=not confirm_reset):
        state.reset_to_original()
        ui.set_flash("success", "The data was reset to the original.")
        st.rerun()

st.divider()
st.subheader("1. Choose an operation")
operation = st.selectbox("Operation", OPERATIONS, key="clean_operation")


def propose():
    """Collect the options for the selected operation and return a CleaningResult (a preview)."""
    if operation == "Drop rows with missing values":
        cols = st.multiselect("Only consider these columns (empty = all columns)", list(df.columns), key="drop_na_cols")
        how = st.radio("Drop a row when", ["any", "all"], format_func=lambda v: "any selected value is missing" if v == "any" else "all selected values are missing", horizontal=True, key="drop_na_how")
        return cleaning.drop_missing_rows(df, cols or None, how)
    if operation in ("Fill missing numbers (mean)", "Fill missing numbers (median)"):
        method = "mean" if "mean" in operation else "median"
        options = [c for c in numeric_cols if c in with_missing]
        if not options:
            raise CleaningError("No numeric column has missing values.")
        cols = st.multiselect("Numeric columns to fill", options, default=options, key=f"fill_num_{method}")
        return cleaning.fill_missing_numeric(df, cols, method)
    if operation == "Fill missing categories (mode)":
        options = [c for c in with_missing if c not in numeric_cols]
        if not options:
            raise CleaningError("No categorical column has missing values.")
        cols = st.multiselect("Columns to fill with their most frequent value", options, default=options, key="fill_mode")
        return cleaning.fill_missing_mode(df, cols)
    if operation == "Fill missing with a constant":
        if not with_missing:
            raise CleaningError("No column has missing values.")
        cols = st.multiselect("Columns", with_missing, key="fill_const_cols")
        value = st.text_input("Constant value (converted to each column's type)", key="fill_const_value")
        return cleaning.fill_missing_constant(df, cols, value)
    if operation == "Convert blank text to missing":
        return cleaning.blank_to_missing(df)
    if operation == "Remove duplicate rows":
        cols = st.multiselect("Compare only these columns (empty = all columns)", list(df.columns), key="dup_cols")
        keep = st.radio("Keep", ["first", "last"], horizontal=True, key="dup_keep")
        return cleaning.remove_duplicates(df, cols or None, keep)
    if operation == "Convert column type":
        column = st.selectbox("Column", list(df.columns), key="conv_col")
        target = st.selectbox("Convert to", cleaning.CONVERSION_TARGETS, key="conv_target")
        fmt = st.text_input("Date format (optional, e.g. %d/%m/%Y)", key="conv_fmt") if target == "datetime" else None
        info = cleaning.preview_conversion(df, column, target, fmt or None)
        mode = "raise"
        if info["failures"]:
            st.warning(f"{info['failures']} of {info['values']} value(s) cannot be converted (e.g. {', '.join(info['examples'])}).")
            mode = "coerce" if st.checkbox("Convert anyway and turn those values into missing values", key="conv_coerce") else "raise"
        return cleaning.convert_dtype(df, column, target, mode, fmt or None)
    if operation == "Rename columns":
        cols = st.multiselect("Columns to rename", list(df.columns), key="ren_cols")
        mapping = {c: st.text_input(f"New name for '{c}'", value=c, key=f"ren_{c}") for c in cols}
        return cleaning.rename_columns(df, mapping)
    if operation == "Drop columns":
        cols = st.multiselect("Columns to drop", list(df.columns), key="dropcols")
        return cleaning.drop_columns(df, cols)
    if operation == "Remove outliers (IQR)":
        summary = profiling.outlier_summary(df)
        st.dataframe(summary, hide_index=True)
        st.caption("Outliers are flagged by the IQR rule. They may be genuine values, so removal needs explicit confirmation.")
        options = [c for c in numeric_cols if not summary.empty and c in set(summary["Column"])]
        cols = st.multiselect("Columns to check", options, key="out_cols")
        factor = st.number_input("IQR factor", min_value=0.5, max_value=5.0, value=float(IQR_FACTOR), step=0.5, key="out_factor")
        confirmed = st.checkbox("I understand this deletes the rows that contain outliers", key="out_confirm")
        return cleaning.remove_outliers_iqr(df, cols, factor, confirm=confirmed)
    raise CleaningError("Unknown operation.")


st.subheader("2. Review the proposed change")
result = None
try:
    result = propose()
except CleaningError as exc:
    st.warning(str(exc))

if result is not None:
    st.dataframe(cleaning.summarize_change(df, result.df), hide_index=True)
    st.info(result.message)
    for note in result.skipped:
        st.caption(f"Skipped - {note}")
    st.subheader("3. Apply")
    if st.button("Apply this change", type="primary", key="apply_clean"):
        state.commit_cleaning(result.df, result.entry)
        ui.set_flash("success", result.message)
        st.rerun()

# ------------------------------------------------------------------ history and export
st.divider()
history, export = st.tabs(["Cleaning history", "Download cleaned data"])
with history:
    log = st.session_state["cleaning_log"]
    if not log:
        st.info("No cleaning steps have been applied. You are working with the original data.")
    else:
        table = pd.DataFrame(log)[["timestamp", "operation", "columns", "rows_before", "rows_after", "cells_changed", "details"]]
        table["columns"] = table["columns"].map(lambda c: ", ".join(c))
        st.dataframe(table, hide_index=True)
    with st.expander("Full audit trail (including undo and reset)"):
        trail = st.session_state["audit_trail"]
        if trail:
            frame = pd.DataFrame(trail)
            frame["columns"] = frame["columns"].map(lambda c: ", ".join(c) if isinstance(c, list) else c)
            st.dataframe(frame[["timestamp", "event", "operation", "columns"]], hide_index=True)
        else:
            st.caption("Nothing recorded yet.")
with export:
    name = st.session_state["dataset_name"].rsplit(".", 1)[0]
    st.write(f"Current data version: **{state.version_label()}** ({len(df):,} rows).")
    st.download_button("Download as CSV", cleaning.to_csv_bytes(df), file_name=f"{name}_cleaned.csv", mime="text/csv")
    if st.checkbox("Prepare an Excel (.xlsx) file", key="prepare_xlsx"):
        st.download_button("Download as Excel", cleaning.to_xlsx_bytes(df), file_name=f"{name}_cleaned.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

ui.show_footer()
