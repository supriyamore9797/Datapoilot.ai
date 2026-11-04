"""Cleaning operations.

Every function is pure: it receives a DataFrame, works on a copy and returns a
:class:`CleaningResult`. Nothing is committed until the caller (the Clean Data page)
passes the result to ``state.commit_cleaning``. This makes "preview" simply "call the
function and show the result", and guarantees the original data can never be modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from src.config import IQR_FACTOR
from src.profiling import column_kind, iqr_outlier_mask
from src.utils import now_iso


class CleaningError(ValueError):
    """Raised with a user-friendly message when an operation is invalid."""


@dataclass
class CleaningResult:
    """The cleaned DataFrame, the audit-log entry, and columns that were skipped (with reasons)."""

    df: pd.DataFrame
    entry: dict[str, Any]
    skipped: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        return self.entry["details"]


# ------------------------------------------------------------------------ helpers
def _require_rows(df: pd.DataFrame) -> None:
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        raise CleaningError("The dataset is empty; there is nothing to clean.")


def _require_columns(df: pd.DataFrame, columns: list[str] | None, what: str = "column") -> list[str]:
    if not columns:
        raise CleaningError(f"Select at least one {what}.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise CleaningError(f"Unknown column(s): {', '.join(map(str, missing))}.")
    return list(columns)


def _entry(operation: str, columns: list[str], before: pd.DataFrame, after: pd.DataFrame,
           cells_changed: int, details: str) -> dict[str, Any]:
    return {
        "timestamp": now_iso(),
        "operation": operation,
        "columns": list(columns),
        "rows_before": int(before.shape[0]),
        "rows_after": int(after.shape[0]),
        "rows_affected": abs(int(before.shape[0]) - int(after.shape[0])),
        "cells_changed": int(cells_changed),
        "details": details,
    }


def summarize_change(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Before/after table of the headline counts, for the 'proposed change' preview."""
    def numbers(frame: pd.DataFrame) -> list[int]:
        return [frame.shape[0], frame.shape[1], int(frame.isna().sum().sum()), int(frame.duplicated().sum()) if len(frame) else 0]

    labels = ["Rows", "Columns", "Missing cells", "Duplicate rows"]
    b, a = numbers(before), numbers(after)
    return pd.DataFrame({"Metric": labels, "Before": b, "After": a, "Change": [x - y for x, y in zip(a, b)]})


# ------------------------------------------------------------------ missing values
def drop_missing_rows(df: pd.DataFrame, columns: list[str] | None = None, how: str = "any") -> CleaningResult:
    """Drop rows with missing values (in ``columns`` or anywhere). ``how`` is ``any`` or ``all``."""
    _require_rows(df)
    if how not in ("any", "all"):
        raise CleaningError("'how' must be 'any' or 'all'.")
    subset = _require_columns(df, columns) if columns else list(df.columns)
    out = df.dropna(subset=subset, how=how)
    removed = len(df) - len(out)
    if removed == 0:
        raise CleaningError("No rows contain missing values for the selected columns; nothing to drop.")
    if out.empty:
        raise CleaningError("This would remove every row. Choose fewer columns or fill the missing values instead.")
    detail = f"Dropped {removed} row(s) with missing values in {len(subset)} column(s) (rule: {how})."
    return CleaningResult(out.reset_index(drop=True), _entry("Drop rows with missing values", subset, df, out, removed, detail))


def fill_missing_numeric(df: pd.DataFrame, columns: list[str], method: str) -> CleaningResult:
    """Fill missing values of numeric columns with their mean or median."""
    _require_rows(df)
    if method not in ("mean", "median"):
        raise CleaningError("Method must be 'mean' or 'median'.")
    cols = _require_columns(df, columns)
    out = df.copy()
    changed, applied, skipped = 0, [], []
    for col in cols:
        series = out[col]
        if column_kind(series) != "numeric":
            skipped.append(f"{col}: not a numeric column")
        elif series.isna().sum() == 0:
            skipped.append(f"{col}: no missing values")
        elif series.notna().sum() == 0:
            skipped.append(f"{col}: every value is missing, so no {method} can be computed")
        else:
            floated = series.astype("float64")
            value = float(floated.mean() if method == "mean" else floated.median())
            n = int(series.isna().sum())
            out[col] = floated.fillna(value)
            changed += n
            applied.append(col)
    if not applied:
        raise CleaningError("Nothing was filled. " + "; ".join(skipped))
    detail = f"Filled {changed} missing value(s) in {len(applied)} column(s) with the {method}."
    return CleaningResult(out, _entry(f"Fill missing (numeric {method})", applied, df, out, changed, detail), skipped)


def fill_missing_mode(df: pd.DataFrame, columns: list[str]) -> CleaningResult:
    """Fill missing values with each column's most frequent value (first one if tied)."""
    _require_rows(df)
    cols = _require_columns(df, columns)
    out = df.copy()
    changed, applied, skipped = 0, [], []
    for col in cols:
        series = out[col]
        if column_kind(series) == "datetime":
            skipped.append(f"{col}: datetime column (use a constant instead)")
        elif series.isna().sum() == 0:
            skipped.append(f"{col}: no missing values")
        else:
            modes = series.dropna().mode()
            if modes.empty:
                skipped.append(f"{col}: every value is missing, so there is no mode")
                continue
            n = int(series.isna().sum())
            out[col] = series.fillna(modes.iloc[0])
            changed += n
            applied.append(col)
    if not applied:
        raise CleaningError("Nothing was filled. " + "; ".join(skipped))
    detail = f"Filled {changed} missing value(s) in {len(applied)} column(s) with the mode."
    return CleaningResult(out, _entry("Fill missing (mode)", applied, df, out, changed, detail), skipped)


def fill_missing_constant(df: pd.DataFrame, columns: list[str], value: str) -> CleaningResult:
    """Fill missing values with a user-supplied constant, converted to each column's type."""
    _require_rows(df)
    cols = _require_columns(df, columns)
    if value is None or str(value).strip() == "":
        raise CleaningError("Enter a value to fill with.")
    text = str(value).strip()
    out = df.copy()
    changed, applied, skipped = 0, [], []
    for col in cols:
        series = out[col]
        n = int(series.isna().sum())
        if n == 0:
            skipped.append(f"{col}: no missing values")
            continue
        kind = column_kind(series)
        try:
            if kind == "numeric":
                number = float(text)
                if pdt.is_integer_dtype(series) and number.is_integer():
                    out[col] = series.fillna(int(number))
                else:
                    out[col] = series.astype("float64").fillna(number)
            elif kind == "boolean":
                out[col] = series.fillna(_parse_bool(text))
            elif kind == "datetime":
                out[col] = series.fillna(pd.Timestamp(text))
            elif isinstance(series.dtype, pd.CategoricalDtype):
                if text not in series.cat.categories:
                    series = series.cat.add_categories([text])
                out[col] = series.fillna(text)
            else:
                out[col] = series.fillna(text)
        except (ValueError, TypeError) as exc:
            raise CleaningError(f"'{text}' cannot be used to fill the {kind} column '{col}'.") from exc
        changed += n
        applied.append(col)
    if not applied:
        raise CleaningError("Nothing was filled. " + "; ".join(skipped))
    detail = f"Filled {changed} missing value(s) in {len(applied)} column(s) with the constant '{text}'."
    return CleaningResult(out, _entry("Fill missing (constant)", applied, df, out, changed, detail), skipped)


def blank_to_missing(df: pd.DataFrame, columns: list[str] | None = None) -> CleaningResult:
    """Turn empty / whitespace-only text values into real missing values."""
    _require_rows(df)
    cols = _require_columns(df, columns) if columns else [c for c in df.columns if column_kind(df[c]) == "categorical"]
    out = df.copy()
    changed, applied = 0, []
    for col in cols:
        if column_kind(out[col]) != "categorical":
            continue
        mask = out[col].map(lambda v: isinstance(v, str) and not v.strip()).astype(bool)
        n = int(mask.sum())
        if n:
            out[col] = out[col].mask(mask)
            changed += n
            applied.append(col)
    if not applied:
        raise CleaningError("No blank text values were found in the selected columns.")
    detail = f"Converted {changed} blank text value(s) to missing in {len(applied)} column(s)."
    return CleaningResult(out, _entry("Blank text to missing", applied, df, out, changed, detail))


# -------------------------------------------------------------------- duplicates
def remove_duplicates(df: pd.DataFrame, columns: list[str] | None = None, keep: str = "first") -> CleaningResult:
    """Remove duplicate rows, comparing all columns or only ``columns``. ``keep``: first or last."""
    _require_rows(df)
    if keep not in ("first", "last"):
        raise CleaningError("'keep' must be 'first' or 'last'.")
    subset = _require_columns(df, columns) if columns else None
    out = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    removed = len(df) - len(out)
    if removed == 0:
        raise CleaningError("No duplicate rows were found; nothing to remove.")
    scope = "selected columns" if subset else "all columns"
    detail = f"Removed {removed} duplicate row(s) (compared on {scope}, kept the {keep} occurrence)."
    return CleaningResult(out, _entry("Remove duplicate rows", subset or list(df.columns), df, out, removed, detail))


# ----------------------------------------------------------------- type conversion
CONVERSION_TARGETS = ("integer", "float", "string", "category", "datetime", "boolean")
_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


def _parse_bool(text: str) -> bool:
    lowered = str(text).strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"not a boolean: {text}")


def _convert(series: pd.Series, target: str, datetime_format: str | None) -> pd.Series:
    """Convert leniently: values that cannot be converted become missing."""
    if target == "float":
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if target == "integer":
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        fractional = numeric.notna() & (numeric % 1 != 0)
        numeric = numeric.mask(fractional | np.isinf(numeric))
        return numeric.astype("Int64")
    if target == "string":
        return series.astype("string")
    if target == "category":
        return series.astype("category")
    if target == "datetime":
        if pdt.is_datetime64_any_dtype(series):
            return series
        text = series.astype("object").where(series.notna(), None)
        if datetime_format:
            return pd.to_datetime(text, format=datetime_format, errors="coerce")
        return pd.to_datetime(text, format="mixed", errors="coerce")
    if target == "boolean":
        def to_bool(v: Any) -> Any:
            if pd.isna(v):
                return pd.NA
            try:
                return _parse_bool(str(v))
            except ValueError:
                return pd.NA
        return series.map(to_bool).astype("boolean")
    raise CleaningError(f"Unsupported target type '{target}'. Choose one of: {', '.join(CONVERSION_TARGETS)}.")


def preview_conversion(df: pd.DataFrame, column: str, target: str, datetime_format: str | None = None) -> dict[str, Any]:
    """Report how many values a conversion would turn into missing values, with examples."""
    _require_rows(df)
    _require_columns(df, [column])
    converted = _convert(df[column], target, datetime_format)
    failed = df[column].notna() & converted.isna()
    return {
        "column": column,
        "target": target,
        "values": int(df[column].notna().sum()),
        "failures": int(failed.sum()),
        "examples": [str(v) for v in df.loc[failed, column].drop_duplicates().head(5)],
    }


def convert_dtype(df: pd.DataFrame, column: str, target: str, errors: str = "raise",
                  datetime_format: str | None = None) -> CleaningResult:
    """Convert a column's type. With ``errors='raise'`` any unconvertible value is an error;
    with ``errors='coerce'`` such values become missing and are counted in the audit log."""
    _require_rows(df)
    _require_columns(df, [column])
    if errors not in ("raise", "coerce"):
        raise CleaningError("'errors' must be 'raise' or 'coerce'.")
    info = preview_conversion(df, column, target, datetime_format)
    if info["failures"] and errors == "raise":
        raise CleaningError(
            f"{info['failures']} value(s) in '{column}' cannot be converted to {target} "
            f"(for example: {', '.join(info['examples'])}). Choose 'turn into missing' to proceed anyway."
        )
    out = df.copy()
    out[column] = _convert(df[column], target, datetime_format)
    detail = f"Converted '{column}' from {df[column].dtype} to {out[column].dtype}."
    if info["failures"]:
        detail += f" {info['failures']} unconvertible value(s) became missing."
    return CleaningResult(out, _entry("Convert data type", [column], df, out, info["values"], detail))


# --------------------------------------------------------------- rename / drop cols
def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> CleaningResult:
    """Rename columns. New names must be non-blank and the final names must be unique."""
    _require_rows(df)
    cleaned = {old: str(new).strip() for old, new in mapping.items() if str(new).strip() != old}
    if not cleaned:
        raise CleaningError("No column names were changed.")
    _require_columns(df, list(cleaned))
    if any(not new for new in cleaned.values()):
        raise CleaningError("New column names cannot be blank.")
    final = [cleaned.get(c, c) for c in df.columns]
    if len(set(final)) != len(final):
        dupes = sorted({n for n in final if final.count(n) > 1})
        raise CleaningError(f"Column names must be unique. Duplicate name(s): {', '.join(dupes)}.")
    out = df.rename(columns=cleaned)
    detail = "Renamed " + ", ".join(f"'{o}' to '{n}'" for o, n in cleaned.items()) + "."
    return CleaningResult(out, _entry("Rename columns", list(cleaned), df, out, 0, detail))


def drop_columns(df: pd.DataFrame, columns: list[str]) -> CleaningResult:
    """Drop the selected columns (at least one column must remain)."""
    _require_rows(df)
    cols = _require_columns(df, columns)
    if len(cols) >= df.shape[1]:
        raise CleaningError("You cannot drop every column.")
    out = df.drop(columns=cols)
    detail = f"Dropped {len(cols)} column(s): {', '.join(cols)}."
    return CleaningResult(out, _entry("Drop columns", cols, df, out, 0, detail))


# ----------------------------------------------------------------------- outliers
def remove_outliers_iqr(df: pd.DataFrame, columns: list[str], factor: float = IQR_FACTOR,
                        confirm: bool = False) -> CleaningResult:
    """Remove rows with IQR outliers in any selected numeric column. Requires ``confirm=True``."""
    _require_rows(df)
    if not confirm:
        raise CleaningError("Removing outliers deletes data and needs explicit confirmation.")
    cols = _require_columns(df, columns)
    non_numeric = [c for c in cols if column_kind(df[c]) != "numeric"]
    if non_numeric:
        raise CleaningError(f"Outlier removal needs numeric columns: {', '.join(non_numeric)}.")
    if factor <= 0:
        raise CleaningError("The IQR factor must be positive.")
    mask = pd.Series(False, index=df.index)
    for col in cols:
        mask |= iqr_outlier_mask(df[col], factor)
    removed = int(mask.sum())
    if removed == 0:
        raise CleaningError("No outliers were found in the selected columns.")
    out = df.loc[~mask].reset_index(drop=True)
    if out.empty:
        raise CleaningError("This would remove every row. Use a larger IQR factor or fewer columns.")
    detail = f"Removed {removed} row(s) containing IQR outliers (factor {factor:g}) in {len(cols)} column(s)."
    return CleaningResult(out, _entry("Remove outliers (IQR)", cols, df, out, removed, detail))


# ------------------------------------------------------------------------ export
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 CSV (with BOM so Excel on Windows shows special characters correctly)."""
    return df.to_csv(index=False).encode("utf-8-sig")


def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    """Excel workbook with a single sheet."""
    from io import BytesIO

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return buffer.getvalue()
