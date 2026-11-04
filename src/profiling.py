"""Dataset profiling: overview numbers, per-column profiles, quality checks, insights.

Every value returned here is computed from the DataFrame that is passed in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from src.config import HIGH_CARDINALITY_THRESHOLD, IQR_FACTOR
from src.utils import pct


# --------------------------------------------------------------------------- kinds
def column_kind(series: pd.Series) -> str:
    """Classify a column as ``numeric``, ``boolean``, ``datetime`` or ``categorical``."""
    if pdt.is_bool_dtype(series):
        return "boolean"
    if pdt.is_numeric_dtype(series):
        return "numeric"
    if pdt.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if column_kind(df[c]) == "numeric"]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Text, category and boolean columns."""
    return [c for c in df.columns if column_kind(df[c]) in ("categorical", "boolean")]


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if column_kind(df[c]) == "datetime"]


# ----------------------------------------------------------------------- overview
def dataset_overview(df: pd.DataFrame) -> dict[str, float | int]:
    """Headline numbers for a dataset. Safe for empty DataFrames."""
    rows, cols = df.shape
    cells = rows * cols
    missing = int(df.isna().sum().sum()) if cells else 0
    duplicates = int(df.duplicated().sum()) if rows else 0
    kinds = [column_kind(df[c]) for c in df.columns]
    return {
        "rows": rows,
        "columns": cols,
        "numeric_columns": kinds.count("numeric"),
        "categorical_columns": kinds.count("categorical") + kinds.count("boolean"),
        "datetime_columns": kinds.count("datetime"),
        "missing_cells": missing,
        "missing_pct": pct(missing, cells, 2),
        "duplicate_rows": duplicates,
        "duplicate_pct": pct(duplicates, rows, 2),
        "memory_bytes": int(df.memory_usage(deep=True).sum()) if cols else 0,
    }


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, missing values, unique values and an example."""
    rows = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        missing = int(series.isna().sum())
        example = "" if non_null.empty else str(non_null.iloc[0])
        rows.append(
            {
                "Column": col,
                "Type": str(series.dtype),
                "Kind": column_kind(series),
                "Missing": missing,
                "Missing %": pct(missing, n, 2),
                "Unique": int(series.nunique(dropna=True)),
                "Example": example[:40],
            }
        )
    return pd.DataFrame(rows, columns=["Column", "Type", "Kind", "Missing", "Missing %", "Unique", "Example"])


def missing_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """Missing counts per column, largest first."""
    n = len(df)
    counts = df.isna().sum()
    out = pd.DataFrame({"Column": counts.index.astype(str), "Missing": counts.to_numpy()})
    out["Missing %"] = [pct(int(m), n, 2) for m in out["Missing"]]
    return out.sort_values(["Missing", "Column"], ascending=[False, True], ignore_index=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numeric columns (empty frame if there are none)."""
    cols = numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    desc = df[cols].astype("float64").describe().T
    desc["skew"] = df[cols].astype("float64").skew()
    desc["missing"] = df[cols].isna().sum()
    desc.index.name = "Column"
    return desc.rename(columns={"count": "non-null"}).reset_index()


def categorical_frequency(series: pd.Series, top: int = 10) -> pd.DataFrame:
    """Most frequent values of one column with counts and percentages of all rows."""
    counts = series.astype("object").where(series.notna(), "(missing)").value_counts()
    head = counts.head(top)
    return pd.DataFrame(
        {
            "Value": head.index.astype(str),
            "Count": head.to_numpy(),
            "Percent": [pct(int(c), len(series), 2) for c in head.to_numpy()],
        }
    )


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per categorical column: unique values, most frequent value and its share."""
    rows = []
    for col in categorical_columns(df):
        non_null = df[col].dropna()
        if non_null.empty:
            rows.append({"Column": col, "Unique": 0, "Top value": "(all missing)", "Top count": 0, "Top %": 0.0})
            continue
        counts = non_null.astype(str).value_counts()
        rows.append(
            {
                "Column": col,
                "Unique": int(counts.size),
                "Top value": str(counts.index[0]),
                "Top count": int(counts.iloc[0]),
                "Top %": pct(int(counts.iloc[0]), len(df), 2),
            }
        )
    return pd.DataFrame(rows, columns=["Column", "Unique", "Top value", "Top count", "Top %"])


# ---------------------------------------------------------------------- outliers
def iqr_bounds(series: pd.Series, factor: float = IQR_FACTOR) -> tuple[float, float] | None:
    """Return (lower, upper) IQR fences, or ``None`` when there are fewer than 4 values."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 4:
        return None
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    spread = q3 - q1
    return float(q1 - factor * spread), float(q3 + factor * spread)


def iqr_outlier_mask(series: pd.Series, factor: float = IQR_FACTOR) -> pd.Series:
    """Boolean mask that is True for values outside the IQR fences (missing values are False)."""
    bounds = iqr_bounds(series, factor)
    if bounds is None:
        return pd.Series(False, index=series.index)
    values = pd.to_numeric(series, errors="coerce")
    return (values < bounds[0]) | (values > bounds[1])


def outlier_summary(df: pd.DataFrame, factor: float = IQR_FACTOR) -> pd.DataFrame:
    """Potential outliers per numeric column using the IQR rule. Nothing is removed."""
    rows = []
    for col in numeric_columns(df):
        bounds = iqr_bounds(df[col], factor)
        if bounds is None:
            continue
        count = int(iqr_outlier_mask(df[col], factor).sum())
        rows.append(
            {
                "Column": col,
                "Lower fence": round(bounds[0], 4),
                "Upper fence": round(bounds[1], 4),
                "Outliers": count,
                "Outlier %": pct(count, int(df[col].notna().sum()), 2),
            }
        )
    return pd.DataFrame(rows, columns=["Column", "Lower fence", "Upper fence", "Outliers", "Outlier %"])


# ---------------------------------------------------------------- type problems
def classify_text_column(series: pd.Series) -> str | None:
    """Detect text columns with type problems.

    Returns ``"mixed"`` (numbers and text together), ``"numeric_text"`` (all values look like
    numbers but are stored as text), ``"mostly_numeric_text"`` (>= 90 % numeric-looking) or ``None``.
    """
    if column_kind(series) != "categorical":
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    if series.dtype == object:
        has_number = any(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in non_null)
        has_text = any(isinstance(v, str) for v in non_null)
        if has_number and has_text:
            return "mixed"
    text = non_null.astype(str).str.strip()
    share = float(pd.to_numeric(text, errors="coerce").notna().mean())
    if share == 1.0:
        return "numeric_text"
    if share >= 0.9:
        return "mostly_numeric_text"
    return None


def count_blank_strings(series: pd.Series) -> int:
    """Number of values that are empty or whitespace-only text (not counted as missing by pandas)."""
    if column_kind(series) != "categorical":
        return 0
    non_null = series.dropna()
    return int(sum(isinstance(v, str) and not v.strip() for v in non_null))


# ----------------------------------------------------------------- quality issues
@dataclass(frozen=True)
class Issue:
    severity: str  # high | medium | low
    column: str
    category: str
    message: str


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def quality_issues(df: pd.DataFrame) -> list[Issue]:
    """List potential data-quality problems. These are flags, not automatic fixes."""
    issues: list[Issue] = []
    if df.shape[0] == 0 or df.shape[1] == 0:
        return [Issue("high", "(dataset)", "empty", "The dataset has no rows or no columns.")]

    n = len(df)
    dups = int(df.duplicated().sum())
    if dups:
        issues.append(Issue("medium" if pct(dups, n) >= 1 else "low", "(dataset)", "duplicates",
                            f"{dups} duplicate row(s) ({pct(dups, n, 2)}% of rows)."))

    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        share = pct(missing, n, 1)
        if missing == n:
            issues.append(Issue("high", col, "missing", "Every value is missing."))
            continue
        if share >= 30:
            issues.append(Issue("high", col, "missing", f"{missing} missing values ({share}%)."))
        elif share >= 5:
            issues.append(Issue("medium", col, "missing", f"{missing} missing values ({share}%)."))
        elif missing:
            issues.append(Issue("low", col, "missing", f"{missing} missing values ({share}%)."))

        unique = int(series.nunique(dropna=True))
        if unique <= 1:
            issues.append(Issue("medium", col, "constant", "Constant column: it has a single value and carries no information."))

        text_issue = classify_text_column(series)
        if text_issue == "mixed":
            issues.append(Issue("medium", col, "types", "Mixed types: numbers and text appear in the same column."))
        elif text_issue == "numeric_text":
            issues.append(Issue("medium", col, "types", "All values look numeric but are stored as text; consider converting the type."))
        elif text_issue == "mostly_numeric_text":
            issues.append(Issue("medium", col, "types", "At least 90% of values look numeric but some are not; check for typos."))

        blanks = count_blank_strings(series)
        if blanks:
            issues.append(Issue("medium", col, "blank", f"{blanks} blank text value(s) that pandas does not count as missing."))

        kind = column_kind(series)
        if kind == "categorical" and unique > HIGH_CARDINALITY_THRESHOLD:
            issues.append(Issue("low", col, "cardinality", f"High cardinality: {unique} distinct values."))
        if n >= 20 and unique == int(series.notna().sum()) and kind in ("categorical", "numeric") and not pdt.is_float_dtype(series):
            issues.append(Issue("low", col, "identifier", "Every value is unique; this may be an identifier column."))

        if kind == "numeric":
            outliers = int(iqr_outlier_mask(series).sum())
            if outliers:
                share_out = pct(outliers, int(series.notna().sum()), 1)
                issues.append(Issue("medium" if share_out >= 5 else "low", col, "outliers",
                                    f"{outliers} potential outlier(s) by the IQR rule ({share_out}%)."))

    issues.sort(key=lambda i: (_SEVERITY_ORDER[i.severity], i.column))
    return issues


def issues_to_frame(issues: list[Issue]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Severity": i.severity, "Column": i.column, "Category": i.category, "Issue": i.message} for i in issues],
        columns=["Severity", "Column", "Category", "Issue"],
    )


# ---------------------------------------------------------------- column detail
def column_detail(df: pd.DataFrame, column: str) -> dict:
    """Facts about a single column, computed from the data."""
    if column not in df.columns:
        raise KeyError(f"Column {column!r} not found.")
    series = df[column]
    kind = column_kind(series)
    detail: dict = {
        "column": column,
        "dtype": str(series.dtype),
        "kind": kind,
        "rows": len(series),
        "missing": int(series.isna().sum()),
        "missing_pct": pct(int(series.isna().sum()), len(series), 2),
        "unique": int(series.nunique(dropna=True)),
    }
    if kind == "numeric" and series.notna().any():
        values = series.astype("float64")
        detail["stats"] = {
            "mean": float(values.mean()), "median": float(values.median()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()), "skew": float(values.skew()),
        }
        detail["outliers_iqr"] = int(iqr_outlier_mask(series).sum())
    elif kind in ("categorical", "boolean"):
        detail["top_values"] = categorical_frequency(series, top=10)
    elif kind == "datetime" and series.notna().any():
        detail["range"] = (str(series.min()), str(series.max()))
    return detail


# ------------------------------------------------------------------ correlations
def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix of numeric columns (constant and all-missing columns are dropped)."""
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"Unsupported correlation method: {method}")
    candidates = columns if columns is not None else numeric_columns(df)
    usable = [c for c in candidates if c in df.columns and column_kind(df[c]) == "numeric" and df[c].nunique(dropna=True) > 1]
    if len(usable) < 2:
        return pd.DataFrame()
    return df[usable].astype("float64").corr(method=method)


def top_correlations(df: pd.DataFrame, n: int = 5, method: str = "pearson") -> pd.DataFrame:
    """The ``n`` strongest absolute pairwise correlations between numeric columns."""
    matrix = correlation_matrix(df, method=method)
    if matrix.empty:
        return pd.DataFrame(columns=["Column A", "Column B", "Correlation"])
    pairs = []
    cols = list(matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            value = matrix.loc[a, b]
            if pd.notna(value):
                pairs.append({"Column A": a, "Column B": b, "Correlation": round(float(value), 4)})
    out = pd.DataFrame(pairs, columns=["Column A", "Column B", "Correlation"])
    if out.empty:
        return out
    order = out["Correlation"].abs().sort_values(ascending=False).index
    return out.loc[order].head(n).reset_index(drop=True)


# ---------------------------------------------------------------------- insights
def generate_insights(df: pd.DataFrame) -> list[str]:
    """Plain-language facts computed from the data (no speculation)."""
    if df.empty:
        return ["The dataset is empty, so no insights could be computed."]
    out: list[str] = []
    overview = dataset_overview(df)
    out.append(f"The dataset has {overview['rows']:,} rows and {overview['columns']} columns "
               f"({overview['numeric_columns']} numeric, {overview['categorical_columns']} categorical, "
               f"{overview['datetime_columns']} datetime).")
    if overview["missing_cells"]:
        worst = missing_by_column(df).iloc[0]
        out.append(f"{overview['missing_cells']:,} cells ({overview['missing_pct']}%) are missing; "
                   f"'{worst['Column']}' has the most ({int(worst['Missing'])}, {worst['Missing %']}%).")
    else:
        out.append("No missing values were found.")
    if overview["duplicate_rows"]:
        out.append(f"{overview['duplicate_rows']} duplicate row(s) were found ({overview['duplicate_pct']}%).")
    corr = top_correlations(df, n=1)
    if not corr.empty:
        row = corr.iloc[0]
        out.append(f"The strongest Pearson correlation is between '{row['Column A']}' and '{row['Column B']}' "
                   f"(r = {row['Correlation']:.2f}). Correlation does not imply causation.")
    cats = categorical_summary(df)
    cats = cats[cats["Unique"] > 1]
    if not cats.empty:
        row = cats.sort_values("Unique").iloc[0]
        out.append(f"In '{row['Column']}', the most frequent value is '{row['Top value']}' ({row['Top %']}% of rows).")
    nums = numeric_summary(df)
    if not nums.empty and nums["skew"].notna().any():
        row = nums.loc[nums["skew"].abs().idxmax()]
        out.append(f"'{row['Column']}' has the most skewed distribution (skewness {row['skew']:.2f}).")
    return out
