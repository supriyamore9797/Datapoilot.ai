"""College Student Data Filtering and Analysis.

Pure business logic for detecting college/student attributes, multi-criteria filtering,
search, distribution summaries, and demographic charting.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from src.cleaning import to_csv_bytes, to_xlsx_bytes
from src.profiling import column_kind

MISSING_VALUE_LABEL = "(Missing / Unspecified)"

# Recognizers for common college student attributes
ATTRIBUTE_PATTERNS: dict[str, list[str]] = {
    "District": [r"\bdist(rict)?\b", r"\bjilha\b"],
    "City": [r"\bcity\b", r"\btown\b", r"\bnative\b", r"\bplace\b", r"\blocation\b", r"\btaluka\b"],
    "State": [r"\bstate\b", r"\bprovince\b", r"\bregion\b"],
    "Course": [r"\bcourse\b", r"\bprogram\b", r"\bdegree\b"],
    "Department": [r"\bdept\b", r"\bdepartment\b", r"\bbranch\b", r"\bstream\b", r"\bspecialization\b", r"\bdiscipline\b"],
    "Academic Year": [r"\byear\b", r"\bacademic[_\s]*year\b", r"\bclass\b", r"\bsem(ester)?\b", r"\bbatch\b"],
    "Gender": [r"\bgender\b", r"\bsex\b"],
    "Category": [r"\bcat(egory)?\b", r"\bcaste\b", r"\bquota\b", r"\breservation\b"],
    "Admission Status": [r"\bstatus\b", r"\badmission[_\s]*status\b", r"\badmission[_\s]*type\b", r"\bresult\b"],
}

SEARCH_PATTERNS: list[str] = [
    r"\bname\b", r"\bstudent[_\s]*name\b", r"\bfirst[_\s]*name\b", r"\blast[_\s]*name\b",
    r"\bid\b", r"\bstudent[_\s]*id\b", r"\broll\b", r"\broll[_\s]*no\b", r"\bprn\b",
    r"\breg(istration)?[_\s]*no\b", r"\benrollment\b", r"\bemail\b", r"\bmobile\b", r"\bphone\b",
]


def detect_student_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Inspect DataFrame columns and return classified candidate filter and search fields.

    Returns:
        A dictionary with:
        - 'recommended_filters': Columns matching known student/college demographic attributes.
        - 'searchable_columns': Columns suitable for text search (names, IDs, roll numbers).
        - 'all_filter_candidates': All columns suitable for filtering (cardinality <= 60).
    """
    if df is None or df.empty:
        return {"recommended_filters": [], "searchable_columns": [], "all_filter_candidates": []}

    recommended: list[str] = []
    searchable: list[str] = []
    all_candidates: list[str] = []

    for col in df.columns:
        col_str = str(col).strip().lower()
        series = df[col]
        nunique = series.nunique(dropna=True)

        # Check search candidates (names, IDs, text)
        is_search = any(bool(re.search(pat, col_str)) for pat in SEARCH_PATTERNS)
        if is_search or column_kind(series) == "categorical":
            searchable.append(col)

        # Check filter candidates (cardinality between 1 and 60, or object/categorical)
        is_attr = any(bool(re.search(pat, col_str)) for patterns in ATTRIBUTE_PATTERNS.values() for pat in patterns)
        if (is_attr and nunique <= 120) or (nunique > 0 and nunique <= 60 and column_kind(series) in ("categorical", "boolean", "numeric")):
            all_candidates.append(col)

        if is_attr and nunique <= 120:
            recommended.append(col)

    # If no specific patterns matched, fall back to all reasonable categorical columns
    if not recommended:
        recommended = [c for c in df.columns if 1 < df[c].nunique(dropna=True) <= 30][:5]

    # Ensure recommended columns come first in all_candidates without duplicates
    ordered_candidates: list[str] = []
    for c in recommended + all_candidates + list(df.columns):
        if c in df.columns and c not in ordered_candidates:
            if df[c].nunique(dropna=True) <= 100:
                ordered_candidates.append(c)

    # Ensure searchable list has at least some columns
    if not searchable:
        searchable = [c for c in df.columns if column_kind(df[c]) == "categorical"]

    return {
        "recommended_filters": recommended,
        "searchable_columns": searchable,
        "all_filter_candidates": ordered_candidates,
    }


def get_unique_filter_values(series: pd.Series) -> list[str]:
    """Return sorted unique string values for a column, with missing values represented at the end."""
    non_null = series.dropna()
    # Sort logically
    unique_vals = sorted({str(v).strip() for v in non_null if str(v).strip() != ""})
    if series.isna().any():
        unique_vals.append(MISSING_VALUE_LABEL)
    return unique_vals


def filter_students(
    df: pd.DataFrame,
    filters: dict[str, list[str]] | None = None,
    search_query: str = "",
    search_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter records by multiple criteria (AND logic across columns) and free-text search.

    Args:
        df: Input student DataFrame.
        filters: Mapping of {column_name: [list of allowed values]}.
        search_query: Substring to search for across search_columns.
        search_columns: Specific columns to search in (or all object columns if None).

    Returns:
        Filtered DataFrame copy. Does NOT modify the original input DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    mask = pd.Series(True, index=df.index)

    # 1. Apply multi-column categorical filters (AND logic across columns)
    if filters:
        for col, selected in filters.items():
            if col not in df.columns or not selected:
                continue

            series = df[col]
            include_missing = MISSING_VALUE_LABEL in selected
            clean_selected = [v for v in selected if v != MISSING_VALUE_LABEL]

            # Compare as strings to prevent int/str type mismatches
            series_str = series.map(lambda v: "" if pd.isna(v) else str(v).strip())
            col_mask = series_str.isin(clean_selected)

            if include_missing:
                col_mask |= series.isna()

            mask &= col_mask

    # 2. Apply free-text search (OR logic across selected search columns)
    query = str(search_query).strip().lower()
    if query:
        search_cols = [c for c in (search_columns or df.columns) if c in df.columns]
        if search_cols:
            search_mask = pd.Series(False, index=df.index)
            for col in search_cols:
                col_text = df[col].astype(str).str.lower()
                search_mask |= col_text.str.contains(query, regex=False, na=False)
            mask &= search_mask

    return df.loc[mask].reset_index(drop=True)


def summarize_filter_results(total: int, filtered: int) -> dict[str, Any]:
    """Compute headline counts and percentages."""
    pct = round(100.0 * filtered / total, 1) if total > 0 else 0.0
    return {
        "total_records": int(total),
        "matching_records": int(filtered),
        "excluded_records": int(max(0, total - filtered)),
        "match_percentage": float(pct),
    }


def compute_column_breakdown(df: pd.DataFrame, column: str, top_n: int = 15) -> pd.DataFrame:
    """Compute frequency counts and percentages for a single column."""
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame(columns=[column, "Count", "Share %"])

    counts = df[column].astype(str).replace("nan", MISSING_VALUE_LABEL).value_counts()
    top = counts.head(top_n)
    total = len(df)
    return pd.DataFrame({
        column: top.index,
        "Count": top.to_numpy(),
        "Share %": (100.0 * top / total).round(1).to_numpy(),
    })


def generate_student_insights(df: pd.DataFrame, key_columns: list[str] | None = None) -> list[str]:
    """Generate concise, readable takeaways for college administrators."""
    if df is None or df.empty:
        return ["No records match the active filters to generate insights."]

    insights: list[str] = [f"Total matching cohort: {len(df):,} student record(s)."]
    cols_to_check = key_columns or list(df.columns)

    for col in cols_to_check:
        if col not in df.columns:
            continue
        col_lower = str(col).lower()
        counts = df[col].dropna().astype(str).value_counts()
        if counts.empty:
            continue

        top_val = counts.index[0]
        top_count = int(counts.iloc[0])
        top_pct = round(100.0 * top_count / len(df), 1)

        if any(re.search(pat, col_lower) for pat in ATTRIBUTE_PATTERNS["District"]):
            insights.append(
                f"District representation: {top_val} is the top district with {top_count} students ({top_pct}% of filtered cohort)."
            )
        elif any(re.search(pat, col_lower) for pat in ATTRIBUTE_PATTERNS["Department"]):
            insights.append(
                f"Department breakdown: {top_val} has the highest enrollment with {top_count} students ({top_pct}%)."
            )
        elif any(re.search(pat, col_lower) for pat in ATTRIBUTE_PATTERNS["Academic Year"]):
            insights.append(
                f"Academic Year: {top_val} accounts for {top_count} students ({top_pct}% of cohort)."
            )
        elif any(re.search(pat, col_lower) for pat in ATTRIBUTE_PATTERNS["Gender"]):
            insights.append(
                f"Gender distribution: {top_val} represents {top_count} students ({top_pct}%)."
            )

    return insights[:5]


def build_breakdown_chart(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
    chart_type: str = "bar",
    top_n: int = 15,
) -> Any:
    """Build a Plotly figure representing student counts across values of column."""
    if df is None or df.empty or column not in df.columns:
        return None

    import plotly.express as px
    import plotly.graph_objects as go

    summary_df = compute_column_breakdown(df, column, top_n=top_n)
    if summary_df.empty:
        return None

    chart_title = title or f"Student Distribution by {column}"

    if chart_type == "pie":
        fig = px.pie(
            summary_df,
            names=column,
            values="Count",
            title=chart_title,
            hole=0.4,
            color_discrete_sequence=["#4F46E5", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"],
        )
    else:
        fig = px.bar(
            summary_df,
            x=column,
            y="Count",
            text="Count",
            title=chart_title,
            color_discrete_sequence=["#4F46E5"],
        )
        fig.update_traces(textposition="outside")

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=column,
        yaxis_title="Student Count",
        font=dict(color="#1F2937", size=12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def make_college_roster_sample(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic, deterministic sample dataset of Maharashtra college engineering students."""
    rng = np.random.default_rng(seed)

    first_names = [
        "Aarav", "Sneha", "Rohan", "Ananya", "Aditya", "Pooja", "Siddharth", "Priyanka",
        "Tanmay", "Neha", "Omkar", "Pratiksha", "Kunal", "Sayali", "Abhishek", "Shruti",
        "Vedant", "Mrunal", "Akash", "Gauri", "Shubham", "Tejaswini", "Nikhil", "Divya",
        "Yash", "Ishita", "Saurabh", "Sakshi", "Pranav", "Ruturaj"
    ]
    last_names = [
        "Patil", "Deshmukh", "Kulkarni", "Joshi", "Shinde", "Pawar", "Chavan", "Kadam",
        "More", "Gaikwad", "Jadhav", "Bhosale", "Thorat", "Salunkhe", "Wagh", "Mane",
        "Tambe", "Ghuge", "Suryavanshi", "Sawant", "Mohite", "Bhandari", "Shelar"
    ]

    districts = ["Pune", "Nashik", "Satara", "Mumbai", "Kolhapur", "Ahmednagar", "Nagpur", "Solapur"]
    dist_p = [0.38, 0.16, 0.12, 0.10, 0.09, 0.07, 0.05, 0.03]

    departments = [
        "AIML",
        "Computer Engineering",
        "Information Technology",
        "Mechanical Engineering",
        "Civil Engineering",
        "Electronics & Telecommunication",
    ]
    dept_p = [0.24, 0.26, 0.18, 0.12, 0.08, 0.12]

    years = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
    genders = ["Female", "Male"]
    categories = ["Open", "OBC", "EWS", "SC", "ST", "NT"]
    cat_p = [0.40, 0.25, 0.15, 0.10, 0.04, 0.06]
    statuses = ["Confirmed", "Provisional", "Management Quota"]
    status_p = [0.82, 0.12, 0.06]

    names = [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n)]
    roll_numbers = [f"PRN2024{i:03d}" for i in range(1, n + 1)]
    student_districts = rng.choice(districts, n, p=dist_p).tolist()
    student_depts = rng.choice(departments, n, p=dept_p).tolist()
    student_years = rng.choice(years, n, p=[0.28, 0.26, 0.24, 0.22]).tolist()
    student_genders = rng.choice(genders, n, p=[0.44, 0.56]).tolist()
    student_categories = rng.choice(categories, n, p=cat_p).tolist()
    student_statuses = rng.choice(statuses, n, p=status_p).tolist()

    cgpa = np.clip(rng.normal(7.8, 0.9, n), 5.5, 9.9).round(2)
    attendance = np.clip(rng.normal(82, 8, n), 60, 99).astype(int)

    # City is predominantly same as district with realistic towns
    cities = []
    for d in student_districts:
        if d == "Pune":
            cities.append(rng.choice(["Pune", "Pimpri-Chinchwad", "Baramati", "Khadki"], p=[0.65, 0.20, 0.10, 0.05]))
        elif d == "Nashik":
            cities.append(rng.choice(["Nashik", "Malegaon", "Sinnar"], p=[0.70, 0.20, 0.10]))
        elif d == "Satara":
            cities.append(rng.choice(["Satara", "Karad", "Phaltan"], p=[0.60, 0.30, 0.10]))
        elif d == "Kolhapur":
            cities.append(rng.choice(["Kolhapur", "Ichalkaranji"], p=[0.75, 0.25]))
        else:
            cities.append(d)

    df = pd.DataFrame({
        "roll_no": roll_numbers,
        "student_name": names,
        "district": student_districts,
        "city": cities,
        "department": student_depts,
        "academic_year": student_years,
        "gender": student_genders,
        "category": student_categories,
        "admission_status": student_statuses,
        "cgpa": cgpa,
        "attendance_pct": attendance,
    })

    # Inject a few intentional missing values in district & city (~2%) to test missing value filters
    miss_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[miss_idx, "district"] = np.nan
    df.loc[miss_idx, "city"] = np.nan

    return df

