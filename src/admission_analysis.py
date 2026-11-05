"""Admission Interest Analysis module for DataPilot AI.

Analyzes admission calling sheets (CSV or Excel) to classify student interest,
track follow-ups, detect calling reachability, and support manual category overrides.
"""

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd

CATEGORIES = [
    "Interested",
    "Not Interested",
    "Follow-up / Undecided",
    "Unreachable",
    "Other / Needs Review",
    "No Remark",
]

CATEGORY_COLORS = {
    "Interested": "#10B981",            # Emerald green
    "Not Interested": "#EF4444",        # Rose red
    "Follow-up / Undecided": "#F59E0B", # Amber
    "Unreachable": "#64748B",           # Slate
    "Other / Needs Review": "#6366F1",  # Indigo
    "No Remark": "#9CA3AF",             # Muted grey
}

# Compiled regex patterns for normalized text classification
_FOLLOWUP_PATTERNS = [
    r"call\s*later|call\s*back|\bcallback\b|ring\s*later|call\s*after|call\s*tomorrow|call\s*again",
    r"think\s*about\s*it|\bthinking\b|needs?\s*time|give\s*time",
    r"not\s*sure|\bmaybe\b|\blater\b",
    r"interested\s*but",
    r"\b(?:course|stream|branch)?\s*not\s*(?:yet\s*)?decid(?:ed|e)?\b|\bnot\s+decid(?:ed|e)?\b|\byet\s+not\s+decid(?:ed|e)?\b|\bundecided\b",
    r"waiting\s*for|wait\s*for",
    r"after\s*(?:cet|mhcet|result|exam|cap|round|neet)",
    r"will\s*decide|\bdeside\b",
    r"will\s*inform|will\s*tell|will\s*let\s*(?:you\s*)?know",
    r"\bdiscussing\b|\bconfused\b|\bconsidering\b",
    r"\bconfirming\b|will\s*confirm",
    r"follow\s*up",
]

_UNREACHABLE_PATTERNS = [
    r"not\s*received",
    r"not\s*reachable|\bunreachable\b",
    r"switch(?:ed)?\s*off",
    r"no\s*answer|not\s*answer(?:ing|ed)?",
    r"not\s*connected",
    r"not\s*pick(?:ing|ed)?|didn\'?t\s*pick|did\s*not\s*pick",
    r"wrong\s*num(?:ber)?|wrng\s*no",
    r"cut\s*the\s*call|call\s*cut|\bdisconnected\b",
    r"not\s*respond(?:ing)?|no\s*response|not\s*responding\s*phone|not\s*respond\s*phone",
    r"out\s*of\s*coverage|network\s*issue",
    r"call\s*busy|line\s*busy|\bbusy\b|\bengage\b",
    r"incoming\s*(?:call\s*)?not\s*available|incoming\s*barred",
    r"(?:number\s*)?(?:does\s*not|doesn\'t)\s*exist|invalid\s*number",
    r"ringing\s*only|not\s*talking",
]

_NOT_INTERESTED_PATTERNS = [
    r"not\s*interested",
    r"no\s*interest",
    r"not\s*willing",
    r"will\s*not\s*take\s*admission|won\'?t\s*take\s*admission",
    r"not\s*joining",
    r"not\s*want",
    r"already\s*(?:taken\s*)?admission|taken\s*admission",
    r"already\s*admitted|admitted\s*in|admitted\s*at",
    r"admission\s*done|done\s*in\s+[a-z]+",
    r"taken\s*at\s+[a-z]+|taken\s*somewhere",
    r"(?:doing|joined)\s+(?:diploma|iti|degree|engineering|bsc|bpharm|nursing)",
    r"taking\s*drop|repeat\s*neet|prepar(?:ing|ation)\s*for\s*neet|neet\s*prep",
    r"failed\s*in\s*12|fail\s*in\s*12",
    r"opted\s*out|\bdeclined\b|\brejected\b",
    r"not\s*planning",
    r"cancel\s*admission",
    r"join\s*police",
]

_INTERESTED_PATTERNS = [
    r"\binterested\b",
    r"wants?\s*admission|want\s*admission",
    r"will\s*take\s*admission",
    r"will\s*visit|visiting|visit\s*campus",
    r"willing\s*to\s*take\s*admission",
    r"looking\s*for",
    r"taking\s*admission",
]

_FOLLOWUP_RE = re.compile("|".join(_FOLLOWUP_PATTERNS), re.IGNORECASE)
_UNREACHABLE_RE = re.compile("|".join(_UNREACHABLE_PATTERNS), re.IGNORECASE)
_NOT_INTERESTED_RE = re.compile("|".join(_NOT_INTERESTED_PATTERNS), re.IGNORECASE)
_INTERESTED_RE = re.compile("|".join(_INTERESTED_PATTERNS), re.IGNORECASE)


def normalize_remark(remark: Any) -> str:
    """Normalize calling remarks: lowercase, trim, strip punctuation, and fix common spelling variants."""
    if remark is None or pd.isna(remark):
        return ""
    text = str(remark).strip()
    if not text or text.lower() in {"nan", "none", "null", "-", "nil", "na", "n/a", "."}:
        return ""

    # 1. Lowercase
    t = text.lower()

    # 2. Replace punctuation with spaces
    t = re.sub(r"[^\w\s]", " ", t)

    # 3. Collapse multiple whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # 4. Correct common spelling and phonetic variations
    replacements = [
        (r"\bint[er]{1,3}e?[st]{1,3}ed\b", "interested"),  # intrested, interesed, intersted, interessted
        (r"\brec[ei]{1,2}ved?\b", "received"),             # recieved, recived, recvd
        (r"\br[ea]{1,2}c?h?able\b", "reachable"),          # rechable, reachble, recable
        (r"\bswich\b", "switch"),
        (r"\bdeside\b", "decide"),
        (r"\bde[sc]i[scd]ed\b", "decided"),
        (r"\bbusey\b", "busy"),
        (r"\badmi[s]{1,2}ion\b", "admission"),
    ]
    for pattern, repl in replacements:
        t = re.sub(pattern, repl, t)

    return t


def classify_remark(remark: Any, google_form_filled: Any = None) -> str:
    """Classify a student's calling remark into one of the standardized Interest_Status categories.

    Ordered rule execution:
    1. Blank or missing remarks always classify as 'No Remark' (Google Form filled = Yes
       is never treated as proof of interest).
    2. Conflicting / deferred remarks (e.g. 'not interested now, call later', 'interested but needs time')
       are classified as 'Follow-up / Undecided'.
    3. Unreachable / No Response ('not received', 'not reachable', 'switched off', etc.) are classified
       as 'Unreachable'. Specifically, 'not received' becomes 'Unreachable', never 'Not Interested'.
    4. Negative-interest rules are evaluated strictly BEFORE positive-interest rules ('not interested',
       'no interest', 'not willing', 'already taken admission', etc.).
    5. Explicit positive phrases ('interested', 'intrested', 'wants admission', 'will take admission',
       'will visit') are classified as 'Interested'.
    6. Remarks that do not clearly fit a category fall into 'Other / Needs Review'.
    """
    t = normalize_remark(remark)
    if not t:
        return "No Remark"

    # Completed callback indicating refusal (e.g. 'not interested after call back') is Not Interested
    if re.search(r"not\s*interested\s*(?:after|on|when)\s*call", t):
        return "Not Interested"

    # Step 1: Conflicting or follow-up deferred intent (checked first)
    if _FOLLOWUP_RE.search(t):
        return "Follow-up / Undecided"

    # Step 2: Unreachable / No Response (e.g., 'not received' -> Unreachable)
    if _UNREACHABLE_RE.search(t):
        return "Unreachable"

    # Step 3: Negative Interest (checked before positive interest)
    if _NOT_INTERESTED_RE.search(t):
        return "Not Interested"

    # Step 4: Explicit positive interest
    if _INTERESTED_RE.search(t):
        return "Interested"

    # Step 5: Fallback for ambiguous remarks
    return "Other / Needs Review"


def detect_admission_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Detect calling sheet columns (Student Name, Faculty, Location, Remarks, Form Status).

    Returns a dictionary of standard role -> detected column name (or None).
    """
    detected: dict[str, str | None] = {
        "student_name": None,
        "faculty": None,
        "location": None,
        "google_form": None,
        "remarks": None,
        "contact": None,
    }

    cols = list(df.columns)
    lowered = {c: str(c).strip().lower() for c in cols}

    # Remarks
    for c, low in lowered.items():
        if any(k in low for k in ["remark", "comment", "calling status", "call feedback", "feedback", "response"]):
            detected["remarks"] = c
            break

    # Student Name
    for c, low in lowered.items():
        if any(k in low for k in ["student name", "name of the student", "candidate name", "applicant name", "student_name"]):
            detected["student_name"] = c
            break
    if not detected["student_name"]:
        for c, low in lowered.items():
            if "student" in low and "contact" not in low and "number" not in low:
                detected["student_name"] = c
                break

    # Faculty
    for c, low in lowered.items():
        if any(k in low for k in ["faculty", "caller", "counselor", "staff", "assigned", "mentor", "agent"]):
            detected["faculty"] = c
            break

    # Location / Address
    for c, low in lowered.items():
        if any(k in low for k in ["address", "location", "city", "district", "place", "town", "native", "region"]):
            detected["location"] = c
            break

    # Google Form
    for c, low in lowered.items():
        if any(k in low for k in ["google form", "form filled", "gform", "form status", "form_filled"]):
            detected["google_form"] = c
            break

    # Contact
    for c, low in lowered.items():
        if any(k in low for k in ["contact", "mobile", "phone", "cell", "student contact"]):
            detected["contact"] = c
            break

    return detected


def classify_dataframe(
    df: pd.DataFrame,
    remark_col: str,
    form_col: str | None = None,
    overrides: dict[Any, str] | None = None,
) -> pd.DataFrame:
    """Return a copy of the dataframe with added Interest_Status classification columns.

    Preserves all original remarks and columns intact.
    """
    out = df.copy()
    overrides = overrides or {}

    def _get_cat(row: pd.Series) -> tuple[str, bool]:
        idx = row.name
        if idx in overrides and overrides[idx] in CATEGORIES:
            return overrides[idx], True
        form_val = row[form_col] if form_col and form_col in row else None
        cat = classify_remark(row[remark_col], google_form_filled=form_val)
        return cat, False

    results = out.apply(_get_cat, axis=1)
    status_list = [r[0] for r in results]
    override_list = [r[1] for r in results]

    # Add normalized Interest_Status as the primary category column
    out["Interest_Status"] = status_list
    out["Category"] = status_list  # Maintained for backwards compatibility
    out["Is Override"] = override_list
    out["Classification Type"] = out["Is Override"].map(lambda o: "Manual Correction" if o else "Automated")
    return out


def compute_admission_kpis(df: pd.DataFrame, category_col: str = "Interest_Status") -> dict[str, Any]:
    """Compute counts and percentages for all Interest_Status categories."""
    total = len(df)
    counts = df[category_col].value_counts().to_dict() if not df.empty and category_col in df.columns else {}

    kpis: dict[str, Any] = {
        "total_records": total,
        "counts": {cat: counts.get(cat, 0) for cat in CATEGORIES},
        "percentages": {
            cat: round((counts.get(cat, 0) / total * 100), 1) if total > 0 else 0.0
            for cat in CATEGORIES
        },
    }
    # Grouped KPIs for dashboard cards
    kpis["interested"] = kpis["counts"]["Interested"]
    kpis["not_interested"] = kpis["counts"]["Not Interested"]
    kpis["undecided"] = kpis["counts"]["Follow-up / Undecided"]
    kpis["unreachable"] = kpis["counts"]["Unreachable"]
    kpis["unknown"] = kpis["counts"]["Other / Needs Review"] + kpis["counts"]["No Remark"]
    kpis["manual_overrides_count"] = int(df["Is Override"].sum()) if "Is Override" in df.columns else 0
    return kpis


def filter_admission_dataframe(
    df: pd.DataFrame,
    faculty_col: str | None = None,
    selected_faculties: list[str] | None = None,
    location_col: str | None = None,
    selected_locations: list[str] | None = None,
    category_col: str = "Interest_Status",
    selected_categories: list[str] | None = None,
    search_query: str = "",
    search_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Filter records by faculty, location, exact Interest_Status, and student search text.

    IMPORTANT: Category filtering is done by exact comparison against Interest_Status,
    NEVER by substring matching on the raw remarks.
    """
    filtered = df.copy()

    # Exact Interest_Status match
    if category_col in filtered.columns and selected_categories is not None:
        filtered = filtered[filtered[category_col].isin(selected_categories)]

    # Location match: normalize by trimming whitespace and comparing case-insensitively
    if location_col and location_col in filtered.columns and selected_locations:
        norm_selected_locations = {str(loc).strip().lower() for loc in selected_locations if str(loc).strip()}
        if norm_selected_locations:
            filtered = filtered[filtered[location_col].astype(str).str.strip().str.lower().isin(norm_selected_locations)]

    # Faculty match: normalize by trimming whitespace and comparing case-insensitively
    if faculty_col and faculty_col in filtered.columns and selected_faculties:
        norm_selected_faculties = {str(f).strip().lower() for f in selected_faculties if str(f).strip()}
        if norm_selected_faculties:
            filtered = filtered[filtered[faculty_col].astype(str).str.strip().str.lower().isin(norm_selected_faculties)]

    # Targeted search by student name or contact number (not substring matching on raw remarks)
    if search_query and search_cols:
        q = search_query.strip().lower()
        if q:
            match_mask = pd.Series(False, index=filtered.index)
            for col in search_cols:
                if col in filtered.columns:
                    match_mask |= filtered[col].astype(str).str.lower().str.contains(q, na=False)
            filtered = filtered[match_mask]

    return filtered


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export dataframe as UTF-8 CSV with BOM for seamless Excel compatibility."""
    return df.to_csv(index=False).encode("utf-8-sig")


def to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Admission Analysis") -> bytes:
    """Export dataframe as an Excel workbook (.xlsx)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buf.getvalue()


def build_admission_chart(counts: dict[str, int]) -> Any:
    """Create a clean, color-coded Plotly bar chart showing Interest_Status distribution."""
    import plotly.express as px

    items = [
        {"Category": cat, "Students": counts.get(cat, 0)}
        for cat in CATEGORIES
    ]
    chart_df = pd.DataFrame(items)
    if chart_df["Students"].sum() == 0:
        return None

    fig = px.bar(
        chart_df,
        x="Students",
        y="Category",
        orientation="h",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        text="Students",
        title="Admission Calling Response Distribution",
    )
    fig.update_layout(
        xaxis_title="Student Count",
        yaxis_title="",
        showlegend=False,
        height=320,
        margin=dict(l=10, r=30, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(textposition="outside")
    return fig
