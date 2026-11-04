"""Central configuration constants for DataPilot AI."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "DataPilot AI"
APP_TAGLINE = "Intelligent Data Science Workbench"

# --- Authorship & Attribution -----------------------------------------------
AUTHOR_NAME = "Supriya S More"
COLLEGE_NAME = ""  # Update with your college/department if desired
COPYRIGHT_YEAR = "2026"


def get_app_footer_text() -> str:
    """Return the professional attribution string for the application footer."""
    if COLLEGE_NAME.strip():
        return f"{APP_NAME} · Developed by {AUTHOR_NAME} ({COLLEGE_NAME}) · {COPYRIGHT_YEAR}"
    return f"{APP_NAME} · Developed by {AUTHOR_NAME} · {COPYRIGHT_YEAR}"


def get_report_developer_text() -> str:
    """Return the developer credit line for generated PDF reports."""
    if COLLEGE_NAME.strip():
        return f"Developed by {AUTHOR_NAME} ({COLLEGE_NAME}) · {COPYRIGHT_YEAR}"
    return f"Developed by {AUTHOR_NAME} · {COPYRIGHT_YEAR}"


def get_copyright_text() -> str:
    """Return copyright notice."""
    return f"© {COPYRIGHT_YEAR} {AUTHOR_NAME}. All rights reserved."


# --- Paths (always relative to the project root, never hard-coded) -----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Upload limits -----------------------------------------------------------
ALLOWED_EXTENSIONS: tuple[str, ...] = (".csv", ".xlsx")
MAX_FILE_MB = 50
MAX_ROWS = 500_000
MAX_COLUMNS = 1_000
CSV_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

# --- Display limits ----------------------------------------------------------
PREVIEW_ROWS = 100
MAX_PLOT_POINTS = 20_000
HIGH_CARDINALITY_THRESHOLD = 50
MAX_BAR_CATEGORIES = 30

# --- Cleaning ----------------------------------------------------------------
MAX_UNDO_STEPS = 5
IQR_FACTOR = 1.5

# --- Machine learning --------------------------------------------------------
RANDOM_STATE = 42
MIN_ML_ROWS = 30
MAX_CLASSES = 30
MAX_OHE_CARDINALITY = 50
DEFAULT_TEST_SIZE = 0.2
LEAKAGE_CORRELATION = 0.98
IMBALANCE_MINORITY_SHARE = 0.20
LARGE_TRAINING_ROWS = 50_000
