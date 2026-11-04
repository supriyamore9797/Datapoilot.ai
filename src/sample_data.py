"""Deterministic synthetic sample datasets.

All data is generated with fixed random seeds. No real student information is used.

Intentional defects (so the cleaning tools have something to demonstrate):

* ``student_performance`` - about 4% missing values in ``attendance_pct``,
  ``assignment_score`` and ``sleep_hours`` plus 15 exact duplicate rows.
* ``student_regression``  - about 3% missing values in ``sleep_hours`` and
  ``parental_support``.
* ``sales_data``          - deliberately clean; ``revenue`` always equals
  ``units_sold * unit_price``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SAMPLE_DATA_DIR

SEED = 42


@dataclass(frozen=True)
class SampleInfo:
    """Metadata describing one built-in sample dataset."""

    key: str
    filename: str
    title: str
    description: str
    suggested_task: str
    suggested_target: str
    defects: str


def _inject_missing(
    df: pd.DataFrame, columns: list[str], fraction: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Blank out ``fraction`` of the values in each listed column (in place on a copy)."""
    out = df.copy()
    for col in columns:
        mask = rng.random(len(out)) < fraction
        out[col] = out[col].astype(object).where(~mask, np.nan) if out[col].dtype == object else out[col].mask(mask)
    return out


def make_student_performance(n: int = 600, seed: int = SEED) -> pd.DataFrame:
    """Classification dataset: predict Pass/Fail from study habits."""
    rng = np.random.default_rng(seed)
    study_hours = np.clip(rng.normal(5.0, 2.0, n), 0.5, 12.0)
    attendance = np.clip(rng.normal(78, 12, n), 40, 100)
    assignment = np.clip(rng.normal(70, 15, n) + 1.5 * (study_hours - 5), 0, 100)
    previous = np.clip(rng.normal(65, 15, n) + 2.0 * (study_hours - 5), 0, 100)
    sleep = np.clip(rng.normal(7.0, 1.2, n), 3.0, 10.0)

    score = (
        0.045 * (previous - 65)
        + 0.035 * (assignment - 70)
        + 0.020 * (attendance - 78)
        + 0.30 * (study_hours - 5)
        + 0.15 * (sleep - 7)
    )
    prob_pass = 1.0 / (1.0 + np.exp(-(score + 0.4)))
    passed = rng.random(n) < prob_pass

    df = pd.DataFrame(
        {
            "study_hours": study_hours.round(1),
            "attendance_pct": attendance.round(1),
            "assignment_score": assignment.round(1),
            "previous_score": previous.round(1),
            "sleep_hours": sleep.round(1),
            "result": np.where(passed, "Pass", "Fail"),
        }
    )
    df = _inject_missing(df, ["attendance_pct", "assignment_score", "sleep_hours"], 0.04, rng)
    duplicates = df.sample(15, random_state=seed)
    return pd.concat([df, duplicates], ignore_index=True)


def make_student_regression(n: int = 500, seed: int = SEED + 1) -> pd.DataFrame:
    """Regression dataset: predict a continuous final exam score."""
    rng = np.random.default_rng(seed)
    department = rng.choice(["CSE", "ECE", "MECH", "CIVIL", "IT"], n, p=[0.30, 0.22, 0.18, 0.12, 0.18])
    internet = rng.choice(["Yes", "No"], n, p=[0.82, 0.18])
    extracurricular = rng.choice(["Yes", "No"], n, p=[0.45, 0.55])
    support = rng.choice(["Low", "Medium", "High"], n, p=[0.25, 0.50, 0.25])

    study_hours = np.clip(rng.normal(5.0, 2.0, n), 0.5, 12.0)
    attendance = np.clip(rng.normal(80, 10, n), 45, 100)
    previous = np.clip(rng.normal(66, 14, n), 15, 100)
    sleep = np.clip(rng.normal(7.0, 1.1, n), 4.0, 10.0)

    support_bonus = pd.Series(support).map({"Low": -2.5, "Medium": 0.0, "High": 2.5}).to_numpy()
    dept_bonus = pd.Series(department).map(
        {"CSE": 1.5, "ECE": 0.5, "MECH": -0.5, "CIVIL": -1.0, "IT": 1.0}
    ).to_numpy()
    final = (
        12
        + 0.55 * previous
        + 2.2 * study_hours
        + 0.12 * attendance
        + 0.8 * (sleep - 7)
        + support_bonus
        + dept_bonus
        + np.where(internet == "Yes", 1.2, 0.0)
        + rng.normal(0, 4.0, n)
    )
    df = pd.DataFrame(
        {
            "department": department,
            "study_hours": study_hours.round(1),
            "attendance_pct": attendance.round(1),
            "previous_score": previous.round(1),
            "sleep_hours": sleep.round(1),
            "internet_access": internet,
            "extracurricular": extracurricular,
            "parental_support": support,
            "final_score": np.clip(final, 0, 100).round(1),
        }
    )
    return _inject_missing(df, ["sleep_hours", "parental_support"], 0.03, rng)


def make_sales_data(n: int = 800, seed: int = SEED + 2) -> pd.DataFrame:
    """Sales dataset with a consistent ``revenue = units_sold * unit_price`` relationship."""
    rng = np.random.default_rng(seed)
    prices = {"Laptop": 720.0, "Phone": 480.0, "Tablet": 310.0, "Headphones": 85.0, "Monitor": 220.0, "Keyboard": 45.0}
    products = rng.choice(list(prices), n, p=[0.14, 0.22, 0.14, 0.22, 0.14, 0.14])
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.27, 0.23, 0.28, 0.22])
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 366, n), unit="D")

    base_units = pd.Series(products).map(
        {"Laptop": 4, "Phone": 7, "Tablet": 5, "Headphones": 12, "Monitor": 5, "Keyboard": 14}
    ).to_numpy()
    day_of_year = dates.day_of_year if hasattr(dates, "day_of_year") else dates.dayofyear
    seasonal = 1.0 + 0.25 * np.sin((day_of_year.to_numpy() / 366.0) * 2 * np.pi)
    units = np.maximum(1, rng.poisson(base_units * seasonal)).astype(int)
    unit_price = (pd.Series(products).map(prices).to_numpy() * rng.uniform(0.95, 1.05, n)).round(2)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "product": products,
            "region": region,
            "units_sold": units,
            "unit_price": unit_price,
        }
    )
    df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


SAMPLES: dict[str, SampleInfo] = {
    "student_performance": SampleInfo(
        key="student_performance",
        filename="student_performance.csv",
        title="Student performance (classification)",
        description="615 rows of study habits with a Pass/Fail result.",
        suggested_task="classification",
        suggested_target="result",
        defects="~4% missing values in 3 columns and 15 duplicate rows (intentional, for cleaning demos).",
    ),
    "student_regression": SampleInfo(
        key="student_regression",
        filename="student_regression.csv",
        title="Student performance (regression)",
        description="500 rows with numeric and categorical features and a continuous final_score.",
        suggested_task="regression",
        suggested_target="final_score",
        defects="~3% missing values in sleep_hours and parental_support (intentional).",
    ),
    "sales_data": SampleInfo(
        key="sales_data",
        filename="sales_data.csv",
        title="Sales analysis",
        description="800 transactions across 6 products and 4 regions over 2024.",
        suggested_task="regression",
        suggested_target="revenue",
        defects="None - this dataset is intentionally clean. Note: revenue = units_sold x unit_price.",
    ),
}

_GENERATORS = {
    "student_performance": make_student_performance,
    "student_regression": make_student_regression,
    "sales_data": make_sales_data,
}


def generate_all(directory: Path = SAMPLE_DATA_DIR) -> list[Path]:
    """Write all three sample CSV files into ``directory`` and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key, info in SAMPLES.items():
        path = directory / info.filename
        _GENERATORS[key]().to_csv(path, index=False, encoding="utf-8")
        written.append(path)
    return written


def load_sample(key: str, directory: Path = SAMPLE_DATA_DIR) -> pd.DataFrame:
    """Load a built-in sample dataset, regenerating the CSV if it is missing."""
    if key not in SAMPLES:
        raise KeyError(f"Unknown sample dataset: {key!r}")
    path = directory / SAMPLES[key].filename
    if not path.exists():
        generate_all(directory)
    return pd.read_csv(path)


if __name__ == "__main__":
    for written_path in generate_all():
        print(f"wrote {written_path}")
