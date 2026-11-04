"""Safe loading and validation of uploaded CSV / Excel files.

Uploaded bytes are only ever parsed as data by pandas. Nothing in an uploaded file is
executed, evaluated or written to disk.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import PurePath

import pandas as pd
from openpyxl import load_workbook

from src.config import (
    ALLOWED_EXTENSIONS,
    CSV_ENCODINGS,
    MAX_COLUMNS,
    MAX_FILE_MB,
    MAX_ROWS,
)


class DataLoadError(Exception):
    """Raised with a user-friendly message when a file cannot be loaded."""


@dataclass
class LoadResult:
    """A successfully loaded dataset plus any non-fatal notes about it."""

    df: pd.DataFrame
    filename: str
    warnings: list[str] = field(default_factory=list)


def normalize_column_names(raw_names: list[object]) -> tuple[list[str], list[str]]:
    """Make column names non-blank, stripped and unique.

    Rules (predictable and reported to the user):

    * leading/trailing whitespace is stripped;
    * blank or missing names become ``unnamed_<position>`` (1-based);
    * a repeated name keeps its first occurrence and later ones get ``_2``, ``_3``...

    Returns the new names and a list of human-readable change notes.
    """
    notes: list[str] = []
    result: list[str] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_names, start=1):
        is_blank = raw is None or (isinstance(raw, float) and pd.isna(raw))
        text = "" if is_blank else str(raw).strip()
        if not text:
            text = f"unnamed_{position}"
            notes.append(f"Column {position} had no name and was renamed to '{text}'.")
        elif str(raw) != text:
            notes.append(f"Whitespace was trimmed from column name '{raw}'.")
        candidate, counter = text, 1
        while candidate in seen:
            counter += 1
            candidate = f"{text}_{counter}"
        if candidate != text:
            notes.append(f"Duplicate column name '{text}' was renamed to '{candidate}'.")
        seen.add(candidate)
        result.append(candidate)
    return result, notes


def _validate_upload(data: bytes, filename: str, max_mb: float) -> str:
    """Check extension and size. Returns the lower-cased extension."""
    if not filename:
        raise DataLoadError("The uploaded file has no name, so its type cannot be determined.")
    extension = PurePath(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        shown = extension or "no extension"
        raise DataLoadError(f"Unsupported file type ({shown}). Please upload one of: {allowed}.")
    if len(data) == 0:
        raise DataLoadError("The uploaded file is empty (0 bytes).")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise DataLoadError(f"The file is {size_mb:.1f} MB, which is above the {max_mb:g} MB limit.")
    return extension


def _decode_csv(data: bytes) -> tuple[str, str]:
    """Decode CSV bytes trying common encodings. Returns (text, encoding used)."""
    if b"\x00" in data[:4096]:
        raise DataLoadError(
            "This file contains binary data and does not look like a text CSV file. "
            "If it is an Excel file, save it as .xlsx and upload that instead."
        )
    for encoding in CSV_ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise DataLoadError("The file's text encoding could not be determined. Save it as UTF-8 CSV and retry.")


def _read_csv(data: bytes, warnings: list[str]) -> pd.DataFrame:
    text, encoding = _decode_csv(data)
    if encoding not in ("utf-8-sig", "utf-8"):
        warnings.append(f"The file was not valid UTF-8; it was read as {encoding}. Check special characters.")
    if not text.strip():
        raise DataLoadError("The CSV file contains no data (only whitespace).")

    text = text.lstrip("\ufeff")
    separator = ","
    first_line = text.splitlines()[0]
    for candidate in (";", "\t", "|"):
        if first_line.count(candidate) > first_line.count(separator):
            separator = candidate
    if separator != ",":
        warnings.append(f"Detected {separator!r} as the column separator instead of a comma.")

    try:
        df = pd.read_csv(io.StringIO(text), sep=separator)
        raw_header = next(csv.reader(io.StringIO(text), delimiter=separator), [])
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The CSV file contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(
            f"The CSV file is malformed and could not be parsed: {str(exc).splitlines()[0]}"
        ) from exc
    except (ValueError, csv.Error) as exc:
        raise DataLoadError(f"The CSV file could not be read: {str(exc).splitlines()[0]}") from exc

    # If data rows are wider than the header, pandas silently shifts leading columns into the index
    # (and newer pandas versions disguise that index as a RangeIndex), so count the raw fields instead.
    widest = max((len(row) for row in csv.reader(io.StringIO(text), delimiter=separator)), default=0)
    if widest > len(raw_header) or len(raw_header) != df.shape[1]:
        raise DataLoadError(
            "The CSV rows do not all have the same number of fields as the header row. "
            "Fix the file (for example an unquoted comma inside a value) and upload it again."
        )
    names, notes = normalize_column_names(raw_header)
    warnings.extend(notes)
    df.columns = names
    return df


def _excel_header_row(data: bytes) -> list[object]:
    """Return the raw first-row values of the first worksheet (needed to detect blank/duplicate names)."""
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        sheet = workbook.worksheets[0]
        first = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
        return list(first)
    finally:
        workbook.close()


def _read_excel(data: bytes, warnings: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(data), engine="openpyxl", header=0)
        raw_header = _excel_header_row(data)
    except Exception as exc:  # openpyxl raises many unrelated error types for corrupt files
        raise DataLoadError(
            "The Excel file could not be read. It may be corrupted, password-protected or not a real .xlsx file."
        ) from exc
    if df.shape[1] == 0:
        raise DataLoadError("The Excel sheet contains no data.")
    width = df.shape[1]
    raw_header = (raw_header + [None] * width)[:width]
    names, notes = normalize_column_names(raw_header)
    warnings.extend(notes)
    df.columns = names
    warnings.append("Only the first worksheet of the Excel file was loaded.")
    return df


def load_dataframe(
    data: bytes,
    filename: str,
    max_mb: float = MAX_FILE_MB,
    max_rows: int = MAX_ROWS,
    max_columns: int = MAX_COLUMNS,
) -> LoadResult:
    """Parse uploaded bytes into a DataFrame, validating size, shape and column names.

    Raises :class:`DataLoadError` with a readable message for any problem.
    The dataset is never silently truncated: files above the row limit are rejected.
    """
    extension = _validate_upload(data, filename, max_mb)
    warnings: list[str] = []
    df = _read_csv(data, warnings) if extension == ".csv" else _read_excel(data, warnings)

    if df.shape[1] == 0:
        raise DataLoadError("The file has no columns.")
    if df.shape[0] == 0:
        raise DataLoadError("The file has column headers but no data rows.")
    if df.shape[1] > max_columns:
        raise DataLoadError(f"The file has {df.shape[1]:,} columns, above the limit of {max_columns:,}.")
    if df.shape[0] > max_rows:
        raise DataLoadError(
            f"The file has {df.shape[0]:,} rows, above the limit of {max_rows:,}. "
            "It was not loaded (DataPilot never truncates data silently); please upload a smaller file."
        )
    if df.isna().all().all():
        raise DataLoadError("Every cell in the file is empty.")

    return LoadResult(df=df, filename=PurePath(filename).name, warnings=warnings)
