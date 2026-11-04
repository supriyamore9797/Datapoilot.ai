from io import BytesIO

import pandas as pd
import pytest

from src.data_loader import DataLoadError, load_dataframe, normalize_column_names


def _xlsx(df: pd.DataFrame, header: bool = True) -> bytes:
    buffer = BytesIO()
    df.to_excel(buffer, index=False, header=header, engine="openpyxl")
    return buffer.getvalue()


def test_valid_csv_loads_with_correct_shape_and_values():
    result = load_dataframe(b"a,b\n1,x\n2,y\n3,z\n", "data.csv")
    assert result.df.shape == (3, 2)
    assert list(result.df.columns) == ["a", "b"]
    assert result.df["a"].tolist() == [1, 2, 3]
    assert result.filename == "data.csv"


def test_valid_excel_loads_and_warns_about_first_sheet_only():
    df = pd.DataFrame({"n": [1, 2, 3], "t": ["a", "b", "c"]})
    result = load_dataframe(_xlsx(df), "book.xlsx")
    assert result.df.equals(df)
    assert any("first worksheet" in w for w in result.warnings)


def test_empty_bytes_rejected():
    with pytest.raises(DataLoadError, match="empty"):
        load_dataframe(b"", "x.csv")


def test_whitespace_only_csv_rejected():
    with pytest.raises(DataLoadError, match="no data"):
        load_dataframe(b"   \n  \n", "x.csv")


def test_header_only_csv_rejected():
    with pytest.raises(DataLoadError, match="no data rows"):
        load_dataframe(b"a,b,c\n", "x.csv")


def test_all_empty_cells_rejected():
    with pytest.raises(DataLoadError, match="empty"):
        load_dataframe(b"a,b\n,\n,\n", "x.csv")


@pytest.mark.parametrize("content", [b"a,b\n1,2\n3,4,5,6,7\n", b"a,b\n1,2,3\n4,5,6\n"])
def test_malformed_csv_gives_friendly_error(content):
    with pytest.raises(DataLoadError):
        load_dataframe(content, "bad.csv")


def test_binary_file_with_csv_extension_rejected():
    with pytest.raises(DataLoadError, match="binary"):
        load_dataframe(b"PK\x03\x04\x00\x00\x00binary", "fake.csv")


def test_corrupt_excel_rejected():
    with pytest.raises(DataLoadError, match="Excel"):
        load_dataframe(b"this is not a zip file", "x.xlsx")


@pytest.mark.parametrize("name", ["data.txt", "data.xls", "archive.zip", "noextension", "script.py"])
def test_unsupported_extension_rejected(name):
    with pytest.raises(DataLoadError, match="Unsupported"):
        load_dataframe(b"a,b\n1,2\n", name)


def test_missing_filename_rejected():
    with pytest.raises(DataLoadError):
        load_dataframe(b"a\n1\n", "")


def test_extension_check_is_case_insensitive():
    assert load_dataframe(b"a\n1\n", "DATA.CSV").df.shape == (1, 1)


def test_file_size_limit_enforced():
    with pytest.raises(DataLoadError, match="limit"):
        load_dataframe(b"a\n" + b"1\n" * 2000, "big.csv", max_mb=0.001)


def test_row_limit_rejects_instead_of_truncating():
    data = b"a\n" + b"1\n" * 50
    with pytest.raises(DataLoadError, match="never truncates"):
        load_dataframe(data, "x.csv", max_rows=10)


def test_column_limit_enforced():
    with pytest.raises(DataLoadError, match="columns"):
        load_dataframe(b"a,b,c\n1,2,3\n", "x.csv", max_columns=2)


def test_duplicate_and_blank_column_names_made_unique_and_reported():
    result = load_dataframe(b"a,a,,a\n1,2,3,4\n", "x.csv")
    assert list(result.df.columns) == ["a", "a_2", "unnamed_3", "a_3"]
    assert any("Duplicate column name" in w for w in result.warnings)
    assert any("no name" in w for w in result.warnings)


def test_excel_duplicate_and_blank_headers_handled():
    df = pd.DataFrame([[1, 2, 3]], columns=["x", "x", None])
    result = load_dataframe(_xlsx(df), "d.xlsx")
    assert len(set(result.df.columns)) == 3
    assert "x_2" in result.df.columns


def test_semicolon_separator_detected():
    result = load_dataframe(b"a;b\n1;2\n3;4\n", "x.csv")
    assert list(result.df.columns) == ["a", "b"]
    assert any("separator" in w for w in result.warnings)


def test_latin1_file_decodes_with_warning():
    result = load_dataframe("name\ncaf\xe9\n".encode("latin-1"), "x.csv")
    assert result.df["name"].iloc[0] == "caf\xe9"
    assert any("not valid UTF-8" in w for w in result.warnings)


def test_utf8_bom_is_removed_from_first_column_name():
    result = load_dataframe("﻿a,b\n1,2\n".encode("utf-8"), "x.csv")
    assert list(result.df.columns) == ["a", "b"]


def test_normalize_column_names_trims_whitespace():
    names, notes = normalize_column_names([" a ", "b"])
    assert names == ["a", "b"]
    assert len(notes) == 1
