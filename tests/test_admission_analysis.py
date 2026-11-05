"""Tests for Admission Interest Analysis: spelling variations, negative rules,

conflicting remarks, blank remarks, ambiguous responses, column detection,
exact Interest_Status filtering, manual overrides, and export.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.admission_analysis import (
    CATEGORIES,
    build_admission_chart,
    classify_dataframe,
    classify_remark,
    compute_admission_kpis,
    detect_admission_columns,
    filter_admission_dataframe,
    normalize_remark,
    to_csv_bytes,
    to_xlsx_bytes,
)


def test_spelling_variations():
    """Verify common typos and phonetic variants in caller remarks."""
    assert classify_remark("intrested") == "Interested"
    assert classify_remark("intersted") == "Interested"
    assert classify_remark("interessted") == "Interested"
    assert classify_remark("interested in nursing") == "Interested"
    assert classify_remark("looking for engineering") == "Interested"

    assert classify_remark("not intrested") == "Not Interested"
    assert classify_remark("not interesed") == "Not Interested"
    assert classify_remark("not intersted") == "Not Interested"

    assert classify_remark("not recieved") == "Unreachable"
    assert classify_remark("not recived") == "Unreachable"
    assert classify_remark("not rechable") == "Unreachable"
    assert classify_remark("swich off") == "Unreachable"
    assert classify_remark("busey") == "Unreachable"

    assert classify_remark("not deciced yet") == "Follow-up / Undecided"
    assert classify_remark("will deside after CET") == "Follow-up / Undecided"


def test_negative_interest_takes_precedence_over_positive_interest():
    """Ensure negative phrases are never mistakenly classified as interested."""
    assert classify_remark("not interested") == "Not Interested"
    assert classify_remark("not interested in Pune") == "Not Interested"
    assert classify_remark("Not Intrested after call back") == "Not Interested"
    assert classify_remark("He said not interested in college") == "Not Interested"
    assert classify_remark("already taken admission") == "Not Interested"
    assert classify_remark("Taken admission in another college") == "Not Interested"
    assert classify_remark("already admitted") == "Not Interested"
    assert classify_remark("admission done") == "Not Interested"
    assert classify_remark("taking drop for neet") == "Not Interested"
    assert classify_remark("failed in 12th") == "Not Interested"


def test_not_received_becomes_unreachable_not_not_interested():
    """Explicitly verify that 'not received' classifies as Unreachable and NOT Not Interested."""
    status_received = classify_remark("not received")
    assert status_received == "Unreachable"
    assert status_received != "Not Interested"

    status_recieved = classify_remark("not recieved")
    assert status_recieved == "Unreachable"
    assert status_recieved != "Not Interested"


def test_interested_does_not_return_not_interested_and_vice_versa():
    """Verify that positive interest remarks and negative remarks never cross-classify."""
    positive_samples = [
        "interested",
        "intrested",
        "wants admission",
        "will take admission",
        "will visit",
        "interested in btech",
    ]
    for text in positive_samples:
        cat = classify_remark(text)
        assert cat == "Interested"
        assert cat != "Not Interested"

    negative_samples = [
        "not interested",
        "not intrested",
        "no interest",
        "not willing",
        "will not take admission",
        "not joining",
    ]
    for text in negative_samples:
        cat = classify_remark(text)
        assert cat == "Not Interested"
        assert cat != "Interested"


def test_conflicting_remarks_handled_correctly():
    """Remarks combining negative words with follow-up instructions must classify as Follow-up / Undecided."""
    assert classify_remark("not interested now, call later") == "Follow-up / Undecided"
    assert classify_remark("interested but needs time") == "Follow-up / Undecided"
    assert classify_remark("busy now, call tomorrow") == "Follow-up / Undecided"
    assert classify_remark("think about it") == "Follow-up / Undecided"
    assert classify_remark("not sure") == "Follow-up / Undecided"
    assert classify_remark("maybe") == "Follow-up / Undecided"


def test_blank_remarks_and_google_form_filled():
    """Blank remarks or Google Form Filled = Yes must NEVER be classified as Interested."""
    assert classify_remark(None) == "No Remark"
    assert classify_remark("") == "No Remark"
    assert classify_remark("   ") == "No Remark"
    assert classify_remark(float("nan")) == "No Remark"
    assert classify_remark("-") == "No Remark"
    assert classify_remark("N/A") == "No Remark"
    assert classify_remark("nil") == "No Remark"

    # Even if Google Form is filled, a blank remark is NOT proof of interest
    assert classify_remark(None, google_form_filled="Yes") == "No Remark"
    assert classify_remark("", google_form_filled="Yes") == "No Remark"
    assert classify_remark("not interested", google_form_filled="Yes") == "Not Interested"


def test_unreachable_remarks():
    """Test standard unreachable calling statuses."""
    assert classify_remark("Not Reachable") == "Unreachable"
    assert classify_remark("Wrong number") == "Unreachable"
    assert classify_remark("Switched off") == "Unreachable"
    assert classify_remark("cut the call") == "Unreachable"
    assert classify_remark("Number does not exist") == "Unreachable"
    assert classify_remark("no answer") == "Unreachable"
    assert classify_remark("not connected") == "Unreachable"


def test_ambiguous_responses_classified_as_other_or_followup():
    """Ambiguous or unclassified remarks fall into Other / Needs Review or Follow-up."""
    assert classify_remark("relative of staff") == "Other / Needs Review"
    assert classify_remark("student was asleep") == "Other / Needs Review"
    assert classify_remark("call was answered by neighbour") == "Other / Needs Review"


def test_detect_admission_columns():
    """Verify dynamic detection of columns across typical naming variations."""
    df = pd.DataFrame({
        "Sr. No": [1],
        "Name of the Faculty": ["Prof. Patil"],
        "Name of the Student": ["Aarav Shinde"],
        "Student Contact Number": ["9876543210"],
        "Address": ["Pune"],
        "Google Form Filled": ["Yes"],
        "Remarks, if any.": ["Interested in BCA"],
    })
    detected = detect_admission_columns(df)
    assert detected["student_name"] == "Name of the Student"
    assert detected["faculty"] == "Name of the Faculty"
    assert detected["location"] == "Address"
    assert detected["google_form"] == "Google Form Filled"
    assert detected["remarks"] == "Remarks, if any."
    assert detected["contact"] == "Student Contact Number"


def test_classify_dataframe_creates_interest_status_column_and_overrides():
    """Verify dataframe classification creates Interest_Status and respects manual overrides."""
    df = pd.DataFrame({
        "Student": ["Alice", "Bob", "Charlie", "David"],
        "Remarks": ["interested", "not interested", "switched off", "waiting for cet"],
    })
    # Automated
    res = classify_dataframe(df, remark_col="Remarks")
    assert "Interest_Status" in res.columns
    assert res.loc[0, "Interest_Status"] == "Interested"
    assert res.loc[1, "Interest_Status"] == "Not Interested"
    assert res.loc[2, "Interest_Status"] == "Unreachable"
    assert res.loc[3, "Interest_Status"] == "Follow-up / Undecided"
    assert not res["Is Override"].any()

    # Manual override row 1 (Bob) from Not Interested to Interested
    res_ov = classify_dataframe(df, remark_col="Remarks", overrides={1: "Interested"})
    assert res_ov.loc[1, "Interest_Status"] == "Interested"
    assert bool(res_ov.loc[1, "Is Override"]) is True
    assert res_ov.loc[1, "Classification Type"] == "Manual Correction"


def test_compute_kpis():
    """Verify KPI counts and percentages match the categories."""
    df = pd.DataFrame({
        "Interest_Status": ["Interested", "Interested", "Not Interested", "No Remark"],
        "Is Override": [False, False, False, False],
    })
    kpis = compute_admission_kpis(df, category_col="Interest_Status")
    assert kpis["total_records"] == 4
    assert kpis["interested"] == 2
    assert kpis["not_interested"] == 1
    assert kpis["counts"]["No Remark"] == 1
    assert kpis["percentages"]["Interested"] == 50.0
    assert kpis["percentages"]["Not Interested"] == 25.0


def test_filter_exact_interest_status_does_not_substring_match_remarks():
    """Crucial bug fix test: Selecting 'Interested' must NOT return rows with 'Not Interested'."""
    df = pd.DataFrame({
        "Student": ["S1", "S2", "S3"],
        "Remarks": ["interested in bca", "not interested in pune", "call later"],
        "Interest_Status": ["Interested", "Not Interested", "Follow-up / Undecided"],
    })
    # When user filters for 'Interested'
    filtered = filter_admission_dataframe(
        df,
        category_col="Interest_Status",
        selected_categories=["Interested"],
    )
    # MUST contain S1 and MUST NOT contain S2 ('Not Interested')
    assert len(filtered) == 1
    assert filtered.iloc[0]["Student"] == "S1"
    assert "not interested" not in filtered.iloc[0]["Remarks"].lower()

    # When user filters for 'Not Interested'
    filtered_not = filter_admission_dataframe(
        df,
        category_col="Interest_Status",
        selected_categories=["Not Interested"],
    )
    assert len(filtered_not) == 1
    assert filtered_not.iloc[0]["Student"] == "S2"

    # When all filters are cleared
    cleared = filter_admission_dataframe(
        df,
        category_col="Interest_Status",
        selected_categories=list(CATEGORIES),
    )
    assert len(cleared) == 3


def test_clear_admission_filters_callback_logic():
    """Verify session state reset logic works cleanly and repeatedly."""
    state_mock = {
        "adm_filt_faculty": ["Prof. Shinde"],
        "adm_filt_location": ["Pune"],
        "adm_filt_category": ["Interested"],
        "adm_search_input": "Aarav",
    }

    # Simulate clear_admission_filters
    state_mock["adm_filt_faculty"] = []
    state_mock["adm_filt_location"] = []
    state_mock["adm_filt_category"] = list(CATEGORIES)
    state_mock["adm_search_input"] = ""

    assert state_mock["adm_filt_faculty"] == []
    assert state_mock["adm_filt_location"] == []
    assert state_mock["adm_filt_category"] == CATEGORIES
    assert state_mock["adm_search_input"] == ""



def test_filter_and_export():
    """Verify faculty/location/category filtering and CSV/Excel export."""
    df = pd.DataFrame({
        "Student": ["S1", "S2", "S3"],
        "Faculty": ["F1", "F1", "F2"],
        "Location": ["Pune", "Nashik", "Pune"],
        "Interest_Status": ["Interested", "Not Interested", "Interested"],
        "Remarks": ["interested", "not interested", "intrested"],
    })
    filtered = filter_admission_dataframe(
        df,
        faculty_col="Faculty",
        selected_faculties=["F1"],
        location_col="Location",
        selected_locations=["Pune"],
        category_col="Interest_Status",
        selected_categories=["Interested"],
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["Student"] == "S1"

    # Export
    csv_bytes = to_csv_bytes(filtered)
    assert b"S1" in csv_bytes
    assert b"S2" not in csv_bytes

    xlsx_bytes = to_xlsx_bytes(filtered)
    assert len(xlsx_bytes) > 1000


def test_admission_calling_dataset_if_available():
    """Verify full end-to-end classification on the actual calling workbook if present."""
    sample_path = Path("sample_data/admission_calling_data.xlsx")
    if not sample_path.exists():
        pytest.skip("sample_data/admission_calling_data.xlsx is not in workspace")

    df = pd.read_excel(sample_path, sheet_name="Calling Data")
    cols = detect_admission_columns(df)
    assert cols["remarks"] is not None

    classified = classify_dataframe(df, remark_col=cols["remarks"], form_col=cols["google_form"])
    kpis = compute_admission_kpis(classified, category_col="Interest_Status")

    assert kpis["total_records"] == 8100
    assert kpis["interested"] > 0
    assert kpis["not_interested"] > 0
    assert kpis["unreachable"] > 0
    assert kpis["undecided"] > 0
    assert kpis["counts"]["No Remark"] > 0

    # Ensure chart builds without error
    fig = build_admission_chart(kpis["counts"])
    assert fig is not None


def test_user_six_pune_example_records_classification_and_filtering():
    """Verify exact classifications and case-insensitive location filtering on user's 6 example records."""
    data = [
        {"Location": "PUNE", "Remarks": "Intrested, BTech", "Expected": "Interested"},
        {"Location": "PUNE", "Remarks": "Intrested- BSc", "Expected": "Interested"},
        {"Location": "PUNE", "Remarks": "Intrested, course not decide", "Expected": "Follow-up / Undecided"},
        {"Location": "PUNE", "Remarks": "Will Visit", "Expected": "Interested"},
        {"Location": "PUNE", "Remarks": "Will Visit", "Expected": "Interested"},
        {"Location": "PUNE", "Remarks": "Will Visit", "Expected": "Interested"},
    ]
    df = pd.DataFrame(data)

    # 1. Test individual classification for each remark
    for item in data:
        actual = classify_remark(item["Remarks"])
        assert actual == item["Expected"], f"Failed for {item['Remarks']!r}: expected {item['Expected']!r}, got {actual!r}"

    # 2. Test classify_dataframe creates Interest_Status matching expectations
    classified = classify_dataframe(df, remark_col="Remarks")
    assert classified["Interest_Status"].tolist() == [d["Expected"] for d in data]

    # 3. Test case-insensitive location filtering with PUNE, Pune, and ' pune '
    for loc_filter in ["PUNE", "Pune", " pune ", "pUnE"]:
        filtered = filter_admission_dataframe(
            classified,
            location_col="Location",
            selected_locations=[loc_filter],
            category_col="Interest_Status",
            selected_categories=["Interested"],
        )
        # Should return exactly the 5 Interested records, excluding the Follow-up record
        assert len(filtered) == 5
        assert "Intrested, course not decide" not in filtered["Remarks"].values

    # 4. Test Follow-up / Undecided returns the course not decide record
    filtered_undecided = filter_admission_dataframe(
        classified,
        location_col="Location",
        selected_locations=["Pune"],
        category_col="Interest_Status",
        selected_categories=["Follow-up / Undecided"],
    )
    assert len(filtered_undecided) == 1
    assert filtered_undecided.iloc[0]["Remarks"] == "Intrested, course not decide"

