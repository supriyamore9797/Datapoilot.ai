"""Admission Interest Analysis page for DataPilot AI.

Enables colleges to classify calling remarks, track student interest,
filter by faculty and location, manually correct categories, and export reports.
"""

from __future__ import annotations

from pathlib import Path
import openpyxl
import pandas as pd
import streamlit as st

from src import state, ui
from src.admission_analysis import (
    CATEGORIES,
    CATEGORY_COLORS,
    build_admission_chart,
    classify_dataframe,
    compute_admission_kpis,
    detect_admission_columns,
    filter_admission_dataframe,
    to_csv_bytes,
    to_xlsx_bytes,
)
from src.data_loader import DataLoadError, load_dataframe

ui.setup_page(
    "Admission Interest Analysis",
    "Automated calling response classification, interest tracking & follow-up management for college admissions.",
)
ui.show_flash()

SAMPLE_EXCEL_PATH = Path("sample_data/admission_calling_data.xlsx")

# Initialize session state for overrides and filters
if "admission_overrides" not in st.session_state:
    st.session_state["admission_overrides"] = {}
if "adm_filt_faculty" not in st.session_state:
    st.session_state["adm_filt_faculty"] = []
if "adm_filt_location" not in st.session_state:
    st.session_state["adm_filt_location"] = []
if "adm_filt_category" not in st.session_state:
    st.session_state["adm_filt_category"] = list(CATEGORIES)
if "adm_search_input" not in st.session_state:
    st.session_state["adm_search_input"] = ""


def clear_admission_filters() -> None:
    """Safely reset all filter session state variables before widgets render."""
    st.session_state["adm_filt_faculty"] = []
    st.session_state["adm_filt_location"] = []
    st.session_state["adm_filt_category"] = list(CATEGORIES)
    st.session_state["adm_search_input"] = ""


# ---------------------------------------------------------------- dataset resolution
working_df = None
dataset_title = ""

# Handle custom uploaded / local admission calling dataframe in session
if not state.has_data():
    st.info("No admission dataset is loaded yet. Upload your admission calling sheet below or load the built-in sample calling workbook.")
    c_sample, c_upload = st.columns([1, 1], gap="medium")

    with c_sample, st.container(border=True):
        st.subheader("Demo Admission Workbook")
        st.markdown(
            "Load the real **Admission Calling Workbook** with 8,100 student records across faculties, "
            "contact details, Google Form statuses, and real-world caller remarks."
        )
        if SAMPLE_EXCEL_PATH.exists():
            sheet_options = ["Calling Data", "Calling Data 2"]
            selected_sheet = st.selectbox("Select Excel Sheet", sheet_options, key="empty_sample_sheet")
            if st.button("Load Admission Calling Workbook", type="primary", key="btn_load_sample_calling"):
                df_loaded = pd.read_excel(SAMPLE_EXCEL_PATH, sheet_name=selected_sheet)
                state.load_dataset(df_loaded, f"admission_{selected_sheet.lower().replace(' ', '_')}.xlsx", "sample")
                ui.set_flash("success", f"Loaded '{selected_sheet}' ({len(df_loaded):,} records).")
                st.rerun()
        else:
            st.caption("Sample file not found on disk.")

    with c_upload, st.container(border=True):
        st.subheader("Upload Calling Sheet")
        uploaded_file = st.file_uploader("Upload CSV or Excel (.xlsx) file", type=["csv", "xlsx"], key="empty_calling_upload")
        if uploaded_file is not None:
            try:
                res = load_dataframe(uploaded_file.getvalue(), uploaded_file.name)
                state.load_dataset(res.df, res.filename, "upload", res.warnings)
                ui.set_flash("success", f"Loaded '{res.filename}' ({len(res.df):,} rows).")
                st.rerun()
            except DataLoadError as exc:
                st.error(str(exc))
    ui.show_footer()
    st.stop()

# Data exists in state
working_df = state.get_working()
dataset_title = st.session_state.get("dataset_name", "Admission Dataset")
ui.data_badges()

# Option to switch sheet if current file is an Excel file with multiple sheets
if SAMPLE_EXCEL_PATH.exists() and "admission" in dataset_title.lower():
    with st.expander("Switch Sheet / Upload Alternate Calling File", expanded=False):
        st.markdown(
            "The sample workbook has two sheets:  \n"
            "- **Calling Data** (8,100 records across **NASHIK**, **AHMEDNAGAR**, and **PUNE**)  \n"
            "- **Calling Data 2** (5,425 records exclusively from **PUNE**)"
        )
        c_switch, c_alt = st.columns([1, 1])
        with c_switch:
            sheet_choice = st.selectbox("Switch Sheet from Sample Workbook", ["Calling Data", "Calling Data 2"], key="switch_sheet_select")
            if st.button("Load Selected Sheet", key="btn_switch_sheet"):
                new_df = pd.read_excel(SAMPLE_EXCEL_PATH, sheet_name=sheet_choice)
                state.load_dataset(new_df, f"admission_{sheet_choice.lower().replace(' ', '_')}.xlsx", "sample")
                st.session_state["admission_overrides"] = {}
                ui.set_flash("success", f"Switched to '{sheet_choice}' ({len(new_df):,} rows).")
                st.rerun()
        with c_alt:
            alt_file = st.file_uploader("Upload another file", type=["csv", "xlsx"], key="alt_calling_upload")
            if alt_file is not None:
                try:
                    res = load_dataframe(alt_file.getvalue(), alt_file.name)
                    state.load_dataset(res.df, res.filename, "upload", res.warnings)
                    st.session_state["admission_overrides"] = {}
                    ui.set_flash("success", f"Loaded '{res.filename}' ({len(res.df):,} rows).")
                    st.rerun()
                except DataLoadError as exc:
                    st.error(str(exc))

# ---------------------------------------------------------------- column mapping
detected = detect_admission_columns(working_df)
cols_list = list(working_df.columns)

with st.expander("Column Detection & Mapping", expanded=False):
    st.caption("Verify or adjust dynamically detected columns for student details and caller remarks.")
    m1, m2, m3 = st.columns(3)
    with m1:
        remarks_idx = cols_list.index(detected["remarks"]) if detected["remarks"] in cols_list else 0
        mapped_remarks = st.selectbox("Remarks Column (Required)", cols_list, index=remarks_idx, key="adm_map_remarks")
        student_idx = cols_list.index(detected["student_name"]) if detected["student_name"] in cols_list else 0
        mapped_student = st.selectbox("Student Name Column", ["(None)"] + cols_list, index=student_idx + 1 if detected["student_name"] in cols_list else 0, key="adm_map_student")
    with m2:
        faculty_idx = cols_list.index(detected["faculty"]) if detected["faculty"] in cols_list else 0
        mapped_faculty = st.selectbox("Faculty / Caller Column", ["(None)"] + cols_list, index=faculty_idx + 1 if detected["faculty"] in cols_list else 0, key="adm_map_faculty")
        loc_idx = cols_list.index(detected["location"]) if detected["location"] in cols_list else 0
        mapped_location = st.selectbox("Location / Address Column", ["(None)"] + cols_list, index=loc_idx + 1 if detected["location"] in cols_list else 0, key="adm_map_location")
    with m3:
        form_idx = cols_list.index(detected["google_form"]) if detected["google_form"] in cols_list else 0
        mapped_form = st.selectbox("Google Form Column", ["(None)"] + cols_list, index=form_idx + 1 if detected["google_form"] in cols_list else 0, key="adm_map_form")
        contact_idx = cols_list.index(detected["contact"]) if detected["contact"] in cols_list else 0
        mapped_contact = st.selectbox("Contact Number Column", ["(None)"] + cols_list, index=contact_idx + 1 if detected["contact"] in cols_list else 0, key="adm_map_contact")

student_col = None if mapped_student == "(None)" else mapped_student
faculty_col = None if mapped_faculty == "(None)" else mapped_faculty
location_col = None if mapped_location == "(None)" else mapped_location
form_col = None if mapped_form == "(None)" else mapped_form
contact_col = None if mapped_contact == "(None)" else mapped_contact

# ---------------------------------------------------------------- automated classification
st.info(
    "**Automated Heuristic Classification**: Remarks are classified using case-insensitive keyword analysis and spelling correction "
    "(e.g. *intrested*, *not recieved*, *rechable*). Negative-interest rules take strict precedence over positive rules. "
    "Blank remarks and *'Google Form Filled = Yes'* are never counted as proof of interest. "
    "You can manually review and correct individual classifications below."
)

classified_df = classify_dataframe(
    working_df,
    remark_col=mapped_remarks,
    form_col=form_col,
    overrides=st.session_state["admission_overrides"],
)

kpis = compute_admission_kpis(classified_df, category_col="Interest_Status")

# ---------------------------------------------------------------- KPI cards
ui.metric_cards([
    ("Total Records", f"{kpis['total_records']:,}", None),
    ("Interested", f"{kpis['interested']:,}", f"{kpis['percentages']['Interested']}% of total"),
    ("Not Interested", f"{kpis['not_interested']:,}", f"{kpis['percentages']['Not Interested']}% of total"),
    ("Follow-up / Undecided", f"{kpis['undecided']:,}", f"{kpis['percentages']['Follow-up / Undecided']}% of total"),
    ("Unreachable", f"{kpis['unreachable']:,}", f"{kpis['percentages']['Unreachable']}% of total"),
    ("Unknown (No Remark / Review)", f"{kpis['unknown']:,}", f"{round(kpis['percentages']['Other / Needs Review'] + kpis['percentages']['No Remark'], 1)}% of total"),
])

if kpis["manual_overrides_count"] > 0:
    st.caption(f"Active manual category corrections: **{kpis['manual_overrides_count']} record(s)**.")

# ---------------------------------------------------------------- charts and breakdown
st.divider()
st.subheader("1. Response & Location Distribution")

tab_overall, tab_location = st.tabs(["Response Distribution", "Location-Wise Breakdown"])

with tab_overall:
    chart_col, table_col = st.columns([3, 2], gap="medium")
    with chart_col:
        fig = build_admission_chart(kpis["counts"])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No records to chart.")

    with table_col:
        st.markdown("**Category Breakdown**")
        breakdown_data = [
            {"Interest Status": cat, "Count": f"{kpis['counts'][cat]:,}", "Share": f"{kpis['percentages'][cat]}%"}
            for cat in CATEGORIES
        ]
        st.dataframe(pd.DataFrame(breakdown_data), hide_index=True, use_container_width=True)

with tab_location:
    if location_col and location_col in classified_df.columns:
        st.markdown("**Student Interest Distribution by Location / City**")
        loc_cross = pd.crosstab(
            classified_df[location_col].astype(str).str.strip(),
            classified_df["Interest_Status"],
        )
        for c in CATEGORIES:
            if c not in loc_cross.columns:
                loc_cross[c] = 0
        loc_cross = loc_cross[CATEGORIES]
        loc_cross["Total Students"] = loc_cross.sum(axis=1)
        loc_cross = loc_cross.sort_values(by="Total Students", ascending=False).reset_index()
        loc_cross.rename(columns={location_col: "Location"}, inplace=True)
        st.dataframe(loc_cross, hide_index=True, use_container_width=True)
    else:
        st.info("No location column mapped.")

# ---------------------------------------------------------------- filters
st.divider()
st.subheader("2. Filter Student Records")

if location_col and location_col in classified_df.columns:
    loc_vals = classified_df[location_col].dropna().astype(str).str.strip()
    location_options = sorted([v for v in loc_vals.unique() if v])
else:
    location_options = []

if len(location_options) == 1 and "pune" in location_options[0].lower():
    st.info(
        f"Notice: The active sheet currently contains only records from **{location_options[0]}**. "
        "If you want to explore records from **Nashik** and **Ahmednagar**, open 'Switch Sheet' above and load **Calling Data**."
    )

fc1, fc2, fc3 = st.columns([1, 1, 1], gap="medium")

with fc1:
    faculty_options = sorted([str(x).strip() for x in classified_df[faculty_col].dropna().unique() if str(x).strip()]) if faculty_col else []
    selected_faculties = st.multiselect("Filter by Faculty", faculty_options, key="adm_filt_faculty")

with fc2:
    selected_locations = st.multiselect("Filter by Location", location_options, key="adm_filt_location")

with fc3:
    selected_categories = st.multiselect(
        "Filter by Interest Status (Exact Match)",
        CATEGORIES,
        key="adm_filt_category",
    )

sc1, sc2 = st.columns([4, 1], gap="medium")
with sc1:
    search_cols = [c for c in [student_col, contact_col] if c is not None]
    search_q = st.text_input("Search by Student Name or Contact Number", key="adm_search_input")
with sc2:
    st.write("")
    st.write("")
    st.button(
        "Clear All Filters",
        key="btn_clear_filters",
        on_click=clear_admission_filters,
        use_container_width=True,
    )

# Apply filters
filtered_df = filter_admission_dataframe(
    classified_df,
    faculty_col=faculty_col,
    selected_faculties=selected_faculties,
    location_col=location_col,
    selected_locations=selected_locations,
    category_col="Interest_Status",
    selected_categories=selected_categories,
    search_query=search_q,
    search_cols=search_cols,
)

st.markdown(f"**Showing {len(filtered_df):,} matching record(s)** out of {len(classified_df):,} total records.")

# Filter Diagnostics (Collapsible pipeline audit)
with st.expander("Filter Diagnostics & Pipeline Step Counts", expanded=False):
    st.caption("Inspect step counts to verify how filters and search isolate records.")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Total Loaded", f"{len(classified_df):,}")
    with d2:
        loc_step_df = filter_admission_dataframe(classified_df, location_col=location_col, selected_locations=selected_locations)
        st.metric("After Location Filter", f"{len(loc_step_df):,}")
    with d3:
        status_step_df = filter_admission_dataframe(classified_df, category_col="Interest_Status", selected_categories=selected_categories)
        st.metric("After Status Filter", f"{len(status_step_df):,}")
    with d4:
        st.metric("Final Filtered Records", f"{len(filtered_df):,}")

# Sorting controls
sort_c1, sort_c2 = st.columns([3, 2], gap="medium")
with sort_c1:
    sort_options = ["(Original Order)"]
    if location_col and location_col in filtered_df.columns:
        sort_options.append("Location")
    if "Interest_Status" in filtered_df.columns:
        sort_options.append("Interest Status")
    if student_col and student_col in filtered_df.columns:
        sort_options.append("Student Name")
    if faculty_col and faculty_col in filtered_df.columns:
        sort_options.append("Faculty")
    sort_choice = st.selectbox("Sort Table By", sort_options, key="adm_sort_field")

with sort_c2:
    sort_dir = st.radio("Sort Order", ["Ascending (A → Z)", "Descending (Z → A)"], horizontal=True, key="adm_sort_dir")

asc = sort_dir.startswith("Ascending")
if sort_choice == "Location" and location_col:
    filtered_df = filtered_df.sort_values(by=location_col, ascending=asc)
elif sort_choice == "Interest Status":
    filtered_df = filtered_df.sort_values(by="Interest_Status", ascending=asc)
elif sort_choice == "Student Name" and student_col:
    filtered_df = filtered_df.sort_values(by=student_col, ascending=asc)
elif sort_choice == "Faculty" and faculty_col:
    filtered_df = filtered_df.sort_values(by=faculty_col, ascending=asc)

# Preview columns
display_cols = []
for c in [student_col, faculty_col, location_col, contact_col, form_col, mapped_remarks, "Interest_Status", "Classification Type"]:
    if c and c in filtered_df.columns and c not in display_cols:
        display_cols.append(c)

col_config: dict[str, Any] = {
    "Interest_Status": st.column_config.TextColumn("Interest Status", width="medium"),
    "Classification Type": st.column_config.TextColumn("Audit Type", width="small"),
}
if mapped_remarks and mapped_remarks in filtered_df.columns:
    col_config[mapped_remarks] = st.column_config.TextColumn("Remarks", width="large")
if student_col and student_col in filtered_df.columns:
    col_config[student_col] = st.column_config.TextColumn("Student Name", width="medium")
if faculty_col and faculty_col in filtered_df.columns:
    col_config[faculty_col] = st.column_config.TextColumn("Faculty", width="medium")
if location_col and location_col in filtered_df.columns:
    col_config[location_col] = st.column_config.TextColumn("Location", width="medium")
if contact_col and contact_col in filtered_df.columns:
    col_config[contact_col] = st.column_config.TextColumn("Contact", width="small")
if form_col and form_col in filtered_df.columns:
    col_config[form_col] = st.column_config.TextColumn("Google Form", width="small")

st.dataframe(
    ui.arrow_safe(filtered_df[display_cols] if display_cols else filtered_df),
    hide_index=True,
    use_container_width=True,
    height=450,
    column_config=col_config,
)

# ---------------------------------------------------------------- manual correction panel
st.divider()
st.subheader("3. Manual Category Review & Correction")
st.write(
    "If an automated category needs human adjustment (for example, a student later called back to confirm interest), "
    "select the record index to override its classification."
)

mc1, mc2, mc3 = st.columns([2, 2, 1])

with mc1:
    record_indices = filtered_df.index.tolist()
    if record_indices:
        target_idx = st.selectbox(
            "Select Record Index to Correct",
            record_indices,
            format_func=lambda idx: f"Row {idx}: {filtered_df.loc[idx, student_col] if student_col else 'Record'} ({filtered_df.loc[idx, 'Interest_Status']})",
            key="adm_override_target",
        )
    else:
        target_idx = None
        st.caption("No matching records available.")

with mc2:
    current_cat = filtered_df.loc[target_idx, "Interest_Status"] if target_idx is not None else CATEGORIES[0]
    cat_idx = CATEGORIES.index(current_cat) if current_cat in CATEGORIES else 0
    new_cat = st.selectbox("Assign New Category", CATEGORIES, index=cat_idx, key="adm_override_new_cat")

with mc3:
    st.write("")
    st.write("")
    if st.button("Apply Correction", type="primary", disabled=target_idx is None, key="btn_apply_override"):
        st.session_state["admission_overrides"][target_idx] = new_cat
        ui.set_flash("success", f"Updated row {target_idx} category to '{new_cat}'.")
        st.rerun()

if st.session_state["admission_overrides"]:
    if st.button("Reset All Manual Corrections", key="btn_reset_overrides"):
        st.session_state["admission_overrides"] = {}
        ui.set_flash("info", "Reset all manual corrections to automated classification.")
        st.rerun()

# ---------------------------------------------------------------- export section
st.divider()
st.subheader("4. Export Filtered Records")
st.write(
    f"Download the **{len(filtered_df):,} matching record(s)**. "
    "The exported file includes all original calling data plus the assigned category and classification audit type."
)

raw_name = dataset_title.rsplit(".", 1)[0]
exp1, exp2 = st.columns(2)

with exp1:
    st.download_button(
        "Download Filtered CSV",
        data=to_csv_bytes(filtered_df),
        file_name=f"{raw_name}_admission_filtered.csv",
        mime="text/csv",
        disabled=filtered_df.empty,
        type="primary",
        key="btn_download_admission_csv",
    )

with exp2:
    st.download_button(
        "Download Filtered Excel (.xlsx)",
        data=to_xlsx_bytes(filtered_df, sheet_name="Admission Filtered"),
        file_name=f"{raw_name}_admission_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=filtered_df.empty,
        key="btn_download_admission_xlsx",
    )

ui.show_footer()

