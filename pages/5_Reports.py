"""Reports page: generate and download a PDF report built from the current data and results."""

import streamlit as st

from src import profiling, reporting, state, ui
from src.config import REPORTS_DIR

ui.setup_page("Reports", "Generate a PDF report from your current data, cleaning history and model results. It is created locally on this computer.")
ui.show_flash()
df = ui.require_data()

name = st.session_state["dataset_name"]
log = st.session_state["cleaning_log"]
results = state.current_ml_results()

st.subheader("What the report will contain")
ui.metric_cards([
    ("Data version", "Original" if not log else "Cleaned", state.version_label()),
    ("Cleaning steps", len(log), "listed in the report" if log else "none applied"),
    ("Model results", "Included" if results else "Not available", f"{results.task}, target '{results.target}'" if results else "train models in ML Studio to include them"),
])
if st.session_state.get("ml_stale_notice") and results is None:
    st.info(st.session_state["ml_stale_notice"])

with st.expander("Insights that will appear in the report"):
    for line in profiling.generate_insights(df):
        st.markdown(f"- {line}")

if st.button("Generate PDF report", type="primary"):
    try:
        with st.spinner("Building the report..."):
            pdf = reporting.build_pdf_report(df, name, state.version_label(), log, results)
        path_note = ""
        try:
            saved = reporting.save_report(pdf, name, REPORTS_DIR)
            path_note = f" A copy was saved to {saved.name} in the reports folder."
        except OSError as exc:
            path_note = f" (A copy could not be saved to the reports folder: {exc.strerror or exc}.)"
        state.store_report(pdf, f"datapilot_report_{name.rsplit('.', 1)[0]}.pdf")
        ui.set_flash("success", "The report was generated." + path_note)
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))

report = state.current_report()
if report is None:
    st.info("No report has been generated for the current data and results yet.")
else:
    st.success(f"A report generated at {report['created']} matches the current data{' and model results' if report['ml_present'] else ''}.")
    st.download_button("Download PDF report", report["pdf"], file_name=report["file_name"], mime="application/pdf")
    st.caption("If you change the data or retrain models, generate the report again so it matches.")

ui.show_footer()
