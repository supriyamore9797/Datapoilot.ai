"""PDF report generation with ReportLab (fully local).

The report only contains values computed from the supplied data and results. Sections that
have no data (no cleaning, no trained model) say so explicitly instead of being filled in.
Text is sanitised for the built-in Helvetica font (Latin-1 / Windows-1252 characters); other
characters are replaced with '?'.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import (
    APP_NAME,
    AUTHOR_NAME,
    COPYRIGHT_YEAR,
    get_copyright_text,
    get_report_developer_text,
)
from src.ml_pipeline import MLResults
from src.profiling import (
    column_profile,
    dataset_overview,
    generate_insights,
    issues_to_frame,
    missing_by_column,
    quality_issues,
)
from src.utils import human_bytes, safe_filename

ACCENT = colors.HexColor("#4F46E5")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
LIGHT = colors.HexColor("#F3F4F8")
BORDER = colors.HexColor("#D1D5DB")
MAX_TABLE_ROWS = 60
PAGE_WIDTH = A4[0] - 4 * cm


def _t(value: Any) -> str:
    """Make arbitrary text safe for ReportLab paragraphs and the Helvetica font."""
    text = "" if value is None else str(value)
    text = text.encode("cp1252", errors="replace").decode("cp1252")
    return escape(text)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24, textColor=INK, spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=11, textColor=MUTED, alignment=TA_CENTER, spaceAfter=4),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=ACCENT, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, textColor=INK, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13, textColor=INK, spaceAfter=4),
        "note": ParagraphStyle("note", parent=base["BodyText"], fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=4),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=8, leading=10, textColor=INK),
        "head": ParagraphStyle("head", parent=base["BodyText"], fontSize=8, leading=10, textColor=colors.white, fontName="Helvetica-Bold"),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontSize=9.5, leading=13, leftIndent=12, bulletIndent=2, spaceAfter=2),
    }


def _table(rows: list[list[Any]], header: list[str], widths: list[float] | None = None, styles: dict | None = None) -> Table:
    """Table with wrapped cells so long column names never overflow the page."""
    st = styles or _styles()
    data = [[Paragraph(_t(h), st["head"]) for h in header]]
    data += [[Paragraph(_t(c), st["cell"]) for c in row] for row in rows]
    if widths is None:
        widths = [PAGE_WIDTH / len(header)] * len(header)
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _frame_table(frame: pd.DataFrame, styles: dict, max_rows: int = MAX_TABLE_ROWS, widths: list[float] | None = None) -> list[Any]:
    shown = frame.head(max_rows)
    cells = [[("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else v)) for v in row] for row in shown.itertuples(index=False)]
    out: list[Any] = [_table(cells, [str(c) for c in shown.columns], widths, styles)]
    if len(frame) > max_rows:
        out.append(Paragraph(_t(f"Showing the first {max_rows} of {len(frame)} rows."), styles["note"]))
    return out


def _bar_chart(items: list[tuple[str, float]], title: str, value_fmt: str = "{:.4g}", width: float = PAGE_WIDTH) -> Drawing:
    """Simple horizontal bar chart drawn with ReportLab shapes (no extra dependencies)."""
    row_h, top, left = 16, 22, 150
    height = top + row_h * len(items) + 6
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 12, _t(title), fontName="Helvetica-Bold", fontSize=9, fillColor=INK))
    peak = max((abs(v) for _, v in items), default=0) or 1.0
    usable = width - left - 60
    for i, (label, value) in enumerate(items):
        y = height - top - row_h * (i + 1) + 3
        text = _t(label)
        text = text if len(text) <= 30 else text[:27] + "..."
        drawing.add(String(0, y + 2, text, fontName="Helvetica", fontSize=8, fillColor=INK))
        drawing.add(Rect(left, y, max(abs(value) / peak * usable, 1), 10, fillColor=ACCENT, strokeColor=None))
        drawing.add(String(left + max(abs(value) / peak * usable, 1) + 4, y + 2, value_fmt.format(value), fontName="Helvetica", fontSize=8, fillColor=MUTED))
    return drawing


def _footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, f"{APP_NAME} - generated locally. Values are computed from the dataset; they are not guarantees.")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _bullets(items: list[str], styles: dict) -> list[Any]:
    return [Paragraph(_t(i), styles["bullet"], bulletText="•") for i in items]


# ------------------------------------------------------------------------ sections
def _section_dataset(df: pd.DataFrame, name: str, version: str, styles: dict) -> list[Any]:
    o = dataset_overview(df)
    rows = [
        ["Dataset", name], ["Data version used", version], ["Rows", f"{o['rows']:,}"], ["Columns", o["columns"]],
        ["Numeric columns", o["numeric_columns"]], ["Categorical columns", o["categorical_columns"]],
        ["Datetime columns", o["datetime_columns"]], ["Missing cells", f"{o['missing_cells']:,} ({o['missing_pct']}%)"],
        ["Duplicate rows", f"{o['duplicate_rows']:,} ({o['duplicate_pct']}%)"], ["Memory usage", human_bytes(o["memory_bytes"])],
    ]
    out: list[Any] = [Paragraph("1. Dataset summary", styles["h1"]),
                      _table(rows, ["Property", "Value"], [PAGE_WIDTH * 0.35, PAGE_WIDTH * 0.65], styles),
                      Paragraph("2. Columns", styles["h1"])]
    prof = column_profile(df)[["Column", "Type", "Missing", "Missing %", "Unique", "Example"]]
    out += _frame_table(prof, styles, widths=[PAGE_WIDTH * f for f in (0.26, 0.12, 0.10, 0.12, 0.10, 0.30)])
    return out


def _section_quality(df: pd.DataFrame, styles: dict) -> list[Any]:
    out: list[Any] = [Paragraph("3. Data quality and missing values", styles["h1"])]
    missing = missing_by_column(df)
    missing = missing[missing["Missing"] > 0]
    if missing.empty:
        out.append(Paragraph("No missing values were found in this version of the data.", styles["body"]))
    else:
        out.append(_bar_chart([(r["Column"], float(r["Missing %"])) for _, r in missing.head(10).iterrows()],
                              "Missing values by column (% of rows)", "{:.1f}%"))
        out += _frame_table(missing, styles, max_rows=20)
    out.append(Paragraph("Potential data-quality issues", styles["h2"]))
    issues = issues_to_frame(quality_issues(df))
    if issues.empty:
        out.append(Paragraph("No data-quality issues were flagged by the automatic checks.", styles["body"]))
    else:
        out += _frame_table(issues[["Severity", "Column", "Issue"]], styles, max_rows=25,
                            widths=[PAGE_WIDTH * f for f in (0.12, 0.23, 0.65)])
        out.append(Paragraph("These are automatic flags for review, not confirmed errors.", styles["note"]))
    return out


def _section_cleaning(log: list[dict[str, Any]], styles: dict) -> list[Any]:
    out: list[Any] = [Paragraph("4. Cleaning operations", styles["h1"])]
    if not log:
        out.append(Paragraph("No cleaning operations were applied. This report describes the original data.", styles["body"]))
        return out
    rows = [[e["timestamp"], e["operation"], ", ".join(e.get("columns", []))[:80], f"{e['rows_before']} -> {e['rows_after']}", e["details"]] for e in log]
    out.append(_table(rows, ["Time", "Operation", "Columns", "Rows", "Details"],
                      [PAGE_WIDTH * f for f in (0.16, 0.18, 0.17, 0.10, 0.39)], styles))
    return out


def _section_insights(df: pd.DataFrame, styles: dict) -> list[Any]:
    return [Paragraph("5. Selected insights (computed from the data)", styles["h1"]), *_bullets(generate_insights(df), styles)]


def _section_ml(results: MLResults | None, styles: dict) -> list[Any]:
    out: list[Any] = [Paragraph("6. Machine learning", styles["h1"])]
    if results is None:
        out.append(Paragraph("Not available: no model has been trained on this version of the data.", styles["body"]))
        return out
    cfg, split = results.config, results.split
    rows = [
        ["Task", results.task], ["Target", results.target], ["Features used", ", ".join(results.plan.features)],
        ["Train / test rows", f"{split['train_rows']} / {split['test_rows']} (test share {split['test_size']:.0%}, "
                              f"{'stratified' if split['stratified'] else 'not stratified'})"],
        ["Random state", cfg["random_state"]], ["Models", ", ".join(cfg["models"])],
        ["Class weighting", "balanced" if cfg.get("balance_classes") else "none"],
        ["Selection criterion", results.selection_note.split(" on the held-out")[0].replace("Best model chosen by ", "")],
    ]
    out.append(_table(rows, ["Setting", "Value"], [PAGE_WIDTH * 0.25, PAGE_WIDTH * 0.75], styles))
    if results.plan.excluded:
        out.append(Paragraph("Columns excluded from the features", styles["h2"]))
        out += _frame_table(pd.DataFrame({"Column": list(results.plan.excluded), "Reason": list(results.plan.excluded.values())}),
                            styles, max_rows=20, widths=[PAGE_WIDTH * 0.25, PAGE_WIDTH * 0.75])
    for warning in results.warnings:
        out.append(Paragraph(_t("Note: " + warning), styles["note"]))

    out.append(Paragraph("7. Model comparison (held-out test performance)", styles["h1"]))
    out += _frame_table(results.comparison, styles)
    out.append(Paragraph(_t(results.averaging_note), styles["note"]))
    baseline = ", ".join(f"{k} {v:.4g}" for k, v in results.baseline.items())
    label = "majority-class" if results.task == "classification" else "predict-the-mean"
    out.append(Paragraph(_t(f"Baseline ({label}) on the same test set: {baseline}."), styles["note"]))
    metric_col = {"f1": "F1", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "rmse": "RMSE", "mae": "MAE", "r2": "R2"}[results.selection_metric]
    out.append(_bar_chart([(r["Model"], float(r[metric_col])) for _, r in results.comparison.iterrows()], f"{metric_col} by model (test set)"))
    out.append(Paragraph(_t(results.selection_note), styles["body"]))
    if results.failed:
        out.append(Paragraph(_t("Models that could not be trained: " + "; ".join(f"{n} ({r})" for n, r in results.failed.items())), styles["note"]))

    best = results.outcomes[results.best_model]
    out.append(Paragraph(_t(f"Best model: {best.name}"), styles["h2"]))
    if best.confusion is not None:
        cm = best.confusion.reset_index().rename(columns={"index": ""})
        out.append(KeepTogether(_frame_table(cm, styles)))
    if best.importance is not None and not best.importance.empty:
        out.append(_bar_chart([(r["Feature"], float(r["Importance"])) for _, r in best.importance.head(10).iterrows()],
                              "Feature importance (impurity-based, from training data)"))
    return out


def _section_limits(results: MLResults | None, styles: dict) -> list[Any]:
    points = [
        "Statistics describe this dataset only; they do not necessarily generalise to other data.",
        "Correlation does not imply causation.",
        "Automatic quality flags are heuristics and need human review.",
        "Charts in this report are simple summaries; interactive charts are available in the application.",
    ]
    if results is not None:
        points += [
            "Model metrics come from a single train/test split; another split could give different numbers.",
            "Models use fixed default settings and were not tuned; results are a baseline, not a final model.",
            "For imbalanced classes, accuracy alone can mislead; compare F1, recall and balanced accuracy with the baseline.",
        ]
        if results.task == "regression":
            points.append("R-squared can be negative on test data when a model fits worse than predicting the mean. Lower MAE/RMSE is better.")
    return [Paragraph("8. Limitations and interpretation", styles["h1"]), *_bullets(points, styles)]


def _section_attribution(styles: dict) -> list[Any]:
    return [
        Spacer(1, 1.2 * cm),
        Paragraph(_t(f"Generated using {APP_NAME}"), ParagraphStyle("a_app", parent=styles["subtitle"], fontName="Helvetica-Bold", fontSize=9.5, textColor=INK, spaceAfter=2)),
        Paragraph(_t(get_report_developer_text()), ParagraphStyle("a_dev", parent=styles["subtitle"], fontName="Helvetica", fontSize=8.5, textColor=MUTED, spaceAfter=2)),
        Paragraph(_t(get_copyright_text()), ParagraphStyle("a_cpr", parent=styles["subtitle"], fontName="Helvetica", fontSize=8, textColor=MUTED, spaceAfter=0)),
    ]


# --------------------------------------------------------------------- entry point
def build_pdf_report(
    df: pd.DataFrame,
    dataset_name: str,
    version_label: str,
    cleaning_log: list[dict[str, Any]] | None = None,
    ml_results: MLResults | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Build the report and return the PDF as bytes."""
    if df is None or df.shape[1] == 0:
        raise ValueError("Cannot build a report without a dataset.")
    styles = _styles()
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    story: list[Any] = [
        Spacer(1, 3 * cm),
        Paragraph(_t(f"{APP_NAME} — Data Analysis Report"), ParagraphStyle("t", parent=styles["title"], alignment=TA_CENTER)),
        Paragraph(_t(f"Dataset: {dataset_name}"), styles["subtitle"]),
        Paragraph(_t(f"Data version: {version_label}"), styles["subtitle"]),
        Paragraph(_t(f"Generated on {stamp}"), styles["subtitle"]),
        Spacer(1, 1 * cm),
        Paragraph(_t("This report was produced locally from the data you loaded. Every number is computed from the dataset "
                     "or from the model results shown; nothing is estimated or invented."), styles["note"]),
        PageBreak(),
    ]
    story += _section_dataset(df, dataset_name, version_label, styles)
    story += _section_quality(df, styles)
    story += _section_cleaning(cleaning_log or [], styles)
    story += _section_insights(df, styles)
    story += _section_ml(ml_results, styles)
    story += _section_limits(ml_results, styles)
    story += _section_attribution(styles)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"{APP_NAME} report - {dataset_name}", author=f"{APP_NAME} - {AUTHOR_NAME}")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def save_report(pdf: bytes, dataset_name: str, directory: Path) -> Path:
    """Write the PDF into ``directory`` (created if needed) and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"datapilot_report_{safe_filename(dataset_name)}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    path.write_bytes(pdf)
    return path
