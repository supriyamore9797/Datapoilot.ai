"""Shared Streamlit helpers: theme, page header, empty states, metric cards, sidebar status.

All processing lives in the other ``src`` modules; this file only deals with presentation.
Custom CSS only styles elements created here (classes prefixed ``dp-``), so it does not depend
on Streamlit's internal markup.
"""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st
from pandas.api import types as pdt

from src import state
from src.config import APP_NAME, APP_TAGLINE, get_app_footer_text
from src.utils import human_bytes

_CSS = """
<style>
.dp-hero { border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.1rem 1.4rem; background: #F8F9FC; margin-bottom: 0.75rem; }
.dp-hero h1 { margin: 0; font-size: 1.8rem; color: #1F2937; padding: 0; }
.dp-hero p { margin: 0.3rem 0 0 0; color: #4B5563; font-size: 0.98rem; }
.dp-badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
            border: 1px solid #C7D2FE; background: #EEF2FF; color: #3730A3; margin-right: 6px; }
.dp-badge-warn { border-color: #FCD34D; background: #FFFBEB; color: #92400E; }
.dp-step { border: 1px solid #E5E7EB; border-radius: 10px; padding: 0.6rem 0.8rem; background: #FFFFFF; font-size: 0.92rem; }
.dp-step b { color: #4F46E5; }
.dp-footer { margin-top: 3.5rem; padding-top: 1rem; padding-bottom: 1.5rem; border-top: 1px solid #E5E7EB; text-align: center; color: #6B7280; font-size: 0.83rem; font-weight: 500; }
</style>
"""


def configure_page() -> None:
    """Call once, first, from the app entry point."""
    st.set_page_config(page_title=f"{APP_NAME} - {APP_TAGLINE}", page_icon="\U0001F4CA", layout="wide")


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` that Streamlit can always display (mixed-type object columns become text)."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            kinds = {type(v) for v in out[col].dropna().head(2000)}
            if len(kinds) > 1:
                out[col] = out[col].map(lambda v: v if pd.isna(v) else str(v))
    return out


def badge(text: str, warn: bool = False) -> str:
    return f'<span class="dp-badge{" dp-badge-warn" if warn else ""}">{escape(text)}</span>'


def setup_page(title: str, subtitle: str) -> None:
    """Common start of every page: state, CSS and header."""
    state.init_state()
    inject_css()
    st.markdown(f'<div class="dp-hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)


def data_badges() -> None:
    """Show which dataset and which data version (original / cleaned) the page is using."""
    if state.has_data():
        df = state.get_working()
        label = state.version_label()
        st.markdown(
            badge(f"Dataset: {st.session_state['dataset_name']}") + badge(f"{len(df):,} rows x {df.shape[1]} columns")
            + badge(label, warn=not label.startswith("Original")),
            unsafe_allow_html=True,
        )


def require_data() -> pd.DataFrame:
    """Return the working DataFrame, or show a helpful empty state and stop the page."""
    state.init_state()
    if not state.has_data():
        st.info("No dataset is loaded yet. Go to **Home**, upload a CSV/Excel file or load a built-in sample dataset, then come back here.")
        st.page_link("app.py", label="Go to Home", icon="\U0001F3E0")
        st.stop()
    data_badges()
    return state.get_working()


def metric_cards(items: list[tuple[str, Any, str | None]]) -> None:
    """A row of bordered metric cards: (label, value, optional caption)."""
    columns = st.columns(len(items))
    for column, (label, value, caption) in zip(columns, items):
        with column, st.container(border=True):
            st.metric(label, value)
            if caption:
                st.caption(caption)


def show_flash() -> None:
    """Show (once) a message stored by an earlier run, e.g. after a rerun."""
    flash = st.session_state.pop("flash", None)
    if flash:
        level, text = flash
        getattr(st, level, st.info)(text)


def set_flash(level: str, text: str) -> None:
    st.session_state["flash"] = (level, text)


def sidebar_status() -> None:
    """Dataset status shown in the sidebar on every page."""
    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        if state.has_data():
            df = state.get_working()
            st.caption(f"**{st.session_state['dataset_name']}**")
            st.caption(f"{len(df):,} rows x {df.shape[1]} columns")
            st.caption(state.version_label())
            st.caption(f"Memory: {human_bytes(int(df.memory_usage(deep=True).sum()))}")
        else:
            st.caption("No dataset loaded.")
        st.caption("Data stays on this computer. Work is kept in memory only and is lost when the session ends.")
        st.divider()
        st.caption(get_app_footer_text())


def show_footer() -> None:
    """Render a subtle, professional authorship footer at the bottom of the page."""
    st.markdown(f'<div class="dp-footer">{escape(get_app_footer_text())}</div>', unsafe_allow_html=True)


def is_numeric_series(series: pd.Series) -> bool:
    return pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series)
