"""Centralised workflow state.

Every function takes an optional ``state`` mapping. In the app this is
``st.session_state``; the tests pass a plain ``dict`` so the logic can be verified
without Streamlit.

Design rules
------------
* ``original_df`` is set once when a dataset is loaded and never modified afterwards.
* ``working_df`` is replaced (never mutated) by each cleaning step.
* ``dataset_id`` changes when a new dataset is loaded; ``data_version`` changes whenever
  the working data changes. Model and report results remember both, so results that
  belong to older data are discarded instead of being shown as current.
* State lives in memory only: it is lost when the browser session or server ends.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pandas as pd

from src.config import MAX_UNDO_STEPS
from src.utils import now_iso


def _fresh_defaults() -> dict[str, Any]:
    return {
        "original_df": None,
        "working_df": None,
        "dataset_name": None,
        "dataset_source": None,
        "load_warnings": [],
        "dataset_id": 0,
        "data_version": 0,
        "cleaning_log": [],  # operations currently reflected in working_df
        "audit_trail": [],  # every event, including undo / reset
        "undo_stack": [],  # snapshots of working_df before each operation
        "ml_results": None,
        "ml_stale_notice": None,
        "ml_settings": {},
        "report_state": None,
    }


def _store(state: MutableMapping | None) -> MutableMapping:
    if state is not None:
        return state
    import streamlit as st  # imported lazily so tests do not need Streamlit

    return st.session_state


def init_state(state: MutableMapping | None = None) -> None:
    """Create any missing state keys without touching existing values."""
    store = _store(state)
    for key, value in _fresh_defaults().items():
        if key not in store:
            store[key] = value


def has_data(state: MutableMapping | None = None) -> bool:
    store = _store(state)
    df = store.get("working_df")
    return df is not None and not df.empty


def get_working(state: MutableMapping | None = None) -> pd.DataFrame | None:
    """Return the working DataFrame. Callers must treat it as read-only."""
    return _store(state).get("working_df")


def get_original(state: MutableMapping | None = None) -> pd.DataFrame | None:
    """Return the untouched original DataFrame. Callers must treat it as read-only."""
    return _store(state).get("original_df")


def load_dataset(
    df: pd.DataFrame,
    name: str,
    source: str = "upload",
    warnings: list[str] | None = None,
    state: MutableMapping | None = None,
) -> None:
    """Install a new dataset and invalidate everything that depended on the old one."""
    store = _store(state)
    init_state(store)
    next_id = int(store.get("dataset_id", 0)) + 1
    store.update(_fresh_defaults())
    store["original_df"] = df.copy()
    store["working_df"] = df.copy()
    store["dataset_name"] = name
    store["dataset_source"] = source
    store["load_warnings"] = list(warnings or [])
    store["dataset_id"] = next_id


def version_label(state: MutableMapping | None = None) -> str:
    """Human-readable label of the data version currently in use."""
    store = _store(state)
    count = len(store.get("cleaning_log", []))
    if count == 0:
        return "Original data"
    return f"Cleaned data ({count} operation{'s' if count != 1 else ''} applied)"


def dataset_signature(state: MutableMapping | None = None) -> tuple[int, int]:
    store = _store(state)
    return int(store.get("dataset_id", 0)), int(store.get("data_version", 0))


def invalidate_ml(reason: str, state: MutableMapping | None = None) -> None:
    """Drop model and report results that belong to older data."""
    store = _store(state)
    had_results = store.get("ml_results") is not None or store.get("report_state") is not None
    store["ml_results"] = None
    store["report_state"] = None
    if had_results:
        store["ml_stale_notice"] = reason


def commit_cleaning(
    new_df: pd.DataFrame,
    entry: dict[str, Any],
    state: MutableMapping | None = None,
) -> None:
    """Make ``new_df`` the working data and record the operation."""
    store = _store(state)
    stack: list[pd.DataFrame] = store["undo_stack"]
    stack.append(store["working_df"])
    del stack[:-MAX_UNDO_STEPS]
    store["working_df"] = new_df
    store["cleaning_log"].append(entry)
    store["audit_trail"].append({**entry, "event": "applied"})
    store["data_version"] += 1
    invalidate_ml("The data was cleaned after the models were trained, so the old results were cleared.", store)


def can_undo(state: MutableMapping | None = None) -> bool:
    return bool(_store(state).get("undo_stack"))


def undo_last(state: MutableMapping | None = None) -> dict[str, Any] | None:
    """Undo the most recent cleaning operation. Returns the undone log entry."""
    store = _store(state)
    if not store["undo_stack"] or not store["cleaning_log"]:
        return None
    store["working_df"] = store["undo_stack"].pop()
    undone = store["cleaning_log"].pop()
    store["audit_trail"].append(
        {"timestamp": now_iso(), "operation": f"Undo: {undone['operation']}", "event": "undone", "columns": undone.get("columns", [])}
    )
    store["data_version"] += 1
    invalidate_ml("A cleaning step was undone after the models were trained, so the old results were cleared.", store)
    return undone


def reset_to_original(state: MutableMapping | None = None) -> None:
    """Restore the working data to a copy of the original and clear cleaning history."""
    store = _store(state)
    if store.get("original_df") is None:
        return
    store["working_df"] = store["original_df"].copy()
    store["cleaning_log"] = []
    store["undo_stack"] = []
    store["audit_trail"].append({"timestamp": now_iso(), "operation": "Reset to original", "event": "reset", "columns": []})
    store["data_version"] += 1
    invalidate_ml("The data was reset to the original, so the old model results were cleared.", store)


def store_ml_results(results: Any, settings: dict[str, Any], state: MutableMapping | None = None) -> None:
    """Keep model results together with the data signature they were produced from."""
    store = _store(state)
    store["ml_results"] = {"signature": dataset_signature(store), "results": results}
    store["ml_settings"] = dict(settings)
    store["ml_stale_notice"] = None
    store["report_state"] = None


def current_ml_results(state: MutableMapping | None = None) -> Any | None:
    """Return model results only if they match the current dataset and data version."""
    store = _store(state)
    payload = store.get("ml_results")
    if not payload or payload["signature"] != dataset_signature(store):
        return None
    return payload["results"]


def store_report(pdf_bytes: bytes, file_name: str, state: MutableMapping | None = None) -> None:
    store = _store(state)
    store["report_state"] = {
        "signature": dataset_signature(store),
        "ml_present": current_ml_results(store) is not None,
        "pdf": pdf_bytes,
        "file_name": file_name,
        "created": now_iso(),
    }


def current_report(state: MutableMapping | None = None) -> dict[str, Any] | None:
    store = _store(state)
    report = store.get("report_state")
    if not report or report["signature"] != dataset_signature(store):
        return None
    if report["ml_present"] != (current_ml_results(store) is not None):
        return None
    return report
