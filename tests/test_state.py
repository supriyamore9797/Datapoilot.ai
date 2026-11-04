"""Workflow-state tests: original data is immutable, undo/reset are consistent, stale results are discarded.

``state`` functions accept a plain dict, so no Streamlit is needed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import cleaning, state
from src.ml_pipeline import train_models


@pytest.fixture
def store(classification_df) -> dict:
    s: dict = {}
    state.load_dataset(classification_df, "demo.csv", "upload", ["note"], s)
    return s


def _apply(store: dict, result) -> None:
    state.commit_cleaning(result.df, result.entry, store)


def test_load_dataset_keeps_independent_original_and_working_copies(store, classification_df):
    assert state.has_data(store)
    assert store["original_df"] is not store["working_df"]
    store["working_df"].loc[0, "x2"] = 12345.0  # even direct mutation of the working copy...
    assert store["original_df"].loc[0, "x2"] != 12345.0  # ...cannot reach the original
    assert classification_df.loc[0, "x2"] != 12345.0  # ...or the caller's DataFrame
    assert store["load_warnings"] == ["note"]


def test_cleaning_never_touches_original(store):
    original = store["original_df"].copy(deep=True)
    _apply(store, cleaning.drop_missing_rows(store["working_df"]))
    _apply(store, cleaning.fill_missing_constant(store["working_df"], ["label"], "x") if store["working_df"]["label"].isna().any()
           else cleaning.rename_columns(store["working_df"], {"x1": "feature_one"}))
    pd.testing.assert_frame_equal(store["original_df"], original)
    assert len(store["working_df"]) < len(original)


def test_version_label_and_data_version_track_changes(store):
    assert state.version_label(store) == "Original data"
    v0 = state.dataset_signature(store)
    _apply(store, cleaning.drop_missing_rows(store["working_df"]))
    assert state.version_label(store) == "Cleaned data (1 operation applied)"
    assert state.dataset_signature(store) != v0
    assert len(store["cleaning_log"]) == 1


def test_undo_restores_previous_data_and_pops_the_log(store):
    before = store["working_df"].copy()
    _apply(store, cleaning.drop_missing_rows(store["working_df"]))
    assert state.can_undo(store)
    undone = state.undo_last(store)
    assert undone["operation"] == "Drop rows with missing values"
    pd.testing.assert_frame_equal(store["working_df"], before)
    assert store["cleaning_log"] == []
    assert not state.can_undo(store)
    assert state.undo_last(store) is None  # nothing left to undo
    assert [e["event"] for e in store["audit_trail"]] == ["applied", "undone"]


def test_reset_restores_original_and_clears_history(store):
    original = store["original_df"].copy()
    _apply(store, cleaning.drop_missing_rows(store["working_df"]))
    _apply(store, cleaning.rename_columns(store["working_df"], {"x1": "renamed"}))
    state.reset_to_original(store)
    pd.testing.assert_frame_equal(store["working_df"], original)
    assert store["cleaning_log"] == [] and store["undo_stack"] == []
    assert store["audit_trail"][-1]["event"] == "reset"
    assert state.version_label(store) == "Original data"
    store["working_df"].loc[0, "x2"] = -99.0  # the reset copy must be independent too
    assert store["original_df"].loc[0, "x2"] != -99.0


def test_undo_stack_is_bounded():
    s: dict = {}
    state.load_dataset(pd.DataFrame({"a": range(50)}), "d", state=s)
    for i in range(10):
        df = s["working_df"]
        _apply(s, cleaning.CleaningResult(df.iloc[:-1].reset_index(drop=True), {"operation": f"op{i}", "columns": []}))
    from src.config import MAX_UNDO_STEPS

    assert len(s["undo_stack"]) == MAX_UNDO_STEPS


def test_model_results_are_tied_to_the_data_version(store):
    results = train_models(store["working_df"], "label", "classification", ["x1", "x2", "group"], ["Decision Tree"])
    state.store_ml_results(results, {"target": "label"}, store)
    assert state.current_ml_results(store) is results
    # Cleaning changes the data, so the old results must disappear instead of looking current.
    _apply(store, cleaning.drop_missing_rows(store["working_df"]))
    assert state.current_ml_results(store) is None
    assert "cleaned after the models were trained" in store["ml_stale_notice"]


def test_undo_and_reset_also_invalidate_results(store):
    _apply(store, cleaning.drop_missing_rows(store["working_df"]))
    results = train_models(store["working_df"], "label", "classification", ["x1", "x2"], ["Decision Tree"])
    state.store_ml_results(results, {}, store)
    state.undo_last(store)
    assert state.current_ml_results(store) is None
    state.store_ml_results(results, {}, store)  # (stored against the current signature)
    state.reset_to_original(store)
    assert state.current_ml_results(store) is None


def test_loading_a_new_dataset_discards_everything_from_the_old_one(store):
    results = train_models(store["working_df"], "label", "classification", ["x1", "x2"], ["Decision Tree"])
    state.store_ml_results(results, {}, store)
    old_id = store["dataset_id"]
    state.load_dataset(pd.DataFrame({"a": [1, 2, 3]}), "other.csv", "upload", state=store)
    assert store["dataset_id"] == old_id + 1
    assert state.current_ml_results(store) is None
    assert store["cleaning_log"] == [] and store["undo_stack"] == []
    assert store["dataset_name"] == "other.csv"


def test_report_is_only_current_when_data_and_results_match(store):
    state.store_report(b"%PDF-1.4 fake", "r.pdf", store)
    assert state.current_report(store) is not None
    results = train_models(store["working_df"], "label", "classification", ["x1", "x2"], ["Decision Tree"])
    state.store_ml_results(results, {}, store)  # new model results -> older report no longer matches
    assert state.current_report(store) is None
    state.store_report(b"%PDF-1.4 fake2", "r.pdf", store)
    assert state.current_report(store)["ml_present"] is True
    _apply(store, cleaning.drop_missing_rows(store["working_df"]))
    assert state.current_report(store) is None


def test_init_state_does_not_overwrite_existing_values(store):
    store["dataset_name"] = "keep-me.csv"
    state.init_state(store)
    assert store["dataset_name"] == "keep-me.csv"
