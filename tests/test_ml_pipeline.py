"""ML correctness tests: real training, metric verification, leakage-safe preprocessing, failure modes."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src import ml_pipeline as ml
from src.ml_pipeline import MLError, train_models

FEATURES = ["x1", "x2", "group"]


def _split(df, target, features, test_size=0.2, stratify=False, seed=42):
    """Reproduce the split train_models performs, so tests can recompute metrics independently."""
    labelled = df[df[target].notna()]
    y = labelled[target].astype(str) if stratify else labelled[target]
    return train_test_split(labelled[features], y, test_size=test_size, random_state=seed, stratify=y if stratify else None)


# ------------------------------------------------------------------ classification
def test_classification_trains_all_models_and_beats_baseline(classification_df):
    res = train_models(classification_df, "label", "classification", FEATURES)
    assert set(res.outcomes) == set(ml.CLASSIFICATION_MODELS)
    assert not res.failed
    assert res.best_model in res.outcomes
    assert len(res.comparison) == 3
    best = res.comparison.iloc[0]
    assert best["F1"] >= res.comparison["F1"].max() - 1e-12  # sorted by the selection metric
    assert best["Accuracy"] > res.baseline["Accuracy"]
    assert res.split["stratified"] is True
    for outcome in res.outcomes.values():
        assert outcome.confusion.to_numpy().sum() == res.split["test_rows"]
        assert {"precision", "recall", "f1-score", "support"} <= set(outcome.report.columns)


def test_classification_metrics_match_independent_recomputation(classification_df):
    res = train_models(classification_df, "label", "classification", FEATURES, ["Random Forest"], positive_class="yes")
    pipeline = res.outcomes["Random Forest"].pipeline
    _, X_test, _, y_test = _split(classification_df, "label", FEATURES, stratify=True)
    predicted = pipeline.predict(X_test)
    metrics = res.outcomes["Random Forest"].metrics
    assert metrics["Accuracy"] == pytest.approx(accuracy_score(y_test, predicted))
    assert metrics["F1"] == pytest.approx(f1_score(y_test, predicted, pos_label="yes", average="binary"))
    assert res.split["test_rows"] == len(y_test)
    assert "positive class 'yes'" in res.averaging_note


def test_multiclass_uses_weighted_averaging_and_says_so():
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["cls"] = np.select([df.a > 0.5, df.a < -0.5], ["hi", "lo"], "mid")
    res = train_models(df, "cls", "classification", ["a", "b"], ["Decision Tree"])
    assert "weighted" in res.averaging_note
    assert res.outcomes["Decision Tree"].confusion.shape == (3, 3)


def test_class_imbalance_is_flagged_and_balanced_weights_accepted():
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame({"a": rng.normal(size=n)})
    df["y"] = np.where(df.a + rng.normal(0, 0.3, n) > 1.6, "rare", "common")
    assert (df.y == "rare").mean() < 0.2
    res = train_models(df, "y", "classification", ["a"], ["Logistic Regression"], balance_classes=True)
    assert any("imbalance" in w.lower() for w in res.warnings)
    assert res.config["balance_classes"] is True


# --------------------------------------------------------------------- regression
def test_regression_trains_and_metrics_are_correct(regression_df):
    res = train_models(regression_df, "y", "regression", ["a", "b", "kind"], selection_metric="rmse")
    assert set(res.outcomes) == set(ml.REGRESSION_MODELS)
    assert res.comparison["RMSE"].is_monotonic_increasing  # lower is better, best first
    linear = res.outcomes["Linear Regression"]
    assert linear.metrics["R2"] > 0.9  # y is (nearly) linear in the features
    assert linear.metrics["R2"] > res.baseline["R2"]
    _, X_test, _, y_test = _split(regression_df, "y", ["a", "b", "kind"])
    predicted = linear.pipeline.predict(X_test)
    assert linear.metrics["MAE"] == pytest.approx(mean_absolute_error(y_test, predicted))
    assert linear.metrics["R2"] == pytest.approx(r2_score(y_test, predicted))
    assert linear.metrics["RMSE"] == pytest.approx(np.sqrt(np.mean((y_test - predicted) ** 2)))
    assert linear.metrics["RMSE"] >= linear.metrics["MAE"]


def test_r2_selection_metric_picks_highest_r2(regression_df):
    res = train_models(regression_df, "y", "regression", ["a", "b", "kind"], selection_metric="r2")
    assert res.comparison["R2"].is_monotonic_decreasing
    assert res.best_model == res.comparison.iloc[0]["Model"]


def test_negative_r2_is_possible_and_reported_honestly():
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"noise": rng.normal(size=400), "y": rng.normal(size=400)})
    res = train_models(df, "y", "regression", ["noise"], ["Decision Tree"])
    assert res.outcomes["Decision Tree"].metrics["R2"] < 0.05  # a tree fitted to pure noise generalises no better than the mean
    assert "negative" in res.averaging_note


# --------------------------------------------------------- preprocessing / leakage
def test_missing_values_and_categoricals_are_handled_inside_the_pipeline(classification_df):
    assert classification_df[["x1", "group"]].isna().any().all()
    res = train_models(classification_df, "label", "classification", FEATURES, ["Logistic Regression"])
    pipeline = res.outcomes["Logistic Regression"].pipeline
    new_rows = pd.DataFrame({"x1": [np.nan, 0.2], "x2": [0.1, np.nan], "group": [None, "never-seen-category"]})
    predictions = pipeline.predict(new_rows)  # NaNs and an unseen category must not crash
    assert len(predictions) == 2
    encoder = pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    assert set(encoder.categories_[0]) == {"a", "b", "c"}


def test_preprocessing_statistics_come_from_training_rows_only(classification_df):
    res = train_models(classification_df, "label", "classification", FEATURES, ["Decision Tree"])
    pipeline = res.outcomes["Decision Tree"].pipeline
    X_train, X_test, _, _ = _split(classification_df, "label", FEATURES, stratify=True)
    imputer = pipeline.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    learned = dict(zip(["x1", "x2"], imputer.statistics_))
    assert learned["x1"] == pytest.approx(X_train["x1"].median())
    # ...and they differ from what the full data (including test rows) would have given.
    assert learned["x1"] != pytest.approx(classification_df["x1"].median(), abs=1e-12)
    assert len(X_test) == res.split["test_rows"]


def test_split_happens_before_fitting_and_sizes_add_up(classification_df):
    res = train_models(classification_df, "label", "classification", FEATURES, ["Decision Tree"], test_size=0.3)
    total = classification_df["label"].notna().sum()
    assert res.split["train_rows"] + res.split["test_rows"] == total
    assert res.split["test_rows"] == pytest.approx(0.3 * total, abs=1.5)


def test_target_is_never_used_as_a_feature(classification_df):
    res = train_models(classification_df, "label", "classification", FEATURES + ["label"], ["Decision Tree"])
    assert "label" not in res.plan.features
    assert "label" in res.plan.excluded


def test_obvious_leakage_columns_are_excluded_with_a_warning(regression_df):
    df = regression_df.copy()
    df["y_copy"] = df["y"] * 1.0  # identical value
    df["y_scaled"] = df["y"] * 2.5 + 1  # perfectly correlated, derived from the target
    res = train_models(df, "y", "regression", ["a", "b", "kind", "y_copy", "y_scaled"], ["Linear Regression"])
    assert "y_copy" not in res.plan.features and "y_scaled" not in res.plan.features
    assert "leakage" in res.plan.excluded["y_copy"]
    assert any("leakage" in w for w in res.warnings)
    assert res.outcomes["Linear Regression"].metrics["R2"] < 0.9999  # not a perfect score from a leaked column


def test_label_encoded_copy_of_a_classification_target_is_flagged(classification_df):
    df = classification_df.copy()
    df["label_code"] = (df["label"] == "yes").astype(int)
    plan = ml.plan_features(df, "label", None, "classification")
    assert "label_code" in plan.excluded and "leakage" in plan.excluded["label_code"]


def test_constant_identifier_and_high_cardinality_columns_are_excluded():
    rng = np.random.default_rng(2)
    n = 100
    df = pd.DataFrame({
        "good": rng.normal(size=n),
        "const": 1.0,
        "row_id": np.arange(n),
        "name": [f"person_{i}" for i in range(n)],
        "all_missing": np.nan,
    })
    df["y"] = df["good"] * 2 + rng.normal(0, 2.0, n)
    plan = ml.plan_features(df, "y", None, "regression")
    assert plan.features == ["good"]
    assert set(plan.excluded) == {"const", "row_id", "name", "all_missing"}


def test_datetime_columns_are_excluded_not_crashing(regression_df):
    df = regression_df.copy()
    df["when"] = pd.date_range("2024-01-01", periods=len(df))
    res = train_models(df, "y", "regression", ["a", "when"], ["Linear Regression"])
    assert res.plan.features == ["a"]
    assert "datetime" in res.plan.excluded["when"]


# ----------------------------------------------------------------- failure modes
def test_rows_with_missing_target_are_removed_and_reported(classification_df):
    df = classification_df.copy()
    df.loc[:19, "label"] = np.nan
    res = train_models(df, "label", "classification", FEATURES, ["Decision Tree"])
    assert res.split["dropped_missing_target"] == 20
    assert res.split["train_rows"] + res.split["test_rows"] == len(df) - 20
    assert any("missing target" in w for w in res.warnings)


def test_too_few_rows_is_a_clear_error(classification_df):
    with pytest.raises(MLError, match="at least"):
        train_models(classification_df.head(10), "label", "classification", FEATURES)


def test_single_class_target_is_rejected(classification_df):
    df = classification_df.copy()
    df["label"] = "yes"
    with pytest.raises(MLError, match="only one class"):
        train_models(df, "label", "classification", FEATURES)


def test_constant_regression_target_is_rejected(regression_df):
    df = regression_df.copy()
    df["y"] = 3.0
    with pytest.raises(MLError, match="constant"):
        train_models(df, "y", "regression", ["a", "b"])


def test_text_target_cannot_be_used_for_regression(classification_df):
    with pytest.raises(MLError, match="must be numeric"):
        train_models(classification_df, "label", "regression", ["x1", "x2"])


def test_continuous_target_cannot_be_used_for_classification(regression_df):
    with pytest.raises(MLError, match="classes"):
        train_models(regression_df, "y", "classification", ["a", "b"])


def test_all_missing_target_and_unknown_target(classification_df):
    df = classification_df.copy()
    df["label"] = np.nan
    with pytest.raises(MLError, match="no values"):
        train_models(df, "label", "classification", FEATURES)
    with pytest.raises(MLError, match="not found"):
        train_models(classification_df, "nope", "classification", FEATURES)


def test_invalid_settings_are_rejected(classification_df):
    with pytest.raises(MLError, match="Task must be"):
        train_models(classification_df, "label", "clustering", FEATURES)
    with pytest.raises(MLError, match="test share"):
        train_models(classification_df, "label", "classification", FEATURES, test_size=0.9)
    with pytest.raises(MLError, match="not available"):
        train_models(classification_df, "label", "classification", FEATURES, ["Linear Regression"])


def test_no_usable_features_is_explained(classification_df):
    df = classification_df[["label"]].assign(const=1)
    with pytest.raises(MLError, match="No usable feature columns"):
        train_models(df, "label", "classification")


def test_one_failing_model_does_not_stop_the_others(classification_df, monkeypatch):
    original = ml._make_estimator

    def flaky(name, *args, **kwargs):
        if name == "Decision Tree":
            raise RuntimeError("boom")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(ml, "_make_estimator", flaky)
    res = train_models(classification_df, "label", "classification", FEATURES)
    assert "Decision Tree" in res.failed and "boom" in res.failed["Decision Tree"]
    assert set(res.outcomes) == {"Logistic Regression", "Random Forest"}
    assert "Decision Tree" not in set(res.comparison["Model"])


def test_stratification_falls_back_when_a_class_is_too_small():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"a": rng.normal(size=60)})
    df["y"] = ["common"] * 59 + ["unique"]
    res = train_models(df, "y", "classification", ["a"], ["Decision Tree"])
    assert res.split["stratified"] is False
    assert any("stratified split was not possible" in w for w in res.warnings)


# ------------------------------------------------------ reproducibility / helpers
def test_results_are_reproducible_for_the_same_seed(classification_df):
    a = train_models(classification_df, "label", "classification", FEATURES, random_state=7)
    b = train_models(classification_df, "label", "classification", FEATURES, random_state=7)
    pd.testing.assert_frame_equal(a.comparison.drop(columns="Fit time (s)"), b.comparison.drop(columns="Fit time (s)"))
    c = train_models(classification_df, "label", "classification", FEATURES, random_state=8)
    assert not a.comparison.drop(columns="Fit time (s)").equals(c.comparison.drop(columns="Fit time (s)"))


def test_feature_importance_only_for_tree_models_and_sums_to_one(classification_df):
    res = train_models(classification_df, "label", "classification", FEATURES)
    assert res.outcomes["Logistic Regression"].importance is None
    forest = res.outcomes["Random Forest"].importance
    assert set(forest["Feature"]) <= set(FEATURES)  # one-hot columns are folded back into "group"
    assert forest["Importance"].sum() == pytest.approx(1.0)


def test_suggest_task():
    assert ml.suggest_task(pd.Series(["a", "b", "a"]))[0] == "classification"
    assert ml.suggest_task(pd.Series([0, 1, 1, 0, 1] * 4))[0] == "classification"
    assert ml.suggest_task(pd.Series(np.linspace(0, 100, 50)))[0] == "regression"
    assert ml.suggest_task(pd.Series(range(100)))[0] == "regression"


def test_exported_pipeline_round_trips_and_predicts_raw_rows(classification_df):
    res = train_models(classification_df, "label", "classification", FEATURES, ["Random Forest"])
    payload = joblib.load(io.BytesIO(ml.export_pipeline_bytes(res, "Random Forest")))
    assert payload["target"] == "label" and payload["features"] == FEATURES
    raw = classification_df[FEATURES].head(5)  # un-preprocessed rows, containing NaNs
    assert list(payload["pipeline"].predict(raw)) == list(res.outcomes["Random Forest"].pipeline.predict(raw))
    with pytest.raises(MLError):
        ml.export_pipeline_bytes(res, "Not trained")
