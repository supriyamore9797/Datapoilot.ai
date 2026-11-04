"""Leakage-safe machine learning for classification and regression.

Method (the same for every model):

1. Rows with a missing target are removed (and counted).
2. Features are chosen; constant, all-missing, identifier-like, datetime, very-high-cardinality
   and target-leaking columns are excluded with a reason.
3. The data is split into train and test sets BEFORE any statistic is learned.
4. A scikit-learn ``Pipeline`` (type coercion -> imputation -> scaling / one-hot encoding -> model)
   is fitted on the training rows only, then applied unchanged to the held-out test rows.
5. All reported metrics are held-out test-set metrics. The test set is never used to choose
   hyper-parameters (the models use fixed, modest defaults).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.config import (
    IMBALANCE_MINORITY_SHARE,
    LARGE_TRAINING_ROWS,
    LEAKAGE_CORRELATION,
    MAX_CLASSES,
    MAX_OHE_CARDINALITY,
    MIN_ML_ROWS,
    RANDOM_STATE,
)
from src.profiling import column_kind
from src.utils import now_iso

CLASSIFICATION_MODELS = ("Logistic Regression", "Decision Tree", "Random Forest")
REGRESSION_MODELS = ("Linear Regression", "Decision Tree", "Random Forest")
CLASSIFICATION_METRICS = {"f1": "F1-score", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy"}
REGRESSION_METRICS = {"rmse": "RMSE (lower is better)", "mae": "MAE (lower is better)", "r2": "R² (higher is better)"}
MAX_PREDICTION_ROWS = 5000


class MLError(ValueError):
    """Raised with a user-friendly message when a model cannot be trained."""


# ------------------------------------------------------------------- data classes
@dataclass
class FeaturePlan:
    features: list[str]
    numeric: list[str]
    categorical: list[str]
    excluded: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelOutcome:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    fit_seconds: float
    confusion: pd.DataFrame | None = None
    report: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    importance: pd.DataFrame | None = None


@dataclass
class MLResults:
    task: str
    target: str
    config: dict[str, Any]
    plan: FeaturePlan
    outcomes: dict[str, ModelOutcome]
    failed: dict[str, str]
    comparison: pd.DataFrame
    best_model: str | None
    selection_metric: str
    selection_note: str
    averaging_note: str
    baseline: dict[str, float]
    split: dict[str, Any]
    warnings: list[str]
    class_distribution: pd.DataFrame | None
    created: str = field(default_factory=now_iso)


# ----------------------------------------------------------------- target helpers
def suggest_task(series: pd.Series) -> tuple[str, str]:
    """Suggest ``classification`` or ``regression`` for a target column, with the reason."""
    non_null = series.dropna()
    kind = column_kind(series)
    if kind in ("categorical", "boolean"):
        return "classification", "The target is categorical (text or boolean), so it holds class labels."
    if kind == "datetime":
        return "regression", "Datetime targets are not supported; convert the column first."
    unique = int(non_null.nunique())
    is_integral = bool(len(non_null)) and bool((non_null.astype("float64") % 1 == 0).all())
    if is_integral and unique <= 10:
        return "classification", f"The target is numeric but has only {unique} whole-number values, so it looks like class labels."
    return "regression", f"The target is numeric with {unique} distinct values, so it looks continuous."


def _prepare_target(y: pd.Series, task: str) -> pd.Series:
    if task == "classification":
        if column_kind(y) == "numeric" and bool((y.astype("float64") % 1 == 0).all()):
            y = y.astype("int64")
        return y.astype(str)
    if column_kind(y) != "numeric":
        raise MLError("A regression target must be numeric. Choose classification, or convert the column to a number first.")
    y = y.astype("float64").replace([np.inf, -np.inf], np.nan)
    if y.isna().any():
        raise MLError("The target contains infinite values. Clean the data first.")
    return y


def class_labels(series: pd.Series) -> list[str]:
    """The class labels (as text, sorted) that training will use for a classification target."""
    return sorted(_prepare_target(series.dropna(), "classification").unique())


# --------------------------------------------------------------------- leakage
def detect_leakage(df: pd.DataFrame, target: str, columns: list[str], task: str) -> dict[str, str]:
    """Flag columns that are obviously derived from the target. Returns {column: reason}."""
    flagged: dict[str, str] = {}
    y = df[target]
    for col in columns:
        if col == target:
            continue
        x = df[col]
        pair = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
        if len(pair) < 10:
            continue
        if pair["x"].astype(str).equals(pair["y"].astype(str)):
            flagged[col] = "identical to the target"
            continue
        if column_kind(x) == "numeric" and column_kind(y) == "numeric" and pair["x"].nunique() > 1 and pair["y"].nunique() > 1:
            r = float(pair["x"].astype("float64").corr(pair["y"].astype("float64")))
            if abs(r) >= LEAKAGE_CORRELATION:
                flagged[col] = f"almost perfectly correlated with the target (r = {r:.3f}); it is probably derived from it"
                continue
        if task == "classification" and column_kind(x) == "numeric" and pair["y"].nunique() == 2:
            codes = pd.factorize(pair["y"].astype(str), sort=True)[0]
            if pair["x"].nunique() > 1:
                r = float(np.corrcoef(pair["x"].astype("float64"), codes)[0, 1])
                if abs(r) >= LEAKAGE_CORRELATION:
                    flagged[col] = f"almost perfectly tracks the binary target (r = {r:.3f})"
                    continue
        nx, ny = pair["x"].nunique(), pair["y"].nunique()
        if 2 <= nx <= MAX_OHE_CARDINALITY and nx == ny:
            per_x = pair.groupby(pair["x"].astype(str))["y"].nunique().max()
            per_y = pair.groupby(pair["y"].astype(str))["x"].nunique().max()
            if per_x == 1 and per_y == 1:
                flagged[col] = "a one-to-one relabelling of the target (each value maps to exactly one target value)"
    return flagged


def plan_features(df: pd.DataFrame, target: str, requested: list[str] | None, task: str) -> FeaturePlan:
    """Decide which columns can be used as features, and why others are excluded."""
    excluded: dict[str, str] = {}
    candidates = [c for c in (requested if requested else df.columns) if c in df.columns]
    if requested and target in requested:
        excluded[target] = "this is the target column"
    candidates = [c for c in candidates if c != target]
    rows = len(df)
    usable: list[str] = []
    for col in candidates:
        s = df[col]
        kind = column_kind(s)
        unique = int(s.nunique(dropna=True))
        if s.notna().sum() == 0:
            excluded[col] = "every value is missing"
        elif unique <= 1:
            excluded[col] = "constant column (no information)"
        elif kind == "datetime":
            excluded[col] = "datetime columns are not supported as features (convert to numbers first)"
        elif kind in ("categorical", "boolean") and unique > MAX_OHE_CARDINALITY:
            excluded[col] = f"too many distinct categories ({unique}) for one-hot encoding"
        elif rows >= 20 and unique == int(s.notna().sum()) and kind != "boolean" and not pd.api.types.is_float_dtype(s):
            excluded[col] = "every value is unique, so it looks like an identifier"
        else:
            usable.append(col)
    for col, reason in detect_leakage(df, target, usable, task).items():
        excluded[col] = f"possible target leakage: {reason}"
        usable.remove(col)
    numeric = [c for c in usable if column_kind(df[c]) == "numeric"]
    categorical = [c for c in usable if c not in numeric]
    return FeaturePlan(features=usable, numeric=numeric, categorical=categorical, excluded=excluded)


# -------------------------------------------------------------------- pipelines
def coerce_features(X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Normalise dtypes (no statistics are learned here): numbers become float, categories become text."""
    out = X.copy()
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)
    for col in categorical:
        series = out[col]
        out[col] = series.astype(object).where(series.notna(), np.nan).map(str, na_action="ignore")
    return out[numeric + categorical]


def _make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(estimator: Any, plan: FeaturePlan, scale_numeric: bool) -> Pipeline:
    """coerce -> (impute [+ scale] numerics | impute + one-hot categoricals) -> model."""
    transformers = []
    if plan.numeric:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), plan.numeric))
    if plan.categorical:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", _make_encoder())]), plan.categorical))
    return Pipeline([
        ("coerce", FunctionTransformer(coerce_features, kw_args={"numeric": plan.numeric, "categorical": plan.categorical})),
        ("prep", ColumnTransformer(transformers, remainder="drop")),
        ("model", estimator),
    ])


def _make_estimator(name: str, task: str, random_state: int, balance: bool, large: bool) -> tuple[Any, bool]:
    """Return (estimator, needs_scaling)."""
    weight = "balanced" if balance and task == "classification" else None
    trees = 50 if large else 100
    if task == "classification":
        if name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, class_weight=weight, random_state=random_state), True
        if name == "Decision Tree":
            return DecisionTreeClassifier(max_depth=6, class_weight=weight, random_state=random_state), False
        if name == "Random Forest":
            return RandomForestClassifier(n_estimators=trees, max_depth=12, class_weight=weight, random_state=random_state, n_jobs=-1), False
    else:
        if name == "Linear Regression":
            return LinearRegression(), False
        if name == "Decision Tree":
            return DecisionTreeRegressor(max_depth=6, random_state=random_state), False
        if name == "Random Forest":
            return RandomForestRegressor(n_estimators=trees, max_depth=12, random_state=random_state, n_jobs=-1), False
    raise MLError(f"Unknown model '{name}' for {task}.")


# --------------------------------------------------------------------- metrics
def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: Any, y_pred: Any, labels: list[str], positive: str | None) -> dict[str, float]:
    binary = len(labels) == 2
    kwargs: dict[str, Any] = {"average": "binary", "pos_label": positive} if binary else {"average": "weighted"}
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, labels=labels, zero_division=0, **kwargs)),
        "Recall": float(recall_score(y_true, y_pred, labels=labels, zero_division=0, **kwargs)),
        "F1": float(f1_score(y_true, y_pred, labels=labels, zero_division=0, **kwargs)),
        "Balanced accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _feature_importance(pipeline: Pipeline, plan: FeaturePlan) -> pd.DataFrame | None:
    """Impurity-based importance for tree models, summed back to the original columns."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    prep: ColumnTransformer = pipeline.named_steps["prep"]
    owners: list[str] = list(plan.numeric)
    if plan.categorical:
        encoder = prep.named_transformers_["cat"].named_steps["onehot"]
        for col, cats in zip(plan.categorical, encoder.categories_):
            owners += [col] * len(cats)
    values = np.asarray(model.feature_importances_)
    if len(owners) != len(values):
        return None
    grouped = pd.Series(values, index=owners).groupby(level=0).sum().sort_values(ascending=False)
    return pd.DataFrame({"Feature": grouped.index, "Importance": grouped.to_numpy()})


# ------------------------------------------------------------------- main entry
def train_models(
    df: pd.DataFrame,
    target: str,
    task: str,
    features: list[str] | None = None,
    models: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    balance_classes: bool = False,
    selection_metric: str | None = None,
    positive_class: str | None = None,
) -> MLResults:
    """Train and evaluate the selected models with a leakage-safe pipeline."""
    if task not in ("classification", "regression"):
        raise MLError("Task must be 'classification' or 'regression'.")
    if target not in df.columns:
        raise MLError(f"Target column '{target}' was not found.")
    if not 0.1 <= test_size <= 0.5:
        raise MLError("The test share must be between 10% and 50%.")
    available = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    chosen = list(models) if models else list(available)
    bad = [m for m in chosen if m not in available]
    if bad:
        raise MLError(f"Model(s) not available for {task}: {', '.join(bad)}.")

    warnings: list[str] = []
    labelled = df[df[target].notna()]
    dropped = len(df) - len(labelled)
    if labelled.empty:
        raise MLError("The target column has no values.")
    if dropped:
        warnings.append(f"{dropped} row(s) with a missing target were removed before training.")
    if len(labelled) < MIN_ML_ROWS:
        raise MLError(f"Only {len(labelled)} usable rows; at least {MIN_ML_ROWS} are needed for a train/test split.")

    y_all = _prepare_target(labelled[target], task)
    class_distribution = None
    labels: list[str] = []
    if task == "classification":
        counts = y_all.value_counts()
        if len(counts) < 2:
            raise MLError("The target has only one class; classification needs at least two.")
        if len(counts) > MAX_CLASSES:
            raise MLError(f"The target has {len(counts)} classes (limit {MAX_CLASSES}). It may be continuous; try regression.")
        labels = sorted(counts.index)
        class_distribution = pd.DataFrame({"Class": counts.index, "Rows": counts.to_numpy(),
                                           "Share %": (100 * counts / counts.sum()).round(2).to_numpy()})
        if counts.min() / counts.sum() < IMBALANCE_MINORITY_SHARE:
            warnings.append(f"Class imbalance: the smallest class is only {100 * counts.min() / counts.sum():.1f}% of rows. "
                            "Accuracy alone can look good while the minority class is missed; check F1, recall and balanced accuracy.")
    elif y_all.nunique() < 2:
        raise MLError("The target is constant; there is nothing to predict.")

    plan = plan_features(labelled, target, features, task)
    if not plan.features:
        reasons = "; ".join(f"{c}: {r}" for c, r in plan.excluded.items()) or "no candidate columns"
        raise MLError(f"No usable feature columns remain. Excluded: {reasons}.")
    leaked = {c: r for c, r in plan.excluded.items() if r.startswith("possible target leakage")}
    if leaked:
        warnings.append("Excluded to avoid target leakage: " + "; ".join(f"{c} ({r.split(': ', 1)[1]})" for c, r in leaked.items()) + ".")

    X = labelled[plan.features]
    n_test = int(np.ceil(test_size * len(X)))
    stratify = None
    stratified = False
    if task == "classification":
        counts = y_all.value_counts()
        if counts.min() >= 2 and n_test >= len(counts) and len(X) - n_test >= len(counts):
            stratify, stratified = y_all, True
        else:
            warnings.append("A stratified split was not possible (a class has fewer than 2 rows or the test set is too small); "
                            "a plain random split was used, so a class may be missing from the test set.")
    X_train, X_test, y_train, y_test = train_test_split(X, y_all, test_size=test_size, random_state=random_state, stratify=stratify)
    large = len(X_train) > LARGE_TRAINING_ROWS
    if large:
        warnings.append("Large dataset: random forests use 50 trees instead of 100 to keep training fast.")
    if len(y_test) < 30:
        warnings.append(f"The test set has only {len(y_test)} rows, so the metrics are noisy.")

    positive = None
    if task == "classification" and len(labels) == 2:
        positive = str(positive_class) if positive_class is not None and str(positive_class) in labels else labels[-1]
    if task == "classification":
        averaging = (f"Binary target: precision, recall and F1 are computed for the positive class '{positive}'." if positive else
                     "Multiclass target: precision, recall and F1 use weighted averaging (each class counts in proportion to its number of test rows).")
    else:
        averaging = "Regression: MAE and RMSE are in the target's units (lower is better). R² can be negative on test data if a model predicts worse than the mean."

    outcomes: dict[str, ModelOutcome] = {}
    failed: dict[str, str] = {}
    for name in chosen:
        try:
            estimator, scale = _make_estimator(name, task, random_state, balance_classes, large)
            pipeline = build_pipeline(estimator, plan, scale)
            started = time.perf_counter()
            pipeline.fit(X_train, y_train)
            seconds = time.perf_counter() - started
            predicted = pipeline.predict(X_test)
            outcome = ModelOutcome(name=name, pipeline=pipeline, metrics={}, fit_seconds=seconds)
            if task == "classification":
                outcome.metrics = classification_metrics(y_test, predicted, labels, positive)
                matrix = confusion_matrix(y_test, predicted, labels=labels)
                outcome.confusion = pd.DataFrame(matrix, index=[f"Actual: {c}" for c in labels],
                                                 columns=[f"Predicted: {c}" for c in labels])
                report = classification_report(y_test, predicted, labels=labels, output_dict=True, zero_division=0)
                outcome.report = pd.DataFrame(report).T.rename_axis("Class").reset_index()
            else:
                outcome.metrics = regression_metrics(y_test, predicted)
                frame = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": predicted})
                frame["Residual"] = frame["Actual"] - frame["Predicted"]
                outcome.predictions = frame.head(MAX_PREDICTION_ROWS)
            outcome.importance = _feature_importance(pipeline, plan)
            outcomes[name] = outcome
        except Exception as exc:  # recorded and shown to the user; other models continue
            failed[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else 'training failed'}"

    if not outcomes:
        raise MLError("No model could be trained. " + "; ".join(f"{n}: {r}" for n, r in failed.items()))

    if task == "classification":
        metric = selection_metric if selection_metric in CLASSIFICATION_METRICS else "f1"
        column = {"f1": "F1", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy"}[metric]
        higher_is_better, label = True, CLASSIFICATION_METRICS[metric]
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        baseline = classification_metrics(y_test, dummy.predict(X_test), labels, positive)
    else:
        metric = selection_metric if selection_metric in REGRESSION_METRICS else "rmse"
        column = {"rmse": "RMSE", "mae": "MAE", "r2": "R2"}[metric]
        higher_is_better, label = metric == "r2", REGRESSION_METRICS[metric]
        dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
        baseline = regression_metrics(y_test, dummy.predict(X_test))

    rows = [{"Model": n, **{k: round(v, 4) for k, v in o.metrics.items()}, "Fit time (s)": round(o.fit_seconds, 3)}
            for n, o in outcomes.items()]
    comparison = pd.DataFrame(rows).sort_values(column, ascending=not higher_is_better, kind="stable", ignore_index=True)
    best = str(comparison.iloc[0]["Model"])
    note = (f"Best model chosen by {label} on the held-out test set. "
            "Results depend on this dataset and this particular split; do not treat small differences as meaningful.")
    split = {"train_rows": len(X_train), "test_rows": len(X_test), "test_size": test_size, "stratified": stratified,
             "dropped_missing_target": dropped, "random_state": random_state}
    config = {"target": target, "task": task, "features": plan.features, "models": chosen, "test_size": test_size,
              "random_state": random_state, "balance_classes": balance_classes, "selection_metric": metric,
              "positive_class": positive, "sklearn_version": sklearn.__version__}
    return MLResults(task, target, config, plan, outcomes, failed, comparison, best, metric, note, averaging,
                     baseline, split, warnings, class_distribution)


# ----------------------------------------------------------------------- export
def export_pipeline_bytes(results: MLResults, model_name: str) -> bytes:
    """Serialise the fitted pipeline (preprocessing + model) and metadata with joblib.

    SECURITY: joblib files are pickles. Only load files you created yourself or fully trust;
    loading an untrusted file can execute arbitrary code.
    """
    if model_name not in results.outcomes:
        raise MLError(f"Model '{model_name}' was not trained.")
    payload = {
        "pipeline": results.outcomes[model_name].pipeline,
        "model_name": model_name,
        "target": results.target,
        "task": results.task,
        "features": results.plan.features,
        "metrics_on_test": results.outcomes[model_name].metrics,
        "sklearn_version": sklearn.__version__,
        "created": results.created,
    }
    buffer = BytesIO()
    joblib.dump(payload, buffer)
    return buffer.getvalue()
