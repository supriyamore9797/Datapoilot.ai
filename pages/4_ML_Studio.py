"""ML Studio page: train, compare and inspect models with a leakage-safe pipeline."""

import streamlit as st

from src import ml_pipeline as ml
from src import state, ui, visualization
from src.ml_pipeline import MLError

ui.setup_page("ML Studio", "Pick a target, train models and compare them on a held-out test set. Preprocessing is fitted on training data only.")
df = ui.require_data()
settings = st.session_state.get("ml_settings", {})

if st.session_state.get("ml_stale_notice"):
    st.info(st.session_state["ml_stale_notice"] + " Train again to get results for the current data.")

# ----------------------------------------------------------------------- 1. target
st.subheader("1. Target and task")
columns = list(df.columns)
default_target = settings.get("target") if settings.get("target") in columns else columns[-1]
target = st.selectbox("Target column (what you want to predict)", columns, index=columns.index(default_target), key="ml_target")
suggested, reason = ml.suggest_task(df[target])
st.caption(f"Suggestion: **{suggested}**. {reason}")
task = st.radio("Task", ["classification", "regression"], index=["classification", "regression"].index(suggested), horizontal=True, key=f"ml_task_{target}")

# --------------------------------------------------------------------- 2. features
st.subheader("2. Features")
labelled = df[df[target].notna()]
plan = ml.plan_features(labelled, target, None, task)
chosen = st.multiselect("Feature columns (defaults are the safe columns)", [c for c in columns if c != target],
                        default=plan.features, key=f"ml_features_{target}_{task}")
final_plan = ml.plan_features(labelled, target, chosen, task) if chosen else plan
dropped = {**plan.excluded, **final_plan.excluded}
if dropped:
    with st.expander(f"{len(dropped)} column(s) will not be used - see why"):
        for col, why in dropped.items():
            st.markdown(f"- **{col}**: {why}")
st.caption(f"{len(final_plan.features)} feature(s) will be used: {', '.join(final_plan.features) or 'none'}.")

# ----------------------------------------------------------------------- 3. models
st.subheader("3. Models and settings")
available = ml.CLASSIFICATION_MODELS if task == "classification" else ml.REGRESSION_MODELS
models = st.multiselect("Models to train", available, default=list(available), key=f"ml_models_{task}")
c1, c2, c3 = st.columns(3)
test_pct = c1.slider("Test set share (%)", 10, 40, 20, step=5, key="ml_test")
metrics = ml.CLASSIFICATION_METRICS if task == "classification" else ml.REGRESSION_METRICS
metric = c2.selectbox("Choose the best model by", list(metrics), format_func=metrics.get, key=f"ml_metric_{task}")
balance, positive = False, None
if task == "classification":
    balance = c3.checkbox("Balance class weights", help="Gives rare classes more weight during training. Useful for imbalanced targets.", key="ml_balance")
    classes = ml.class_labels(labelled[target])
    if len(classes) == 2:
        positive = st.selectbox("Positive class (used for precision, recall and F1)", classes, index=1, key="ml_pos")
    elif len(classes) > ml.MAX_CLASSES:
        st.warning(f"The target has {len(classes)} distinct values, which is a lot for classification. Consider regression.")

if st.button("Train models", type="primary", disabled=not models or not final_plan.features):
    try:
        with st.spinner("Training models..."):
            results = ml.train_models(df, target, task, final_plan.features, models, test_pct / 100, balance_classes=balance,
                                      selection_metric=metric, positive_class=positive)
        state.store_ml_results(results, {"target": target, "task": task})
        st.rerun()
    except MLError as exc:
        st.error(str(exc))

# ----------------------------------------------------------------------- results
results = state.current_ml_results()
if results is None:
    st.info("No results yet. Configure the options above and click **Train models**.")
    st.stop()

st.divider()
st.subheader("Results - held-out test performance")
st.caption(f"Target '{results.target}' ({results.task}), trained {results.created}. Train rows: {results.split['train_rows']:,}; "
           f"test rows: {results.split['test_rows']:,}{' (stratified split)' if results.split['stratified'] else ''}.")
for warning in results.warnings:
    st.warning(warning)
for name, why in results.failed.items():
    st.error(f"{name} could not be trained: {why}")

st.dataframe(results.comparison, hide_index=True)
st.success(f"Best model: **{results.best_model}**. {results.selection_note}")
st.caption(results.averaging_note)
baseline = ", ".join(f"{k} {v:.3f}" for k, v in results.baseline.items())
st.caption(f"Baseline ({'always predict the majority class' if results.task == 'classification' else 'always predict the training mean'}): {baseline}. "
           "A useful model should clearly beat this baseline.")
if results.class_distribution is not None:
    with st.expander("Class distribution of the target"):
        st.dataframe(results.class_distribution, hide_index=True)

st.subheader("Inspect a model")
names = list(results.outcomes)
chosen_model = st.selectbox("Model", names, index=names.index(results.best_model), key="ml_inspect")
outcome = results.outcomes[chosen_model]
ui.metric_cards([(k, f"{v:.4f}", None) for k, v in outcome.metrics.items()])

if results.task == "classification":
    left, right = st.columns(2)
    left.plotly_chart(visualization.confusion_matrix_figure(outcome.confusion))
    right.dataframe(outcome.report, hide_index=True)
    st.caption("Rows are the actual classes and columns the predicted classes. 'support' is the number of test rows per class.")
else:
    left, right = st.columns(2)
    left.plotly_chart(visualization.actual_vs_predicted_figure(outcome.predictions))
    right.plotly_chart(visualization.residuals_figure(outcome.predictions))
    st.caption("Lower MAE and RMSE are better. R² can be negative on test data when a model is worse than predicting the mean.")

if outcome.importance is not None:
    st.plotly_chart(visualization.importance_figure(outcome.importance))
    st.caption("Impurity-based importance from the training data. It can favour columns with many distinct values and does not prove cause and effect.")
else:
    st.caption("Feature importance is only shown for tree-based models; this model does not provide a comparable measure.")

st.subheader("Downloads")
st.download_button("Model comparison (CSV)", results.comparison.to_csv(index=False).encode("utf-8"), file_name="model_comparison.csv", mime="text/csv")
if outcome.confusion is not None:
    st.download_button("Confusion matrix (CSV)", outcome.confusion.to_csv().encode("utf-8"), file_name=f"confusion_{chosen_model}.csv", mime="text/csv")
    st.download_button("Classification report (CSV)", outcome.report.to_csv(index=False).encode("utf-8"), file_name=f"report_{chosen_model}.csv", mime="text/csv")
st.warning("Model files are Python pickles (joblib). Only load a model file you created yourself or fully trust: loading a file from an untrusted source can run arbitrary code.")
if st.checkbox("Prepare the fitted pipeline (preprocessing + model) for download", key="ml_prepare_export"):
    st.download_button("Download pipeline (.joblib)", ml.export_pipeline_bytes(results, chosen_model),
                       file_name=f"pipeline_{chosen_model.replace(' ', '_').lower()}.joblib", mime="application/octet-stream")

ui.show_footer()
