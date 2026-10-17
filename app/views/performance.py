"""Model performance presentation over immutable, validated evidence."""

import altair as alt
import pandas as pd
import streamlit as st

from app.config import COHORT_NAME
from app.data_loader import ArtifactError
from app.demo import aggregate_summary
from app.performance_loader import METRICS, MODEL_ORDER, absolute_differences, load_curves, load_performance

COLORS = ["#38a9d6", "#cf8d3a", "#8e79d6"]


def comparison_chart(frame: pd.DataFrame, metric: str) -> alt.Chart:
    return (alt.Chart(frame).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X(f"{metric}:Q", title=METRICS[metric], scale=alt.Scale(domain=[0, 1], nice=False)),
        y=alt.Y("model:N", title=None, sort=MODEL_ORDER),
        color=alt.Color("model:N", scale=alt.Scale(domain=MODEL_ORDER, range=COLORS), legend=None),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip(f"{metric}:Q", title=METRICS[metric], format=".6f"),
                 alt.Tooltip("customers:Q", title="Evaluation customers", format=",")],
    ).properties(height=180))


def curve_chart(frame: pd.DataFrame, *, pr: bool) -> alt.Chart:
    x, y = ("recall", "precision") if pr else ("false_positive_rate", "true_positive_rate")
    titles = {"recall": "Recall", "precision": "Precision", "false_positive_rate": "False positive rate", "true_positive_rate": "True positive rate"}
    # Inline values avoid Altair's 5,000-row DataFrame limit without sampling.
    return (alt.Chart(alt.Data(values=frame.to_dict("records")))
            .mark_line(color=COLORS[0], strokeWidth=2, interpolate="step-before" if pr else "linear")
            .encode(x=alt.X(f"{x}:Q", title=titles[x], scale=alt.Scale(domain=[0, 1], nice=False)),
                    y=alt.Y(f"{y}:Q", title=titles[y], scale=alt.Scale(domain=[0, 1], nice=False)),
                    order=alt.Order("point:Q"),
                    tooltip=[alt.Tooltip(f"{x}:Q", title=titles[x], format=".6f"),
                             alt.Tooltip(f"{y}:Q", title=titles[y], format=".6f")])
            .properties(height=290))


def render():
    st.header("Model Performance")
    st.write("Frozen customer-level evaluation · AMP-LDP and its preregistered baselines")
    try:
        evidence = load_performance()
    except ArtifactError as exc:
        st.error(f"Model performance unavailable: {exc}")
        st.info("Restore the reviewed evidence and catalog. No historical experiment is substituted.")
        return
    frozen = evidence.frozen
    left, right = st.columns(2)
    left.metric("AMP-LDP ROC-AUC", f"{frozen.roc_auc:.6f}")
    right.metric("AMP-LDP Average Precision", f"{frozen.average_precision:.6f}")
    counts = st.columns(3)
    for column, label, value in zip(counts, ["Evaluation customers", "Theft-labelled customers", "Theft-label prevalence"],
                                    [f"{frozen.customers:,}", f"{frozen.theft_customers:,}", f"{evidence.prevalence:.4%}"]):
        column.metric(label, value)
    st.caption(f"Stage 27B · {COHORT_NAME}. Positive evaluation label: FLAG = 1. Metrics displayed to six decimals; comparisons use full precision.")

    st.subheader("Frozen baseline comparison")
    st.write("All three methods were trained on DEV_TRAIN only and evaluated on the same 6,356 SGCC customers with the same customer-level ROC-AUC and Average Precision definitions.")
    st.dataframe(evidence.comparison.rename(columns={"model": "Model", "customers": "Customers", **METRICS}),
                 hide_index=True, width="stretch", column_config={
                     "ROC-AUC": st.column_config.NumberColumn(format="%.6f"),
                     "Average Precision": st.column_config.NumberColumn(format="%.6f")})
    columns = st.columns(2)
    for column, metric in zip(columns, METRICS):
        with column:
            st.altair_chart(comparison_chart(evidence.comparison, metric), width="stretch", theme="streamlit")
    st.caption("Both metric axes span 0–1. Higher is better. Bar lengths show the complete metric scale.")
    st.write("**Absolute differences in metric units**")
    st.dataframe(absolute_differences(evidence).rename(columns={"comparison": "Comparison", **METRICS}),
                 hide_index=True, width="stretch", column_config={
                     "ROC-AUC": st.column_config.NumberColumn(format="%+.6f"),
                     "Average Precision": st.column_config.NumberColumn(format="%+.6f")})
    st.caption("AMP-LDP minus each baseline; these are absolute metric differences, not relative percentage improvements. This view makes no statistical significance claim.")
    with st.expander("What the baselines measure and why these models are included"):
        st.write("**DIRECT:** six distribution statistics over 17 standardized window summaries produce 102 customer features, followed by the frozen Stage 25B HGB. It tests aggregation without learned encoding.")
        st.write("**FULL_48:** 48 engineered customer-profile features, including chronology and persistence, followed by the original frozen Stage 17B HGB. Its extra temporal information is an intended representation difference.")
        st.write("**AMP-LDP:** learned latent distributions with the frozen five-seed HGB ensemble. It has the highest observed ROC-AUC and AP among these three methods on this cohort.")
        st.write("FC/CNN/LSTM reconstruction experiments share historical source splits but use normal-window training, reconstruction-MSE scores and primarily window-level evaluation. Their customer aggregates are diagnostics under a different protocol, so they are excluded here. Stage 13 and Stage 24B exports also do not match this protocol. Stage 23B FULL_48 uses a different DEV_TRAIN+DEV_VAL refit and cannot substitute for the Stage 27B comparator.")

    st.subheader("Supported evaluation curves")
    curves = load_curves(evidence)
    if curves.roc is None:
        st.warning(curves.message)
    else:
        st.caption(curves.message)
        columns = st.columns(2)
        for column, title, frame, pr in ((columns[0], "AMP-LDP ROC", curves.roc, False),
                                          (columns[1], "AMP-LDP precision–recall", curves.pr, True)):
            with column:
                st.write(f"**{title}**")
                values = frame.assign(point=range(len(frame)))
                st.altair_chart(curve_chart(values, pr=pr), width="stretch", theme="streamlit")
        st.caption("Coordinates come solely from existing scores and labels. ROC-AUC and AP reproduce the frozen values exactly. The PR curve uses steps; AP is the non-interpolated sum of recall increments weighted by precision, not trapezoidal PR area.")
    st.info("FULL_48 and DIRECT curves are unavailable: no matching original Stage 27B per-customer predictions or curve artifacts were located. Their verified aggregate metrics are shown above.")
    st.download_button("Download aggregate evidence summary", aggregate_summary(evidence, curves),
                       file_name="amp-frozen-evaluation-summary.json", mime="application/json", on_click="ignore")
    st.caption("Local aggregate metrics, cohort disclosure, definitions and source hashes only. Customer identifiers, presenter examples and consumption histories are excluded.")

    st.subheader("How to interpret the evidence")
    st.write("**ROC-AUC** measures how well scores rank a theft-labelled customer above a normal-labelled customer across thresholds (ties receive half credit). It summarizes discrimination, not the fraction of customers correctly classified.")
    st.write("**Average Precision (AP)** summarizes precision–recall performance using sklearn's original average_precision_score definition. Precision asks how many selected customers are theft-labelled; recall asks how many theft-labelled customers are found. With only 8.5274% positive labels, AP helps assess retrieval when the positive class is uncommon. Prevalence describes this cohort; it is not a guaranteed achievable performance level.")
    st.write("**Ranking and operating decisions:** no applicable frozen AMP-LDP binary operating threshold and matching confusion matrix are established. This demonstration prioritizes risk ranking. No 0.5 cutoff or test-selected threshold is applied.")

    st.subheader("Frozen evaluation methodology")
    st.write("30-day non-overlapping consumption windows → 17 summary features → learned 17 → 32 → 16 encoder → six statistics per latent dimension → 96 customer features → five frozen HistGradientBoosting models → unweighted arithmetic mean of customer scores (seeds 42–46).")
    st.write("The six statistics are mean, stabilized population standard deviation, q10, median, q90 and maximum. Standardization references and fitted models come from DEV_TRAIN only. Learned representations encode consumption patterns; distribution aggregation summarizes their variation across a customer's windows. This representation does not establish causal explanations or confirmed theft days.")
    st.write("The encoder, aggregation, ensemble and comparator policy were frozen before the governed terminal evaluation. Stage 26 confirmation supported the frozen method before Stage 27B; this page does not make a new model selection.")
    with st.expander("Cohort disclosure and limitations", expanded=True):
        st.write(frozen.disclosure)
        st.write("Scores prioritize customers for review. They are not established as calibrated theft probabilities, and a high score is not proof of theft. Exact theft-day localization and causal attribution are not established. Results apply to this frozen SGCC cohort and protocol; independent external utility generalization and real-world operational validity have not been established.")
    with st.expander("Evidence provenance"):
        st.caption(f"Original evidence commit: {frozen.catalog['evidence_commit']}")
        st.write("Research text hashes normalize CRLF to LF. Derived curve files use LF bytes. Artifact hashes, schema, model identity, cohort, labels and frozen metric agreement are checked before display. Parsed immutable content is cached; current file hashes are checked on every load.")
        records = {**{key: frozen.catalog["artifacts"][key] for key in ("metrics", "integrity", "model", "protocol", "access")},
                   **{key: evidence.catalog[key] for key in ("comparators", "evaluator", "method_document", "historical_architectures")}}
        if curves.provenance:
            records.update({"curve_provenance": evidence.catalog["curves"], "curve_coordinates": curves.provenance["coordinates"],
                            "reconstructed_scores": curves.provenance["source_scores"], "score_provenance": curves.provenance["source_provenance"]})
        for label, record in records.items():
            st.text(f"{label}: {record['path']}")
            st.caption(f"SHA-256: {record['sha256']}")
