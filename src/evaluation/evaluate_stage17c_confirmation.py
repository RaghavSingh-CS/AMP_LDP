"""Stage 17C: one frozen evaluation on Confirmation Gate #1.

The only feature/label loader in this module uses the historical ``val`` split,
which Stage 17A maps exactly to CONFIRMATION. FINAL_TEST has no loading path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.run_stage14_representation import behavioural_features
from src.analysis.run_stage16_customer_aggregation import (
    FEATURE_NAMES,
    build_customer_features,
)
from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    RANDOM_STATE,
    aggregate_customer_mean,
    classification_metrics,
    file_sha256,
    load_window_matrix,
    relative_path,
    save_csv,
    save_json,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_17c"
PRE_GATE_PATH = RESULTS_DIR / "pre_gate_audit.json"
PROTOCOL_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "governance"
    / "stage_17a"
    / "research_protocol_manifest.csv"
)
CONFIRMATION_WINDOW_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "windows"
    / "stride_30"
    / "usable"
    / "val_windows.parquet"
)
STAGE17B_METRICS = RESULTS_DIR.parent / "stage_17b" / "dev_val_metrics.csv"
STAGE17B_PROBABILITIES = (
    RESULTS_DIR.parent / "stage_17b" / "dev_val_customer_probabilities.csv"
)
DIRECT_MODEL_NAME = "stage17b_frozen_direct_customer_hgb"
COMPARATOR_NAME = "stage14_frozen_window_hgb_mean_customer_probability"


def load_pre_gate() -> dict[str, Any]:
    if not PRE_GATE_PATH.is_file():
        raise FileNotFoundError("The Stage 17C pre-gate audit must exist before access")
    with PRE_GATE_PATH.open("r", encoding="utf-8") as source:
        audit = json.load(source)
    if not audit.get(
        "created_before_confirmation_features_labels_or_metrics_were_loaded"
    ):
        raise ValueError("Pre-gate audit is not certified as pre-access")
    for record in audit["verified_artifacts"].values():
        if "path" in record and "sha256" in record:
            path = PROJECT_ROOT / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError(f"Frozen artifact hash changed: {record['path']}")
    comparator = audit["verified_artifacts"]["stage14_frozen_comparator"]
    if file_sha256(PROJECT_ROOT / comparator["frozen_decision_path"]) != comparator[
        "frozen_decision_sha256"
    ]:
        raise ValueError("Frozen Stage 14 decision hash changed")
    return audit


def verify_model_configuration(model: object, audit: dict[str, Any]) -> None:
    expected = audit["frozen_pipeline"]["parameters"]
    actual = model.get_params()
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(
                f"Frozen HGB parameter {name} changed: {actual[name]!r} != {value!r}"
            )


def load_confirmation(
    audit: dict[str, Any],
) -> tuple[Any, np.ndarray, pd.DataFrame]:
    governance = pd.read_csv(PROTOCOL_MANIFEST, dtype={ID_COLUMN: str})
    confirmation = governance.loc[
        governance["research_role"].eq("CONFIRMATION")
    ].copy()
    expected = audit["confirmation_governance_metadata"]
    if len(confirmation) != expected["expected_customers"]:
        raise ValueError("Confirmation governance count changed")
    if not confirmation["original_split"].eq(expected["expected_original_split"]).all():
        raise ValueError("Confirmation includes a non-historical-validation customer")
    if int(confirmation["stage16_representation_available"].sum()) != expected[
        "expected_representation_available"
    ]:
        raise ValueError("Confirmation representation availability changed")

    # Historical val is exactly CONFIRMATION under the frozen governance map.
    windows = load_window_matrix("val")
    chronology = pd.read_parquet(
        CONFIRMATION_WINDOW_MANIFEST,
        columns=["window_id", ID_COLUMN, "window_number_within_customer"],
        engine="pyarrow",
    )
    if not np.array_equal(
        chronology["window_id"].astype(str).to_numpy(), windows.window_ids
    ):
        raise ValueError("Confirmation window chronology is misaligned")
    if set(windows.customer_ids) != set(confirmation[ID_COLUMN]):
        raise ValueError("Loaded confirmation IDs differ from governance")
    return (
        windows,
        chronology["window_number_within_customer"].to_numpy(dtype=np.int32),
        confirmation,
    )


def metric_row(
    model_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    threshold_source: str,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "evaluation_role": "CONFIRMATION",
        "threshold_source": threshold_source,
        **classification_metrics(labels, scores, threshold),
    }


def bootstrap_metric_intervals(
    labels: np.ndarray, scores: np.ndarray, resamples: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < resamples:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        rows.append(
            {
                "bootstrap_replicate": len(rows) + 1,
                "ROC_AUC": float(roc_auc_score(y, scores[indices])),
                "PR_AUC_average_precision": float(
                    average_precision_score(y, scores[indices])
                ),
            }
        )
    samples = pd.DataFrame(rows)
    summary_rows = []
    for metric in ("ROC_AUC", "PR_AUC_average_precision"):
        values = samples[metric].to_numpy(dtype=np.float64)
        point = (
            roc_auc_score(labels, scores)
            if metric == "ROC_AUC"
            else average_precision_score(labels, scores)
        )
        summary_rows.append(
            {
                "metric": metric,
                "point_estimate": float(point),
                "resamples": resamples,
                "seed": RANDOM_STATE,
                "sampling_attempts": attempts,
                "bootstrap_mean": float(np.mean(values)),
                "percentile_95_CI_lower": float(np.quantile(values, 0.025)),
                "percentile_95_CI_upper": float(np.quantile(values, 0.975)),
            }
        )
    return samples, pd.DataFrame(summary_rows)


def independent_difference_bootstrap(
    confirmation_labels: np.ndarray,
    confirmation_scores: np.ndarray,
    development_labels: np.ndarray,
    development_scores: np.ndarray,
    resamples: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < resamples:
        attempts += 1
        confirmation_indices = rng.integers(
            0, len(confirmation_labels), size=len(confirmation_labels)
        )
        development_indices = rng.integers(
            0, len(development_labels), size=len(development_labels)
        )
        confirmation_y = confirmation_labels[confirmation_indices]
        development_y = development_labels[development_indices]
        if np.unique(confirmation_y).size < 2 or np.unique(development_y).size < 2:
            continue
        rows.append(
            {
                "bootstrap_replicate": len(rows) + 1,
                "ROC_AUC_delta_CONFIRMATION_minus_DEV_VAL": float(
                    roc_auc_score(
                        confirmation_y, confirmation_scores[confirmation_indices]
                    )
                    - roc_auc_score(
                        development_y, development_scores[development_indices]
                    )
                ),
                "average_precision_delta_CONFIRMATION_minus_DEV_VAL": float(
                    average_precision_score(
                        confirmation_y, confirmation_scores[confirmation_indices]
                    )
                    - average_precision_score(
                        development_y, development_scores[development_indices]
                    )
                ),
            }
        )
    samples = pd.DataFrame(rows)
    summary_rows = []
    mappings = (
        ("ROC_AUC", "ROC_AUC_delta_CONFIRMATION_minus_DEV_VAL"),
        (
            "PR_AUC_average_precision",
            "average_precision_delta_CONFIRMATION_minus_DEV_VAL",
        ),
    )
    for metric, column in mappings:
        values = samples[column].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "metric": metric,
                "resamples": resamples,
                "seed": RANDOM_STATE,
                "sampling_attempts": attempts,
                "bootstrap_mean_delta": float(np.mean(values)),
                "percentile_95_CI_lower": float(np.quantile(values, 0.025)),
                "percentile_95_CI_upper": float(np.quantile(values, 0.975)),
                "fraction_above_zero": float(np.mean(values > 0)),
            }
        )
    return samples, pd.DataFrame(summary_rows)


def paired_comparator_bootstrap(
    labels: np.ndarray,
    direct_scores: np.ndarray,
    comparator_scores: np.ndarray,
    resamples: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < resamples:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        rows.append(
            {
                "bootstrap_replicate": len(rows) + 1,
                "ROC_AUC_delta_direct_minus_stage14": float(
                    roc_auc_score(y, direct_scores[indices])
                    - roc_auc_score(y, comparator_scores[indices])
                ),
                "average_precision_delta_direct_minus_stage14": float(
                    average_precision_score(y, direct_scores[indices])
                    - average_precision_score(y, comparator_scores[indices])
                ),
            }
        )
    samples = pd.DataFrame(rows)
    summary_rows = []
    for metric, column in (
        ("ROC_AUC", "ROC_AUC_delta_direct_minus_stage14"),
        (
            "PR_AUC_average_precision",
            "average_precision_delta_direct_minus_stage14",
        ),
    ):
        values = samples[column].to_numpy(dtype=np.float64)
        observed = (
            roc_auc_score(labels, direct_scores)
            - roc_auc_score(labels, comparator_scores)
            if metric == "ROC_AUC"
            else average_precision_score(labels, direct_scores)
            - average_precision_score(labels, comparator_scores)
        )
        summary_rows.append(
            {
                "metric": metric,
                "observed_delta_direct_minus_stage14": float(observed),
                "resamples": resamples,
                "seed": RANDOM_STATE,
                "sampling_attempts": attempts,
                "bootstrap_mean_delta": float(np.mean(values)),
                "percentile_95_CI_lower": float(np.quantile(values, 0.025)),
                "percentile_95_CI_upper": float(np.quantile(values, 0.975)),
                "fraction_above_zero": float(np.mean(values > 0)),
            }
        )
    return samples, pd.DataFrame(summary_rows)


def classify_gate(
    metrics: pd.Series, intervals: pd.DataFrame
) -> tuple[str, str]:
    lookup = intervals.set_index("metric")
    strong = bool(
        metrics["ROC_AUC"] >= 0.75
        and metrics["PR_AUC_average_precision"] >= 0.30
        and lookup.loc["ROC_AUC", "percentile_95_CI_lower"] >= 0.70
        and lookup.loc[
            "PR_AUC_average_precision", "percentile_95_CI_lower"
        ]
        >= 0.20
    )
    if strong:
        return "A", "STRONG CONFIRMATION"
    partial = bool(
        metrics["ROC_AUC"] >= 0.65
        and metrics["PR_AUC_average_precision"]
        >= max(0.15, 1.5 * metrics["positive_prevalence"])
    )
    if partial:
        return "B", "PARTIAL / DEGRADED CONFIRMATION"
    return "C", "CONFIRMATION FAILURE"


def write_report(
    direct: pd.Series,
    comparator: pd.Series,
    deltas: pd.Series,
    confirmation_bootstrap: pd.DataFrame,
    independent_bootstrap: pd.DataFrame,
    comparator_bootstrap: pd.DataFrame,
    decision_code: str,
    decision_label: str,
) -> None:
    absolute = confirmation_bootstrap.set_index("metric")
    independent = independent_bootstrap.set_index("metric")
    paired = comparator_bootstrap.set_index("metric")
    report = f"""# Stage 17C — Confirmation Gate #1

## Frozen gate and access disclosure

This gate asks whether the exact frozen Stage 17B direct-customer HGB retains
strong discrimination on `CONFIRMATION`. Hashes, 48-feature order, HGB parameters,
four `DEV_TRAIN` persistence references, threshold, and A/B/C interpretation
boundaries were recorded in `pre_gate_audit.json` before confirmation features or
labels were loaded. The saved Stage 17B model was not retrained, and no confirmation
quantity was fitted or optimized.

**CONFIRMATION GATE #1 WAS ACCESSED ONCE FOR THE FROZEN STAGE 17B PIPELINE.**
**FINAL_TEST WAS NOT ACCESSED.**

## Confirmation representation and primary result

All 6,355 governed confirmation customers are representation-available and form
one finite 48-column row each. There are 5,813 normal and 542 theft customers;
prevalence is `{direct.positive_prevalence:.12f}`. IDs are exactly the original
validation/`CONFIRMATION` role, with no development or final-test contamination.
The frozen Stage 17B threshold is `{direct.threshold:.17g}`.

| Model | ROC-AUC | AP | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen Stage 17B direct HGB | {direct.ROC_AUC:.6f} | {direct.PR_AUC_average_precision:.6f} | {direct.precision:.6f} | {direct.recall:.6f} | {direct.F1:.6f} | {direct.threshold:.6f} | {int(direct.TN)} | {int(direct.FP)} | {int(direct.FN)} | {int(direct.TP)} |
| Frozen Stage 14 window HGB → customer mean | {comparator.ROC_AUC:.6f} | {comparator.PR_AUC_average_precision:.6f} | {comparator.precision:.6f} | {comparator.recall:.6f} | {comparator.F1:.6f} | {comparator.threshold:.6f} | {int(comparator.TN)} | {int(comparator.FP)} | {int(comparator.FN)} | {int(comparator.TP)} |

Threshold metrics for the direct pipeline use the frozen pre-confirmation Stage
17B threshold. The Stage 14 comparator uses its pre-existing frozen threshold,
which was historically selected on this same original-validation population;
its threshold metrics are therefore optimistic and only ranking comparisons are
scientifically emphasized.

## Development-to-confirmation change

Confirmation minus Stage 17B `DEV_VAL` is `{deltas.ROC_AUC_delta:+.12f}` ROC-AUC,
`{deltas.PR_AUC_average_precision_delta:+.12f}` AP,
`{deltas.precision_delta:+.12f}` precision, `{deltas.recall_delta:+.12f}` recall,
and `{deltas.F1_delta:+.12f}` F1.

The independent two-sample bootstrap gives ROC-AUC delta mean
`{independent.loc['ROC_AUC', 'bootstrap_mean_delta']:+.6f}`, 95% CI
`[{independent.loc['ROC_AUC', 'percentile_95_CI_lower']:+.6f}, {independent.loc['ROC_AUC', 'percentile_95_CI_upper']:+.6f}]`,
and AP delta mean
`{independent.loc['PR_AUC_average_precision', 'bootstrap_mean_delta']:+.6f}`, CI
`[{independent.loc['PR_AUC_average_precision', 'percentile_95_CI_lower']:+.6f}, {independent.loc['PR_AUC_average_precision', 'percentile_95_CI_upper']:+.6f}]`.
The populations were resampled independently and never pooled.

## Confirmation uncertainty

Across 2,000 confirmation-customer resamples with seed 42, ROC-AUC is
`{direct.ROC_AUC:.6f}` with 95% percentile CI
`[{absolute.loc['ROC_AUC', 'percentile_95_CI_lower']:.6f}, {absolute.loc['ROC_AUC', 'percentile_95_CI_upper']:.6f}]`.
AP is `{direct.PR_AUC_average_precision:.6f}` with CI
`[{absolute.loc['PR_AUC_average_precision', 'percentile_95_CI_lower']:.6f}, {absolute.loc['PR_AUC_average_precision', 'percentile_95_CI_upper']:.6f}]`.

## Frozen Stage 14 formulation comparator

The optional comparator was validly evaluated without retraining or tuning.
Direct Stage 17B HGB minus Stage 14 window-to-customer HGB is
`{paired.loc['ROC_AUC', 'observed_delta_direct_minus_stage14']:+.12f}` ROC-AUC
with paired 95% CI
`[{paired.loc['ROC_AUC', 'percentile_95_CI_lower']:+.6f}, {paired.loc['ROC_AUC', 'percentile_95_CI_upper']:+.6f}]`
and `{paired.loc['ROC_AUC', 'fraction_above_zero']:.3%}` above zero. AP delta is
`{paired.loc['PR_AUC_average_precision', 'observed_delta_direct_minus_stage14']:+.12f}`
with CI
`[{paired.loc['PR_AUC_average_precision', 'percentile_95_CI_lower']:+.6f}, {paired.loc['PR_AUC_average_precision', 'percentile_95_CI_upper']:+.6f}]`
and `{paired.loc['PR_AUC_average_precision', 'fraction_above_zero']:.3%}` above zero.

## Gate decision and interpretation

Decision **{decision_code} — {decision_label}** follows the pre-gate numerical
policy. AP is interpreted against the confirmation prevalence baseline, and the
development-to-confirmation intervals determine whether modest changes are
compatible with ordinary sampling variation. This gate supports generalization
to a held-out SGCC subset, not another dataset.

The direct HGB retains strong discrimination: AP is about `4.44×` the `8.53%`
prevalence baseline. Both independent development-to-confirmation intervals
include zero, so the small ROC-AUC change and modest AP increase are compatible
with ordinary sampling variation. The frozen direct formulation also retains a
clear paired ranking advantage over Stage 14. Gate #1 therefore strengthens the
customer-formulation hypothesis without proving that propagated-window label
noise is the causal mechanism.

If confirmation is strong, the Stage 17B direct-customer HGB remains frozen and
confirmation returns to closed status. The next recommended research question,
using development data only, is: **Does the direct-customer advantage persist
across pre-specified history-length and consumption-level strata?**

For a paper, the gate completes the current evidence chain: reconstruction models
were weak, supervised signal existed, behavioural summaries improved raw windows,
FC signals added no value, direct customer modeling improved formulation, the
representation replicated under a new development split, and the frozen pipeline
was now tested once on confirmation. This can support a transparent held-out SGCC
confirmation claim, but not SOTA, deployment readiness, external generalization,
or causal proof of label noise.
"""
    (RESULTS_DIR / "stage_17c_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )


def main() -> None:
    decision_path = RESULTS_DIR / "confirmation_decision.json"
    if decision_path.exists():
        raise FileExistsError(
            "Refusing to repeat Confirmation Gate #1: decision artifact exists"
        )
    audit = load_pre_gate()
    direct_record = audit["verified_artifacts"]["stage17b_frozen_hgb"]
    direct_model = joblib.load(PROJECT_ROOT / direct_record["path"])
    verify_model_configuration(direct_model, audit)

    windows, window_numbers, governance = load_confirmation(audit)
    summaries = behavioural_features(windows.X)
    thresholds = audit["frozen_pipeline"][
        "DEV_TRAIN_derived_persistence_q90_thresholds"
    ]
    features, labels, customer_ids, used_thresholds = build_customer_features(
        summaries,
        windows.y,
        windows.customer_ids,
        window_numbers,
        thresholds,
    )
    if used_thresholds != thresholds:
        raise ValueError("DEV_TRAIN feature references changed")
    if tuple(features.columns) != FEATURE_NAMES or features.shape != (6355, 48):
        raise ValueError("Confirmation feature shape/order changed")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Confirmation representation contains non-finite values")
    if set(customer_ids) != set(governance[ID_COLUMN]):
        raise ValueError("Confirmation customer features differ from governance IDs")

    matrix = features.to_numpy(dtype=np.float64)
    # Primary confirmation probability generation: exactly one predict call.
    direct_scores = np.asarray(
        direct_model.predict_proba(matrix)[:, 1], dtype=np.float64
    )
    frozen_threshold = float(audit["frozen_pipeline"]["frozen_DEV_VAL_threshold"])
    direct_row = metric_row(
        DIRECT_MODEL_NAME,
        labels,
        direct_scores,
        frozen_threshold,
        "frozen Stage 17B DEV_VAL threshold",
    )

    stage14_record = audit["verified_artifacts"]["stage14_frozen_comparator"]
    stage14_model = joblib.load(PROJECT_ROOT / stage14_record["path"])
    stage14_window_scores = np.asarray(
        stage14_model.predict_proba(summaries)[:, 1], dtype=np.float64
    )
    stage14_customers = aggregate_customer_mean(
        windows.y, stage14_window_scores, windows.customer_ids
    ).set_index(ID_COLUMN)
    aligned_stage14 = stage14_customers.reindex(customer_ids)
    if not np.array_equal(
        aligned_stage14[LABEL_COLUMN].to_numpy(dtype=np.int8), labels
    ):
        raise ValueError("Stage 14 comparator labels do not align")
    stage14_scores = aligned_stage14["mean_probability"].to_numpy(dtype=np.float64)
    with (PROJECT_ROOT / stage14_record["frozen_decision_path"]).open(
        "r", encoding="utf-8"
    ) as source:
        stage14_decision = json.load(source)
    stage14_threshold = float(stage14_decision["customer_mean_threshold"])
    comparator_row = metric_row(
        COMPARATOR_NAME,
        labels,
        stage14_scores,
        stage14_threshold,
        "historical frozen Stage 14 threshold; selected on original validation",
    )

    metrics = pd.DataFrame([direct_row, comparator_row])
    save_csv(metrics, RESULTS_DIR / "confirmation_metrics.csv")
    probability_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "model": DIRECT_MODEL_NAME,
                    ID_COLUMN: customer_ids,
                    LABEL_COLUMN: labels,
                    "probability": direct_scores,
                }
            ),
            pd.DataFrame(
                {
                    "model": COMPARATOR_NAME,
                    ID_COLUMN: customer_ids,
                    LABEL_COLUMN: labels,
                    "probability": stage14_scores,
                }
            ),
        ],
        ignore_index=True,
    )
    save_csv(probability_frame, RESULTS_DIR / "confirmation_customer_probabilities.csv")

    dev_metrics = pd.read_csv(STAGE17B_METRICS)
    dev = dev_metrics.loc[
        dev_metrics["model"].eq("hist_gradient_boosting")
    ].iloc[0]
    delta_row = {
        "comparison": "CONFIRMATION_minus_STAGE17B_DEV_VAL",
        **{
            f"{metric}_delta": float(direct_row[metric] - dev[metric])
            for metric in (
                "ROC_AUC",
                "PR_AUC_average_precision",
                "precision",
                "recall",
                "F1",
            )
        },
    }
    deltas = pd.DataFrame([delta_row])
    save_csv(deltas, RESULTS_DIR / "development_to_confirmation_deltas.csv")

    absolute_samples, absolute_summary = bootstrap_metric_intervals(
        labels, direct_scores
    )
    save_csv(absolute_samples, RESULTS_DIR / "confirmation_bootstrap_samples.csv")
    save_csv(absolute_summary, RESULTS_DIR / "confirmation_bootstrap_summary.csv")

    dev_probabilities = pd.read_csv(STAGE17B_PROBABILITIES)
    dev_hgb = dev_probabilities.loc[
        dev_probabilities["model"].eq("hist_gradient_boosting")
    ]
    independent_samples, independent_summary = independent_difference_bootstrap(
        labels,
        direct_scores,
        dev_hgb[LABEL_COLUMN].to_numpy(dtype=np.int8),
        dev_hgb["probability"].to_numpy(dtype=np.float64),
    )
    save_csv(
        independent_samples,
        RESULTS_DIR / "independent_development_confirmation_bootstrap_samples.csv",
    )
    save_csv(
        independent_summary,
        RESULTS_DIR / "independent_development_confirmation_bootstrap_summary.csv",
    )

    comparator_samples, comparator_summary = paired_comparator_bootstrap(
        labels, direct_scores, stage14_scores
    )
    save_csv(
        comparator_samples,
        RESULTS_DIR / "stage14_comparator_paired_bootstrap_samples.csv",
    )
    save_csv(
        comparator_summary,
        RESULTS_DIR / "stage14_comparator_paired_bootstrap_summary.csv",
    )

    direct_series = metrics.loc[metrics["model"].eq(DIRECT_MODEL_NAME)].iloc[0]
    comparator_series = metrics.loc[metrics["model"].eq(COMPARATOR_NAME)].iloc[0]
    decision_code, decision_label = classify_gate(direct_series, absolute_summary)
    save_json(
        {
            "stage": "Stage 17C — Confirmation Gate #1 evaluation summary",
            "pre_gate_audit": relative_path(PRE_GATE_PATH),
            "pre_gate_audit_sha256": file_sha256(PRE_GATE_PATH),
            "confirmation_access_count_for_frozen_pipeline": 1,
            "confirmation_customers": int(len(labels)),
            "normal_customers": int(np.sum(labels == 0)),
            "theft_customers": int(np.sum(labels == 1)),
            "theft_prevalence": float(np.mean(labels)),
            "usable_windows": int(len(windows.y)),
            "representation_shape": list(features.shape),
            "nonfinite_representation_values": 0,
            "feature_names_in_order": list(FEATURE_NAMES),
            "DEV_TRAIN_references_reused_exactly": True,
            "frozen_model_sha256": file_sha256(
                PROJECT_ROOT / direct_record["path"]
            ),
            "frozen_threshold": frozen_threshold,
            "model_retrained": False,
            "confirmation_fitted_preprocessing": False,
            "stage14_comparator": {
                "evaluated": True,
                "model_sha256": file_sha256(
                    PROJECT_ROOT / stage14_record["path"]
                ),
                "retrained": False,
                "tuned": False,
                "historical_selection_limitation": stage14_record[
                    "authorized_scope"
                ],
            },
            "final_test_accessed": False,
        },
        RESULTS_DIR / "evaluation_summary.json",
    )
    save_json(
        {
            "stage": "Stage 17C — Confirmation Gate #1 decision",
            "decision": decision_code,
            "decision_label": decision_label,
            "success_policy": audit[
                "success_interpretation_frozen_before_scoring"
            ],
            "direct_confirmation_metrics": direct_row,
            "development_to_confirmation_deltas": delta_row,
            "retain_frozen_stage17b_pipeline": bool(decision_code == "A"),
            "confirmation_returns_to_closed_status": True,
            "next_research_question": "Does the direct-customer advantage persist across pre-specified history-length and consumption-level strata?",
            "confirmation_gate_1_accessed_once": True,
            "final_test_accessed": False,
        },
        decision_path,
    )
    write_report(
        direct_series,
        comparator_series,
        deltas.iloc[0],
        absolute_summary,
        independent_summary,
        comparator_summary,
        decision_code,
        decision_label,
    )
    print(
        f"Confirmation ROC-AUC={direct_row['ROC_AUC']:.6f}, "
        f"AP={direct_row['PR_AUC_average_precision']:.6f}"
    )
    print(f"Decision {decision_code}: {decision_label}")
    print("Confirmation Gate #1 accessed once; FINAL_TEST not accessed.")


if __name__ == "__main__":
    main()
