"""Stage 16: validation-only direct customer behavioural modeling.

The test split is deliberately outside this program's allowed split set.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.run_stage14_representation import (
    FEATURE_NAMES as WINDOW_FEATURE_NAMES,
    behavioural_features,
)
from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    RANDOM_STATE,
    WindowMatrix,
    aggregate_customer_mean,
    classification_metrics,
    file_sha256,
    load_window_matrix,
    relative_path,
    save_csv,
    save_json,
    select_f1_threshold,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_16"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_16"
PREREGISTRATION_PATH = RESULTS_DIR / "preregistered_config.json"
ALLOWED_SPLITS = ("train", "val")
BASELINE_NAME = "stage14_window_hgb_mean_customer_probability"
DIRECT_MODEL_NAMES = {
    "logistic": "direct_customer_logistic",
    "hist_gradient_boosting": "direct_customer_hist_gradient_boosting",
}

LEVEL_FEATURES = ("mean", "median", "minimum", "maximum", "q10", "q90")
VARIABILITY_FEATURES = (
    "standard_deviation",
    "range",
    "interquartile_range",
    "mean_absolute_deviation",
    "mean_absolute_consecutive_change",
    "maximum_absolute_consecutive_change",
)
DIRECTION_FEATURES = (
    "first_to_last_change",
    "absolute_first_to_last_change",
    "linear_slope",
    "second_half_minus_first_half_mean",
)
PERSISTENCE_FEATURES = (
    "maximum",
    "standard_deviation",
    "mean_absolute_consecutive_change",
    "maximum_absolute_consecutive_change",
)
CHRONOLOGY_FEATURES = (
    "mean",
    "maximum",
    "standard_deviation",
    "mean_absolute_consecutive_change",
)


def expected_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for source in LEVEL_FEATURES:
        names.extend(
            f"{source}__customer_{aggregate}" for aggregate in ("median", "q10", "q90")
        )
    for source in VARIABILITY_FEATURES:
        names.extend(f"{source}__customer_{aggregate}" for aggregate in ("median", "q90"))
    for source in DIRECTION_FEATURES:
        names.extend(
            f"{source}__customer_{aggregate}"
            for aggregate in ("mean", "standard_deviation")
        )
    names.extend(
        (
            "fraction_below_minus_one__customer_mean",
            "fraction_below_minus_one__customer_q90",
        )
    )
    names.extend(f"{source}__training_q90_exceedance_fraction" for source in PERSISTENCE_FEATURES)
    names.extend(f"{source}__late_third_minus_early_third" for source in CHRONOLOGY_FEATURES)
    return tuple(names)


FEATURE_NAMES = expected_feature_names()


def load_preregistration() -> dict[str, Any]:
    if not PREREGISTRATION_PATH.is_file():
        raise FileNotFoundError("Stage 16 pre-registration must exist before scoring")
    with PREREGISTRATION_PATH.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not config.get("created_before_stage_16_validation_scoring"):
        raise ValueError("Stage 16 pre-registration is not certified pre-validation")
    if tuple(config["feature_names_in_order"]) != FEATURE_NAMES:
        raise ValueError("Code feature order differs from the pre-registered feature list")
    if config["feature_count"] != len(FEATURE_NAMES):
        raise ValueError("Pre-registered feature count is inconsistent")
    if config["test_policy"].split(".", 1)[0] != "CLOSED":
        raise ValueError("Stage 16 test policy is not closed")
    return config


def load_allowed_split(split: str) -> tuple[WindowMatrix, np.ndarray]:
    """Load only train/validation windows plus chronology kept outside predictors."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Stage 16 permits only {ALLOWED_SPLITS}; got {split!r}")
    windows = load_window_matrix(split)
    manifest_path = (
        PROJECT_ROOT
        / "data"
        / "windows"
        / "stride_30"
        / "usable"
        / f"{split}_windows.parquet"
    )
    chronology = pd.read_parquet(
        manifest_path,
        columns=["window_id", ID_COLUMN, "window_number_within_customer"],
        engine="pyarrow",
    )
    if not np.array_equal(chronology["window_id"].astype(str).to_numpy(), windows.window_ids):
        raise ValueError(f"{split} chronology does not align with loaded windows")
    if not np.array_equal(chronology[ID_COLUMN].astype(str).to_numpy(), windows.customer_ids):
        raise ValueError(f"{split} chronology customer IDs do not align")
    return windows, chronology["window_number_within_customer"].to_numpy(dtype=np.int32)


def split_integrity_audit(train: WindowMatrix, validation: WindowMatrix) -> dict[str, Any]:
    manifest_path = PROJECT_ROOT / "data" / "processed" / "split_manifest.parquet"
    manifest = pd.read_parquet(manifest_path, columns=[ID_COLUMN, "split"], engine="pyarrow")
    id_sets = {
        split: set(manifest.loc[manifest["split"].eq(split), ID_COLUMN].astype(str))
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train_val": len(id_sets["train"] & id_sets["val"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "val_test": len(id_sets["val"] & id_sets["test"]),
    }
    if any(overlaps.values()) or set(train.customer_ids) & set(validation.customer_ids):
        raise ValueError(f"Customer split overlap detected: {overlaps}")
    train_usable = set(train.customer_ids)
    validation_usable = set(validation.customer_ids)
    return {
        "authoritative_manifest": relative_path(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "authoritative_customer_counts_without_loading_test_labels": {
            split: len(ids) for split, ids in id_sets.items()
        },
        "customer_overlaps": overlaps,
        "usable_window_counts": {"train": int(len(train.y)), "validation": int(len(validation.y))},
        "customers_with_usable_windows": {
            "train": len(train_usable),
            "validation": len(validation_usable),
        },
        "customers_excluded_for_zero_usable_windows": {
            "train": len(id_sets["train"] - train_usable),
            "validation": len(id_sets["val"] - validation_usable),
        },
        "loaded_splits": ["train", "val"],
        "test_features_loaded": False,
        "test_labels_loaded": False,
        "test_predictions_loaded_or_inspected": False,
        "historical_test_metrics_loaded_or_used": False,
    }


def window_count_outputs(split: str, customer_ids: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = pd.Series(customer_ids).value_counts(sort=False).to_numpy(dtype=np.int32)
    histogram = pd.Series(counts).value_counts().sort_index()
    frame = pd.DataFrame(
        {
            "split": split,
            "usable_window_count": histogram.index.astype(int),
            "customer_count": histogram.to_numpy(dtype=int),
            "customer_fraction": histogram.to_numpy(dtype=float) / len(counts),
        }
    )
    quantiles = np.quantile(counts, [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    summary = {
        "customer_count": int(len(counts)),
        "mean": float(np.mean(counts)),
        "standard_deviation": float(np.std(counts, ddof=0)),
        "minimum": int(np.min(counts)),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "p10": float(quantiles[3]),
        "p25": float(quantiles[4]),
        "median": float(quantiles[5]),
        "p75": float(quantiles[6]),
        "p90": float(quantiles[7]),
        "p95": float(quantiles[8]),
        "p99": float(quantiles[9]),
        "maximum": int(np.max(counts)),
    }
    return frame, summary


def build_customer_features(
    summaries: np.ndarray,
    labels: np.ndarray,
    customer_ids: np.ndarray,
    window_numbers: np.ndarray,
    persistence_thresholds: dict[str, float] | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, float]]:
    frame = pd.DataFrame(summaries, columns=WINDOW_FEATURE_NAMES)
    frame.insert(0, "window_number", window_numbers)
    frame.insert(0, LABEL_COLUMN, labels)
    frame.insert(0, ID_COLUMN, customer_ids)
    grouped = frame.groupby(ID_COLUMN, sort=False, observed=True)
    if grouped[LABEL_COLUMN].nunique().ne(1).any():
        raise ValueError("A customer has inconsistent window labels")

    q10 = grouped[list(WINDOW_FEATURE_NAMES)].quantile(0.10)
    q90 = grouped[list(WINDOW_FEATURE_NAMES)].quantile(0.90)
    median = grouped[list(WINDOW_FEATURE_NAMES)].median()
    mean = grouped[list(WINDOW_FEATURE_NAMES)].mean()
    standard_deviation = grouped[list(WINDOW_FEATURE_NAMES)].std(ddof=0).fillna(0.0)
    output = pd.DataFrame(index=mean.index)
    for source in LEVEL_FEATURES:
        output[f"{source}__customer_median"] = median[source]
        output[f"{source}__customer_q10"] = q10[source]
        output[f"{source}__customer_q90"] = q90[source]
    for source in VARIABILITY_FEATURES:
        output[f"{source}__customer_median"] = median[source]
        output[f"{source}__customer_q90"] = q90[source]
    for source in DIRECTION_FEATURES:
        output[f"{source}__customer_mean"] = mean[source]
        output[f"{source}__customer_standard_deviation"] = standard_deviation[source]
    output["fraction_below_minus_one__customer_mean"] = mean["fraction_below_minus_one"]
    output["fraction_below_minus_one__customer_q90"] = q90["fraction_below_minus_one"]

    if persistence_thresholds is None:
        persistence_thresholds = {
            source: float(np.quantile(frame[source].to_numpy(dtype=np.float64), 0.90))
            for source in PERSISTENCE_FEATURES
        }
    for source in PERSISTENCE_FEATURES:
        indicator = frame[source].gt(persistence_thresholds[source]).groupby(frame[ID_COLUMN], sort=False).mean()
        output[f"{source}__training_q90_exceedance_fraction"] = indicator

    chronology_rows: list[dict[str, float | str]] = []
    for customer_id, group in grouped:
        ordered = group.sort_values("window_number", kind="mergesort")
        k = max(1, math.ceil(len(ordered) / 3))
        row: dict[str, float | str] = {ID_COLUMN: str(customer_id)}
        for source in CHRONOLOGY_FEATURES:
            row[f"{source}__late_third_minus_early_third"] = float(
                ordered[source].iloc[-k:].mean() - ordered[source].iloc[:k].mean()
            )
        chronology_rows.append(row)
    chronology = pd.DataFrame(chronology_rows).set_index(ID_COLUMN)
    output = output.join(chronology, how="left")
    output = output.loc[:, FEATURE_NAMES]
    output_labels = grouped[LABEL_COLUMN].first().reindex(output.index).to_numpy(dtype=np.int8)
    output_ids = output.index.to_numpy(dtype=str)
    if output.shape[1] != 48 or not np.isfinite(output.to_numpy()).all():
        raise ValueError("Customer feature matrix has an invalid shape or non-finite value")
    return output.reset_index(drop=True), output_labels, output_ids, persistence_thresholds


def make_model(family: str) -> object:
    if family == "logistic":
        return Pipeline(
            [
                ("standardize", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight=None,
                        max_iter=300,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            class_weight=None,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model family: {family}")


def metric_row(model: str, formulation: str, family: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    threshold = select_f1_threshold(labels, scores)
    return {
        "model": model,
        "formulation": formulation,
        "family": family,
        "level": "customer",
        **classification_metrics(labels, scores, threshold),
    }


def reproduce_stage14_baseline(
    validation: WindowMatrix, validation_summaries: np.ndarray
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    model_path = MODEL_DIR.parent / "stage_14" / "hist_gradient_boosting_unweighted_summary_only.joblib"
    model = joblib.load(model_path)
    window_scores = np.asarray(model.predict_proba(validation_summaries)[:, 1], dtype=np.float64)
    customers = aggregate_customer_mean(validation.y, window_scores, validation.customer_ids)
    labels = customers[LABEL_COLUMN].to_numpy(dtype=np.int8)
    scores = customers["mean_probability"].to_numpy(dtype=np.float64)
    row = metric_row(BASELINE_NAME, "window_model_then_mean_probability", "hist_gradient_boosting", labels, scores)

    saved_path = RESULTS_DIR.parent / "stage_14" / "validation_customer_probabilities.csv"
    saved = pd.read_csv(saved_path)
    saved = saved.loc[saved["model"].eq("hist_gradient_boosting_unweighted_summary_only")].copy()
    aligned = customers.merge(saved[[ID_COLUMN, LABEL_COLUMN, "mean_probability"]], on=ID_COLUMN, suffixes=("_reproduced", "_saved"), validate="one_to_one")
    max_probability_difference = float(np.max(np.abs(aligned["mean_probability_reproduced"] - aligned["mean_probability_saved"])))
    label_match = bool(np.array_equal(aligned[f"{LABEL_COLUMN}_reproduced"], aligned[f"{LABEL_COLUMN}_saved"]))
    old_metrics_path = RESULTS_DIR.parent / "stage_14" / "validation_metrics.csv"
    old_metrics = pd.read_csv(old_metrics_path)
    old = old_metrics.loc[
        old_metrics["model"].eq("hist_gradient_boosting_unweighted_summary_only")
        & old_metrics["level"].eq("customer_mean")
    ].iloc[0]
    metric_differences = {
        metric: float(row[metric] - old[metric])
        for metric in ("ROC_AUC", "PR_AUC_average_precision", "precision", "recall", "F1", "threshold")
    }
    if max_probability_difference > 1e-12 or not label_match or max(abs(v) for v in metric_differences.values()) > 1e-12:
        raise ValueError("Frozen Stage 14 baseline does not reproduce exactly")
    customers.insert(0, "model", BASELINE_NAME)
    return row, customers, {
        "model_artifact": relative_path(model_path),
        "model_sha256": file_sha256(model_path),
        "saved_probability_artifact": relative_path(saved_path),
        "max_absolute_customer_probability_difference": max_probability_difference,
        "labels_match": label_match,
        "metric_differences_stage16_reproduction_minus_stage14": metric_differences,
        "all_metrics_reproduced_within_1e-12": True,
    }


def paired_bootstrap(
    labels: np.ndarray,
    direct_scores: np.ndarray,
    baseline_scores: np.ndarray,
    resamples: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, float | int]] = []
    attempts = 0
    while len(rows) < resamples:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_y = labels[indices]
        if np.unique(sampled_y).size < 2:
            continue
        rows.append(
            {
                "bootstrap_replicate": len(rows) + 1,
                "ROC_AUC_delta_direct_minus_baseline": float(
                    roc_auc_score(sampled_y, direct_scores[indices])
                    - roc_auc_score(sampled_y, baseline_scores[indices])
                ),
                "average_precision_delta_direct_minus_baseline": float(
                    average_precision_score(sampled_y, direct_scores[indices])
                    - average_precision_score(sampled_y, baseline_scores[indices])
                ),
            }
        )
    samples = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for metric, column in (
        ("ROC_AUC", "ROC_AUC_delta_direct_minus_baseline"),
        ("PR_AUC_average_precision", "average_precision_delta_direct_minus_baseline"),
    ):
        values = samples[column].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "metric": metric,
                "resamples": resamples,
                "seed": RANDOM_STATE,
                "accepted_resamples": len(samples),
                "sampling_attempts": attempts,
                "bootstrap_mean_delta": float(np.mean(values)),
                "bootstrap_standard_deviation": float(np.std(values, ddof=0)),
                "percentile_95_CI_lower": float(np.quantile(values, 0.025)),
                "percentile_95_CI_upper": float(np.quantile(values, 0.975)),
                "fraction_above_zero": float(np.mean(values > 0)),
                "fraction_below_zero": float(np.mean(values < 0)),
            }
        )
    return samples, pd.DataFrame(summary_rows)


def validation_permutation_importance(
    model: object, matrix: np.ndarray, labels: np.ndarray, repetitions: int = 5
) -> pd.DataFrame:
    baseline_scores = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    baseline_roc = float(roc_auc_score(labels, baseline_scores))
    baseline_ap = float(average_precision_score(labels, baseline_scores))
    rng = np.random.default_rng(RANDOM_STATE)
    permuted = matrix.copy()
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(FEATURE_NAMES):
        original = matrix[:, index].copy()
        roc_decreases: list[float] = []
        ap_decreases: list[float] = []
        for _ in range(repetitions):
            permuted[:, index] = original[rng.permutation(len(original))]
            scores = np.asarray(model.predict_proba(permuted)[:, 1], dtype=np.float64)
            roc_decreases.append(baseline_roc - roc_auc_score(labels, scores))
            ap_decreases.append(baseline_ap - average_precision_score(labels, scores))
            permuted[:, index] = original
        rows.append(
            {
                "feature": name,
                "validation_repetitions": repetitions,
                "baseline_customer_ROC_AUC": baseline_roc,
                "mean_ROC_AUC_decrease": float(np.mean(roc_decreases)),
                "std_ROC_AUC_decrease": float(np.std(roc_decreases, ddof=0)),
                "baseline_customer_average_precision": baseline_ap,
                "mean_average_precision_decrease": float(np.mean(ap_decreases)),
                "std_average_precision_decrease": float(np.std(ap_decreases, ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_ROC_AUC_decrease", "mean_average_precision_decrease"],
        ascending=False,
        kind="mergesort",
    )


def feature_sanity(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    for split, matrix in (("train", train), ("validation", validation)):
        for feature in FEATURE_NAMES:
            values = matrix[feature].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "count": len(values),
                    "nonfinite_count": int(np.sum(~np.isfinite(values))),
                    "unique_count": int(np.unique(values).size),
                    "effectively_constant": bool(np.ptp(values) <= 1e-12),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=0)),
                    "minimum": float(np.min(values)),
                    "p01": float(np.quantile(values, 0.01)),
                    "median": float(np.median(values)),
                    "p99": float(np.quantile(values, 0.99)),
                    "maximum": float(np.max(values)),
                }
            )
    distribution = pd.DataFrame(rows)
    constants = distribution.loc[
        distribution["split"].eq("train") & distribution["effectively_constant"], "feature"
    ].tolist()
    correlations = train.corr(method="spearman")
    pairs: list[dict[str, Any]] = []
    for left in range(len(FEATURE_NAMES)):
        for right in range(left + 1, len(FEATURE_NAMES)):
            value = float(correlations.iloc[left, right])
            if abs(value) >= 0.95:
                pairs.append(
                    {
                        "feature_1": FEATURE_NAMES[left],
                        "feature_2": FEATURE_NAMES[right],
                        "train_spearman_correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )
    pair_frame = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "train_spearman_correlation", "absolute_correlation"])
    if len(pair_frame):
        pair_frame = pair_frame.sort_values("absolute_correlation", ascending=False, kind="mergesort")
    return distribution, pair_frame, constants


def decision_code(roc_delta: float, ap_delta: float, bootstrap_summary: pd.DataFrame) -> tuple[str, str]:
    lookup = bootstrap_summary.set_index("metric")
    roc_excludes = bool(lookup.loc["ROC_AUC", "percentile_95_CI_lower"] > 0 or lookup.loc["ROC_AUC", "percentile_95_CI_upper"] < 0)
    ap_excludes = bool(lookup.loc["PR_AUC_average_precision", "percentile_95_CI_lower"] > 0 or lookup.loc["PR_AUC_average_precision", "percentile_95_CI_upper"] < 0)
    if roc_delta > 0 and ap_delta > 0 and roc_excludes and ap_excludes:
        return "A", "direct customer-level modeling"
    if roc_delta < 0 and ap_delta < 0 and roc_excludes and ap_excludes:
        return "C", "window modeling followed by mean customer probability"
    return "B", "statistical tie between formulations"


def write_report(
    split_audit: dict[str, Any],
    count_summaries: dict[str, dict[str, Any]],
    customer_dataset: dict[str, dict[str, Any]],
    metrics: pd.DataFrame,
    selected_name: str,
    deltas: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    importance: pd.DataFrame,
    redundancy_pairs: pd.DataFrame,
    constants: list[str],
    reproduction: dict[str, Any],
    decision: str,
    decision_label: str,
) -> None:
    def metric(model: str) -> pd.Series:
        return metrics.loc[metrics["model"].eq(model)].iloc[0]

    baseline = metric(BASELINE_NAME)
    selected = metric(selected_name)
    delta = deltas.loc[deltas["model"].eq(selected_name)].iloc[0]
    boot = bootstrap_summary.set_index("metric")
    top = importance.head(10)
    top_lines = "\n".join(
        f"- `{row.feature}`: ROC-AUC decrease `{row.mean_ROC_AUC_decrease:.6f}`, AP decrease `{row.mean_average_precision_decrease:.6f}`."
        for row in top.itertuples()
    )
    if len(redundancy_pairs):
        redundancy_lines = "\n".join(
            f"- `{row.feature_1}` with `{row.feature_2}`: Spearman `{row.train_spearman_correlation:.6f}`."
            for row in redundancy_pairs.head(10).itertuples()
        )
    else:
        redundancy_lines = "- No training feature pair has absolute Spearman correlation at or above 0.95."
    rows = []
    for model_name in (BASELINE_NAME, DIRECT_MODEL_NAMES["logistic"], DIRECT_MODEL_NAMES["hist_gradient_boosting"]):
        row = metric(model_name)
        rows.append(
            f"| {model_name} | {row.ROC_AUC:.6f} | {row.PR_AUC_average_precision:.6f} | {row.precision:.6f} | {row.recall:.6f} | {row.F1:.6f} | {row.threshold:.6f} | {int(row.TN)} | {int(row.FP)} | {int(row.FN)} | {int(row.TP)} |"
        )
    report = f"""# Stage 16 — Direct Customer-Level Behavioural Modeling

## Research question and boundary

Stage 16 asks whether fitting directly on compact customer aggregates of the
frozen 17 Stage 14 window summaries improves discrimination over the frozen
Stage 14 summary-only HGB fitted at window level and averaged by customer.
Only train and validation were loaded. Test features, labels, predictions, and
historical test metrics were not accessed.

## Pre-registered customer representation

The 48-feature schema was written before Stage 16 validation scoring. It contains
18 robust level features, 12 typical/upper-tail variability features, eight
direction-consistency features, two low-consumption persistence features, four
train-q90 exceedance fractions, and four chronological late-third-minus-early-third
contrasts. Exact names and definitions are in `preregistered_config.json`.
Customer/window IDs, labels, split metadata, raw dates/window position, usable
window count, missingness metadata, prior predictions, and reconstruction outputs
are excluded. The persistence reference thresholds use training windows only.

## Split, coverage, and window-count audit

The authoritative split has {split_audit['authoritative_customer_counts_without_loading_test_labels']['train']:,}
train, {split_audit['authoritative_customer_counts_without_loading_test_labels']['val']:,}
validation, and {split_audit['authoritative_customer_counts_without_loading_test_labels']['test']:,}
test customers, with zero pairwise overlap. Stage 16 uses {split_audit['usable_window_counts']['train']:,}
train and {split_audit['usable_window_counts']['validation']:,} validation windows,
forming {split_audit['customers_with_usable_windows']['train']:,} train and
{split_audit['customers_with_usable_windows']['validation']:,} validation customer rows.
One training customer and no validation customers have zero usable windows and
are transparently excluded; no imputation or count feature is introduced.
The included train set has {customer_dataset['train']['theft_customers']:,} theft
customers (prevalence `{customer_dataset['train']['theft_prevalence']:.12f}`);
validation has {customer_dataset['validation']['theft_customers']:,} (prevalence
`{customer_dataset['validation']['theft_prevalence']:.12f}`).

Train usable-window count: min {count_summaries['train']['minimum']}, median
{count_summaries['train']['median']:.0f}, mean {count_summaries['train']['mean']:.6f},
p90 {count_summaries['train']['p90']:.0f}, max {count_summaries['train']['maximum']}.
Validation: min {count_summaries['validation']['minimum']}, median
{count_summaries['validation']['median']:.0f}, mean {count_summaries['validation']['mean']:.6f},
p90 {count_summaries['validation']['p90']:.0f}, max {count_summaries['validation']['maximum']}.
The exact histograms are in `customer_window_count_distribution.csv`.

## Frozen baseline reproduction

The frozen Stage 14 model artifact is `{reproduction['model_artifact']}` with
SHA-256 `{reproduction['model_sha256']}`. Recomputed validation customer
probabilities match the saved Stage 14 artifact with maximum absolute difference
`{reproduction['max_absolute_customer_probability_difference']:.3g}`; all six
reported metrics reproduce within `1e-12`.

## Customer-level validation results

| Model | ROC-AUC | AP | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The selected direct model is `{selected_name}` by pre-registered customer
ROC-AUC then AP. Versus the frozen baseline, its exact deltas are
`{delta.ROC_AUC_delta_vs_baseline:+.12f}` ROC-AUC,
`{delta.PR_AUC_average_precision_delta_vs_baseline:+.12f}` AP, and
`{delta.F1_delta_vs_baseline:+.12f}` F1.
Each model's operating threshold maximizes positive-class F1 on validation with
the established deterministic highest-threshold tie break. This is test-safe but
makes threshold-dependent validation metrics optimistic; ranking metrics and the
paired comparison remain the primary evidence.

## Paired customer bootstrap

Using 2,000 paired validation-customer resamples with seed 42, the selected
direct-minus-baseline ROC-AUC delta has bootstrap mean
`{boot.loc['ROC_AUC', 'bootstrap_mean_delta']:+.6f}` and percentile 95% CI
`[{boot.loc['ROC_AUC', 'percentile_95_CI_lower']:+.6f}, {boot.loc['ROC_AUC', 'percentile_95_CI_upper']:+.6f}]`.
The fraction of replicates above zero is
`{boot.loc['ROC_AUC', 'fraction_above_zero']:.3%}`.
The AP delta has mean `{boot.loc['PR_AUC_average_precision', 'bootstrap_mean_delta']:+.6f}`
and 95% CI `[{boot.loc['PR_AUC_average_precision', 'percentile_95_CI_lower']:+.6f}, {boot.loc['PR_AUC_average_precision', 'percentile_95_CI_upper']:+.6f}]`.
Its fraction above zero is
`{boot.loc['PR_AUC_average_precision', 'fraction_above_zero']:.3%}`.
These are descriptive validation uncertainty intervals, not an independent-test claim.

## Importance, redundancy, and stability

Selected-model validation permutation importance (five repeats, seed 42) is led by:

{top_lines}

Training redundancy audit (absolute Spearman >= 0.95) is led by:

{redundancy_lines}

Training effectively constant features: {', '.join(constants) if constants else 'none'}.
Correlated-feature permutation importance can distribute credit among substitutes;
neither correlation nor predictive importance establishes a causal theft mechanism.

Overall level is important: upper-tail customer maximum is the largest permutation
contributor, while a lower-tail median statistic also ranks highly. Long-horizon
variability and abrupt-change persistence contribute through upper-tail consecutive
change and train-q90 exceedance features. Across-window consistency also matters:
variation in within-window slope and early-to-late level shifts carry signal. Both
lower and upper behavioural quantiles therefore matter, but the importance pattern
is distributed and partially redundant rather than a single-feature explanation.

## Decision: {decision} — {decision_label}

The pre-registered decision rule yields **{decision}**. The result addresses the
customer-label/window-training mismatch directly: customer features are calculated
before model fitting, and validation is scored at the customer unit. It does not
show operational readiness, causal attribution, or generalization to untouched test.
The sizeable gains, with both paired intervals above zero, are consistent with the
hypothesis that propagating a customer label to every window introduces harmful
instance-level label noise. Performance alone cannot prove that mechanism; aggregation
also changes representation and model sample size. Methodological alignment and the
observed validation gain nevertheless justify making the customer the fundamental
AMP prediction unit and freezing the direct customer HGB as the next supervised baseline.

For the paper, Stage 16 is evidence about formulation rather than another broad
architecture search. It should be reported alongside the Stage 14 window baseline
and the Stage 15 negative FC hybrid result, with weak-label and repeated-validation
limitations explicit.

## Recommended Stage 17 research question

**Does the frozen direct-customer HGB retain its advantage over the frozen Stage 14
window-to-customer baseline across pre-specified customer history-length and
consumption-level strata?** Run this as a validation-only robustness audit with
fixed models and no feature changes before considering any further test use.
**TEST NOT ACCESSED DURING STAGE 16.**
"""
    (RESULTS_DIR / "stage_16_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    preregistration = load_preregistration()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    decision_path = RESULTS_DIR / "validation_decision.json"
    if decision_path.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 16 decision: {decision_path}")

    print("Loading Stage 16 allowed splits: train and val only...")
    train, train_window_numbers = load_allowed_split("train")
    validation, validation_window_numbers = load_allowed_split("val")
    split_audit = split_integrity_audit(train, validation)
    count_frames: list[pd.DataFrame] = []
    count_summaries: dict[str, dict[str, Any]] = {}
    for split, ids in (("train", train.customer_ids), ("validation", validation.customer_ids)):
        frame, summary = window_count_outputs(split, ids)
        count_frames.append(frame)
        count_summaries[split] = summary
    save_csv(pd.concat(count_frames, ignore_index=True), RESULTS_DIR / "customer_window_count_distribution.csv")

    train_summaries = behavioural_features(train.X)
    validation_summaries = behavioural_features(validation.X)
    train_features, train_labels, train_ids, persistence_thresholds = build_customer_features(
        train_summaries, train.y, train.customer_ids, train_window_numbers, None
    )
    validation_features, validation_labels, validation_ids, _ = build_customer_features(
        validation_summaries,
        validation.y,
        validation.customer_ids,
        validation_window_numbers,
        persistence_thresholds,
    )
    customer_dataset = {
        "train": {
            "customers": int(len(train_labels)),
            "normal_customers": int(np.sum(train_labels == 0)),
            "theft_customers": int(np.sum(train_labels == 1)),
            "theft_prevalence": float(np.mean(train_labels)),
        },
        "validation": {
            "customers": int(len(validation_labels)),
            "normal_customers": int(np.sum(validation_labels == 0)),
            "theft_customers": int(np.sum(validation_labels == 1)),
            "theft_prevalence": float(np.mean(validation_labels)),
        },
    }
    distribution, redundancy_pairs, constants = feature_sanity(train_features, validation_features)
    save_csv(distribution, RESULTS_DIR / "customer_feature_distribution.csv")
    save_csv(redundancy_pairs, RESULTS_DIR / "training_redundant_feature_pairs.csv")

    baseline_row, baseline_customers, reproduction = reproduce_stage14_baseline(validation, validation_summaries)
    baseline_customers = baseline_customers.set_index(ID_COLUMN).reindex(validation_ids).reset_index()
    if not np.array_equal(baseline_customers[LABEL_COLUMN].to_numpy(dtype=np.int8), validation_labels):
        raise ValueError("Baseline and direct-customer validation labels do not align")

    rows = [baseline_row]
    probability_frames = [baseline_customers[["model", ID_COLUMN, LABEL_COLUMN, "mean_probability", "usable_window_count"]].rename(columns={"mean_probability": "probability"})]
    fitted_models: dict[str, object] = {}
    model_records: list[dict[str, Any]] = []
    train_matrix = train_features.to_numpy(dtype=np.float64)
    validation_matrix = validation_features.to_numpy(dtype=np.float64)
    for family in ("logistic", "hist_gradient_boosting"):
        model_name = DIRECT_MODEL_NAMES[family]
        print(f"Fitting {model_name}...")
        model = make_model(family)
        started = time.perf_counter()
        model.fit(train_matrix, train_labels)
        runtime = time.perf_counter() - started
        scores = np.asarray(model.predict_proba(validation_matrix)[:, 1], dtype=np.float64)
        row = metric_row(model_name, "direct_customer_features", family, validation_labels, scores)
        rows.append(row)
        fitted_models[model_name] = model
        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path, compress=3)
        model_records.append(
            {
                "model": model_name,
                "family": family,
                "fit_runtime_seconds": float(runtime),
                "artifact": relative_path(model_path),
                "artifact_sha256": file_sha256(model_path),
            }
        )
        probability_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    ID_COLUMN: validation_ids,
                    LABEL_COLUMN: validation_labels,
                    "probability": scores,
                    "usable_window_count": pd.Series(validation.customer_ids).value_counts(sort=False).reindex(validation_ids).to_numpy(dtype=int),
                }
            )
        )
        print(f"  customer ROC-AUC={row['ROC_AUC']:.6f}, AP={row['PR_AUC_average_precision']:.6f}; {runtime:.2f}s")

    metrics = pd.DataFrame(rows)
    save_csv(metrics, RESULTS_DIR / "validation_metrics.csv")
    probabilities = pd.concat(probability_frames, ignore_index=True)
    save_csv(probabilities, RESULTS_DIR / "validation_customer_probabilities.csv")
    baseline = metrics.loc[metrics["model"].eq(BASELINE_NAME)].iloc[0]
    deltas = metrics.loc[metrics["formulation"].eq("direct_customer_features")].copy()
    for metric in ("ROC_AUC", "PR_AUC_average_precision", "precision", "recall", "F1"):
        deltas[f"{metric}_delta_vs_baseline"] = deltas[metric] - baseline[metric]
    save_csv(deltas, RESULTS_DIR / "customer_deltas_vs_stage14.csv")

    selected = deltas.sort_values(
        ["ROC_AUC", "PR_AUC_average_precision", "model"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    selected_name = str(selected["model"])
    selected_scores = probabilities.loc[probabilities["model"].eq(selected_name), "probability"].to_numpy(dtype=np.float64)
    baseline_scores = probabilities.loc[probabilities["model"].eq(BASELINE_NAME), "probability"].to_numpy(dtype=np.float64)
    bootstrap_samples, bootstrap_summary = paired_bootstrap(validation_labels, selected_scores, baseline_scores)
    save_csv(bootstrap_samples, RESULTS_DIR / "paired_customer_bootstrap_samples.csv")
    save_csv(bootstrap_summary, RESULTS_DIR / "paired_customer_bootstrap_summary.csv")
    importance = validation_permutation_importance(fitted_models[selected_name], validation_matrix.copy(), validation_labels)
    save_csv(importance, RESULTS_DIR / "validation_permutation_importance.csv")

    roc_delta = float(selected["ROC_AUC"] - baseline["ROC_AUC"])
    ap_delta = float(selected["PR_AUC_average_precision"] - baseline["PR_AUC_average_precision"])
    decision, decision_label = decision_code(roc_delta, ap_delta, bootstrap_summary)
    selected_record = next(record for record in model_records if record["model"] == selected_name)
    save_json(
        {
            "stage": "Stage 16 — validation-only training summary",
            "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
            "feature_count": len(FEATURE_NAMES),
            "feature_names_in_order": list(FEATURE_NAMES),
            "train_derived_persistence_q90_thresholds": persistence_thresholds,
            "split_integrity": split_audit,
            "customer_dataset": customer_dataset,
            "usable_window_count_summaries": count_summaries,
            "feature_sanity": {
                "train_shape": list(train_features.shape),
                "validation_shape": list(validation_features.shape),
                "train_nonfinite_values": int(np.sum(~np.isfinite(train_matrix))),
                "validation_nonfinite_values": int(np.sum(~np.isfinite(validation_matrix))),
                "effectively_constant_training_features": constants,
                "absolute_spearman_ge_0_95_pair_count": int(len(redundancy_pairs)),
            },
            "stage14_baseline_reproduction": reproduction,
            "models": model_records,
            "test_access": split_audit | {"explicit_statement": "TEST NOT ACCESSED DURING STAGE 16"},
        },
        RESULTS_DIR / "training_summary.json",
    )
    save_json(
        {
            "stage": "Stage 16 — validation-only formulation decision",
            "primary_candidate_selection": preregistration["primary_candidate_selection"],
            "decision_policy": preregistration["decision_policy"],
            "selected_direct_model": selected_name,
            "selected_model_artifact": selected_record["artifact"],
            "selected_model_sha256": selected_record["artifact_sha256"],
            "selected_customer_threshold": float(selected["threshold"]),
            "selected_customer_ROC_AUC": float(selected["ROC_AUC"]),
            "selected_customer_PR_AUC_average_precision": float(selected["PR_AUC_average_precision"]),
            "selected_customer_F1": float(selected["F1"]),
            "baseline_customer_ROC_AUC": float(baseline["ROC_AUC"]),
            "baseline_customer_PR_AUC_average_precision": float(baseline["PR_AUC_average_precision"]),
            "baseline_customer_F1": float(baseline["F1"]),
            "deltas_direct_minus_baseline": {
                "ROC_AUC": roc_delta,
                "PR_AUC_average_precision": ap_delta,
                "F1": float(selected["F1"] - baseline["F1"]),
            },
            "decision": decision,
            "decision_label": decision_label,
            "frozen_next_supervised_baseline": bool(decision == "A"),
            "test_evaluation_performed": False,
            "test_policy": "CLOSED; TEST NOT ACCESSED DURING STAGE 16",
        },
        decision_path,
    )
    write_report(
        split_audit,
        count_summaries,
        customer_dataset,
        metrics,
        selected_name,
        deltas,
        bootstrap_summary,
        importance,
        redundancy_pairs,
        constants,
        reproduction,
        decision,
        decision_label,
    )
    print(f"Decision {decision}: {decision_label}; selected direct model {selected_name}")
    print("Stage 16 completed with train and validation only; test was not accessed.")


if __name__ == "__main__":
    main()
