"""Stage 14: validation-only behavioural representation comparison.

The test split is deliberately outside this program's allowed split set.
"""

from __future__ import annotations

import json
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


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_14"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_14"
PREREGISTRATION_PATH = RESULTS_DIR / "preregistered_config.json"
ALLOWED_SPLITS = ("train", "val")
REPRESENTATIONS = ("RAW_ONLY", "SUMMARY_ONLY", "RAW_PLUS_SUMMARY")
FEATURE_NAMES = (
    "mean",
    "median",
    "minimum",
    "maximum",
    "q10",
    "q90",
    "standard_deviation",
    "range",
    "interquartile_range",
    "mean_absolute_deviation",
    "first_to_last_change",
    "absolute_first_to_last_change",
    "mean_absolute_consecutive_change",
    "maximum_absolute_consecutive_change",
    "linear_slope",
    "fraction_below_minus_one",
    "second_half_minus_first_half_mean",
)


def load_preregistration() -> dict[str, Any]:
    if not PREREGISTRATION_PATH.is_file():
        raise FileNotFoundError("Stage 14 pre-registration must exist before scoring")
    with PREREGISTRATION_PATH.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not config.get("created_before_stage_14_validation_scoring"):
        raise ValueError("Pre-registration does not certify pre-validation creation")
    if tuple(config["feature_names_in_order"]) != FEATURE_NAMES:
        raise ValueError("Code feature order differs from the pre-registered feature list")
    if config["test_policy"].split(".", 1)[0] != "CLOSED":
        raise ValueError("Stage 14 test policy is not closed")
    return config


def load_allowed_split(split: str) -> WindowMatrix:
    """Enforce the closed-test boundary before delegating to the shared loader."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Stage 14 permits only {ALLOWED_SPLITS}; got {split!r}")
    return load_window_matrix(split)


def behavioural_features(raw: np.ndarray) -> np.ndarray:
    """Compute the 17 deterministic, pre-registered window summaries."""
    if raw.ndim != 2 or raw.shape[1] != 30:
        raise ValueError(f"Expected a (n, 30) raw matrix; got {raw.shape}")
    values = np.asarray(raw, dtype=np.float32)
    mean = values.mean(axis=1, dtype=np.float64)
    quantiles = np.quantile(
        values, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1, method="linear"
    )
    minimum = values.min(axis=1)
    maximum = values.max(axis=1)
    standard_deviation = values.std(axis=1, dtype=np.float64, ddof=0)
    differences = np.diff(values, axis=1)
    absolute_differences = np.abs(differences)
    centered_positions = np.arange(30, dtype=np.float64) - 14.5
    slope_denominator = float(np.square(centered_positions).sum())

    summaries = np.column_stack(
        [
            mean,
            quantiles[2],
            minimum,
            maximum,
            quantiles[0],
            quantiles[4],
            standard_deviation,
            maximum - minimum,
            quantiles[3] - quantiles[1],
            np.mean(np.abs(values - mean[:, None]), axis=1, dtype=np.float64),
            values[:, -1] - values[:, 0],
            np.abs(values[:, -1] - values[:, 0]),
            absolute_differences.mean(axis=1, dtype=np.float64),
            absolute_differences.max(axis=1),
            (values.astype(np.float64, copy=False) @ centered_positions)
            / slope_denominator,
            np.mean(values <= -1.0, axis=1),
            values[:, 15:].mean(axis=1, dtype=np.float64)
            - values[:, :15].mean(axis=1, dtype=np.float64),
        ]
    ).astype(np.float32, copy=False)
    if summaries.shape != (len(values), len(FEATURE_NAMES)):
        raise ValueError("Behavioural feature matrix has an unexpected shape")
    if not np.isfinite(summaries).all():
        raise ValueError("Behavioural feature matrix contains NaN or infinity")
    return summaries


def representation_matrix(
    raw: np.ndarray, summaries: np.ndarray, representation: str
) -> np.ndarray:
    if representation == "RAW_ONLY":
        return raw
    if representation == "SUMMARY_ONLY":
        return summaries
    if representation == "RAW_PLUS_SUMMARY":
        return np.ascontiguousarray(np.column_stack([raw, summaries]), dtype=np.float32)
    raise ValueError(f"Unknown representation: {representation}")


def feature_distribution(
    split: str, summaries: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(FEATURE_NAMES):
        values = summaries[:, index].astype(np.float64, copy=False)
        rows.append(
            {
                "split": split,
                "feature": name,
                "count": int(values.size),
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
    return pd.DataFrame(rows)


def make_model(family: str, weighting: str) -> object:
    class_weight: str | None = "balanced" if weighting == "balanced" else None
    if family == "logistic":
        return Pipeline(
            [
                ("standardize", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight=class_weight,
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
            class_weight=class_weight,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown family: {family}")


def metric_rows(
    model_name: str,
    family: str,
    weighting: str,
    representation: str,
    labels: np.ndarray,
    scores: np.ndarray,
    customer_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    window_threshold = select_f1_threshold(labels, scores)
    window_metrics = classification_metrics(labels, scores, window_threshold)
    customers = aggregate_customer_mean(labels, scores, customer_ids)
    customer_labels = customers[LABEL_COLUMN].to_numpy(dtype=np.int8)
    customer_scores = customers["mean_probability"].to_numpy(dtype=np.float64)
    customer_threshold = select_f1_threshold(customer_labels, customer_scores)
    customer_metrics = classification_metrics(
        customer_labels, customer_scores, customer_threshold
    )
    shared = {
        "model": model_name,
        "family": family,
        "weighting": weighting,
        "representation": representation,
    }
    rows = [
        {**shared, "level": "window", **window_metrics},
        {**shared, "level": "customer_mean", **customer_metrics},
    ]
    customers.insert(0, "model", model_name)
    customers.insert(1, "family", family)
    customers.insert(2, "weighting", weighting)
    customers.insert(3, "representation", representation)
    return rows, customers


def stage13_reproduction_check(metrics: pd.DataFrame) -> dict[str, Any]:
    stage13_path = (
        PROJECT_ROOT / "results" / "supervised" / "stage_13" / "validation_metrics.csv"
    )
    stage13 = pd.read_csv(stage13_path)
    comparisons: list[dict[str, Any]] = []
    mapping = {
        ("logistic", "unweighted"): "logistic_unweighted",
        ("logistic", "balanced"): "logistic_balanced",
        (
            "hist_gradient_boosting",
            "unweighted",
        ): "hist_gradient_boosting_unweighted",
        (
            "hist_gradient_boosting",
            "balanced",
        ): "hist_gradient_boosting_balanced",
    }
    for (family, weighting), old_name in mapping.items():
        for level in ("window", "customer_mean"):
            current = metrics.loc[
                metrics["family"].eq(family)
                & metrics["weighting"].eq(weighting)
                & metrics["representation"].eq("RAW_ONLY")
                & metrics["level"].eq(level)
            ].iloc[0]
            prior = stage13.loc[
                stage13["model"].eq(old_name) & stage13["level"].eq(level)
            ].iloc[0]
            differences = {
                metric: float(current[metric] - prior[metric])
                for metric in (
                    "ROC_AUC",
                    "PR_AUC_average_precision",
                    "precision",
                    "recall",
                    "F1",
                    "threshold",
                )
            }
            comparisons.append(
                {
                    "family": family,
                    "weighting": weighting,
                    "level": level,
                    "stage13_model": old_name,
                    "differences_stage14_minus_stage13": differences,
                    "exact_within_1e-12": bool(
                        max(abs(value) for value in differences.values()) <= 1e-12
                    ),
                }
            )
    if not all(item["exact_within_1e-12"] for item in comparisons):
        raise ValueError("Stage 14 RAW_ONLY does not reproduce Stage 13 validation")
    return {
        "stage13_validation_artifact": relative_path(stage13_path),
        "comparisons": comparisons,
        "all_raw_only_metrics_reproduced_within_1e-12": True,
    }


def customer_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    customers = metrics.loc[metrics["level"].eq("customer_mean")].copy()
    rows: list[dict[str, Any]] = []
    for (family, weighting), group in customers.groupby(
        ["family", "weighting"], sort=False
    ):
        raw = group.loc[group["representation"].eq("RAW_ONLY")].iloc[0]
        for _, row in group.iterrows():
            output = row.to_dict()
            for metric in ("ROC_AUC", "PR_AUC_average_precision"):
                absolute = float(row[metric] - raw[metric])
                output[f"{metric}_absolute_delta_vs_raw"] = absolute
                output[f"{metric}_relative_delta_vs_raw_percent"] = float(
                    100.0 * absolute / raw[metric]
                )
            rows.append(output)
    return pd.DataFrame(rows)


def split_integrity_audit(train: WindowMatrix, validation: WindowMatrix) -> dict[str, Any]:
    manifest_path = PROJECT_ROOT / "data" / "processed" / "split_manifest.parquet"
    # Deliberately omit FLAG: Stage 14 needs test assignment integrity, not test labels.
    manifest = pd.read_parquet(manifest_path, columns=[ID_COLUMN, "split"])
    id_sets = {
        split: set(manifest.loc[manifest["split"].eq(split), ID_COLUMN].astype(str))
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train_val": len(id_sets["train"] & id_sets["val"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "val_test": len(id_sets["val"] & id_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Customer overlap detected: {overlaps}")
    if set(train.customer_ids) & set(validation.customer_ids):
        raise ValueError("Train and validation window customer IDs overlap")
    return {
        "authoritative_manifest": relative_path(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "customer_counts_without_loading_test_labels": {
            split: len(ids) for split, ids in id_sets.items()
        },
        "customer_overlaps": overlaps,
        "train_windows": int(len(train.y)),
        "validation_windows": int(len(validation.y)),
        "train_window_prevalence": float(train.y.mean()),
        "validation_window_prevalence": float(validation.y.mean()),
        "loaded_splits": ["train", "val"],
        "test_features_loaded": False,
        "test_labels_loaded": False,
        "test_predictions_inspected": False,
    }


def permutation_importance(
    model: object,
    representation: str,
    raw_validation: np.ndarray,
    summary_validation: np.ndarray,
    labels: np.ndarray,
    customer_ids: np.ndarray,
    repetitions: int = 5,
) -> pd.DataFrame:
    matrix = representation_matrix(raw_validation, summary_validation, representation)
    base_scores = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    base_customers = aggregate_customer_mean(labels, base_scores, customer_ids)
    base_y = base_customers[LABEL_COLUMN].to_numpy(dtype=np.int8)
    base_probability = base_customers["mean_probability"].to_numpy(dtype=np.float64)
    base_roc = float(roc_auc_score(base_y, base_probability))
    base_ap = float(average_precision_score(base_y, base_probability))

    if representation == "SUMMARY_ONLY":
        feature_indices = list(range(len(FEATURE_NAMES)))
        names = list(FEATURE_NAMES)
    elif representation == "RAW_PLUS_SUMMARY":
        feature_indices = list(range(30, 30 + len(FEATURE_NAMES)))
        names = list(FEATURE_NAMES)
    else:
        feature_indices = list(range(30))
        names = [f"raw_position_{index + 1:02d}" for index in range(30)]

    rng = np.random.default_rng(RANDOM_STATE)
    permuted = matrix.copy()
    rows: list[dict[str, Any]] = []
    for name, index in zip(names, feature_indices):
        original = matrix[:, index].copy()
        roc_decreases: list[float] = []
        ap_decreases: list[float] = []
        for _ in range(repetitions):
            permuted[:, index] = original[rng.permutation(len(original))]
            scores = np.asarray(model.predict_proba(permuted)[:, 1], dtype=np.float64)
            customers = aggregate_customer_mean(labels, scores, customer_ids)
            customer_y = customers[LABEL_COLUMN].to_numpy(dtype=np.int8)
            probabilities = customers["mean_probability"].to_numpy(dtype=np.float64)
            roc_decreases.append(base_roc - roc_auc_score(customer_y, probabilities))
            ap_decreases.append(
                base_ap - average_precision_score(customer_y, probabilities)
            )
            permuted[:, index] = original
        rows.append(
            {
                "feature": name,
                "validation_repetitions": repetitions,
                "baseline_customer_ROC_AUC": base_roc,
                "mean_ROC_AUC_decrease": float(np.mean(roc_decreases)),
                "std_ROC_AUC_decrease": float(np.std(roc_decreases, ddof=0)),
                "baseline_customer_average_precision": base_ap,
                "mean_average_precision_decrease": float(np.mean(ap_decreases)),
                "std_average_precision_decrease": float(
                    np.std(ap_decreases, ddof=0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_ROC_AUC_decrease", "mean_average_precision_decrease"],
        ascending=False,
        kind="mergesort",
    )


def main() -> None:
    preregistration = load_preregistration()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    decision_path = RESULTS_DIR / "validation_decision.json"
    if decision_path.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 14 decision: {decision_path}")

    print("Loading Stage 14 allowed splits: train and val only...")
    train = load_allowed_split("train")
    validation = load_allowed_split("val")
    split_audit = split_integrity_audit(train, validation)
    train_summaries = behavioural_features(train.X)
    validation_summaries = behavioural_features(validation.X)

    distributions = pd.concat(
        [
            feature_distribution("train", train_summaries),
            feature_distribution("val", validation_summaries),
        ],
        ignore_index=True,
    )
    if distributions["nonfinite_count"].ne(0).any():
        raise ValueError("Non-finite behavioural feature found")
    if distributions["effectively_constant"].any():
        constants = distributions.loc[
            distributions["effectively_constant"], ["split", "feature"]
        ].to_dict("records")
        raise ValueError(f"Effectively constant feature found: {constants}")
    save_csv(distributions, RESULTS_DIR / "feature_distribution_summary.csv")

    rows: list[dict[str, Any]] = []
    customer_frames: list[pd.DataFrame] = []
    model_records: list[dict[str, Any]] = []
    fitted_models: dict[str, object] = {}
    for family in ("logistic", "hist_gradient_boosting"):
        for weighting in ("unweighted", "balanced"):
            for representation in REPRESENTATIONS:
                model_name = f"{family}_{weighting}_{representation.lower()}"
                print(f"Fitting {model_name}...")
                train_matrix = representation_matrix(
                    train.X, train_summaries, representation
                )
                validation_matrix = representation_matrix(
                    validation.X, validation_summaries, representation
                )
                model = make_model(family, weighting)
                started = time.perf_counter()
                model.fit(train_matrix, train.y)
                runtime = time.perf_counter() - started
                scores = np.asarray(
                    model.predict_proba(validation_matrix)[:, 1], dtype=np.float64
                )
                result_rows, customers = metric_rows(
                    model_name,
                    family,
                    weighting,
                    representation,
                    validation.y,
                    scores,
                    validation.customer_ids,
                )
                rows.extend(result_rows)
                customer_frames.append(customers)
                model_path = MODEL_DIR / f"{model_name}.joblib"
                joblib.dump(model, model_path, compress=3)
                fitted_models[model_name] = model
                customer_row = next(
                    row for row in result_rows if row["level"] == "customer_mean"
                )
                model_records.append(
                    {
                        "model": model_name,
                        "family": family,
                        "weighting": weighting,
                        "representation": representation,
                        "fit_runtime_seconds": float(runtime),
                        "artifact": relative_path(model_path),
                        "artifact_sha256": file_sha256(model_path),
                        "customer_ROC_AUC": customer_row["ROC_AUC"],
                        "customer_PR_AUC_average_precision": customer_row[
                            "PR_AUC_average_precision"
                        ],
                    }
                )
                print(
                    f"  customer ROC-AUC={customer_row['ROC_AUC']:.6f}, "
                    f"AP={customer_row['PR_AUC_average_precision']:.6f}; "
                    f"{runtime:.2f}s"
                )

    metrics = pd.DataFrame(rows)
    save_csv(metrics, RESULTS_DIR / "validation_metrics.csv")
    save_csv(customer_deltas(metrics), RESULTS_DIR / "customer_representation_deltas.csv")
    save_csv(
        pd.concat(customer_frames, ignore_index=True),
        RESULTS_DIR / "validation_customer_probabilities.csv",
    )
    reproduction = stage13_reproduction_check(metrics)

    primary = metrics.loc[metrics["level"].eq("customer_mean")].sort_values(
        ["ROC_AUC", "PR_AUC_average_precision", "model"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    selected = primary.iloc[0]
    selected_name = str(selected["model"])
    selected_record = next(
        record for record in model_records if record["model"] == selected_name
    )
    selected_window = metrics.loc[
        metrics["model"].eq(selected_name) & metrics["level"].eq("window")
    ].iloc[0]

    tree_primary = primary.loc[primary["family"].eq("hist_gradient_boosting")]
    best_tree = tree_primary.iloc[0]
    best_tree_name = str(best_tree["model"])
    importance = permutation_importance(
        fitted_models[best_tree_name],
        str(best_tree["representation"]),
        validation.X,
        validation_summaries,
        validation.y,
        validation.customer_ids,
    )
    save_csv(importance, RESULTS_DIR / "validation_permutation_importance.csv")

    save_json(
        {
            "stage": "Stage 14 — validation-only training summary",
            "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
            "models": model_records,
            "split_integrity": split_audit,
            "feature_sanity": {
                "feature_count": len(FEATURE_NAMES),
                "train_nonfinite_values": int(np.sum(~np.isfinite(train_summaries))),
                "validation_nonfinite_values": int(
                    np.sum(~np.isfinite(validation_summaries))
                ),
                "effectively_constant_features": [],
                "clipping_or_additional_transformation": (
                    "none; logistic uses its pre-registered train-fitted StandardScaler and "
                    "HistGradientBoosting is rank/bin based"
                ),
            },
            "stage13_raw_only_reproduction": reproduction,
            "test_access": {
                "loaded_splits": ["train", "val"],
                "test_features_loaded": False,
                "test_labels_loaded": False,
                "test_predictions_loaded_or_inspected": False,
                "stage13_historical_test_metrics_used_for_selection": False,
            },
        },
        RESULTS_DIR / "training_summary.json",
    )
    save_json(
        {
            "stage": "Stage 14 — validation-only representation decision",
            "selection_policy": preregistration["primary_selection_policy"],
            "selected_model": selected_name,
            "selected_family": selected["family"],
            "selected_weighting": selected["weighting"],
            "selected_representation": selected["representation"],
            "selected_model_artifact": selected_record["artifact"],
            "selected_model_sha256": selected_record["artifact_sha256"],
            "customer_mean_threshold": float(selected["threshold"]),
            "customer_mean_ROC_AUC": float(selected["ROC_AUC"]),
            "customer_mean_PR_AUC_average_precision": float(
                selected["PR_AUC_average_precision"]
            ),
            "window_threshold": float(selected_window["threshold"]),
            "window_ROC_AUC": float(selected_window["ROC_AUC"]),
            "window_PR_AUC_average_precision": float(
                selected_window["PR_AUC_average_precision"]
            ),
            "best_tree_for_interpretation": best_tree_name,
            "test_evaluation_performed": False,
            "test_policy": "CLOSED after Stage 13; no Stage 14 test evaluation permitted",
        },
        decision_path,
    )
    print(
        f"Selected validation-only baseline: {selected_name}; customer "
        f"ROC-AUC={selected['ROC_AUC']:.6f}, "
        f"AP={selected['PR_AUC_average_precision']:.6f}"
    )
    print("Stage 14 completed with train and validation only; test was not accessed.")


if __name__ == "__main__":
    main()
