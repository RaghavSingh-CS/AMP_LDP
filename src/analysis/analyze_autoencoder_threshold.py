"""Stage 5 validation-only reconstruction analysis and threshold selection.

This module never reads the test split and never retrains the Autoencoder. It
uses the completed Stage 4 validation reconstruction-error artifact only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.models.autoencoder import Autoencoder, architecture_config, count_trainable_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "autoencoder"
RESULTS_DIR = PROJECT_ROOT / "results" / "autoencoder" / "stride_30"
ANALYSIS_DIR = RESULTS_DIR / "threshold_analysis"

CHECKPOINT_PATH = MODEL_DIR / "ae_stride30_baseline.pt"
CONFIG_PATH = MODEL_DIR / "ae_stride30_baseline_config.json"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"
TRAINING_SUMMARY_PATH = RESULTS_DIR / "training_summary.json"
TRAINING_CURVE_PATH = RESULTS_DIR / "training_curve.png"
ERRORS_PATH = RESULTS_DIR / "val_reconstruction_errors.parquet"
ERROR_DISTRIBUTION_PATH = RESULTS_DIR / "validation_error_distribution.png"

THRESHOLD_PATH = MODEL_DIR / "ae_stride30_threshold.json"
METRICS_PATH = ANALYSIS_DIR / "threshold_metrics.csv"
SUMMARY_PATH = ANALYSIS_DIR / "threshold_summary.json"
REPORT_PATH = ANALYSIS_DIR / "threshold_report.md"
PR_CURVE_PATH = ANALYSIS_DIR / "precision_recall_curve.png"
TRADEOFF_PATH = ANALYSIS_DIR / "threshold_tradeoff.png"
CUSTOMER_DIAGNOSTICS_PATH = ANALYSIS_DIR / "customer_level_diagnostics.csv"
MISSINGNESS_DIAGNOSTICS_PATH = ANALYSIS_DIR / "missingness_group_summary.csv"

STAGE4_PATHS = (
    CHECKPOINT_PATH,
    CONFIG_PATH,
    HISTORY_PATH,
    TRAINING_SUMMARY_PATH,
    TRAINING_CURVE_PATH,
    ERRORS_PATH,
    ERROR_DISTRIBUTION_PATH,
    PROJECT_ROOT / "src" / "data" / "window_dataset.py",
    PROJECT_ROOT / "src" / "models" / "autoencoder.py",
    PROJECT_ROOT / "src" / "training" / "train_autoencoder.py",
)
EXPECTED_ARCHITECTURE = [30, 16, 8, 4, 8, 16, 30]
NORMAL_PERCENTILES = (0.90, 0.95, 0.975, 0.99)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one protected input without loading it wholly into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage4_hashes() -> dict[str, str]:
    """Hash every Stage 4 input that Stage 5 must leave unchanged."""
    missing = [str(path) for path in STAGE4_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Stage 4 artifacts are missing: {missing}")
    return {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in STAGE4_PATHS
    }


def load_json(path: Path) -> dict[str, object]:
    """Load one JSON object."""
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def validate_stage4_artifacts() -> dict[str, object]:
    """Load the checkpoint and verify the completed Stage 4 evidence."""
    config = load_json(CONFIG_PATH)
    training_summary = load_json(TRAINING_SUMMARY_PATH)
    history = pd.read_csv(HISTORY_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = Autoencoder()
    model.load_state_dict(checkpoint["model_state_dict"])

    if architecture_config()["layer_dimensions"] != EXPECTED_ARCHITECTURE:
        raise ValueError("Source Autoencoder architecture differs from the Stage 4 baseline")
    if checkpoint["architecture_configuration"]["layer_dimensions"] != EXPECTED_ARCHITECTURE:
        raise ValueError("Checkpoint architecture differs from the Stage 4 baseline")
    if config["model"]["layer_dimensions"] != EXPECTED_ARCHITECTURE:
        raise ValueError("Model configuration architecture is inconsistent")
    if count_trainable_parameters(model) != 1362:
        raise ValueError("Stage 4 checkpoint does not match the 1,362-parameter baseline")
    if config["training_population"] != "usable stride-30 train windows with FLAG == 0 only":
        raise ValueError("Stage 4 training population policy is inconsistent")
    if config["early_stopping_population"] != (
        "usable stride-30 validation windows with FLAG == 0 only"
    ):
        raise ValueError("Stage 4 validation population policy is inconsistent")
    if config["early_stopping"]["monitor"] != "normal_validation_mse":
        raise ValueError("Stage 4 did not monitor normal validation MSE")
    if config["test_set_policy"] != "not loaded or used during Stage 4":
        raise ValueError("Stage 4 test-set policy is inconsistent")
    if training_summary["test_set_used"] is not False:
        raise ValueError("Stage 4 summary reports test-set use")
    if training_summary["anomaly_threshold_selected"] is not False:
        raise ValueError("Stage 4 unexpectedly selected an anomaly threshold")
    if list(history.columns) != ["epoch", "train_mse", "val_normal_mse"]:
        raise ValueError("Training history schema is unexpected")
    if len(history) != int(training_summary["epochs_executed"]):
        raise ValueError("Training history length disagrees with training summary")
    best_epoch = int(checkpoint["best_epoch"])
    best_row = history.loc[history["epoch"].eq(best_epoch)]
    if len(best_row) != 1 or not np.isclose(
        best_row.iloc[0]["val_normal_mse"],
        float(checkpoint["best_validation_loss"]),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Best checkpoint loss is not aligned with training history")

    return {
        "checkpoint_loaded": True,
        "architecture": EXPECTED_ARCHITECTURE,
        "trainable_parameters": 1362,
        "device": training_summary["device"],
        "training_normal_windows": int(training_summary["training_normal_window_count"]),
        "validation_normal_windows": int(
            training_summary["validation_normal_window_count"]
        ),
        "validation_theft_windows": int(
            training_summary["validation_theft_window_count"]
        ),
        "maximum_epochs": int(config["maximum_epochs"]),
        "epochs_executed": int(training_summary["epochs_executed"]),
        "best_epoch": best_epoch,
        "best_validation_mse": float(checkpoint["best_validation_loss"]),
        "early_stopping_occurred": bool(
            training_summary["early_stopping_occurred"]
        ),
        "test_set_used": False,
        "anomaly_threshold_selected_during_stage4": False,
        "config": config,
    }


def load_validation_errors() -> pd.DataFrame:
    """Load and validate the Stage 4 validation-only reconstruction scores."""
    errors = pd.read_parquet(ERRORS_PATH, engine="pyarrow")
    expected_columns = [
        "window_id",
        "CONS_NO",
        "FLAG",
        "reconstruction_mse",
        "missing_fraction",
    ]
    if list(errors.columns) != expected_columns:
        raise ValueError(f"Validation error columns differ from {expected_columns}")
    if errors["window_id"].isna().any() or not errors["window_id"].is_unique:
        raise ValueError("Validation window IDs must be non-missing and unique")
    if errors["CONS_NO"].isna().any():
        raise ValueError("Validation customer IDs contain missing values")
    if set(errors["FLAG"].unique()) != {0, 1}:
        raise ValueError("Validation errors must contain both FLAG=0 and FLAG=1")
    scores = errors["reconstruction_mse"].to_numpy(dtype=np.float64, copy=False)
    fractions = errors["missing_fraction"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Reconstruction errors must be finite and nonnegative")
    if not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions >= 1)):
        raise ValueError("Validation missing fractions are outside Stage 3C eligibility")
    return errors


def distribution_summary(values: Iterable[float]) -> dict[str, float | int]:
    """Return the requested reconstruction-error distribution statistics."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty score population")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def summarize_error_distributions(errors: pd.DataFrame) -> dict[str, object]:
    """Summarize normal/theft-labelled scores and their distribution overlap."""
    normal = errors.loc[errors["FLAG"].eq(0), "reconstruction_mse"].to_numpy(
        dtype=np.float64
    )
    theft = errors.loc[errors["FLAG"].eq(1), "reconstruction_mse"].to_numpy(
        dtype=np.float64
    )
    normal_summary = distribution_summary(normal)
    theft_summary = distribution_summary(theft)
    exceedance: dict[str, object] = {}
    for percentile in NORMAL_PERCENTILES:
        threshold = float(np.quantile(normal, percentile))
        count = int(np.count_nonzero(theft > threshold))
        exceedance[f"normal_p{percentile * 100:g}"] = {
            "threshold": threshold,
            "theft_windows_exceeding": count,
            "theft_proportion_exceeding": count / len(theft),
        }
    return {
        "FLAG_0_normal": normal_summary,
        "FLAG_1_theft_labelled": theft_summary,
        "theft_to_normal_median_ratio": theft_summary["median"]
        / normal_summary["median"],
        "theft_to_normal_mean_ratio": theft_summary["mean"] / normal_summary["mean"],
        "overlap": {
            "score_ranges_overlap": bool(
                max(normal_summary["minimum"], theft_summary["minimum"])
                <= min(normal_summary["maximum"], theft_summary["maximum"])
            ),
            "theft_at_or_below_normal_median_proportion": float(
                np.mean(theft <= normal_summary["median"])
            ),
            "normal_above_theft_median_proportion": float(
                np.mean(normal > theft_summary["median"])
            ),
        },
        "theft_exceeding_normal_percentiles": exceedance,
    }


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Divide metric components, returning zero for an empty denominator."""
    return float(numerator / denominator) if denominator else 0.0


def calculate_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Calculate weakly supervised validation metrics for score > threshold."""
    predicted = scores > threshold
    positive = labels == 1
    negative = ~positive
    tp = int(np.count_nonzero(predicted & positive))
    tn = int(np.count_nonzero(~predicted & negative))
    fp = int(np.count_nonzero(predicted & negative))
    fn = int(np.count_nonzero(~predicted & positive))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    fpr = safe_divide(fp, fp + tn)
    return {
        "threshold": float(threshold),
        "comparison_rule": "reconstruction_mse > threshold",
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": safe_divide(2 * precision * recall, precision + recall),
        "specificity": specificity,
        "false_positive_rate": fpr,
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": safe_divide(tp + tn, len(labels)),
        "predicted_anomaly_count": int(np.count_nonzero(predicted)),
        "predicted_anomaly_percentage": 100.0 * np.mean(predicted),
    }


def evaluate_percentile_thresholds(errors: pd.DataFrame) -> pd.DataFrame:
    """Evaluate normal-only percentile thresholds on all validation labels."""
    labels = errors["FLAG"].to_numpy(dtype=np.int8, copy=False)
    scores = errors["reconstruction_mse"].to_numpy(dtype=np.float64, copy=False)
    normal = scores[labels == 0]
    rows: list[dict[str, object]] = []
    for percentile in NORMAL_PERCENTILES:
        row: dict[str, object] = {
            "candidate": f"normal_p{percentile * 100:g}",
            "selection_family": "normal_validation_percentile",
            "normal_percentile": percentile,
        }
        row.update(calculate_metrics(labels, scores, float(np.quantile(normal, percentile))))
        rows.append(row)
    return pd.DataFrame(rows)


def supervised_metric_arrays(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate every unique observed score efficiently using sorted arrays."""
    thresholds = np.unique(scores)
    normal = np.sort(scores[labels == 0])
    theft = np.sort(scores[labels == 1])
    tp = len(theft) - np.searchsorted(theft, thresholds, side="right")
    fp = len(normal) - np.searchsorted(normal, thresholds, side="right")
    fn = len(theft) - tp
    tn = len(normal) - fp
    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0
    )
    recall = tp / len(theft)
    specificity = tn / len(normal)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return thresholds, {
        "F1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "false_positive_rate": fp / len(normal),
    }


def best_index(
    values: np.ndarray, false_positive_rates: np.ndarray, thresholds: np.ndarray
) -> int:
    """Maximize a metric; break numerical ties by lower FPR then higher threshold."""
    maximum = np.max(values)
    candidates = np.flatnonzero(np.isclose(values, maximum, rtol=0, atol=1e-15))
    ordering = np.lexsort((-thresholds[candidates], false_positive_rates[candidates]))
    return int(candidates[ordering[0]])


def evaluate_supervised_thresholds(errors: pd.DataFrame) -> pd.DataFrame:
    """Select best-F1 and best-balanced-accuracy validation candidates."""
    labels = errors["FLAG"].to_numpy(dtype=np.int8, copy=False)
    scores = errors["reconstruction_mse"].to_numpy(dtype=np.float64, copy=False)
    thresholds, arrays = supervised_metric_arrays(labels, scores)
    rows: list[dict[str, object]] = []
    for candidate, target in (
        ("best_validation_f1", "F1"),
        ("best_validation_balanced_accuracy", "balanced_accuracy"),
    ):
        index = best_index(
            arrays[target], arrays["false_positive_rate"], thresholds
        )
        row: dict[str, object] = {
            "candidate": candidate,
            "selection_family": "supervised_validation_labels",
            "normal_percentile": np.nan,
        }
        row.update(calculate_metrics(labels, scores, float(thresholds[index])))
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_customer_level_scores(errors: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare mean, maximum, and p95 customer score aggregation diagnostically."""
    label_counts = errors.groupby("CONS_NO")["FLAG"].nunique()
    if not label_counts.eq(1).all():
        raise ValueError("A validation customer has inconsistent FLAG values")
    customers = (
        errors.groupby("CONS_NO", sort=False)
        .agg(
            FLAG=("FLAG", "first"),
            window_count=("window_id", "size"),
            mean_mse=("reconstruction_mse", "mean"),
            maximum_mse=("reconstruction_mse", "max"),
            p95_mse=("reconstruction_mse", lambda values: values.quantile(0.95)),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    aucs: dict[str, object] = {}
    for aggregation in ("mean_mse", "maximum_mse", "p95_mse"):
        roc_auc = float(roc_auc_score(customers["FLAG"], customers[aggregation]))
        pr_auc = float(average_precision_score(customers["FLAG"], customers[aggregation]))
        aucs[aggregation] = {"ROC_AUC": roc_auc, "PR_AUC": pr_auc}
        for flag in (0, 1):
            values = customers.loc[customers["FLAG"].eq(flag), aggregation]
            summary = distribution_summary(values)
            rows.append(
                {
                    "aggregation": aggregation,
                    "FLAG": flag,
                    "ROC_AUC": roc_auc,
                    "PR_AUC": pr_auc,
                    **summary,
                }
            )
    diagnostic = {
        "customer_counts": {
            "FLAG_0": int(customers["FLAG"].eq(0).sum()),
            "FLAG_1": int(customers["FLAG"].eq(1).sum()),
        },
        "aggregation_auc": aucs,
    }
    return pd.DataFrame(rows), diagnostic


def analyze_missingness_relationship(
    errors: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Measure overall/by-label correlation and score summaries by missing count."""
    correlations: dict[str, object] = {}
    populations = [("overall", errors), *[(f"FLAG_{flag}", group) for flag, group in errors.groupby("FLAG")]]
    for name, group in populations:
        correlations[name] = {
            "count": int(len(group)),
            "pearson": float(
                group["missing_fraction"].corr(group["reconstruction_mse"], method="pearson")
            ),
            "spearman": float(
                group["missing_fraction"].corr(group["reconstruction_mse"], method="spearman")
            ),
        }

    working = errors[["FLAG", "missing_fraction", "reconstruction_mse"]].copy()
    working["missing_count"] = np.rint(working["missing_fraction"] * 30).astype(np.int8)
    rows: list[dict[str, object]] = []
    for population, group in populations:
        grouped = (
            group.assign(
                missing_count=np.rint(group["missing_fraction"] * 30).astype(np.int8)
            )
            .groupby("missing_count", sort=True)["reconstruction_mse"]
            .agg(["count", "mean", "median", "std", "max"])
            .reset_index()
        )
        grouped.insert(0, "population", population)
        grouped["missing_fraction"] = grouped["missing_count"] / 30.0
        rows.extend(grouped.to_dict("records"))
    return pd.DataFrame(rows), correlations


def select_baseline_threshold(metrics: pd.DataFrame) -> tuple[pd.Series, str]:
    """Adopt normal p95 as a validation-only, false-alarm-controlled baseline."""
    selected = metrics.loc[metrics["candidate"].eq("normal_p95")]
    if len(selected) != 1:
        raise ValueError("Normal p95 candidate is missing or duplicated")
    rationale = (
        "The normal-validation 95th percentile fixes the validation normal-window "
        "false-positive rate near 5% without fitting the cutoff to noisy customer-level "
        "theft labels. It avoids the impractical false-alarm volume of the best-F1 "
        "candidate while retaining more theft-labelled windows than normal p99."
    )
    return selected.iloc[0], rationale


def save_json(payload: dict[str, object], path: Path) -> None:
    """Atomically write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, allow_nan=False)
            output.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_text(text: str, path: Path) -> None:
    """Atomically write UTF-8 text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_precision_recall_curve(
    errors: pd.DataFrame, metrics: pd.DataFrame, selected: pd.Series
) -> None:
    """Plot the weakly supervised validation precision-recall curve."""
    labels = errors["FLAG"].to_numpy(dtype=np.int8, copy=False)
    scores = errors["reconstruction_mse"].to_numpy(dtype=np.float64, copy=False)
    precision, recall, _ = precision_recall_curve(labels, scores)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(recall, precision, label="Validation PR curve", linewidth=1.5)
    axis.axhline(np.mean(labels), color="gray", linestyle="--", label="Theft-label prevalence")
    axis.scatter(
        [selected["recall"]],
        [selected["precision"]],
        color="tab:red",
        s=55,
        label="Selected normal p95",
        zorder=3,
    )
    best_f1 = metrics.loc[metrics["candidate"].eq("best_validation_f1")].iloc[0]
    axis.scatter(
        [best_f1["recall"]],
        [best_f1["precision"]],
        color="tab:green",
        s=45,
        label="Best validation F1",
        zorder=3,
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Validation Precision–Recall Curve (Weak Window Labels)")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    temporary = PR_CURVE_PATH.with_name(PR_CURVE_PATH.stem + ".tmp.png")
    try:
        figure.savefig(temporary, dpi=150)
        temporary.replace(PR_CURVE_PATH)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def save_threshold_tradeoff(errors: pd.DataFrame, selected: pd.Series) -> None:
    """Plot validation threshold tradeoffs across a compact quantile grid."""
    labels = errors["FLAG"].to_numpy(dtype=np.int8, copy=False)
    scores = errors["reconstruction_mse"].to_numpy(dtype=np.float64, copy=False)
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.999, 300)))
    rows = [calculate_metrics(labels, scores, float(threshold)) for threshold in thresholds]
    grid = pd.DataFrame(rows)
    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(grid["threshold"], grid["recall"], label="Recall")
    axes[0].plot(grid["threshold"], grid["specificity"], label="Specificity")
    axes[0].plot(grid["threshold"], grid["false_positive_rate"], label="FPR")
    axes[0].set_ylabel("Rate")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(grid["threshold"], grid["precision"], label="Precision")
    axes[1].plot(grid["threshold"], grid["F1"], label="F1")
    axes[1].plot(
        grid["threshold"],
        grid["predicted_anomaly_percentage"] / 100,
        label="Predicted anomaly fraction",
    )
    axes[1].set_xlabel("Reconstruction-MSE threshold (log scale)")
    axes[1].set_ylabel("Rate")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.axvline(
            selected["threshold"], color="tab:red", linestyle="--", label="Selected p95"
        )
        axis.set_xscale("log")
        axis.set_ylim(0, 1)
    figure.suptitle("Validation Threshold Tradeoffs (Weak Window Labels)")
    figure.tight_layout()
    temporary = TRADEOFF_PATH.with_name(TRADEOFF_PATH.stem + ".tmp.png")
    try:
        figure.savefig(temporary, dpi=150)
        temporary.replace(TRADEOFF_PATH)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def format_metrics_table(metrics: pd.DataFrame) -> str:
    """Render a concise Markdown candidate table."""
    lines = [
        "| Candidate | Threshold | Precision | Recall | F1 | Specificity | FPR | Predicted anomaly |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.threshold:.9g} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.F1:.4f} | {row.specificity:.4f} | "
            f"{row.false_positive_rate:.4f} | {row.predicted_anomaly_percentage:.2f}% |"
        )
    return "\n".join(lines)


def build_report(
    stage4: dict[str, object],
    distributions: dict[str, object],
    metrics: pd.DataFrame,
    selected: pd.Series,
    rationale: str,
    auc: dict[str, float],
    customer: dict[str, object],
    missingness: dict[str, object],
) -> str:
    """Build the human-readable Stage 5 evidence report."""
    normal = distributions["FLAG_0_normal"]
    theft = distributions["FLAG_1_theft_labelled"]
    best_f1 = metrics.loc[metrics["candidate"].eq("best_validation_f1")].iloc[0]
    best_balanced = metrics.loc[
        metrics["candidate"].eq("best_validation_balanced_accuracy")
    ].iloc[0]
    customer_aucs = customer["aggregation_auc"]
    return f"""# Stage 5 — Validation-Based Autoencoder Threshold

## Scope and Stage 4 verification

The completed Stage 4 checkpoint loaded successfully with architecture
`30 → 16 → 8 → 4 → 8 → 16 → 30` and 1,362 trainable parameters. Training was
executed manually on {str(stage4['device']).upper()} using {stage4['training_normal_windows']:,}
normal train windows. Early stopping occurred after {stage4['epochs_executed']} epochs;
the best qualifying checkpoint was epoch {stage4['best_epoch']} with normal-validation
MSE {stage4['best_validation_mse']:.12f}. Test data was not used.

## Validation reconstruction errors

| Population | Count | Mean | Median | Std | Min | P25 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLAG=0 | {normal['count']:,} | {normal['mean']:.8f} | {normal['median']:.8f} | {normal['standard_deviation']:.8f} | {normal['minimum']:.8f} | {normal['p25']:.8f} | {normal['p75']:.8f} | {normal['p90']:.8f} | {normal['p95']:.8f} | {normal['p99']:.8f} | {normal['maximum']:.8f} |
| FLAG=1 | {theft['count']:,} | {theft['mean']:.8f} | {theft['median']:.8f} | {theft['standard_deviation']:.8f} | {theft['minimum']:.8f} | {theft['p25']:.8f} | {theft['p75']:.8f} | {theft['p90']:.8f} | {theft['p95']:.8f} | {theft['p99']:.8f} | {theft['maximum']:.8f} |

The theft-labelled/normal median ratio is
{distributions['theft_to_normal_median_ratio']:.4f}; the mean ratio is
{distributions['theft_to_normal_mean_ratio']:.4f}. The score ranges overlap
substantially: {100 * distributions['overlap']['theft_at_or_below_normal_median_proportion']:.2f}%
of theft-labelled windows are at or below the normal median. FLAG is customer-level,
so these are weakly supervised window-level comparisons, not confirmed theft-event metrics.

Validation ROC-AUC is {auc['ROC_AUC']:.6f}; validation PR-AUC is
{auc['PR_AUC']:.6f}. These values describe ranking under weak customer labels only.

## Candidate thresholds

{format_metrics_table(metrics)}

The best-F1 threshold ({best_f1['threshold']:.9g}) attains F1
{best_f1['F1']:.4f}, but flags {best_f1['predicted_anomaly_percentage']:.2f}% of
all validation windows and has FPR {best_f1['false_positive_rate']:.4f}. The
best-balanced-accuracy candidate has balanced accuracy
{best_balanced['balanced_accuracy']:.4f}, with similarly high false-positive volume.

## Selected baseline threshold

**Threshold:** `{selected['threshold']:.12g}`  
**Method:** normal validation 95th percentile  
**Precision:** {selected['precision']:.6f}  
**Recall:** {selected['recall']:.6f}  
**F1:** {selected['F1']:.6f}  
**Specificity:** {selected['specificity']:.6f}  
**False-positive rate:** {selected['false_positive_rate']:.6f}  
**Predicted anomaly rate:** {selected['predicted_anomaly_percentage']:.4f}%

{rationale} This is a validation-selected baseline, not a globally optimal or
business-cost-optimal threshold. Lower cutoffs catch more theft-labelled windows
but generate more false alarms; higher cutoffs reduce false alarms but miss more
theft-labelled windows.

## Customer-level diagnostic

Validation contains {customer['customer_counts']['FLAG_0']:,} normal and
{customer['customer_counts']['FLAG_1']:,} theft-labelled customers. Customer
mean/max/p95 ROC-AUC values are {customer_aucs['mean_mse']['ROC_AUC']:.4f},
{customer_aucs['maximum_mse']['ROC_AUC']:.4f}, and
{customer_aucs['p95_mse']['ROC_AUC']:.4f}; their PR-AUC values are
{customer_aucs['mean_mse']['PR_AUC']:.4f},
{customer_aucs['maximum_mse']['PR_AUC']:.4f}, and
{customer_aucs['p95_mse']['PR_AUC']:.4f}. P95 aggregation improves separation
modestly relative to window-level ROC-AUC/PR-AUC, but separation remains weak and
does not justify a separate customer threshold in this stage.

## Missingness diagnostic

Missing fraction has weak negative association with reconstruction MSE:
overall Pearson {missingness['overall']['pearson']:.4f} and Spearman
{missingness['overall']['spearman']:.4f}; FLAG=0 Pearson/Spearman are
{missingness['FLAG_0']['pearson']:.4f}/{missingness['FLAG_0']['spearman']:.4f},
and FLAG=1 values are
{missingness['FLAG_1']['pearson']:.4f}/{missingness['FLAG_1']['spearman']:.4f}.
This does not indicate that the current Autoencoder score is primarily a proxy
for missing fraction, although missingness remains a potential confounder for
later sensitivity analysis.

## Boundary

No Autoencoder retraining, test-data access, test scoring, final test evaluation,
GAN work, or alternative-stride analysis was performed. The selected threshold
is now frozen in `models/autoencoder/ae_stride30_threshold.json` for later use.
"""


def write_outputs(
    errors: pd.DataFrame,
    stage4: dict[str, object],
    hashes: dict[str, str],
    distributions: dict[str, object],
    metrics: pd.DataFrame,
    selected: pd.Series,
    rationale: str,
    auc: dict[str, float],
    customer_table: pd.DataFrame,
    customer_diagnostic: dict[str, object],
    missingness_table: pd.DataFrame,
    missingness_diagnostic: dict[str, object],
) -> None:
    """Persist the threshold, evidence tables, report, and two concise plots."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(metrics, METRICS_PATH)
    save_csv(customer_table, CUSTOMER_DIAGNOSTICS_PATH)
    save_csv(missingness_table, MISSINGNESS_DIAGNOSTICS_PATH)

    selected_metrics = {
        key: int(selected[key]) if key in {"TP", "TN", "FP", "FN", "predicted_anomaly_count"}
        else float(selected[key])
        for key in (
            "threshold",
            "TP",
            "TN",
            "FP",
            "FN",
            "precision",
            "recall",
            "F1",
            "specificity",
            "false_positive_rate",
            "balanced_accuracy",
            "accuracy",
            "predicted_anomaly_count",
            "predicted_anomaly_percentage",
        )
    }
    threshold_artifact: dict[str, object] = {
        "stage": "Stage 5 — Validation-Based Anomaly Threshold",
        "model_checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "stride": 30,
        "window_length": 30,
        "threshold_value": selected_metrics["threshold"],
        "threshold_selection_method": "normal_validation_95th_percentile",
        "anomaly_rule": "reconstruction_mse > threshold_value",
        "validation_metrics": selected_metrics,
        "validation_sample_counts": {
            "normal_windows": int(errors["FLAG"].eq(0).sum()),
            "theft_labelled_windows": int(errors["FLAG"].eq(1).sum()),
            "total_windows": int(len(errors)),
        },
        "seed": int(stage4["config"]["seed"]),
        "model_context": {
            "architecture": EXPECTED_ARCHITECTURE,
            "latent_dimension": 4,
            "loss": stage4["config"]["loss"],
            "optimizer": stage4["config"]["optimizer"],
            "learning_rate": stage4["config"]["learning_rate"],
            "best_epoch": stage4["best_epoch"],
            "best_validation_mse": stage4["best_validation_mse"],
        },
        "selection_rationale": rationale,
        "weak_supervision_note": (
            "FLAG is customer-level; FLAG=1 windows are not confirmed theft events."
        ),
        "test_data_accessed": False,
        "stage4_input_sha256": hashes,
    }
    save_json(threshold_artifact, THRESHOLD_PATH)

    summary: dict[str, object] = {
        "stage": "Stage 5 — Validation-Based Anomaly Threshold",
        "stage4_verification": {key: value for key, value in stage4.items() if key != "config"},
        "validation_error_distributions": distributions,
        "validation_ranking_metrics": auc,
        "candidate_thresholds": metrics.replace({np.nan: None}).to_dict("records"),
        "selected_threshold": threshold_artifact,
        "customer_level_diagnostic": customer_diagnostic,
        "missingness_diagnostic": missingness_diagnostic,
        "test_data_accessed": False,
        "stage4_artifacts_unchanged": True,
    }
    save_json(summary, SUMMARY_PATH)
    report = build_report(
        stage4,
        distributions,
        metrics,
        selected,
        rationale,
        auc,
        customer_diagnostic,
        missingness_diagnostic,
    )
    save_text(report, REPORT_PATH)
    save_precision_recall_curve(errors, metrics, selected)
    save_threshold_tradeoff(errors, selected)


def main() -> None:
    """Run the complete validation-only Stage 5 analysis."""
    hashes_before = stage4_hashes()
    stage4 = validate_stage4_artifacts()
    errors = load_validation_errors()
    if len(errors) != (
        stage4["validation_normal_windows"] + stage4["validation_theft_windows"]
    ):
        raise ValueError("Validation error count disagrees with Stage 4 summary")

    distributions = summarize_error_distributions(errors)
    percentile_metrics = evaluate_percentile_thresholds(errors)
    supervised_metrics = evaluate_supervised_thresholds(errors)
    metrics = pd.concat([percentile_metrics, supervised_metrics], ignore_index=True)
    selected, rationale = select_baseline_threshold(metrics)
    labels = errors["FLAG"].to_numpy(dtype=np.int8, copy=False)
    scores = errors["reconstruction_mse"].to_numpy(dtype=np.float64, copy=False)
    auc = {
        "ROC_AUC": float(roc_auc_score(labels, scores)),
        "PR_AUC": float(average_precision_score(labels, scores)),
        "theft_label_prevalence": float(np.mean(labels)),
    }
    customer_table, customer_diagnostic = analyze_customer_level_scores(errors)
    customer_diagnostic["window_level_auc"] = auc
    missingness_table, missingness_diagnostic = analyze_missingness_relationship(errors)

    write_outputs(
        errors,
        stage4,
        hashes_before,
        distributions,
        metrics,
        selected,
        rationale,
        auc,
        customer_table,
        customer_diagnostic,
        missingness_table,
        missingness_diagnostic,
    )
    hashes_after = stage4_hashes()
    if hashes_before != hashes_after:
        changed = [path for path in hashes_before if hashes_before[path] != hashes_after[path]]
        raise RuntimeError(f"Stage 4 artifacts changed during Stage 5: {changed}")

    print("\nStage 5 validation-only threshold report")
    print(f"Validation windows: {len(errors):,}")
    print(f"ROC-AUC: {auc['ROC_AUC']:.6f}")
    print(f"PR-AUC: {auc['PR_AUC']:.6f}")
    print(f"Selected threshold: {selected['threshold']:.12g} (normal validation p95)")
    print(
        f"Precision={selected['precision']:.6f}, recall={selected['recall']:.6f}, "
        f"F1={selected['F1']:.6f}, specificity={selected['specificity']:.6f}, "
        f"FPR={selected['false_positive_rate']:.6f}, "
        f"predicted anomalies={selected['predicted_anomaly_percentage']:.4f}%"
    )
    print(f"Threshold artifact: {THRESHOLD_PATH}")
    print(f"Analysis directory: {ANALYSIS_DIR}")
    print("Stage 4 artifacts unchanged: True")
    print("Test data accessed: False")


if __name__ == "__main__":
    main()
