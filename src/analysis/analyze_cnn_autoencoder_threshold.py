"""Stage 10 CNN validation-only analysis and P95 threshold freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis import analyze_autoencoder_threshold as shared
from src.models.cnn_autoencoder import (
    CNNAutoencoder,
    architecture_config,
    count_trainable_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "cnn_autoencoder"
RESULTS_DIR = PROJECT_ROOT / "results" / "cnn_autoencoder" / "stride_30"
ANALYSIS_DIR = RESULTS_DIR / "threshold_analysis"
CHECKPOINT_PATH = MODEL_DIR / "cnn_ae_stride30_baseline.pt"
CONFIG_PATH = MODEL_DIR / "cnn_ae_stride30_baseline_config.json"
THRESHOLD_PATH = MODEL_DIR / "cnn_ae_stride30_threshold.json"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"
SUMMARY_PATH = RESULTS_DIR / "training_summary.json"
ERRORS_PATH = RESULTS_DIR / "val_reconstruction_errors.parquet"
TRAINING_PATHS = (
    CHECKPOINT_PATH,
    CONFIG_PATH,
    HISTORY_PATH,
    SUMMARY_PATH,
    RESULTS_DIR / "training_curve.png",
    ERRORS_PATH,
    RESULTS_DIR / "validation_error_distribution.png",
    PROJECT_ROOT / "src" / "models" / "cnn_autoencoder.py",
    PROJECT_ROOT / "src" / "training" / "train_cnn_autoencoder.py",
)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    missing = [str(path) for path in TRAINING_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CNN training artifacts are missing: {missing}")
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path)
        for path in TRAINING_PATHS
    }


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def verify_training() -> dict[str, object]:
    """Verify the manual CNN run without invoking training code."""
    config = load_json(CONFIG_PATH)
    summary = load_json(SUMMARY_PATH)
    history = pd.read_csv(HISTORY_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = CNNAutoencoder()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    expected_architecture = architecture_config()
    if checkpoint["architecture_configuration"] != expected_architecture:
        raise ValueError("CNN checkpoint architecture differs from Stage 9")
    if config["model"] != expected_architecture:
        raise ValueError("CNN config architecture differs from Stage 9")
    if count_trainable_parameters(model) != 2209:
        raise ValueError("CNN trainable parameter count changed")
    if checkpoint["input_channels"] != 1 or checkpoint["input_length"] != 30:
        raise ValueError("CNN checkpoint input shape changed")
    if checkpoint["latent_dim"] != 8 or checkpoint["stride"] != 30:
        raise ValueError("CNN checkpoint latent/stride configuration changed")
    if config["training_population"] != "usable stride-30 train windows with FLAG == 0 only":
        raise ValueError("CNN training population policy changed")
    if config["early_stopping_population"] != "usable stride-30 validation windows with FLAG == 0 only":
        raise ValueError("CNN early-stopping population policy changed")
    if config["test_set_policy"] != "not loaded or used during CNN training":
        raise ValueError("CNN training test policy changed")
    if summary["test_set_used"] is not False or summary["anomaly_threshold_selected"] is not False:
        raise ValueError("CNN training summary violates the Stage 9 boundary")
    if list(history.columns) != ["epoch", "train_mse", "val_normal_mse"]:
        raise ValueError("CNN training history schema changed")
    if len(history) != int(summary["epochs_executed"]):
        raise ValueError("CNN history length differs from training summary")
    best_epoch = int(checkpoint["best_epoch"])
    best_row = history.loc[history["epoch"].eq(best_epoch)]
    if len(best_row) != 1 or not np.isclose(
        best_row.iloc[0]["val_normal_mse"],
        float(checkpoint["best_validation_loss"]), rtol=0, atol=1e-12,
    ):
        raise ValueError("CNN best checkpoint is not aligned with its history")
    if not np.isclose(
        float(summary["best_validation_mse"]),
        float(checkpoint["best_validation_loss"]), rtol=0, atol=1e-15,
    ):
        raise ValueError("CNN checkpoint and summary best losses disagree")
    return {
        "device": summary["device"],
        "training_normal_windows": int(summary["training_normal_window_count"]),
        "validation_normal_windows": int(summary["validation_normal_window_count"]),
        "validation_theft_windows": int(summary["validation_theft_window_count"]),
        "maximum_epochs": int(config["maximum_epochs"]),
        "epochs_executed": int(summary["epochs_executed"]),
        "best_epoch": best_epoch,
        "best_validation_mse": float(checkpoint["best_validation_loss"]),
        "runtime_seconds": float(summary["runtime_seconds"]),
        "early_stopping_occurred": bool(summary["early_stopping_occurred"]),
        "trainable_parameters": 2209,
        "test_set_used": False,
        "threshold_selected_during_training": False,
        "config": config,
    }


def load_errors() -> pd.DataFrame:
    errors = pd.read_parquet(ERRORS_PATH, engine="pyarrow")
    expected = ["window_id", "CONS_NO", "FLAG", "reconstruction_mse", "missing_fraction"]
    if list(errors.columns) != expected:
        raise ValueError(f"CNN validation score columns differ from {expected}")
    if not errors["window_id"].is_unique or errors["window_id"].isna().any():
        raise ValueError("CNN validation window IDs must be unique and present")
    scores = errors["reconstruction_mse"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("CNN validation scores must be finite and nonnegative")
    if set(errors["FLAG"].unique()) != {0, 1}:
        raise ValueError("CNN validation scores must contain FLAG 0 and 1")
    return errors


def configure_shared_plots() -> None:
    """Point shared plotting helpers only at isolated CNN output paths."""
    shared.ANALYSIS_DIR = ANALYSIS_DIR
    shared.PR_CURVE_PATH = ANALYSIS_DIR / "precision_recall_curve.png"
    shared.TRADEOFF_PATH = ANALYSIS_DIR / "threshold_tradeoff.png"


def build_report(
    verification: dict[str, object], distributions: dict[str, object],
    metrics: pd.DataFrame, selected: pd.Series, auc: dict[str, float],
    customer: dict[str, object], missingness: dict[str, object],
) -> str:
    normal = distributions["FLAG_0_normal"]
    theft = distributions["FLAG_1_theft_labelled"]
    candidate_table = shared.format_metrics_table(metrics)
    exceedance = distributions["theft_exceeding_normal_percentiles"]
    return f"""# Stage 10 — CNN Validation Threshold Analysis

The manually trained 2,209-parameter CNN checkpoint uses input `(B,1,30)`,
latent dimension 8, and the Stage 9 architecture. It ran on
{str(verification['device']).upper()} for {verification['epochs_executed']} epochs;
epoch {verification['best_epoch']} is the best qualifying checkpoint with normal
validation MSE {verification['best_validation_mse']:.12f}. Test data was not accessed.

| Population | Count | Mean | Median | Std | Min | P25 | P50 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | {normal['count']:,} | {normal['mean']:.8f} | {normal['median']:.8f} | {normal['standard_deviation']:.8f} | {normal['minimum']:.8f} | {normal['p25']:.8f} | {normal['p50']:.8f} | {normal['p75']:.8f} | {normal['p90']:.8f} | {normal['p95']:.8f} | {normal['p99']:.8f} | {normal['maximum']:.8f} |
| Theft-labelled | {theft['count']:,} | {theft['mean']:.8f} | {theft['median']:.8f} | {theft['standard_deviation']:.8f} | {theft['minimum']:.8f} | {theft['p25']:.8f} | {theft['p50']:.8f} | {theft['p75']:.8f} | {theft['p90']:.8f} | {theft['p95']:.8f} | {theft['p99']:.8f} | {theft['maximum']:.8f} |

Theft/normal median and mean ratios are
{distributions['theft_to_normal_median_ratio']:.4f} and
{distributions['theft_to_normal_mean_ratio']:.4f}. Theft-labelled exceedance of
normal P90/P95/P97.5/P99 is
{100*exceedance['normal_p90']['theft_proportion_exceeding']:.2f}% /
{100*exceedance['normal_p95']['theft_proportion_exceeding']:.2f}% /
{100*exceedance['normal_p97.5']['theft_proportion_exceeding']:.2f}% /
{100*exceedance['normal_p99']['theft_proportion_exceeding']:.2f}%.

Validation ROC-AUC is {auc['ROC_AUC']:.6f}; PR-AUC is {auc['PR_AUC']:.6f}.
FLAG is customer-level weak supervision.

## Candidate thresholds

{candidate_table}

The authoritative threshold is normal-validation P95:
`{selected['threshold']:.12g}`. Precision/recall/F1 are
{selected['precision']:.6f}/{selected['recall']:.6f}/{selected['F1']:.6f};
specificity is {selected['specificity']:.6f}, FPR is
{selected['false_positive_rate']:.6f}, balanced accuracy is
{selected['balanced_accuracy']:.6f}, and anomaly rate is
{selected['predicted_anomaly_percentage']:.4f}%. Best-F1 and
best-balanced-accuracy candidates are diagnostic only.

Customer mean/max/P95 ROC-AUC values are
{customer['aggregation_auc']['mean_mse']['ROC_AUC']:.4f}/
{customer['aggregation_auc']['maximum_mse']['ROC_AUC']:.4f}/
{customer['aggregation_auc']['p95_mse']['ROC_AUC']:.4f}. Overall
missingness Pearson/Spearman values are
{missingness['overall']['pearson']:.4f}/{missingness['overall']['spearman']:.4f}.
No retraining, test scoring, or test-driven selection occurred.
"""


def main() -> None:
    hashes_before = protected_hashes()
    verification = verify_training()
    errors = load_errors()
    if len(errors) != verification["validation_normal_windows"] + verification["validation_theft_windows"]:
        raise ValueError("CNN validation score count differs from training summary")
    distributions = shared.summarize_error_distributions(errors)
    percentile_metrics = shared.evaluate_percentile_thresholds(errors)
    supervised_metrics = shared.evaluate_supervised_thresholds(errors)
    metrics = pd.concat([percentile_metrics, supervised_metrics], ignore_index=True)
    selected, rationale = shared.select_baseline_threshold(metrics)
    labels = errors["FLAG"].to_numpy(dtype=np.int8, copy=False)
    scores = errors["reconstruction_mse"].to_numpy(dtype=np.float64, copy=False)
    auc = {
        "ROC_AUC": float(roc_auc_score(labels, scores)),
        "PR_AUC": float(average_precision_score(labels, scores)),
        "theft_label_prevalence": float(np.mean(labels)),
    }
    customer_table, customer = shared.analyze_customer_level_scores(errors)
    missingness_table, missingness = shared.analyze_missingness_relationship(errors)
    selected_metrics = {
        key: int(selected[key]) if key in {"TP", "TN", "FP", "FN", "predicted_anomaly_count"}
        else float(selected[key])
        for key in ("threshold", "TP", "TN", "FP", "FN", "precision", "recall", "F1", "specificity", "false_positive_rate", "balanced_accuracy", "accuracy", "predicted_anomaly_count", "predicted_anomaly_percentage")
    }
    threshold_artifact = {
        "stage": "Stage 10 — CNN Validation-Based Anomaly Threshold",
        "model_checkpoint": CHECKPOINT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "stride": 30, "window_length": 30,
        "threshold_value": selected_metrics["threshold"],
        "threshold_selection_method": "normal_validation_95th_percentile",
        "anomaly_rule": "reconstruction_mse > threshold_value",
        "validation_metrics": selected_metrics,
        "validation_sample_counts": {
            "normal_windows": int(errors["FLAG"].eq(0).sum()),
            "theft_labelled_windows": int(errors["FLAG"].eq(1).sum()),
            "total_windows": int(len(errors)),
        },
        "model_context": {
            "architecture": architecture_config(), "trainable_parameters": 2209,
            "latent_dimension": 8, "loss": "MSELoss", "optimizer": "Adam",
            "learning_rate": 0.001, "best_epoch": verification["best_epoch"],
            "best_validation_mse": verification["best_validation_mse"],
        },
        "selection_rationale": rationale,
        "weak_supervision_note": "FLAG is customer-level; positive windows are not confirmed theft events.",
        "test_data_accessed": False,
        "cnn_training_input_sha256": hashes_before,
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    shared.save_json(threshold_artifact, THRESHOLD_PATH)
    shared.save_csv(metrics, ANALYSIS_DIR / "threshold_metrics.csv")
    shared.save_csv(customer_table, ANALYSIS_DIR / "customer_level_diagnostics.csv")
    shared.save_csv(missingness_table, ANALYSIS_DIR / "missingness_group_summary.csv")
    shared.save_json({
        "stage": "Stage 10 — CNN Validation Threshold",
        "training_verification": {k:v for k,v in verification.items() if k != "config"},
        "validation_error_distributions": distributions,
        "validation_ranking_metrics": auc,
        "candidate_thresholds": metrics.replace({np.nan: None}).to_dict("records"),
        "selected_threshold": threshold_artifact,
        "customer_level_diagnostic": customer,
        "missingness_diagnostic": missingness,
        "test_data_accessed": False,
        "cnn_training_artifacts_unchanged": True,
    }, ANALYSIS_DIR / "threshold_summary.json")
    shared.save_text(build_report(verification, distributions, metrics, selected, auc, customer, missingness), ANALYSIS_DIR / "threshold_report.md")
    configure_shared_plots()
    shared.save_precision_recall_curve(errors, metrics, selected)
    shared.save_threshold_tradeoff(errors, selected)
    if hashes_before != protected_hashes():
        raise RuntimeError("CNN training artifacts changed during threshold selection")
    print(f"Validation windows: {len(errors):,}")
    print(f"ROC-AUC: {auc['ROC_AUC']:.6f}; PR-AUC: {auc['PR_AUC']:.6f}")
    print(f"Frozen normal-validation P95 threshold: {selected['threshold']:.12g}")
    print(f"Threshold artifact: {THRESHOLD_PATH}")
    print("CNN training artifacts unchanged: True")
    print("Test data accessed: False")


if __name__ == "__main__":
    main()
