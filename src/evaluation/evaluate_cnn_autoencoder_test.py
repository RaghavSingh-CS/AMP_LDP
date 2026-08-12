"""One-time frozen test evaluation for the Stage 9 stride-30 CNN Autoencoder.

This module is inference-only. It contains no optimizer, backward pass,
threshold search, or model-selection logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.evaluation import evaluate_autoencoder_test as shared
from src.data.window_dataset import ProcessedSplitData, WindowDataset
from src.models.cnn_autoencoder import (
    CNNAutoencoder,
    architecture_config,
    count_trainable_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "cnn_autoencoder"
RESULTS_DIR = PROJECT_ROOT / "results" / "cnn_autoencoder" / "stride_30"
OUTPUT_DIR = RESULTS_DIR / "test_evaluation"
CHECKPOINT_PATH = MODEL_DIR / "cnn_ae_stride30_baseline.pt"
THRESHOLD_PATH = MODEL_DIR / "cnn_ae_stride30_threshold.json"
FILTER_SUMMARY_PATH = PROJECT_ROOT / "data/windows/stride_30/usable/filter_summary.json"
TEST_MANIFEST_PATH = FILTER_SUMMARY_PATH.parent / "test_windows.parquet"
THRESHOLD_SUMMARY_PATH = RESULTS_DIR / "threshold_analysis/threshold_summary.json"

ERRORS_PATH = OUTPUT_DIR / "test_reconstruction_errors.parquet"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
REPORT_PATH = OUTPUT_DIR / "test_report.md"
CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.png"
ERROR_DISTRIBUTION_PATH = OUTPUT_DIR / "test_error_distribution.png"
VALIDATION_COMPARISON_PATH = OUTPUT_DIR / "validation_vs_test.csv"
CUSTOMER_DIAGNOSTICS_PATH = OUTPUT_DIR / "customer_level_test_diagnostics.csv"
CUSTOMER_ANOMALY_PATH = OUTPUT_DIR / "customer_anomaly_fraction_summary.csv"
MISSINGNESS_PATH = OUTPUT_DIR / "missingness_test_summary.csv"
TOP_CASES_PATH = OUTPUT_DIR / "top_test_error_cases.csv"
FINAL_SUMMARY_PATH = OUTPUT_DIR / "baseline_final_summary.json"

THRESHOLD_OUTPUTS = (
    THRESHOLD_PATH,
    RESULTS_DIR / "threshold_analysis/threshold_metrics.csv",
    THRESHOLD_SUMMARY_PATH,
    RESULTS_DIR / "threshold_analysis/threshold_report.md",
    RESULTS_DIR / "threshold_analysis/precision_recall_curve.png",
    RESULTS_DIR / "threshold_analysis/threshold_tradeoff.png",
    RESULTS_DIR / "threshold_analysis/customer_level_diagnostics.csv",
    RESULTS_DIR / "threshold_analysis/missingness_group_summary.csv",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def protected_paths() -> tuple[Path, ...]:
    threshold = load_json(THRESHOLD_PATH)
    filter_summary = load_json(FILTER_SUMMARY_PATH)
    paths = [PROJECT_ROOT / relative for relative in threshold["cnn_training_input_sha256"]]
    paths.extend(THRESHOLD_OUTPUTS)
    paths.extend(
        PROJECT_ROOT / relative
        for relative in filter_summary["integrity"]["protected_sha256"]
    )
    paths.extend(
        FILTER_SUMMARY_PATH.parent / f"{split}_windows.parquet"
        for split in ("train", "val", "test")
    )
    paths.extend((FILTER_SUMMARY_PATH, TEST_MANIFEST_PATH))
    unique = tuple(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected CNN lineage artifacts are missing: {missing}")
    return unique


def protected_hashes() -> dict[str, str]:
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path)
        for path in protected_paths()
    }


def load_frozen_inputs(device: torch.device) -> tuple[CNNAutoencoder, dict, float, dict]:
    threshold_artifact = load_json(THRESHOLD_PATH)
    threshold = float(threshold_artifact["threshold_value"])
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Frozen CNN threshold must be finite and nonnegative")
    if threshold_artifact["threshold_selection_method"] != "normal_validation_95th_percentile":
        raise ValueError("Unexpected CNN threshold-selection method")
    if threshold_artifact["anomaly_rule"] != "reconstruction_mse > threshold_value":
        raise ValueError("Unexpected CNN anomaly rule")
    if threshold_artifact["test_data_accessed"] is not False:
        raise ValueError("Threshold artifact test-access flag is invalid")
    if Path(threshold_artifact["model_checkpoint"]) != Path(
        "models/cnn_autoencoder/cnn_ae_stride30_baseline.pt"
    ):
        raise ValueError("CNN threshold references a different checkpoint")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    if checkpoint["architecture_configuration"] != architecture_config():
        raise ValueError("CNN checkpoint architecture differs from source")
    if checkpoint["window_length"] != 30 or checkpoint["stride"] != 30:
        raise ValueError("CNN checkpoint window configuration changed")
    if checkpoint["latent_dim"] != 8:
        raise ValueError("CNN checkpoint latent dimension changed")
    model = CNNAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if count_trainable_parameters(model) != 2209:
        raise ValueError("CNN parameter count changed")
    model.eval()
    return model, checkpoint, threshold, threshold_artifact


def load_test_dataset() -> tuple[ProcessedSplitData, WindowDataset, WindowDataset]:
    store = ProcessedSplitData.load("test")
    normal = WindowDataset("test", 0, stride=30, store=store, include_metadata=True)
    theft = WindowDataset("test", 1, stride=30, store=store, include_metadata=True)
    expected = int(load_json(FILTER_SUMMARY_PATH)["splits"]["test"]["retained_windows"])
    if len(normal) + len(theft) != expected:
        raise ValueError("Test-window count differs from the frozen usable manifest")
    return store, normal, theft


@torch.inference_mode()
def score_dataset(
    model: CNNAutoencoder,
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches: list[np.ndarray] = []
    for inputs in loader:
        inputs = inputs.to(device=device, dtype=torch.float32).unsqueeze(1)
        reconstructions = model(inputs)
        scores = torch.mean((reconstructions - inputs) ** 2, dim=(1, 2))
        if not torch.isfinite(scores).all() or torch.any(scores < 0):
            raise FloatingPointError("Invalid CNN test reconstruction error")
        batches.append(scores.cpu().numpy())
    values = np.concatenate(batches).astype(np.float32, copy=False)
    metadata = dataset.metadata
    if len(metadata) != len(values):
        raise RuntimeError("CNN test scores and metadata are misaligned")
    metadata["reconstruction_mse"] = values
    return metadata


def compute_errors(model, normal, theft, batch_size, device) -> pd.DataFrame:
    errors = pd.concat(
        [score_dataset(model, normal, batch_size, device),
         score_dataset(model, theft, batch_size, device)],
        ignore_index=True,
    )
    dates = pd.read_parquet(TEST_MANIFEST_PATH, columns=["window_id", "start_date", "end_date"])
    errors = errors.merge(dates, on="window_id", how="left", validate="one_to_one")
    if errors[["start_date", "end_date"]].isna().any().any():
        raise ValueError("CNN test date metadata did not align")
    return errors


def configure_shared_paths() -> None:
    shared.CONFUSION_PATH = CONFUSION_PATH
    shared.ERROR_DISTRIBUTION_PATH = ERROR_DISTRIBUTION_PATH
    shared.THRESHOLD_PATH = THRESHOLD_PATH
    shared.VALIDATION_THRESHOLD_SUMMARY_PATH = THRESHOLD_SUMMARY_PATH


def build_report(threshold, metrics, distributions, comparison, generalization, customer, missingness) -> str:
    normal = distributions["FLAG_0_normal"]
    theft = distributions["FLAG_1_theft_labelled"]
    rows = "\n".join(
        f"| {row.metric} | {row.validation:.6f} | {row.test:.6f} | {row.absolute_difference:.6f} |"
        for row in comparison.itertuples(index=False)
    )
    return f"""# Stage 10 — Frozen CNN Test Evaluation

The frozen CNN checkpoint was evaluated once with validation P95 threshold
`{threshold:.12g}` and rule `reconstruction_mse > threshold`. No retraining,
threshold adjustment, or test-driven selection occurred.

## Window-level results

Test windows: {normal['count'] + theft['count']:,} ({normal['count']:,} normal;
{theft['count']:,} theft-labelled).

| Metric | Value |
|---|---:|
| TP / TN / FP / FN | {metrics['TP']:,} / {metrics['TN']:,} / {metrics['FP']:,} / {metrics['FN']:,} |
| Precision | {metrics['precision']:.6f} |
| Recall | {metrics['recall']:.6f} |
| F1 | {metrics['F1']:.6f} |
| Specificity | {metrics['specificity']:.6f} |
| ROC-AUC | {metrics['ROC_AUC']:.6f} |
| PR-AUC | {metrics['PR_AUC']:.6f} |
| Predicted anomaly percentage | {metrics['predicted_anomaly_percentage']:.6f}% |

## Validation versus test

Classification: **{generalization}**.

| Metric | Validation | Test | Absolute difference |
|---|---:|---:|---:|
{rows}

Customer p95-MSE ROC-AUC is {customer['aggregation_auc']['p95_mse']['ROC_AUC']:.6f}.
The reconstruction-MSE/missing-fraction Spearman correlation is
{missingness['overall']['spearman']:.6f}.

`FLAG` is customer-level weak supervision: positive windows are not confirmed
theft events. Customer aggregates remain diagnostics, not a tuned classifier.
"""


def save_outputs(errors, threshold, threshold_artifact, checkpoint, hashes) -> dict[str, object]:
    configure_shared_paths()
    errors = shared.apply_threshold(errors, threshold)
    metrics = shared.calculate_window_metrics(errors, threshold)
    distributions = shared.summarize_error_distributions(errors, threshold)
    customer_table, customer_anomaly, customer = shared.calculate_customer_diagnostics(errors)
    missing_table, missingness = shared.analyze_missingness(errors)
    comparison, generalization = shared.compare_validation_test(metrics)
    cases = shared.top_error_cases(errors)

    shared.save_parquet(errors, ERRORS_PATH)
    shared.save_csv(comparison, VALIDATION_COMPARISON_PATH)
    shared.save_csv(customer_table, CUSTOMER_DIAGNOSTICS_PATH)
    shared.save_csv(customer_anomaly, CUSTOMER_ANOMALY_PATH)
    shared.save_csv(missing_table, MISSINGNESS_PATH)
    shared.save_csv(cases, TOP_CASES_PATH)
    shared.save_confusion_matrix(metrics)
    shared.save_error_distribution(errors, threshold)

    payload = {
        "stage": "Stage 10 — Frozen CNN Test Evaluation",
        "frozen_threshold": threshold,
        "threshold_source": THRESHOLD_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "window_level_test_metrics": metrics,
        "error_distributions": distributions,
        "validation_vs_test": {"classification": generalization, "metrics": comparison.to_dict("records")},
        "customer_level_diagnostic": customer,
        "missingness_diagnostic": missingness,
        "retraining_performed": False,
        "threshold_modified": False,
        "test_driven_tuning_performed": False,
        "protected_pretest_sha256": hashes,
    }
    shared.save_json(payload, METRICS_PATH)
    final = {
        "artifact_role": "final stride-30 CNN Autoencoder performance record",
        "model": {
            "architecture": architecture_config(),
            "trainable_parameters": 2209,
            "checkpoint": CHECKPOINT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_sha256": hashes[CHECKPOINT_PATH.relative_to(PROJECT_ROOT).as_posix()],
            "best_epoch": checkpoint["best_epoch"],
            "best_validation_mse": checkpoint["best_validation_loss"],
        },
        "threshold": {
            "value": threshold,
            "selection_method": threshold_artifact["threshold_selection_method"],
            "artifact": THRESHOLD_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "artifact_sha256": hashes[THRESHOLD_PATH.relative_to(PROJECT_ROOT).as_posix()],
            "rule": "reconstruction_mse > threshold",
        },
        "test_sample_counts": {
            "normal_windows": int(errors["FLAG"].eq(0).sum()),
            "theft_labelled_windows": int(errors["FLAG"].eq(1).sum()),
            "total_windows": int(len(errors)),
        },
        "test_metrics": {key: metrics[key] for key in (
            "TP", "TN", "FP", "FN", "precision", "recall", "F1", "specificity",
            "false_positive_rate", "balanced_accuracy", "accuracy", "ROC_AUC", "PR_AUC",
            "predicted_anomaly_percentage")},
        "validation_test_generalization": generalization,
        "weak_supervision_caveat": "FLAG is customer-level; window metrics are not confirmed theft-event metrics.",
        "frozen_evaluation_guarantees": {
            "checkpoint_unchanged": True, "threshold_unchanged": True,
            "retraining_performed": False, "test_driven_tuning_performed": False,
        },
    }
    shared.save_json(final, FINAL_SUMMARY_PATH)
    shared.save_text(build_report(threshold, metrics, distributions, comparison, generalization, customer, missingness), REPORT_PATH)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen CNN checkpoint on test exactly once.")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    before = protected_hashes()
    model, checkpoint, threshold, threshold_artifact = load_frozen_inputs(device)
    _store, normal, theft = load_test_dataset()
    errors = compute_errors(model, normal, theft, args.batch_size, device)
    payload = save_outputs(errors, threshold, threshold_artifact, checkpoint, before)
    after = protected_hashes()
    if before != after:
        raise RuntimeError("A protected pre-test artifact changed during CNN evaluation")
    metrics = payload["window_level_test_metrics"]
    print(f"CNN test windows: {len(errors):,}")
    print(f"ROC-AUC: {metrics['ROC_AUC']:.6f}; PR-AUC: {metrics['PR_AUC']:.6f}; F1: {metrics['F1']:.6f}")
    print("CNN checkpoint and frozen threshold unchanged: True")
    print("Retraining or test-driven tuning performed: False")


if __name__ == "__main__":
    main()
