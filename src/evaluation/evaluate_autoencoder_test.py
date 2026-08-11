"""Stage 6: one-time frozen test evaluation of the stride-30 Autoencoder.

The checkpoint and Stage 5 threshold are immutable inputs. This module performs
inference only: it contains no optimizer, backward pass, threshold search, or
test-driven model selection.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader

from src.analysis.analyze_autoencoder_threshold import (
    analyze_customer_level_scores,
    analyze_missingness_relationship,
    calculate_metrics,
    distribution_summary,
)
from src.data.window_dataset import ProcessedSplitData, WindowDataset
from src.models.autoencoder import Autoencoder, architecture_config, count_trainable_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "autoencoder"
RESULTS_DIR = PROJECT_ROOT / "results" / "autoencoder" / "stride_30"
OUTPUT_DIR = RESULTS_DIR / "test_evaluation"

CHECKPOINT_PATH = MODEL_DIR / "ae_stride30_baseline.pt"
THRESHOLD_PATH = MODEL_DIR / "ae_stride30_threshold.json"
STAGE3C_SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "windows" / "stride_30" / "usable" / "filter_summary.json"
)
TEST_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "windows" / "stride_30" / "usable" / "test_windows.parquet"
)
VALIDATION_THRESHOLD_SUMMARY_PATH = (
    RESULTS_DIR / "threshold_analysis" / "threshold_summary.json"
)

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

EXPECTED_ARCHITECTURE = [30, 16, 8, 4, 8, 16, 30]
WINDOW_LENGTH = 30
STRIDE = 30
DEFAULT_BATCH_SIZE = 256

STAGE5_OUTPUTS = (
    MODEL_DIR / "ae_stride30_threshold.json",
    RESULTS_DIR / "threshold_analysis" / "threshold_metrics.csv",
    RESULTS_DIR / "threshold_analysis" / "threshold_summary.json",
    RESULTS_DIR / "threshold_analysis" / "threshold_report.md",
    RESULTS_DIR / "threshold_analysis" / "precision_recall_curve.png",
    RESULTS_DIR / "threshold_analysis" / "threshold_tradeoff.png",
    RESULTS_DIR / "threshold_analysis" / "customer_level_diagnostics.csv",
    RESULTS_DIR / "threshold_analysis" / "missingness_group_summary.csv",
    PROJECT_ROOT / "src" / "analysis" / "analyze_autoencoder_threshold.py",
)

def configure_stride(stride: int) -> None:
    """Route frozen evaluation to isolated artifacts for one trained stride."""
    global STRIDE, RESULTS_DIR, OUTPUT_DIR, CHECKPOINT_PATH, THRESHOLD_PATH
    global STAGE3C_SUMMARY_PATH, TEST_MANIFEST_PATH
    global VALIDATION_THRESHOLD_SUMMARY_PATH, ERRORS_PATH, METRICS_PATH
    global REPORT_PATH, CONFUSION_PATH, ERROR_DISTRIBUTION_PATH
    global VALIDATION_COMPARISON_PATH, CUSTOMER_DIAGNOSTICS_PATH
    global CUSTOMER_ANOMALY_PATH, MISSINGNESS_PATH, TOP_CASES_PATH
    global FINAL_SUMMARY_PATH, STAGE5_OUTPUTS

    if stride not in (30, 15):
        raise ValueError(f"Supported Autoencoder strides are 30 and 15; got {stride}")
    STRIDE = stride
    RESULTS_DIR = PROJECT_ROOT / "results" / "autoencoder" / f"stride_{stride}"
    OUTPUT_DIR = RESULTS_DIR / "test_evaluation"
    CHECKPOINT_PATH = MODEL_DIR / f"ae_stride{stride}_baseline.pt"
    THRESHOLD_PATH = MODEL_DIR / f"ae_stride{stride}_threshold.json"
    STAGE3C_SUMMARY_PATH = (
        PROJECT_ROOT
        / "data"
        / "windows"
        / f"stride_{stride}"
        / "usable"
        / "filter_summary.json"
    )
    TEST_MANIFEST_PATH = STAGE3C_SUMMARY_PATH.parent / "test_windows.parquet"
    VALIDATION_THRESHOLD_SUMMARY_PATH = (
        RESULTS_DIR / "threshold_analysis" / "threshold_summary.json"
    )
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
    STAGE5_OUTPUTS = (
        THRESHOLD_PATH,
        RESULTS_DIR / "threshold_analysis" / "threshold_metrics.csv",
        RESULTS_DIR / "threshold_analysis" / "threshold_summary.json",
        RESULTS_DIR / "threshold_analysis" / "threshold_report.md",
        RESULTS_DIR / "threshold_analysis" / "precision_recall_curve.png",
        RESULTS_DIR / "threshold_analysis" / "threshold_tradeoff.png",
        RESULTS_DIR / "threshold_analysis" / "customer_level_diagnostics.csv",
        RESULTS_DIR / "threshold_analysis" / "missingness_group_summary.csv",
        PROJECT_ROOT / "src" / "analysis" / "analyze_autoencoder_threshold.py",
    )



def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash of one protected artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    """Load one JSON object."""
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def protected_paths() -> tuple[Path, ...]:
    """Resolve the immutable Stage 1–5 lineage without changing it."""
    stage3c = load_json(STAGE3C_SUMMARY_PATH)
    threshold = load_json(THRESHOLD_PATH)
    paths: list[Path] = [
        PROJECT_ROOT / relative for relative in stage3c["integrity"]["protected_sha256"]
    ]
    paths.extend(
        [
            PROJECT_ROOT
            / "data"
            / "windows"
            / f"stride_{STRIDE}"
            / "usable"
            / f"{split}_windows.parquet"
            for split in ("train", "val", "test")
        ]
    )
    paths.append(STAGE3C_SUMMARY_PATH)
    paths.extend(
        PROJECT_ROOT / relative for relative in threshold["stage4_input_sha256"]
    )
    paths.extend(STAGE5_OUTPUTS)
    unique = tuple(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected Stage 1–5 artifacts are missing: {missing}")
    return unique


def protected_hashes() -> dict[str, str]:
    """Hash immutable Stage 1–5 lineage before and after test evaluation."""
    return {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in protected_paths()
    }


def load_frozen_threshold() -> tuple[float, dict[str, object]]:
    """Load the authoritative Stage 5 threshold and validate its semantics."""
    artifact = load_json(THRESHOLD_PATH)
    threshold = float(artifact["threshold_value"])
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Frozen threshold must be finite and nonnegative")
    if artifact["threshold_selection_method"] != "normal_validation_95th_percentile":
        raise ValueError("Unexpected frozen-threshold selection method")
    if artifact["anomaly_rule"] != "reconstruction_mse > threshold_value":
        raise ValueError("Unexpected frozen anomaly comparison rule")
    if Path(artifact["model_checkpoint"]) != Path(f"models/autoencoder/ae_stride{STRIDE}_baseline.pt"):
        raise ValueError("Threshold artifact references a different checkpoint")
    if int(artifact["stride"]) != STRIDE:
        raise ValueError("Threshold artifact stride differs from requested stride")
    return threshold, artifact


def load_model(device: torch.device) -> tuple[Autoencoder, dict[str, object]]:
    """Load the frozen Stage 4 model exactly and put it in inference mode."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    architecture = checkpoint["architecture_configuration"]
    if architecture["layer_dimensions"] != EXPECTED_ARCHITECTURE:
        raise ValueError("Checkpoint architecture is not the frozen Stage 4 baseline")
    if checkpoint["input_dimension"] != 30 or checkpoint["latent_dimension"] != 4:
        raise ValueError("Checkpoint input or latent dimension changed")
    if checkpoint["window_length"] != WINDOW_LENGTH or checkpoint["stride"] != STRIDE:
        raise ValueError("Checkpoint window configuration changed")
    model = Autoencoder().to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint state mismatch: {incompatible}")
    if architecture_config()["layer_dimensions"] != EXPECTED_ARCHITECTURE:
        raise ValueError("Source model architecture changed")
    if count_trainable_parameters(model) != 1362:
        raise ValueError("Source model parameter count changed")
    model.eval()
    if model.training:
        raise RuntimeError("Model did not enter eval mode")
    return model, checkpoint


def load_test_dataset() -> tuple[ProcessedSplitData, WindowDataset, WindowDataset]:
    """Build normal and theft-labelled views sharing one compact test matrix."""
    store = ProcessedSplitData.load("test")
    normal = WindowDataset(
        "test", 0, stride=STRIDE, store=store, include_metadata=True
    )
    theft = WindowDataset(
        "test", 1, stride=STRIDE, store=store, include_metadata=True
    )
    expected = int(load_json(STAGE3C_SUMMARY_PATH)["splits"]["test"]["retained_windows"])
    if len(normal) + len(theft) != expected:
        raise ValueError("Usable test-window count differs from Stage 3C")
    if normal.flag != 0 or theft.flag != 1:
        raise ValueError("Test Dataset FLAG filters are inconsistent")
    return store, normal, theft


def make_loader(
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
) -> DataLoader[Tensor]:
    """Create a deterministic, non-shuffled inference DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


@torch.inference_mode()
def score_dataset(
    model: Autoencoder,
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """Calculate per-window reconstruction MSE without gradients."""
    if torch.is_grad_enabled():
        raise RuntimeError("Gradient tracking must be disabled during inference")
    loader = make_loader(dataset, batch_size, device)
    batches: list[np.ndarray] = []
    for inputs in loader:
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        reconstructions = model(inputs)
        scores = torch.mean((reconstructions - inputs) ** 2, dim=1)
        if not torch.isfinite(scores).all() or torch.any(scores < 0):
            raise FloatingPointError("Invalid test reconstruction error encountered")
        batches.append(scores.cpu().numpy())
    score_values = np.concatenate(batches).astype(np.float32, copy=False)
    metadata = dataset.metadata
    if len(score_values) != len(metadata):
        raise RuntimeError("Test scores and metadata are misaligned")
    metadata["reconstruction_mse"] = score_values
    return metadata


def compute_reconstruction_errors(
    model: Autoencoder,
    normal: WindowDataset,
    theft: WindowDataset,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """Score all usable test windows and restore manifest date metadata."""
    errors = pd.concat(
        [
            score_dataset(model, normal, batch_size, device),
            score_dataset(model, theft, batch_size, device),
        ],
        ignore_index=True,
    )
    dates = pd.read_parquet(
        TEST_MANIFEST_PATH,
        columns=["window_id", "start_date", "end_date"],
        engine="pyarrow",
    )
    if dates["window_id"].duplicated().any():
        raise ValueError("Test manifest contains duplicate window IDs")
    errors = errors.merge(dates, on="window_id", how="left", validate="one_to_one")
    if errors[["start_date", "end_date"]].isna().any().any():
        raise ValueError("Test date metadata did not align with reconstruction scores")
    return errors


def apply_threshold(errors: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Apply the frozen strict-greater-than anomaly rule exactly once."""
    output = errors.copy()
    output["predicted_anomaly"] = (
        output["reconstruction_mse"].to_numpy(dtype=np.float64) > threshold
    ).astype(np.int8)
    return output.loc[
        :,
        [
            "window_id",
            "CONS_NO",
            "FLAG",
            "reconstruction_mse",
            "missing_fraction",
            "predicted_anomaly",
            "start_date",
            "end_date",
        ],
    ]


def calculate_window_metrics(errors: pd.DataFrame, threshold: float) -> dict[str, object]:
    """Calculate frozen-threshold and threshold-independent test metrics."""
    labels = errors["FLAG"].to_numpy(dtype=np.int8, copy=False)
    scores = errors["reconstruction_mse"].to_numpy(dtype=np.float64, copy=False)
    metrics = calculate_metrics(labels, scores, threshold)
    predicted = (scores > threshold).astype(np.int8)
    if not np.array_equal(predicted, errors["predicted_anomaly"].to_numpy(np.int8)):
        raise ValueError("Saved frozen-threshold predictions are inconsistent")
    metrics["ROC_AUC"] = float(roc_auc_score(labels, scores))
    metrics["PR_AUC"] = float(average_precision_score(labels, scores))
    metrics["theft_label_prevalence"] = float(np.mean(labels))
    return metrics


def summarize_error_distributions(
    errors: pd.DataFrame, threshold: float
) -> dict[str, object]:
    """Summarize test errors and theft exceedance of diagnostic cutoffs."""
    normal = errors.loc[errors["FLAG"].eq(0), "reconstruction_mse"].to_numpy(
        dtype=np.float64
    )
    theft = errors.loc[errors["FLAG"].eq(1), "reconstruction_mse"].to_numpy(
        dtype=np.float64
    )
    normal_summary = distribution_summary(normal)
    theft_summary = distribution_summary(theft)
    exceedance: dict[str, object] = {
        "frozen_threshold": {
            "threshold": threshold,
            "theft_windows_exceeding": int(np.count_nonzero(theft > threshold)),
            "theft_proportion_exceeding": float(np.mean(theft > threshold)),
            "diagnostic_only": False,
        }
    }
    for percentile in (0.90, 0.95, 0.99):
        cutoff = float(np.quantile(normal, percentile))
        exceedance[f"normal_test_p{percentile * 100:g}"] = {
            "threshold": cutoff,
            "theft_windows_exceeding": int(np.count_nonzero(theft > cutoff)),
            "theft_proportion_exceeding": float(np.mean(theft > cutoff)),
            "diagnostic_only": True,
        }
    return {
        "FLAG_0_normal": normal_summary,
        "FLAG_1_theft_labelled": theft_summary,
        "theft_to_normal_median_ratio": theft_summary["median"]
        / normal_summary["median"],
        "theft_to_normal_mean_ratio": theft_summary["mean"] / normal_summary["mean"],
        "theft_exceedance": exceedance,
    }


def calculate_customer_diagnostics(
    errors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Calculate diagnostic customer aggregation and frozen-window flag summaries."""
    diagnostic_table, diagnostic = analyze_customer_level_scores(errors)
    customers = (
        errors.groupby("CONS_NO", sort=False)
        .agg(
            FLAG=("FLAG", "first"),
            window_count=("window_id", "size"),
            anomalous_window_count=("predicted_anomaly", "sum"),
        )
        .reset_index()
    )
    customers["anomalous_window_fraction"] = (
        customers["anomalous_window_count"] / customers["window_count"]
    )
    rows: list[dict[str, object]] = []
    for flag in (0, 1):
        group = customers.loc[customers["FLAG"].eq(flag)]
        for measure in ("anomalous_window_count", "anomalous_window_fraction"):
            rows.append(
                {
                    "FLAG": flag,
                    "measure": measure,
                    **distribution_summary(group[measure]),
                }
            )
    diagnostic["frozen_window_threshold_customer_summary"] = {
        "normal_mean_anomalous_fraction": float(
            customers.loc[customers["FLAG"].eq(0), "anomalous_window_fraction"].mean()
        ),
        "theft_mean_anomalous_fraction": float(
            customers.loc[customers["FLAG"].eq(1), "anomalous_window_fraction"].mean()
        ),
        "customer_classifier_created": False,
    }
    return diagnostic_table, pd.DataFrame(rows), diagnostic


def analyze_missingness(
    errors: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reuse the Stage 5 correlation/grouped-summary definition on test data."""
    return analyze_missingness_relationship(errors)


def compare_validation_test(test_metrics: dict[str, object]) -> tuple[pd.DataFrame, str]:
    """Compare saved Stage 5 validation metrics with frozen Stage 6 test metrics."""
    threshold = load_json(THRESHOLD_PATH)
    stage5 = load_json(VALIDATION_THRESHOLD_SUMMARY_PATH)
    validation = threshold["validation_metrics"]
    validation_auc = stage5["validation_ranking_metrics"]
    pairs = {
        "precision": (validation["precision"], test_metrics["precision"]),
        "recall": (validation["recall"], test_metrics["recall"]),
        "F1": (validation["F1"], test_metrics["F1"]),
        "specificity": (validation["specificity"], test_metrics["specificity"]),
        "false_positive_rate": (
            validation["false_positive_rate"],
            test_metrics["false_positive_rate"],
        ),
        "ROC_AUC": (validation_auc["ROC_AUC"], test_metrics["ROC_AUC"]),
        "PR_AUC": (validation_auc["PR_AUC"], test_metrics["PR_AUC"]),
        "predicted_anomaly_percentage": (
            validation["predicted_anomaly_percentage"],
            test_metrics["predicted_anomaly_percentage"],
        ),
    }
    rows = [
        {
            "metric": metric,
            "validation": float(values[0]),
            "test": float(values[1]),
            "signed_difference_test_minus_validation": float(values[1] - values[0]),
            "absolute_difference": float(abs(values[1] - values[0])),
        }
        for metric, values in pairs.items()
    ]
    differences = {row["metric"]: row["absolute_difference"] for row in rows}
    stable = (
        differences["precision"] <= 0.03
        and differences["recall"] <= 0.03
        and differences["F1"] <= 0.03
        and differences["false_positive_rate"] <= 0.02
        and differences["ROC_AUC"] <= 0.03
        and differences["PR_AUC"] <= 0.03
    )
    if stable:
        interpretation = "broadly stable"
    elif (
        test_metrics["ROC_AUC"] < validation_auc["ROC_AUC"] - 0.10
        or test_metrics["PR_AUC"] < validation_auc["PR_AUC"] - 0.05
        or test_metrics["F1"] < validation["F1"] - 0.08
    ):
        interpretation = "substantially degraded"
    elif (
        test_metrics["ROC_AUC"] > validation_auc["ROC_AUC"]
        and test_metrics["PR_AUC"] > validation_auc["PR_AUC"]
        and test_metrics["F1"] > validation["F1"]
    ):
        interpretation = "unexpectedly improved"
    else:
        interpretation = "moderately degraded"
    return pd.DataFrame(rows), interpretation


def top_error_cases(errors: pd.DataFrame, count: int = 15) -> pd.DataFrame:
    """Return small, non-causal FP/TP/FN metadata samples for investigation."""
    predicted = errors["predicted_anomaly"].eq(1)
    categories = (
        ("highest_scoring_false_positive", errors.loc[errors["FLAG"].eq(0) & predicted]),
        ("highest_scoring_true_positive", errors.loc[errors["FLAG"].eq(1) & predicted]),
        (
            "highest_error_false_negative_below_threshold",
            errors.loc[errors["FLAG"].eq(1) & ~predicted],
        ),
    )
    frames: list[pd.DataFrame] = []
    for category, frame in categories:
        selected = frame.nlargest(count, "reconstruction_mse").copy()
        selected.insert(0, "case_category", category)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def save_json(payload: dict[str, object], path: Path) -> None:
    """Atomically save JSON."""
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
    """Atomically save CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically save compressed per-window test errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_text(text: str, path: Path) -> None:
    """Atomically save UTF-8 Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_confusion_matrix(metrics: dict[str, object]) -> None:
    """Plot the weak-label test confusion matrix with explicit semantics."""
    matrix = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                fontsize=12,
            )
    axis.set_xticks([0, 1], ["Predicted normal", "Predicted anomaly"])
    axis.set_yticks([0, 1], ["Actual normal", "Actual theft-labelled"])
    axis.set_xlabel("Frozen threshold prediction")
    axis.set_ylabel("Weak customer-level window label")
    axis.set_title("Frozen Autoencoder Test Confusion Matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    temporary = CONFUSION_PATH.with_name(CONFUSION_PATH.stem + ".tmp.png")
    try:
        figure.savefig(temporary, dpi=150)
        temporary.replace(CONFUSION_PATH)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def save_error_distribution(errors: pd.DataFrame, threshold: float) -> None:
    """Plot right-skewed test MSE without changing the saved score values."""
    figure, axis = plt.subplots(figsize=(8, 5))
    epsilon = np.finfo(np.float32).eps
    for flag, label in ((0, "Normal test"), (1, "Theft-labelled test")):
        scores = errors.loc[errors["FLAG"].eq(flag), "reconstruction_mse"].to_numpy()
        axis.hist(
            np.log10(scores.astype(np.float64) + epsilon),
            bins=80,
            density=True,
            alpha=0.45,
            label=label,
        )
    axis.axvline(
        np.log10(threshold + epsilon),
        color="tab:red",
        linestyle="--",
        label="Frozen validation threshold",
    )
    axis.set_xlabel("log10(reconstruction MSE + float32 epsilon)")
    axis.set_ylabel("Density")
    axis.set_title("Frozen Autoencoder Test Error Distribution")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    temporary = ERROR_DISTRIBUTION_PATH.with_name(
        ERROR_DISTRIBUTION_PATH.stem + ".tmp.png"
    )
    try:
        figure.savefig(temporary, dpi=150)
        temporary.replace(ERROR_DISTRIBUTION_PATH)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def build_report(
    threshold: float,
    metrics: dict[str, object],
    distributions: dict[str, object],
    comparison: pd.DataFrame,
    generalization: str,
    customer: dict[str, object],
    missingness: dict[str, object],
) -> str:
    """Build the human-readable frozen baseline test report."""
    normal = distributions["FLAG_0_normal"]
    theft = distributions["FLAG_1_theft_labelled"]
    exceedance = distributions["theft_exceedance"]
    p95_customer = customer["aggregation_auc"]["p95_mse"]
    comparison_lines = [
        "| Metric | Validation | Test | Absolute difference |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        comparison_lines.append(
            f"| {row.metric} | {row.validation:.6f} | {row.test:.6f} | "
            f"{row.absolute_difference:.6f} |"
        )
    return f"""# Stage 6 — Frozen Test Evaluation

## Frozen evaluation policy

The Stage 4 checkpoint was evaluated once using threshold `{threshold:.12g}`
loaded from `{THRESHOLD_PATH.relative_to(PROJECT_ROOT).as_posix()}`. The rule was
`reconstruction_mse > threshold`. No retraining, threshold adjustment,
architecture change, or test-driven selection occurred.

## Window-level test results

Test windows: {normal['count'] + theft['count']:,} ({normal['count']:,} normal;
{theft['count']:,} theft-labelled).

| Metric | Value |
|---|---:|
| TP | {metrics['TP']:,} |
| TN | {metrics['TN']:,} |
| FP | {metrics['FP']:,} |
| FN | {metrics['FN']:,} |
| Precision | {metrics['precision']:.6f} |
| Recall | {metrics['recall']:.6f} |
| F1 | {metrics['F1']:.6f} |
| Specificity | {metrics['specificity']:.6f} |
| False-positive rate | {metrics['false_positive_rate']:.6f} |
| Balanced accuracy | {metrics['balanced_accuracy']:.6f} |
| Accuracy | {metrics['accuracy']:.6f} |
| Predicted anomaly rate | {metrics['predicted_anomaly_percentage']:.4f}% |
| ROC-AUC | {metrics['ROC_AUC']:.6f} |
| PR-AUC | {metrics['PR_AUC']:.6f} |

FLAG is customer-level. These are weakly supervised window-label metrics, not
confirmed theft-event detection metrics.

## Reconstruction-error distributions

| Population | Count | Mean | Median | Std | Min | P25 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLAG=0 | {normal['count']:,} | {normal['mean']:.8f} | {normal['median']:.8f} | {normal['standard_deviation']:.8f} | {normal['minimum']:.8f} | {normal['p25']:.8f} | {normal['p75']:.8f} | {normal['p90']:.8f} | {normal['p95']:.8f} | {normal['p99']:.8f} | {normal['maximum']:.8f} |
| FLAG=1 | {theft['count']:,} | {theft['mean']:.8f} | {theft['median']:.8f} | {theft['standard_deviation']:.8f} | {theft['minimum']:.8f} | {theft['p25']:.8f} | {theft['p75']:.8f} | {theft['p90']:.8f} | {theft['p95']:.8f} | {theft['p99']:.8f} | {theft['maximum']:.8f} |

The theft/normal median ratio is
{distributions['theft_to_normal_median_ratio']:.4f}; the mean ratio is
{distributions['theft_to_normal_mean_ratio']:.4f}. Theft-labelled exceedance is
{100 * exceedance['frozen_threshold']['theft_proportion_exceeding']:.2f}% at the
frozen threshold. Diagnostic exceedance of normal test p90/p95/p99 is
{100 * exceedance['normal_test_p90']['theft_proportion_exceeding']:.2f}% /
{100 * exceedance['normal_test_p95']['theft_proportion_exceeding']:.2f}% /
{100 * exceedance['normal_test_p99']['theft_proportion_exceeding']:.2f}%. Test
percentiles were not used to change the threshold.

## Validation versus test

{chr(10).join(comparison_lines)}

Generalization is classified as **{generalization}** under the prespecified
absolute-difference checks. The model and threshold remain unchanged.

## Customer and missingness diagnostics

Test contains {customer['customer_counts']['FLAG_0']:,} normal and
{customer['customer_counts']['FLAG_1']:,} theft-labelled customers. Customer p95
aggregation has ROC-AUC {p95_customer['ROC_AUC']:.6f} and PR-AUC
{p95_customer['PR_AUC']:.6f}. No customer classifier or customer cutoff was
created. Mean anomalous-window fractions are
{customer['frozen_window_threshold_customer_summary']['normal_mean_anomalous_fraction']:.4f}
for normal customers and
{customer['frozen_window_threshold_customer_summary']['theft_mean_anomalous_fraction']:.4f}
for theft-labelled customers.

Missing-fraction correlation with reconstruction MSE is weak: overall
Pearson/Spearman {missingness['overall']['pearson']:.4f}/
{missingness['overall']['spearman']:.4f}, normal
{missingness['FLAG_0']['pearson']:.4f}/{missingness['FLAG_0']['spearman']:.4f},
and theft-labelled {missingness['FLAG_1']['pearson']:.4f}/
{missingness['FLAG_1']['spearman']:.4f}.

## Interpretation and boundary

This frozen fully connected stride-{STRIDE} Autoencoder is classified after test
evaluation according to its ranking, F1, recall, false-positive rate, and
validation/test consistency in `baseline_final_summary.json`. It establishes a
rigorous baseline for future models. No stride comparison, GAN, ensemble,
dashboard, or deployment work was performed.
"""


def write_reports(
    errors: pd.DataFrame,
    threshold: float,
    threshold_artifact: dict[str, object],
    checkpoint: dict[str, object],
    metrics: dict[str, object],
    distributions: dict[str, object],
    comparison: pd.DataFrame,
    generalization: str,
    customer_table: pd.DataFrame,
    customer_anomaly_table: pd.DataFrame,
    customer_diagnostic: dict[str, object],
    missingness_table: pd.DataFrame,
    missingness_diagnostic: dict[str, object],
    cases: pd.DataFrame,
    hashes: dict[str, str],
) -> None:
    """Persist all frozen test evaluation artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet(errors, ERRORS_PATH)
    save_csv(comparison, VALIDATION_COMPARISON_PATH)
    save_csv(customer_table, CUSTOMER_DIAGNOSTICS_PATH)
    save_csv(customer_anomaly_table, CUSTOMER_ANOMALY_PATH)
    save_csv(missingness_table, MISSINGNESS_PATH)
    save_csv(cases, TOP_CASES_PATH)

    model_assessment = classify_baseline_strength(metrics)
    metrics_payload: dict[str, object] = {
        "stage": "Stage 6 — Frozen Test Evaluation",
        "frozen_threshold": threshold,
        "threshold_source": str(THRESHOLD_PATH.relative_to(PROJECT_ROOT)),
        "anomaly_rule": "reconstruction_mse > frozen_threshold",
        "window_level_test_metrics": metrics,
        "error_distributions": distributions,
        "validation_vs_test": {
            "classification": generalization,
            "metrics": comparison.to_dict("records"),
        },
        "customer_level_diagnostic": customer_diagnostic,
        "missingness_diagnostic": missingness_diagnostic,
        "weak_supervision_note": (
            "FLAG is customer-level; FLAG=1 test windows are not confirmed theft events."
        ),
        "model_strength_interpretation": model_assessment,
        "retraining_performed": False,
        "threshold_modified": False,
        "protected_stage1_stage5_sha256": hashes,
    }
    save_json(metrics_payload, METRICS_PATH)

    final_summary: dict[str, object] = {
        "artifact_role": f"final stride-{STRIDE} fully connected Autoencoder performance record",
        "model": {
            "architecture": EXPECTED_ARCHITECTURE,
            "latent_dimension": 4,
            "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": hashes[
                str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT))
            ],
        },
        "preprocessing": {
            "window_length": WINDOW_LENGTH,
            "stride": STRIDE,
            "usable_window_rule": "missing_fraction < 1.0; at least one observed reading",
            "scaler_policy": "Stage 2 feature-wise StandardScaler fitted on train customers only",
            "model_input": "30 scaled consumption values only",
        },
        "threshold": {
            "value": threshold,
            "selection_method": threshold_artifact["threshold_selection_method"],
            "validation_source": "Stage 5 validation reconstruction errors",
            "artifact": str(THRESHOLD_PATH.relative_to(PROJECT_ROOT)),
            "artifact_sha256": hashes[str(THRESHOLD_PATH.relative_to(PROJECT_ROOT))],
            "rule": "reconstruction_mse > threshold",
        },
        "test_sample_counts": {
            "normal_windows": int(errors["FLAG"].eq(0).sum()),
            "theft_labelled_windows": int(errors["FLAG"].eq(1).sum()),
            "total_windows": int(len(errors)),
        },
        "test_metrics": {
            key: metrics[key]
            for key in (
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
                "ROC_AUC",
                "PR_AUC",
                "predicted_anomaly_percentage",
            )
        },
        "validation_test_generalization": generalization,
        "model_strength_interpretation": model_assessment,
        "weak_supervision_caveat": (
            "FLAG is customer-level; window metrics do not measure confirmed theft events."
        ),
        "frozen_evaluation_guarantees": {
            "checkpoint_unchanged": True,
            "threshold_unchanged": True,
            "retraining_performed": False,
            "test_driven_tuning_performed": False,
        },
    }
    save_json(final_summary, FINAL_SUMMARY_PATH)
    save_confusion_matrix(metrics)
    save_error_distribution(errors, threshold)
    save_text(
        build_report(
            threshold,
            metrics,
            distributions,
            comparison,
            generalization,
            customer_diagnostic,
            missingness_diagnostic,
        ),
        REPORT_PATH,
    )


def classify_baseline_strength(metrics: dict[str, object]) -> str:
    """Classify baseline evidence conservatively without changing any decision."""
    roc_auc = float(metrics["ROC_AUC"])
    pr_auc = float(metrics["PR_AUC"])
    prevalence = float(metrics["theft_label_prevalence"])
    f1 = float(metrics["F1"])
    recall = float(metrics["recall"])
    if roc_auc < 0.55 and pr_auc <= prevalence * 1.10:
        return "non-informative"
    if roc_auc < 0.65 or f1 < 0.25 or recall < 0.25 or pr_auc < prevalence * 2:
        return "weak"
    if roc_auc < 0.80 or f1 < 0.60:
        return "moderate"
    return "strong"


def positive_int(value: str) -> int:
    """Argparse converter for positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    """Expose stride and batch size; all modeling decisions remain frozen."""
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen Autoencoder stride on test exactly once."
    )
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--stride", type=int, choices=(30, 15), default=30)
    return parser.parse_args(arguments)


def main() -> None:
    """Run Stage 6 frozen test inference and evaluation."""
    args = parse_args()
    configure_stride(args.stride)
    hashes_before = protected_hashes()
    threshold, threshold_artifact = load_frozen_threshold()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(device)
    store, normal, theft = load_test_dataset()
    print(f"Selected device: {device}")
    print(f"Usable test windows: {len(normal) + len(theft):,}")
    print(f"Normal: {len(normal):,}; theft-labelled: {len(theft):,}")
    print(f"Frozen threshold: {threshold:.12g}")
    print(f"Test split matrix memory: {store.memory_bytes / 1024**2:.1f} MiB")

    errors = compute_reconstruction_errors(
        model, normal, theft, args.batch_size, device
    )
    errors = apply_threshold(errors, threshold)
    metrics = calculate_window_metrics(errors, threshold)
    distributions = summarize_error_distributions(errors, threshold)
    customer_table, customer_anomaly_table, customer_diagnostic = (
        calculate_customer_diagnostics(errors)
    )
    missingness_table, missingness_diagnostic = analyze_missingness(errors)
    comparison, generalization = compare_validation_test(metrics)
    cases = top_error_cases(errors)

    write_reports(
        errors,
        threshold,
        threshold_artifact,
        checkpoint,
        metrics,
        distributions,
        comparison,
        generalization,
        customer_table,
        customer_anomaly_table,
        customer_diagnostic,
        missingness_table,
        missingness_diagnostic,
        cases,
        hashes_before,
    )

    hashes_after = protected_hashes()
    if hashes_before != hashes_after:
        changed = [
            path for path in hashes_before if hashes_before[path] != hashes_after[path]
        ]
        raise RuntimeError(f"Protected Stage 1–5 artifacts changed: {changed}")

    print(
        f"TP={metrics['TP']:,}, TN={metrics['TN']:,}, "
        f"FP={metrics['FP']:,}, FN={metrics['FN']:,}"
    )
    print(
        f"Precision={metrics['precision']:.6f}, recall={metrics['recall']:.6f}, "
        f"F1={metrics['F1']:.6f}, specificity={metrics['specificity']:.6f}, "
        f"FPR={metrics['false_positive_rate']:.6f}"
    )
    print(f"ROC-AUC={metrics['ROC_AUC']:.6f}; PR-AUC={metrics['PR_AUC']:.6f}")
    print(f"Validation/test generalization: {generalization}")
    print(f"Baseline strength: {classify_baseline_strength(metrics)}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Checkpoint unchanged: True")
    print("Threshold unchanged: True")
    print("Retraining performed: False")


if __name__ == "__main__":
    main()
