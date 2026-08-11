"""Stage 8: compare frozen stride-30 and stride-15 Autoencoder artifacts.

The script reads saved Stage 3C–6 artifacts only. It does not load window data,
run inference, select thresholds, retrain models, or rewrite either stride's
authoritative results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "autoencoder" / "stride_comparison"
CSV_PATH = OUTPUT_DIR / "stride30_vs_stride15.csv"
JSON_PATH = OUTPUT_DIR / "stride30_vs_stride15.json"
REPORT_PATH = OUTPUT_DIR / "stride_comparison_report.md"
STRIDES = (30, 15)


def load_json(path: Path) -> dict[str, object]:
    """Load one JSON object."""
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one artifact without loading it completely."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths(stride: int) -> tuple[Path, ...]:
    """Return authoritative inputs for one completed stride."""
    result_dir = PROJECT_ROOT / "results" / "autoencoder" / f"stride_{stride}"
    return (
        PROJECT_ROOT / "data" / "windows" / f"stride_{stride}" / "window_summary.json",
        PROJECT_ROOT / "data" / "windows" / f"stride_{stride}" / "usable" / "filter_summary.json",
        PROJECT_ROOT / "models" / "autoencoder" / f"ae_stride{stride}_baseline.pt",
        PROJECT_ROOT / "models" / "autoencoder" / f"ae_stride{stride}_baseline_config.json",
        PROJECT_ROOT / "models" / "autoencoder" / f"ae_stride{stride}_threshold.json",
        result_dir / "training_summary.json",
        result_dir / "threshold_analysis" / "threshold_summary.json",
        result_dir / "test_evaluation" / "test_metrics.json",
        result_dir / "test_evaluation" / "baseline_final_summary.json",
    )


def protected_hashes() -> dict[str, str]:
    """Hash all saved inputs and fail if the comparison is incomplete."""
    paths = tuple(path for stride in STRIDES for path in artifact_paths(stride))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Comparison inputs are missing: {missing}")
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path) for path in paths
    }


def load_stride(stride: int) -> dict[str, object]:
    """Load one stride's saved summaries without recalculation."""
    result_dir = PROJECT_ROOT / "results" / "autoencoder" / f"stride_{stride}"
    return {
        "window": load_json(
            PROJECT_ROOT / "data" / "windows" / f"stride_{stride}" / "window_summary.json"
        ),
        "filter": load_json(
            PROJECT_ROOT
            / "data"
            / "windows"
            / f"stride_{stride}"
            / "usable"
            / "filter_summary.json"
        ),
        "training": load_json(result_dir / "training_summary.json"),
        "threshold": load_json(result_dir / "threshold_analysis" / "threshold_summary.json"),
        "test": load_json(result_dir / "test_evaluation" / "test_metrics.json"),
    }


def nested(payload: dict[str, object], *keys: str) -> float:
    """Read and cast one nested saved metric."""
    value: object = payload
    for key in keys:
        value = value[key]  # type: ignore[index]
    return float(value)


def add_row(
    rows: list[dict[str, object]],
    category: str,
    metric: str,
    value30: float,
    value15: float,
    unit: str,
) -> None:
    """Append one auditable comparison row."""
    difference = value15 - value30
    relative = 100.0 * difference / abs(value30) if value30 else None
    rows.append(
        {
            "category": category,
            "metric": metric,
            "stride_30": value30,
            "stride_15": value15,
            "absolute_difference_stride15_minus_stride30": difference,
            "relative_difference_percent": relative,
            "unit": unit,
        }
    )


def build_rows(data: dict[int, dict[str, object]]) -> pd.DataFrame:
    """Build the requested volume, training, validation, test, and customer table."""
    a, b = data[30], data[15]
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        add_row(
            rows,
            "window_volume",
            f"usable_{split}_windows",
            nested(a, "filter", "splits", split, "retained_windows"),
            nested(b, "filter", "splits", split, "retained_windows"),
            "windows",
        )
    add_row(
        rows,
        "window_volume",
        "windows_per_customer",
        nested(a, "window", "configuration", "windows_per_customer"),
        nested(b, "window", "configuration", "windows_per_customer"),
        "windows/customer",
    )
    for metric, unit in (
        ("epochs_executed", "epochs"),
        ("best_validation_mse", "MSE"),
        ("runtime_seconds", "seconds"),
    ):
        add_row(
            rows,
            "training",
            metric,
            nested(a, "training", metric),
            nested(b, "training", metric),
            unit,
        )
    add_row(rows, "training", "trainable_parameters", 1362, 1362, "parameters")

    validation_metrics = (
        "threshold",
        "precision",
        "recall",
        "F1",
        "specificity",
        "false_positive_rate",
        "balanced_accuracy",
        "predicted_anomaly_percentage",
    )
    for metric in validation_metrics:
        unit = "percent" if metric == "predicted_anomaly_percentage" else "fraction"
        add_row(
            rows,
            "validation_p95",
            metric,
            nested(a, "threshold", "selected_threshold", "validation_metrics", metric),
            nested(b, "threshold", "selected_threshold", "validation_metrics", metric),
            unit,
        )
    for metric in ("ROC_AUC", "PR_AUC"):
        add_row(
            rows,
            "validation_p95",
            metric,
            nested(a, "threshold", "validation_ranking_metrics", metric),
            nested(b, "threshold", "validation_ranking_metrics", metric),
            "fraction",
        )

    test_metrics = (
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
    for metric in test_metrics:
        unit = "percent" if metric == "predicted_anomaly_percentage" else "fraction"
        add_row(
            rows,
            "frozen_test",
            metric,
            nested(a, "test", "window_level_test_metrics", metric),
            nested(b, "test", "window_level_test_metrics", metric),
            unit,
        )

    for aggregation in ("mean_mse", "maximum_mse", "p95_mse"):
        for metric in ("ROC_AUC", "PR_AUC"):
            add_row(
                rows,
                "customer_test",
                f"{aggregation}_{metric}",
                nested(a, "test", "customer_level_diagnostic", "aggregation_auc", aggregation, metric),
                nested(b, "test", "customer_level_diagnostic", "aggregation_auc", aggregation, metric),
                "fraction",
            )
    for label in ("normal", "theft"):
        key = f"{label}_mean_anomalous_fraction"
        add_row(
            rows,
            "customer_test",
            key,
            nested(a, "test", "customer_level_diagnostic", "frozen_window_threshold_customer_summary", key),
            nested(b, "test", "customer_level_diagnostic", "frozen_window_threshold_customer_summary", key),
            "fraction",
        )
    return pd.DataFrame(rows)


def fixed_design() -> dict[str, object]:
    """Record the controlled ablation boundary."""
    return {
        "fixed": [
            "customer split",
            "cleaning and train-fitted StandardScaler",
            "window length 30",
            "missing_count < 30 usable-window rule",
            "30-16-8-4-8-16-30 architecture and latent dimension 4",
            "MSE loss and Adam optimizer",
            "learning rate 0.001 and batch size 256",
            "early stopping policy and seed 42",
            "normal-validation P95 threshold method",
            "one-time frozen-test evaluation method",
        ],
        "changed": "stride 30 to stride 15",
        "weak_supervision": (
            "FLAG is customer-level; window metrics do not measure confirmed theft events."
        ),
    }


def build_report(frame: pd.DataFrame, data: dict[int, dict[str, object]]) -> str:
    """Create the concise controlled-comparison report."""
    def row(category: str, metric: str) -> pd.Series:
        return frame.loc[
            frame["category"].eq(category) & frame["metric"].eq(metric)
        ].iloc[0]

    test_lines = [
        "| Metric | Stride 30 | Stride 15 | Δ 15−30 |",
        "|---|---:|---:|---:|",
    ]
    for metric in (
        "precision", "recall", "F1", "specificity", "false_positive_rate",
        "balanced_accuracy", "ROC_AUC", "PR_AUC", "predicted_anomaly_percentage",
    ):
        item = row("frozen_test", metric)
        test_lines.append(
            f"| {metric} | {item.stride_30:.6f} | {item.stride_15:.6f} | "
            f"{item.absolute_difference_stride15_minus_stride30:+.6f} |"
        )

    customer_lines = [
        "| Aggregation | Metric | Stride 30 | Stride 15 | Δ 15−30 |",
        "|---|---|---:|---:|---:|",
    ]
    for aggregation in ("mean_mse", "maximum_mse", "p95_mse"):
        for metric in ("ROC_AUC", "PR_AUC"):
            item = row("customer_test", f"{aggregation}_{metric}")
            customer_lines.append(
                f"| {aggregation} | {metric} | {item.stride_30:.6f} | "
                f"{item.stride_15:.6f} | "
                f"{item.absolute_difference_stride15_minus_stride30:+.6f} |"
            )

    mse30 = row("training", "best_validation_mse")
    mse15 = float(mse30.stride_15)
    runtime = row("training", "runtime_seconds")
    test_roc = row("frozen_test", "ROC_AUC")
    test_pr = row("frozen_test", "PR_AUC")
    test_f1 = row("frozen_test", "F1")
    p95_pr = row("customer_test", "p95_mse_PR_AUC")
    missing30 = data[30]["test"]["missingness_diagnostic"]
    missing15 = data[15]["test"]["missingness_diagnostic"]

    return f"""# Stage 8 — Stride-30 vs Stride-15 Autoencoder Comparison

## Controlled design

The customer split, preprocessing, train-fitted scaler, 30-day window length,
usable-window rule, architecture, loss, optimizer, learning rate, batch size,
early stopping, seed, normal-validation P95 threshold method, and frozen-test
method are fixed. Only stride changes from 30 to 15. FLAG remains weak
customer-level supervision.

Stride 15 uses 67 windows/customer versus 34 for stride 30 and increases usable
training windows from 708,308 to 1,520,861. CPU training runtime changes from
{float(runtime.stride_30):.1f}s to {float(runtime.stride_15):.1f}s.

## Reconstruction versus detection

Best normal-validation MSE decreases from {float(mse30.stride_30):.8f} to
{mse15:.8f}. This lower reconstruction loss does **not** translate into better
window-level anomaly discrimination: test ROC-AUC changes by
{float(test_roc.absolute_difference_stride15_minus_stride30):+.6f}, PR-AUC by
{float(test_pr.absolute_difference_stride15_minus_stride30):+.6f}, and F1 by
{float(test_f1.absolute_difference_stride15_minus_stride30):+.6f}.

## Frozen test comparison

{chr(10).join(test_lines)}

## Customer-level test comparison

{chr(10).join(customer_lines)}

P95 customer PR-AUC improves by
{float(p95_pr.absolute_difference_stride15_minus_stride30):+.6f}, but the other
customer-level changes are small and mixed. No customer threshold is selected.

## Missingness diagnostic

Overall test Pearson/Spearman correlations are
{missing30['overall']['pearson']:.4f}/{missing30['overall']['spearman']:.4f}
for stride 30 and
{missing15['overall']['pearson']:.4f}/{missing15['overall']['spearman']:.4f}
for stride 15. Both remain weak; stride 15 is modestly more negative.

## Conclusion and stride-7 recommendation

Stride 15 is **effectively equivalent and slightly worse at the window level**,
not a meaningful improvement over stride 30. The small P95 customer PR-AUC gain
does not offset lower window ROC-AUC, PR-AUC, F1, and recall.

Do **not** execute stride 7 next under the current fully connected baseline.
Stride 15 approximately doubles window volume and materially increases CPU time
without improving core detection. Stride 7 would produce 144 windows/customer,
about 2.15× the stride-15 volume, while current evidence gives no reason to
expect a worthwhile anomaly-discrimination gain. It remains unexecuted and may
be reconsidered only after a model or hypothesis that can exploit denser overlap.
"""


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """Write comparison artifacts while proving all inputs remain unchanged."""
    hashes_before = protected_hashes()
    data = {stride: load_stride(stride) for stride in STRIDES}
    frame = build_rows(data)
    save_csv(frame, CSV_PATH)
    payload = {
        "stage": "Stage 8 — Controlled Stride Ablation",
        "comparison": "stride 30 versus stride 15",
        "design": fixed_design(),
        "rows": frame.where(pd.notna(frame), None).to_dict("records"),
        "interpretation": "effectively equivalent; stride 15 slightly worse at window level",
        "stride7_recommendation": {
            "execute_next": False,
            "windows_per_customer_if_run": 144,
            "reason": (
                "Higher overlap doubled stride-15 volume without improving core frozen-test "
                "metrics; additional CPU cost is not justified for this baseline."
            ),
        },
        "input_sha256": hashes_before,
        "inputs_recalculated_or_modified": False,
    }
    save_text(json.dumps(payload, indent=2) + "\n", JSON_PATH)
    save_text(build_report(frame, data), REPORT_PATH)
    hashes_after = protected_hashes()
    if hashes_before != hashes_after:
        changed = [key for key in hashes_before if hashes_before[key] != hashes_after[key]]
        raise RuntimeError(f"Protected comparison inputs changed: {changed}")
    print(f"Comparison rows: {len(frame)}")
    print("Interpretation: effectively equivalent; stride 15 slightly worse at window level")
    print("Run stride 7 next: False")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Protected stride artifacts unchanged: True")


if __name__ == "__main__":
    main()
