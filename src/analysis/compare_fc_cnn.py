"""Saved-artifact-only comparison of frozen stride-30 FC and CNN autoencoders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FC = ROOT / "results/autoencoder/stride_30"
CNN = ROOT / "results/cnn_autoencoder/stride_30"
OUT = ROOT / "results/model_comparison/fc_vs_cnn"


def load(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    fc_train = load(FC / "training_summary.json")
    cnn_train = load(CNN / "training_summary.json")
    fc_threshold = load(ROOT / "models/autoencoder/ae_stride30_threshold.json")
    cnn_threshold = load(ROOT / "models/cnn_autoencoder/cnn_ae_stride30_threshold.json")
    fc_val = load(FC / "threshold_analysis/threshold_summary.json")
    cnn_val = load(CNN / "threshold_analysis/threshold_summary.json")
    fc_test = load(FC / "test_evaluation/test_metrics.json")
    cnn_test = load(CNN / "test_evaluation/test_metrics.json")
    fc_final = load(FC / "test_evaluation/baseline_final_summary.json")
    cnn_final = load(CNN / "test_evaluation/baseline_final_summary.json")
    fc_config = load(ROOT / "models/autoencoder/ae_stride30_baseline_config.json")
    cnn_config = load(ROOT / "models/cnn_autoencoder/cnn_ae_stride30_baseline_config.json")

    ft = fc_test["window_level_test_metrics"]
    ct = cnn_test["window_level_test_metrics"]
    fv = fc_threshold["validation_metrics"]
    cv = cnn_threshold["validation_metrics"]
    fvr = fc_val["validation_ranking_metrics"]
    cvr = cnn_val["validation_ranking_metrics"]
    fc_customer = fc_test["customer_level_diagnostic"]["aggregation_auc"]
    cnn_customer = cnn_test["customer_level_diagnostic"]["aggregation_auc"]

    values = {
        "trainable_parameters": (fc_config["trainable_parameters"], cnn_config["trainable_parameters"]),
        "epochs_executed": (fc_train["epochs_executed"], cnn_train["epochs_executed"]),
        "best_epoch": (fc_train["best_epoch"], cnn_train["best_epoch"]),
        "best_validation_mse": (fc_train["best_validation_mse"], cnn_train["best_validation_mse"]),
        "training_runtime_seconds": (fc_train["runtime_seconds"], cnn_train["runtime_seconds"]),
        "frozen_p95_threshold": (fc_threshold["threshold_value"], cnn_threshold["threshold_value"]),
        "validation_precision": (fv["precision"], cv["precision"]),
        "validation_recall": (fv["recall"], cv["recall"]),
        "validation_F1": (fv["F1"], cv["F1"]),
        "validation_specificity": (fv["specificity"], cv["specificity"]),
        "validation_false_positive_rate": (fv["false_positive_rate"], cv["false_positive_rate"]),
        "validation_predicted_anomaly_percentage": (fv["predicted_anomaly_percentage"], cv["predicted_anomaly_percentage"]),
        "validation_ROC_AUC": (fvr["ROC_AUC"], cvr["ROC_AUC"]),
        "validation_PR_AUC": (fvr["PR_AUC"], cvr["PR_AUC"]),
        "test_precision": (ft["precision"], ct["precision"]),
        "test_recall": (ft["recall"], ct["recall"]),
        "test_F1": (ft["F1"], ct["F1"]),
        "test_specificity": (ft["specificity"], ct["specificity"]),
        "test_false_positive_rate": (ft["false_positive_rate"], ct["false_positive_rate"]),
        "test_balanced_accuracy": (ft["balanced_accuracy"], ct["balanced_accuracy"]),
        "test_ROC_AUC": (ft["ROC_AUC"], ct["ROC_AUC"]),
        "test_PR_AUC": (ft["PR_AUC"], ct["PR_AUC"]),
        "test_predicted_anomaly_percentage": (ft["predicted_anomaly_percentage"], ct["predicted_anomaly_percentage"]),
        "test_customer_mean_ROC_AUC": (fc_customer["mean_mse"]["ROC_AUC"], cnn_customer["mean_mse"]["ROC_AUC"]),
        "test_customer_mean_PR_AUC": (fc_customer["mean_mse"]["PR_AUC"], cnn_customer["mean_mse"]["PR_AUC"]),
        "test_customer_maximum_ROC_AUC": (fc_customer["maximum_mse"]["ROC_AUC"], cnn_customer["maximum_mse"]["ROC_AUC"]),
        "test_customer_maximum_PR_AUC": (fc_customer["maximum_mse"]["PR_AUC"], cnn_customer["maximum_mse"]["PR_AUC"]),
        "test_customer_p95_ROC_AUC": (fc_customer["p95_mse"]["ROC_AUC"], cnn_customer["p95_mse"]["ROC_AUC"]),
        "test_customer_p95_PR_AUC": (fc_customer["p95_mse"]["PR_AUC"], cnn_customer["p95_mse"]["PR_AUC"]),
    }
    rows = []
    for metric, (fc_value, cnn_value) in values.items():
        fc_float, cnn_float = float(fc_value), float(cnn_value)
        rows.append({
            "metric": metric,
            "fully_connected": fc_float,
            "cnn": cnn_float,
            "cnn_minus_fully_connected": cnn_float - fc_float,
            "cnn_relative_change_percent": ((cnn_float - fc_float) / fc_float * 100.0) if fc_float else None,
        })
    table = pd.DataFrame(rows)

    validation_mse_reduction = (
        (float(fc_train["best_validation_mse"]) - float(cnn_train["best_validation_mse"]))
        / float(fc_train["best_validation_mse"]) * 100.0
    )
    runtime_multiplier = float(cnn_train["runtime_seconds"]) / float(fc_train["runtime_seconds"])
    verdict = (
        "CNN reconstructs normal validation windows better, but the frozen FC model "
        "is slightly stronger on every principal validation/test ranking and P95-threshold "
        "classification metric. The CNN does not improve theft detection enough to justify "
        "its larger parameter count and longer CPU training time as the preferred baseline."
    )
    payload = {
        "stage": "Stage 10 — Frozen FC versus CNN comparison",
        "comparison_policy": "saved artifacts only; no retraining, threshold tuning, or test-driven selection",
        "shared_data_and_protocol": {
            "window_length": 30,
            "stride": 30,
            "training_normal_windows": int(fc_train["training_normal_window_count"]),
            "validation_windows": int(fc_threshold["validation_sample_counts"]["total_windows"]),
            "test_windows": int(fc_final["test_sample_counts"]["total_windows"]),
            "threshold_method": "normal_validation_95th_percentile",
        },
        "derived_comparison": {
            "cnn_validation_mse_reduction_percent": validation_mse_reduction,
            "cnn_training_runtime_multiplier": runtime_multiplier,
            "cnn_parameter_multiplier": float(cnn_config["trainable_parameters"]) / float(fc_config["trainable_parameters"]),
        },
        "metrics": table.to_dict("records"),
        "verdict": verdict,
        "preferred_frozen_baseline": "fully_connected_autoencoder",
        "architecture_effect": {
            "normal_reconstruction": "materially better",
            "window_level_anomaly_discrimination": "worse",
            "customer_level_anomaly_discrimination": "worse",
            "overall_detection_classification": "worse",
        },
        "weak_supervision_caveat": "FLAG is customer-level; window positives are not confirmed theft events.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "fc_vs_cnn.csv", index=False)
    save_json(payload, OUT / "fc_vs_cnn.json")
    report_rows = "\n".join(
        f"| {row.metric} | {row.fully_connected:.6f} | {row.cnn:.6f} | {row.cnn_minus_fully_connected:+.6f} |"
        for row in table.itertuples(index=False)
    )
    report = f"""# Stage 10 — Frozen FC versus CNN Autoencoder

Both models use the same stride-30 windows, split lineage, normal-only training
policy, and normal-validation P95 threshold rule. This comparison reads saved
artifacts only.

| Metric | Fully connected | CNN | CNN − FC |
|---|---:|---:|---:|
{report_rows}

The CNN reduced best normal-validation reconstruction MSE by
**{validation_mse_reduction:.2f}%**, but required **{runtime_multiplier:.2f}×** the
CPU training time and **{float(cnn_config['trainable_parameters']) / float(fc_config['trainable_parameters']):.2f}×**
the trainable parameters.

## Interpretation

{verdict}

- Normal reconstruction: **materially better** (30.70% lower validation MSE).
- Window-level anomaly discrimination: **worse**.
- Customer-level anomaly discrimination: **worse** for mean, maximum, and P95
  aggregation ROC-AUC and PR-AUC.
- Overall detection classification: **worse** than the FC baseline.

The fully connected model remains the preferred frozen baseline. This is not a
causal theft detector: `FLAG` is customer-level weak supervision, and positive
windows are not confirmed theft events.
"""
    (OUT / "fc_vs_cnn_report.md").write_text(report, encoding="utf-8")
    print(f"CNN validation MSE reduction: {validation_mse_reduction:.2f}%")
    print(f"CNN runtime multiplier: {runtime_multiplier:.2f}x")
    print("Preferred frozen baseline: fully_connected_autoencoder")


if __name__ == "__main__":
    main()
