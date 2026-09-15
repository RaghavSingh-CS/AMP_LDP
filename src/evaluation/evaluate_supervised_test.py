"""Stage 13: one-time test evaluation of the frozen supervised model."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.analysis.supervised_signal_utils import (
    LABEL_COLUMN,
    PROJECT_ROOT,
    RESULTS_DIR,
    aggregate_customer_mean,
    classification_metrics,
    file_sha256,
    load_window_matrix,
    save_csv,
    save_json,
)


def main() -> None:
    frozen_path = RESULTS_DIR / "frozen_decision.json"
    test_path = RESULTS_DIR / "test_metrics.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("Run the validation-only Stage 13 training phase first")
    if test_path.exists():
        raise FileExistsError(
            f"Refusing to repeat or overwrite the frozen test evaluation: {test_path}"
        )
    with frozen_path.open("r", encoding="utf-8") as source:
        frozen = json.load(source)
    if not frozen.get("created_before_test_inference"):
        raise ValueError("Frozen decision does not certify pre-test creation")

    model_path = PROJECT_ROOT / frozen["selected_model_artifact"]
    if file_sha256(model_path) != frozen["selected_model_sha256"]:
        raise ValueError("Selected model artifact changed after the decision was frozen")
    model = joblib.load(model_path)

    print("Loading test windows after validating the frozen model and thresholds...")
    test = load_window_matrix("test")
    scores = np.asarray(model.predict_proba(test.X)[:, 1], dtype=np.float64)
    window_metrics = classification_metrics(
        test.y, scores, float(frozen["window_threshold"])
    )
    customers = aggregate_customer_mean(test.y, scores, test.customer_ids)
    customer_metrics = classification_metrics(
        customers[LABEL_COLUMN].to_numpy(dtype=np.int8),
        customers["mean_probability"].to_numpy(dtype=np.float64),
        float(frozen["customer_mean_threshold"]),
    )

    payload = {
        "stage": "Stage 13 — frozen supervised test evaluation",
        "selected_model": frozen["selected_model"],
        "selected_model_sha256_verified": True,
        "thresholds_modified_after_validation": False,
        "retraining_performed": False,
        "window_level": window_metrics,
        "customer_mean_level": customer_metrics,
        "weak_supervision_note": (
            "FLAG is customer-level. FLAG=1 windows are not confirmed theft events, and these "
            "scores do not establish deployment performance."
        ),
    }
    save_json(payload, test_path)
    save_csv(
        pd.DataFrame(
            [
                {"model": frozen["selected_model"], "level": "window", **window_metrics},
                {
                    "model": frozen["selected_model"],
                    "level": "customer_mean",
                    **customer_metrics,
                },
            ]
        ),
        RESULTS_DIR / "test_metrics.csv",
    )
    customers.insert(0, "model", frozen["selected_model"])
    save_csv(customers, RESULTS_DIR / "test_customer_probabilities.csv")
    print(
        f"Frozen test window ROC-AUC={window_metrics['ROC_AUC']:.6f}, "
        f"AP={window_metrics['PR_AUC_average_precision']:.6f}, "
        f"F1={window_metrics['F1']:.6f}"
    )
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
