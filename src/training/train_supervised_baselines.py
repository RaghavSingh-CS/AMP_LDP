"""Stage 13: validation-only supervised signal diagnostic and model freeze."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    MODEL_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_DIR,
    aggregate_customer_mean,
    classification_metrics,
    file_sha256,
    load_window_matrix,
    relative_path,
    save_csv,
    save_json,
    select_f1_threshold,
)


MODEL_SPECS: tuple[tuple[str, object], ...] = (
    ("dummy_prior", DummyClassifier(strategy="prior", random_state=RANDOM_STATE)),
    (
        "logistic_unweighted",
        Pipeline(
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
        ),
    ),
    (
        "logistic_balanced",
        Pipeline(
            [
                ("standardize", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=300,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    ),
    (
        "hist_gradient_boosting_unweighted",
        HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            class_weight=None,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "hist_gradient_boosting_balanced",
        HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
    ),
)


def split_audit() -> dict[str, Any]:
    """Audit aggregate labels and established split artifacts without test features."""
    manifest_path = PROJECT_ROOT / "data" / "processed" / "split_manifest.parquet"
    filter_path = (
        PROJECT_ROOT
        / "data"
        / "windows"
        / "stride_30"
        / "usable"
        / "filter_summary.json"
    )
    manifest = pd.read_parquet(manifest_path, engine="pyarrow")
    with filter_path.open("r", encoding="utf-8") as source:
        filter_summary = json.load(source)
    if len(manifest) != 42_367 or manifest[ID_COLUMN].nunique() != len(manifest):
        raise ValueError("Authoritative split manifest is incomplete or duplicated")
    id_sets = {
        name: set(manifest.loc[manifest["split"].eq(name), ID_COLUMN].astype(str))
        for name in ("train", "val", "test")
    }
    overlaps = {
        "train_val": len(id_sets["train"] & id_sets["val"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "val_test": len(id_sets["val"] & id_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Customer leakage detected: {overlaps}")

    splits: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        customer_rows = manifest.loc[manifest["split"].eq(name)]
        customer_counts = customer_rows[LABEL_COLUMN].value_counts()
        window_summary = filter_summary["splits"][name]
        splits[name] = {
            "customers": int(len(customer_rows)),
            "normal_customers": int(customer_counts.get(0, 0)),
            "theft_customers": int(customer_counts.get(1, 0)),
            "customer_theft_prevalence": float(customer_rows[LABEL_COLUMN].mean()),
            "usable_windows": int(window_summary["retained_windows"]),
            "normal_usable_windows": int(window_summary["retained_normal_windows"]),
            "theft_usable_windows": int(window_summary["retained_theft_windows"]),
            "window_theft_prevalence": float(
                window_summary["retained_theft_windows"]
                / window_summary["retained_windows"]
            ),
            "customers_with_usable_windows": int(
                window_summary["customer_coverage"]["retained_customers"]
            ),
        }

    return {
        "stage": "Stage 13 — supervised signal diagnostic split audit",
        "authoritative_assignment": relative_path(manifest_path),
        "split_method": "customer-level 70/15/15 stratified by FLAG; random_state=42",
        "customer_overlap": overlaps,
        "total_customers": int(len(manifest)),
        "normal_customers": int(manifest[LABEL_COLUMN].eq(0).sum()),
        "theft_customers": int(manifest[LABEL_COLUMN].eq(1).sum()),
        "overall_customer_theft_prevalence": float(manifest[LABEL_COLUMN].mean()),
        "canonical_stride_30_windows": int(filter_summary["overall"]["original_windows"]),
        "usable_stride_30_windows": int(filter_summary["overall"]["retained_windows"]),
        "splits": splits,
        "label_granularity": (
            "FLAG is customer-level. Every window inherits its customer's FLAG; "
            "a FLAG=1 window is not a confirmed theft event."
        ),
        "supervised_features": [
            "30 ordered float32 consumption values from each existing usable stride-30 window",
            "Stage 1 customer-wise imputed and log1p-transformed",
            "Stage 2 StandardScaled with training-customer statistics only",
            "logistic regression adds a train-window-fitted StandardScaler inside its pipeline",
        ],
        "excluded_from_model_input": [
            "FLAG/target",
            "CONS_NO/customer identifier",
            "split indicator",
            "window_id",
            "window_number_within_customer",
            "start_feature_index and end_feature_index",
            "start_date and end_date",
            "missingness mask, missing_count, and missing_fraction",
            "all reconstruction errors, anomaly flags, and prior model outputs",
        ],
        "test_access_during_validation_phase": (
            "Aggregate customer counts were read from the authoritative split manifest and "
            "aggregate window counts from the pre-existing Stage 3C summary solely for the "
            "requested split audit. No test feature matrix or per-window test labels were loaded, "
            "and no test result can affect model/configuration/threshold selection."
        ),
    }


def model_config() -> dict[str, Any]:
    return {
        "stage": "Stage 13 — simple supervised baselines",
        "seed": RANDOM_STATE,
        "input": "existing usable stride-30 windows; 30 scaled consumption values only",
        "models": {
            "dummy_prior": {"strategy": "prior"},
            "logistic_unweighted": {
                "standardization": "StandardScaler fitted on training windows only",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 300,
                "class_weight": None,
            },
            "logistic_balanced": {
                "standardization": "StandardScaler fitted on training windows only",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 300,
                "class_weight": "balanced",
            },
            "hist_gradient_boosting_unweighted": {
                "learning_rate": 0.1,
                "max_iter": 100,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 100,
                "l2_regularization": 1.0,
                "class_weight": None,
                "early_stopping": False,
            },
            "hist_gradient_boosting_balanced": {
                "learning_rate": 0.1,
                "max_iter": 100,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 100,
                "l2_regularization": 1.0,
                "class_weight": "balanced",
                "early_stopping": False,
            },
        },
        "imbalance_policy": "compare unweighted and balanced class weights; no resampling",
        "window_threshold_policy": "maximize positive-class F1 on validation",
        "customer_aggregation": "arithmetic mean of usable-window predicted probabilities",
        "customer_threshold_policy": "maximize positive-class F1 on validation customer means",
        "selection_policy": (
            "highest validation window ROC-AUC, then window average precision, among non-dummy "
            "models; no test information"
        ),
        "test_policy": (
            "Stages 6, 8, 10, and 12 establish one frozen test evaluation after validation-only "
            "selection. Stage 13 follows that policy for the selected supervised model only."
        ),
    }


def metric_row(model_name: str, level: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"model": model_name, "level": level, **metrics}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frozen_path = RESULTS_DIR / "frozen_decision.json"
    if frozen_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing frozen Stage 13 decision: {frozen_path}"
        )

    audit = split_audit()
    save_json(audit, RESULTS_DIR / "split_audit.json")
    save_json(model_config(), RESULTS_DIR / "experiment_config.json")

    print("Loading train and validation windows only...")
    train = load_window_matrix("train")
    validation = load_window_matrix("val")
    print(f"train={len(train.y):,}, validation={len(validation.y):,}")

    metric_rows: list[dict[str, Any]] = []
    customer_frames: list[pd.DataFrame] = []
    model_records: list[dict[str, Any]] = []
    for model_name, model in MODEL_SPECS:
        print(f"Fitting {model_name}...")
        started = time.perf_counter()
        model.fit(train.X, train.y)
        runtime = time.perf_counter() - started
        scores = np.asarray(model.predict_proba(validation.X)[:, 1], dtype=np.float64)
        window_threshold = select_f1_threshold(validation.y, scores)
        window_metrics = classification_metrics(
            validation.y, scores, window_threshold
        )

        customers = aggregate_customer_mean(
            validation.y, scores, validation.customer_ids
        )
        customer_y = customers[LABEL_COLUMN].to_numpy(dtype=np.int8)
        customer_scores = customers["mean_probability"].to_numpy(dtype=np.float64)
        customer_threshold = select_f1_threshold(customer_y, customer_scores)
        customer_metrics = classification_metrics(
            customer_y, customer_scores, customer_threshold
        )

        metric_rows.extend(
            [
                metric_row(model_name, "window", window_metrics),
                metric_row(model_name, "customer_mean", customer_metrics),
            ]
        )
        customers.insert(0, "model", model_name)
        customer_frames.append(customers)

        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path, compress=3)
        model_records.append(
            {
                "model": model_name,
                "fit_runtime_seconds": float(runtime),
                "artifact": relative_path(model_path),
                "artifact_sha256": file_sha256(model_path),
                "window_threshold": window_threshold,
                "customer_mean_threshold": customer_threshold,
                "window_ROC_AUC": window_metrics["ROC_AUC"],
                "window_PR_AUC_average_precision": window_metrics[
                    "PR_AUC_average_precision"
                ],
            }
        )
        print(
            f"  window ROC-AUC={window_metrics['ROC_AUC']:.6f}, "
            f"AP={window_metrics['PR_AUC_average_precision']:.6f}, "
            f"F1={window_metrics['F1']:.6f}; {runtime:.2f}s"
        )

    metrics = pd.DataFrame(metric_rows)
    save_csv(metrics, RESULTS_DIR / "validation_metrics.csv")
    save_csv(
        pd.concat(customer_frames, ignore_index=True),
        RESULTS_DIR / "validation_customer_probabilities.csv",
    )
    save_json(
        {"stage": "Stage 13 validation model records", "models": model_records},
        RESULTS_DIR / "training_summary.json",
    )

    window_rows = metrics.loc[
        metrics["level"].eq("window") & metrics["model"].ne("dummy_prior")
    ].sort_values(
        ["ROC_AUC", "PR_AUC_average_precision", "model"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    selected_name = str(window_rows.iloc[0]["model"])
    selected_record = next(
        record for record in model_records if record["model"] == selected_name
    )
    selected_customer = metrics.loc[
        metrics["model"].eq(selected_name)
        & metrics["level"].eq("customer_mean")
    ].iloc[0]
    frozen = {
        "stage": "Stage 13 — validation-only frozen supervised decision",
        "created_before_test_inference": True,
        "test_features_or_per_window_labels_loaded": False,
        "selection_rule": model_config()["selection_policy"],
        "selected_model": selected_name,
        "selected_model_artifact": selected_record["artifact"],
        "selected_model_sha256": selected_record["artifact_sha256"],
        "window_threshold": float(selected_record["window_threshold"]),
        "window_threshold_source": "maximum validation positive-class F1",
        "customer_aggregation": "mean predicted probability over usable windows",
        "customer_mean_threshold": float(selected_customer["threshold"]),
        "customer_threshold_source": "maximum validation customer-level positive-class F1",
        "validation_window_ROC_AUC": float(selected_record["window_ROC_AUC"]),
        "validation_window_PR_AUC_average_precision": float(
            selected_record["window_PR_AUC_average_precision"]
        ),
        "test_evaluation_authorized_by_methodology": True,
        "methodology_basis": model_config()["test_policy"],
    }
    save_json(frozen, frozen_path)
    print(f"Frozen validation-only selection: {selected_name}")
    print(f"Frozen decision: {frozen_path}")
    print("Test split feature matrix was not loaded.")


if __name__ == "__main__":
    main()
