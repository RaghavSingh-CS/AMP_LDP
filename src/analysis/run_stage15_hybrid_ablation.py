"""Stage 15: validation-only frozen-FC hybrid ablation.

Only train and validation windows are permitted. The FC checkpoint is loaded in
evaluation mode and is never trained or adapted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.run_stage14_representation import (
    FEATURE_NAMES as SUMMARY_FEATURE_NAMES,
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
from src.models.autoencoder import Autoencoder, count_trainable_parameters


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_15"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_15"
PREREGISTRATION_PATH = RESULTS_DIR / "preregistered_config.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "autoencoder" / "ae_stride30_baseline.pt"
FC_CONFIG_PATH = (
    PROJECT_ROOT / "models" / "autoencoder" / "ae_stride30_baseline_config.json"
)
SAVED_VALIDATION_MSE_PATH = (
    PROJECT_ROOT
    / "results"
    / "autoencoder"
    / "stride_30"
    / "val_reconstruction_errors.parquet"
)
ALLOWED_SPLITS = ("train", "val")
RECON_FEATURE_NAMES = (
    "reconstruction_mse",
    "reconstruction_mae",
    "maximum_absolute_reconstruction_error",
    "early_half_reconstruction_mse",
    "late_half_reconstruction_mse",
    "late_minus_early_reconstruction_mse",
)
LATENT_FEATURE_NAMES = ("latent_1", "latent_2", "latent_3", "latent_4")
FC_FEATURE_NAMES = (*RECON_FEATURE_NAMES, *LATENT_FEATURE_NAMES)
REPRESENTATIONS = (
    "SUMMARY_ONLY",
    "FC_RECON_ONLY",
    "FC_ONLY",
    "SUMMARY_PLUS_RECON",
    "SUMMARY_PLUS_FC",
)


def load_preregistration() -> dict[str, Any]:
    if not PREREGISTRATION_PATH.is_file():
        raise FileNotFoundError("Stage 15 pre-registration must exist before scoring")
    with PREREGISTRATION_PATH.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not config.get("created_before_stage_15_validation_scoring"):
        raise ValueError("Stage 15 pre-registration is not certified pre-validation")
    if tuple(config["reconstruction_feature_names_in_order"]) != RECON_FEATURE_NAMES:
        raise ValueError("Reconstruction feature order differs from pre-registration")
    if tuple(config["latent_feature_names_in_order"]) != LATENT_FEATURE_NAMES:
        raise ValueError("Latent feature order differs from pre-registration")
    if not config["test_policy"].startswith("CLOSED"):
        raise ValueError("Stage 15 test policy is not closed")
    return config


def load_allowed_split(split: str) -> WindowMatrix:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Stage 15 permits only {ALLOWED_SPLITS}; got {split!r}")
    return load_window_matrix(split)


def load_frozen_fc(config: dict[str, Any]) -> tuple[Autoencoder, dict[str, Any]]:
    frozen = config["frozen_fc"]
    if file_sha256(CHECKPOINT_PATH) != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen FC checkpoint hash changed")
    if file_sha256(FC_CONFIG_PATH) != frozen["config_sha256"]:
        raise ValueError("Frozen FC config hash changed")
    with FC_CONFIG_PATH.open("r", encoding="utf-8") as source:
        saved_config = json.load(source)
    if saved_config["model"]["layer_dimensions"] != frozen["architecture"]:
        raise ValueError("Frozen FC architecture differs from pre-registration")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if checkpoint["architecture_configuration"]["layer_dimensions"] != frozen[
        "architecture"
    ]:
        raise ValueError("Checkpoint architecture metadata is inconsistent")
    if checkpoint["best_epoch"] != 30 or checkpoint["stride"] != 30:
        raise ValueError("Unexpected frozen FC checkpoint lineage")
    model = Autoencoder().cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if count_trainable_parameters(model) != 0:
        raise RuntimeError("Frozen FC parameters still require gradients")
    return model, checkpoint


@torch.no_grad()
def extract_fc_features(model: Autoencoder, raw: np.ndarray, batch_size: int) -> np.ndarray:
    """Extract six residual summaries and the frozen four-unit bottleneck."""
    if raw.ndim != 2 or raw.shape[1] != 30:
        raise ValueError(f"Expected (n, 30) FC input; got {raw.shape}")
    output = np.empty((len(raw), len(FC_FEATURE_NAMES)), dtype=np.float32)
    for first in range(0, len(raw), batch_size):
        last = min(first + batch_size, len(raw))
        inputs = torch.from_numpy(raw[first:last]).to(dtype=torch.float32, device="cpu")
        latent = model.encoder(inputs)
        reconstruction = model.decoder(latent)
        residual = reconstruction - inputs
        squared = residual.square()
        absolute = residual.abs()
        early_mse = squared[:, :15].mean(dim=1)
        late_mse = squared[:, 15:].mean(dim=1)
        batch = torch.column_stack(
            [
                squared.mean(dim=1),
                absolute.mean(dim=1),
                absolute.max(dim=1).values,
                early_mse,
                late_mse,
                late_mse - early_mse,
                latent,
            ]
        )
        if not torch.isfinite(batch).all():
            raise FloatingPointError("Non-finite frozen-FC feature encountered")
        output[first:last] = batch.numpy()
    return output


def representation_matrix(
    summaries: np.ndarray, fc_features: np.ndarray, representation: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    recon = fc_features[:, : len(RECON_FEATURE_NAMES)]
    if representation == "SUMMARY_ONLY":
        return summaries, tuple(SUMMARY_FEATURE_NAMES)
    if representation == "FC_RECON_ONLY":
        return recon, RECON_FEATURE_NAMES
    if representation == "FC_ONLY":
        return fc_features, FC_FEATURE_NAMES
    if representation == "SUMMARY_PLUS_RECON":
        return (
            np.ascontiguousarray(np.column_stack([summaries, recon]), dtype=np.float32),
            (*SUMMARY_FEATURE_NAMES, *RECON_FEATURE_NAMES),
        )
    if representation == "SUMMARY_PLUS_FC":
        return (
            np.ascontiguousarray(
                np.column_stack([summaries, fc_features]), dtype=np.float32
            ),
            (*SUMMARY_FEATURE_NAMES, *FC_FEATURE_NAMES),
        )
    raise ValueError(f"Unknown representation: {representation}")


def make_hgb() -> HistGradientBoostingClassifier:
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


def evaluate(
    model: HistGradientBoostingClassifier,
    representation: str,
    matrix: np.ndarray,
    validation: WindowMatrix,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    scores = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    window_threshold = select_f1_threshold(validation.y, scores)
    window_metrics = classification_metrics(validation.y, scores, window_threshold)
    customers = aggregate_customer_mean(validation.y, scores, validation.customer_ids)
    customer_y = customers[LABEL_COLUMN].to_numpy(dtype=np.int8)
    customer_scores = customers["mean_probability"].to_numpy(dtype=np.float64)
    customer_threshold = select_f1_threshold(customer_y, customer_scores)
    customer_metrics = classification_metrics(
        customer_y, customer_scores, customer_threshold
    )
    rows = [
        {"representation": representation, "level": "window", **window_metrics},
        {
            "representation": representation,
            "level": "customer_mean",
            **customer_metrics,
        },
    ]
    customers.insert(0, "representation", representation)
    return rows, customers


def verify_saved_validation_mse(
    validation: WindowMatrix, fc_features: np.ndarray
) -> dict[str, Any]:
    saved = pd.read_parquet(
        SAVED_VALIDATION_MSE_PATH,
        columns=["window_id", ID_COLUMN, LABEL_COLUMN, "reconstruction_mse"],
    )
    computed = pd.DataFrame(
        {
            "window_id": validation.window_ids,
            ID_COLUMN: validation.customer_ids,
            LABEL_COLUMN: validation.y,
            "computed_mse": fc_features[:, 0],
        }
    )
    if saved["window_id"].duplicated().any() or computed["window_id"].duplicated().any():
        raise ValueError("Validation reconstruction artifacts contain duplicate IDs")
    aligned = saved.merge(
        computed,
        on="window_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_saved", "_computed"),
        indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Computed and saved validation FC windows do not align")
    if not np.array_equal(
        aligned[f"{ID_COLUMN}_saved"].astype(str).to_numpy(),
        aligned[f"{ID_COLUMN}_computed"].astype(str).to_numpy(),
    ):
        raise ValueError("FC validation customer alignment differs")
    if not np.array_equal(
        aligned[f"{LABEL_COLUMN}_saved"].to_numpy(dtype=np.int8),
        aligned[f"{LABEL_COLUMN}_computed"].to_numpy(dtype=np.int8),
    ):
        raise ValueError("FC validation labels differ after alignment")
    differences = (
        aligned["reconstruction_mse"].to_numpy(dtype=np.float64)
        - aligned["computed_mse"].to_numpy(dtype=np.float64)
    )
    maximum = float(np.max(np.abs(differences)))
    if maximum > 1e-6:
        raise ValueError(f"Computed FC MSE differs from saved output by {maximum}")
    return {
        "saved_artifact": relative_path(SAVED_VALIDATION_MSE_PATH),
        "saved_artifact_sha256": file_sha256(SAVED_VALIDATION_MSE_PATH),
        "aligned_windows": int(len(aligned)),
        "missing_or_extra_windows": 0,
        "customer_ids_aligned": True,
        "labels_aligned": True,
        "maximum_absolute_mse_difference": maximum,
        "verified_within_tolerance": 1e-6,
    }


def distribution_summary(
    split: str, features: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(FC_FEATURE_NAMES):
        values = features[:, index].astype(np.float64, copy=False)
        rows.append(
            {
                "split": split,
                "feature": name,
                "count": int(len(values)),
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


def stage14_control_check(metrics: pd.DataFrame) -> dict[str, Any]:
    stage14_path = (
        PROJECT_ROOT / "results" / "supervised" / "stage_14" / "validation_metrics.csv"
    )
    stage14 = pd.read_csv(stage14_path)
    prior = stage14.loc[
        stage14["family"].eq("hist_gradient_boosting")
        & stage14["weighting"].eq("unweighted")
        & stage14["representation"].eq("SUMMARY_ONLY")
    ]
    comparisons: list[dict[str, Any]] = []
    for level in ("window", "customer_mean"):
        current = metrics.loc[
            metrics["representation"].eq("SUMMARY_ONLY")
            & metrics["level"].eq(level)
        ].iloc[0]
        old = prior.loc[prior["level"].eq(level)].iloc[0]
        differences = {
            metric: float(current[metric] - old[metric])
            for metric in (
                "ROC_AUC",
                "PR_AUC_average_precision",
                "precision",
                "recall",
                "F1",
                "threshold",
            )
        }
        comparisons.append({"level": level, "differences": differences})
    maximum = max(
        abs(value)
        for comparison in comparisons
        for value in comparison["differences"].values()
    )
    if maximum > 1e-12:
        raise ValueError("Stage 15 SUMMARY_ONLY does not reproduce Stage 14")
    return {
        "stage14_artifact": relative_path(stage14_path),
        "comparisons": comparisons,
        "maximum_absolute_metric_difference": float(maximum),
        "reproduced_within_1e-12": True,
    }


def customer_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    customer = metrics.loc[metrics["level"].eq("customer_mean")].copy()
    baseline = customer.loc[customer["representation"].eq("SUMMARY_ONLY")].iloc[0]
    rows: list[dict[str, Any]] = []
    for _, row in customer.iterrows():
        output = row.to_dict()
        for metric in ("ROC_AUC", "PR_AUC_average_precision", "F1"):
            output[f"{metric}_absolute_delta_vs_summary"] = float(
                row[metric] - baseline[metric]
            )
        rows.append(output)
    return pd.DataFrame(rows)


def fc_summary_correlations(
    validation_summaries: np.ndarray, validation_fc: np.ndarray
) -> pd.DataFrame:
    summary_frame = pd.DataFrame(validation_summaries, columns=SUMMARY_FEATURE_NAMES)
    fc_frame = pd.DataFrame(validation_fc, columns=FC_FEATURE_NAMES)
    combined = pd.concat([summary_frame, fc_frame], axis=1)
    pearson = combined.corr(method="pearson")
    spearman = combined.corr(method="spearman")
    rows: list[dict[str, Any]] = []
    for fc_name in FC_FEATURE_NAMES:
        for summary_name in SUMMARY_FEATURE_NAMES:
            rows.append(
                {
                    "fc_feature": fc_name,
                    "summary_feature": summary_name,
                    "pearson": float(pearson.loc[fc_name, summary_name]),
                    "spearman": float(spearman.loc[fc_name, summary_name]),
                    "absolute_pearson": float(abs(pearson.loc[fc_name, summary_name])),
                    "absolute_spearman": float(
                        abs(spearman.loc[fc_name, summary_name])
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["fc_feature", "absolute_spearman"],
        ascending=[True, False],
        kind="mergesort",
    )


def hybrid_permutation_importance(
    model: HistGradientBoostingClassifier,
    matrix: np.ndarray,
    feature_names: tuple[str, ...],
    validation: WindowMatrix,
    repetitions: int = 5,
) -> pd.DataFrame:
    base_scores = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    base_customers = aggregate_customer_mean(
        validation.y, base_scores, validation.customer_ids
    )
    base_y = base_customers[LABEL_COLUMN].to_numpy(dtype=np.int8)
    base_probabilities = base_customers["mean_probability"].to_numpy(dtype=np.float64)
    base_roc = float(roc_auc_score(base_y, base_probabilities))
    base_ap = float(average_precision_score(base_y, base_probabilities))
    rng = np.random.default_rng(RANDOM_STATE)
    permuted = matrix.copy()
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        original = matrix[:, index].copy()
        roc_decreases: list[float] = []
        ap_decreases: list[float] = []
        for _ in range(repetitions):
            permuted[:, index] = original[rng.permutation(len(original))]
            scores = np.asarray(model.predict_proba(permuted)[:, 1], dtype=np.float64)
            customers = aggregate_customer_mean(
                validation.y, scores, validation.customer_ids
            )
            labels = customers[LABEL_COLUMN].to_numpy(dtype=np.int8)
            probabilities = customers["mean_probability"].to_numpy(dtype=np.float64)
            roc_decreases.append(base_roc - roc_auc_score(labels, probabilities))
            ap_decreases.append(
                base_ap - average_precision_score(labels, probabilities)
            )
            permuted[:, index] = original
        rows.append(
            {
                "feature": name,
                "feature_source": "frozen_fc" if name in FC_FEATURE_NAMES else "stage14_summary",
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


def paired_customer_bootstrap(
    baseline: pd.DataFrame,
    hybrid: pd.DataFrame,
    replicates: int = 2000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [ID_COLUMN, LABEL_COLUMN, "mean_probability"]
    base = baseline.loc[:, columns].sort_values(ID_COLUMN).reset_index(drop=True)
    candidate = hybrid.loc[:, columns].sort_values(ID_COLUMN).reset_index(drop=True)
    if not base[ID_COLUMN].equals(candidate[ID_COLUMN]):
        raise ValueError("Bootstrap customer IDs do not align")
    if not base[LABEL_COLUMN].equals(candidate[LABEL_COLUMN]):
        raise ValueError("Bootstrap customer labels do not align")
    labels = base[LABEL_COLUMN].to_numpy(dtype=np.int8)
    base_scores = base["mean_probability"].to_numpy(dtype=np.float64)
    hybrid_scores = candidate["mean_probability"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    for replicate in range(1, replicates + 1):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_y = labels[indices]
        if np.unique(sampled_y).size != 2:
            raise RuntimeError("Customer bootstrap replicate contains only one class")
        base_roc = roc_auc_score(sampled_y, base_scores[indices])
        hybrid_roc = roc_auc_score(sampled_y, hybrid_scores[indices])
        base_ap = average_precision_score(sampled_y, base_scores[indices])
        hybrid_ap = average_precision_score(sampled_y, hybrid_scores[indices])
        rows.append(
            {
                "replicate": replicate,
                "ROC_AUC_delta_hybrid_minus_summary": float(hybrid_roc - base_roc),
                "average_precision_delta_hybrid_minus_summary": float(
                    hybrid_ap - base_ap
                ),
            }
        )
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "unit": "customer",
        "paired": True,
        "replicates": replicates,
        "seed": RANDOM_STATE,
        "comparison": "SUMMARY_PLUS_FC minus SUMMARY_ONLY",
    }
    for column, label in (
        ("ROC_AUC_delta_hybrid_minus_summary", "ROC_AUC_delta"),
        (
            "average_precision_delta_hybrid_minus_summary",
            "average_precision_delta",
        ),
    ):
        values = frame[column].to_numpy(dtype=np.float64)
        summary[label] = {
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_standard_deviation": float(np.std(values, ddof=1)),
            "percentile_95_CI": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "proportion_above_zero": float(np.mean(values > 0)),
        }
    return frame, summary


def main() -> None:
    preregistration = load_preregistration()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "validation_results_summary.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 15 results: {result_path}")

    torch.manual_seed(RANDOM_STATE)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, checkpoint = load_frozen_fc(preregistration)
    print("Loading Stage 15 allowed splits: train and val only...")
    train = load_allowed_split("train")
    validation = load_allowed_split("val")
    if set(train.customer_ids) & set(validation.customer_ids):
        raise ValueError("Train and validation customers overlap")

    print("Extracting frozen-FC features without gradients or label input...")
    batch_size = int(preregistration["frozen_fc"]["inference_batch_size"])
    train_fc = extract_fc_features(model, train.X, batch_size)
    validation_fc = extract_fc_features(model, validation.X, batch_size)
    mse_verification = verify_saved_validation_mse(validation, validation_fc)
    train_summaries = behavioural_features(train.X)
    validation_summaries = behavioural_features(validation.X)

    distributions = pd.concat(
        [
            distribution_summary("train", train_fc),
            distribution_summary("val", validation_fc),
        ],
        ignore_index=True,
    )
    if distributions["nonfinite_count"].ne(0).any():
        raise ValueError("Non-finite FC-derived feature found")
    save_csv(distributions, RESULTS_DIR / "fc_feature_distribution_summary.csv")

    rows: list[dict[str, Any]] = []
    customer_outputs: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, HistGradientBoostingClassifier] = {}
    records: list[dict[str, Any]] = []
    feature_names_by_representation: dict[str, tuple[str, ...]] = {}
    for representation in REPRESENTATIONS:
        train_matrix, feature_names = representation_matrix(
            train_summaries, train_fc, representation
        )
        validation_matrix, validation_feature_names = representation_matrix(
            validation_summaries, validation_fc, representation
        )
        if feature_names != validation_feature_names:
            raise ValueError("Train and validation feature names differ")
        feature_names_by_representation[representation] = feature_names
        hgb = make_hgb()
        print(f"Fitting frozen-config HGB on {representation}...")
        started = time.perf_counter()
        hgb.fit(train_matrix, train.y)
        runtime = time.perf_counter() - started
        result_rows, customers = evaluate(
            hgb, representation, validation_matrix, validation
        )
        rows.extend(result_rows)
        customer_outputs[representation] = customers
        fitted_models[representation] = hgb
        model_path = MODEL_DIR / f"hgb_unweighted_{representation.lower()}.joblib"
        joblib.dump(hgb, model_path, compress=3)
        customer_row = next(row for row in result_rows if row["level"] == "customer_mean")
        records.append(
            {
                "representation": representation,
                "feature_count": len(feature_names),
                "feature_names": list(feature_names),
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
            f"AP={customer_row['PR_AUC_average_precision']:.6f}; {runtime:.2f}s"
        )

    metrics = pd.DataFrame(rows)
    save_csv(metrics, RESULTS_DIR / "validation_metrics.csv")
    save_csv(customer_deltas(metrics), RESULTS_DIR / "customer_deltas.csv")
    save_csv(
        pd.concat(customer_outputs.values(), ignore_index=True),
        RESULTS_DIR / "validation_customer_probabilities.csv",
    )
    control = stage14_control_check(metrics)

    correlations = fc_summary_correlations(validation_summaries, validation_fc)
    save_csv(correlations, RESULTS_DIR / "fc_summary_correlations.csv")

    customer_metrics = metrics.loc[metrics["level"].eq("customer_mean")]
    hybrid_metrics = customer_metrics.loc[
        customer_metrics["representation"].isin(
            ["SUMMARY_PLUS_RECON", "SUMMARY_PLUS_FC"]
        )
    ].sort_values(
        ["ROC_AUC", "PR_AUC_average_precision", "representation"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    best_hybrid = str(hybrid_metrics.iloc[0]["representation"])
    best_hybrid_matrix, best_hybrid_names = representation_matrix(
        validation_summaries, validation_fc, best_hybrid
    )
    importance = hybrid_permutation_importance(
        fitted_models[best_hybrid],
        best_hybrid_matrix,
        best_hybrid_names,
        validation,
    )
    save_csv(importance, RESULTS_DIR / "hybrid_permutation_importance.csv")

    bootstrap_frame, bootstrap_summary = paired_customer_bootstrap(
        customer_outputs["SUMMARY_ONLY"],
        customer_outputs["SUMMARY_PLUS_FC"],
        replicates=int(preregistration["bootstrap_policy"]["replicates"]),
    )
    save_csv(bootstrap_frame, RESULTS_DIR / "customer_bootstrap_deltas.csv")
    save_json(bootstrap_summary, RESULTS_DIR / "customer_bootstrap_summary.json")

    summary_baseline = customer_metrics.loc[
        customer_metrics["representation"].eq("SUMMARY_ONLY")
    ].iloc[0]
    full_hybrid = customer_metrics.loc[
        customer_metrics["representation"].eq("SUMMARY_PLUS_FC")
    ].iloc[0]
    save_json(
        {
            "stage": "Stage 15 — frozen-FC validation-only hybrid ablation",
            "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
            "frozen_fc_audit": {
                "checkpoint": relative_path(CHECKPOINT_PATH),
                "checkpoint_sha256_verified": file_sha256(CHECKPOINT_PATH),
                "config": relative_path(FC_CONFIG_PATH),
                "config_sha256_verified": file_sha256(FC_CONFIG_PATH),
                "architecture": [30, 16, 8, 4, 8, 16, 30],
                "parameters": 1362,
                "training_population": "708,308 usable stride-30 FLAG=0 train windows",
                "checkpoint_selection": (
                    "epoch 30 by normal-validation reconstruction MSE; inherited validation "
                    "reuse limitation applies"
                ),
                "input": "30 Stage 2 scaled consumption values",
                "reconstruction_error": "mean((reconstruction - input)^2) over 30 positions",
                "existing_p95_threshold": 0.16436323150992393,
                "threshold_used_as_stage15_feature": False,
                "latent_extraction": "model.encoder output; four-unit ReLU bottleneck",
                "retrained_or_adapted": False,
                "checkpoint_metadata": {
                    "best_epoch": int(checkpoint["best_epoch"]),
                    "best_validation_loss": float(checkpoint["best_validation_loss"]),
                    "stride": int(checkpoint["stride"]),
                },
            },
            "alignment_and_reconstruction_verification": mse_verification,
            "feature_sanity": {
                "train_shape": list(train_fc.shape),
                "validation_shape": list(validation_fc.shape),
                "train_nonfinite": int(np.sum(~np.isfinite(train_fc))),
                "validation_nonfinite": int(np.sum(~np.isfinite(validation_fc))),
                "effectively_constant": distributions.loc[
                    distributions["effectively_constant"], "feature"
                ].drop_duplicates().tolist(),
            },
            "stage14_control": control,
            "model_records": records,
            "best_pre_registered_hybrid_for_importance": best_hybrid,
            "central_customer_delta": {
                "comparison": "SUMMARY_PLUS_FC minus SUMMARY_ONLY",
                "ROC_AUC": float(full_hybrid["ROC_AUC"] - summary_baseline["ROC_AUC"]),
                "PR_AUC_average_precision": float(
                    full_hybrid["PR_AUC_average_precision"]
                    - summary_baseline["PR_AUC_average_precision"]
                ),
                "F1": float(full_hybrid["F1"] - summary_baseline["F1"]),
            },
            "bootstrap": bootstrap_summary,
            "inherited_validation_reuse_limitation": preregistration[
                "inherited_validation_reuse_limitation"
            ],
            "test_access": {
                "loaded_splits": ["train", "val"],
                "test_features_loaded": False,
                "test_labels_loaded": False,
                "test_predictions_loaded_or_inspected": False,
                "historical_test_metrics_used": False,
            },
        },
        result_path,
    )
    print(f"Best pre-registered hybrid: {best_hybrid}")
    print(
        "Central SUMMARY_PLUS_FC customer delta: "
        f"ROC-AUC={full_hybrid['ROC_AUC'] - summary_baseline['ROC_AUC']:+.6f}, "
        "AP="
        f"{full_hybrid['PR_AUC_average_precision'] - summary_baseline['PR_AUC_average_precision']:+.6f}"
    )
    print("Stage 15 completed with train and validation only; test was not accessed.")


if __name__ == "__main__":
    main()
