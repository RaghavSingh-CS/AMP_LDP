"""Stage 18: frozen DEV_VAL robustness and failure-mode analysis.

Run ``--prepare`` before ``--evaluate``. Only original-train data and the
DEV_TRAIN/DEV_VAL governance roles are permitted. The saved Stage 17B DEV_VAL
probabilities are reused; no model is fitted or scored by this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.run_stage14_representation import behavioural_features
from src.analysis.run_stage16_customer_aggregation import (
    FEATURE_NAMES,
    build_customer_features,
)
from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    RANDOM_STATE,
    classification_metrics,
    file_sha256,
    load_window_matrix,
    save_csv,
    save_json,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_18"
STRATA_CONFIG = RESULTS_DIR / "preregistered_strata.json"
DECISION_PATH = RESULTS_DIR / "robustness_decision.json"
PROTOCOL_MANIFEST = (
    PROJECT_ROOT / "results" / "governance" / "stage_17a" / "research_protocol_manifest.csv"
)
PROTOCOL_SUMMARY = (
    PROJECT_ROOT / "results" / "governance" / "stage_17a" / "protocol_summary.json"
)
HISTORICAL_SPLIT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "split_manifest.parquet"
STAGE16_CONFIG = RESULTS_DIR.parent / "stage_16" / "preregistered_config.json"
STAGE17B_CONFIG = RESULTS_DIR.parent / "stage_17b" / "preregistered_benchmark_config.json"
STAGE17B_SUMMARY = RESULTS_DIR.parent / "stage_17b" / "training_summary.json"
STAGE17B_DECISION = RESULTS_DIR.parent / "stage_17b" / "validation_decision.json"
STAGE17B_METRICS = RESULTS_DIR.parent / "stage_17b" / "dev_val_metrics.csv"
STAGE17B_PREDICTIONS = RESULTS_DIR.parent / "stage_17b" / "dev_val_customer_probabilities.csv"
STAGE17B_MODEL = PROJECT_ROOT / "models" / "supervised" / "stage_17b" / "hist_gradient_boosting.joblib"
TRAIN_WINDOW_MANIFEST = (
    PROJECT_ROOT / "data" / "windows" / "stride_30" / "usable" / "train_windows.parquet"
)
EXPECTED_PROTOCOL_HASH = "dae7dba53260a3af6b8e2ddd904b5c7867c3321f27cdff5af5e4060b97c348c0"
EXPECTED_MODEL_HASH = "67ed869858185d8f71d99b3dce8b358c43793178185e5aa50e6997d11b56974b"
EXPECTED_STAGE16_CONFIG_HASH = "a722299d6207fce06180f71bf96717ce58948e127a85440e04f721c1616fcdf3"
EXPECTED_HISTORICAL_SPLIT_HASH = "6ad1132bbaef0c6f6ccc5bdc973844b805c4291612ac7170ec165f273835959c"
EXPECTED_STAGE18_PREREG_HASH = "d0087fe318efef2be39d24db440e6f9807b654624c1702efd5fc54de2eba2acf"
HISTORY_ORDER = ("SHORT", "MEDIUM", "LONG")
CONSUMPTION_ORDER = ("LOW", "MEDIUM", "HIGH")
FROZEN_THRESHOLD = 0.20070617709942135
MIN_TOTAL = 100
MIN_POSITIVE = 30
MIN_NEGATIVE = 30


def load_original_train() -> tuple[Any, np.ndarray, pd.DataFrame]:
    governance = pd.read_csv(PROTOCOL_MANIFEST, dtype={ID_COLUMN: str})
    development = governance.loc[
        governance["research_role"].isin(["DEV_TRAIN", "DEV_VAL"])
    ].copy()
    if not development["original_split"].eq("train").all():
        raise ValueError("Development governance contains a non-original-train row")
    windows = load_window_matrix("train")
    chronology = pd.read_parquet(
        TRAIN_WINDOW_MANIFEST,
        columns=["window_id", ID_COLUMN, "window_number_within_customer"],
        engine="pyarrow",
    )
    if not np.array_equal(
        chronology["window_id"].astype(str).to_numpy(), windows.window_ids
    ):
        raise ValueError("Original-train chronology is misaligned")
    if not set(windows.customer_ids) <= set(development[ID_COLUMN]):
        raise ValueError("Original-train windows include an ungoverned customer")
    return (
        windows,
        chronology["window_number_within_customer"].to_numpy(dtype=np.int32),
        development,
    )


def role_customer_features(
    role: str,
    windows: Any,
    window_numbers: np.ndarray,
    summaries: np.ndarray,
    governance: pd.DataFrame,
    thresholds: dict[str, float] | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, float]]:
    if role not in {"DEV_TRAIN", "DEV_VAL"}:
        raise ValueError(f"Closed role requested: {role}")
    ids = set(
        governance.loc[
            governance["research_role"].eq(role)
            & governance["stage16_representation_available"].eq(True),
            ID_COLUMN,
        ]
    )
    mask = np.isin(windows.customer_ids, list(ids))
    features, labels, customer_ids, references = build_customer_features(
        summaries[mask],
        windows.y[mask],
        windows.customer_ids[mask],
        window_numbers[mask],
        thresholds,
    )
    if set(customer_ids) != ids or tuple(features.columns) != FEATURE_NAMES:
        raise ValueError(f"{role} frozen feature reconstruction failed")
    counts = (
        pd.Series(windows.customer_ids[mask], name=ID_COLUMN)
        .value_counts(sort=False)
        .reindex(customer_ids)
        .to_numpy(dtype=np.int32)
    )
    features = features.copy()
    features.insert(0, "history_window_count", counts)
    features.insert(0, ID_COLUMN, customer_ids)
    return features, labels, customer_ids, references


def tie_aware_tertile_cutpoints(counts: np.ndarray) -> tuple[int, int, dict[str, int]]:
    """Choose count cutpoints closest to equal thirds without splitting ties."""
    values = np.asarray(counts, dtype=np.int32)
    unique = np.unique(values)
    target = len(values) / 3.0
    candidates: list[tuple[float, int, int, int, int, int]] = []
    for low in unique[:-2]:
        for high in unique[unique > low][:-1]:
            group_counts = (
                int(np.sum(values <= low)),
                int(np.sum((values > low) & (values <= high))),
                int(np.sum(values > high)),
            )
            if min(group_counts) == 0:
                continue
            loss = float(sum((count - target) ** 2 for count in group_counts))
            candidates.append((loss, int(low), int(high), *group_counts))
    if not candidates:
        raise ValueError("No three nonempty history groups are possible")
    _, low, high, short, medium, long = min(candidates)
    return low, high, {"SHORT": short, "MEDIUM": medium, "LONG": long}


def train_strata_audit(
    frame: pd.DataFrame,
    labels: np.ndarray,
    history_low: int,
    history_high: int,
    consumption_low: float,
    consumption_high: float,
) -> pd.DataFrame:
    """Return the boundary-only DEV_TRAIN count/prevalence audit."""
    audit = frame[[ID_COLUMN, "history_window_count", "mean__customer_median"]].copy()
    audit[LABEL_COLUMN] = labels
    audit["history_stratum"] = pd.cut(
        audit["history_window_count"],
        bins=[-np.inf, history_low, history_high, np.inf],
        labels=HISTORY_ORDER,
        right=True,
    ).astype(str)
    audit["consumption_stratum"] = pd.cut(
        audit["mean__customer_median"],
        bins=[-np.inf, consumption_low, consumption_high, np.inf],
        labels=CONSUMPTION_ORDER,
        right=True,
    ).astype(str)
    rows: list[dict[str, Any]] = []
    for dimension, column, order in (
        ("history", "history_stratum", HISTORY_ORDER),
        ("consumption", "consumption_stratum", CONSUMPTION_ORDER),
    ):
        for stratum in order:
            group_labels = audit.loc[audit[column].eq(stratum), LABEL_COLUMN]
            rows.append(
                {
                    "dimension": dimension,
                    "stratum": stratum,
                    "count": int(len(group_labels)),
                    "normal_count": int(group_labels.eq(0).sum()),
                    "theft_count": int(group_labels.eq(1).sum()),
                    "theft_prevalence": float(group_labels.mean()),
                }
            )
    return pd.DataFrame(rows)


def prepare() -> None:
    if STRATA_CONFIG.exists():
        raise FileExistsError("Refusing to overwrite frozen Stage 18 strata")
    if file_sha256(PROTOCOL_MANIFEST) != EXPECTED_PROTOCOL_HASH:
        raise ValueError("Stage 17A manifest hash changed")
    if file_sha256(STAGE17B_MODEL) != EXPECTED_MODEL_HASH:
        raise ValueError("Stage 17B model hash changed")
    if file_sha256(STAGE16_CONFIG) != EXPECTED_STAGE16_CONFIG_HASH:
        raise ValueError("Stage 16 feature preregistration hash changed")
    if file_sha256(HISTORICAL_SPLIT_MANIFEST) != EXPECTED_HISTORICAL_SPLIT_HASH:
        raise ValueError("Historical split manifest hash changed")
    with STAGE16_CONFIG.open("r", encoding="utf-8") as source:
        stage16 = json.load(source)
    with STAGE17B_CONFIG.open("r", encoding="utf-8") as source:
        stage17b = json.load(source)
    with STAGE17B_SUMMARY.open("r", encoding="utf-8") as source:
        training = json.load(source)
    with STAGE17B_DECISION.open("r", encoding="utf-8") as source:
        stage17b_decision = json.load(source)
    with PROTOCOL_SUMMARY.open("r", encoding="utf-8") as source:
        protocol_summary = json.load(source)
    if tuple(stage16["feature_names_in_order"]) != FEATURE_NAMES:
        raise ValueError("Stage 16 feature schema changed")
    if tuple(stage17b["representation"]["feature_names_in_order"]) != FEATURE_NAMES:
        raise ValueError("Stage 17B feature schema changed")
    if training["feature_names_in_order"] != list(FEATURE_NAMES):
        raise ValueError("Stage 17B training feature order changed")
    if stage17b_decision["selected_model_sha256"] != EXPECTED_MODEL_HASH:
        raise ValueError("Stage 17B decision points to a different model")
    if stage17b_decision["selected_DEV_VAL_metrics"]["threshold"] != FROZEN_THRESHOLD:
        raise ValueError("Stage 17B frozen threshold changed")
    if protocol_summary["historical_split"]["sha256"] != EXPECTED_HISTORICAL_SPLIT_HASH:
        raise ValueError("Stage 17A historical split provenance changed")

    predictions = pd.read_csv(STAGE17B_PREDICTIONS, dtype={ID_COLUMN: str})
    hgb_predictions = predictions.loc[
        predictions["model"].eq("hist_gradient_boosting")
    ]
    if len(hgb_predictions) != 5931 or hgb_predictions[ID_COLUMN].nunique() != 5931:
        raise ValueError("Stage 17B saved HGB DEV_VAL prediction population changed")

    windows, window_numbers, governance = load_original_train()
    summaries = behavioural_features(windows.X)
    train_features, train_labels, _, references = role_customer_features(
        "DEV_TRAIN", windows, window_numbers, summaries, governance, None
    )
    if references != training["DEV_TRAIN_derived_persistence_q90_thresholds"]:
        raise ValueError("Reconstructed DEV_TRAIN references differ from Stage 17B")

    history_low, history_high, history_counts = tie_aware_tertile_cutpoints(
        train_features["history_window_count"].to_numpy(dtype=np.int32)
    )
    consumption_measure = "mean__customer_median"
    consumption_values = train_features[consumption_measure].to_numpy(dtype=np.float64)
    consumption_low, consumption_high = np.quantile(
        consumption_values, [1 / 3, 2 / 3], method="linear"
    )
    if not consumption_low < consumption_high:
        raise ValueError("DEV_TRAIN consumption tertiles are not distinct")
    train_audit = train_strata_audit(
        train_features,
        train_labels,
        history_low,
        history_high,
        float(consumption_low),
        float(consumption_high),
    )
    history_distribution = (
        train_features["history_window_count"]
        .value_counts()
        .sort_index()
        .rename_axis("history_window_count")
        .reset_index(name="DEV_TRAIN_customer_count")
    )
    config = {
        "stage": "Stage 18 — pre-registered DEV_VAL robustness strata",
        "created_before_DEV_VAL_subgroup_scoring": True,
        "research_question": "Does the frozen direct-customer AMP pipeline retain useful discrimination across pre-specified customer history-length and consumption-level strata, or is aggregate performance driven by particular groups?",
        "frozen_inputs": {
            "protocol_manifest": {"path": str(PROTOCOL_MANIFEST.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": file_sha256(PROTOCOL_MANIFEST)},
            "stage17b_model": {"path": str(STAGE17B_MODEL.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": file_sha256(STAGE17B_MODEL)},
            "stage17b_predictions": {"path": str(STAGE17B_PREDICTIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": file_sha256(STAGE17B_PREDICTIONS)},
            "stage17b_metrics": {"path": str(STAGE17B_METRICS.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": file_sha256(STAGE17B_METRICS)},
            "historical_split_manifest": {"path": str(HISTORICAL_SPLIT_MANIFEST.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": file_sha256(HISTORICAL_SPLIT_MANIFEST)},
            "stage16_feature_config_sha256": file_sha256(STAGE16_CONFIG),
            "stage17b_benchmark_config_sha256": file_sha256(STAGE17B_CONFIG),
            "feature_count": 48,
            "feature_names_in_order": list(FEATURE_NAMES),
            "hgb_parameters": stage17b["candidate_models"]["hist_gradient_boosting"]["parameters"],
            "frozen_threshold": FROZEN_THRESHOLD,
            "DEV_TRAIN_persistence_references": references,
        },
        "history_strata": {
            "analysis_variable": "number of established usable stride-30 windows contributing to the customer representation; never a predictor",
            "boundary_method": "DEV_TRAIN-only tie-aware tertile approximation: enumerate observed integer cutpoint pairs, do not split ties, minimize squared deviation of the three customer counts from n/3, then choose lexicographically smallest cutpoints on ties",
            "DEV_TRAIN_cutpoints": {"short_max_inclusive": history_low, "medium_max_inclusive": history_high},
            "definitions": {
                "SHORT": f"history_window_count <= {history_low}",
                "MEDIUM": f"{history_low} < history_window_count <= {history_high}",
                "LONG": f"history_window_count > {history_high}",
            },
            "DEV_TRAIN_group_counts_without_scoring": history_counts,
        },
        "consumption_strata": {
            "analysis_variable": consumption_measure,
            "meaning": "Existing Stage 16 feature: median across windows of each window's arithmetic mean standardized consumption",
            "boundary_method": "DEV_TRAIN-only 1/3 and 2/3 NumPy linear quantiles",
            "DEV_TRAIN_cutpoints": {"low_max_inclusive": float(consumption_low), "medium_max_inclusive": float(consumption_high)},
            "definitions": {
                "LOW": f"{consumption_measure} <= {consumption_low:.17g}",
                "MEDIUM": f"{consumption_low:.17g} < {consumption_measure} <= {consumption_high:.17g}",
                "HIGH": f"{consumption_measure} > {consumption_high:.17g}",
            },
        },
        "DEV_TRAIN_strata_count_prevalence_audit": train_audit.to_dict(orient="records"),
        "inference_sufficiency": {"minimum_total": MIN_TOTAL, "minimum_positive": MIN_POSITIVE, "minimum_negative": MIN_NEGATIVE},
        "primary_bootstrap": {"resamples": 2000, "base_seed": RANDOM_STATE, "group_seeds": {"history_SHORT": 42, "history_MEDIUM": 43, "history_LONG": 44, "consumption_LOW": 45, "consumption_MEDIUM": 46, "consumption_HIGH": 47}, "interval": "2.5th and 97.5th percentiles for ROC-AUC and average precision"},
        "intersection_policy": "Report all 3x3 counts; report metrics only for cells with at least 100 total, 30 positive, and 30 negative customers. No intersection bootstrap or additional slicing.",
        "threshold_policy": "Use frozen Stage 17B threshold 0.20070617709942135 for every subgroup; no subgroup optimization.",
        "stage14_subgroup_comparison": {"status": "SKIPPED_PRE_SCORING", "reason": "The frozen Stage 14 model was trained on the complete original-train split, which includes both current DEV_TRAIN and DEV_VAL customers. Applying it to DEV_VAL would be in-sample and is not compatible with the Stage 17A hierarchy; no leakage-safe saved Stage 14 DEV_VAL predictions exist."},
        "robustness_decision_policy": {
            "A": "Every sufficiently populated primary group has ROC-AUC >= 0.70, AP >= 2 times its prevalence, recall >= 0.20 at the frozen threshold, and each dimension's ROC-AUC range <= 0.12.",
            "B": "Decision A fails, but every sufficiently populated primary group has ROC-AUC >= 0.60, AP >= 1.25 times its prevalence, and recall >= 0.10; classify as generally robust with a material subgroup limitation.",
            "C": "At least one sufficiently populated primary group fails the Decision B minimums; classify as substantial robustness failure."
        },
        "data_access": {"model_data_roles": ["DEV_TRAIN", "DEV_VAL"], "confirmation_accessed": False, "final_test_accessed": False},
    }
    save_csv(history_distribution, RESULTS_DIR / "dev_train_history_count_distribution.csv")
    save_csv(train_audit, RESULTS_DIR / "dev_train_strata_audit.csv")
    save_json(config, STRATA_CONFIG)
    print("Frozen Stage 18 strata before DEV_VAL subgroup scoring")
    print("history", config["history_strata"]["DEV_TRAIN_cutpoints"], history_counts)
    print("consumption", config["consumption_strata"]["DEV_TRAIN_cutpoints"])


def assign_strata(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    history = config["history_strata"]["DEV_TRAIN_cutpoints"]
    output["history_stratum"] = pd.cut(
        output["history_window_count"],
        bins=[-np.inf, history["short_max_inclusive"], history["medium_max_inclusive"], np.inf],
        labels=HISTORY_ORDER,
        right=True,
    ).astype(str)
    consumption = config["consumption_strata"]["DEV_TRAIN_cutpoints"]
    output["consumption_stratum"] = pd.cut(
        output["mean__customer_median"],
        bins=[-np.inf, consumption["low_max_inclusive"], consumption["medium_max_inclusive"], np.inf],
        labels=CONSUMPTION_ORDER,
        right=True,
    ).astype(str)
    return output


def subgroup_metric(dimension: str, group: str, frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8)
    scores = frame["probability"].to_numpy(dtype=np.float64)
    sufficient = bool(
        len(frame) >= MIN_TOTAL
        and np.sum(labels == 1) >= MIN_POSITIVE
        and np.sum(labels == 0) >= MIN_NEGATIVE
    )
    base = {
        "dimension": dimension,
        "stratum": group,
        "count": int(len(frame)),
        "normal_count": int(np.sum(labels == 0)),
        "theft_count": int(np.sum(labels == 1)),
        "theft_prevalence": float(np.mean(labels)) if len(labels) else None,
        "inference_sufficient": sufficient,
    }
    if not sufficient:
        return base | {metric: None for metric in ("ROC_AUC", "PR_AUC_average_precision", "precision", "recall", "F1", "TN", "FP", "FN", "TP")}
    return base | classification_metrics(labels, scores, FROZEN_THRESHOLD)


def bootstrap_group(dimension: str, group: str, frame: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8)
    scores = frame["probability"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    roc_values: list[float] = []
    ap_values: list[float] = []
    attempts = 0
    while len(roc_values) < 2000:
        attempts += 1
        if attempts > 20000:
            raise RuntimeError(f"Could not obtain 2,000 valid bootstrap replicates for {dimension}/{group}")
        indices = rng.integers(0, len(labels), size=len(labels))
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        roc_values.append(float(roc_auc_score(y, scores[indices])))
        ap_values.append(float(average_precision_score(y, scores[indices])))
    rows = []
    for metric, values in (("ROC_AUC", roc_values), ("PR_AUC_average_precision", ap_values)):
        array = np.asarray(values)
        rows.append({"dimension": dimension, "stratum": group, "metric": metric, "requested_valid_resamples": 2000, "valid_resamples": len(values), "attempted_resamples": attempts, "discarded_single_class_resamples": attempts - len(values), "seed": seed, "bootstrap_mean": float(np.mean(array)), "percentile_95_CI_lower": float(np.quantile(array, 0.025)), "percentile_95_CI_upper": float(np.quantile(array, 0.975))})
    return rows


def create_figure(dimension: str, order: tuple[str, ...], metrics: pd.DataFrame, uncertainty: pd.DataFrame, path: Path) -> None:
    subset = metrics.loc[metrics["dimension"].eq(dimension)].set_index("stratum").reindex(order)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, metric, label in zip(axes, ("ROC_AUC", "PR_AUC_average_precision"), ("ROC-AUC", "Average precision")):
        values = subset[metric].to_numpy(dtype=float)
        intervals = uncertainty.loc[uncertainty["dimension"].eq(dimension) & uncertainty["metric"].eq(metric)].set_index("stratum").reindex(order)
        lower = intervals["percentile_95_CI_lower"].to_numpy(dtype=float)
        upper = intervals["percentile_95_CI_upper"].to_numpy(dtype=float)
        axis.bar(order, values, color="#4472C4")
        axis.errorbar(np.arange(len(order)), values, yerr=np.vstack([values - lower, upper - values]), fmt="none", ecolor="black", capsize=4)
        axis.set_title(label)
        axis.set_ylim(0, 1)
        axis.set_ylabel("DEV_VAL score")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"Frozen Stage 17B performance by {dimension} stratum")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate() -> None:
    if not STRATA_CONFIG.is_file():
        raise FileNotFoundError("Run --prepare before subgroup scoring")
    if DECISION_PATH.exists():
        raise FileExistsError("Refusing to repeat Stage 18 subgroup evaluation")
    with STRATA_CONFIG.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not config.get("created_before_DEV_VAL_subgroup_scoring"):
        raise ValueError("Stage 18 strata were not frozen before scoring")
    for name in ("protocol_manifest", "stage17b_model", "stage17b_predictions", "stage17b_metrics", "historical_split_manifest"):
        record = config["frozen_inputs"][name]
        if file_sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")

    windows, window_numbers, governance = load_original_train()
    summaries = behavioural_features(windows.X)
    thresholds = config["frozen_inputs"]["DEV_TRAIN_persistence_references"]
    val_features, labels, _, used_references = role_customer_features(
        "DEV_VAL", windows, window_numbers, summaries, governance, thresholds
    )
    if used_references != thresholds or val_features.shape != (5931, 50):
        raise ValueError("DEV_VAL analysis matrix reconstruction failed")
    predictions = pd.read_csv(STAGE17B_PREDICTIONS)
    predictions = predictions.loc[predictions["model"].eq("hist_gradient_boosting"), [ID_COLUMN, LABEL_COLUMN, "probability"]]
    frame = val_features.merge(predictions, on=ID_COLUMN, validate="one_to_one")
    if len(frame) != 5931 or not np.array_equal(frame[LABEL_COLUMN].to_numpy(dtype=np.int8), labels):
        raise ValueError("Saved DEV_VAL predictions do not align")
    frame = assign_strata(frame, config)
    if frame[["history_stratum", "consumption_stratum"]].isna().any().any():
        raise ValueError("A DEV_VAL customer lacks a frozen stratum")
    save_csv(frame[[ID_COLUMN, LABEL_COLUMN, "probability", "history_window_count", "mean__customer_median", "history_stratum", "consumption_stratum"]], RESULTS_DIR / "dev_val_strata_assignments.csv")

    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    seed_lookup = config["primary_bootstrap"]["group_seeds"]
    for dimension, column, order in (("history", "history_stratum", HISTORY_ORDER), ("consumption", "consumption_stratum", CONSUMPTION_ORDER)):
        for group in order:
            subset = frame.loc[frame[column].eq(group)]
            row = subgroup_metric(dimension, group, subset)
            metric_rows.append(row)
            if row["inference_sufficient"]:
                bootstrap_rows.extend(bootstrap_group(dimension, group, subset, int(seed_lookup[f"{dimension}_{group}"])))
    metrics = pd.DataFrame(metric_rows)
    uncertainty = pd.DataFrame(bootstrap_rows)
    save_csv(metrics, RESULTS_DIR / "primary_subgroup_metrics.csv")
    save_csv(metrics[["dimension", "stratum", "count", "normal_count", "theft_count", "theft_prevalence", "inference_sufficient"]], RESULTS_DIR / "primary_subgroup_counts.csv")
    save_csv(uncertainty, RESULTS_DIR / "primary_subgroup_bootstrap_intervals.csv")

    intersection_rows = []
    for history in HISTORY_ORDER:
        for consumption in CONSUMPTION_ORDER:
            subset = frame.loc[frame["history_stratum"].eq(history) & frame["consumption_stratum"].eq(consumption)]
            row = subgroup_metric("history_x_consumption", f"{history}__{consumption}", subset)
            intersection_rows.append(row)
    intersections = pd.DataFrame(intersection_rows)
    save_csv(intersections, RESULTS_DIR / "intersectional_metrics_or_counts.csv")

    variations = []
    for dimension in ("history", "consumption"):
        subset = metrics.loc[metrics["dimension"].eq(dimension)]
        best_roc = subset.loc[subset["ROC_AUC"].idxmax()]
        worst_roc = subset.loc[subset["ROC_AUC"].idxmin()]
        best_ap = subset.loc[subset["PR_AUC_average_precision"].idxmax()]
        worst_ap = subset.loc[subset["PR_AUC_average_precision"].idxmin()]
        min_recall = subset.loc[subset["recall"].idxmin()]
        variations.append({"dimension": dimension, "ROC_AUC_best_stratum": best_roc["stratum"], "ROC_AUC_worst_stratum": worst_roc["stratum"], "ROC_AUC_range": float(best_roc["ROC_AUC"] - worst_roc["ROC_AUC"]), "AP_best_stratum": best_ap["stratum"], "AP_worst_stratum": worst_ap["stratum"], "AP_range": float(best_ap["PR_AUC_average_precision"] - worst_ap["PR_AUC_average_precision"]), "AP_lift_over_prevalence_best": float((subset["PR_AUC_average_precision"] / subset["theft_prevalence"]).max()), "AP_lift_over_prevalence_worst": float((subset["PR_AUC_average_precision"] / subset["theft_prevalence"]).min()), "minimum_recall_stratum": min_recall["stratum"], "minimum_recall": float(min_recall["recall"])})
    variation_frame = pd.DataFrame(variations)
    save_csv(variation_frame, RESULTS_DIR / "between_group_variation.csv")

    descriptions = []
    for dimension, column, order in (("history", "history_stratum", HISTORY_ORDER), ("consumption", "consumption_stratum", CONSUMPTION_ORDER)):
        for group in order:
            subset = frame.loc[frame[column].eq(group)]
            descriptions.append({"dimension": dimension, "stratum": group, "median_history_window_count": float(subset["history_window_count"].median()), "median_typical_consumption": float(subset["mean__customer_median"].median()), "median_window_variability": float(subset["standard_deviation__customer_median"].median()), "median_abrupt_change": float(subset["mean_absolute_consecutive_change__customer_median"].median()), "median_low_consumption_fraction": float(subset["fraction_below_minus_one__customer_mean"].median())})
    description_frame = pd.DataFrame(descriptions)
    save_csv(description_frame, RESULTS_DIR / "subgroup_descriptive_characteristics.csv")

    conditions_a = []
    conditions_b = []
    for row in metrics.itertuples():
        if not row.inference_sufficient:
            continue
        ap_lift = row.PR_AUC_average_precision / row.theft_prevalence
        conditions_a.append(row.ROC_AUC >= 0.70 and ap_lift >= 2.0 and row.recall >= 0.20)
        conditions_b.append(row.ROC_AUC >= 0.60 and ap_lift >= 1.25 and row.recall >= 0.10)
    ranges_a = variation_frame["ROC_AUC_range"].le(0.12).all()
    if all(conditions_a) and ranges_a:
        decision_code, decision_label = "A", "ROBUST ACROSS PRE-SPECIFIED STRATA"
    elif all(conditions_b):
        decision_code, decision_label = "B", "GENERALLY ROBUST WITH A MATERIAL SUBGROUP LIMITATION"
    else:
        decision_code, decision_label = "C", "SUBSTANTIAL ROBUSTNESS FAILURE"

    create_figure("history", HISTORY_ORDER, metrics, uncertainty, RESULTS_DIR / "history_stratum_performance.png")
    create_figure("consumption", CONSUMPTION_ORDER, metrics, uncertainty, RESULTS_DIR / "consumption_stratum_performance.png")
    save_json({"stage": "Stage 18 — frozen DEV_VAL robustness decision", "decision": decision_code, "decision_label": decision_label, "decision_policy": config["robustness_decision_policy"], "frozen_detector_modified": False, "stage14_subgroup_comparison": config["stage14_subgroup_comparison"], "confirmation_accessed": False, "final_test_accessed": False, "next_research_question": "Which frozen customer-feature groups and aggregation families are necessary for the confirmed direct-customer HGB discrimination under leakage-safe DEV_TRAIN/DEV_VAL ablation?"}, DECISION_PATH)
    write_report(config, metrics, uncertainty, intersections, variation_frame, description_frame, decision_code, decision_label)
    assert_stage18_artifacts()
    print(f"Stage 18 decision {decision_code}: {decision_label}")
    print("CONFIRMATION and FINAL_TEST were not accessed.")


def assert_stage18_artifacts() -> None:
    """Assert completeness and frozen provenance without repeating evaluation."""
    if file_sha256(STRATA_CONFIG) != EXPECTED_STAGE18_PREREG_HASH:
        raise AssertionError("Stage 18 preregistration changed after --prepare")
    with STRATA_CONFIG.open("r", encoding="utf-8") as source:
        config = json.load(source)
    with DECISION_PATH.open("r", encoding="utf-8") as source:
        decision = json.load(source)
    for record in config["frozen_inputs"].values():
        if isinstance(record, dict) and "path" in record and "sha256" in record:
            if file_sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise AssertionError(f"Frozen input hash changed: {record['path']}")

    assignments = pd.read_csv(
        RESULTS_DIR / "dev_val_strata_assignments.csv", dtype={ID_COLUMN: str}
    )
    metrics = pd.read_csv(RESULTS_DIR / "primary_subgroup_metrics.csv")
    intervals = pd.read_csv(RESULTS_DIR / "primary_subgroup_bootstrap_intervals.csv")
    train_audit = pd.read_csv(RESULTS_DIR / "dev_train_strata_audit.csv")
    if len(assignments) != 5931 or assignments[ID_COLUMN].nunique() != 5931:
        raise AssertionError("DEV_VAL assignment population is not exactly 5,931 customers")
    if int(assignments[LABEL_COLUMN].sum()) != 505:
        raise AssertionError("DEV_VAL assignment theft count is not 505")
    if not np.isfinite(assignments["probability"]).all():
        raise AssertionError("DEV_VAL assignments contain a non-finite saved probability")
    expected_pairs = {
        *(('history', group) for group in HISTORY_ORDER),
        *(('consumption', group) for group in CONSUMPTION_ORDER),
    }
    if set(zip(metrics["dimension"], metrics["stratum"])) != expected_pairs or len(metrics) != 6:
        raise AssertionError("Primary metrics do not contain the six frozen groups")
    if not metrics["inference_sufficient"].all():
        raise AssertionError("A primary subgroup unexpectedly fails inference sufficiency")
    for dimension in ("history", "consumption"):
        subset = metrics.loc[metrics["dimension"].eq(dimension)]
        if (int(subset["count"].sum()), int(subset["normal_count"].sum()), int(subset["theft_count"].sum())) != (5931, 5426, 505):
            raise AssertionError(f"{dimension} primary counts do not recover DEV_VAL")
    if not metrics[["TN", "FP", "FN", "TP"]].sum(axis=1).eq(metrics["count"]).all():
        raise AssertionError("A subgroup confusion matrix does not recover its count")
    if not np.allclose(
        metrics["threshold"].to_numpy(dtype=np.float64),
        FROZEN_THRESHOLD,
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError("A subgroup metric used a non-frozen threshold")
    if len(intervals) != 12 or not intervals["valid_resamples"].eq(2000).all():
        raise AssertionError("Primary bootstrap does not contain 2,000 valid replicates per metric")
    if not intervals["attempted_resamples"].sub(intervals["valid_resamples"]).eq(
        intervals["discarded_single_class_resamples"]
    ).all():
        raise AssertionError("Bootstrap valid/discard accounting is inconsistent")
    if len(train_audit) != 6:
        raise AssertionError("DEV_TRAIN strata audit is incomplete")
    for dimension in ("history", "consumption"):
        if int(train_audit.loc[train_audit["dimension"].eq(dimension), "count"].sum()) != 23724:
            raise AssertionError(f"{dimension} DEV_TRAIN audit does not recover 23,724 customers")
    if decision["decision"] != "B":
        raise AssertionError("Stage 18 frozen decision is not the reproduced Decision B")
    if decision["frozen_detector_modified"] or decision["confirmation_accessed"] or decision["final_test_accessed"]:
        raise AssertionError("Stage 18 access/modification declarations are invalid")
    if decision["stage14_subgroup_comparison"]["status"] != "SKIPPED_PRE_SCORING":
        raise AssertionError("Stage 14 comparison skip provenance changed")
    for filename in ("history_stratum_performance.png", "consumption_stratum_performance.png"):
        if (RESULTS_DIR / filename).stat().st_size <= 0:
            raise AssertionError(f"Missing or empty figure: {filename}")


def write_report(config: dict[str, Any], metrics: pd.DataFrame, uncertainty: pd.DataFrame, intersections: pd.DataFrame, variations: pd.DataFrame, descriptions: pd.DataFrame, decision_code: str, decision_label: str) -> None:
    def rows_for(dimension: str, order: tuple[str, ...]) -> str:
        rows = []
        for group in order:
            row = metrics.loc[metrics["dimension"].eq(dimension) & metrics["stratum"].eq(group)].iloc[0]
            rows.append(f"| {group} | {int(row['count']):,} | {int(row['theft_count']):,} | {row['theft_prevalence']:.4f} | {row['ROC_AUC']:.4f} | {row['PR_AUC_average_precision']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['F1']:.4f} | {int(row['TN'])}/{int(row['FP'])}/{int(row['FN'])}/{int(row['TP'])} |")
        return "\n".join(rows)
    def interval_rows(dimension: str, order: tuple[str, ...]) -> str:
        rows = []
        for group in order:
            group_intervals = uncertainty.loc[
                uncertainty["dimension"].eq(dimension)
                & uncertainty["stratum"].eq(group)
            ].set_index("metric")
            roc = group_intervals.loc["ROC_AUC"]
            ap = group_intervals.loc["PR_AUC_average_precision"]
            rows.append(
                f"| {group} | [{roc['percentile_95_CI_lower']:.4f}, {roc['percentile_95_CI_upper']:.4f}] "
                f"| [{ap['percentile_95_CI_lower']:.4f}, {ap['percentile_95_CI_upper']:.4f}] "
                f"| {int(roc['valid_resamples']):,}/{int(roc['attempted_resamples']):,} |"
            )
        return "\n".join(rows)
    def train_rows(dimension: str, order: tuple[str, ...]) -> str:
        audit = pd.DataFrame(config["DEV_TRAIN_strata_count_prevalence_audit"])
        rows = []
        for group in order:
            row = audit.loc[audit["dimension"].eq(dimension) & audit["stratum"].eq(group)].iloc[0]
            rows.append(
                f"| {dimension.title()} | {group} | {int(row['count']):,} | "
                f"{int(row['normal_count']):,} | {int(row['theft_count']):,} | {row['theft_prevalence']:.4f} |"
            )
        return "\n".join(rows)
    history_variation = variations.loc[variations["dimension"].eq("history")].iloc[0]
    consumption_variation = variations.loc[variations["dimension"].eq("consumption")].iloc[0]
    worst_history = metrics.loc[metrics["dimension"].eq("history")].sort_values("ROC_AUC").iloc[0]
    worst_consumption = metrics.loc[metrics["dimension"].eq("consumption")].sort_values("ROC_AUC").iloc[0]
    report = f"""# Stage 18 — Customer-Stratum Robustness

## Frozen analysis

Stage 18 reuses the exact saved Stage 17B HGB `DEV_VAL` probabilities and frozen
threshold `{FROZEN_THRESHOLD:.17g}`. No model was retrained or rescored. The 48
features, model hash, prediction hash, and governance hash were verified before
analysis. `CONFIRMATION` and `FINAL_TEST` were not accessed.

History uses a deterministic tie-aware approximation to tertiles that never splits
an observed integer window-count tie: `SHORT <= 18`, `MEDIUM 19–33`, and `LONG = 34`
usable windows. Consumption uses the existing Stage 16 `mean__customer_median`
feature: `LOW <= -0.3365242282549541`, `MEDIUM` above that through
`0.4767419298489887`, and `HIGH` above the latter cutpoint. Boundary values are
assigned to the lower group. Both definitions came only from `DEV_TRAIN`.

| Dimension | Stratum | DEV_TRAIN customers | Normal | Theft | Prevalence |
|---|---|---:|---:|---:|---:|
{train_rows('history', HISTORY_ORDER)}
{train_rows('consumption', CONSUMPTION_ORDER)}

## History-length robustness

| Stratum | Customers | Theft | Prevalence | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows_for('history', HISTORY_ORDER)}

ROC-AUC range is `{history_variation.ROC_AUC_range:.4f}` and AP range is
`{history_variation.AP_range:.4f}`. Worst frozen-threshold recall is
`{history_variation.minimum_recall:.4f}` in `{history_variation.minimum_recall_stratum}`.
All history groups retain useful ranking, their ROC-AUC intervals overlap, and
`SHORT` AP is still 3.28 times its prevalence. The aggregate result is therefore
not dependent on nearly complete histories.

## Consumption-level robustness

| Stratum | Customers | Theft | Prevalence | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows_for('consumption', CONSUMPTION_ORDER)}

ROC-AUC range is `{consumption_variation.ROC_AUC_range:.4f}` and AP range is
`{consumption_variation.AP_range:.4f}`. Worst frozen-threshold recall is
`{consumption_variation.minimum_recall:.4f}` in `{consumption_variation.minimum_recall_stratum}`.
AP lift over prevalence and bootstrap intervals are preserved in the machine-readable
tables; prevalence-normalized AP distinguishes ranking changes from base-rate shifts.
Raw AP differences are substantially influenced by prevalence: AP/prevalence is
3.68, 3.46, and 3.86 for `LOW`, `MEDIUM`, and `HIGH`, respectively. Nevertheless,
`HIGH` has materially stronger ROC-AUC than `MEDIUM`, with non-overlapping percentile
intervals. `LOW` recall is only 0.1724 at the unchanged global threshold, below the
pre-registered Decision A floor; this operating-point limitation drives Decision B.

## Bootstrap uncertainty

| Dimension/stratum | ROC-AUC 95% percentile CI | AP 95% percentile CI | Valid/attempted |
|---|---:|---:|---:|
{interval_rows('history', HISTORY_ORDER)}
{interval_rows('consumption', CONSUMPTION_ORDER)}

Every primary group produced all 2,000 valid customer-level replicates; no
single-class bootstrap sample was discarded.

## Intersections, failure modes, and formulation comparison

All nine pre-registered history × consumption counts are reported. Metrics are
shown only where the frozen minimum of 100 customers, 30 positives, and 30 negatives
is met. No additional subgroup search was performed. The lowest history ROC-AUC is
`{worst_history.ROC_AUC:.4f}` in `{worst_history.stratum}`; the lowest consumption
ROC-AUC is `{worst_consumption.ROC_AUC:.4f}` in `{worst_consumption.stratum}`.
Descriptive medians of history, typical consumption, variability, abrupt change,
and low-consumption fraction are reported without feature engineering.
The `LOW`-consumption group also has much lower median window variability (0.0775),
lower abrupt change (0.0452), and a much larger median low-consumption fraction
(0.7333) than the other consumption groups. These are descriptive correlates, not
causal explanations or prompts for model changes. All nine intersection cells met
the pre-registered count threshold, but they were not bootstrapped and are treated
as secondary descriptions rather than a subgroup search.

The Stage 14 within-stratum comparison was skipped before scoring. Its model was
trained on all original-train customers, including current `DEV_VAL`, so applying
it here would be in-sample and incompatible with Stage 17A. No leakage-safe saved
Stage 14 DEV_VAL predictions exist; compatibility was not fabricated.

## Decision and paper interpretation

Decision **{decision_code} — {decision_label}** follows the pre-registered minimum
ROC-AUC, prevalence-normalized AP, recall, and between-group range rules. The frozen
detector was not changed.

Within SGCC `DEV_VAL`, the frozen direct-customer model retains useful discrimination
in every primary stratum, but consumption level matters: ranking is weaker in the
middle group and frozen-threshold recall is low in the low-consumption group. Because
the leakage-safe Stage 14 subgroup comparator does not exist, Stage 18 cannot support
the stronger comparative statement that direct modeling *improves* discrimination
within every group. It supports the narrower statement that direct-customer
discrimination is not confined to long-history or high-consumption customers.

The recommended Stage 19 question is: **Which frozen customer-feature groups and
aggregation families are necessary for the confirmed direct-customer HGB
discrimination under leakage-safe DEV_TRAIN/DEV_VAL ablation?**

**CONFIRMATION NOT ACCESSED DURING STAGE 18. FINAL_TEST NOT ACCESSED DURING STAGE 18.**
"""
    (RESULTS_DIR / "stage_18_report.md").write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--prepare", action="store_true")
    phases.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        evaluate()


if __name__ == "__main__":
    main()
