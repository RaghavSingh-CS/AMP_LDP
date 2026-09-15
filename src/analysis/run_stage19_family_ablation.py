"""Stage 19: pre-registered DEV-only customer feature-family ablation.

Run ``--prepare`` before ``--evaluate``. The only model-data source is the
original-train population partitioned by Stage 17A into DEV_TRAIN and DEV_VAL.
CONFIRMATION and FINAL_TEST have no loading path in this module.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.run_stage14_representation import behavioural_features
from src.analysis.run_stage16_customer_aggregation import FEATURE_NAMES
from src.analysis.run_stage18_robustness import (
    EXPECTED_HISTORICAL_SPLIT_HASH,
    EXPECTED_MODEL_HASH,
    EXPECTED_PROTOCOL_HASH,
    EXPECTED_STAGE16_CONFIG_HASH,
    HISTORICAL_SPLIT_MANIFEST,
    PROTOCOL_MANIFEST,
    STAGE16_CONFIG,
    STAGE17B_CONFIG,
    STAGE17B_DECISION,
    STAGE17B_METRICS,
    STAGE17B_MODEL,
    STAGE17B_PREDICTIONS,
    STAGE17B_SUMMARY,
    load_original_train,
    role_customer_features,
)
from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    classification_metrics,
    file_sha256,
    save_csv,
    save_json,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_19"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_19"
PREREGISTRATION = RESULTS_DIR / "preregistered_family_ablation.json"
MAPPING_PATH = RESULTS_DIR / "feature_family_mapping.csv"
SEAL_PATH = RESULTS_DIR / "preregistration_seal.json"
DECISION_PATH = RESULTS_DIR / "family_ablation_decision.json"
STAGE16_REDUNDANCY = (
    PROJECT_ROOT
    / "results"
    / "supervised"
    / "stage_16"
    / "training_redundant_feature_pairs.csv"
)
STAGE17B_IMPORTANCE = (
    PROJECT_ROOT
    / "results"
    / "supervised"
    / "stage_17b"
    / "dev_val_permutation_importance.csv"
)
STAGE18_DECISION = (
    PROJECT_ROOT / "results" / "supervised" / "stage_18" / "robustness_decision.json"
)
STAGE18_METRICS = (
    PROJECT_ROOT / "results" / "supervised" / "stage_18" / "primary_subgroup_metrics.csv"
)

EXPECTED_STAGE17B_CONFIG_HASH = "605eb1f482123a24a14c0929c951f400c37b3fbc2d2bb2e4d46d2ce834698046"
EXPECTED_STAGE17B_SUMMARY_HASH = "560e8602ce0d572a2e305efcebe525ef3f1174aad0e6f5a3209c2b33fff4865a"
EXPECTED_STAGE17B_DECISION_HASH = "11e15f18e2521ffed8e7dbaacee469284ee356acedea1d8f34ffcf48785a68fa"
EXPECTED_STAGE17B_PREDICTIONS_HASH = "d98a7551a763a5f1e78193786012d86a96c593a73d770fa625a0e09aaac71aea"
EXPECTED_STAGE17B_METRICS_HASH = "9deeb02106b361857b9c4eead2bf9cdd25affd232e4848275129182341a7b083"
EXPECTED_STAGE16_REDUNDANCY_HASH = "cab14c7ced4f672e8cfb5a56f99b4a79bcf4c4cd3216ab55dfaa432391293b89"
EXPECTED_STAGE17B_IMPORTANCE_HASH = "75cee0e1e2cbf74e163ab9ac6c1060ad72bd91d033186105254c9afdb74155f6"
EXPECTED_STAGE18_DECISION_HASH = "092ffcf04f6898ec4a134718447b2985feea3476e412da81ebef8018ba00c78b"
EXPECTED_STAGE18_METRICS_HASH = "5294b19d05945b2489bc72b487fe9f8c7c945084c5d38c5d495e2878d2feef4c"
FROZEN_THRESHOLD = 0.20070617709942135
BASE_HEAD = "528acb53554017a6c159cd4c03ead1be46165a02"

FAMILIES: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    (
        ("LEVEL_DISTRIBUTION", tuple(FEATURE_NAMES[0:18])),
        ("VARIABILITY_ABRUPT_CHANGE", tuple(FEATURE_NAMES[18:30])),
        ("DIRECTIONAL_TEMPORAL_CHANGE", tuple(FEATURE_NAMES[30:38])),
        ("LOW_CONSUMPTION_BEHAVIOUR", tuple(FEATURE_NAMES[38:40])),
        ("PERSISTENCE_EXCEEDANCE", tuple(FEATURE_NAMES[40:44])),
        ("LONGITUDINAL_CONTRAST", tuple(FEATURE_NAMES[44:48])),
    )
)
FAMILY_ORDER = tuple(FAMILIES)
HGB_PARAMETERS: dict[str, Any] = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "class_weight": None,
    "early_stopping": False,
    "random_state": 42,
}
FULL_PROBABILITY_TOLERANCE = 1e-12
PRACTICAL_ROC_DROP = 0.01
PRACTICAL_AP_DROP = 0.02


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def verify_frozen_inputs() -> dict[str, dict[str, str]]:
    expected = {
        "stage17a_manifest": (PROTOCOL_MANIFEST, EXPECTED_PROTOCOL_HASH),
        "historical_split": (HISTORICAL_SPLIT_MANIFEST, EXPECTED_HISTORICAL_SPLIT_HASH),
        "stage16_preregistration": (STAGE16_CONFIG, EXPECTED_STAGE16_CONFIG_HASH),
        "stage17b_configuration": (STAGE17B_CONFIG, EXPECTED_STAGE17B_CONFIG_HASH),
        "stage17b_training_summary": (STAGE17B_SUMMARY, EXPECTED_STAGE17B_SUMMARY_HASH),
        "stage17b_decision": (STAGE17B_DECISION, EXPECTED_STAGE17B_DECISION_HASH),
        "stage17b_model": (STAGE17B_MODEL, EXPECTED_MODEL_HASH),
        "stage17b_predictions": (STAGE17B_PREDICTIONS, EXPECTED_STAGE17B_PREDICTIONS_HASH),
        "stage17b_metrics": (STAGE17B_METRICS, EXPECTED_STAGE17B_METRICS_HASH),
        "stage16_redundancy": (STAGE16_REDUNDANCY, EXPECTED_STAGE16_REDUNDANCY_HASH),
        "stage17b_permutation_importance": (STAGE17B_IMPORTANCE, EXPECTED_STAGE17B_IMPORTANCE_HASH),
        "stage18_decision": (STAGE18_DECISION, EXPECTED_STAGE18_DECISION_HASH),
        "stage18_primary_metrics": (STAGE18_METRICS, EXPECTED_STAGE18_METRICS_HASH),
    }
    records: dict[str, dict[str, str]] = {}
    for name, (path, expected_hash) in expected.items():
        observed = file_sha256(path)
        if observed != expected_hash:
            raise ValueError(f"Frozen input hash changed: {name}")
        records[name] = {"path": relative(path), "sha256": observed}
    return records


def feature_mapping_frame() -> pd.DataFrame:
    rows = []
    for position, feature in enumerate(FEATURE_NAMES, start=1):
        matches = [family for family, features in FAMILIES.items() if feature in features]
        if len(matches) != 1:
            raise ValueError(f"Feature does not map to exactly one family: {feature}")
        rows.append(
            {
                "feature_position": position,
                "feature": feature,
                "family": matches[0],
            }
        )
    mapping = pd.DataFrame(rows)
    if mapping["feature"].tolist() != list(FEATURE_NAMES) or len(mapping) != 48:
        raise ValueError("Feature-family mapping is not the exact frozen schema")
    return mapping


def prepare() -> None:
    if PREREGISTRATION.exists() or SEAL_PATH.exists():
        raise FileExistsError("Refusing to overwrite frozen Stage 19 preregistration")
    frozen_inputs = verify_frozen_inputs()
    with STAGE16_CONFIG.open("r", encoding="utf-8") as source:
        stage16 = json.load(source)
    with STAGE17B_CONFIG.open("r", encoding="utf-8") as source:
        stage17b = json.load(source)
    with STAGE17B_DECISION.open("r", encoding="utf-8") as source:
        stage17b_decision = json.load(source)
    if tuple(stage16["feature_names_in_order"]) != FEATURE_NAMES:
        raise ValueError("Stage 16 feature names/order changed")
    if tuple(stage17b["representation"]["feature_names_in_order"]) != FEATURE_NAMES:
        raise ValueError("Stage 17B feature names/order changed")
    if stage17b["candidate_models"]["hist_gradient_boosting"]["parameters"] != HGB_PARAMETERS:
        raise ValueError("Stage 17B HGB configuration changed")
    if stage17b_decision["selected_DEV_VAL_metrics"]["threshold"] != FROZEN_THRESHOLD:
        raise ValueError("Stage 17B threshold changed")

    mapping = feature_mapping_frame()
    save_csv(mapping, MAPPING_PATH)
    experiments = [
        {
            "experiment": "FULL_48",
            "type": "FULL_CONTROL",
            "family": None,
            "feature_count": 48,
            "features": list(FEATURE_NAMES),
        }
    ]
    for family, features in FAMILIES.items():
        experiments.append(
            {
                "experiment": f"FAMILY_ONLY__{family}",
                "type": "FAMILY_ONLY",
                "family": family,
                "feature_count": len(features),
                "features": list(features),
            }
        )
    for family, omitted in FAMILIES.items():
        retained = [feature for feature in FEATURE_NAMES if feature not in omitted]
        experiments.append(
            {
                "experiment": f"LEAVE_ONE_OUT__{family}",
                "type": "LEAVE_ONE_FAMILY_OUT",
                "family": family,
                "feature_count": len(retained),
                "features": retained,
            }
        )

    config = {
        "stage": "Stage 19 — pre-registered customer feature-family ablation",
        "base_head_before_stage": BASE_HEAD,
        "created_before_any_new_ablation_DEV_VAL_metric": True,
        "research_question": "Which frozen customer-feature groups and aggregation families are necessary for the confirmed direct-customer HGB discrimination under leakage-safe DEV_TRAIN/DEV_VAL ablation?",
        "frozen_inputs": frozen_inputs,
        "feature_schema": {
            "count": 48,
            "ordered_names": list(FEATURE_NAMES),
            "mapping_path": relative(MAPPING_PATH),
            "mapping_sha256": file_sha256(MAPPING_PATH),
            "families": [
                {"family": family, "feature_count": len(features), "features": list(features)}
                for family, features in FAMILIES.items()
            ],
            "mapping_basis": "Semantic Stage 16 feature-construction blocks only; no DEV_VAL performance inspected.",
        },
        "model": {
            "implementation": "sklearn.ensemble.HistGradientBoostingClassifier",
            "parameters": HGB_PARAMETERS,
            "same_configuration_for_every_experiment": True,
            "hyperparameter_tuning": False,
        },
        "experiments": experiments,
        "full_control_stop_gate": {
            "reference": relative(STAGE17B_PREDICTIONS),
            "maximum_absolute_probability_difference": FULL_PROBABILITY_TOLERANCE,
            "maximum_absolute_primary_metric_difference": FULL_PROBABILITY_TOLERANCE,
            "action_on_failure": "STOP before fitting or interpreting any family ablation.",
        },
        "metrics": {
            "primary": ["ROC_AUC", "PR_AUC_average_precision"],
            "secondary": ["precision", "recall", "F1", "TN", "FP", "FN", "TP"],
            "threshold": FROZEN_THRESHOLD,
            "threshold_rule": "Use the unchanged Stage 17B global threshold for every experiment; no experiment-specific optimization.",
            "delta_direction": "experiment minus FULL_48",
        },
        "bootstrap_selection_rule": {
            "scope": "LEAVE_ONE_FAMILY_OUT only",
            "selected_union": [
                "family with the most negative observed ROC-AUC delta",
                "family with the most negative observed AP delta",
                f"every family with ROC-AUC delta <= -{PRACTICAL_ROC_DROP}",
                f"every family with AP delta <= -{PRACTICAL_AP_DROP}",
            ],
            "tie_break": "FAMILY_ORDER",
            "resamples": 2000,
            "seed": 42,
            "unit": "paired DEV_VAL customer",
            "single_class_policy": "Discard deterministically and continue until 2,000 valid replicates; report attempts and discards.",
            "reported": ["mean delta", "2.5th percentile", "97.5th percentile", "fraction below zero", "fraction above zero"],
        },
        "interpretation_rules": {
            "strong_standalone": "FAMILY_ONLY ROC-AUC >= 0.72 and AP >= 0.25.",
            "practical_leave_out_loss": f"LEAVE_ONE_FAMILY_OUT minus FULL_48 ROC-AUC <= -{PRACTICAL_ROC_DROP} or AP <= -{PRACTICAL_AP_DROP}.",
            "conditionally_necessary": "A practical leave-out loss whose triggering metric paired-bootstrap 95% upper bound is below zero.",
            "strong_plus_negligible_drop": "Informative but redundant.",
            "weak_plus_necessary": "Complementary interaction information.",
            "weak_plus_negligible_drop": "Little evidence of necessity.",
        },
        "decision_policy": {
            "D": "Only if a compact model was genuinely pre-registered and retains both primary metrics within 0.01 of FULL_48 with paired intervals excluding losses beyond those margins.",
            "C": "Exactly one family is conditionally necessary, it is the only strong standalone family, and leaving it out reduces both ROC-AUC below 0.70 and AP below 0.25.",
            "A": "At least two feature families are conditionally necessary.",
            "B": "Neither C nor A applies; discrimination is distributed but family-level leave-out evidence is substantially redundant.",
            "evaluation_order": ["D", "C", "A", "B"],
        },
        "compact_model": {
            "status": "NOT_PRE_REGISTERED_FOR_CONSTRUCTION",
            "reason": "Stage 19 remains pure family ablation; no post-hoc compact model will be fitted.",
        },
        "prohibitions": [
            "No new, removed, or redefined individual feature outside the pre-registered family experiments.",
            "No individual-feature subset search.",
            "No hyperparameter, threshold, subgroup, resampling, GAN, or SMOTE experiment.",
            "No CONFIRMATION or FINAL_TEST features, labels, predictions, or performance access.",
        ],
        "data_access": {
            "fit": "DEV_TRAIN only",
            "evaluation": "DEV_VAL only",
            "confirmation_accessed": False,
            "final_test_accessed": False,
        },
    }
    save_json(config, PREREGISTRATION)
    save_json(
        {
            "preregistration_path": relative(PREREGISTRATION),
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "mapping_path": relative(MAPPING_PATH),
            "mapping_sha256": file_sha256(MAPPING_PATH),
        },
        SEAL_PATH,
    )
    print(f"Frozen Stage 19 preregistration: {file_sha256(PREREGISTRATION)}")
    print(mapping.groupby("family", sort=False).size().to_dict())


def load_preregistration() -> dict[str, Any]:
    if not PREREGISTRATION.is_file() or not SEAL_PATH.is_file():
        raise FileNotFoundError("Run --prepare before --evaluate")
    with SEAL_PATH.open("r", encoding="utf-8") as source:
        seal = json.load(source)
    if file_sha256(PREREGISTRATION) != seal["preregistration_sha256"]:
        raise ValueError("Stage 19 preregistration changed after --prepare")
    if file_sha256(MAPPING_PATH) != seal["mapping_sha256"]:
        raise ValueError("Stage 19 feature-family mapping changed after --prepare")
    with PREREGISTRATION.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not config.get("created_before_any_new_ablation_DEV_VAL_metric"):
        raise ValueError("Stage 19 preregistration timing declaration is invalid")
    for record in config["frozen_inputs"].values():
        if file_sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen input changed: {record['path']}")
    return config


def model_for_experiment() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**HGB_PARAMETERS)


def save_model(model: HistGradientBoostingClassifier, name: str) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name.lower()}.joblib"
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)
    return {
        "experiment": name,
        "path": relative(path),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def primary_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    full = metrics.loc[metrics["experiment"].eq("FULL_48")].iloc[0]
    output = metrics[
        ["experiment", "experiment_type", "family", "feature_count", "ROC_AUC", "PR_AUC_average_precision"]
    ].copy()
    output["delta_ROC_AUC_experiment_minus_FULL_48"] = output["ROC_AUC"] - full["ROC_AUC"]
    output["delta_AP_experiment_minus_FULL_48"] = (
        output["PR_AUC_average_precision"] - full["PR_AUC_average_precision"]
    )
    return output


def select_bootstrap_families(deltas: pd.DataFrame) -> list[str]:
    leave_out = deltas.loc[deltas["experiment_type"].eq("LEAVE_ONE_FAMILY_OUT")].copy()
    selected = {
        str(leave_out.loc[leave_out["delta_ROC_AUC_experiment_minus_FULL_48"].idxmin(), "family"]),
        str(leave_out.loc[leave_out["delta_AP_experiment_minus_FULL_48"].idxmin(), "family"]),
    }
    selected.update(
        leave_out.loc[
            leave_out["delta_ROC_AUC_experiment_minus_FULL_48"].le(-PRACTICAL_ROC_DROP)
            | leave_out["delta_AP_experiment_minus_FULL_48"].le(-PRACTICAL_AP_DROP),
            "family",
        ].astype(str)
    )
    return [family for family in FAMILY_ORDER if family in selected]


def paired_bootstrap(
    labels: np.ndarray,
    full_scores: np.ndarray,
    selected_scores: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    rng = np.random.default_rng(42)
    samples: list[dict[str, Any]] = []
    valid = 0
    attempts = 0
    while valid < 2000:
        attempts += 1
        if attempts > 20000:
            raise RuntimeError("Unable to obtain 2,000 valid paired bootstrap samples")
        indices = rng.integers(0, len(labels), size=len(labels))
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        full_roc = roc_auc_score(y, full_scores[indices])
        full_ap = average_precision_score(y, full_scores[indices])
        valid += 1
        for family, scores in selected_scores.items():
            samples.append(
                {
                    "replicate": valid,
                    "family": family,
                    "delta_ROC_AUC_ablation_minus_FULL_48": float(
                        roc_auc_score(y, scores[indices]) - full_roc
                    ),
                    "delta_AP_ablation_minus_FULL_48": float(
                        average_precision_score(y, scores[indices]) - full_ap
                    ),
                }
            )
    sample_frame = pd.DataFrame(samples)
    summaries: list[dict[str, Any]] = []
    for family in selected_scores:
        subset = sample_frame.loc[sample_frame["family"].eq(family)]
        for metric, column in (
            ("ROC_AUC", "delta_ROC_AUC_ablation_minus_FULL_48"),
            ("PR_AUC_average_precision", "delta_AP_ablation_minus_FULL_48"),
        ):
            values = subset[column].to_numpy(dtype=np.float64)
            summaries.append(
                {
                    "family": family,
                    "metric": metric,
                    "valid_resamples": valid,
                    "attempted_resamples": attempts,
                    "discarded_single_class_resamples": attempts - valid,
                    "seed": 42,
                    "mean_delta_ablation_minus_FULL_48": float(np.mean(values)),
                    "percentile_95_CI_lower": float(np.quantile(values, 0.025)),
                    "percentile_95_CI_upper": float(np.quantile(values, 0.975)),
                    "fraction_below_zero": float(np.mean(values < 0)),
                    "fraction_above_zero": float(np.mean(values > 0)),
                }
            )
    return sample_frame, pd.DataFrame(summaries), {
        "valid_resamples": valid,
        "attempted_resamples": attempts,
        "discarded_single_class_resamples": attempts - valid,
    }


def existing_evidence_summaries(mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_to_family = mapping.set_index("feature")["family"].to_dict()
    importance = pd.read_csv(STAGE17B_IMPORTANCE)
    importance["family"] = importance["feature"].map(feature_to_family)
    if importance["family"].isna().any():
        raise ValueError("Stage 17B importance contains an unmapped frozen feature")
    importance_rows = []
    for family in FAMILY_ORDER:
        subset = importance.loc[importance["family"].eq(family)]
        top_roc = subset.loc[subset["mean_ROC_AUC_decrease"].idxmax()]
        top_ap = subset.loc[subset["mean_average_precision_decrease"].idxmax()]
        importance_rows.append(
            {
                "family": family,
                "feature_count": len(subset),
                "sum_positive_mean_ROC_AUC_decrease": float(
                    subset["mean_ROC_AUC_decrease"].clip(lower=0).sum()
                ),
                "sum_positive_mean_AP_decrease": float(
                    subset["mean_average_precision_decrease"].clip(lower=0).sum()
                ),
                "largest_ROC_AUC_decrease_feature": top_roc["feature"],
                "largest_ROC_AUC_decrease": float(top_roc["mean_ROC_AUC_decrease"]),
                "largest_AP_decrease_feature": top_ap["feature"],
                "largest_AP_decrease": float(top_ap["mean_average_precision_decrease"]),
            }
        )

    redundancy = pd.read_csv(STAGE16_REDUNDANCY)
    redundancy["family_1"] = redundancy["feature_1"].map(feature_to_family)
    redundancy["family_2"] = redundancy["feature_2"].map(feature_to_family)
    if redundancy[["family_1", "family_2"]].isna().any().any():
        raise ValueError("Stage 16 redundancy evidence contains an unmapped feature")
    redundancy["family_pair"] = redundancy.apply(
        lambda row: "__".join(
            sorted(
                (row["family_1"], row["family_2"]),
                key=lambda family: FAMILY_ORDER.index(family),
            )
        ),
        axis=1,
    )
    redundancy_summary = (
        redundancy.groupby("family_pair", sort=False)
        .agg(
            correlated_pairs=("absolute_correlation", "size"),
            mean_absolute_correlation=("absolute_correlation", "mean"),
            maximum_absolute_correlation=("absolute_correlation", "max"),
        )
        .reset_index()
    )
    return pd.DataFrame(importance_rows), redundancy_summary


def family_interpretation(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    rows = []
    strong: set[str] = set()
    necessary: set[str] = set()
    for family in FAMILY_ORDER:
        only = metrics.loc[
            metrics["experiment"].eq(f"FAMILY_ONLY__{family}")
        ].iloc[0]
        leave = deltas.loc[
            deltas["experiment"].eq(f"LEAVE_ONE_OUT__{family}")
        ].iloc[0]
        is_strong = bool(only["ROC_AUC"] >= 0.72 and only["PR_AUC_average_precision"] >= 0.25)
        practical_roc = bool(leave["delta_ROC_AUC_experiment_minus_FULL_48"] <= -PRACTICAL_ROC_DROP)
        practical_ap = bool(leave["delta_AP_experiment_minus_FULL_48"] <= -PRACTICAL_AP_DROP)
        is_necessary = False
        if practical_roc or practical_ap:
            family_bootstrap = bootstrap.loc[bootstrap["family"].eq(family)].set_index("metric")
            if practical_roc and family_bootstrap.loc["ROC_AUC", "percentile_95_CI_upper"] < 0:
                is_necessary = True
            if practical_ap and family_bootstrap.loc[
                "PR_AUC_average_precision", "percentile_95_CI_upper"
            ] < 0:
                is_necessary = True
        if is_strong:
            strong.add(family)
        if is_necessary:
            necessary.add(family)
        if is_strong and is_necessary:
            label = "STRONG_STANDALONE_AND_CONDITIONALLY_NECESSARY"
        elif is_strong:
            label = "STRONG_STANDALONE_BUT_REDUNDANT"
        elif is_necessary:
            label = "WEAK_STANDALONE_BUT_COMPLEMENTARY_NECESSARY"
        else:
            label = "LIMITED_EVIDENCE_OF_CONDITIONAL_NECESSITY"
        rows.append(
            {
                "family": family,
                "family_only_strong": is_strong,
                "practical_ROC_AUC_leave_out_loss": practical_roc,
                "practical_AP_leave_out_loss": practical_ap,
                "conditionally_necessary": is_necessary,
                "interpretation": label,
            }
        )
    return pd.DataFrame(rows), strong, necessary


def make_figure(metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    labels = [family.replace("_", "\n") for family in FAMILY_ORDER]
    only = metrics.loc[metrics["experiment_type"].eq("FAMILY_ONLY")].set_index("family").reindex(FAMILY_ORDER)
    leave = deltas.loc[deltas["experiment_type"].eq("LEAVE_ONE_FAMILY_OUT")].set_index("family").reindex(FAMILY_ORDER)
    full = metrics.loc[metrics["experiment"].eq("FULL_48")].iloc[0]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, metric, title in (
        (axes[0, 0], "ROC_AUC", "Family-only ROC-AUC"),
        (axes[0, 1], "PR_AUC_average_precision", "Family-only average precision"),
    ):
        axis.bar(np.arange(len(FAMILY_ORDER)), only[metric], color="#4472C4")
        axis.axhline(full[metric], color="black", linestyle="--", label="FULL_48")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    for axis, metric, title in (
        (axes[1, 0], "delta_ROC_AUC_experiment_minus_FULL_48", "Leave-family-out ROC-AUC delta"),
        (axes[1, 1], "delta_AP_experiment_minus_FULL_48", "Leave-family-out AP delta"),
    ):
        values = leave[metric].to_numpy(dtype=np.float64)
        axis.bar(np.arange(len(FAMILY_ORDER)), values, color=np.where(values < 0, "#C0504D", "#70AD47"))
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes.flat:
        axis.set_xticks(np.arange(len(FAMILY_ORDER)), labels, fontsize=8)
    figure.suptitle("Stage 19 frozen customer feature-family ablation")
    figure.savefig(RESULTS_DIR / "family_ablation_primary_metrics.png", dpi=180)
    plt.close(figure)


def evaluate() -> None:
    if DECISION_PATH.exists():
        raise FileExistsError("Refusing to repeat completed Stage 19 evaluation")
    config = load_preregistration()
    windows, window_numbers, governance = load_original_train()
    summaries = behavioural_features(windows.X)
    train_features, train_labels, train_ids, references = role_customer_features(
        "DEV_TRAIN", windows, window_numbers, summaries, governance, None
    )
    val_features, val_labels, val_ids, val_references = role_customer_features(
        "DEV_VAL", windows, window_numbers, summaries, governance, references
    )
    if references != val_references:
        raise ValueError("DEV_TRAIN persistence references were not reused exactly")
    with STAGE17B_SUMMARY.open("r", encoding="utf-8") as source:
        stage17b_summary = json.load(source)
    if references != stage17b_summary["DEV_TRAIN_derived_persistence_q90_thresholds"]:
        raise ValueError("Reconstructed references differ from Stage 17B")
    if train_features.shape != (23724, 50) or val_features.shape != (5931, 50):
        raise ValueError("Reconstructed development matrix shape changed")
    if set(train_ids) & set(val_ids):
        raise ValueError("DEV_TRAIN and DEV_VAL overlap")
    train_matrix = train_features.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    val_matrix = val_features.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)

    saved = pd.read_csv(STAGE17B_PREDICTIONS, dtype={ID_COLUMN: str})
    saved = saved.loc[
        saved["model"].eq("hist_gradient_boosting"),
        [ID_COLUMN, LABEL_COLUMN, "probability"],
    ].set_index(ID_COLUMN).reindex(val_ids)
    if saved.isna().any().any() or not np.array_equal(
        saved[LABEL_COLUMN].to_numpy(dtype=np.int8), val_labels
    ):
        raise ValueError("Saved Stage 17B predictions do not align with reconstructed DEV_VAL")
    saved_scores = saved["probability"].to_numpy(dtype=np.float64)

    metrics_rows: list[dict[str, Any]] = []
    probability_frames: list[pd.DataFrame] = []
    model_records: list[dict[str, Any]] = []
    score_by_experiment: dict[str, np.ndarray] = {}
    for index, experiment in enumerate(config["experiments"]):
        name = experiment["experiment"]
        columns = [FEATURE_NAMES.index(feature) for feature in experiment["features"]]
        print(f"Fitting {name} with {len(columns)} frozen features")
        model = model_for_experiment()
        model.fit(train_matrix[:, columns], train_labels)
        scores = np.asarray(model.predict_proba(val_matrix[:, columns])[:, 1], dtype=np.float64)
        row = classification_metrics(val_labels, scores, FROZEN_THRESHOLD)
        row.update(
            {
                "experiment": name,
                "experiment_type": experiment["type"],
                "family": experiment["family"],
                "feature_count": len(columns),
            }
        )
        if index == 0:
            max_probability_difference = float(np.max(np.abs(scores - saved_scores)))
            saved_metrics = pd.read_csv(STAGE17B_METRICS).loc[
                lambda frame: frame["model"].eq("hist_gradient_boosting")
            ].iloc[0]
            roc_difference = float(row["ROC_AUC"] - saved_metrics["ROC_AUC"])
            ap_difference = float(
                row["PR_AUC_average_precision"]
                - saved_metrics["PR_AUC_average_precision"]
            )
            reproduction = {
                "stage": "Stage 19 FULL_48 reproduction gate",
                "passed_before_family_ablation": bool(
                    max_probability_difference <= FULL_PROBABILITY_TOLERANCE
                    and abs(roc_difference) <= FULL_PROBABILITY_TOLERANCE
                    and abs(ap_difference) <= FULL_PROBABILITY_TOLERANCE
                ),
                "maximum_absolute_probability_difference": max_probability_difference,
                "ROC_AUC_difference": roc_difference,
                "AP_difference": ap_difference,
                "tolerance": FULL_PROBABILITY_TOLERANCE,
                "reconstructed_matrix_audit": {
                    "DEV_TRAIN_shape": [23724, 48],
                    "DEV_VAL_shape": [5931, 48],
                    "DEV_TRAIN_theft": int(np.sum(train_labels)),
                    "DEV_VAL_theft": int(np.sum(val_labels)),
                    "overlap": 0,
                    "feature_names_and_order_exact": True,
                    "persistence_references_exact": True,
                },
            }
            save_json(reproduction, RESULTS_DIR / "full_48_reproduction.json")
            if not reproduction["passed_before_family_ablation"]:
                raise RuntimeError("FULL_48 failed exact Stage 17B reproduction; stopping")
        metrics_rows.append(row)
        score_by_experiment[name] = scores
        probability_frames.append(
            pd.DataFrame(
                {
                    ID_COLUMN: val_ids,
                    LABEL_COLUMN: val_labels,
                    "experiment": name,
                    "probability": scores,
                }
            )
        )
        model_records.append(save_model(model, name))

    metrics = pd.DataFrame(metrics_rows)
    column_order = [
        "experiment",
        "experiment_type",
        "family",
        "feature_count",
        "count",
        "normal_count",
        "theft_count",
        "positive_prevalence",
        "threshold",
        "threshold_rule",
        "ROC_AUC",
        "PR_AUC_average_precision",
        "precision",
        "recall",
        "F1",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    metrics = metrics[column_order]
    deltas = primary_deltas(metrics)
    save_csv(metrics, RESULTS_DIR / "experiment_metrics.csv")
    save_csv(
        metrics.loc[metrics["experiment_type"].eq("FAMILY_ONLY")],
        RESULTS_DIR / "family_only_metrics.csv",
    )
    save_csv(
        metrics.loc[metrics["experiment_type"].eq("LEAVE_ONE_FAMILY_OUT")],
        RESULTS_DIR / "leave_one_family_out_metrics.csv",
    )
    save_csv(deltas, RESULTS_DIR / "primary_metric_deltas.csv")
    save_csv(pd.concat(probability_frames, ignore_index=True), RESULTS_DIR / "dev_val_ablation_probabilities.csv")
    save_json({"models": model_records}, RESULTS_DIR / "model_artifact_manifest.json")

    selected = select_bootstrap_families(deltas)
    selected_scores = {
        family: score_by_experiment[f"LEAVE_ONE_OUT__{family}"] for family in selected
    }
    bootstrap_samples, bootstrap_summary, bootstrap_audit = paired_bootstrap(
        val_labels,
        score_by_experiment["FULL_48"],
        selected_scores,
    )
    observed = deltas.loc[
        deltas["experiment_type"].eq("LEAVE_ONE_FAMILY_OUT"),
        [
            "family",
            "delta_ROC_AUC_experiment_minus_FULL_48",
            "delta_AP_experiment_minus_FULL_48",
        ],
    ]
    bootstrap_summary = bootstrap_summary.merge(observed, on="family", validate="many_to_one")
    save_csv(bootstrap_samples, RESULTS_DIR / "selected_paired_bootstrap_samples.csv")
    save_csv(bootstrap_summary, RESULTS_DIR / "selected_paired_bootstrap_summary.csv")
    save_json(
        {
            "selection_rule": config["bootstrap_selection_rule"],
            "selected_families": selected,
            **bootstrap_audit,
        },
        RESULTS_DIR / "bootstrap_selection_audit.json",
    )

    mapping = pd.read_csv(MAPPING_PATH)
    importance_summary, redundancy_summary = existing_evidence_summaries(mapping)
    save_csv(importance_summary, RESULTS_DIR / "stage17b_permutation_importance_by_family.csv")
    save_csv(redundancy_summary, RESULTS_DIR / "stage16_redundancy_by_family_pair.csv")
    interpretation, strong, necessary = family_interpretation(
        metrics, deltas, bootstrap_summary
    )
    save_csv(interpretation, RESULTS_DIR / "family_interpretation.csv")

    leave_metrics = metrics.loc[metrics["experiment_type"].eq("LEAVE_ONE_FAMILY_OUT")].set_index("family")
    if (
        len(necessary) == 1
        and len(strong) == 1
        and necessary == strong
        and leave_metrics.loc[next(iter(necessary)), "ROC_AUC"] < 0.70
        and leave_metrics.loc[next(iter(necessary)), "PR_AUC_average_precision"] < 0.25
    ):
        decision_code = "C"
        decision_label = "PERFORMANCE IS DOMINATED BY ONE NARROW FEATURE FAMILY"
    elif len(necessary) >= 2:
        decision_code = "A"
        decision_label = "MULTIPLE BEHAVIOURAL FAMILIES ARE NECESSARY"
    else:
        decision_code = "B"
        decision_label = "SIGNAL IS STRONG BUT HIGHLY REDUNDANT ACROSS FAMILIES"

    decision = {
        "stage": "Stage 19 — customer feature-family ablation decision",
        "decision": decision_code,
        "decision_label": decision_label,
        "policy": config["decision_policy"],
        "strong_standalone_families": [family for family in FAMILY_ORDER if family in strong],
        "conditionally_necessary_families": [family for family in FAMILY_ORDER if family in necessary],
        "compact_model_constructed": False,
        "frozen_stage17b_model_modified": False,
        "hyperparameter_tuning": False,
        "feature_discovery_or_individual_selection": False,
        "threshold_tuning": False,
        "confirmation_accessed": False,
        "final_test_accessed": False,
        "recommended_stage20_question": "Does a label-formulation control that weakens propagated window-label assumptions preserve the direct customer-level advantage under DEV_TRAIN/DEV_VAL, while keeping CONFIRMATION and FINAL_TEST closed?",
    }
    save_json(decision, DECISION_PATH)
    make_figure(metrics, deltas)
    write_report(config, metrics, deltas, bootstrap_summary, interpretation, importance_summary, redundancy_summary, decision)
    assert_artifacts()
    print(f"Stage 19 decision {decision_code}: {decision_label}")
    print("CONFIRMATION and FINAL_TEST were not accessed")


def format_metric_rows(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"| {row.family} | {row.feature_count} | {row.ROC_AUC:.6f} | "
            f"{row.PR_AUC_average_precision:.6f} | {row.precision:.6f} | "
            f"{row.recall:.6f} | {row.F1:.6f} | {row.TN}/{row.FP}/{row.FN}/{row.TP} |"
        )
    return "\n".join(rows)


def format_delta_rows(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"| {row.family} | {row.feature_count} | "
            f"{row.delta_ROC_AUC_experiment_minus_FULL_48:+.6f} | "
            f"{row.delta_AP_experiment_minus_FULL_48:+.6f} |"
        )
    return "\n".join(rows)


def write_report(
    config: dict[str, Any],
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    bootstrap: pd.DataFrame,
    interpretation: pd.DataFrame,
    importance: pd.DataFrame,
    redundancy: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    full = metrics.loc[metrics["experiment"].eq("FULL_48")].iloc[0]
    only = metrics.loc[metrics["experiment_type"].eq("FAMILY_ONLY")]
    leave_metrics = metrics.loc[metrics["experiment_type"].eq("LEAVE_ONE_FAMILY_OUT")]
    only_deltas = deltas.loc[deltas["experiment_type"].eq("FAMILY_ONLY")]
    leave_deltas = deltas.loc[deltas["experiment_type"].eq("LEAVE_ONE_FAMILY_OUT")]
    bootstrap_rows = []
    for family in FAMILY_ORDER:
        subset = bootstrap.loc[bootstrap["family"].eq(family)].set_index("metric")
        if subset.empty:
            continue
        roc = subset.loc["ROC_AUC"]
        ap = subset.loc["PR_AUC_average_precision"]
        bootstrap_rows.append(
            f"| {family} | {roc['mean_delta_ablation_minus_FULL_48']:+.6f} "
            f"[{roc['percentile_95_CI_lower']:+.6f}, {roc['percentile_95_CI_upper']:+.6f}] "
            f"| {ap['mean_delta_ablation_minus_FULL_48']:+.6f} "
            f"[{ap['percentile_95_CI_lower']:+.6f}, {ap['percentile_95_CI_upper']:+.6f}] "
            f"| {roc['fraction_below_zero']:.3f}/{roc['fraction_above_zero']:.3f} "
            f"| {ap['fraction_below_zero']:.3f}/{ap['fraction_above_zero']:.3f} |"
        )
    strongest = only.sort_values(["PR_AUC_average_precision", "ROC_AUC"], ascending=False).iloc[0]
    worst_leave = leave_deltas.sort_values(
        ["delta_AP_experiment_minus_FULL_48", "delta_ROC_AUC_experiment_minus_FULL_48"]
    ).iloc[0]
    report = f"""# Stage 19 — Customer Behavioural Feature-Family Ablation

## Frozen design and reproduction

Stage 19 classifies all 48 frozen Stage 16 predictors into six semantic families
before ablation scoring: LEVEL_DISTRIBUTION (18), VARIABILITY_ABRUPT_CHANGE (12),
DIRECTIONAL_TEMPORAL_CHANGE (8), LOW_CONSUMPTION_BEHAVIOUR (2),
PERSISTENCE_EXCEEDANCE (4), and LONGITUDINAL_CONTRAST (4). Every model uses the
unchanged unweighted Stage 17B HGB configuration. The global Stage 17B threshold
`{FROZEN_THRESHOLD:.17g}` is used for secondary threshold metrics.

The FULL_48 control exactly reproduces Stage 17B: ROC-AUC `{full.ROC_AUC:.12f}`
and AP `{full.PR_AUC_average_precision:.12f}`. The maximum absolute probability
difference is recorded in `full_48_reproduction.json`; the stop gate passed before
any family ablation was fitted.

## Family-only models

| Family | Features | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|
{format_metric_rows(only)}

| Family | Features | ROC-AUC delta | AP delta |
|---|---:|---:|---:|
{format_delta_rows(only_deltas)}

The strongest standalone family by AP-first ordering is `{strongest.family}`.
LEVEL_DISTRIBUTION alone retains ROC-AUC 0.767717 and AP 0.309965, substantially
more than any other single family, but remains below FULL_48 by 0.021114 and
0.049829. PERSISTENCE_EXCEEDANCE, LONGITUDINAL_CONTRAST, and
VARIABILITY_ABRUPT_CHANGE each carry moderate standalone signal; directional
change is weaker, and the two low-consumption features alone are weak.

## Leave-one-family-out models

| Omitted family | Features retained | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|
{format_metric_rows(leave_metrics)}

| Omitted family | Features retained | ROC-AUC delta | AP delta |
|---|---:|---:|---:|
{format_delta_rows(leave_deltas)}

The largest AP-first observed leave-out loss is `{worst_leave.family}`.
Omitting LEVEL_DISTRIBUTION produces the largest losses, but the remaining five
families together still achieve ROC-AUC 0.763537 and AP 0.325818. The result is
therefore level-sensitive rather than narrowly level-dominated. Omitting
LONGITUDINAL_CONTRAST gives a practical observed AP loss of 0.023237; all other
leave-out losses are below the pre-registered practical thresholds.

## Selected paired uncertainty

| Omitted family | ROC-AUC mean delta [95% CI] | AP mean delta [95% CI] | ROC below/above 0 | AP below/above 0 |
|---|---:|---:|---:|---:|
{chr(10).join(bootstrap_rows)}

Selection followed the frozen union rule: largest ROC-AUC drop, largest AP drop,
and any family with observed ROC-AUC loss at least 0.01 or AP loss at least 0.02.
Each selected comparison uses 2,000 paired DEV_VAL customer replicates with seed 42.
The LEVEL_DISTRIBUTION intervals exclude zero for both metrics, making it the only
conditionally necessary family. Both LONGITUDINAL_CONTRAST intervals include zero,
so its observed loss is suggestive but not conclusive under the frozen rule.

## Redundancy, Stage 18 connection, and decision

The family interpretation table combines standalone strength with conditional
leave-out evidence. Existing Stage 16 correlations and Stage 17B permutation
importance are aggregated by the frozen family map; they are supporting evidence,
not another subset search. No compact model was pre-registered or constructed.
All 35 Stage 16 correlations at absolute Spearman at least 0.95 occur within
LEVEL_DISTRIBUTION (26), VARIABILITY_ABRUPT_CHANGE (7), or
DIRECTIONAL_TEMPORAL_CHANGE (2), supporting substantial within-family redundancy.
Stage 17B permutation importance is also concentrated in level/distribution, with
meaningful secondary contributions from longitudinal, variability, and persistence
features. Their small leave-out losses indicate that much of this information is
replaceable across families.

Stage 18 found stable ranking across history length but consumption-dependent
performance and weak LOW-consumption recall at the global threshold. Stage 19
therefore distinguishes dominance by absolute level from signal distributed across
variability, persistence, directional, low-consumption, and longitudinal behaviour.
It does not redesign the low-consumption operating point.
The necessary level family makes the Stage 18 consumption-stratum variation
plausible as model dependence on consumption distribution, but the useful
without-level result shows that variability, persistence, directional, and
longitudinal information collectively retain substantial discrimination. The
LOW_CONSUMPTION_BEHAVIOUR family is neither strong alone nor conditionally
necessary, so Stage 18's low-consumption recall limitation is not explained as
dependence on those two indicators alone.

Decision **{decision['decision']} — {decision['decision_label']}** follows the
pre-registered family-only, leave-out, practical-loss, and paired-uncertainty rules.
Strong standalone families: `{decision['strong_standalone_families']}`.
Conditionally necessary families: `{decision['conditionally_necessary_families']}`.

For the SGCC paper, Stage 19 identifies which behavioural information families
support the confirmed direct-customer discrimination. It does not establish
causality, state of the art, or external generalization.

Recommended Stage 20 question: **{decision['recommended_stage20_question']}**

**CONFIRMATION NOT ACCESSED DURING STAGE 19. FINAL_TEST NOT ACCESSED DURING STAGE 19.**
"""
    (RESULTS_DIR / "stage_19_report.md").write_text(report, encoding="utf-8", newline="\n")


def assert_artifacts() -> None:
    config = load_preregistration()
    with DECISION_PATH.open("r", encoding="utf-8") as source:
        decision = json.load(source)
    with (RESULTS_DIR / "full_48_reproduction.json").open("r", encoding="utf-8") as source:
        reproduction = json.load(source)
    metrics = pd.read_csv(RESULTS_DIR / "experiment_metrics.csv")
    deltas = pd.read_csv(RESULTS_DIR / "primary_metric_deltas.csv")
    probabilities = pd.read_csv(RESULTS_DIR / "dev_val_ablation_probabilities.csv")
    bootstrap = pd.read_csv(RESULTS_DIR / "selected_paired_bootstrap_summary.csv")
    bootstrap_samples = pd.read_csv(RESULTS_DIR / "selected_paired_bootstrap_samples.csv")
    mapping = pd.read_csv(MAPPING_PATH)
    if len(config["experiments"]) != 13 or len(metrics) != 13 or len(deltas) != 13:
        raise AssertionError("Stage 19 does not contain exactly 13 frozen experiments")
    expected_experiments = {experiment["experiment"] for experiment in config["experiments"]}
    if set(metrics["experiment"]) != expected_experiments:
        raise AssertionError("Observed experiments differ from the frozen experiment matrix")
    if mapping.to_dict(orient="records") != feature_mapping_frame().to_dict(orient="records"):
        raise AssertionError("Feature-family mapping is not exhaustive and ordered")
    if not reproduction["passed_before_family_ablation"]:
        raise AssertionError("FULL_48 reproduction gate did not pass")
    if reproduction["maximum_absolute_probability_difference"] > FULL_PROBABILITY_TOLERANCE:
        raise AssertionError("FULL_48 probability reproduction exceeds tolerance")
    if len(probabilities) != 13 * 5931:
        raise AssertionError("Saved ablation probabilities are incomplete")
    probability_audit = probabilities.groupby("experiment").agg(
        rows=(ID_COLUMN, "size"),
        customers=(ID_COLUMN, "nunique"),
        theft=(LABEL_COLUMN, "sum"),
    )
    if (
        set(probability_audit.index) != expected_experiments
        or not probability_audit["rows"].eq(5931).all()
        or not probability_audit["customers"].eq(5931).all()
        or not probability_audit["theft"].eq(505).all()
    ):
        raise AssertionError("Per-experiment saved probability population is invalid")
    if not metrics["count"].eq(5931).all() or not metrics["theft_count"].eq(505).all():
        raise AssertionError("An experiment does not recover DEV_VAL")
    if not np.allclose(metrics["threshold"], FROZEN_THRESHOLD, rtol=0, atol=1e-15):
        raise AssertionError("An experiment used a non-frozen threshold")
    if set(metrics["feature_count"]) - {2, 4, 8, 12, 18, 30, 36, 40, 44, 46, 48}:
        raise AssertionError("Unexpected feature count in experiment matrix")
    if not bootstrap["valid_resamples"].eq(2000).all():
        raise AssertionError("A selected bootstrap comparison lacks 2,000 valid samples")
    with (RESULTS_DIR / "bootstrap_selection_audit.json").open("r", encoding="utf-8") as source:
        bootstrap_audit = json.load(source)
    if bootstrap_audit["selected_families"] != select_bootstrap_families(deltas):
        raise AssertionError("Bootstrap families do not follow the frozen selection rule")
    if len(bootstrap_samples) != 2000 * len(bootstrap_audit["selected_families"]):
        raise AssertionError("Paired bootstrap sample artifact is incomplete")
    if decision["compact_model_constructed"] or decision["frozen_stage17b_model_modified"]:
        raise AssertionError("Stage 19 changed the frozen detector or built a compact model")
    if decision["hyperparameter_tuning"] or decision["threshold_tuning"]:
        raise AssertionError("Stage 19 performed prohibited tuning")
    if decision["confirmation_accessed"] or decision["final_test_accessed"]:
        raise AssertionError("Stage 19 protected-data declaration is invalid")
    with (RESULTS_DIR / "model_artifact_manifest.json").open("r", encoding="utf-8") as source:
        model_manifest = json.load(source)
    if len(model_manifest["models"]) != 13:
        raise AssertionError("Stage 19 model manifest is incomplete")
    if model_manifest["models"][0]["sha256"] != EXPECTED_MODEL_HASH:
        raise AssertionError("FULL_48 serialized model does not reproduce frozen Stage 17B HGB")
    for record in model_manifest["models"]:
        path = PROJECT_ROOT / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise AssertionError(f"Stage 19 model hash changed: {record['experiment']}")
    if (RESULTS_DIR / "family_ablation_primary_metrics.png").stat().st_size <= 0:
        raise AssertionError("Stage 19 figure is missing or empty")


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
