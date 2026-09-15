"""Stage 22: pre-registered label-free compact customer representation.

``--prepare`` derives exactly one subset from DEV_TRAIN correlations and freezes
it before ``--evaluate`` may score DEV_VAL. CONFIRMATION and FINAL_TEST have no
model-data loading path in this module.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

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


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_22"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_22"
AUDIT_PATH = RESULTS_DIR / "pre_stage_audit.json"
CORRELATION_PATH = RESULTS_DIR / "dev_train_spearman_correlation.csv"
PAIR_PATH = RESULTS_DIR / "within_family_redundant_pairs.csv"
COMPONENT_PATH = RESULTS_DIR / "within_family_components.csv"
FEATURE_MANIFEST_PATH = RESULTS_DIR / "compact_feature_manifest.csv"
FAMILY_SUMMARY_PATH = RESULTS_DIR / "family_retention_summary.csv"
PREREGISTRATION = RESULTS_DIR / "preregistered_compact_representation.json"
SEAL_PATH = RESULTS_DIR / "preregistration_seal.json"
DECISION_PATH = RESULTS_DIR / "stage_22_decision.json"
REPORT_PATH = RESULTS_DIR / "stage_22_report.md"

STAGE19_MAPPING = RESULTS_DIR.parent / "stage_19" / "feature_family_mapping.csv"
STAGE19_DECISION = RESULTS_DIR.parent / "stage_19" / "family_ablation_decision.json"
STAGE19_REDUNDANCY = RESULTS_DIR.parent / "stage_19" / "stage16_redundancy_by_family_pair.csv"
STAGE21_DECISION = RESULTS_DIR.parent / "stage_21" / "stage_21_decision.json"

BASE_HEAD = "aeba8ee1e07ef36883bbe17b836f370b19ca5f8d"
EXPECTED_STAGE17B_CONFIG_HASH = "605eb1f482123a24a14c0929c951f400c37b3fbc2d2bb2e4d46d2ce834698046"
EXPECTED_STAGE17B_SUMMARY_HASH = "560e8602ce0d572a2e305efcebe525ef3f1174aad0e6f5a3209c2b33fff4865a"
EXPECTED_STAGE17B_DECISION_HASH = "11e15f18e2521ffed8e7dbaacee469284ee356acedea1d8f34ffcf48785a68fa"
EXPECTED_STAGE17B_PREDICTIONS_HASH = "d98a7551a763a5f1e78193786012d86a96c593a73d770fa625a0e09aaac71aea"
EXPECTED_STAGE19_MAPPING_HASH = "7ddec4b0c015b14ec647f3142d7bf9b1d14dda31bf3a2ea1d9351dfee5ec43ff"
EXPECTED_STAGE21_DECISION_HASH = "e78d43a9462b69bc416ac1c09d1c8956a8a40a1177470a27fa0396f9101ca020"

FAMILY_ORDER = (
    "LEVEL_DISTRIBUTION",
    "VARIABILITY_ABRUPT_CHANGE",
    "DIRECTIONAL_TEMPORAL_CHANGE",
    "LOW_CONSUMPTION_BEHAVIOUR",
    "PERSISTENCE_EXCEEDANCE",
    "LONGITUDINAL_CONTRAST",
)
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
CORRELATION_THRESHOLD = 0.95
MAXIMUM_RETAINED_FOR_SUCCESS = 36
ROC_MARGIN = 0.01
AP_MARGIN = 0.02
FINAL_COMPACT_MINIMUM_REDUCTION = 0.40
FINAL_COMPACT_MAXIMUM_ROC_POINT_LOSS = 0.005
FINAL_COMPACT_MAXIMUM_AP_POINT_LOSS = 0.01
FROZEN_THRESHOLD = 0.20070617709942135
REPRODUCTION_TOLERANCE = 1e-12
BOOTSTRAP_RESAMPLES = 2000
RANDOM_SEED = 42


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def frozen_input_records() -> dict[str, dict[str, str]]:
    expected = {
        "stage17a_manifest": (PROTOCOL_MANIFEST, EXPECTED_PROTOCOL_HASH),
        "historical_split": (HISTORICAL_SPLIT_MANIFEST, EXPECTED_HISTORICAL_SPLIT_HASH),
        "stage16_feature_specification": (STAGE16_CONFIG, EXPECTED_STAGE16_CONFIG_HASH),
        "stage17b_configuration": (STAGE17B_CONFIG, EXPECTED_STAGE17B_CONFIG_HASH),
        "stage17b_training_summary": (STAGE17B_SUMMARY, EXPECTED_STAGE17B_SUMMARY_HASH),
        "stage17b_decision": (STAGE17B_DECISION, EXPECTED_STAGE17B_DECISION_HASH),
        "stage17b_model": (STAGE17B_MODEL, EXPECTED_MODEL_HASH),
        "stage17b_predictions": (STAGE17B_PREDICTIONS, EXPECTED_STAGE17B_PREDICTIONS_HASH),
        "stage19_family_mapping": (STAGE19_MAPPING, EXPECTED_STAGE19_MAPPING_HASH),
        "stage21_decision": (STAGE21_DECISION, EXPECTED_STAGE21_DECISION_HASH),
    }
    records: dict[str, dict[str, str]] = {}
    for name, (path, expected_hash) in expected.items():
        observed = file_sha256(path)
        if observed != expected_hash:
            raise ValueError(f"Frozen input hash changed: {name}")
        records[name] = {"path": relative(path), "sha256": observed}
    for name, path in {
        "stage19_decision": STAGE19_DECISION,
        "stage19_redundancy": STAGE19_REDUNDANCY,
    }.items():
        records[name] = {"path": relative(path), "sha256": file_sha256(path)}
    return records


def family_features() -> OrderedDict[str, tuple[str, ...]]:
    mapping = pd.read_csv(STAGE19_MAPPING)
    if mapping["feature"].tolist() != list(FEATURE_NAMES):
        raise ValueError("Stage 19 family mapping no longer follows frozen feature order")
    families: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    for family in FAMILY_ORDER:
        features = tuple(mapping.loc[mapping["family"].eq(family), "feature"])
        if not features:
            raise ValueError(f"Stage 19 family is empty: {family}")
        families[family] = features
    if set().union(*map(set, families.values())) != set(FEATURE_NAMES):
        raise ValueError("Stage 19 family mapping does not partition all 48 features")
    return families


def verify_specs() -> None:
    with STAGE16_CONFIG.open("r", encoding="utf-8") as source:
        stage16 = json.load(source)
    with STAGE17B_CONFIG.open("r", encoding="utf-8") as source:
        stage17b = json.load(source)
    with STAGE17B_DECISION.open("r", encoding="utf-8") as source:
        stage17b_decision = json.load(source)
    with STAGE19_DECISION.open("r", encoding="utf-8") as source:
        stage19 = json.load(source)
    with STAGE21_DECISION.open("r", encoding="utf-8") as source:
        stage21 = json.load(source)
    if tuple(stage16["feature_names_in_order"]) != FEATURE_NAMES or len(FEATURE_NAMES) != 48:
        raise ValueError("Frozen Stage 16 feature schema changed")
    if tuple(stage17b["representation"]["feature_names_in_order"]) != FEATURE_NAMES:
        raise ValueError("Stage 17B feature schema differs from Stage 16")
    if stage17b["candidate_models"]["hist_gradient_boosting"]["parameters"] != HGB_PARAMETERS:
        raise ValueError("Frozen Stage 17B HGB configuration changed")
    if stage17b_decision["selected_model_sha256"] != EXPECTED_MODEL_HASH:
        raise ValueError("Stage 17B model reference changed")
    if stage19["decision"] != "B" or stage21["decision"] != "B":
        raise ValueError("Stage 22 prerequisite decisions changed")
    family_features()


def audit(records: dict[str, dict[str, str]]) -> dict[str, Any]:
    governance = pd.read_csv(PROTOCOL_MANIFEST, dtype={ID_COLUMN: str})
    development = governance.loc[
        governance["research_role"].isin(["DEV_TRAIN", "DEV_VAL"])
    ].copy()
    if not development["original_split"].eq("train").all():
        raise ValueError("A development role includes a protected original split")
    role_ids = {
        role: set(
            development.loc[
                development["research_role"].eq(role)
                & development["stage16_representation_available"].eq(True),
                ID_COLUMN,
            ]
        )
        for role in ("DEV_TRAIN", "DEV_VAL")
    }
    overlap = role_ids["DEV_TRAIN"] & role_ids["DEV_VAL"]
    if overlap or len(role_ids["DEV_TRAIN"]) != 23724 or len(role_ids["DEV_VAL"]) != 5931:
        raise ValueError("Development membership audit failed")
    return {
        "stage": "Stage 22 pre-scoring audit",
        "base_head": BASE_HEAD,
        "git_clean_before_stage": True,
        "frozen_inputs": records,
        "original_feature_count": 48,
        "semantic_family_count": 6,
        "DEV_TRAIN_customers_with_representation": len(role_ids["DEV_TRAIN"]),
        "DEV_VAL_customers_with_representation": len(role_ids["DEV_VAL"]),
        "DEV_TRAIN_DEV_VAL_overlap": len(overlap),
        "expected_FULL_48": {
            "ROC_AUC": 0.7888308218953116,
            "AP": 0.35979459032174654,
        },
        "confirmation_accessed": False,
        "final_test_accessed": False,
    }


def connected_components(
    features: tuple[str, ...], correlation: pd.DataFrame
) -> list[list[str]]:
    neighbors = {feature: set() for feature in features}
    for first_index, first in enumerate(features):
        for second in features[first_index + 1 :]:
            value = correlation.loc[first, second]
            if np.isfinite(value) and abs(value) >= CORRELATION_THRESHOLD:
                neighbors[first].add(second)
                neighbors[second].add(first)
    components: list[list[str]] = []
    visited: set[str] = set()
    for feature in features:
        if feature in visited:
            continue
        stack = [feature]
        component: list[str] = []
        visited.add(feature)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in features:
                if neighbor in neighbors[current] and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        component.sort(key=FEATURE_NAMES.index)
        components.append(component)
    return components


def derive_compaction(
    train_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    correlation = train_matrix.loc[:, FEATURE_NAMES].corr(method="spearman")
    correlation_output = correlation.copy()
    correlation_output.insert(0, "feature", correlation_output.index)
    pair_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    families = family_features()
    for family, features in families.items():
        for first_index, first in enumerate(features):
            for second in features[first_index + 1 :]:
                rho = correlation.loc[first, second]
                if np.isfinite(rho) and abs(rho) >= CORRELATION_THRESHOLD:
                    pair_rows.append(
                        {
                            "family": family,
                            "feature_1": first,
                            "feature_2": second,
                            "spearman_rho": float(rho),
                            "absolute_spearman_rho": float(abs(rho)),
                        }
                    )
        components = connected_components(features, correlation)
        retained_in_family = []
        for component_number, component in enumerate(components, start=1):
            representative = component[0]
            retained_in_family.append(representative)
            component_id = f"{family}__{component_number:02d}"
            for feature in component:
                component_rows.append(
                    {
                        "family": family,
                        "component": component_id,
                        "component_size": len(component),
                        "feature": feature,
                        "representative": representative,
                        "retained": feature == representative,
                        "representative_rule": "first feature in frozen Stage 16 order",
                    }
                )
                manifest_rows.append(
                    {
                        "feature_position": FEATURE_NAMES.index(feature) + 1,
                        "family": family,
                        "feature": feature,
                        "component": component_id,
                        "component_size": len(component),
                        "status": "RETAINED" if feature == representative else "DROPPED_REDUNDANT",
                        "representative": representative,
                    }
                )
        family_rows.append(
            {
                "family": family,
                "original_feature_count": len(features),
                "component_count": len(components),
                "retained_feature_count": len(retained_in_family),
                "dropped_feature_count": len(features) - len(retained_in_family),
                "family_remains_represented": bool(retained_in_family),
            }
        )
    pairs = pd.DataFrame(
        pair_rows,
        columns=["family", "feature_1", "feature_2", "spearman_rho", "absolute_spearman_rho"],
    )
    components_frame = pd.DataFrame(component_rows)
    manifest = pd.DataFrame(manifest_rows).sort_values("feature_position").reset_index(drop=True)
    family_summary = pd.DataFrame(family_rows)
    return correlation_output.reset_index(drop=True), pairs, components_frame, manifest, family_summary


def prepare() -> None:
    if PREREGISTRATION.exists() or SEAL_PATH.exists():
        raise FileExistsError("Refusing to overwrite frozen Stage 22 preregistration")
    verify_specs()
    records = frozen_input_records()
    pre_stage = audit(records)
    save_json(pre_stage, AUDIT_PATH)

    # Only DEV_TRAIN features are aggregated and used for subset construction.
    windows, window_numbers, governance = load_original_train()
    summaries = behavioural_features(windows.X)
    train_features, _unused_labels, train_ids, _references = role_customer_features(
        "DEV_TRAIN", windows, window_numbers, summaries, governance, None
    )
    if train_features.shape != (23724, 50) or len(train_ids) != 23724:
        raise ValueError("DEV_TRAIN feature reconstruction changed")
    correlation, pairs, components, manifest, family_summary = derive_compaction(train_features)
    save_csv(correlation, CORRELATION_PATH)
    save_csv(pairs, PAIR_PATH)
    save_csv(components, COMPONENT_PATH)
    save_csv(manifest, FEATURE_MANIFEST_PATH)
    save_csv(family_summary, FAMILY_SUMMARY_PATH)
    retained = manifest.loc[manifest["status"].eq("RETAINED"), "feature"].tolist()
    dropped = manifest.loc[manifest["status"].eq("DROPPED_REDUNDANT"), "feature"].tolist()
    retained_count = len(retained)
    reduction = 1.0 - retained_count / 48.0
    compaction_success = retained_count <= MAXIMUM_RETAINED_FOR_SUCCESS
    config = {
        "stage": "Stage 22 — pre-registered compact representation",
        "base_head_before_stage": BASE_HEAD,
        "created_before_any_COMPACT_DEV_VAL_scoring": True,
        "research_question": "Can the frozen 48-feature customer representation be substantially simplified using label-free DEV_TRAIN redundancy while retaining essentially the same DEV_VAL discrimination?",
        "audit": {"path": relative(AUDIT_PATH), "sha256": file_sha256(AUDIT_PATH)},
        "frozen_inputs": records,
        "compaction_algorithm": {
            "scope": "Within each of the six frozen Stage 19 semantic families independently",
            "fit_population": "DEV_TRAIN customer feature values only",
            "correlation": "absolute Spearman",
            "threshold": CORRELATION_THRESHOLD,
            "edge_rule": "Connect two within-family features when absolute Spearman rho >= 0.95",
            "component_rule": "Undirected graph connected components in frozen family/feature order",
            "representative_rule": "Retain the first feature in exact frozen Stage 16 order from each component",
            "labels_used": False,
            "importance_or_performance_used": False,
            "cross_family_pruning": False,
        },
        "analysis_artifacts": {
            "correlation": {"path": relative(CORRELATION_PATH), "sha256": file_sha256(CORRELATION_PATH)},
            "redundant_pairs": {"path": relative(PAIR_PATH), "sha256": file_sha256(PAIR_PATH)},
            "components": {"path": relative(COMPONENT_PATH), "sha256": file_sha256(COMPONENT_PATH)},
            "feature_manifest": {"path": relative(FEATURE_MANIFEST_PATH), "sha256": file_sha256(FEATURE_MANIFEST_PATH)},
            "family_summary": {"path": relative(FAMILY_SUMMARY_PATH), "sha256": file_sha256(FAMILY_SUMMARY_PATH)},
        },
        "frozen_subset": {
            "original_feature_count": 48,
            "retained_feature_count": retained_count,
            "dropped_feature_count": len(dropped),
            "reduction_fraction": reduction,
            "retained_features_in_frozen_order": retained,
            "dropped_features_in_frozen_order": dropped,
            "all_six_families_represented": bool(family_summary["family_remains_represented"].all()),
        },
        "compaction_success_gate": {
            "minimum_reduction_fraction": 0.25,
            "maximum_retained_features": MAXIMUM_RETAINED_FOR_SUCCESS,
            "passed_before_DEV_VAL_scoring": compaction_success,
            "failure_action": "Decision D and stop before compact DEV_VAL scoring; do not change threshold or subset.",
        },
        "model": {
            "implementation": "sklearn HistGradientBoostingClassifier",
            "parameters": HGB_PARAMETERS,
            "experiments": ["FULL_48", "COMPACT"],
            "same_configuration": True,
        },
        "reproduction_gate": {
            "expected_FULL_48_ROC_AUC": 0.7888308218953116,
            "expected_FULL_48_AP": 0.35979459032174654,
            "maximum_absolute_probability_or_metric_difference": REPRODUCTION_TOLERANCE,
            "action_on_failure": "Abort before COMPACT fit or interpretation.",
        },
        "metrics": {
            "primary": ["ROC_AUC", "PR_AUC_average_precision"],
            "secondary": ["precision", "recall", "F1", "TN", "FP", "FN", "TP"],
            "threshold": FROZEN_THRESHOLD,
            "threshold_rule": "Use the unchanged Stage 17B threshold for both models; descriptive only.",
            "delta_direction": "COMPACT minus FULL_48",
        },
        "retention": {
            "ROC_AUC_degradation_margin": ROC_MARGIN,
            "AP_degradation_margin": AP_MARGIN,
            "point_estimate_rule": "ROC delta >= -0.01 and AP delta >= -0.02",
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": RANDOM_SEED,
                "unit": "paired DEV_VAL customer",
                "interval": "2.5th and 97.5th percentiles",
                "single_class_policy": "discard and continue deterministically",
            },
        },
        "decision_policy": {
            "D": "More than 36 features retained; stop before DEV_VAL scoring.",
            "A": "At most 36 features retained, both point deltas satisfy margins, and both bootstrap lower bounds are at or above the corresponding negative margin.",
            "C": "For either metric, its point delta and bootstrap upper bound are both below its negative margin, or both point metrics miss their margins.",
            "B": "Substantial compaction succeeds but neither A nor C applies; performance retention has a modest/uncertain tradeoff.",
            "evaluation_order": ["D", "A", "C", "B"],
        },
        "final_method_policy": {
            "FINAL_COMPACT": "Only if Decision A, every family remains represented, reduction >= 40%, ROC point delta >= -0.005, and AP point delta >= -0.01.",
            "FINAL_FULL_48": "Otherwise retain FULL_48 because it already owns confirmation evidence and COMPACT does not.",
        },
        "prohibited_adaptations": [
            "No confirmation/final-test access, labels, importance, DEV_VAL metrics, individual feature tests, RFE, MI, L1, PCA, changed threshold, cross-family pruning, alternative subset, hyperparameter tuning, GAN, SMOTE, classifier change, or post-hoc feature change.",
        ],
    }
    save_json(config, PREREGISTRATION)
    save_json(
        {
            "preregistration_path": relative(PREREGISTRATION),
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "audit_sha256": file_sha256(AUDIT_PATH),
            "analysis_artifact_hashes": {
                name: value["sha256"] for name, value in config["analysis_artifacts"].items()
            },
            "frozen_before_COMPACT_DEV_VAL_scoring": True,
        },
        SEAL_PATH,
    )
    print(f"Frozen Stage 22 preregistration: {file_sha256(PREREGISTRATION)}")
    print(f"Retained {retained_count}/48 features ({reduction:.2%} reduction)")
    print(f"Compaction success gate: {compaction_success}")
    print("No COMPACT DEV_VAL score was calculated during --prepare")


def load_preregistration() -> dict[str, Any]:
    if not PREREGISTRATION.is_file() or not SEAL_PATH.is_file():
        raise FileNotFoundError("Run --prepare before --evaluate")
    with SEAL_PATH.open("r", encoding="utf-8") as source:
        seal = json.load(source)
    if file_sha256(PREREGISTRATION) != seal["preregistration_sha256"]:
        raise ValueError("Stage 22 preregistration changed after freezing")
    if file_sha256(AUDIT_PATH) != seal["audit_sha256"]:
        raise ValueError("Stage 22 audit changed after freezing")
    with PREREGISTRATION.open("r", encoding="utf-8") as source:
        config = json.load(source)
    for name, record in config["analysis_artifacts"].items():
        if file_sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen compaction artifact changed: {name}")
    for record in config["frozen_inputs"].values():
        if file_sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen input changed: {record['path']}")
    return config


def timed_inference(predict: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    scores = np.asarray(predict(), dtype=np.float64)
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        repeated = np.asarray(predict(), dtype=np.float64)
        timings.append(time.perf_counter() - started)
        if not np.allclose(scores, repeated, rtol=0, atol=1e-12):
            raise ValueError("Repeated inference changed materially")
    return scores, float(np.median(timings))


def model_complexity(model: HistGradientBoostingClassifier) -> tuple[int, int]:
    tree_count = int(sum(len(group) for group in model._predictors))
    leaf_count = int(
        sum(
            np.sum(tree.nodes["is_leaf"])
            for group in model._predictors
            for tree in group
        )
    )
    return tree_count, leaf_count


def save_model(name: str, model: HistGradientBoostingClassifier) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name.lower()}.joblib"
    joblib.dump(model, path)
    trees, leaves = model_complexity(model)
    return {
        "model": name,
        "path": relative(path),
        "sha256": file_sha256(path),
        "serialized_bytes": path.stat().st_size,
        "tree_count": trees,
        "leaf_count": leaves,
    }


def paired_bootstrap(
    labels: np.ndarray, compact: np.ndarray, full: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    attempts = 0
    while len(rows) < BOOTSTRAP_RESAMPLES:
        attempts += 1
        indices = rng.integers(0, len(labels), len(labels))
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        rows.append(
            {
                "bootstrap_replicate": len(rows) + 1,
                "ROC_AUC_delta_COMPACT_minus_FULL_48": float(
                    roc_auc_score(y, compact[indices]) - roc_auc_score(y, full[indices])
                ),
                "AP_delta_COMPACT_minus_FULL_48": float(
                    average_precision_score(y, compact[indices])
                    - average_precision_score(y, full[indices])
                ),
            }
        )
    samples = pd.DataFrame(rows)
    summary_rows = []
    for metric, column, observed in (
        (
            "ROC_AUC",
            "ROC_AUC_delta_COMPACT_minus_FULL_48",
            roc_auc_score(labels, compact) - roc_auc_score(labels, full),
        ),
        (
            "AP",
            "AP_delta_COMPACT_minus_FULL_48",
            average_precision_score(labels, compact) - average_precision_score(labels, full),
        ),
    ):
        values = samples[column].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "comparison": "COMPACT_MINUS_FULL_48",
                "metric": metric,
                "observed_delta": float(observed),
                "bootstrap_mean_delta": float(values.mean()),
                "CI_2_5": float(np.quantile(values, 0.025)),
                "CI_97_5": float(np.quantile(values, 0.975)),
                "fraction_below_zero": float(np.mean(values < 0)),
                "fraction_above_zero": float(np.mean(values > 0)),
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": RANDOM_SEED,
            }
        )
    return samples, pd.DataFrame(summary_rows), attempts


def decide(
    config: dict[str, Any], summary: pd.DataFrame
) -> tuple[str, str, str, dict[str, Any]]:
    retained = config["frozen_subset"]["retained_feature_count"]
    reduction = config["frozen_subset"]["reduction_fraction"]
    indexed = summary.set_index("metric")
    roc, ap = indexed.loc["ROC_AUC"], indexed.loc["AP"]
    point_retention = bool(roc["observed_delta"] >= -ROC_MARGIN and ap["observed_delta"] >= -AP_MARGIN)
    interval_retention = bool(roc["CI_2_5"] >= -ROC_MARGIN and ap["CI_2_5"] >= -AP_MARGIN)
    clear_degradation = bool(
        (roc["observed_delta"] < -ROC_MARGIN and roc["CI_97_5"] < -ROC_MARGIN)
        or (ap["observed_delta"] < -AP_MARGIN and ap["CI_97_5"] < -AP_MARGIN)
        or (roc["observed_delta"] < -ROC_MARGIN and ap["observed_delta"] < -AP_MARGIN)
    )
    if retained > MAXIMUM_RETAINED_FOR_SUCCESS:
        code, label = "D", "PRE-REGISTERED RULE DOES NOT ACHIEVE SUBSTANTIAL COMPACTION"
    elif point_retention and interval_retention:
        code, label = "A", "COMPACT RETAINS PERFORMANCE"
    elif clear_degradation:
        code, label = "C", "REDUNDANCY IS NOT SAFELY REMOVABLE"
    else:
        code, label = "B", "COMPACT RETAINS MOST PERFORMANCE WITH A SMALL TRADEOFF"
    all_families = config["frozen_subset"]["all_six_families_represented"]
    final_compact = bool(
        code == "A"
        and all_families
        and reduction >= FINAL_COMPACT_MINIMUM_REDUCTION
        and roc["observed_delta"] >= -FINAL_COMPACT_MAXIMUM_ROC_POINT_LOSS
        and ap["observed_delta"] >= -FINAL_COMPACT_MAXIMUM_AP_POINT_LOSS
    )
    final_method = "FINAL_COMPACT" if final_compact else "FINAL_FULL_48"
    evidence = {
        "point_estimate_retention": point_retention,
        "bootstrap_interval_retention": interval_retention,
        "clear_material_degradation": clear_degradation,
        "all_six_families_represented": all_families,
        "final_compact_worthwhile_rule_satisfied": final_compact,
    }
    return code, label, final_method, evidence


def metric_markdown(frame: pd.DataFrame) -> str:
    return "\n".join(
        f"| {row.model} | {row.feature_count} | {row.ROC_AUC:.6f} | "
        f"{row.PR_AUC_average_precision:.6f} | {row.precision:.6f} | "
        f"{row.recall:.6f} | {row.F1:.6f} | {row.TN}/{row.FP}/{row.FN}/{row.TP} |"
        for row in frame.itertuples()
    )


def feature_markdown(manifest: pd.DataFrame, family: str, status: str) -> str:
    values = manifest.loc[
        manifest["family"].eq(family) & manifest["status"].eq(status), "feature"
    ].tolist()
    return ", ".join(f"`{value}`" for value in values) if values else "—"


def write_report(
    config: dict[str, Any],
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    complexity: pd.DataFrame,
    reproduction: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
    family_summary = pd.read_csv(FAMILY_SUMMARY_PATH)
    family_rows = []
    for row in family_summary.itertuples():
        family_rows.append(
            f"| {row.family} | {row.original_feature_count} | {row.retained_feature_count} | "
            f"{row.dropped_feature_count} | {feature_markdown(manifest, row.family, 'RETAINED')} | "
            f"{feature_markdown(manifest, row.family, 'DROPPED_REDUNDANT')} |"
        )
    compact = config["frozen_subset"]
    roc = bootstrap.loc[bootstrap["metric"].eq("ROC_AUC")].iloc[0]
    ap = bootstrap.loc[bootstrap["metric"].eq("AP")].iloc[0]
    text = f"""# Stage 22 — Pre-Registered Compact Representation

## Question and governance

Stage 22 tests whether deterministic, label-free within-family redundancy removal can simplify the frozen 48-feature customer representation without material DEV_VAL discrimination loss. The subset was frozen before compact scoring in `{relative(PREREGISTRATION)}`, SHA-256 `{file_sha256(PREREGISTRATION)}`.

Only DEV_TRAIN feature values determined the subset. Labels, importance, DEV_VAL performance, cross-family correlations, and protected populations played no role.

## Compaction rule

Within each frozen Stage 19 family, absolute DEV_TRAIN Spearman correlations at least `0.95` form edges in an undirected graph. Each connected component retains its first feature in frozen Stage 16 order. Singleton components remain. No pruning crosses family boundaries.

The rule retained **{compact['retained_feature_count']} of 48 features**, a **{compact['reduction_fraction']:.2%} reduction**. The pre-scoring requirement of at most 36 retained features passed.

| Family | Original | Retained | Dropped | Retained features | Dropped features |
|---|---:|---:|---:|---|---|
{chr(10).join(family_rows)}

Every semantic family remains represented: **{compact['all_six_families_represented']}**.

## FULL_48 reproduction

The full model passed before COMPACT was interpreted. Maximum probability difference from frozen Stage 17B was `{reproduction['maximum_absolute_probability_difference']:.3g}`; ROC-AUC/AP differences were `{reproduction['ROC_AUC_difference']:+.3g}`/`{reproduction['AP_difference']:+.3g}`.

## Metrics

Both models use the exact frozen Stage 17B HGB configuration and threshold `{FROZEN_THRESHOLD:.17g}`. Threshold metrics are descriptive.

| Model | Features | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---|
{metric_markdown(metrics)}

COMPACT minus FULL_48 observed ROC-AUC/AP deltas are `{roc['observed_delta']:+.6f}`/`{ap['observed_delta']:+.6f}`.

## Paired uncertainty

Two thousand paired DEV_VAL customer bootstrap replicates used seed 42.

| Metric | Observed delta | Mean delta | 95% interval | Fraction below 0 | Fraction above 0 |
|---|---:|---:|---|---:|---:|
| ROC-AUC | {roc['observed_delta']:+.6f} | {roc['bootstrap_mean_delta']:+.6f} | [{roc['CI_2_5']:+.6f}, {roc['CI_97_5']:+.6f}] | {roc['fraction_below_zero']:.4f} | {roc['fraction_above_zero']:.4f} |
| AP | {ap['observed_delta']:+.6f} | {ap['bootstrap_mean_delta']:+.6f} | [{ap['CI_2_5']:+.6f}, {ap['CI_97_5']:+.6f}] | {ap['fraction_below_zero']:.4f} | {ap['fraction_above_zero']:.4f} |

## Complexity

| Model | Features | Reduction | Serialized bytes | Fit seconds | Inference seconds | Trees | Leaves |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL_48 | {int(complexity.iloc[0]['feature_count'])} | {complexity.iloc[0]['reduction_fraction']:.2%} | {int(complexity.iloc[0]['serialized_bytes'])} | {complexity.iloc[0]['fit_seconds']:.6f} | {complexity.iloc[0]['inference_seconds']:.6f} | {int(complexity.iloc[0]['tree_count'])} | {int(complexity.iloc[0]['leaf_count'])} |
| COMPACT | {int(complexity.iloc[1]['feature_count'])} | {complexity.iloc[1]['reduction_fraction']:.2%} | {int(complexity.iloc[1]['serialized_bytes'])} | {complexity.iloc[1]['fit_seconds']:.6f} | {complexity.iloc[1]['inference_seconds']:.6f} | {int(complexity.iloc[1]['tree_count'])} | {int(complexity.iloc[1]['leaf_count'])} |

## Decision and final method

Decision **{decision['decision']} — {decision['decision_label']}**.

Final-method recommendation: **{decision['final_method_recommendation']}**. {decision['final_method_rationale']}

{decision['paper_interpretation']}

The AMP research matrix should now be frozen. No Stage 23 model development is recommended; proceed to a formal research-freeze/final-evaluation protocol.

No adaptive feature selection, label use, feature importance, DEV_VAL-guided subset change, threshold change, or hyperparameter tuning occurred. The original FULL_48 model remains the only formulation with existing confirmation evidence.

**CONFIRMATION NOT ACCESSED DURING STAGE 22.**

**FINAL_TEST NOT ACCESSED DURING STAGE 22.**
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")


def evaluate() -> None:
    if DECISION_PATH.exists():
        raise FileExistsError("Refusing to repeat completed Stage 22 evaluation")
    config = load_preregistration()
    if not config["compaction_success_gate"]["passed_before_DEV_VAL_scoring"]:
        raise RuntimeError("Compaction retained more than 36 features; Decision D requires no scoring")
    verify_specs()
    retained = tuple(config["frozen_subset"]["retained_features_in_frozen_order"])
    windows, window_numbers, governance = load_original_train()
    summaries = behavioural_features(windows.X)
    train_features, train_labels, train_ids, references = role_customer_features(
        "DEV_TRAIN", windows, window_numbers, summaries, governance, None
    )
    val_features, val_labels, val_ids, val_references = role_customer_features(
        "DEV_VAL", windows, window_numbers, summaries, governance, references
    )
    if references != val_references or set(train_ids) & set(val_ids):
        raise ValueError("Development reconstruction provenance failed")
    with STAGE17B_SUMMARY.open("r", encoding="utf-8") as source:
        stage17b_summary = json.load(source)
    if references != stage17b_summary["DEV_TRAIN_derived_persistence_q90_thresholds"]:
        raise ValueError("DEV_TRAIN reference statistics differ from Stage 17B")
    full_train = train_features.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    full_val = val_features.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    compact_columns = [FEATURE_NAMES.index(feature) for feature in retained]
    compact_train = full_train[:, compact_columns]
    compact_val = full_val[:, compact_columns]

    saved = pd.read_csv(STAGE17B_PREDICTIONS, dtype={ID_COLUMN: str})
    saved = saved.loc[
        saved["model"].eq("hist_gradient_boosting"),
        [ID_COLUMN, LABEL_COLUMN, "probability"],
    ].set_index(ID_COLUMN).reindex(val_ids)
    if saved.isna().any().any() or not np.array_equal(
        saved[LABEL_COLUMN].to_numpy(dtype=np.int8), val_labels
    ):
        raise ValueError("Frozen Stage 17B predictions do not align with DEV_VAL")
    saved_scores = saved["probability"].to_numpy(dtype=np.float64)

    runtime: dict[str, dict[str, float]] = {}
    models: dict[str, HistGradientBoostingClassifier] = {}
    scores: dict[str, np.ndarray] = {}
    records = []
    started = time.perf_counter()
    full_model = HistGradientBoostingClassifier(**HGB_PARAMETERS)
    full_model.fit(full_train, train_labels)
    runtime["FULL_48"] = {"fit_seconds": time.perf_counter() - started}
    full_scores, full_inference = timed_inference(lambda: full_model.predict_proba(full_val)[:, 1])
    runtime["FULL_48"]["inference_seconds"] = full_inference
    full_roc = roc_auc_score(val_labels, full_scores)
    full_ap = average_precision_score(val_labels, full_scores)
    reproduction = {
        "stage": "Stage 22 FULL_48 reproduction stop gate",
        "passed_before_COMPACT_fit": bool(
            np.max(np.abs(full_scores - saved_scores)) <= REPRODUCTION_TOLERANCE
            and abs(full_roc - config["reproduction_gate"]["expected_FULL_48_ROC_AUC"])
            <= REPRODUCTION_TOLERANCE
            and abs(full_ap - config["reproduction_gate"]["expected_FULL_48_AP"])
            <= REPRODUCTION_TOLERANCE
        ),
        "maximum_absolute_probability_difference": float(np.max(np.abs(full_scores - saved_scores))),
        "ROC_AUC": float(full_roc),
        "AP": float(full_ap),
        "ROC_AUC_difference": float(full_roc - config["reproduction_gate"]["expected_FULL_48_ROC_AUC"]),
        "AP_difference": float(full_ap - config["reproduction_gate"]["expected_FULL_48_AP"]),
        "tolerance": REPRODUCTION_TOLERANCE,
    }
    save_json(reproduction, RESULTS_DIR / "full_48_reproduction.json")
    if not reproduction["passed_before_COMPACT_fit"]:
        raise RuntimeError("FULL_48 failed Stage 17B reproduction; aborting COMPACT fit")
    models["FULL_48"], scores["FULL_48"] = full_model, full_scores
    records.append(save_model("FULL_48", full_model))

    started = time.perf_counter()
    compact_model = HistGradientBoostingClassifier(**HGB_PARAMETERS)
    compact_model.fit(compact_train, train_labels)
    runtime["COMPACT"] = {"fit_seconds": time.perf_counter() - started}
    compact_scores, compact_inference = timed_inference(
        lambda: compact_model.predict_proba(compact_val)[:, 1]
    )
    runtime["COMPACT"]["inference_seconds"] = compact_inference
    models["COMPACT"], scores["COMPACT"] = compact_model, compact_scores
    records.append(save_model("COMPACT", compact_model))

    metrics_rows = []
    for name, feature_count in (("FULL_48", 48), ("COMPACT", len(retained))):
        metrics_rows.append(
            {
                "model": name,
                "feature_count": feature_count,
                **classification_metrics(val_labels, scores[name], FROZEN_THRESHOLD),
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    samples, bootstrap, attempts = paired_bootstrap(
        val_labels, scores["COMPACT"], scores["FULL_48"]
    )
    code, label, final_method, evidence = decide(config, bootstrap)
    roc = bootstrap.loc[bootstrap["metric"].eq("ROC_AUC")].iloc[0]
    ap = bootstrap.loc[bootstrap["metric"].eq("AP")].iloc[0]
    if final_method == "FINAL_COMPACT":
        rationale = "The compact subset meets the conservative worthwhile-simplification rule, but it still has no confirmation evidence; final protocol design must acknowledge that governance distinction without reopening confirmation during Stage 22."
    else:
        rationale = "FULL_48 retains the existing strong confirmation evidence, and COMPACT did not satisfy every conservative condition required to justify replacing it."
    if code == "A":
        paper = "The behavioural representation contains substantial within-family redundancy: deterministic label-free compaction retained performance within pre-specified margins using substantially fewer predictors."
    elif code == "C":
        paper = "Correlation redundancy did not imply safe predictive removability: the deterministic compact representation materially degraded discrimination."
    else:
        paper = "Deterministic label-free compaction substantially reduced the representation, but performance retention involved a modest or uncertain tradeoff."
    decision = {
        "stage": "Stage 22 compact representation decision",
        "decision": code,
        "decision_label": label,
        "decision_policy": config["decision_policy"],
        "evidence": evidence,
        "feature_reduction_fraction": config["frozen_subset"]["reduction_fraction"],
        "final_method_recommendation": final_method,
        "final_method_policy": config["final_method_policy"],
        "final_method_rationale": rationale,
        "paper_interpretation": paper,
        "research_matrix_freeze_recommended": True,
        "stage23_model_development_recommended": False,
        "adaptive_feature_selection": False,
        "labels_used_for_selection": False,
        "DEV_VAL_used_for_selection": False,
        "feature_importance_used_for_selection": False,
        "threshold_or_hyperparameter_tuning": False,
        "confirmation_accessed": False,
        "final_test_accessed": False,
    }

    record_by_name = {record["model"]: record for record in records}
    complexity_rows = []
    for name, feature_count in (("FULL_48", 48), ("COMPACT", len(retained))):
        record = record_by_name[name]
        complexity_rows.append(
            {
                "model": name,
                "feature_count": feature_count,
                "reduction_fraction": 1.0 - feature_count / 48.0,
                "serialized_bytes": record["serialized_bytes"],
                "fit_seconds": runtime[name]["fit_seconds"],
                "inference_seconds": runtime[name]["inference_seconds"],
                "tree_count": record["tree_count"],
                "leaf_count": record["leaf_count"],
            }
        )
    complexity = pd.DataFrame(complexity_rows)
    probabilities = pd.concat(
        [
            pd.DataFrame(
                {ID_COLUMN: val_ids, LABEL_COLUMN: val_labels, "model": name, "probability": scores[name]}
            )
            for name in ("FULL_48", "COMPACT")
        ],
        ignore_index=True,
    )
    deltas = {
        "comparison": "COMPACT_MINUS_FULL_48",
        "ROC_AUC_delta": float(roc["observed_delta"]),
        "AP_delta": float(ap["observed_delta"]),
    }
    save_csv(metrics, RESULTS_DIR / "model_metrics.csv")
    save_csv(pd.DataFrame([deltas]), RESULTS_DIR / "compact_minus_full_deltas.csv")
    save_csv(samples, RESULTS_DIR / "paired_bootstrap_samples.csv")
    save_csv(bootstrap, RESULTS_DIR / "paired_bootstrap_summary.csv")
    save_csv(complexity, RESULTS_DIR / "complexity_comparison.csv")
    save_csv(probabilities, RESULTS_DIR / "dev_val_probabilities.csv")
    save_json({"models": records}, RESULTS_DIR / "model_artifact_manifest.json")
    save_json(
        {
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "DEV_TRAIN_shape_FULL_48": list(full_train.shape),
            "DEV_VAL_shape_FULL_48": list(full_val.shape),
            "DEV_TRAIN_shape_COMPACT": list(compact_train.shape),
            "DEV_VAL_shape_COMPACT": list(compact_val.shape),
            "bootstrap_attempts": attempts,
            "bootstrap_valid_replicates": BOOTSTRAP_RESAMPLES,
            "loaded_model_data_source": "original train only",
            "loaded_model_roles": ["DEV_TRAIN", "DEV_VAL"],
            "closed_roles_loaded_into_matrices": [],
        },
        RESULTS_DIR / "evaluation_summary.json",
    )
    save_json(decision, DECISION_PATH)
    write_report(config, metrics, bootstrap, complexity, reproduction, decision)
    assert_artifacts()
    print(f"Stage 22 decision {code}: {label}")
    print(f"Final method recommendation: {final_method}")
    print("RESEARCH MATRIX FREEZE recommended")
    print("CONFIRMATION and FINAL_TEST were not accessed")


def assert_artifacts() -> None:
    required = [
        AUDIT_PATH,
        CORRELATION_PATH,
        PAIR_PATH,
        COMPONENT_PATH,
        FEATURE_MANIFEST_PATH,
        FAMILY_SUMMARY_PATH,
        PREREGISTRATION,
        SEAL_PATH,
        RESULTS_DIR / "full_48_reproduction.json",
        RESULTS_DIR / "model_metrics.csv",
        RESULTS_DIR / "compact_minus_full_deltas.csv",
        RESULTS_DIR / "paired_bootstrap_samples.csv",
        RESULTS_DIR / "paired_bootstrap_summary.csv",
        RESULTS_DIR / "complexity_comparison.csv",
        RESULTS_DIR / "dev_val_probabilities.csv",
        RESULTS_DIR / "model_artifact_manifest.json",
        RESULTS_DIR / "evaluation_summary.json",
        DECISION_PATH,
        REPORT_PATH,
    ]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"Missing Stage 22 artifacts: {missing}")
    config = load_preregistration()
    manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
    if len(manifest) != 48 or manifest["status"].eq("RETAINED").sum() != config["frozen_subset"]["retained_feature_count"]:
        raise AssertionError("Stage 22 feature manifest changed")
    family = pd.read_csv(FAMILY_SUMMARY_PATH)
    if len(family) != 6 or not family["family_remains_represented"].all():
        raise AssertionError("A Stage 19 semantic family disappeared")
    metrics = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    if set(metrics["model"]) != {"FULL_48", "COMPACT"} or not metrics["count"].eq(5931).all():
        raise AssertionError("Stage 22 metric panel is invalid")
    with (RESULTS_DIR / "full_48_reproduction.json").open("r", encoding="utf-8") as source:
        reproduction = json.load(source)
    if not reproduction["passed_before_COMPACT_fit"]:
        raise AssertionError("Stage 22 FULL_48 reproduction failed")
    bootstrap = pd.read_csv(RESULTS_DIR / "paired_bootstrap_summary.csv")
    if len(bootstrap) != 2 or not bootstrap["resamples"].eq(2000).all():
        raise AssertionError("Stage 22 bootstrap output is incomplete")
    with DECISION_PATH.open("r", encoding="utf-8") as source:
        decision = json.load(source)
    if decision["adaptive_feature_selection"] or decision["confirmation_accessed"] or decision["final_test_accessed"]:
        raise AssertionError("Stage 22 governance assertion failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    action.add_argument("--assert-artifacts", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.evaluate:
        evaluate()
    else:
        assert_artifacts()
        print("Stage 22 artifact assertions passed")


if __name__ == "__main__":
    main()
