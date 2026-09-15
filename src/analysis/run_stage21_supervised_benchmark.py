"""Stage 21: independent strong supervised benchmark on frozen customer features.

Run ``--prepare`` before ``--evaluate``. Only original-train customers assigned
DEV_TRAIN or DEV_VAL are loaded. CONFIRMATION and FINAL_TEST have no model-data
loading path in this module.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_21"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_21"
AUDIT_PATH = RESULTS_DIR / "pre_stage_audit.json"
ENVIRONMENT_PATH = RESULTS_DIR / "environment_package_audit.json"
PREREGISTRATION = RESULTS_DIR / "preregistered_benchmark_specification.json"
SEAL_PATH = RESULTS_DIR / "preregistration_seal.json"
DECISION_PATH = RESULTS_DIR / "stage_21_decision.json"
REPORT_PATH = RESULTS_DIR / "stage_21_report.md"

STAGE19_DECISION = RESULTS_DIR.parent / "stage_19" / "family_ablation_decision.json"
STAGE19_REPORT = RESULTS_DIR.parent / "stage_19" / "stage_19_report.md"
STAGE20B_DECISION = RESULTS_DIR.parent / "stage_20b" / "stage_20b_decision.json"
STAGE20B_REPORT = RESULTS_DIR.parent / "stage_20b" / "stage_20b_report.md"
STAGE17B_METRICS = RESULTS_DIR.parent / "stage_17b" / "dev_val_metrics.csv"

BASE_HEAD = "17ef49343f094a72fbf48bf27282f149b25b8315"
EXPECTED_STAGE17B_CONFIG_HASH = "605eb1f482123a24a14c0929c951f400c37b3fbc2d2bb2e4d46d2ce834698046"
EXPECTED_STAGE17B_SUMMARY_HASH = "560e8602ce0d572a2e305efcebe525ef3f1174aad0e6f5a3209c2b33fff4865a"
EXPECTED_STAGE17B_DECISION_HASH = "11e15f18e2521ffed8e7dbaacee469284ee356acedea1d8f34ffcf48785a68fa"
EXPECTED_STAGE17B_PREDICTIONS_HASH = "d98a7551a763a5f1e78193786012d86a96c593a73d770fa625a0e09aaac71aea"
EXPECTED_STAGE17B_METRICS_HASH = "9deeb02106b361857b9c4eead2bf9cdd25affd232e4848275129182341a7b083"

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
LOGISTIC_PARAMETERS: dict[str, Any] = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 300,
    "class_weight": None,
    "random_state": 42,
}
RANDOM_FOREST_PARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": None,
    "n_jobs": -1,
    "random_state": 42,
}
LIGHTGBM_PARAMETERS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "class_weight": None,
    "random_state": 42,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}
MODEL_ORDER = ("LOGISTIC_REGRESSION", "RANDOM_FOREST", "HGB", "LIGHTGBM")
NON_HGB_ORDER = ("LOGISTIC_REGRESSION", "RANDOM_FOREST", "LIGHTGBM")
ENSEMBLE_NAME = "BOOST_MEAN_HGB_LIGHTGBM"
FIXED_THRESHOLD = 0.5
RANDOM_SEED = 42
BOOTSTRAP_RESAMPLES = 2000
ROC_EQUIVALENCE_MARGIN = 0.01
AP_EQUIVALENCE_MARGIN = 0.02
REPRODUCTION_TOLERANCE = 1e-12


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def input_records() -> dict[str, dict[str, str]]:
    expected = {
        "stage17a_manifest": (PROTOCOL_MANIFEST, EXPECTED_PROTOCOL_HASH),
        "historical_split": (HISTORICAL_SPLIT_MANIFEST, EXPECTED_HISTORICAL_SPLIT_HASH),
        "stage16_feature_specification": (STAGE16_CONFIG, EXPECTED_STAGE16_CONFIG_HASH),
        "stage17b_configuration": (STAGE17B_CONFIG, EXPECTED_STAGE17B_CONFIG_HASH),
        "stage17b_training_summary": (STAGE17B_SUMMARY, EXPECTED_STAGE17B_SUMMARY_HASH),
        "stage17b_decision": (STAGE17B_DECISION, EXPECTED_STAGE17B_DECISION_HASH),
        "stage17b_model": (STAGE17B_MODEL, EXPECTED_MODEL_HASH),
        "stage17b_predictions": (STAGE17B_PREDICTIONS, EXPECTED_STAGE17B_PREDICTIONS_HASH),
        "stage17b_metrics": (STAGE17B_METRICS, EXPECTED_STAGE17B_METRICS_HASH),
    }
    records: dict[str, dict[str, str]] = {}
    for name, (path, expected_hash) in expected.items():
        observed = file_sha256(path)
        if observed != expected_hash:
            raise ValueError(f"Frozen input hash changed: {name}")
        records[name] = {"path": relative(path), "sha256": observed}
    for name, path in {
        "stage19_decision": STAGE19_DECISION,
        "stage19_report": STAGE19_REPORT,
        "stage20b_decision": STAGE20B_DECISION,
        "stage20b_report": STAGE20B_REPORT,
    }.items():
        records[name] = {"path": relative(path), "sha256": file_sha256(path)}
    return records


def verify_specs() -> None:
    with STAGE16_CONFIG.open("r", encoding="utf-8") as source:
        stage16 = json.load(source)
    with STAGE17B_CONFIG.open("r", encoding="utf-8") as source:
        stage17b = json.load(source)
    with STAGE17B_DECISION.open("r", encoding="utf-8") as source:
        stage17b_decision = json.load(source)
    with STAGE19_DECISION.open("r", encoding="utf-8") as source:
        stage19 = json.load(source)
    with STAGE20B_DECISION.open("r", encoding="utf-8") as source:
        stage20b = json.load(source)
    if tuple(stage16["feature_names_in_order"]) != FEATURE_NAMES or len(FEATURE_NAMES) != 48:
        raise ValueError("Stage 16 frozen 48-feature schema changed")
    if tuple(stage17b["representation"]["feature_names_in_order"]) != FEATURE_NAMES:
        raise ValueError("Stage 17B representation differs from Stage 16")
    if stage17b["candidate_models"]["hist_gradient_boosting"]["parameters"] != HGB_PARAMETERS:
        raise ValueError("Stage 17B HGB configuration changed")
    if stage17b["candidate_models"]["lightgbm"]["parameters"] != LIGHTGBM_PARAMETERS:
        raise ValueError("Stage 17B LightGBM configuration changed")
    if stage17b_decision["selected_model_sha256"] != EXPECTED_MODEL_HASH:
        raise ValueError("Stage 17B selected-model hash changed")
    if stage19["decision"] != "B":
        raise ValueError("Unexpected Stage 19 family-ablation conclusion")
    if stage20b["decision"] != "E":
        raise ValueError("Unexpected Stage 20B conclusion")


def environment_audit() -> dict[str, Any]:
    import importlib.util

    availability = {
        "scikit_learn": {"available": True, "version": sklearn.__version__},
        "lightgbm": {"available": True, "version": lightgbm.__version__},
        "xgboost": {
            "available": bool(importlib.util.find_spec("xgboost")),
            "included": False,
            "reason": "Absent from the established environment; installing another compiled boosting runtime is not a lightweight requirement for this bounded panel.",
        },
        "catboost": {
            "available": bool(importlib.util.find_spec("catboost")),
            "included": False,
            "reason": "Absent from the established environment; installing another large compiled boosting runtime is not a lightweight requirement for this bounded panel.",
        },
    }
    if availability["xgboost"]["available"] or availability["catboost"]["available"]:
        raise ValueError("Package availability changed after the pre-stage environment audit")
    return {
        "stage": "Stage 21 pre-scoring environment/package audit",
        "packages": availability,
        "environment_modified": False,
        "package_installation_performed": False,
        "included_panel": list(MODEL_ORDER),
        "included_fixed_ensemble": ENSEMBLE_NAME,
    }


def governance_audit(records: dict[str, dict[str, str]]) -> dict[str, Any]:
    governance = pd.read_csv(PROTOCOL_MANIFEST, dtype={ID_COLUMN: str})
    development = governance.loc[
        governance["research_role"].isin(["DEV_TRAIN", "DEV_VAL"])
    ].copy()
    if not development["original_split"].eq("train").all():
        raise ValueError("Development governance includes a protected original split")
    ids = {
        role: set(
            development.loc[
                development["research_role"].eq(role)
                & development["stage16_representation_available"].eq(True),
                ID_COLUMN,
            ]
        )
        for role in ("DEV_TRAIN", "DEV_VAL")
    }
    overlap = ids["DEV_TRAIN"] & ids["DEV_VAL"]
    if overlap or len(ids["DEV_TRAIN"]) != 23724 or len(ids["DEV_VAL"]) != 5931:
        raise ValueError("Development membership audit failed")
    return {
        "stage": "Stage 21 pre-scoring audit",
        "base_head": BASE_HEAD,
        "git_clean_before_stage": True,
        "frozen_inputs": records,
        "feature_count": 48,
        "features_in_order": list(FEATURE_NAMES),
        "DEV_TRAIN_customers_with_representation": len(ids["DEV_TRAIN"]),
        "DEV_VAL_customers_with_representation": len(ids["DEV_VAL"]),
        "DEV_TRAIN_DEV_VAL_overlap": len(overlap),
        "stage17b_expected_DEV_VAL": {
            "ROC_AUC": 0.7888308218953116,
            "AP": 0.35979459032174654,
            "maximum_reproduction_tolerance": REPRODUCTION_TOLERANCE,
        },
        "confirmation_accessed": False,
        "final_test_accessed": False,
    }


def prepare() -> None:
    if PREREGISTRATION.exists() or SEAL_PATH.exists():
        raise FileExistsError("Refusing to overwrite frozen Stage 21 preregistration")
    verify_specs()
    records = input_records()
    environment = environment_audit()
    audit = governance_audit(records)
    save_json(environment, ENVIRONMENT_PATH)
    save_json(audit, AUDIT_PATH)
    config = {
        "stage": "Stage 21 — independent strong supervised benchmark",
        "base_head_before_stage": BASE_HEAD,
        "created_before_any_new_benchmark_DEV_VAL_scoring": True,
        "research_question": "Is the confirmed AMP customer-level behavioural result competitive against independently specified strong conventional supervised learners under an exact representation-matched comparison?",
        "scientific_independence": "No published pipeline is reproduced. Literature motivates only the inclusion of conventional strong tabular learners.",
        "audit": {"path": relative(AUDIT_PATH), "sha256": file_sha256(AUDIT_PATH)},
        "environment_audit": {
            "path": relative(ENVIRONMENT_PATH),
            "sha256": file_sha256(ENVIRONMENT_PATH),
        },
        "frozen_inputs": records,
        "data_roles": {
            "fit": "DEV_TRAIN only",
            "evaluate_once": "DEV_VAL only",
            "closed": ["CONFIRMATION", "FINAL_TEST"],
        },
        "representation": {
            "feature_count": 48,
            "ordered_names": list(FEATURE_NAMES),
            "same_matrix_for_every_main_model": True,
            "new_feature_engineering": False,
            "feature_selection": False,
            "resampling_or_synthetic_data": False,
        },
        "models": {
            "LOGISTIC_REGRESSION": {
                "implementation": "sklearn Pipeline(StandardScaler, LogisticRegression)",
                "parameters": LOGISTIC_PARAMETERS,
                "scaling": "StandardScaler fitted on DEV_TRAIN customers only",
            },
            "RANDOM_FOREST": {
                "implementation": "sklearn RandomForestClassifier",
                "parameters": RANDOM_FOREST_PARAMETERS,
                "justification": "A conventional moderate 300-tree bagging reference with sqrt feature subsampling and minimum leaf size 5; fixed independently of DEV_VAL.",
            },
            "HGB": {
                "implementation": "sklearn HistGradientBoostingClassifier",
                "parameters": HGB_PARAMETERS,
                "status": "Exact frozen Stage 17B reference configuration",
            },
            "LIGHTGBM": {
                "implementation": "lightgbm LGBMClassifier",
                "parameters": LIGHTGBM_PARAMETERS,
                "status": "Exact frozen Stage 17B configuration",
            },
        },
        "excluded_packages": {
            "XGBOOST": environment["packages"]["xgboost"],
            "CATBOOST": environment["packages"]["catboost"],
        },
        "ensemble": {
            "name": ENSEMBLE_NAME,
            "members": ["HGB", "LIGHTGBM"],
            "score": "arithmetic mean of the two probabilities",
            "fallback": "Skip only if exact HGB reproduction or LightGBM fitting fails for a documented implementation reason; do not replace members.",
            "learned_weights_or_meta_model": False,
        },
        "reproduction_stop_gate": {
            "model": "HGB",
            "reference_predictions": relative(STAGE17B_PREDICTIONS),
            "expected_ROC_AUC": 0.7888308218953116,
            "expected_AP": 0.35979459032174654,
            "maximum_absolute_probability_or_metric_difference": REPRODUCTION_TOLERANCE,
            "action_on_failure": "Abort before fitting or interpreting any new benchmark model.",
        },
        "metrics": {
            "primary": ["ROC_AUC", "PR_AUC_average_precision"],
            "secondary": ["precision", "recall", "F1", "TN", "FP", "FN", "TP"],
            "secondary_threshold": FIXED_THRESHOLD,
            "threshold_status": "Fixed probability 0.5 for every model; descriptive only; never drives selection.",
            "complexity": ["serialized model bytes", "meaningful parameter count", "tree and leaf counts"],
            "runtime": "Fit seconds and median of five post-warmup full-DEV_VAL predict_proba timings.",
        },
        "primary_deltas": "Every model and the fixed ensemble minus HGB for ROC-AUC and AP.",
        "bootstrap_selection": {
            "individual_AP": "Highest DEV_VAL AP among LOGISTIC_REGRESSION, RANDOM_FOREST, LIGHTGBM; ties use NON_HGB_ORDER.",
            "individual_ROC": "Highest DEV_VAL ROC-AUC among the same candidates; include separately only if different; ties use NON_HGB_ORDER.",
            "ensemble": ENSEMBLE_NAME,
            "delta_direction": "selected benchmark or ensemble minus HGB",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": RANDOM_SEED,
            "unit": "paired DEV_VAL customer",
            "single_class_policy": "discard and continue deterministically",
            "reported": ["observed delta", "mean delta", "95% percentile interval", "fraction below zero", "fraction above zero"],
        },
        "materiality": {
            "ROC_margin": ROC_EQUIVALENCE_MARGIN,
            "AP_margin": AP_EQUIVALENCE_MARGIN,
            "materially_stronger": "Both observed deltas and both paired 95% lower bounds exceed zero, and observed ROC delta >= 0.01 or AP delta >= 0.02.",
            "materially_weaker": "Both observed deltas and both paired 95% upper bounds are below zero, and observed ROC delta <= -0.01 or AP delta <= -0.02.",
            "boosting_equivalent": "Absolute observed HGB delta is below 0.01 ROC and 0.02 AP and neither model is materially stronger.",
        },
        "decision_policy": {
            "D": "The fixed ensemble is materially stronger than HGB.",
            "C": "D is false and a selected individual benchmark is materially stronger than HGB.",
            "A": "D/C are false and HGB is materially stronger than both selected strongest individual references.",
            "B": "D/C/A are false and LightGBM is boosting-equivalent to HGB.",
            "E": "None of D/C/A/B applies.",
            "evaluation_order": ["D", "C", "A", "B", "E"],
        },
        "next_step_policy": {
            "RUN_ONE_FINAL_COMPACT_REPRESENTATION_EXPERIMENT": "Only for decision B together with committed Stage 19 decision B, because classifier equivalence plus family redundancy jointly justify testing a pre-specified simplification.",
            "FREEZE_RESEARCH_MATRIX_NOW": "For every other outcome unless a serious validity failure is discovered.",
        },
        "prohibited_adaptations": [
            "No confirmation/final-test access, copied literature pipeline, new feature engineering, feature selection, hyperparameter search, class resampling, SMOTE, GAN, learned stacking, learned ensemble weights, threshold hunting, or post-hoc model changes.",
            "Do not modify the frozen 48-feature representation based on Stage 19.",
            "Do not attempt another label-formulation experiment.",
        ],
        "random_seeds": {"models": 42, "bootstrap": 42},
    }
    save_json(config, PREREGISTRATION)
    save_json(
        {
            "preregistration_path": relative(PREREGISTRATION),
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "audit_sha256": file_sha256(AUDIT_PATH),
            "environment_audit_sha256": file_sha256(ENVIRONMENT_PATH),
            "frozen_before_DEV_VAL_scoring": True,
        },
        SEAL_PATH,
    )
    print(f"Frozen Stage 21 preregistration: {file_sha256(PREREGISTRATION)}")
    print("No new benchmark DEV_VAL scores were calculated during --prepare")


def load_preregistration() -> dict[str, Any]:
    if not PREREGISTRATION.is_file() or not SEAL_PATH.is_file():
        raise FileNotFoundError("Run --prepare before --evaluate")
    with SEAL_PATH.open("r", encoding="utf-8") as source:
        seal = json.load(source)
    for path, key in (
        (PREREGISTRATION, "preregistration_sha256"),
        (AUDIT_PATH, "audit_sha256"),
        (ENVIRONMENT_PATH, "environment_audit_sha256"),
    ):
        if file_sha256(path) != seal[key]:
            raise ValueError(f"Sealed Stage 21 file changed: {relative(path)}")
    with PREREGISTRATION.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not config.get("created_before_any_new_benchmark_DEV_VAL_scoring"):
        raise ValueError("Preregistration timing declaration is invalid")
    for record in config["frozen_inputs"].values():
        if file_sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen input changed: {record['path']}")
    return config


def make_models() -> dict[str, Any]:
    return {
        "LOGISTIC_REGRESSION": Pipeline(
            [
                ("standardize", StandardScaler()),
                ("classifier", LogisticRegression(**LOGISTIC_PARAMETERS)),
            ]
        ),
        "RANDOM_FOREST": RandomForestClassifier(**RANDOM_FOREST_PARAMETERS),
        "HGB": HistGradientBoostingClassifier(**HGB_PARAMETERS),
        "LIGHTGBM": LGBMClassifier(**LIGHTGBM_PARAMETERS),
    }


def timed_inference(predict: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    scores = np.asarray(predict(), dtype=np.float64)
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        repeated = np.asarray(predict(), dtype=np.float64)
        timings.append(time.perf_counter() - started)
        if not np.allclose(scores, repeated, rtol=0, atol=1e-12):
            raise ValueError("Repeated inference changed model probabilities materially")
    return scores, float(np.median(timings))


def metric_row(name: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return {
        "model": name,
        "evaluation_role": "DEV_VAL",
        **classification_metrics(labels, scores, FIXED_THRESHOLD),
    }


def model_complexity(name: str, model: Any) -> dict[str, Any]:
    parameter_count: int | None = None
    tree_count: int | None = None
    leaf_count: int | None = None
    if name == "LOGISTIC_REGRESSION":
        classifier = model.named_steps["classifier"]
        parameter_count = int(classifier.coef_.size + classifier.intercept_.size)
    elif name == "RANDOM_FOREST":
        tree_count = len(model.estimators_)
        leaf_count = int(sum(tree.tree_.n_leaves for tree in model.estimators_))
    elif name == "HGB":
        tree_count = int(sum(len(group) for group in model._predictors))
        leaf_count = int(
            sum(
                np.sum(tree.nodes["is_leaf"])
                for group in model._predictors
                for tree in group
            )
        )
    elif name == "LIGHTGBM":
        tree_count = int(model.booster_.num_trees())
        dump = model.booster_.dump_model()

        def count_leaves(node: dict[str, Any]) -> int:
            if "leaf_index" in node:
                return 1
            return count_leaves(node["left_child"]) + count_leaves(node["right_child"])

        leaf_count = int(sum(count_leaves(tree["tree_structure"]) for tree in dump["tree_info"]))
    return {
        "meaningful_parameter_count": parameter_count,
        "tree_count": tree_count,
        "leaf_count": leaf_count,
    }


def save_model(name: str, model: Any) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name.lower()}.joblib"
    joblib.dump(model, path)
    return {
        "model": name,
        "path": relative(path),
        "sha256": file_sha256(path),
        "serialized_bytes": path.stat().st_size,
        **model_complexity(name, model),
    }


def primary_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    reference = metrics.loc[metrics["model"].eq("HGB")].iloc[0]
    rows = []
    for row in metrics.itertuples():
        rows.append(
            {
                "model": row.model,
                "comparator": "HGB",
                "ROC_AUC_delta_model_minus_HGB": float(row.ROC_AUC - reference["ROC_AUC"]),
                "AP_delta_model_minus_HGB": float(
                    row.PR_AUC_average_precision - reference["PR_AUC_average_precision"]
                ),
            }
        )
    return pd.DataFrame(rows)


def select_bootstrap_models(metrics: pd.DataFrame) -> list[str]:
    indexed = metrics.set_index("model")
    by_ap = max(
        NON_HGB_ORDER,
        key=lambda name: (indexed.loc[name, "PR_AUC_average_precision"], -NON_HGB_ORDER.index(name)),
    )
    by_roc = max(
        NON_HGB_ORDER,
        key=lambda name: (indexed.loc[name, "ROC_AUC"], -NON_HGB_ORDER.index(name)),
    )
    selected = [by_ap]
    if by_roc != by_ap:
        selected.append(by_roc)
    selected.append(ENSEMBLE_NAME)
    return selected


def paired_bootstrap(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    selected: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    attempts = 0
    replicate = 0
    while replicate < BOOTSTRAP_RESAMPLES:
        attempts += 1
        indices = rng.integers(0, len(labels), len(labels))
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        replicate += 1
        hgb = scores["HGB"][indices]
        for name in selected:
            candidate = scores[name][indices]
            rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "model": name,
                    "comparator": "HGB",
                    "ROC_AUC_delta_model_minus_HGB": float(
                        roc_auc_score(y, candidate) - roc_auc_score(y, hgb)
                    ),
                    "AP_delta_model_minus_HGB": float(
                        average_precision_score(y, candidate)
                        - average_precision_score(y, hgb)
                    ),
                }
            )
    samples = pd.DataFrame(rows)
    summary_rows = []
    for name in selected:
        subset = samples.loc[samples["model"].eq(name)]
        for metric, column in (
            ("ROC_AUC", "ROC_AUC_delta_model_minus_HGB"),
            ("AP", "AP_delta_model_minus_HGB"),
        ):
            values = subset[column].to_numpy(dtype=np.float64)
            observed = (
                roc_auc_score(labels, scores[name]) - roc_auc_score(labels, scores["HGB"])
                if metric == "ROC_AUC"
                else average_precision_score(labels, scores[name])
                - average_precision_score(labels, scores["HGB"])
            )
            summary_rows.append(
                {
                    "model": name,
                    "comparator": "HGB",
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


def materially_stronger(summary: pd.DataFrame, name: str) -> bool:
    subset = summary.loc[summary["model"].eq(name)].set_index("metric")
    roc, ap = subset.loc["ROC_AUC"], subset.loc["AP"]
    return bool(
        roc["observed_delta"] > 0
        and ap["observed_delta"] > 0
        and roc["CI_2_5"] > 0
        and ap["CI_2_5"] > 0
        and (
            roc["observed_delta"] >= ROC_EQUIVALENCE_MARGIN
            or ap["observed_delta"] >= AP_EQUIVALENCE_MARGIN
        )
    )


def hgb_materially_stronger(summary: pd.DataFrame, individuals: list[str]) -> bool:
    for name in individuals:
        subset = summary.loc[summary["model"].eq(name)].set_index("metric")
        roc, ap = subset.loc["ROC_AUC"], subset.loc["AP"]
        if not (
            roc["observed_delta"] < 0
            and ap["observed_delta"] < 0
            and roc["CI_97_5"] < 0
            and ap["CI_97_5"] < 0
            and (
                roc["observed_delta"] <= -ROC_EQUIVALENCE_MARGIN
                or ap["observed_delta"] <= -AP_EQUIVALENCE_MARGIN
            )
        ):
            return False
    return True


def decide(
    metrics: pd.DataFrame, bootstrap: pd.DataFrame, selected: list[str]
) -> tuple[str, str, dict[str, Any]]:
    individual_selected = [name for name in selected if name != ENSEMBLE_NAME]
    ensemble_stronger = materially_stronger(bootstrap, ENSEMBLE_NAME)
    stronger_individuals = [
        name for name in individual_selected if materially_stronger(bootstrap, name)
    ]
    hgb_clear = hgb_materially_stronger(bootstrap, individual_selected)
    indexed = metrics.set_index("model")
    lightgbm_equivalent = bool(
        abs(indexed.loc["LIGHTGBM", "ROC_AUC"] - indexed.loc["HGB", "ROC_AUC"])
        < ROC_EQUIVALENCE_MARGIN
        and abs(
            indexed.loc["LIGHTGBM", "PR_AUC_average_precision"]
            - indexed.loc["HGB", "PR_AUC_average_precision"]
        )
        < AP_EQUIVALENCE_MARGIN
        and not ensemble_stronger
        and not stronger_individuals
    )
    if ensemble_stronger:
        code, label = "D", "SIMPLE PREREGISTERED ENSEMBLE MATERIALLY STRONGER"
    elif stronger_individuals:
        code, label = "C", "ANOTHER INDIVIDUAL LEARNER MATERIALLY STRONGER"
    elif hgb_clear:
        code, label = "A", "HGB REMAINS CLEARLY STRONG"
    elif lightgbm_equivalent:
        code, label = "B", "STRONG CONVENTIONAL LEARNERS ESSENTIALLY EQUIVALENT"
    else:
        code, label = "E", "BENCHMARK INCONCLUSIVE"
    evidence = {
        "bootstrap_selected_models": selected,
        "materially_stronger_individuals": stronger_individuals,
        "ensemble_materially_stronger": ensemble_stronger,
        "HGB_materially_stronger_than_selected_individuals": hgb_clear,
        "LightGBM_within_frozen_equivalence_margins": lightgbm_equivalent,
    }
    return code, label, evidence


def markdown_metrics(frame: pd.DataFrame) -> str:
    return "\n".join(
        f"| {row.model} | {row.ROC_AUC:.6f} | {row.PR_AUC_average_precision:.6f} | "
        f"{row.precision:.6f} | {row.recall:.6f} | {row.F1:.6f} | "
        f"{row.TN}/{row.FP}/{row.FN}/{row.TP} |"
        for row in frame.itertuples()
    )


def markdown_complexity(frame: pd.DataFrame) -> str:
    def value(item: Any) -> str:
        return "—" if pd.isna(item) else str(int(item))

    return "\n".join(
        f"| {row.model} | {row.fit_seconds:.6f} | {row.inference_seconds:.6f} | "
        f"{row.serialized_bytes} | {value(row.meaningful_parameter_count)} | "
        f"{value(row.tree_count)} | {value(row.leaf_count)} |"
        for row in frame.itertuples()
    )


def markdown_bootstrap(frame: pd.DataFrame) -> str:
    return "\n".join(
        f"| {row.model} | {row.metric} | {row.observed_delta:+.6f} | "
        f"{row.bootstrap_mean_delta:+.6f} | [{row.CI_2_5:+.6f}, {row.CI_97_5:+.6f}] | "
        f"{row.fraction_below_zero:.4f} | {row.fraction_above_zero:.4f} |"
        for row in frame.itertuples()
    )


def write_report(
    metrics: pd.DataFrame,
    complexity: pd.DataFrame,
    bootstrap: pd.DataFrame,
    reproduction: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    text = f"""# Stage 21 — Independent Strong Supervised Benchmark

## Question and governance

Stage 21 tests whether the frozen 48-feature AMP customer representation remains competitive across a minimal, independently specified conventional learner panel. Every main model used identical DEV_TRAIN and DEV_VAL customers, features, and labels. No published pipeline was copied. No feature/model co-adaptation, resampling, or tuning occurred.

The machine-readable preregistration is `{relative(PREREGISTRATION)}` with SHA-256 `{file_sha256(PREREGISTRATION)}`. DEV_TRAIN contained 23,724 customers (2,022 theft); DEV_VAL contained 5,931 customers (505 theft); overlap was zero.

## Environment and panel

scikit-learn `{sklearn.__version__}` and LightGBM `{lightgbm.__version__}` were already available. XGBoost and CatBoost were absent and were excluded without modifying the environment. The frozen panel comprised logistic regression, random forest, HGB, LightGBM, and the fixed arithmetic HGB/LightGBM probability mean.

## HGB reproduction gate

The Stage 17B HGB was reconstructed before new models were interpreted. Maximum probability difference was `{reproduction['maximum_absolute_probability_difference']:.3g}`; ROC-AUC/AP differences were `{reproduction['ROC_AUC_difference']:+.3g}`/`{reproduction['AP_difference']:+.3g}`. The `1e-12` stop gate **passed**.

## Complete benchmark

All rows contain 5,931 customers, 505 theft customers, and prevalence 0.085146. Threshold metrics use fixed probability 0.5 and are descriptive only.

| Model | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---|
{markdown_metrics(metrics)}

## Complexity and runtime

Inference time is the median of five post-warmup full-DEV_VAL probability passes. Parameter counts are reported only where meaningful; tree and leaf counts describe tree ensembles.

| Model | Fit seconds | Inference seconds | Serialized bytes | Parameters | Trees | Leaves |
|---|---:|---:|---:|---:|---:|---:|
{markdown_complexity(complexity)}

## Paired uncertainty

The pre-registered rule selected the strongest non-HGB individual by AP and separately by ROC-AUC if different, plus the fixed ensemble. Deltas are model minus HGB over 2,000 paired DEV_VAL customer replicates with seed 42.

| Model | Metric | Observed delta | Bootstrap mean | 95% interval | Fraction below 0 | Fraction above 0 |
|---|---|---:|---:|---|---:|---:|
{markdown_bootstrap(bootstrap)}

## Decision

Decision **{decision['decision']} — {decision['decision_label']}**.

No individual benchmark materially beat HGB under the frozen joint effect-size and uncertainty rule: `{decision['evidence']['materially_stronger_individuals']}`. The fixed ensemble materially beat HGB: **{decision['evidence']['ensemble_materially_stronger']}**. LightGBM fell within the frozen equivalence margins: **{decision['evidence']['LightGBM_within_frozen_equivalence_margins']}**.

## Paper and next step

{decision['paper_implication']}

Literature is used only to motivate testing strong tabular boosting models; these results do not establish protocol-equivalent superiority over published SGCC studies and are not a state-of-the-art claim.

Next step: **{decision['next_step']}**. {decision['next_step_reason']}

No adaptive tuning or post-hoc model change occurred. The frozen Stage 17B detector and 48-feature representation were not modified.

**CONFIRMATION NOT ACCESSED DURING STAGE 21.**

**FINAL_TEST NOT ACCESSED DURING STAGE 21.**
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")


def evaluate() -> None:
    if DECISION_PATH.exists():
        raise FileExistsError("Refusing to repeat completed Stage 21 evaluation")
    config = load_preregistration()
    verify_specs()
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
        raise ValueError("DEV_TRAIN-derived reference statistics differ from Stage 17B")
    train_X = train_features.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    val_X = val_features.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    if train_X.shape != (23724, 48) or val_X.shape != (5931, 48):
        raise ValueError("Frozen customer matrix shape changed")

    saved = pd.read_csv(STAGE17B_PREDICTIONS, dtype={ID_COLUMN: str})
    saved = saved.loc[
        saved["model"].eq("hist_gradient_boosting"),
        [ID_COLUMN, LABEL_COLUMN, "probability"],
    ].set_index(ID_COLUMN).reindex(val_ids)
    if saved.isna().any().any() or not np.array_equal(
        saved[LABEL_COLUMN].to_numpy(dtype=np.int8), val_labels
    ):
        raise ValueError("Saved Stage 17B predictions do not align with DEV_VAL")
    saved_scores = saved["probability"].to_numpy(dtype=np.float64)

    models = make_models()
    scores: dict[str, np.ndarray] = {}
    model_records = []
    runtime_rows = []

    # The HGB reproduction gate is deliberately completed before new models fit.
    started = time.perf_counter()
    hgb = models["HGB"]
    hgb.fit(train_X, train_labels)
    hgb_fit = time.perf_counter() - started
    hgb_scores, hgb_inference = timed_inference(lambda: hgb.predict_proba(val_X)[:, 1])
    hgb_roc = roc_auc_score(val_labels, hgb_scores)
    hgb_ap = average_precision_score(val_labels, hgb_scores)
    reproduction = {
        "stage": "Stage 21 HGB reproduction stop gate",
        "passed_before_new_benchmark_fit": bool(
            np.max(np.abs(hgb_scores - saved_scores)) <= REPRODUCTION_TOLERANCE
            and abs(hgb_roc - config["reproduction_stop_gate"]["expected_ROC_AUC"])
            <= REPRODUCTION_TOLERANCE
            and abs(hgb_ap - config["reproduction_stop_gate"]["expected_AP"])
            <= REPRODUCTION_TOLERANCE
        ),
        "maximum_absolute_probability_difference": float(np.max(np.abs(hgb_scores - saved_scores))),
        "ROC_AUC": float(hgb_roc),
        "AP": float(hgb_ap),
        "ROC_AUC_difference": float(hgb_roc - config["reproduction_stop_gate"]["expected_ROC_AUC"]),
        "AP_difference": float(hgb_ap - config["reproduction_stop_gate"]["expected_AP"]),
        "tolerance": REPRODUCTION_TOLERANCE,
    }
    save_json(reproduction, RESULTS_DIR / "hgb_reproduction.json")
    if not reproduction["passed_before_new_benchmark_fit"]:
        raise RuntimeError("HGB failed exact Stage 17B reproduction; aborting benchmark")
    scores["HGB"] = hgb_scores
    hgb_record = save_model("HGB", hgb)
    model_records.append(hgb_record)
    runtime_rows.append(
        {"model": "HGB", "fit_seconds": hgb_fit, "inference_seconds": hgb_inference, **hgb_record}
    )

    for name in ("LOGISTIC_REGRESSION", "RANDOM_FOREST", "LIGHTGBM"):
        model = models[name]
        started = time.perf_counter()
        model.fit(train_X, train_labels)
        fit_seconds = time.perf_counter() - started
        model_scores, inference_seconds = timed_inference(
            lambda model=model: model.predict_proba(val_X)[:, 1]
        )
        scores[name] = model_scores
        record = save_model(name, model)
        model_records.append(record)
        runtime_rows.append(
            {"model": name, "fit_seconds": fit_seconds, "inference_seconds": inference_seconds, **record}
        )

    def ensemble_predict() -> np.ndarray:
        return (
            models["HGB"].predict_proba(val_X)[:, 1]
            + models["LIGHTGBM"].predict_proba(val_X)[:, 1]
        ) / 2.0

    ensemble_scores, ensemble_inference = timed_inference(ensemble_predict)
    scores[ENSEMBLE_NAME] = ensemble_scores
    ensemble_path = MODEL_DIR / f"{ENSEMBLE_NAME.lower()}.joblib"
    joblib.dump(
        {"members": ["HGB", "LIGHTGBM"], "weights": [0.5, 0.5], "learned": False},
        ensemble_path,
    )
    ensemble_record = {
        "model": ENSEMBLE_NAME,
        "path": relative(ensemble_path),
        "sha256": file_sha256(ensemble_path),
        "serialized_bytes": ensemble_path.stat().st_size,
        "meaningful_parameter_count": None,
        "tree_count": None,
        "leaf_count": None,
    }
    model_records.append(ensemble_record)
    runtime_rows.append(
        {
            "model": ENSEMBLE_NAME,
            "fit_seconds": 0.0,
            "inference_seconds": ensemble_inference,
            **ensemble_record,
        }
    )

    metrics = pd.DataFrame(
        [metric_row(name, val_labels, scores[name]) for name in (*MODEL_ORDER, ENSEMBLE_NAME)]
    )
    metrics = metrics[
        [
            "model", "evaluation_role", "count", "normal_count", "theft_count",
            "positive_prevalence", "threshold", "threshold_rule", "ROC_AUC",
            "PR_AUC_average_precision", "precision", "recall", "F1", "TN", "FP", "FN", "TP",
        ]
    ]
    deltas = primary_deltas(metrics)
    selected = select_bootstrap_models(metrics)
    bootstrap_samples, bootstrap_summary, attempts = paired_bootstrap(
        val_labels, scores, selected
    )
    code, label, evidence = decide(metrics, bootstrap_summary, selected)
    if code == "B":
        next_step = "RUN ONE FINAL COMPACT-REPRESENTATION EXPERIMENT"
        next_reason = "Stage 19 found strong feature-family redundancy and Stage 21 finds classifier-level boosting equivalence, jointly motivating one pre-specified simplification test rather than more model hunting."
    else:
        next_step = "FREEZE RESEARCH MATRIX NOW"
        next_reason = "The pre-registered compact-experiment condition was not met."
    if code == "B":
        implication = "HGB and LightGBM cluster within the frozen practical margins, supporting the interpretation that AMP's customer representation, rather than a uniquely favorable classifier, carries the development result."
    elif code == "A":
        implication = "HGB retains a material advantage over the strongest included conventional alternatives under the frozen representation."
    elif code == "C":
        implication = "A preregistered individual conventional learner materially exceeds HGB, so one final model-choice decision should be considered before freeze without changing the representation."
    elif code == "D":
        implication = "The fixed arithmetic boosting ensemble materially exceeds HGB, warranting disclosure and one bounded final model-choice decision before freeze."
    else:
        implication = "The benchmark does not support a clean classifier-family conclusion; the frozen HGB remains the unchanged reference pending matrix freeze."
    decision = {
        "stage": "Stage 21 independent strong supervised benchmark decision",
        "decision": code,
        "decision_label": label,
        "decision_policy": config["decision_policy"],
        "evidence": evidence,
        "paper_implication": implication,
        "literature_claim_boundary": "Motivates strong tabular baselines only; no protocol-equivalent published comparison and no state-of-the-art claim.",
        "next_step": next_step,
        "next_step_reason": next_reason,
        "adaptive_tuning": False,
        "feature_or_representation_change": False,
        "resampling_or_synthetic_data": False,
        "threshold_tuning": False,
        "learned_ensemble": False,
        "frozen_stage17b_detector_modified": False,
        "confirmation_accessed": False,
        "final_test_accessed": False,
    }

    complexity = pd.DataFrame(runtime_rows)[
        [
            "model", "fit_seconds", "inference_seconds", "serialized_bytes",
            "meaningful_parameter_count", "tree_count", "leaf_count", "path", "sha256",
        ]
    ]
    probabilities = pd.concat(
        [
            pd.DataFrame(
                {ID_COLUMN: val_ids, LABEL_COLUMN: val_labels, "model": name, "probability": scores[name]}
            )
            for name in (*MODEL_ORDER, ENSEMBLE_NAME)
        ],
        ignore_index=True,
    )
    save_csv(metrics, RESULTS_DIR / "benchmark_metrics.csv")
    save_csv(deltas, RESULTS_DIR / "model_deltas_vs_hgb.csv")
    save_csv(complexity, RESULTS_DIR / "model_complexity_runtime.csv")
    save_csv(probabilities, RESULTS_DIR / "dev_val_benchmark_probabilities.csv")
    save_csv(bootstrap_samples, RESULTS_DIR / "paired_bootstrap_samples.csv")
    save_csv(bootstrap_summary, RESULTS_DIR / "paired_bootstrap_summary.csv")
    save_json(
        {
            "selection_rule": config["bootstrap_selection"],
            "selected_models": selected,
            "attempts": attempts,
            "valid_replicates": BOOTSTRAP_RESAMPLES,
        },
        RESULTS_DIR / "bootstrap_selection_audit.json",
    )
    save_json({"models": model_records}, RESULTS_DIR / "model_artifact_manifest.json")
    save_json(decision, DECISION_PATH)
    write_report(metrics, complexity, bootstrap_summary, reproduction, decision)
    assert_artifacts()
    print(f"Stage 21 decision {code}: {label}")
    print(f"Next step: {next_step}")
    print("CONFIRMATION and FINAL_TEST were not accessed")


def assert_artifacts() -> None:
    required = [
        AUDIT_PATH,
        ENVIRONMENT_PATH,
        PREREGISTRATION,
        SEAL_PATH,
        RESULTS_DIR / "hgb_reproduction.json",
        RESULTS_DIR / "benchmark_metrics.csv",
        RESULTS_DIR / "model_deltas_vs_hgb.csv",
        RESULTS_DIR / "model_complexity_runtime.csv",
        RESULTS_DIR / "dev_val_benchmark_probabilities.csv",
        RESULTS_DIR / "paired_bootstrap_samples.csv",
        RESULTS_DIR / "paired_bootstrap_summary.csv",
        RESULTS_DIR / "bootstrap_selection_audit.json",
        RESULTS_DIR / "model_artifact_manifest.json",
        DECISION_PATH,
        REPORT_PATH,
    ]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"Missing Stage 21 artifacts: {missing}")
    metrics = pd.read_csv(RESULTS_DIR / "benchmark_metrics.csv")
    if len(metrics) != 5 or set(metrics["model"]) != {*MODEL_ORDER, ENSEMBLE_NAME}:
        raise AssertionError("Stage 21 benchmark panel is incomplete")
    if not metrics["count"].eq(5931).all() or not metrics["theft_count"].eq(505).all():
        raise AssertionError("Stage 21 benchmark population changed")
    with (RESULTS_DIR / "hgb_reproduction.json").open("r", encoding="utf-8") as source:
        reproduction = json.load(source)
    if not reproduction["passed_before_new_benchmark_fit"]:
        raise AssertionError("Stage 21 HGB reproduction did not pass")
    bootstrap = pd.read_csv(RESULTS_DIR / "paired_bootstrap_summary.csv")
    if len(bootstrap) not in {4, 6} or not bootstrap["resamples"].eq(2000).all():
        raise AssertionError("Stage 21 bootstrap summary is incomplete")
    with DECISION_PATH.open("r", encoding="utf-8") as source:
        decision = json.load(source)
    if decision["adaptive_tuning"] or decision["confirmation_accessed"] or decision["final_test_accessed"]:
        raise AssertionError("Stage 21 governance assertion failed")


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
        load_preregistration()
        assert_artifacts()
        print("Stage 21 artifact assertions passed")


if __name__ == "__main__":
    main()
