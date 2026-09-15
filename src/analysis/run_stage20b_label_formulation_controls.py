"""Stage 20B: pre-registered label/objective-formulation controls.

Run ``--prepare`` before ``--evaluate``. The only model-data loader is the
original-train population partitioned into DEV_TRAIN and DEV_VAL. Protected
CONFIRMATION and FINAL_TEST populations have no loading path in this module.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.run_stage14_representation import (
    FEATURE_NAMES as WINDOW_FEATURE_NAMES,
    behavioural_features,
)
from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    classification_metrics,
    file_sha256,
    load_window_matrix,
    save_csv,
    save_json,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_20b"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_20b"
AUDIT_PATH = RESULTS_DIR / "pre_stage_audit.json"
PREREGISTRATION = RESULTS_DIR / "preregistered_control_experiment.json"
SEAL_PATH = RESULTS_DIR / "preregistration_seal.json"
DECISION_PATH = RESULTS_DIR / "stage_20b_decision.json"
REPORT_PATH = RESULTS_DIR / "stage_20b_report.md"

PROTOCOL_MANIFEST = (
    PROJECT_ROOT / "results" / "governance" / "stage_17a" / "research_protocol_manifest.csv"
)
PROTOCOL_SUMMARY = (
    PROJECT_ROOT / "results" / "governance" / "stage_17a" / "protocol_summary.json"
)
HISTORICAL_SPLIT = PROJECT_ROOT / "data" / "processed" / "split_manifest.parquet"
TRAIN_WINDOW_MANIFEST = (
    PROJECT_ROOT / "data" / "windows" / "stride_30" / "usable" / "train_windows.parquet"
)
STAGE14_CONFIG = RESULTS_DIR.parent / "stage_14" / "preregistered_config.json"
STAGE16_CONFIG = RESULTS_DIR.parent / "stage_16" / "preregistered_config.json"
STAGE17B_CONFIG = RESULTS_DIR.parent / "stage_17b" / "preregistered_benchmark_config.json"
STAGE17B_SUMMARY = RESULTS_DIR.parent / "stage_17b" / "training_summary.json"
STAGE17B_DECISION = RESULTS_DIR.parent / "stage_17b" / "validation_decision.json"
STAGE17B_MODEL = PROJECT_ROOT / "models" / "supervised" / "stage_17b" / "hist_gradient_boosting.joblib"
STAGE20A_REPORT = RESULTS_DIR.parent / "stage_20a" / "stage_20a_label_formulation_control_design.md"

BASE_HEAD = "b405a2828588d5365662ab25a1c198d1d7c6ea0d"
EXPECTED_PROTOCOL_HASH = "dae7dba53260a3af6b8e2ddd904b5c7867c3321f27cdff5af5e4060b97c348c0"
EXPECTED_HISTORICAL_SPLIT_HASH = "6ad1132bbaef0c6f6ccc5bdc973844b805c4291612ac7170ec165f273835959c"
EXPECTED_STAGE16_CONFIG_HASH = "a722299d6207fce06180f71bf96717ce58948e127a85440e04f721c1616fcdf3"
EXPECTED_STAGE17B_MODEL_HASH = "67ed869858185d8f71d99b3dce8b358c43793178185e5aa50e6997d11b56974b"

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
LOGIT_PARAMETERS: dict[str, Any] = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 300,
    "class_weight": None,
    "random_state": 42,
}
AGGREGATIONS = ("MEAN", "MEDIAN", "Q90", "MAXIMUM")
FIXED_THRESHOLD = 0.5
BOOTSTRAP_RESAMPLES = 2000
RANDOM_SEED = 42
PRACTICAL_ROC_DELTA = 0.01
PRACTICAL_AP_DELTA = 0.02


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def frozen_input_records() -> dict[str, dict[str, str]]:
    expected = {
        "stage17a_governance_manifest": (PROTOCOL_MANIFEST, EXPECTED_PROTOCOL_HASH),
        "historical_split_manifest": (HISTORICAL_SPLIT, EXPECTED_HISTORICAL_SPLIT_HASH),
        "stage16_feature_specification": (STAGE16_CONFIG, EXPECTED_STAGE16_CONFIG_HASH),
        "stage17b_model": (STAGE17B_MODEL, EXPECTED_STAGE17B_MODEL_HASH),
    }
    records: dict[str, dict[str, str]] = {}
    for name, (path, expected_hash) in expected.items():
        observed = file_sha256(path)
        if observed != expected_hash:
            raise ValueError(f"Frozen input hash changed: {name}")
        records[name] = {"path": relative(path), "sha256": observed}
    for name, path in {
        "stage14_window_feature_specification": STAGE14_CONFIG,
        "stage17b_configuration": STAGE17B_CONFIG,
        "stage17b_training_summary": STAGE17B_SUMMARY,
        "stage17b_decision": STAGE17B_DECISION,
        "stage20a_design": STAGE20A_REPORT,
    }.items():
        records[name] = {"path": relative(path), "sha256": file_sha256(path)}
    return records


def verify_feature_and_model_specifications() -> None:
    with STAGE14_CONFIG.open("r", encoding="utf-8") as source:
        stage14 = json.load(source)
    with STAGE16_CONFIG.open("r", encoding="utf-8") as source:
        stage16 = json.load(source)
    with STAGE17B_CONFIG.open("r", encoding="utf-8") as source:
        stage17b = json.load(source)
    with STAGE17B_DECISION.open("r", encoding="utf-8") as source:
        decision = json.load(source)
    if tuple(stage14["feature_names_in_order"]) != WINDOW_FEATURE_NAMES:
        raise ValueError("Frozen Stage 14 17-feature window schema changed")
    if len(WINDOW_FEATURE_NAMES) != 17:
        raise ValueError("Window feature schema is not exactly 17 features")
    if len(stage16["feature_names_in_order"]) != 48:
        raise ValueError("Frozen Stage 16 customer schema is not exactly 48 features")
    if stage17b["candidate_models"]["hist_gradient_boosting"]["parameters"] != HGB_PARAMETERS:
        raise ValueError("Frozen Stage 17B HGB configuration changed")
    if decision["selected_model_sha256"] != EXPECTED_STAGE17B_MODEL_HASH:
        raise ValueError("Stage 17B decision points to a different model")


def governance_audit() -> dict[str, Any]:
    governance = pd.read_csv(PROTOCOL_MANIFEST, dtype={ID_COLUMN: str})
    development = governance.loc[
        governance["research_role"].isin(["DEV_TRAIN", "DEV_VAL"])
    ].copy()
    if not development["original_split"].eq("train").all():
        raise ValueError("A development role includes a non-original-train customer")
    available = development.loc[development["stage16_representation_available"].eq(True)]
    role_ids = {
        role: set(available.loc[available["research_role"].eq(role), ID_COLUMN])
        for role in ("DEV_TRAIN", "DEV_VAL")
    }
    overlap = role_ids["DEV_TRAIN"] & role_ids["DEV_VAL"]
    if overlap:
        raise ValueError("DEV_TRAIN and DEV_VAL customer membership overlaps")

    window_manifest = pd.read_parquet(
        TRAIN_WINDOW_MANIFEST,
        columns=[ID_COLUMN, LABEL_COLUMN],
        engine="pyarrow",
    )
    roles = window_manifest[ID_COLUMN].astype(str).map(
        dict(zip(development[ID_COLUMN], development["research_role"]))
    )
    if roles.isna().any() or not roles.isin(["DEV_TRAIN", "DEV_VAL"]).all():
        raise ValueError("An original-train window lacks an allowed development role")
    counts = {}
    for role in ("DEV_TRAIN", "DEV_VAL"):
        mask = roles.eq(role).to_numpy()
        role_windows = window_manifest.loc[mask]
        customer_labels = role_windows.groupby(ID_COLUMN, sort=False)[LABEL_COLUMN].first()
        if role_windows.groupby(ID_COLUMN)[LABEL_COLUMN].nunique().max() != 1:
            raise ValueError(f"{role} has inconsistent propagated customer labels")
        if set(customer_labels.index.astype(str)) != role_ids[role]:
            raise ValueError(f"{role} usable-window membership differs from governance")
        counts[role] = {
            "customers": int(len(customer_labels)),
            "normal_customers": int(customer_labels.eq(0).sum()),
            "theft_customers": int(customer_labels.eq(1).sum()),
            "theft_prevalence": float(customer_labels.mean()),
            "usable_windows": int(mask.sum()),
        }
    if counts["DEV_TRAIN"]["customers"] != 23724 or counts["DEV_VAL"]["customers"] != 5931:
        raise ValueError("Development customer counts changed")
    return {
        "stage": "Stage 20B pre-scoring audit",
        "base_head": BASE_HEAD,
        "git_clean_before_stage": True,
        "frozen_inputs": frozen_input_records(),
        "window_feature_count": len(WINDOW_FEATURE_NAMES),
        "window_features_in_order": list(WINDOW_FEATURE_NAMES),
        "development_counts": counts,
        "DEV_TRAIN_DEV_VAL_customer_overlap": len(overlap),
        "loaded_data_scope": "Original train governance and window-manifest audit columns only; no model scoring",
        "historical_stage14_model_reused": False,
        "confirmation_accessed": False,
        "final_test_accessed": False,
    }


def comparison_specifications() -> list[dict[str, str]]:
    comparisons = [
        {
            "comparison": "BALANCED_MEAN_MINUS_UNWEIGHTED_MEAN",
            "candidate": "WINDOW_HGB_CUSTOMER_BALANCED_MEAN",
            "comparator": "WINDOW_HGB_UNWEIGHTED_MEAN",
            "question": "Does equal total customer contribution materially help?",
        }
    ]
    for prefix in ("WINDOW_HGB_UNWEIGHTED", "WINDOW_HGB_CUSTOMER_BALANCED"):
        for aggregation in ("MEDIAN", "Q90", "MAXIMUM"):
            comparisons.append(
                {
                    "comparison": f"{prefix}_{aggregation}_MINUS_{prefix}_MEAN",
                    "candidate": f"{prefix}_{aggregation}",
                    "comparator": f"{prefix}_MEAN",
                    "question": "Is arithmetic-mean aggregation hiding useful window evidence?",
                }
            )
    comparisons.append(
        {
            "comparison": "C_LOGIT_MINUS_W_LOGIT",
            "candidate": "C_LOGIT",
            "comparator": "W_LOGIT",
            "question": "Does placing the loss at customer level help in the matched linear control?",
        }
    )
    return comparisons


def prepare() -> None:
    if PREREGISTRATION.exists() or SEAL_PATH.exists():
        raise FileExistsError("Refusing to overwrite frozen Stage 20B preregistration")
    verify_feature_and_model_specifications()
    audit = governance_audit()
    save_json(audit, AUDIT_PATH)
    config = {
        "stage": "Stage 20B — controlled label/objective-formulation experiment",
        "base_head_before_stage": BASE_HEAD,
        "created_before_any_new_DEV_VAL_scoring": True,
        "research_question": "Does aligning training and scoring more closely with the customer-level label provide an advantage that cannot be explained solely by unequal windows per customer or mean-probability aggregation?",
        "causal_limit": "SGCC has no temporal theft-event truth. These controls cannot identify temporal window-label noise.",
        "audit": {"path": relative(AUDIT_PATH), "sha256": file_sha256(AUDIT_PATH)},
        "frozen_inputs": audit["frozen_inputs"],
        "data_roles": {
            "fit": "DEV_TRAIN only",
            "evaluate_once": "DEV_VAL only",
            "closed": ["CONFIRMATION", "FINAL_TEST"],
        },
        "features": {
            "count": 17,
            "ordered_names": list(WINDOW_FEATURE_NAMES),
            "definition": "Exact frozen Stage 14 behavioural summaries of each usable 30-day window.",
            "new_feature_discovery": False,
        },
        "models": {
            "WINDOW_HGB_UNWEIGHTED": {
                "training_unit": "window",
                "target": "customer FLAG propagated to each window; proxy only",
                "parameters": HGB_PARAMETERS,
                "sample_weight": "1 for every DEV_TRAIN window",
            },
            "WINDOW_HGB_CUSTOMER_BALANCED": {
                "training_unit": "window",
                "target": "customer FLAG propagated to each window; proxy only",
                "parameters": HGB_PARAMETERS,
                "sample_weight": "N_windows / (N_customers * n_i) for every window of customer i; equal total customer weight and mean window weight 1",
            },
            "W_LOGIT": {
                "training_unit": "window",
                "target": "customer FLAG propagated to each window; proxy only",
                "parameters": LOGIT_PARAMETERS,
                "sample_weight": "same customer-balanced weights as WINDOW_HGB_CUSTOMER_BALANCED",
                "customer_score": "arithmetic mean of window probabilities, required by the Stage 20B instruction",
            },
            "C_LOGIT": {
                "training_unit": "customer",
                "target": "one customer FLAG",
                "parameters": LOGIT_PARAMETERS,
                "customer_vector": "arithmetic mean of the shared transformed 17-dimensional window vectors",
                "customer_score": "direct logistic probability",
            },
        },
        "shared_logistic_preprocessing": {
            "definition": "Weighted mean and population variance fitted on DEV_TRAIN windows with customer-balanced weights; identical centering/scaling applied to both arms; zero scales replaced by 1.",
            "fit_scope": "DEV_TRAIN only",
            "stage20a_pooling_operator": "arithmetic mean",
            "note": "The explicit Stage 20B request fixes W_LOGIT scoring as mean probability; therefore exact linear mean-logit equivalence is not claimed.",
        },
        "aggregation_panel": {
            "models": ["WINDOW_HGB_UNWEIGHTED", "WINDOW_HGB_CUSTOMER_BALANCED"],
            "rules": {
                "MEAN": "arithmetic mean of window probabilities",
                "MEDIAN": "median of window probabilities",
                "Q90": "90th percentile with NumPy linear interpolation",
                "MAXIMUM": "maximum window probability",
            },
            "selection": "All four reported equally; no winning rule selected; MEAN is the historical primary reference.",
        },
        "metrics": {
            "primary": ["ROC_AUC", "PR_AUC_average_precision"],
            "secondary": ["precision", "recall", "F1", "TN", "FP", "FN", "TP"],
            "secondary_threshold": FIXED_THRESHOLD,
            "threshold_status": "Fixed at 0.5 before scoring; descriptive only; not used for the formulation decision.",
            "report_counts_and_prevalence": True,
        },
        "comparisons": comparison_specifications(),
        "bootstrap": {
            "comparisons": [row["comparison"] for row in comparison_specifications()],
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": RANDOM_SEED,
            "unit": "paired DEV_VAL customer",
            "single_class_policy": "discard and continue deterministically",
            "interval": "2.5th and 97.5th percentiles",
            "reported": ["observed delta", "mean delta", "95% percentile interval", "fraction below zero", "fraction above zero"],
        },
        "practical_evidence_rule": {
            "clear_material_improvement": "Both observed primary deltas positive, both paired 95% lower bounds above zero, and ROC-AUC delta >= 0.01 or AP delta >= 0.02.",
            "aggregation_substantial": "The same alternative rule meets clear material improvement versus MEAN in both HGB arms.",
        },
        "decision_policy": {
            "A": "Only customer multiplicity meets the clear material improvement rule.",
            "B": "Only aggregation choice meets the substantial aggregation rule.",
            "C": "Only C_LOGIT minus W_LOGIT meets the clear material improvement rule.",
            "D": "None of A/B/C is supported and C_LOGIT fails to improve both observed primary metrics; the historical direct-customer advantage appears primarily representational.",
            "E": "More than one mechanism is supported, or the remaining pattern is mixed/inconclusive.",
        },
        "prohibited_adaptations": [
            "No hyperparameter tuning, feature discovery, aggregation selection, threshold tuning, class balancing, SMOTE, GAN, pseudo temporal labels, multiple-instance learning, or post-hoc comparison invention.",
            "Do not reuse historically incompatible Stage 14 fitted models.",
            "Do not load CONFIRMATION or FINAL_TEST features, labels, predictions, or performance.",
            "Do not alter the frozen Stage 17B detector.",
        ],
        "random_seeds": {"models": 42, "bootstrap": 42},
    }
    save_json(config, PREREGISTRATION)
    save_json(
        {
            "preregistration_path": relative(PREREGISTRATION),
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "audit_path": relative(AUDIT_PATH),
            "audit_sha256": file_sha256(AUDIT_PATH),
            "frozen_before_DEV_VAL_scoring": True,
        },
        SEAL_PATH,
    )
    print(f"Frozen Stage 20B preregistration: {file_sha256(PREREGISTRATION)}")
    print("No DEV_VAL scores were calculated during --prepare")


def load_preregistration() -> dict[str, Any]:
    if not PREREGISTRATION.is_file() or not SEAL_PATH.is_file():
        raise FileNotFoundError("Run --prepare before --evaluate")
    with SEAL_PATH.open("r", encoding="utf-8") as source:
        seal = json.load(source)
    if file_sha256(PREREGISTRATION) != seal["preregistration_sha256"]:
        raise ValueError("Stage 20B preregistration changed after freezing")
    if file_sha256(AUDIT_PATH) != seal["audit_sha256"]:
        raise ValueError("Stage 20B audit changed after freezing")
    with PREREGISTRATION.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not config.get("created_before_any_new_DEV_VAL_scoring"):
        raise ValueError("Preregistration timing declaration is invalid")
    for record in config["frozen_inputs"].values():
        if file_sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen input changed: {record['path']}")
    return config


def role_window_data(
    role: str,
    summaries: np.ndarray,
    labels: np.ndarray,
    customer_ids: np.ndarray,
    governance: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if role not in {"DEV_TRAIN", "DEV_VAL"}:
        raise ValueError(f"Closed role requested: {role}")
    allowed_ids = set(
        governance.loc[
            governance["research_role"].eq(role)
            & governance["stage16_representation_available"].eq(True),
            ID_COLUMN,
        ]
    )
    mask = np.isin(customer_ids, list(allowed_ids))
    role_ids = customer_ids[mask]
    if set(role_ids) != allowed_ids:
        raise ValueError(f"{role} window membership differs from governance")
    return summaries[mask], labels[mask], role_ids


def factorized_customers(
    X: np.ndarray, y: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    codes, unique_ids = pd.factorize(ids, sort=False)
    counts = np.bincount(codes)
    customer_X = np.zeros((len(unique_ids), X.shape[1]), dtype=np.float64)
    np.add.at(customer_X, codes, X)
    customer_X /= counts[:, None]
    customer_y_sum = np.bincount(codes, weights=y)
    if not np.all((customer_y_sum == 0) | (customer_y_sum == counts)):
        raise ValueError("A customer has inconsistent propagated window labels")
    customer_y = (customer_y_sum > 0).astype(np.int8)
    return customer_X, customer_y, unique_ids.astype(str), counts


def customer_balanced_weights(ids: np.ndarray) -> np.ndarray:
    codes, unique_ids = pd.factorize(ids, sort=False)
    counts = np.bincount(codes)
    weights = len(ids) / (len(unique_ids) * counts[codes])
    totals = np.bincount(codes, weights=weights)
    if not np.allclose(totals, totals[0], rtol=0, atol=1e-10):
        raise ValueError("Customer-balanced weights do not give equal totals")
    if not np.isclose(weights.mean(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Customer-balanced weights do not have mean 1")
    return weights


def aggregate_probabilities(
    y: np.ndarray, scores: np.ndarray, ids: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame({ID_COLUMN: ids, LABEL_COLUMN: y, "probability": scores})
    grouped = frame.groupby(ID_COLUMN, sort=False)
    if grouped[LABEL_COLUMN].nunique().max() != 1:
        raise ValueError("A customer has inconsistent labels during aggregation")
    result = grouped.agg(
        FLAG=(LABEL_COLUMN, "first"),
        usable_window_count=("probability", "size"),
        MEAN=("probability", "mean"),
        MEDIAN=("probability", "median"),
        MAXIMUM=("probability", "max"),
    )
    result["Q90"] = grouped["probability"].quantile(0.90, interpolation="linear")
    return result.reset_index()


def metric_row(model: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return {
        "model": model,
        "evaluation_role": "DEV_VAL",
        **classification_metrics(labels, scores, FIXED_THRESHOLD),
    }


def save_model(model: Any, name: str) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name.lower()}.joblib"
    joblib.dump(model, path)
    return {
        "model": name,
        "path": relative(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def paired_bootstrap(
    labels: np.ndarray,
    score_by_model: dict[str, np.ndarray],
    comparisons: list[dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(RANDOM_SEED)
    sample_rows: list[dict[str, Any]] = []
    attempts = 0
    replicate = 0
    while replicate < BOOTSTRAP_RESAMPLES:
        attempts += 1
        indices = rng.integers(0, len(labels), len(labels))
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        replicate += 1
        for comparison in comparisons:
            candidate = score_by_model[comparison["candidate"]][indices]
            comparator = score_by_model[comparison["comparator"]][indices]
            sample_rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "comparison": comparison["comparison"],
                    "ROC_AUC_delta": float(
                        roc_auc_score(y, candidate) - roc_auc_score(y, comparator)
                    ),
                    "AP_delta": float(
                        average_precision_score(y, candidate)
                        - average_precision_score(y, comparator)
                    ),
                }
            )
    samples = pd.DataFrame(sample_rows)
    summary_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        subset = samples.loc[samples["comparison"].eq(comparison["comparison"])]
        for metric, column in (("ROC_AUC", "ROC_AUC_delta"), ("AP", "AP_delta")):
            values = subset[column].to_numpy(dtype=np.float64)
            observed = (
                roc_auc_score(labels, score_by_model[comparison["candidate"]])
                - roc_auc_score(labels, score_by_model[comparison["comparator"]])
                if metric == "ROC_AUC"
                else average_precision_score(labels, score_by_model[comparison["candidate"]])
                - average_precision_score(labels, score_by_model[comparison["comparator"]])
            )
            summary_rows.append(
                {
                    "comparison": comparison["comparison"],
                    "candidate": comparison["candidate"],
                    "comparator": comparison["comparator"],
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


def comparison_deltas(
    metrics: pd.DataFrame, comparisons: list[dict[str, str]]
) -> pd.DataFrame:
    indexed = metrics.set_index("model")
    rows = []
    for comparison in comparisons:
        candidate = indexed.loc[comparison["candidate"]]
        comparator = indexed.loc[comparison["comparator"]]
        rows.append(
            {
                **comparison,
                "ROC_AUC_delta": float(candidate["ROC_AUC"] - comparator["ROC_AUC"]),
                "AP_delta": float(
                    candidate["PR_AUC_average_precision"]
                    - comparator["PR_AUC_average_precision"]
                ),
            }
        )
    return pd.DataFrame(rows)


def clear_material(summary: pd.DataFrame, comparison: str) -> bool:
    subset = summary.loc[summary["comparison"].eq(comparison)].set_index("metric")
    roc = subset.loc["ROC_AUC"]
    ap = subset.loc["AP"]
    return bool(
        roc["observed_delta"] > 0
        and ap["observed_delta"] > 0
        and roc["CI_2_5"] > 0
        and ap["CI_2_5"] > 0
        and (
            roc["observed_delta"] >= PRACTICAL_ROC_DELTA
            or ap["observed_delta"] >= PRACTICAL_AP_DELTA
        )
    )


def decide(summary: pd.DataFrame) -> tuple[str, str, dict[str, Any]]:
    multiplicity = clear_material(summary, "BALANCED_MEAN_MINUS_UNWEIGHTED_MEAN")
    supported_aggregators = []
    for rule in ("MEDIAN", "Q90", "MAXIMUM"):
        unweighted = f"WINDOW_HGB_UNWEIGHTED_{rule}_MINUS_WINDOW_HGB_UNWEIGHTED_MEAN"
        balanced = f"WINDOW_HGB_CUSTOMER_BALANCED_{rule}_MINUS_WINDOW_HGB_CUSTOMER_BALANCED_MEAN"
        if clear_material(summary, unweighted) and clear_material(summary, balanced):
            supported_aggregators.append(rule)
    aggregation = bool(supported_aggregators)
    customer_loss = clear_material(summary, "C_LOGIT_MINUS_W_LOGIT")
    mechanisms = sum((multiplicity, aggregation, customer_loss))
    if mechanisms > 1:
        code, label = "E", "MIXED / INCONCLUSIVE"
    elif multiplicity:
        code, label = "A", "CUSTOMER MULTIPLICITY EXPLAINS SUBSTANTIAL WINDOW WEAKNESS"
    elif aggregation:
        code, label = "B", "AGGREGATION CHOICE EXPLAINS SUBSTANTIAL WINDOW WEAKNESS"
    elif customer_loss:
        code, label = "C", "MATCHED CUSTOMER-LEVEL LOSS RETAINS A CLEAR ADVANTAGE"
    else:
        logit = summary.loc[
            summary["comparison"].eq("C_LOGIT_MINUS_W_LOGIT")
        ].set_index("metric")
        if logit.loc["ROC_AUC", "observed_delta"] <= 0 or logit.loc["AP", "observed_delta"] <= 0:
            code, label = "D", "DIRECT-CUSTOMER ADVANTAGE APPEARS PRIMARILY REPRESENTATIONAL"
        else:
            code, label = "E", "MIXED / INCONCLUSIVE"
    evidence = {
        "customer_multiplicity_clear_material_effect": multiplicity,
        "aggregation_clear_material_effect": aggregation,
        "aggregation_rules_supported_in_both_HGB_arms": supported_aggregators,
        "matched_customer_loss_clear_material_advantage": customer_loss,
    }
    return code, label, evidence


def markdown_metrics(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"| {row.model} | {row.count} | {row.theft_count} | {row.positive_prevalence:.6f} | "
            f"{row.ROC_AUC:.6f} | {row.PR_AUC_average_precision:.6f} | "
            f"{row.precision:.6f} | {row.recall:.6f} | {row.F1:.6f} | "
            f"{row.TN}/{row.FP}/{row.FN}/{row.TP} |"
        )
    return "\n".join(rows)


def markdown_bootstrap(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"| {row.comparison} | {row.metric} | {row.observed_delta:+.6f} | "
            f"{row.bootstrap_mean_delta:+.6f} | [{row.CI_2_5:+.6f}, {row.CI_97_5:+.6f}] | "
            f"{row.fraction_below_zero:.4f} | {row.fraction_above_zero:.4f} |"
        )
    return "\n".join(rows)


def paper_claim(code: str, evidence: dict[str, Any]) -> str:
    if code == "A":
        return "Equalizing each customer's training contribution was associated with a material improvement in the window formulation, showing that customer multiplicity was an important confound in SGCC development data."
    if code == "B":
        return "Customer-level discrimination from window models was materially sensitive to the pre-specified score-aggregation rule, showing that mean pooling was an important limitation in SGCC development data."
    if code == "C":
        return "Under a matched linear control using the same local behavioural dimensions, aligning the supervised loss with the customer-level SGCC target was associated with improved customer-level discrimination."
    if code == "D":
        return "The matched loss-placement control did not retain a clear customer-level advantage; the earlier direct-customer gain is therefore more consistent with richer longitudinal representation than with loss placement alone."
    mechanisms = [key for key, value in evidence.items() if value is True]
    if mechanisms:
        return "The pre-specified controls indicate that multiple formulation mechanisms contributed; no single customer-level objective effect can be isolated."
    return "The pre-specified controls did not isolate a clear multiplicity, aggregation, or customer-loss mechanism; the formulation explanation remains inconclusive."


def write_report(
    config: dict[str, Any],
    audit: dict[str, Any],
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    hgb_unweighted = metrics.loc[metrics["model"].str.startswith("WINDOW_HGB_UNWEIGHTED_")]
    hgb_balanced = metrics.loc[metrics["model"].str.startswith("WINDOW_HGB_CUSTOMER_BALANCED_")]
    logits = metrics.loc[metrics["model"].isin(["W_LOGIT", "C_LOGIT"])]
    text = f"""# Stage 20B — Controlled Label/Objective-Formulation Experiment

## Question and governance

Stage 20B tested whether closer alignment with the customer-level SGCC target provides an advantage not explained by unequal windows per customer or arithmetic-mean probability aggregation. SGCC has no temporal theft-event ground truth, so this experiment does not identify mislabeled windows.

The sealed preregistration is `{relative(PREREGISTRATION)}` with SHA-256 `{file_sha256(PREREGISTRATION)}`. Only original-train customers assigned DEV_TRAIN or DEV_VAL were loaded. DEV_TRAIN and DEV_VAL overlap is zero. Historical Stage 14 models were not reused.

## Population

| Role | Customers | Theft | Prevalence | Usable windows |
|---|---:|---:|---:|---:|
| DEV_TRAIN | {audit['development_counts']['DEV_TRAIN']['customers']} | {audit['development_counts']['DEV_TRAIN']['theft_customers']} | {audit['development_counts']['DEV_TRAIN']['theft_prevalence']:.6f} | {audit['development_counts']['DEV_TRAIN']['usable_windows']} |
| DEV_VAL | {audit['development_counts']['DEV_VAL']['customers']} | {audit['development_counts']['DEV_VAL']['theft_customers']} | {audit['development_counts']['DEV_VAL']['theft_prevalence']:.6f} | {audit['development_counts']['DEV_VAL']['usable_windows']} |

## Window HGB aggregation panel

All models use the exact 17 frozen Stage 14 summaries and fixed Stage 17B HGB settings. Customer-balanced weights equalize each customer's total loss contribution and retain mean window weight 1. All four aggregators are reported without selection.

| Model | N | Theft | Prevalence | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{markdown_metrics(hgb_unweighted)}
{markdown_metrics(hgb_balanced)}

## Matched logistic loss-placement control

Both arms use one shared customer-weighted DEV_TRAIN transformation of the same 17 local features and fixed L2 logistic settings. W_LOGIT applies loss to customer-balanced proxy-labelled windows and averages window probabilities; C_LOGIT applies loss once per customer after arithmetic feature pooling.

| Model | N | Theft | Prevalence | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{markdown_metrics(logits)}

Threshold metrics use the pre-specified fixed probability threshold 0.5 and are descriptive only.

## Paired comparisons and uncertainty

Two thousand paired DEV_VAL customer bootstrap replicates used seed 42. Every comparison below was frozen before scoring.

| Comparison | Metric | Observed delta | Bootstrap mean | 95% percentile CI | Fraction below 0 | Fraction above 0 |
|---|---|---:|---:|---|---:|---:|
{markdown_bootstrap(bootstrap)}

## Interpretation and decision

- Customer multiplicity clear material effect: **{decision['evidence']['customer_multiplicity_clear_material_effect']}**.
- Aggregation clear material effect in both HGB arms: **{decision['evidence']['aggregation_clear_material_effect']}**; qualifying rules: `{decision['evidence']['aggregation_rules_supported_in_both_HGB_arms']}`.
- Matched customer-level loss clear material advantage: **{decision['evidence']['matched_customer_loss_clear_material_advantage']}**.

Decision **{decision['decision']} — {decision['decision_label']}**.

The frozen Stage 17B HGB DEV_VAL ROC-AUC/AP (`0.788831`/`0.359795`) is historical context only; it is not treated as a representation-matched Stage 20B comparison.

## Paper claim and next step

{decision['strongest_defensible_paper_claim']}

This is triangulation, not proof of temporal window-label noise. {decision['research_matrix_recommendation']}

No adaptive tuning, feature discovery, aggregation selection, threshold tuning, or post-hoc comparison invention occurred. The frozen Stage 17B detector was not modified.

**CONFIRMATION NOT ACCESSED DURING STAGE 20B.**

**FINAL_TEST NOT ACCESSED DURING STAGE 20B.**
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")


def evaluate() -> None:
    if DECISION_PATH.exists():
        raise FileExistsError("Refusing to repeat completed Stage 20B evaluation")
    config = load_preregistration()
    verify_feature_and_model_specifications()
    with AUDIT_PATH.open("r", encoding="utf-8") as source:
        audit = json.load(source)

    governance = pd.read_csv(PROTOCOL_MANIFEST, dtype={ID_COLUMN: str})
    governance = governance.loc[
        governance["research_role"].isin(["DEV_TRAIN", "DEV_VAL"])
    ].copy()
    if not governance["original_split"].eq("train").all():
        raise ValueError("Development governance includes a protected original split")
    windows = load_window_matrix("train")
    summaries = behavioural_features(windows.X)
    train_X, train_y, train_ids = role_window_data(
        "DEV_TRAIN", summaries, windows.y, windows.customer_ids, governance
    )
    val_X, val_y, val_ids = role_window_data(
        "DEV_VAL", summaries, windows.y, windows.customer_ids, governance
    )
    if set(train_ids) & set(val_ids):
        raise ValueError("DEV_TRAIN and DEV_VAL overlap")
    if train_X.shape != (614976, 17) or val_X.shape != (153947, 17):
        raise ValueError("Development window matrix shape changed")

    train_customer_X_raw, train_customer_y, train_customer_ids, train_counts = factorized_customers(
        train_X, train_y, train_ids
    )
    val_customer_X_raw, val_customer_y, val_customer_ids, val_counts = factorized_customers(
        val_X, val_y, val_ids
    )
    if len(train_customer_y) != 23724 or len(val_customer_y) != 5931:
        raise ValueError("Development customer count changed")
    weights = customer_balanced_weights(train_ids)

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    score_by_model: dict[str, np.ndarray] = {}
    model_records: list[dict[str, Any]] = []
    runtimes: dict[str, float] = {}

    for model_name, sample_weight in (
        ("WINDOW_HGB_UNWEIGHTED", None),
        ("WINDOW_HGB_CUSTOMER_BALANCED", weights),
    ):
        started = time.perf_counter()
        model = HistGradientBoostingClassifier(**HGB_PARAMETERS)
        model.fit(train_X, train_y, sample_weight=sample_weight)
        runtimes[model_name] = time.perf_counter() - started
        window_scores = np.asarray(model.predict_proba(val_X)[:, 1], dtype=np.float64)
        customer_scores = aggregate_probabilities(val_y, window_scores, val_ids).set_index(ID_COLUMN)
        customer_scores = customer_scores.reindex(val_customer_ids)
        if customer_scores.isna().any().any() or not np.array_equal(
            customer_scores[LABEL_COLUMN].to_numpy(dtype=np.int8), val_customer_y
        ):
            raise ValueError(f"{model_name} customer aggregation is misaligned")
        for aggregation in AGGREGATIONS:
            output_name = f"{model_name}_{aggregation}"
            scores = customer_scores[aggregation].to_numpy(dtype=np.float64)
            score_by_model[output_name] = scores
            metrics_rows.append(metric_row(output_name, val_customer_y, scores))
            prediction_frames.append(
                pd.DataFrame(
                    {
                        ID_COLUMN: val_customer_ids,
                        LABEL_COLUMN: val_customer_y,
                        "model": output_name,
                        "probability": scores,
                        "usable_window_count": val_counts,
                    }
                )
            )
        model_records.append(save_model(model, model_name))

    weighted_mean = np.average(train_X.astype(np.float64), axis=0, weights=weights)
    weighted_variance = np.average(
        np.square(train_X.astype(np.float64) - weighted_mean), axis=0, weights=weights
    )
    scale = np.sqrt(weighted_variance)
    scale[scale == 0] = 1.0
    train_scaled = (train_X.astype(np.float64) - weighted_mean) / scale
    val_scaled = (val_X.astype(np.float64) - weighted_mean) / scale
    train_customer_X, check_train_y, check_train_ids, _ = factorized_customers(
        train_scaled, train_y, train_ids
    )
    val_customer_X, check_val_y, check_val_ids, _ = factorized_customers(
        val_scaled, val_y, val_ids
    )
    if not np.array_equal(check_train_y, train_customer_y) or not np.array_equal(
        check_val_y, val_customer_y
    ):
        raise ValueError("Customer labels changed during shared transformation")
    if not np.array_equal(check_train_ids, train_customer_ids) or not np.array_equal(
        check_val_ids, val_customer_ids
    ):
        raise ValueError("Customer order changed during shared transformation")

    started = time.perf_counter()
    w_logit = LogisticRegression(**LOGIT_PARAMETERS)
    w_logit.fit(train_scaled, train_y, sample_weight=weights)
    runtimes["W_LOGIT"] = time.perf_counter() - started
    w_window_scores = np.asarray(w_logit.predict_proba(val_scaled)[:, 1], dtype=np.float64)
    w_frame = aggregate_probabilities(val_y, w_window_scores, val_ids).set_index(ID_COLUMN).reindex(val_customer_ids)
    w_scores = w_frame["MEAN"].to_numpy(dtype=np.float64)

    started = time.perf_counter()
    c_logit = LogisticRegression(**LOGIT_PARAMETERS)
    c_logit.fit(train_customer_X, train_customer_y)
    runtimes["C_LOGIT"] = time.perf_counter() - started
    c_scores = np.asarray(c_logit.predict_proba(val_customer_X)[:, 1], dtype=np.float64)
    for name, scores in (("W_LOGIT", w_scores), ("C_LOGIT", c_scores)):
        score_by_model[name] = scores
        metrics_rows.append(metric_row(name, val_customer_y, scores))
        prediction_frames.append(
            pd.DataFrame(
                {
                    ID_COLUMN: val_customer_ids,
                    LABEL_COLUMN: val_customer_y,
                    "model": name,
                    "probability": scores,
                    "usable_window_count": val_counts,
                }
            )
        )
    model_records.append(
        save_model(
            {"model": w_logit, "weighted_mean": weighted_mean, "scale": scale},
            "W_LOGIT",
        )
    )
    model_records.append(
        save_model(
            {"model": c_logit, "weighted_mean": weighted_mean, "scale": scale},
            "C_LOGIT",
        )
    )

    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics[
        [
            "model", "evaluation_role", "count", "normal_count", "theft_count",
            "positive_prevalence", "threshold", "threshold_rule", "ROC_AUC",
            "PR_AUC_average_precision", "precision", "recall", "F1", "TN", "FP", "FN", "TP",
        ]
    ]
    comparisons = config["comparisons"]
    deltas = comparison_deltas(metrics, comparisons)
    bootstrap_samples, bootstrap_summary, attempts = paired_bootstrap(
        val_customer_y, score_by_model, comparisons
    )
    decision_code, decision_label, evidence = decide(bootstrap_summary)
    claim = paper_claim(decision_code, evidence)
    unresolved = (
        "No serious unresolved internal-validity issue requiring another development experiment was identified; temporal label noise remains inherently unidentifiable without new timing ground truth."
    )
    decision = {
        "stage": "Stage 20B — controlled formulation decision",
        "decision": decision_code,
        "decision_label": decision_label,
        "decision_policy": config["decision_policy"],
        "evidence": evidence,
        "strongest_defensible_paper_claim": claim,
        "research_matrix_recommendation": "Freeze the AMP development experiment matrix.",
        "genuinely_unresolved_validity_issue": unresolved,
        "adaptive_tuning": False,
        "new_feature_discovery": False,
        "aggregation_selection": False,
        "threshold_tuning": False,
        "historical_stage14_model_reused": False,
        "frozen_stage17b_detector_modified": False,
        "temporal_label_noise_claimed": False,
        "confirmation_accessed": False,
        "final_test_accessed": False,
    }

    save_csv(metrics, RESULTS_DIR / "control_metrics.csv")
    save_csv(deltas, RESULTS_DIR / "primary_comparison_deltas.csv")
    save_csv(bootstrap_samples, RESULTS_DIR / "paired_bootstrap_samples.csv")
    save_csv(bootstrap_summary, RESULTS_DIR / "paired_bootstrap_summary.csv")
    save_csv(pd.concat(prediction_frames, ignore_index=True), RESULTS_DIR / "dev_val_customer_probabilities.csv")
    save_json({"models": model_records}, RESULTS_DIR / "model_artifact_manifest.json")
    save_json(
        {
            "stage": "Stage 20B training and evaluation summary",
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "scikit_learn_version": sklearn.__version__,
            "development_counts": audit["development_counts"],
            "matrix_shapes": {
                "DEV_TRAIN_windows_by_features": list(train_X.shape),
                "DEV_VAL_windows_by_features": list(val_X.shape),
                "DEV_TRAIN_customers_by_features": list(train_customer_X.shape),
                "DEV_VAL_customers_by_features": list(val_customer_X.shape),
            },
            "customer_weight_audit": {
                "minimum_window_weight": float(weights.min()),
                "maximum_window_weight": float(weights.max()),
                "mean_window_weight": float(weights.mean()),
                "equal_total_customer_weight": True,
            },
            "shared_scaler": {
                "weighted_mean": weighted_mean.tolist(),
                "scale": scale.tolist(),
                "fit_role": "DEV_TRAIN",
            },
            "fit_runtime_seconds": runtimes,
            "bootstrap_attempts": attempts,
            "bootstrap_valid_replicates": BOOTSTRAP_RESAMPLES,
            "loaded_model_data_source": "original train only",
            "loaded_model_roles": ["DEV_TRAIN", "DEV_VAL"],
            "closed_roles_loaded_into_matrices": [],
        },
        RESULTS_DIR / "training_summary.json",
    )
    save_json(decision, DECISION_PATH)
    write_report(config, audit, metrics, bootstrap_summary, decision)
    assert_artifacts()
    print(f"Stage 20B decision {decision_code}: {decision_label}")
    print("CONFIRMATION and FINAL_TEST were not accessed")


def assert_artifacts() -> None:
    required = [
        AUDIT_PATH,
        PREREGISTRATION,
        SEAL_PATH,
        RESULTS_DIR / "control_metrics.csv",
        RESULTS_DIR / "primary_comparison_deltas.csv",
        RESULTS_DIR / "paired_bootstrap_samples.csv",
        RESULTS_DIR / "paired_bootstrap_summary.csv",
        RESULTS_DIR / "dev_val_customer_probabilities.csv",
        RESULTS_DIR / "model_artifact_manifest.json",
        RESULTS_DIR / "training_summary.json",
        DECISION_PATH,
        REPORT_PATH,
    ]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"Missing Stage 20B artifacts: {missing}")
    metrics = pd.read_csv(RESULTS_DIR / "control_metrics.csv")
    if len(metrics) != 10 or set(metrics["model"]) != {
        *(f"WINDOW_HGB_UNWEIGHTED_{rule}" for rule in AGGREGATIONS),
        *(f"WINDOW_HGB_CUSTOMER_BALANCED_{rule}" for rule in AGGREGATIONS),
        "W_LOGIT", "C_LOGIT",
    }:
        raise AssertionError("Stage 20B metric panel is incomplete")
    if not metrics["count"].eq(5931).all() or not metrics["theft_count"].eq(505).all():
        raise AssertionError("Stage 20B metric populations changed")
    bootstrap = pd.read_csv(RESULTS_DIR / "paired_bootstrap_summary.csv")
    if len(bootstrap) != 16 or not bootstrap["resamples"].eq(2000).all():
        raise AssertionError("Stage 20B bootstrap summary is incomplete")
    with DECISION_PATH.open("r", encoding="utf-8") as source:
        decision = json.load(source)
    if decision["adaptive_tuning"] or decision["confirmation_accessed"] or decision["final_test_accessed"]:
        raise AssertionError("Stage 20B governance assertion failed")


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
        print("Stage 20B artifact assertions passed")


if __name__ == "__main__":
    main()
