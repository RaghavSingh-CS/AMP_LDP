"""Execute the sealed AMP Stage 23B final-evaluation protocol exactly once.

The result-directory and model-directory guards make a second execution fail
before data access.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.run_stage14_representation import behavioural_features  # noqa: E402
from src.analysis.run_stage16_customer_aggregation import (  # noqa: E402
    FEATURE_NAMES,
    PERSISTENCE_FEATURES,
    build_customer_features,
)
from src.analysis.supervised_signal_utils import (  # noqa: E402
    ID_COLUMN,
    LABEL_COLUMN,
    classification_metrics,
    file_sha256,
    load_window_matrix,
    save_csv,
    save_json,
)


EXPECTED_HEAD = "f61c220591b92a587ce6abc733bd5aec4fa24372"
EXPECTED_PROTOCOL_HASH = "dbaa2035586c3e9f8f4ed2654d5a129acb2d767d14ded5b168dfc740d8943993"
THRESHOLD = 0.20070617709942135
RESULTS = ROOT / "results" / "supervised" / "stage_23b"
MODEL_DIR = ROOT / "models" / "supervised" / "stage_23b"
MODEL_PATH = MODEL_DIR / "final_full_48_hgb.joblib"
PROTOCOL_PATH = ROOT / "results" / "supervised" / "stage_23a" / "final_evaluation_protocol.json"
METHOD_PATH = ROOT / "results" / "supervised" / "stage_23a" / "final_method.json"
GOVERNANCE_PATH = ROOT / "results" / "governance" / "stage_17a" / "research_protocol_manifest.csv"
HISTORICAL_SPLIT_PATH = ROOT / "data" / "processed" / "split_manifest.parquet"
STAGE16_PATH = ROOT / "results" / "supervised" / "stage_16" / "preregistered_config.json"
STAGE17B_CONFIG_PATH = ROOT / "results" / "supervised" / "stage_17b" / "preregistered_benchmark_config.json"
STAGE17B_MODEL_PATH = ROOT / "models" / "supervised" / "stage_17b" / "hist_gradient_boosting.joblib"
STAGE17B_DECISION_PATH = ROOT / "results" / "supervised" / "stage_17b" / "validation_decision.json"
SCALER_PATH = ROOT / "models" / "preprocessing" / "scaler.joblib"

PARAMETERS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "class_weight": None,
    "early_stopping": False,
    "random_state": 42,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def string_array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode("utf-8")).hexdigest()


def load_chronology(split: str, expected_window_ids: np.ndarray, expected_customer_ids: np.ndarray) -> np.ndarray:
    path = ROOT / "data" / "windows" / "stride_30" / "usable" / f"{split}_windows.parquet"
    frame = pd.read_parquet(
        path,
        columns=["window_id", ID_COLUMN, "window_number_within_customer"],
        engine="pyarrow",
    )
    if not np.array_equal(frame["window_id"].astype(str).to_numpy(), expected_window_ids):
        raise ValueError(f"{split} chronology window order differs from the model matrix")
    if not np.array_equal(frame[ID_COLUMN].astype(str).to_numpy(), expected_customer_ids):
        raise ValueError(f"{split} chronology customer order differs from the model matrix")
    return frame["window_number_within_customer"].to_numpy(dtype=np.int32)


def write_abort(reason: str, assertions: list[dict[str, Any]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "stage": "Stage 23B — frozen final evaluation",
            "status": "ABORTED_BEFORE_FINAL_TEST_ACCESS",
            "abort_reason": reason,
            "final_test_accessed": False,
            "assertions": assertions,
        },
        RESULTS / "pre_access_audit.json",
    )


def pre_access_audit() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    assertions: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        assertions.append({"assertion": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise ValueError(f"{name}: {detail}")

    try:
        check("git_status_clean", run_git("status", "--porcelain") == "", "no tracked or untracked changes")
        head = run_git("rev-parse", "HEAD")
        check("expected_stage_23a_head", head == EXPECTED_HEAD, head)
        check("protocol_exists", PROTOCOL_PATH.is_file(), str(PROTOCOL_PATH.relative_to(ROOT)))
        protocol_hash = file_sha256(PROTOCOL_PATH)
        check("protocol_sha256", protocol_hash == EXPECTED_PROTOCOL_HASH, protocol_hash)
        protocol = load_json(PROTOCOL_PATH)
        method = load_json(METHOD_PATH)
        sealed = protocol["sealed_inputs"]
        path_hashes = {
            "final_method": (METHOD_PATH, sealed["final_method_sha256"]),
            "stage_17a_governance_manifest": (GOVERNANCE_PATH, sealed["stage_17a_governance_manifest_sha256"]),
            "historical_split_manifest": (HISTORICAL_SPLIT_PATH, sealed["historical_split_manifest_sha256"]),
            "stage_16_feature_specification": (STAGE16_PATH, sealed["stage_16_feature_specification_sha256"]),
            "stage_17b_configuration": (STAGE17B_CONFIG_PATH, sealed["stage_17b_configuration_sha256"]),
            "stage_17b_model": (STAGE17B_MODEL_PATH, sealed["stage_17b_model_sha256"]),
            "stage_17b_validation_decision": (STAGE17B_DECISION_PATH, sealed["stage_17b_validation_decision_sha256"]),
            "preprocessing_scaler": (SCALER_PATH, sealed["preprocessing_scaler_sha256"]),
        }
        verified_hashes: dict[str, dict[str, Any]] = {}
        for name, (path, expected) in path_hashes.items():
            actual = file_sha256(path)
            check(f"{name}_sha256", actual == expected, actual)
            verified_hashes[name] = {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "expected_sha256": expected,
                "actual_sha256": actual,
            }

        feature_spec = load_json(STAGE16_PATH)
        feature_names = tuple(feature_spec["feature_names_in_order"])
        check("stage16_exact_feature_count", feature_spec["feature_count"] == 48 == len(FEATURE_NAMES), len(feature_names))
        check("stage16_exact_feature_order", feature_names == FEATURE_NAMES == tuple(method["feature_schema"]["ordered_names"]), "all three ordered schemas match")

        benchmark = load_json(STAGE17B_CONFIG_PATH)
        benchmark_parameters = benchmark["candidate_models"]["hist_gradient_boosting"]["parameters"]
        check("stage17b_hgb_configuration", benchmark_parameters == PARAMETERS == method["classifier"]["parameters"], benchmark_parameters)
        decision = load_json(STAGE17B_DECISION_PATH)
        check("stage17b_selected_hgb", decision["selected_model"] == "hist_gradient_boosting", decision["selected_model"])
        check("frozen_threshold_exact", float(decision["selected_DEV_VAL_metrics"]["threshold"]) == THRESHOLD == float(method["threshold"]["value"]), THRESHOLD)
        check("no_methodological_choices_remaining", protocol["methodological_choices_remaining_for_stage_23b"] == 0 == method["methodological_choices_remaining_for_stage_23b"], 0)

        governance = pd.read_csv(GOVERNANCE_PATH, dtype={ID_COLUMN: str})
        expected_columns = [ID_COLUMN, "original_split", "research_role", "stage16_representation_available"]
        check("governance_columns", governance.columns.tolist() == expected_columns, governance.columns.tolist())
        check("governance_customer_ids_unique", not governance[ID_COLUMN].duplicated().any(), int(governance[ID_COLUMN].duplicated().sum()))
        check("governance_roles_exact", set(governance["research_role"]) == {"DEV_TRAIN", "DEV_VAL", "CONFIRMATION", "FINAL_TEST"}, sorted(governance["research_role"].unique()))
        expected_role_counts = {"DEV_TRAIN": 23725, "DEV_VAL": 5931, "CONFIRMATION": 6355, "FINAL_TEST": 6356}
        role_counts = governance["research_role"].value_counts().to_dict()
        check("governance_role_counts", role_counts == expected_role_counts, role_counts)
        role_to_original = {"DEV_TRAIN": "train", "DEV_VAL": "train", "CONFIRMATION": "val", "FINAL_TEST": "test"}
        mappings_ok = all(
            governance.loc[governance["research_role"].eq(role), "original_split"].eq(original).all()
            for role, original in role_to_original.items()
        )
        check("role_to_historical_split_mapping", mappings_ok, role_to_original)
        role_sets = {
            role: set(governance.loc[governance["research_role"].eq(role), ID_COLUMN])
            for role in expected_role_counts
        }
        overlap_counts: dict[str, int] = {}
        roles = list(role_sets)
        for i, left in enumerate(roles):
            for right in roles[i + 1 :]:
                overlap_counts[f"{left}__{right}"] = len(role_sets[left] & role_sets[right])
        check("all_role_overlaps_zero", not any(overlap_counts.values()), overlap_counts)
        final_train_rows = governance[governance["research_role"].isin(["DEV_TRAIN", "DEV_VAL"])]
        eligible = final_train_rows[final_train_rows["stage16_representation_available"]]
        excluded = final_train_rows[~final_train_rows["stage16_representation_available"]]
        check("final_training_roles_exact", set(final_train_rows["research_role"]) == {"DEV_TRAIN", "DEV_VAL"}, sorted(final_train_rows["research_role"].unique()))
        check("final_training_eligible_count", len(eligible) == 29655, len(eligible))
        check("one_unusable_original_train_customer", excluded[ID_COLUMN].tolist() == ["B08E0670C8A6933E5EC44715F7A80E9D"], excluded[ID_COLUMN].tolist())
        check("confirmation_excluded_from_training", not set(eligible[ID_COLUMN]) & role_sets["CONFIRMATION"], 0)
        check("final_test_excluded_from_training", not set(eligible[ID_COLUMN]) & role_sets["FINAL_TEST"], 0)
        check("final_train_final_test_overlap_zero", not set(eligible[ID_COLUMN]) & role_sets["FINAL_TEST"], 0)
        check("final_training_population_protocol", protocol["data_roles"]["final_fit"] == ["DEV_TRAIN", "DEV_VAL"], protocol["data_roles"]["final_fit"])

        audit = {
            "stage": "Stage 23B — frozen final evaluation pre-access audit",
            "status": "ALL_PRE_ACCESS_ASSERTIONS_PASSED",
            "created_before_final_test_features_labels_predictions_or_metrics_were_loaded": True,
            "timestamp": datetime.now().astimezone().isoformat(),
            "source_commit": head,
            "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "protocol_sha256": protocol_hash,
            "verified_artifacts": verified_hashes,
            "governance": {
                "role_counts": role_counts,
                "pairwise_overlap_counts": overlap_counts,
                "final_training_roles": ["DEV_TRAIN", "DEV_VAL"],
                "final_training_governed_count": int(len(final_train_rows)),
                "final_training_representation_available_count": int(len(eligible)),
                "excluded_unusable_customer": excluded.to_dict(orient="records"),
                "confirmation_in_final_training": False,
                "final_test_in_final_training": False,
            },
            "implementation_requires_unsealed_methodological_choice": False,
            "final_test_accessed": False,
            "assertions": assertions,
        }
        RESULTS.mkdir(parents=True, exist_ok=False)
        MODEL_DIR.mkdir(parents=True, exist_ok=False)
        save_json(audit, RESULTS / "pre_access_audit.json")
        return protocol, method, governance
    except Exception as exc:
        if not (RESULTS / "pre_access_audit.json").exists():
            write_abort(str(exc), assertions)
        raise


def matrix_integrity(
    role: str,
    features: pd.DataFrame,
    labels: np.ndarray,
    customer_ids: np.ndarray,
    expected_ids: set[str],
    forbidden_ids: set[str],
) -> dict[str, Any]:
    matrix = features.to_numpy(dtype=np.float64)
    if features.shape != (len(expected_ids), 48):
        raise ValueError(f"{role} matrix shape is {features.shape}")
    if tuple(features.columns) != FEATURE_NAMES:
        raise ValueError(f"{role} feature order changed")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{role} matrix contains NaN/inf")
    if len(set(customer_ids)) != len(customer_ids):
        raise ValueError(f"{role} customer IDs are not unique")
    if set(customer_ids) != expected_ids:
        raise ValueError(f"{role} customer membership differs from governance")
    if set(customer_ids) & forbidden_ids:
        raise ValueError(f"{role} contains forbidden customer IDs")
    if len(labels) != len(customer_ids) or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{role} labels do not align one-to-one")
    return {
        "role": role,
        "shape": list(features.shape),
        "feature_count": 48,
        "feature_names_in_order": list(FEATURE_NAMES),
        "feature_schema_sha256": hashlib.sha256(("\n".join(FEATURE_NAMES) + "\n").encode()).hexdigest(),
        "matrix_float64_sha256": array_sha256(matrix),
        "customer_ids_sha256": string_array_sha256(customer_ids),
        "labels_int8_sha256": array_sha256(labels.astype(np.int8)),
        "customer_ids_unique": True,
        "labels_aligned_one_to_one": True,
        "nonfinite_values": 0,
        "forbidden_customer_overlap": 0,
        "normal_customers": int(np.sum(labels == 0)),
        "theft_customers": int(np.sum(labels == 1)),
    }


def bootstrap(labels: np.ndarray, scores: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    attempts = 0
    rejected = 0
    while len(rows) < 2000:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            rejected += 1
            continue
        sampled_scores = scores[indices]
        rows.append(
            {
                "bootstrap_replicate": len(rows) + 1,
                "ROC_AUC": float(roc_auc_score(sampled_labels, sampled_scores)),
                "PR_AUC_average_precision": float(average_precision_score(sampled_labels, sampled_scores)),
            }
        )
    samples = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "method": "customer-level nonparametric percentile bootstrap",
        "valid_replicates": 2000,
        "seed": 42,
        "sampling_attempts": attempts,
        "invalid_single_class_replicates": rejected,
        "same_replicate_indices_for_both_metrics": True,
        "metrics": {},
    }
    for metric, observed in (
        ("ROC_AUC", roc_auc_score(labels, scores)),
        ("PR_AUC_average_precision", average_precision_score(labels, scores)),
    ):
        values = samples[metric].to_numpy(dtype=np.float64)
        summary["metrics"][metric] = {
            "observed_estimate": float(observed),
            "bootstrap_mean": float(np.mean(values)),
            "percentile_95_CI_lower": float(np.quantile(values, 0.025)),
            "percentile_95_CI_upper": float(np.quantile(values, 0.975)),
        }
    return samples, summary


def scientific_status(roc: float, ap: float) -> tuple[str, str]:
    prior_roc = np.array([0.7888308218953116, 0.7890840481602820])
    prior_ap = np.array([0.3597945903217465, 0.3790578558669986])
    roc_delta = roc - float(prior_roc.mean())
    ap_delta = ap - float(prior_ap.mean())
    # Descriptive classification only: jointly near prior evidence is stable;
    # joint material movement identifies improvement or degradation.
    if roc_delta >= 0.03 and ap_delta >= 0.03:
        return "D", "FINAL PERFORMANCE IMPROVEMENT"
    if roc_delta <= -0.10 or ap_delta <= -0.10:
        return "C", "MAJOR FINAL DEGRADATION"
    if roc_delta <= -0.03 or ap_delta <= -0.03:
        return "B", "MODERATE FINAL DEGRADATION"
    return "A", "STRONG STABLE FINAL REPLICATION"


def write_report(
    metrics: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    references: dict[str, Any],
    train_integrity: dict[str, Any],
    test_integrity: dict[str, Any],
    model_record: dict[str, Any],
    gate_hash: str,
    status_code: str,
    status_label: str,
) -> None:
    roc_boot = bootstrap_summary["metrics"]["ROC_AUC"]
    ap_boot = bootstrap_summary["metrics"]["PR_AUC_average_precision"]
    report = f"""# Stage 23B — Frozen Final Evaluation

## Governance and access gate

Every pre-access assertion passed before `FINAL_TEST` features, labels, predictions,
or metrics were loaded. The verified Stage 23A protocol SHA-256 was
`{EXPECTED_PROTOCOL_HASH}`. The exact final-training population was
`DEV_TRAIN + DEV_VAL`: {train_integrity['shape'][0]:,} representation-available
customers. `CONFIRMATION` and `FINAL_TEST` were excluded. The one established
unusable original-training customer (`B08E0670C8A6933E5EC44715F7A80E9D`)
remained excluded and documented.

The four label-free persistence references were recomputed from all
{references['eligible_window_count']:,} usable final-training windows and frozen
before test construction. The final model was serialized before the access gate.
The access-gate artifact SHA-256 is `{gate_hash}`.

## Frozen method

The final matrix used the exact ordered 48-feature Stage 16 representation. One
unweighted `HistGradientBoostingClassifier` was fit once with learning rate 0.1,
100 iterations, 15 maximum leaf nodes, minimum leaf size 100, L2 regularization
1.0, no early stopping, and random seed 42. The model artifact SHA-256 is
`{model_record['model_sha256']}` and its size is {model_record['serialized_size_bytes']:,}
bytes. Fit runtime was {model_record['fit_runtime_seconds']:.6f} seconds.

The fixed threshold was `{THRESHOLD:.17g}`. It was not rederived or recalibrated.
Because the model was refit on the combined development population, calibration
may shift; all threshold-dependent metrics below are secondary/descriptive.

## Final result

The eligible final cohort contains {metrics['count']:,} customers: {metrics['theft_count']:,}
theft and {metrics['normal_count']:,} normal, for prevalence
{metrics['positive_prevalence']:.12f}. There were no exclusions and no overlap
with final training. The exact 48-column test matrix was finite and had one row
per governed eligible customer.

Primary metrics:

- ROC-AUC: **{metrics['ROC_AUC']:.12f}**; customer-bootstrap 95% percentile CI
  [{roc_boot['percentile_95_CI_lower']:.12f}, {roc_boot['percentile_95_CI_upper']:.12f}]
  (bootstrap mean {roc_boot['bootstrap_mean']:.12f}).
- Average Precision: **{metrics['PR_AUC_average_precision']:.12f}**; customer-bootstrap
  95% percentile CI [{ap_boot['percentile_95_CI_lower']:.12f},
  {ap_boot['percentile_95_CI_upper']:.12f}] (bootstrap mean
  {ap_boot['bootstrap_mean']:.12f}).

Secondary frozen-threshold metrics: precision {metrics['precision']:.12f}, recall
{metrics['recall']:.12f}, F1 {metrics['F1']:.12f}; TN={metrics['TN']}, FP={metrics['FP']},
FN={metrics['FN']}, TP={metrics['TP']}.

The bootstrap used 2,000 valid customer-level replicates, seed 42, identical
replicate indices for both primary metrics, and rejected
{bootstrap_summary['invalid_single_class_replicates']} single-class replicates.

## Descriptive comparison and scientific status

Stage 17B `DEV_VAL` recorded ROC-AUC/AP 0.788830821895/0.359794590322.
Stage 17C `CONFIRMATION` recorded 0.789084048160/0.379057855867. The historical
Stage 13 final-cohort window baseline recorded 0.7023135402/0.2193165970.
These are descriptive comparisons only; no superiority test, selection, tuning,
or adaptation followed.

Final status: **{status_code} — {status_label}**. This classification is descriptive
and cannot change the frozen method.

## Historical-access disclosure and closure

This population is a **"final evaluation cohort with disclosed historical baseline
access"**, not a pristine never-seen test cohort. Stage 13 historically evaluated
an obsolete window-level baseline on it. The current FULL_48 direct-customer
method did not yet exist, and Stages 14–22 did not use this population to tune or
select the frozen current method. The historical access is a limitation requiring
disclosure, not grounds for pretending the cohort was untouched.

The final result was accepted as-is without feature engineering, model switching,
hyperparameter tuning, threshold tuning, calibration, subgroup optimization,
resampling, GAN/SMOTE, or any other adaptation.

**AMP FROZEN EXPERIMENT COMPLETE.** No additional model-development experiment
may be added to the current reported study based on this result. Future work must
be explicitly separated from the frozen AMP experiment.
"""
    (RESULTS / "final_evaluation_report.md").write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    protocol, method, governance = pre_access_audit()
    train_ids = set(
        governance.loc[
            governance["research_role"].isin(["DEV_TRAIN", "DEV_VAL"])
            & governance["stage16_representation_available"],
            ID_COLUMN,
        ]
    )
    confirmation_ids = set(governance.loc[governance["research_role"].eq("CONFIRMATION"), ID_COLUMN])
    final_test_ids = set(governance.loc[governance["research_role"].eq("FINAL_TEST"), ID_COLUMN])

    # Sections 2–4: only original-train data are loaded here.
    train_windows = load_window_matrix("train")
    if set(train_windows.customer_ids) != train_ids:
        raise ValueError("Original-train usable-window membership differs from DEV_TRAIN + DEV_VAL")
    if set(train_windows.customer_ids) & (confirmation_ids | final_test_ids):
        raise ValueError("Protected customer entered final-training windows")
    train_chronology = load_chronology("train", train_windows.window_ids, train_windows.customer_ids)
    train_summaries = behavioural_features(train_windows.X)
    train_features, train_labels, train_customer_ids, references = build_customer_features(
        train_summaries,
        train_windows.y,
        train_windows.customer_ids,
        train_chronology,
        None,
    )
    if len(train_windows.y) != 768923:
        raise ValueError(f"Final-training eligible window count is {len(train_windows.y)}")
    reference_payload = {
        "stage": "Stage 23B — frozen final-training persistence references",
        "created_before_final_test_access": True,
        "source_population": ["DEV_TRAIN", "DEV_VAL"],
        "source_original_split": "train",
        "eligible_window_count": int(len(train_windows.y)),
        "label_use": False,
        "quantile": 0.9,
        "interpolation": "linear",
        "comparison": "strictly_greater_than",
        "values": {name: float(references[name]) for name in PERSISTENCE_FEATURES},
    }
    reference_path = RESULTS / "final_training_references.json"
    save_json(reference_payload, reference_path)
    save_json(reference_payload, RESULTS / "final_train_references.json")
    reference_hash = file_sha256(reference_path)

    train_integrity = matrix_integrity(
        "DEV_TRAIN + DEV_VAL",
        train_features,
        train_labels,
        train_customer_ids,
        train_ids,
        confirmation_ids | final_test_ids,
    )
    train_integrity.update(
        {
            "usable_window_count": int(len(train_windows.y)),
            "persistence_reference_artifact": str(reference_path.relative_to(ROOT)).replace("\\", "/"),
            "persistence_reference_sha256": reference_hash,
            "excluded_unusable_original_train_customer": "B08E0670C8A6933E5EC44715F7A80E9D",
        }
    )
    save_json(train_integrity, RESULTS / "final_training_matrix_integrity.json")

    # Section 5: exactly one fit, serialized before any FINAL_TEST loader call.
    model = HistGradientBoostingClassifier(**PARAMETERS)
    started = time.perf_counter()
    model.fit(train_features.to_numpy(dtype=np.float64), train_labels)
    fit_runtime = time.perf_counter() - started
    train_probabilities = model.predict_proba(train_features.to_numpy(dtype=np.float64))[:, 1]
    if not np.isfinite(train_probabilities).all():
        raise ValueError("Final model produced non-finite training probabilities")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    model_hash = file_sha256(MODEL_PATH)
    model_record = {
        "stage": "Stage 23B — final model configuration and fit record",
        "class": "sklearn.ensemble.HistGradientBoostingClassifier",
        "package_version": sklearn.__version__,
        "parameters": PARAMETERS,
        "sample_weight": None,
        "class_weights": None,
        "fit_count": 1,
        "training_population": ["DEV_TRAIN", "DEV_VAL"],
        "training_customer_count": int(len(train_labels)),
        "fit_runtime_seconds": float(fit_runtime),
        "model_artifact": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": model_hash,
        "serialized_size_bytes": MODEL_PATH.stat().st_size,
        "training_probabilities_finite": True,
        "serialized_before_final_test_access": True,
        "frozen_threshold": THRESHOLD,
        "threshold_rederived_after_refit": False,
        "threshold_metrics_are_secondary": True,
        "threshold_interpretation": "Combined-development refitting may shift calibration; operating-point metrics are secondary/descriptive.",
    }
    save_json(model_record, RESULTS / "final_model_config.json")

    # Section 7: the explicit access gate is sealed before the first test-data call.
    gate = {
        "stage": "Stage 23B — FINAL_TEST ACCESS GATE",
        "timestamp": datetime.now().astimezone().isoformat(),
        "source_commit": EXPECTED_HEAD,
        "protocol_sha256": EXPECTED_PROTOCOL_HASH,
        "final_method_sha256": file_sha256(METHOD_PATH),
        "train_reference_artifact_sha256": reference_hash,
        "final_model_sha256": model_hash,
        "frozen_threshold": THRESHOLD,
        "all_abort_conditions_passed": True,
        "pre_access_audit_status": "ALL_PRE_ACCESS_ASSERTIONS_PASSED",
        "training_references_frozen": True,
        "final_model_serialized": True,
        "methodological_modification_permitted_after_gate": False,
        "final_test_access_authorized": True,
    }
    gate_path = RESULTS / "final_test_access_gate.json"
    save_json(gate, gate_path)
    gate_hash = file_sha256(gate_path)

    # First FINAL_TEST feature/label access occurs only below this line.
    test_windows = load_window_matrix("test")
    test_chronology = load_chronology("test", test_windows.window_ids, test_windows.customer_ids)
    test_summaries = behavioural_features(test_windows.X)
    test_features, test_labels, test_customer_ids, reused_references = build_customer_features(
        test_summaries,
        test_windows.y,
        test_windows.customer_ids,
        test_chronology,
        references,
    )
    if reused_references != references:
        raise ValueError("Frozen final-training persistence references changed")
    test_integrity = matrix_integrity(
        "FINAL_TEST",
        test_features,
        test_labels,
        test_customer_ids,
        final_test_ids,
        train_ids | confirmation_ids,
    )
    test_integrity.update(
        {
            "usable_window_count": int(len(test_windows.y)),
            "governed_customer_count": len(final_test_ids),
            "expected_exclusions": 0,
            "unexpected_exclusions": 0,
            "frozen_training_references_reused_exactly": True,
            "final_test_statistics_used_to_influence_construction": False,
        }
    )
    if test_integrity["shape"] != [6356, 48] or test_integrity["theft_customers"] != 542:
        raise ValueError("FINAL_TEST count or class audit differs from sealed expectations")
    save_json(test_integrity, RESULTS / "final_test_matrix_integrity.json")

    scores = np.asarray(model.predict_proba(test_features.to_numpy(dtype=np.float64))[:, 1], dtype=np.float64)
    if not np.isfinite(scores).all() or len(scores) != 6356:
        raise ValueError("FINAL_TEST scoring did not produce one finite probability per customer")
    predictions = pd.DataFrame(
        {
            ID_COLUMN: test_customer_ids,
            LABEL_COLUMN: test_labels,
            "probability": scores,
            "predicted_at_frozen_threshold": (scores >= THRESHOLD).astype(np.int8),
        }
    )
    save_csv(predictions, RESULTS / "final_test_customer_predictions.csv")
    save_csv(predictions, RESULTS / "final_test_predictions.csv")

    metrics = classification_metrics(test_labels, scores, THRESHOLD)
    metrics.update(
        {
            "stage": "Stage 23B — single frozen FINAL_TEST evaluation",
            "primary_metrics": ["ROC_AUC", "PR_AUC_average_precision"],
            "secondary_metrics": ["precision", "recall", "F1", "TN", "FP", "FN", "TP"],
            "model_sha256": model_hash,
            "access_gate_sha256": gate_hash,
            "model_modified_after_scoring": False,
            "threshold_optimized_on_final_test": False,
            "result_accepted_without_adaptation": True,
        }
    )
    save_json(metrics, RESULTS / "final_test_metrics.json")
    save_json(metrics, RESULTS / "final_metrics.json")
    save_csv(pd.DataFrame([metrics]), RESULTS / "final_test_metrics.csv")

    bootstrap_samples, bootstrap_summary = bootstrap(test_labels, scores)
    save_csv(bootstrap_samples, RESULTS / "final_test_bootstrap.csv")
    save_json(bootstrap_summary, RESULTS / "final_bootstrap_summary.json")

    comparison = pd.DataFrame(
        [
            {"evidence": "Stage 17B DEV_VAL", "ROC_AUC": 0.7888308218953116, "PR_AUC_average_precision": 0.3597945903217465, "ROC_AUC_delta_final_minus_prior": metrics["ROC_AUC"] - 0.7888308218953116, "AP_delta_final_minus_prior": metrics["PR_AUC_average_precision"] - 0.3597945903217465, "comparison_role": "descriptive_only"},
            {"evidence": "Stage 17C CONFIRMATION", "ROC_AUC": 0.7890840481602820, "PR_AUC_average_precision": 0.3790578558669986, "ROC_AUC_delta_final_minus_prior": metrics["ROC_AUC"] - 0.7890840481602820, "AP_delta_final_minus_prior": metrics["PR_AUC_average_precision"] - 0.3790578558669986, "comparison_role": "descriptive_only"},
            {"evidence": "Historical Stage 13 FINAL_TEST-cohort window baseline", "ROC_AUC": 0.7023135402, "PR_AUC_average_precision": 0.2193165970, "ROC_AUC_delta_final_minus_prior": metrics["ROC_AUC"] - 0.7023135402, "AP_delta_final_minus_prior": metrics["PR_AUC_average_precision"] - 0.2193165970, "comparison_role": "descriptive_only"},
        ]
    )
    save_csv(comparison, RESULTS / "prior_evidence_comparison.csv")
    status_code, status_label = scientific_status(metrics["ROC_AUC"], metrics["PR_AUC_average_precision"])
    save_json(
        {
            "stage": "Stage 23B — research closure",
            "scientific_status_code": status_code,
            "scientific_status_label": status_label,
            "classification_is_descriptive_only": True,
            "result_accepted_without_adaptation": True,
            "historical_access_disclosure": "final evaluation cohort with disclosed historical baseline access",
            "AMP_FROZEN_EXPERIMENT_COMPLETE": True,
            "future_work_must_be_separated": True,
        },
        RESULTS / "research_closure.json",
    )
    write_report(
        metrics,
        bootstrap_summary,
        reference_payload,
        train_integrity,
        test_integrity,
        model_record,
        gate_hash,
        status_code,
        status_label,
    )
    save_json(
        {
            "stage": "Stage 23B — frozen final evaluation execution audit",
            "source_commit": EXPECTED_HEAD,
            "protocol_sha256": EXPECTED_PROTOCOL_HASH,
            "pre_access_audit_sha256": file_sha256(RESULTS / "pre_access_audit.json"),
            "training_references_sha256": reference_hash,
            "training_matrix_integrity_sha256": file_sha256(RESULTS / "final_training_matrix_integrity.json"),
            "model_sha256": model_hash,
            "access_gate_sha256": gate_hash,
            "test_matrix_integrity_sha256": file_sha256(RESULTS / "final_test_matrix_integrity.json"),
            "predictions_sha256": file_sha256(RESULTS / "final_test_customer_predictions.csv"),
            "metrics_sha256": file_sha256(RESULTS / "final_test_metrics.json"),
            "bootstrap_samples_sha256": file_sha256(RESULTS / "final_test_bootstrap.csv"),
            "bootstrap_summary_sha256": file_sha256(RESULTS / "final_bootstrap_summary.json"),
            "fit_count": 1,
            "score_count": 1,
            "valid_bootstrap_replicates": 2000,
            "methodological_changes_after_access_gate": 0,
            "result_accepted_without_adaptation": True,
            "AMP_FROZEN_EXPERIMENT_COMPLETE": True,
        },
        RESULTS / "execution_audit.json",
    )
    print(json.dumps({"metrics": metrics, "bootstrap": bootstrap_summary, "status": [status_code, status_label]}, indent=2))


if __name__ == "__main__":
    main()
