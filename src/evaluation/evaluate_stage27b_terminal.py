"""Execute the single terminal Study 2 evaluation frozen by Stage 27A.

The historical ``test`` loader is used once for the final evaluation cohort
with disclosed historical baseline access. No model fitting is implemented.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.run_stage14_representation import FEATURE_NAMES as SUMMARY_NAMES, behavioural_features
from src.analysis.run_stage16_customer_aggregation import FEATURE_NAMES as FULL48_NAMES, build_customer_features
from src.analysis.run_stage25b_distribution_experiment import BATCH_SIZE, DistributionModel, masked_distribution
from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    file_sha256,
    load_window_matrix,
    save_csv,
    save_json,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_27b"
STAGE27A_DIR = RESULTS_DIR.parent / "stage_27a"
PREREG_PATH = STAGE27A_DIR / "stage27b_preregistration.json"
MODEL_FREEZE_PATH = STAGE27A_DIR / "final_model_freeze.json"
COMPARATOR_FREEZE_PATH = STAGE27A_DIR / "final_comparator_freeze.json"
CRITERIA_PATH = STAGE27A_DIR / "final_evaluation_criteria.json"
ANCHOR_PATH = STAGE27A_DIR / "anchor_survival_criteria.json"
PRE_ACCESS_PATH = RESULTS_DIR / "pre_final_access_audit.json"
ACCESS_EVENT_PATH = RESULTS_DIR / "final_access_event.json"
GOVERNANCE_PATH = PROJECT_ROOT / "results" / "governance" / "stage_17a" / "research_protocol_manifest.csv"
WINDOW_MANIFEST = PROJECT_ROOT / "data" / "windows" / "stride_30" / "usable" / "test_windows.parquet"
STAGE25B_INTEGRITY = RESULTS_DIR.parent / "stage_25b" / "bag_integrity.json"

SOURCE_COMMIT = "20cd26627b0601965166bab879ab3c46695ac0f0"
STUDY1_COMMIT = "d0601c54f6e9d0f11caac2ebd27f7c86a568a863"
PREREG_SHA256 = "ed2457e3a6eaf4b683c3fd8aaf89cf1f1fca63500a68f0427279813533e92702"
MODEL_FREEZE_SHA256 = "a5ce3e6db799e3f4053a69d457a7d22a9f45db27bda6e6f08ec87c69b232c14e"
COMPARATOR_FREEZE_SHA256 = "e97724ee4e62519d70810c0054297755e5c48c873e7190eafca2e87622e34f19"
CRITERIA_SHA256 = "31ce1b0a7c84ecdfec052fe4a26025075b82d12c6d4fe67d5b3a54cf81cdbb7b"
ANCHOR_SHA256 = "60338370876af16c8bdac5c998a75194e6e116ee8bb3dae2d748e3aed58a88f4"
STAGE25B_SPEC_SHA256 = "006c781899f74f0f62d01282fbf4164ac3ee27d8a61e3f2ea5fcb09b9e445470"
STAGE26_PROTOCOL_SHA256 = "7207a75dfd31d3254aaa3d4507eb9812b38ff87ecda09abfa62178572c8be09b"
COHORT_NAME = "final evaluation cohort with disclosed historical baseline access"
SEEDS = (42, 43, 44, 45, 46)
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2502
EXPECTED = {"customers": 6356, "normal": 5814, "theft": 542, "windows": 165020}
FULL48_THRESHOLDS = {
    "maximum": 1.5190057158470154,
    "standard_deviation": 0.438372477889061,
    "mean_absolute_consecutive_change": 0.2778959274291992,
    "maximum_absolute_consecutive_change": 1.1807788610458374,
}
DEV = {
    "LATENT": (0.7908091586895513, 0.36783736705395026),
    "FULL_48": (0.7888308218953116, 0.35979459032174654),
    "DIRECT": (0.7882334049844351, 0.3492440521718474),
}
CONFIRMATION = {
    "LATENT": (0.8117914548318028, 0.4152873417382383),
    "FULL_48": (0.789084048160282, 0.3790578558669986),
    "DIRECT": (0.7869997137095059, 0.35572458981124844),
}
DISCLOSURE = (
    "The final evaluation cohort had been accessed historically in Stage 13 for an obsolete "
    "supervised window-level baseline. The learned latent-distribution hypothesis, architecture, "
    "pooling rule, five-seed ensemble, controls, and confirmation decision reported in Study 2 "
    "were developed and frozen later without Study 2 row-level access to this cohort. We therefore "
    "treat it as a governed final evaluation cohort with disclosed historical baseline access, not "
    "as a pristine untouched test set."
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, encoding="utf-8").strip()


def stable_hash(values: np.ndarray | list[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def validate_seal() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if git("branch", "--show-current") != "study2/mil" or git("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ValueError("Branch/source HEAD assertion failed")
    if git("rev-parse", "main") != STUDY1_COMMIT:
        raise ValueError("Frozen Study 1 main moved")
    expected_hashes = {
        PREREG_PATH: PREREG_SHA256, MODEL_FREEZE_PATH: MODEL_FREEZE_SHA256,
        COMPARATOR_FREEZE_PATH: COMPARATOR_FREEZE_SHA256, CRITERIA_PATH: CRITERIA_SHA256,
        ANCHOR_PATH: ANCHOR_SHA256,
        RESULTS_DIR.parent / "stage_25b" / "stage25b_execution_spec.json": STAGE25B_SPEC_SHA256,
        RESULTS_DIR.parent / "stage_26" / "stage26_confirmation_protocol.json": STAGE26_PROTOCOL_SHA256,
    }
    for path, expected in expected_hashes.items():
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen artifact changed: {path}")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    model_freeze = json.loads(MODEL_FREEZE_PATH.read_text(encoding="utf-8"))
    comparator_freeze = json.loads(COMPARATOR_FREEZE_PATH.read_text(encoding="utf-8"))
    if tuple(prereg["primary_model"]["seeds"]) != SEEDS or prereg["uncertainty"] != {
        "resamples": 2000, "seed": 2502, "unit": "final-evaluation customer", "paired": True,
        "interval": "percentile 95%", "outputs": ["observed delta", "bootstrap mean delta", "lower and upper 95% bounds", "fraction above zero"]
    }:
        raise ValueError("Seed or bootstrap policy changed")
    for record in model_freeze["checkpoints"]:
        if file_sha256(PROJECT_ROOT / record["encoder"]) != record["encoder_sha256"] or file_sha256(PROJECT_ROOT / record["hgb"]) != record["hgb_sha256"]:
            raise ValueError(f"Frozen latent pipeline changed for seed {record['seed']}")
    for record in comparator_freeze["comparators"].values():
        if file_sha256(PROJECT_ROOT / record["model"]) != record["model_sha256"]:
            raise ValueError(f"Frozen comparator changed: {record['model']}")
    if not PRE_ACCESS_PATH.is_file() or ACCESS_EVENT_PATH.exists():
        raise ValueError("Pre-access audit missing or Stage 27B access already occurred")
    return prereg, model_freeze, comparator_freeze


def open_cohort_once() -> tuple[Any, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    access_event = {
        "stage": "Stage 27B final access event", "timestamp": datetime.now().astimezone().isoformat(),
        "cohort_terminology": COHORT_NAME,
        "event": "Opened the governed final evaluation cohort with disclosed historical baseline access exactly once for frozen terminal inference.",
        "stage27b_access_count": 1, "authorized_by_preregistration_sha256": PREREG_SHA256,
        "disclosure": DISCLOSURE,
    }
    save_json(access_event, ACCESS_EVENT_PATH)
    windows = load_window_matrix("test")
    manifest = pd.read_parquet(WINDOW_MANIFEST, engine="pyarrow")
    governance = pd.read_csv(GOVERNANCE_PATH, dtype={ID_COLUMN: str})
    return windows, manifest, governance, access_event


def prepare_representations(windows: Any, manifest: pd.DataFrame, governance: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    governed = governance.loc[governance["research_role"].eq("FINAL_TEST")].copy()
    if len(governed) != EXPECTED["customers"] or not governed["original_split"].eq("test").all():
        raise ValueError("Governed final membership changed")
    if len(manifest) != EXPECTED["windows"] or manifest["window_id"].duplicated().any():
        raise ValueError("Final window count or uniqueness changed")
    if not np.array_equal(manifest["window_id"].astype(str).to_numpy(), windows.window_ids):
        raise ValueError("Window matrix/manifest order mismatch")
    if not np.array_equal(manifest[ID_COLUMN].astype(str).to_numpy(), windows.customer_ids):
        raise ValueError("Window customer order mismatch")
    if set(windows.customer_ids) != set(governed[ID_COLUMN]):
        raise ValueError("Eligible and governed memberships differ")

    summaries = behavioural_features(windows.X)
    if summaries.shape != (EXPECTED["windows"], 17) or not np.isfinite(summaries).all():
        raise ValueError("SUMMARY_17 integrity failed")
    stage25 = json.loads(STAGE25B_INTEGRITY.read_text(encoding="utf-8"))
    mean = np.asarray(stage25["standardizer"]["mean"], dtype=np.float64)
    scale = np.asarray(stage25["standardizer"]["scale"], dtype=np.float64)
    standardized = ((summaries.astype(np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError("Standardized SUMMARY_17 is nonfinite")

    manifest = manifest.copy(); manifest[ID_COLUMN] = manifest[ID_COLUMN].astype(str); manifest["_row"] = np.arange(len(manifest))
    groups = list(manifest.groupby(ID_COLUMN, sort=False)); max_bag = max(len(group) for _, group in groups)
    x = np.zeros((len(groups), max_bag, 17), dtype=np.float32); mask = np.zeros((len(groups), max_bag), dtype=bool)
    labels = np.empty(len(groups), dtype=np.int8); ids: list[str] = []; window_ids: list[np.ndarray] = []
    for i, (customer, group) in enumerate(groups):
        group = group.sort_values("window_number_within_customer", kind="stable")
        rows = group["_row"].to_numpy(dtype=np.int64); customer_labels = windows.y[rows]
        if np.unique(customer_labels).size != 1:
            raise ValueError("Customer label inconsistency")
        x[i, :len(rows)] = standardized[rows]; mask[i, :len(rows)] = True; labels[i] = int(customer_labels[0])
        ids.append(str(customer)); window_ids.append(group["window_id"].astype(str).to_numpy())
    customer_ids = np.asarray(ids)
    observed = (len(customer_ids), int((labels == 0).sum()), int((labels == 1).sum()), int(mask.sum()))
    if observed != (EXPECTED["customers"], EXPECTED["normal"], EXPECTED["theft"], EXPECTED["windows"]):
        raise ValueError(f"Final count audit changed: {observed}")

    chronology = manifest["window_number_within_customer"].to_numpy(dtype=np.int32)
    full_frame, full_labels, full_ids, used_thresholds = build_customer_features(summaries, windows.y, windows.customer_ids, chronology, FULL48_THRESHOLDS)
    if used_thresholds != FULL48_THRESHOLDS or tuple(full_frame.columns) != FULL48_NAMES or full_frame.shape != (EXPECTED["customers"], 48):
        raise ValueError("FULL_48 representation changed")
    order = pd.Series(np.arange(len(full_ids)), index=full_ids).reindex(customer_ids)
    if order.isna().any():
        raise ValueError("FULL_48 customer alignment failed")
    indices = order.to_numpy(dtype=np.int64)
    full_matrix = full_frame.to_numpy(dtype=np.float64)[indices]
    if not np.array_equal(full_labels[indices].astype(np.int8), labels) or not np.isfinite(full_matrix).all():
        raise ValueError("FULL_48 label/finite integrity failed")

    role_sets = {role: set(governance.loc[governance["research_role"].eq(role), ID_COLUMN]) for role in ("DEV_TRAIN", "DEV_VAL", "CONFIRMATION", "FINAL_TEST")}
    overlaps = {role: len(set(customer_ids) & role_sets[role]) for role in ("DEV_TRAIN", "DEV_VAL", "CONFIRMATION")}
    if any(overlaps.values()):
        raise ValueError(f"Protected split overlap: {overlaps}")
    integrity = {
        "stage": "Stage 27B final integrity", "cohort_terminology": COHORT_NAME,
        "governed_customers": len(governed), "eligible_customers": len(customer_ids),
        "normal_customers": int((labels == 0).sum()), "theft_customers": int((labels == 1).sum()),
        "prevalence": float(labels.mean()), "eligible_windows": int(mask.sum()), "exclusions": 0,
        "duplicate_window_ids": 0, "nonfinite_SUMMARY_17": 0, "nonfinite_FULL_48": 0,
        "split_overlaps": overlaps, "customer_membership_sha256": stable_hash(customer_ids),
        "window_membership_sha256": stable_hash(np.concatenate(window_ids)), "labels_sha256": stable_hash(labels.astype(str)),
        "SUMMARY_17_dimension": 17, "FULL_48_dimension": 48, "max_bag_size": int(max_bag),
        "new_exclusion_rule": False, "label_dependent_preprocessing": False,
    }
    return torch.from_numpy(x), torch.from_numpy(mask), customer_ids, labels, full_matrix, integrity


@torch.no_grad()
def latent_features(model: DistributionModel, x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    model.eval(); parts = []
    for first in range(0, len(x), BATCH_SIZE):
        last = min(first + BATCH_SIZE, len(x)); parts.append(model.represent(x[first:last], mask[first:last]).numpy())
    result = np.concatenate(parts)
    if result.shape != (len(x), 96) or not np.isfinite(result).all():
        raise ValueError("Latent representation integrity failed")
    return result


def direct_features(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        result = masked_distribution(x, mask).numpy()
    if result.shape != (len(x), 102) or not np.isfinite(result).all():
        raise ValueError("Direct representation integrity failed")
    return result


def metric(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(scores) != len(labels) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Prediction integrity failed")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def paired_bootstraps(labels: np.ndarray, probabilities: dict[str, np.ndarray]) -> dict[str, Any]:
    comparisons = {"LATENT_minus_FULL_48": ("LATENT", "FULL_48"), "LATENT_minus_DIRECT": ("LATENT", "DIRECT"), "DIRECT_minus_FULL_48": ("DIRECT", "FULL_48")}
    observed = {name: (metric(probabilities[a], labels)[0] - metric(probabilities[b], labels)[0], metric(probabilities[a], labels)[1] - metric(probabilities[b], labels)[1]) for name, (a, b) in comparisons.items()}
    samples = {name: {"ROC_AUC": np.empty(BOOTSTRAPS), "average_precision": np.empty(BOOTSTRAPS)} for name in comparisons}
    rng = np.random.default_rng(BOOTSTRAP_SEED); attempts = 0
    for i in range(BOOTSTRAPS):
        while True:
            attempts += 1; indices = rng.integers(0, len(labels), len(labels)); y = labels[indices]
            if np.unique(y).size == 2:
                break
        values = {name: metric(scores[indices], y) for name, scores in probabilities.items()}
        for name, (a, b) in comparisons.items():
            samples[name]["ROC_AUC"][i] = values[a][0] - values[b][0]
            samples[name]["average_precision"][i] = values[a][1] - values[b][1]
    result: dict[str, Any] = {"policy": {"resamples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "paired_unit": COHORT_NAME, "shared_indices_across_all_comparisons": True, "sampling_attempts": attempts}}
    for name in comparisons:
        result[name] = {}
        for j, key in enumerate(("ROC_AUC", "average_precision")):
            values = samples[name][key]
            result[name][f"{key}_delta"] = {"observed_delta": observed[name][j], "bootstrap_mean_delta": float(values.mean()), "lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975)), "fraction_above_zero": float(np.mean(values > 0))}
    return result


def stage_stability(final: dict[str, tuple[float, float]]) -> dict[str, Any]:
    cohorts = {"DEV_VAL": DEV, "CONFIRMATION": CONFIRMATION, COHORT_NAME: final}
    result: dict[str, Any] = {"metrics": {}, "ordering": {}, "effects": {}}
    for method in ("LATENT", "FULL_48", "DIRECT"):
        result["metrics"][method] = {
            "DEV_VAL": {"ROC_AUC": DEV[method][0], "average_precision": DEV[method][1]},
            "CONFIRMATION": {"ROC_AUC": CONFIRMATION[method][0], "average_precision": CONFIRMATION[method][1]},
            "FINAL": {"ROC_AUC": final[method][0], "average_precision": final[method][1]},
            "FINAL_minus_DEV_VAL": {"ROC_AUC": final[method][0] - DEV[method][0], "average_precision": final[method][1] - DEV[method][1]},
            "FINAL_minus_CONFIRMATION": {"ROC_AUC": final[method][0] - CONFIRMATION[method][0], "average_precision": final[method][1] - CONFIRMATION[method][1]},
        }
    for cohort, values in cohorts.items():
        result["ordering"][cohort] = {"ROC_AUC": sorted(values, key=lambda name: values[name][0], reverse=True), "average_precision": sorted(values, key=lambda name: values[name][1], reverse=True)}
        result["effects"][cohort] = {
            "LATENT_minus_FULL_48": {"ROC_AUC": values["LATENT"][0] - values["FULL_48"][0], "average_precision": values["LATENT"][1] - values["FULL_48"][1]},
            "LATENT_minus_DIRECT": {"ROC_AUC": values["LATENT"][0] - values["DIRECT"][0], "average_precision": values["LATENT"][1] - values["DIRECT"][1]},
        }
    final_effects = result["effects"][COHORT_NAME]
    result["qualitative_direction_persists"] = bool(all(result["effects"][cohort][comparison][metric] > 0 for cohort in cohorts for comparison in ("LATENT_minus_FULL_48", "LATENT_minus_DIRECT") for metric in ("ROC_AUC", "average_precision")))
    result["final_effects_positive"] = bool(all(value > 0 for comparison in final_effects.values() for value in comparison.values()))
    return result


def classify(seed_frame: pd.DataFrame, primary: dict[str, tuple[float, float]], boot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    latent, full, direct = primary["LATENT"], primary["FULL_48"], primary["DIRECT"]
    lf = (latent[0] - full[0], latent[1] - full[1]); ld = (latent[0] - direct[0], latent[1] - direct[1])
    lf_boot, ld_boot = boot["LATENT_minus_FULL_48"], boot["LATENT_minus_DIRECT"]
    parity_count = int(seed_frame["within_FULL_48_parity_margins"].sum())
    roc_sd = float(seed_frame["ROC_AUC"].std(ddof=0)); ap_sd = float(seed_frame["average_precision"].std(ddof=0))
    seed_roc_delta = seed_frame["ROC_AUC"] - full[0]; seed_ap_delta = seed_frame["average_precision"] - full[1]
    collapse = {
        "fewer_than_4_of_5_parity_seeds": parity_count < 4,
        "ROC_population_SD_above_0.015": roc_sd > 0.015,
        "AP_population_SD_above_0.025": ap_sd > 0.025,
        "any_seed_ROC_more_than_0.030_below_FULL_48": bool((seed_roc_delta < -0.030).any()),
        "any_seed_AP_more_than_0.060_below_FULL_48": bool((seed_ap_delta < -0.060).any()),
    }
    meaningful_roc = ld[0] >= 0.005 and ld_boot["ROC_AUC_delta"]["lower_95"] > 0
    meaningful_ap = ld[1] >= 0.010 and ld_boot["average_precision_delta"]["lower_95"] > 0
    other_support = ((meaningful_roc and ld_boot["average_precision_delta"]["fraction_above_zero"] > 0.5 and ld_boot["average_precision_delta"]["upper_95"] > 0) or (meaningful_ap and ld_boot["ROC_AUC_delta"]["fraction_above_zero"] > 0.5 and ld_boot["ROC_AUC_delta"]["upper_95"] > 0))
    a_conditions = {
        "LATENT_ROC_within_FULL_48_parity": lf[0] >= -0.005,
        "LATENT_AP_within_FULL_48_parity": lf[1] >= -0.010,
        "LATENT_FULL_ROC_lower_bound_within_margin": lf_boot["ROC_AUC_delta"]["lower_95"] >= -0.005,
        "LATENT_FULL_AP_lower_bound_within_margin": lf_boot["average_precision_delta"]["lower_95"] >= -0.010,
        "LATENT_DIRECT_both_observed_positive": ld[0] > 0 and ld[1] > 0,
        "LATENT_DIRECT_meaningful_supported_gain": meaningful_roc or meaningful_ap,
        "LATENT_DIRECT_other_metric_supportive": other_support,
        "at_least_4_of_5_parity_seeds": parity_count >= 4,
        "no_concerning_seed_collapse": not any(collapse.values()),
    }
    median_roc_delta = float(seed_frame["ROC_AUC"].median() - full[0]); median_ap_delta = float(seed_frame["average_precision"].median() - full[1])
    c_conditions = {
        "LATENT_ROC_materially_inferior_to_FULL_48": lf[0] < -0.015,
        "LATENT_AP_materially_inferior_to_FULL_48": lf[1] < -0.030,
        "LATENT_DIRECT_AP_supported_nonpositive": ld[1] <= 0 and ld_boot["average_precision_delta"]["upper_95"] <= 0,
        "two_or_fewer_parity_seeds": parity_count <= 2,
        "median_seed_materially_inferior": median_roc_delta < -0.015 or median_ap_delta < -0.030,
    }
    category = "FINAL A — STRONG FINAL REPLICATION" if all(a_conditions.values()) else ("FINAL C — FAILED FINAL REPLICATION" if any(c_conditions.values()) else "FINAL B — PARTIAL / MIXED FINAL REPLICATION")
    outcome = {"classification": category, "FINAL_A_criterion_audit": a_conditions, "FINAL_C_criterion_audit": c_conditions, "seed_collapse_audit": collapse, "parity_seed_count": parity_count, "concerning_seed_collapse": any(collapse.values())}
    if category.startswith("FINAL A"):
        anchor_class = "A — ANCHOR SURVIVES"
        sentence = "Learning window representations before distribution-preserving customer aggregation adds reproducible customer-level discrimination beyond identical direct aggregation across development, confirmation, and the governed final evaluation."
    elif category.startswith("FINAL B") and lf[0] >= -0.015 and lf[1] >= -0.030 and ld[1] > 0 and ld_boot["average_precision_delta"]["fraction_above_zero"] > 0.5 and ld_boot["average_precision_delta"]["upper_95"] > 0 and parity_count >= 3:
        anchor_class = "B — ANCHOR SURVIVES WITH QUALIFICATION"
        sentence = "Learned distribution-preserving aggregation retains qualified customer-level value across governed stages, although the terminal evaluation does not satisfy every strong-replication condition."
    else:
        anchor_class = "C — ANCHOR DOES NOT SURVIVE FINAL EVALUATION"; sentence = None
    anchor = {"classification": anchor_class, "anchor_sentence": sentence, "stage26_anchor_reassessed": True}
    return outcome, anchor


def main() -> None:
    prereg, model_freeze, comparator_freeze = validate_seal()
    windows, manifest, governance, access_event = open_cohort_once()
    x, mask, customer_ids, labels, full_matrix, integrity = prepare_representations(windows, manifest, governance)
    integrity["access_event_timestamp"] = access_event["timestamp"]; save_json(integrity, RESULTS_DIR / "final_integrity.json")

    full_model = joblib.load(PROJECT_ROOT / comparator_freeze["comparators"]["FULL_48_HGB"]["model"])
    direct_model = joblib.load(PROJECT_ROOT / comparator_freeze["comparators"]["DIRECT_SUMMARY_DISTRIBUTION_HGB"]["model"])
    full_probability = full_model.predict_proba(full_matrix)[:, 1]
    direct_matrix = direct_features(x, mask); direct_probability = direct_model.predict_proba(direct_matrix)[:, 1]
    seed_probabilities: dict[int, np.ndarray] = {}; seed_rows = []
    full_metrics = metric(full_probability, labels)
    for record in model_freeze["checkpoints"]:
        seed = int(record["seed"]); model = DistributionModel()
        checkpoint = torch.load(PROJECT_ROOT / record["encoder"], map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"]); representation = latent_features(model, x, mask)
        hgb = joblib.load(PROJECT_ROOT / record["hgb"]); probability = hgb.predict_proba(representation)[:, 1]
        roc, ap = metric(probability, labels); seed_probabilities[seed] = probability
        seed_rows.append({"model": "LATENT_DISTRIBUTION_HGB", "seed": seed, "customers": len(labels), "ROC_AUC": roc, "average_precision": ap, "ROC_delta_vs_FULL_48": roc - full_metrics[0], "AP_delta_vs_FULL_48": ap - full_metrics[1], "within_FULL_48_parity_margins": bool(roc - full_metrics[0] >= -0.005 and ap - full_metrics[1] >= -0.010)})
    seed_frame = pd.DataFrame(seed_rows); save_csv(seed_frame, RESULTS_DIR / "final_seed_metrics.csv")
    latent_probability = sum(seed_probabilities[seed] for seed in SEEDS) / 5.0
    stacked_mean = np.mean(np.stack([seed_probabilities[seed] for seed in SEEDS]), axis=0)
    if not np.allclose(latent_probability, stacked_mean, atol=1e-15, rtol=0.0):
        raise ValueError("Five-seed arithmetic ensemble assertion failed")
    probabilities = {"LATENT": latent_probability, "FULL_48": full_probability, "DIRECT": direct_probability}
    primary = {name: metric(scores, labels) for name, scores in probabilities.items()}
    primary_frame = pd.DataFrame([{"model": name, "customers": len(labels), "ROC_AUC": values[0], "average_precision": values[1]} for name, values in primary.items()])
    save_csv(primary_frame, RESULTS_DIR / "final_primary_metrics.csv")
    boot = paired_bootstraps(labels, probabilities); save_json(boot, RESULTS_DIR / "final_bootstrap_comparisons.json")
    stability = stage_stability(primary)
    save_json(stability, RESULTS_DIR / "final_stage_stability.json")
    save_json(stability, RESULTS_DIR / "development_confirmation_final_stability.json")
    outcome, anchor = classify(seed_frame, primary, boot)
    outcome.update({"cohort_terminology": COHORT_NAME, "seed_summary": {metric_name: {"mean": float(seed_frame[metric_name].mean()), "std_population": float(seed_frame[metric_name].std(ddof=0)), "median": float(seed_frame[metric_name].median()), "min": float(seed_frame[metric_name].min()), "max": float(seed_frame[metric_name].max())} for metric_name in ("ROC_AUC", "average_precision")}, "no_retraining_or_refitting": True, "no_threshold_optimization": True, "model_development_closed": True})
    save_json(outcome, RESULTS_DIR / "final_outcome.json")
    save_json(outcome, RESULTS_DIR / "stage27b_outcome.json")
    save_json(anchor, RESULTS_DIR / "final_anchor_audit.json")

    if outcome["classification"].startswith("FINAL A"):
        allowed = prereg["allowed_claims_if_supported"]
        level = "LEVEL 1 — strong methodological result"
    elif anchor["classification"].startswith("B"):
        allowed = [claim for claim in prereg["allowed_claims_if_supported"] if not claim.startswith("Reproducible across")]
        level = "LEVEL 2 — useful but qualified empirical result"
    else:
        allowed = ["Development and confirmation evidence for a learned latent-distribution representation", "Terminal result reported without a final-supported anchor"]
        level = "LEVEL 3 — development/confirmation result that fails final support"
    claims = {"allowed_claims": allowed, "prohibited_claims": prereg["prohibited_claims"], "paperworthiness": level, "required_disclosure": DISCLOSURE}
    save_json(claims, RESULTS_DIR / "final_claim_audit.json")

    latent, full, direct = primary["LATENT"], primary["FULL_48"], primary["DIRECT"]
    report = f"""# Stage 27B — terminal Study 2 evaluation\n\n## Cohort governance\n\n{DISCLOSURE}\n\nThe cohort was opened exactly once at `{access_event['timestamp']}` after preregistration `{PREREG_SHA256}` and every pre-access assertion passed. It contained {len(labels):,} customers ({int((labels == 0).sum()):,} normal; {int((labels == 1).sum()):,} theft-labelled), prevalence {labels.mean():.6f}, and {int(mask.sum()):,} eligible windows, with zero exclusions or protected-role overlap.\n\n## Frozen results\n\nLATENT five-seed mean: `{latent[0]:.6f}` ROC / `{latent[1]:.6f}` AP. FULL_48: `{full[0]:.6f}/{full[1]:.6f}`. DIRECT: `{direct[0]:.6f}/{direct[1]:.6f}`. No model was fitted, refit, calibrated, tuned, or selected, and no threshold was optimized.\n\nThe seed-level results, paired uncertainty, stage stability, and exact criterion audits are recorded in the accompanying CSV/JSON artifacts.\n\n## Terminal decision\n\n**{outcome['classification']}**. **{anchor['classification']}**.\n\n{('Final anchor sentence: **' + anchor['anchor_sentence'] + '**') if anchor['anchor_sentence'] else 'The Stage 26 anchor does not survive the terminal evaluation.'}\n\nPaperworthiness: **{level}**. Study 2 model development is permanently closed; no rescue experiment is permitted.\n"""
    (RESULTS_DIR / "stage27b_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"access_event": access_event, "integrity": integrity, "seed_metrics": seed_rows, "primary": primary_frame.to_dict(orient="records"), "outcome": outcome, "anchor": anchor, "paperworthiness": level}, indent=2))


if __name__ == "__main__":
    main()
