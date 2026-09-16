"""Stage 26: one frozen Study 2 evaluation on governed CONFIRMATION.

The only row-level feature loader is historical ``val``, which governance maps
exactly to CONFIRMATION. This module has no FINAL_TEST loader and performs no fit.
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

from src.analysis.run_stage14_representation import FEATURE_NAMES, behavioural_features
from src.analysis.run_stage25b_distribution_experiment import (
    BATCH_SIZE,
    DistributionModel,
    masked_distribution,
)
from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    file_sha256,
    load_window_matrix,
    save_csv,
    save_json,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_26"
PROTOCOL_PATH = RESULTS_DIR / "stage26_confirmation_protocol.json"
EXPECTED_PROTOCOL_SHA256 = "7207a75dfd31d3254aaa3d4507eb9812b38ff87ecda09abfa62178572c8be09b"
SOURCE_COMMIT = "4b7741b6126aa09380627a9a0845d3a95f066d55"
STUDY1_COMMIT = "d0601c54f6e9d0f11caac2ebd27f7c86a568a863"
GOVERNANCE_PATH = PROJECT_ROOT / "results" / "governance" / "stage_17a" / "research_protocol_manifest.csv"
WINDOW_MANIFEST = PROJECT_ROOT / "data" / "windows" / "stride_30" / "usable" / "val_windows.parquet"
STAGE25B_INTEGRITY = RESULTS_DIR.parent / "stage_25b" / "bag_integrity.json"
FULL48_PREDICTIONS = RESULTS_DIR.parent / "stage_17c" / "confirmation_customer_probabilities.csv"
SEEDS = (42, 43, 44, 45, 46)
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2502
FULL48_EXPECTED = (0.789084048160282, 0.3790578558669986)
DEV = {
    "LATENT_DISTRIBUTION_HGB": (0.7908091586895513, 0.36783736705395026),
    "FULL_48_HGB": (0.7888308218953116, 0.35979459032174654),
    "DIRECT_SUMMARY_DISTRIBUTION_HGB": (0.7882334049844351, 0.3492440521718474),
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, encoding="utf-8").strip()


def stable_hash(values: np.ndarray | list[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def validate_pre_access_seal() -> dict[str, Any]:
    if git("branch", "--show-current") != "study2/mil" or git("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ValueError("Branch/source HEAD assertion failed")
    if git("rev-parse", "main") != STUDY1_COMMIT:
        raise ValueError("Frozen Study 1 main moved")
    if file_sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Stage 26 confirmation protocol changed")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if tuple(protocol["models"]["LATENT_DISTRIBUTION_HGB"]["seeds"]) != SEEDS:
        raise ValueError("Frozen seed set changed")
    if protocol["paired_bootstrap"]["resamples"] != BOOTSTRAPS or protocol["paired_bootstrap"]["seed"] != BOOTSTRAP_SEED:
        raise ValueError("Bootstrap policy changed")
    if file_sha256(RESULTS_DIR.parent / "stage_25b" / "stage25b_execution_spec.json") != protocol["inputs"]["stage25b_execution_spec_sha256"]:
        raise ValueError("Stage 25B execution spec changed")
    for checkpoint in protocol["models"]["LATENT_DISTRIBUTION_HGB"]["checkpoints"]:
        if file_sha256(PROJECT_ROOT / checkpoint["encoder"]) != checkpoint["encoder_sha256"]:
            raise ValueError(f"Encoder checkpoint changed for seed {checkpoint['seed']}")
        if file_sha256(PROJECT_ROOT / checkpoint["hgb"]) != checkpoint["hgb_sha256"]:
            raise ValueError(f"HGB checkpoint changed for seed {checkpoint['seed']}")
    direct = protocol["models"]["DIRECT_SUMMARY_DISTRIBUTION_HGB"]
    if file_sha256(PROJECT_ROOT / direct["checkpoint"]) != direct["sha256"]:
        raise ValueError("Direct HGB checkpoint changed")
    full = protocol["models"]["FULL_48_HGB"]
    if file_sha256(PROJECT_ROOT / full["confirmation_predictions"]) != full["predictions_sha256"]:
        raise ValueError("Frozen FULL_48 predictions changed")
    return protocol


def load_confirmation_once(protocol: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, list[np.ndarray], dict[str, Any]]:
    access_timestamp = datetime.now().astimezone().isoformat()
    governance = pd.read_csv(GOVERNANCE_PATH, dtype={ID_COLUMN: str})
    governed = governance.loc[governance["research_role"].eq("CONFIRMATION")].copy()
    if len(governed) != protocol["data"]["expected_customers"] or not governed["original_split"].eq("val").all():
        raise ValueError("Governed CONFIRMATION membership changed")

    windows = load_window_matrix("val")
    manifest = pd.read_parquet(WINDOW_MANIFEST, engine="pyarrow")
    if len(manifest) != protocol["data"]["expected_windows"]:
        raise ValueError("CONFIRMATION window count changed")
    if manifest["window_id"].duplicated().any():
        raise ValueError("Duplicate CONFIRMATION window IDs")
    if not np.array_equal(manifest["window_id"].astype(str).to_numpy(), windows.window_ids):
        raise ValueError("CONFIRMATION matrix/manifest order mismatch")
    if set(manifest[ID_COLUMN].astype(str)) != set(governed[ID_COLUMN].astype(str)):
        raise ValueError("Eligible and governed customer membership differ")

    config = json.loads((RESULTS_DIR.parent / "stage_14" / "preregistered_config.json").read_text(encoding="utf-8"))
    if tuple(config["feature_names_in_order"]) != FEATURE_NAMES or len(FEATURE_NAMES) != 17:
        raise ValueError("SUMMARY_17 schema changed")
    summaries = behavioural_features(windows.X)
    if summaries.shape != (protocol["data"]["expected_windows"], 17) or not np.isfinite(summaries).all():
        raise ValueError("CONFIRMATION SUMMARY_17 integrity failed")
    stage25_integrity = json.loads(STAGE25B_INTEGRITY.read_text(encoding="utf-8"))
    mean = np.asarray(stage25_integrity["standardizer"]["mean"], dtype=np.float64)
    scale = np.asarray(stage25_integrity["standardizer"]["scale"], dtype=np.float64)
    standardized = ((summaries.astype(np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError("Standardized CONFIRMATION features are nonfinite")

    manifest = manifest.copy()
    manifest[ID_COLUMN] = manifest[ID_COLUMN].astype(str)
    manifest["_row"] = np.arange(len(manifest), dtype=np.int64)
    groups = list(manifest.groupby(ID_COLUMN, sort=False))
    max_bag_size = max(len(group) for _, group in groups)
    x = np.zeros((len(groups), max_bag_size, 17), dtype=np.float32)
    mask = np.zeros((len(groups), max_bag_size), dtype=bool)
    labels = np.empty(len(groups), dtype=np.int8)
    customer_ids: list[str] = []
    window_ids: list[np.ndarray] = []
    for i, (customer, group) in enumerate(groups):
        group = group.sort_values("window_number_within_customer", kind="stable")
        rows = group["_row"].to_numpy(dtype=np.int64)
        if group[LABEL_COLUMN].nunique() != 1:
            raise ValueError("Customer bag has inconsistent labels")
        x[i, :len(rows)] = standardized[rows]
        mask[i, :len(rows)] = True
        labels[i] = int(group[LABEL_COLUMN].iloc[0])
        customer_ids.append(str(customer))
        window_ids.append(group["window_id"].astype(str).to_numpy())

    customer_array = np.asarray(customer_ids)
    if len(customer_array) != 6355 or int((labels == 0).sum()) != 5813 or int((labels == 1).sum()) != 542:
        raise ValueError("CONFIRMATION customer/class counts changed")
    integrity = {
        "stage": "Stage 26 confirmation integrity",
        "access_event": {
            "timestamp": access_timestamp,
            "event": "Opened historical val exactly once as governed CONFIRMATION for frozen Stage 26 inference.",
            "confirmation_access_count_for_stage26": 1,
            "final_test_access_count_for_stage26": 0
        },
        "governed_customers": int(len(governed)), "eligible_customers": int(len(customer_array)),
        "normal_customers": int((labels == 0).sum()), "theft_customers": int((labels == 1).sum()),
        "prevalence": float(labels.mean()), "windows": int(mask.sum()), "exclusions": 0,
        "max_bag_size": int(max_bag_size), "duplicate_window_ids": 0,
        "nonfinite_summary17": int((~np.isfinite(standardized)).sum()),
        "customer_membership_sha256": stable_hash(customer_array),
        "window_membership_sha256": stable_hash(np.concatenate(window_ids)),
        "label_sequence_sha256": stable_hash(labels.astype(str)),
        "governance_sha256": file_sha256(GOVERNANCE_PATH), "window_manifest_sha256": file_sha256(WINDOW_MANIFEST),
        "DEV_CONFIRMATION_customer_overlap": 0, "FINAL_TEST_customers_loaded": 0,
        "SUMMARY_17_features_in_order": list(FEATURE_NAMES), "new_exclusions": False,
    }
    development_ids = set(governance.loc[governance["research_role"].isin(["DEV_TRAIN", "DEV_VAL"]), ID_COLUMN])
    integrity["DEV_CONFIRMATION_customer_overlap"] = len(development_ids & set(customer_array))
    if integrity["DEV_CONFIRMATION_customer_overlap"]:
        raise ValueError("Development/CONFIRMATION overlap")
    return torch.from_numpy(x), torch.from_numpy(mask), customer_array, labels, window_ids, integrity


@torch.no_grad()
def latent_representation(model: DistributionModel, x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    model.eval(); parts = []
    for first in range(0, len(x), BATCH_SIZE):
        last = min(first + BATCH_SIZE, len(x))
        parts.append(model.represent(x[first:last], mask[first:last]).numpy())
    result = np.concatenate(parts)
    if result.shape != (len(x), 96) or not np.isfinite(result).all():
        raise ValueError("Latent representation integrity failed")
    return result


def direct_representation(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        result = masked_distribution(x, mask).numpy()
    if result.shape != (len(x), 102) or not np.isfinite(result).all():
        raise ValueError("Direct representation integrity failed")
    return result


def metrics(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(scores) != len(labels) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Prediction integrity failed")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def paired_bootstrap(labels: np.ndarray, candidate: np.ndarray, comparator: np.ndarray) -> dict[str, Any]:
    observed_roc, observed_ap = metrics(candidate, labels)
    comparator_roc, comparator_ap = metrics(comparator, labels)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    roc_values, ap_values = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
    attempts = 0
    for i in range(BOOTSTRAPS):
        while True:
            attempts += 1; indices = rng.integers(0, len(labels), len(labels)); y = labels[indices]
            if np.unique(y).size == 2:
                break
        roc_values[i] = roc_auc_score(y, candidate[indices]) - roc_auc_score(y, comparator[indices])
        ap_values[i] = average_precision_score(y, candidate[indices]) - average_precision_score(y, comparator[indices])
    def describe(observed: float, values: np.ndarray) -> dict[str, float]:
        return {"observed_delta": observed, "bootstrap_mean_delta": float(values.mean()), "lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975)), "fraction_above_zero": float(np.mean(values > 0))}
    return {"resamples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "sampling_attempts": attempts, "ROC_AUC_delta": describe(observed_roc - comparator_roc, roc_values), "average_precision_delta": describe(observed_ap - comparator_ap, ap_values)}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    protocol = validate_pre_access_seal()
    x, mask, customer_ids, labels, _window_ids, integrity = load_confirmation_once(protocol)
    save_json(integrity, RESULTS_DIR / "confirmation_integrity.json")

    full_frame = pd.read_csv(FULL48_PREDICTIONS, dtype={ID_COLUMN: str})
    full_frame = full_frame.loc[full_frame["model"].eq("stage17b_frozen_direct_customer_hgb")].set_index(ID_COLUMN).reindex(customer_ids)
    if full_frame["probability"].isna().any() or not np.array_equal(full_frame[LABEL_COLUMN].to_numpy(np.int8), labels):
        raise ValueError("Frozen FULL_48 predictions do not align")
    full_probability = full_frame["probability"].to_numpy(float)
    if not np.allclose(metrics(full_probability, labels), FULL48_EXPECTED, atol=1e-14):
        raise ValueError("Frozen FULL_48 confirmation metrics changed")

    direct_features = direct_representation(x, mask)
    direct_model = joblib.load(PROJECT_ROOT / protocol["models"]["DIRECT_SUMMARY_DISTRIBUTION_HGB"]["checkpoint"])
    direct_probability = direct_model.predict_proba(direct_features)[:, 1]
    seed_probabilities: dict[int, np.ndarray] = {}
    seed_rows = []
    for record in protocol["models"]["LATENT_DISTRIBUTION_HGB"]["checkpoints"]:
        seed = int(record["seed"]); model = DistributionModel()
        checkpoint = torch.load(PROJECT_ROOT / record["encoder"], map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        representation = latent_representation(model, x, mask)
        hgb = joblib.load(PROJECT_ROOT / record["hgb"])
        probability = hgb.predict_proba(representation)[:, 1]
        roc, ap = metrics(probability, labels); seed_probabilities[seed] = probability
        seed_rows.append({"model": "LATENT_DISTRIBUTION_HGB", "seed": seed, "customers": len(labels), "ROC_AUC": roc, "average_precision": ap, "within_FULL_48_parity_margins": bool(roc - FULL48_EXPECTED[0] >= -0.005 and ap - FULL48_EXPECTED[1] >= -0.010)})

    seed_metrics = pd.DataFrame(seed_rows); save_csv(seed_metrics, RESULTS_DIR / "confirmation_seed_metrics.csv")
    latent_probability = np.mean(np.stack([seed_probabilities[seed] for seed in SEEDS]), axis=0)
    latent_roc, latent_ap = metrics(latent_probability, labels); direct_roc, direct_ap = metrics(direct_probability, labels)
    primary = pd.DataFrame([
        {"model": "LATENT_DISTRIBUTION_HGB_FIVE_SEED_MEAN", "customers": len(labels), "ROC_AUC": latent_roc, "average_precision": latent_ap},
        {"model": "FULL_48_HGB", "customers": len(labels), "ROC_AUC": FULL48_EXPECTED[0], "average_precision": FULL48_EXPECTED[1]},
        {"model": "DIRECT_SUMMARY_DISTRIBUTION_HGB", "customers": len(labels), "ROC_AUC": direct_roc, "average_precision": direct_ap},
    ]); save_csv(primary, RESULTS_DIR / "confirmation_primary_metrics.csv")

    bootstraps = {
        "policy": {"resamples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "unit": "CONFIRMATION customer", "interval": "percentile 95%"},
        "LATENT_minus_FULL_48": paired_bootstrap(labels, latent_probability, full_probability),
        "LATENT_minus_DIRECT": paired_bootstrap(labels, latent_probability, direct_probability),
        "DIRECT_minus_FULL_48": paired_bootstrap(labels, direct_probability, full_probability),
    }; save_json(bootstraps, RESULTS_DIR / "confirmation_bootstrap_comparisons.json")

    confirmation = {"LATENT_DISTRIBUTION_HGB": (latent_roc, latent_ap), "FULL_48_HGB": FULL48_EXPECTED, "DIRECT_SUMMARY_DISTRIBUTION_HGB": (direct_roc, direct_ap)}
    stability = {model: {"DEV_VAL_ROC_AUC": DEV[model][0], "CONFIRMATION_ROC_AUC": values[0], "ROC_AUC_shift": values[0] - DEV[model][0], "DEV_VAL_average_precision": DEV[model][1], "CONFIRMATION_average_precision": values[1], "average_precision_shift": values[1] - DEV[model][1]} for model, values in confirmation.items()}
    dev_order_roc = sorted(DEV, key=lambda model: DEV[model][0], reverse=True); dev_order_ap = sorted(DEV, key=lambda model: DEV[model][1], reverse=True)
    conf_order_roc = sorted(confirmation, key=lambda model: confirmation[model][0], reverse=True); conf_order_ap = sorted(confirmation, key=lambda model: confirmation[model][1], reverse=True)
    stability["ordering"] = {"DEV_VAL_ROC": dev_order_roc, "CONFIRMATION_ROC": conf_order_roc, "DEV_VAL_AP": dev_order_ap, "CONFIRMATION_AP": conf_order_ap, "ROC_order_replicated": dev_order_roc == conf_order_roc, "AP_order_replicated": dev_order_ap == conf_order_ap}
    save_json(stability, RESULTS_DIR / "development_confirmation_stability.json")

    parity_count = int(seed_metrics["within_FULL_48_parity_margins"].sum())
    latent_full_roc, latent_full_ap = latent_roc - FULL48_EXPECTED[0], latent_ap - FULL48_EXPECTED[1]
    latent_direct_roc, latent_direct_ap = latent_roc - direct_roc, latent_ap - direct_ap
    direct_full_roc, direct_full_ap = direct_roc - FULL48_EXPECTED[0], direct_ap - FULL48_EXPECTED[1]
    learned_ap_boot = bootstraps["LATENT_minus_DIRECT"]["average_precision_delta"]
    positive = bool(latent_full_roc >= -0.005 and latent_full_ap >= -0.010 and parity_count >= 4 and latent_direct_ap > 0 and learned_ap_boot["fraction_above_zero"] > 0.5 and learned_ap_boot["upper_95"] > 0)
    failed = bool(latent_full_roc < -0.015 or latent_full_ap < -0.030 or (latent_direct_ap <= 0 and learned_ap_boot["upper_95"] <= 0))
    category = "A" if positive else ("C" if failed else "B")
    labels_by_category = {"A": "POSITIVE REPLICATION", "B": "PARTIAL / MIXED REPLICATION", "C": "FAILED REPLICATION"}
    full_boot = bootstraps["LATENT_minus_FULL_48"]
    anchor = bool(category == "A" and latent_direct_ap >= 0.010 and learned_ap_boot["lower_95"] > 0 and latent_full_ap >= 0 and full_boot["average_precision_delta"]["lower_95"] > 0)
    seed_summary = {metric: {"mean": float(seed_metrics[metric].mean()), "std_population": float(seed_metrics[metric].std(ddof=0)), "median": float(seed_metrics[metric].median()), "min": float(seed_metrics[metric].min()), "max": float(seed_metrics[metric].max())} for metric in ("ROC_AUC", "average_precision")}
    outcome = {
        "confirmation_classification": category, "classification_label": labels_by_category[category],
        "reason": "Complete positive-replication criterion satisfied." if category == "A" else ("Material inferiority or independently unsupported learned-encoder advantage." if category == "C" else "Some Stage 25B behavior replicated, but the complete positive-replication criterion failed."),
        "parity_seed_count": parity_count, "seed_summary": seed_summary,
        "observed_deltas": {"LATENT_minus_FULL_48": {"ROC_AUC": latent_full_roc, "average_precision": latent_full_ap}, "LATENT_minus_DIRECT": {"ROC_AUC": latent_direct_roc, "average_precision": latent_direct_ap}, "DIRECT_minus_FULL_48": {"ROC_AUC": direct_full_roc, "average_precision": direct_full_ap}},
        "ANCHOR_EARNED": "YES" if anchor else "NO",
        "anchor_sentence": "Distribution-preserving learned window representations reproducibly add customer-level precision-recall signal beyond direct distribution aggregation across development and independent confirmation." if anchor else None,
        "no_retraining_or_tuning": True, "instance_level_FLAG_supervision": False, "final_test_access": False,
    }; save_json(outcome, RESULTS_DIR / "stage26_outcome.json")

    report = f"""# Stage 26 — independent confirmation of learned distribution representation\n\n## Governed access\n\nProtocol SHA-256: `{EXPECTED_PROTOCOL_SHA256}`. CONFIRMATION was opened once at `{integrity['access_event']['timestamp']}` for frozen inference. It contained {len(labels):,} customers ({int((labels == 1).sum()):,} theft-labelled) and {int(mask.sum()):,} eligible windows. FINAL_TEST remained closed.\n\n## Results\n\nFive-seed mean LATENT scored ROC-AUC `{latent_roc:.6f}` and AP `{latent_ap:.6f}`. Frozen FULL_48 scored `{FULL48_EXPECTED[0]:.6f}/{FULL48_EXPECTED[1]:.6f}`; frozen DIRECT scored `{direct_roc:.6f}/{direct_ap:.6f}`. LATENT deltas were `{latent_full_roc:+.6f}/{latent_full_ap:+.6f}` versus FULL_48 and `{latent_direct_roc:+.6f}/{latent_direct_ap:+.6f}` versus DIRECT.\n\nIndividual latent seed results are recorded in `confirmation_seed_metrics.csv`; {parity_count}/5 satisfied both parity margins. The three preregistered paired comparisons, including intervals and fractions above zero, are in `confirmation_bootstrap_comparisons.json`.\n\n## Decision\n\n**CONFIRMATION {category} — {labels_by_category[category]}**. **ANCHOR_EARNED = {'YES' if anchor else 'NO'}**. {'The independent result supports a non-tiny, uncertainty-supported learned-encoder AP contribution across cohorts.' if anchor else 'The independent result is scientifically useful replication evidence but is insufficient for a paper anchor under the strict frozen policy.'}\n\nNo model was fitted, selected, tuned, or altered using confirmation. No instance-level FLAG supervision was used. FINAL_TEST was not accessed.\n"""
    (RESULTS_DIR / "stage26_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"protocol_sha256": EXPECTED_PROTOCOL_SHA256, "primary": primary.to_dict(orient="records"), "seed_metrics": seed_rows, "outcome": outcome, "access_event": integrity["access_event"]}, indent=2))


if __name__ == "__main__":
    main()
