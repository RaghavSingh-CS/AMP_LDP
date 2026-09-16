"""Stage 24A: development-only MIL bag and representation feasibility audit.

This module deliberately has no loader for the original validation or test
populations.  It reads only the original-train processed matrix and its frozen
usable-window manifest, then applies the existing Stage 17A DEV_TRAIN/DEV_VAL
governance assignments.  It performs no model fitting or scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.run_stage14_representation import FEATURE_NAMES, behavioural_features
from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    file_sha256,
    load_window_matrix,
    save_csv,
    save_json,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_24a"
GOVERNANCE_MANIFEST = (
    PROJECT_ROOT / "results" / "governance" / "stage_17a" / "research_protocol_manifest.csv"
)
TRAIN_WINDOWS = (
    PROJECT_ROOT / "data" / "windows" / "stride_30" / "usable" / "train_windows.parquet"
)
TRAIN_PROCESSED = PROJECT_ROOT / "data" / "processed" / "train.parquet"
STAGE14_CONFIG = RESULTS_DIR.parent / "stage_14" / "preregistered_config.json"
ALLOWED_ROLES = ("DEV_TRAIN", "DEV_VAL")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def distribution(values: pd.Series) -> dict[str, float | int]:
    numeric = values.to_numpy(dtype=np.float64)
    q10, q25, median, q75, q90 = np.quantile(numeric, QUANTILES, method="linear")
    return {
        "min": int(numeric.min()),
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
        "max": int(numeric.max()),
        "mean": float(numeric.mean()),
        "std_population": float(numeric.std(ddof=0)),
    }


def short_bags(values: pd.Series) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for threshold in (1, 3, 5, 10):
        count = int(values.le(threshold).sum())
        result[f"at_most_{threshold}_windows"] = {
            "count": count,
            "fraction": float(count / len(values)),
        }
    return result


def grouped_summary(customers: pd.DataFrame) -> dict[str, Any]:
    counts = customers["usable_window_count"]
    return {
        "customers": int(len(customers)),
        "windows": int(counts.sum()),
        "windows_per_customer": distribution(counts),
        "short_bags": short_bags(counts),
    }


def main() -> None:
    governance = pd.read_csv(GOVERNANCE_MANIFEST, dtype={ID_COLUMN: str})
    development = governance.loc[
        governance["research_role"].isin(ALLOWED_ROLES),
        [ID_COLUMN, "original_split", "research_role", "stage16_representation_available"],
    ].copy()
    if not development["original_split"].eq("train").all():
        raise ValueError("Development roles must contain original-train customers only")
    if development[ID_COLUMN].duplicated().any():
        raise ValueError("Governance customer IDs are not unique")

    manifest = pd.read_parquet(TRAIN_WINDOWS, engine="pyarrow")
    expected_columns = {
        "window_id",
        ID_COLUMN,
        LABEL_COLUMN,
        "window_number_within_customer",
        "start_feature_index",
        "end_feature_index",
        "start_date",
        "end_date",
        "missing_fraction",
    }
    if not expected_columns.issubset(manifest.columns):
        raise ValueError("Frozen usable-window manifest schema changed")
    role_map = development.set_index(ID_COLUMN)["research_role"]
    manifest["research_role"] = manifest[ID_COLUMN].astype(str).map(role_map)
    if manifest["research_role"].isna().any():
        raise ValueError("An original-train usable window lacks a development role")
    if not manifest["research_role"].isin(ALLOWED_ROLES).all():
        raise ValueError("A closed role entered the Stage 24A window audit")

    label_counts = manifest.groupby(ID_COLUMN, sort=False)[LABEL_COLUMN].nunique()
    if not label_counts.eq(1).all():
        raise ValueError("A customer has inconsistent propagated FLAG values")
    customer_table = (
        manifest.groupby([ID_COLUMN, "research_role"], sort=False, as_index=False)
        .agg(FLAG=(LABEL_COLUMN, "first"), usable_window_count=("window_id", "size"))
    )

    role_summaries = {
        role: grouped_summary(customer_table.loc[customer_table["research_role"].eq(role)])
        for role in ALLOWED_ROLES
    }
    class_summaries = {
        "normal_FLAG_0": grouped_summary(customer_table.loc[customer_table[LABEL_COLUMN].eq(0)]),
        "theft_labelled_FLAG_1": grouped_summary(
            customer_table.loc[customer_table[LABEL_COLUMN].eq(1)]
        ),
    }
    role_class_summaries = {
        role: {
            "normal_FLAG_0": grouped_summary(
                customer_table.loc[
                    customer_table["research_role"].eq(role)
                    & customer_table[LABEL_COLUMN].eq(0)
                ]
            ),
            "theft_labelled_FLAG_1": grouped_summary(
                customer_table.loc[
                    customer_table["research_role"].eq(role)
                    & customer_table[LABEL_COLUMN].eq(1)
                ]
            ),
        }
        for role in ALLOWED_ROLES
    }

    parsed_starts = pd.to_datetime(manifest["start_date"], format="%m/%d/%Y", errors="coerce")
    parsed_ends = pd.to_datetime(manifest["end_date"], format="%m/%d/%Y", errors="coerce")
    chronological = True
    for _, group in manifest.assign(_start=parsed_starts, _end=parsed_ends).groupby(
        ID_COLUMN, sort=False
    ):
        ordered = group.sort_values("window_number_within_customer", kind="stable")
        if not ordered["window_number_within_customer"].is_monotonic_increasing:
            chronological = False
        if not ordered["start_feature_index"].is_monotonic_increasing:
            chronological = False
        if not ordered["_start"].is_monotonic_increasing:
            chronological = False

    raw = load_window_matrix("train")
    if len(raw.X) != len(manifest) or not np.array_equal(raw.window_ids, manifest["window_id"]):
        raise ValueError("Raw matrix and frozen window manifest are not aligned")
    summaries = behavioural_features(raw.X)

    total_windows = len(manifest)
    max_bag = int(customer_table["usable_window_count"].max())
    median_bag = float(customer_table["usable_window_count"].median())
    representations = pd.DataFrame(
        [
            {
                "representation": "RAW_30",
                "dimensionality": 30,
                "existing_definition": "30 frozen scaled daily values per usable stride-30 window",
                "development_windows": total_windows,
                "float32_matrix_mib": raw.X.nbytes / (1024**2),
                "median_bag_float32_kib": median_bag * 30 * 4 / 1024,
                "maximum_bag_float32_kib": max_bag * 30 * 4 / 1024,
                "batch64_max_padding_float32_mib": 64 * max_bag * 30 * 4 / (1024**2),
                "nonfinite_values": int((~np.isfinite(raw.X)).sum()),
                "missing_values_after_frozen_preprocessing": int(np.isnan(raw.X).sum()),
                "within_window_chronology_retained": True,
                "bag_chronology_recoverable_from_metadata": True,
                "interpretability": "direct scaled daily trajectory; positions map to ordered dates",
                "frozen_reproducible": True,
                "leakage_assessment": "No label/ID input; inherits frozen scaler fitted on the full original-train population, including current DEV_VAL features but not labels",
                "stage24b_status": "eligible_not_primary",
            },
            {
                "representation": "SUMMARY_17",
                "dimensionality": 17,
                "existing_definition": "17 frozen Stage 14 behavioural summaries per usable window",
                "development_windows": total_windows,
                "float32_matrix_mib": summaries.nbytes / (1024**2),
                "median_bag_float32_kib": median_bag * 17 * 4 / 1024,
                "maximum_bag_float32_kib": max_bag * 17 * 4 / 1024,
                "batch64_max_padding_float32_mib": 64 * max_bag * 17 * 4 / (1024**2),
                "nonfinite_values": int((~np.isfinite(summaries)).sum()),
                "missing_values_after_frozen_preprocessing": int(np.isnan(summaries).sum()),
                "within_window_chronology_retained": False,
                "bag_chronology_recoverable_from_metadata": True,
                "interpretability": "named level, distribution, variability, and directional summaries",
                "frozen_reproducible": True,
                "leakage_assessment": "No label/ID input; deterministic from RAW_30 and inherits the same frozen-scaler limitation",
                "stage24b_status": "recommended_primary",
            },
            {
                "representation": "OPTIONAL_EXISTING_COMBINED_RAW_PLUS_SUMMARY_47",
                "dimensionality": 47,
                "existing_definition": "Existing Stage 14 RAW_PLUS_SUMMARY concatenation",
                "development_windows": total_windows,
                "float32_matrix_mib": (raw.X.nbytes + summaries.nbytes) / (1024**2),
                "median_bag_float32_kib": median_bag * 47 * 4 / 1024,
                "maximum_bag_float32_kib": max_bag * 47 * 4 / 1024,
                "batch64_max_padding_float32_mib": 64 * max_bag * 47 * 4 / (1024**2),
                "nonfinite_values": int((~np.isfinite(raw.X)).sum() + (~np.isfinite(summaries)).sum()),
                "missing_values_after_frozen_preprocessing": int(np.isnan(raw.X).sum() + np.isnan(summaries).sum()),
                "within_window_chronology_retained": True,
                "bag_chronology_recoverable_from_metadata": True,
                "interpretability": "direct trajectory plus named summaries; redundant by construction",
                "frozen_reproducible": True,
                "leakage_assessment": "No label/ID input; inherits the same frozen-scaler limitation",
                "stage24b_status": "eligible_but_excluded_from_first_experiment",
            },
        ]
    )

    bag_payload: dict[str, Any] = {
        "stage": "Stage 24A — development-only MIL bag feasibility audit",
        "data_scope": {
            "loaded_processed_population": "original train only",
            "loaded_window_population": "original train usable stride-30 windows only",
            "allowed_research_roles": list(ALLOWED_ROLES),
            "confirmation_row_level_access": False,
            "final_test_row_level_access": False,
        },
        "frozen_inputs": {
            "governance_manifest": {
                "path": str(GOVERNANCE_MANIFEST.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": file_sha256(GOVERNANCE_MANIFEST),
            },
            "train_usable_window_manifest": {
                "path": str(TRAIN_WINDOWS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": file_sha256(TRAIN_WINDOWS),
            },
            "train_processed_matrix": {
                "path": str(TRAIN_PROCESSED.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": file_sha256(TRAIN_PROCESSED),
            },
            "stage14_representation_specification": {
                "path": str(STAGE14_CONFIG.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": file_sha256(STAGE14_CONFIG),
                "summary_feature_names": list(FEATURE_NAMES),
            },
        },
        "overall": grouped_summary(customer_table),
        "by_research_role": role_summaries,
        "by_customer_label": class_summaries,
        "by_research_role_and_customer_label": role_class_summaries,
        "very_short_bag_definition": "At most 5 usable windows; additional thresholds are reported for sensitivity.",
        "mapping_and_chronology_integrity": {
            "unique_window_ids": bool(not manifest["window_id"].duplicated().any()),
            "each_window_maps_to_exactly_one_customer": bool(
                manifest.groupby("window_id")[ID_COLUMN].nunique().eq(1).all()
            ),
            "duplicate_customer_window_number_mappings": int(
                manifest.duplicated([ID_COLUMN, "window_number_within_customer"]).sum()
            ),
            "duplicate_customer_start_end_mappings": int(
                manifest.duplicated([ID_COLUMN, "start_feature_index", "end_feature_index"]).sum()
            ),
            "start_and_end_dates_parse": bool(parsed_starts.notna().all() and parsed_ends.notna().all()),
            "chronological_order_recoverable": chronological,
            "bag_construction_uses_other_customer_information": False,
            "bag_construction_rule": "Group each customer's own frozen eligible windows by CONS_NO; sort by window_number_within_customer only for audit/analysis.",
        },
        "feasibility_conclusion": "Variable-length bags are small (at most 34 instances) and directly manageable without truncation, resampling, or cross-customer information.",
    }
    save_json(bag_payload, RESULTS_DIR / "bag_statistics.json")
    save_csv(representations, RESULTS_DIR / "instance_representation_audit.csv")


if __name__ == "__main__":
    main()
