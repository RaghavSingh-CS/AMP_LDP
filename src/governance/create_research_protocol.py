"""Create the Stage 17A AMP development/confirmation governance split.

This module reads customer IDs and historical split roles, train-only labels,
and train-only usable-window membership. It never loads validation/test feature
matrices, row-level test labels, model predictions, or performance metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "governance" / "stage_17a"
HISTORICAL_MANIFEST = PROJECT_ROOT / "data" / "processed" / "split_manifest.parquet"
TRAIN_CUSTOMERS = PROJECT_ROOT / "data" / "processed" / "train.parquet"
TRAIN_USABLE_WINDOWS = (
    PROJECT_ROOT
    / "data"
    / "windows"
    / "stride_30"
    / "usable"
    / "train_windows.parquet"
)
HISTORICAL_SPLIT_AUDIT = (
    PROJECT_ROOT / "results" / "supervised" / "stage_13" / "split_audit.json"
)
OUTPUT_MANIFEST = RESULTS_DIR / "research_protocol_manifest.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "protocol_summary.json"
OUTPUT_REPORT = RESULTS_DIR / "stage_17a_protocol_report.md"

ID_COLUMN = "CONS_NO"
LABEL_COLUMN = "FLAG"
SEED = 42
DEV_VAL_FRACTION = 0.20
ROLE_ORDER = ("DEV_TRAIN", "DEV_VAL", "CONFIRMATION", "FINAL_TEST")
ORIGINAL_SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
    temporary.replace(path)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, set[str], dict[str, Any]]:
    """Load governance metadata only; confirmation/final labels stay sealed."""
    historical = pd.read_parquet(
        HISTORICAL_MANIFEST,
        columns=[ID_COLUMN, "split"],
        engine="pyarrow",
    )
    train_labels = pd.read_parquet(
        TRAIN_CUSTOMERS,
        columns=[ID_COLUMN, LABEL_COLUMN],
        engine="pyarrow",
    )
    usable_train_ids = set(
        pd.read_parquet(
            TRAIN_USABLE_WINDOWS,
            columns=[ID_COLUMN],
            engine="pyarrow",
        )[ID_COLUMN].astype(str)
    )
    with HISTORICAL_SPLIT_AUDIT.open("r", encoding="utf-8") as source:
        historical_audit = json.load(source)

    historical[ID_COLUMN] = historical[ID_COLUMN].astype(str)
    train_labels[ID_COLUMN] = train_labels[ID_COLUMN].astype(str)
    if historical[ID_COLUMN].isna().any() or historical[ID_COLUMN].duplicated().any():
        raise ValueError("Historical customer manifest contains invalid IDs")
    if train_labels[ID_COLUMN].isna().any() or train_labels[ID_COLUMN].duplicated().any():
        raise ValueError("Train customer table contains invalid IDs")
    if set(historical["split"]) != set(ORIGINAL_SPLITS):
        raise ValueError("Historical split labels changed")
    if not train_labels[LABEL_COLUMN].isin([0, 1]).all():
        raise ValueError("Train labels are not binary")
    historical_train_ids = set(
        historical.loc[historical["split"].eq("train"), ID_COLUMN]
    )
    if set(train_labels[ID_COLUMN]) != historical_train_ids:
        raise ValueError("Train-only labels do not match the historical train role")
    if not usable_train_ids <= historical_train_ids:
        raise ValueError("Usable training windows include a non-training customer")
    return historical, train_labels, usable_train_ids, historical_audit


def create_roles(
    historical: pd.DataFrame,
    train_labels: pd.DataFrame,
    usable_train_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Make one frozen stratified 80/20 split of usable original-train IDs."""
    eligible = train_labels.loc[train_labels[ID_COLUMN].isin(usable_train_ids)].copy()
    eligible = eligible.sort_values(ID_COLUMN, kind="mergesort").reset_index(drop=True)
    dev_train_ids, dev_val_ids = train_test_split(
        eligible[ID_COLUMN].to_numpy(),
        test_size=DEV_VAL_FRACTION,
        random_state=SEED,
        shuffle=True,
        stratify=eligible[LABEL_COLUMN].to_numpy(),
    )
    dev_train = set(map(str, dev_train_ids))
    dev_val = set(map(str, dev_val_ids))
    unavailable_train = set(train_labels[ID_COLUMN]) - usable_train_ids
    if len(unavailable_train) != 1:
        raise ValueError(
            "Expected exactly one original-training customer without a Stage 16 representation"
        )
    # Governance must retain every authoritative customer. The unavailable
    # training customer is assigned to DEV_TRAIN, but remains explicitly
    # ineligible for Stage 16 representation fitting until a policy is defined.
    dev_train |= unavailable_train

    role_by_id: dict[str, str] = {}
    role_by_id.update({customer_id: "DEV_TRAIN" for customer_id in dev_train})
    role_by_id.update({customer_id: "DEV_VAL" for customer_id in dev_val})
    role_by_id.update(
        {
            customer_id: "CONFIRMATION"
            for customer_id in historical.loc[
                historical["split"].eq("val"), ID_COLUMN
            ]
        }
    )
    role_by_id.update(
        {
            customer_id: "FINAL_TEST"
            for customer_id in historical.loc[
                historical["split"].eq("test"), ID_COLUMN
            ]
        }
    )
    if len(role_by_id) != len(historical):
        raise ValueError("Research roles do not cover every historical customer exactly once")

    output = historical.rename(columns={"split": "original_split"}).copy()
    output["research_role"] = output[ID_COLUMN].map(role_by_id)
    output["stage16_representation_available"] = True
    output.loc[output[ID_COLUMN].isin(unavailable_train), "stage16_representation_available"] = False
    output = output[
        [ID_COLUMN, "original_split", "research_role", "stage16_representation_available"]
    ]
    procedure = {
        "eligible_population": "Original-train customers with at least one established usable Stage 3C stride-30 window",
        "eligible_customer_count": int(len(eligible)),
        "sort_before_split": f"ascending {ID_COLUMN} with stable mergesort",
        "algorithm": "sklearn.model_selection.train_test_split",
        "test_size_for_DEV_VAL": DEV_VAL_FRACTION,
        "shuffle": True,
        "stratify": f"train-only customer {LABEL_COLUMN}",
        "random_state": SEED,
        "seed_selection": "fixed by repository convention; not screened or optimized",
        "unavailable_customer_policy": "Retain in governance manifest as DEV_TRAIN with stage16_representation_available=false; exclude transparently from representation fitting/evaluation",
        "unavailable_original_train_customer_ids": sorted(unavailable_train),
    }
    return output, procedure


def verify_and_summarize(
    manifest: pd.DataFrame,
    train_labels: pd.DataFrame,
    historical_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(manifest["research_role"]) != set(ROLE_ORDER):
        raise ValueError("Research manifest roles are invalid")
    role_sets = {
        role: set(manifest.loc[manifest["research_role"].eq(role), ID_COLUMN])
        for role in ROLE_ORDER
    }
    overlap_counts: dict[str, int] = {}
    for left_index, left in enumerate(ROLE_ORDER):
        for right in ROLE_ORDER[left_index + 1 :]:
            overlap_counts[f"{left}__{right}"] = len(role_sets[left] & role_sets[right])
    if any(overlap_counts.values()):
        raise ValueError(f"Research role overlap detected: {overlap_counts}")
    union = set().union(*role_sets.values())
    all_ids = set(manifest[ID_COLUMN])
    if union != all_ids or len(union) != int(historical_audit["total_customers"]):
        raise ValueError("Research role union does not match the authoritative population")

    train_with_roles = train_labels.merge(
        manifest[[ID_COLUMN, "original_split", "research_role", "stage16_representation_available"]],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    role_counts: dict[str, dict[str, Any]] = {}
    for role in ("DEV_TRAIN", "DEV_VAL"):
        rows = train_with_roles.loc[train_with_roles["research_role"].eq(role)]
        normal = int(rows[LABEL_COLUMN].eq(0).sum())
        theft = int(rows[LABEL_COLUMN].eq(1).sum())
        available = int(rows["stage16_representation_available"].sum())
        role_counts[role] = {
            "customers": int(len(rows)),
            "normal_customers": normal,
            "theft_customers": theft,
            "theft_prevalence": float(theft / len(rows)),
            "customers_with_stage16_representation": available,
            "customers_without_stage16_representation": int(len(rows) - available),
        }

    historical_role_map = {"CONFIRMATION": "val", "FINAL_TEST": "test"}
    for role, old_split in historical_role_map.items():
        aggregate = historical_audit["splits"][old_split]
        role_counts[role] = {
            "customers": int(aggregate["customers"]),
            "normal_customers": int(aggregate["normal_customers"]),
            "theft_customers": int(aggregate["theft_customers"]),
            "theft_prevalence": float(aggregate["customer_theft_prevalence"]),
            "customers_with_stage16_representation": int(
                aggregate["customers_with_usable_windows"]
            ),
            "customers_without_stage16_representation": int(
                aggregate["customers"]
                - aggregate["customers_with_usable_windows"]
            ),
            "class_count_source": relative(HISTORICAL_SPLIT_AUDIT),
        }
        if len(role_sets[role]) != role_counts[role]["customers"]:
            raise ValueError(f"{role} ID count disagrees with historical aggregate audit")

    original_validation = set(
        manifest.loc[manifest["original_split"].eq("val"), ID_COLUMN]
    )
    original_test = set(
        manifest.loc[manifest["original_split"].eq("test"), ID_COLUMN]
    )
    original_train = set(
        manifest.loc[manifest["original_split"].eq("train"), ID_COLUMN]
    )
    assignment_assertions = {
        "all_original_validation_are_CONFIRMATION": role_sets["CONFIRMATION"] == original_validation,
        "all_original_test_are_FINAL_TEST": role_sets["FINAL_TEST"] == original_test,
        "DEV_TRAIN_only_original_train": role_sets["DEV_TRAIN"] <= original_train,
        "DEV_VAL_only_original_train": role_sets["DEV_VAL"] <= original_train,
    }
    if not all(assignment_assertions.values()):
        raise ValueError(f"Historical role preservation failed: {assignment_assertions}")
    verification = {
        "role_pairwise_overlap_counts": overlap_counts,
        "all_pairwise_overlaps_zero": True,
        "role_union_customer_count": len(union),
        "authoritative_customer_count": int(historical_audit["total_customers"]),
        "union_matches_authoritative_population": True,
        "assignment_assertions": assignment_assertions,
    }
    return role_counts, verification


def write_report(summary: dict[str, Any]) -> None:
    counts = summary["role_counts"]
    rows = "\n".join(
        f"| {role} | {counts[role]['customers']:,} | {counts[role]['normal_customers']:,} | {counts[role]['theft_customers']:,} | {counts[role]['theft_prevalence']:.12f} | {counts[role]['customers_with_stage16_representation']:,} | {counts[role]['customers_without_stage16_representation']:,} |"
        for role in ROLE_ORDER
    )
    report = f"""# Stage 17A — Development and Confirmation Protocol

## Motivation and historical disclosure

Stages 13–16 legitimately used the original validation population for exploratory
model, representation, hybrid, and formulation decisions. That population is no
longer a statistically pristine development holdout. It remains more independent
than data used for routine future iteration, so it is reassigned to `CONFIRMATION`
and may be consulted only at explicit, recorded confirmation gates.

The historical customer split is not changed or overwritten. The hierarchy is:

`original train → DEV_TRAIN + DEV_VAL`

`original validation → CONFIRMATION`

`original test → FINAL_TEST`

## Deterministic development split

Only the 29,655 original-training customers with an established usable Stage 3C
stride-30 window were eligible for the stratified split. IDs were sorted ascending,
then split once with `sklearn.model_selection.train_test_split`, `test_size=0.20`,
`shuffle=True`, customer-`FLAG` stratification, and seed `{SEED}`. The seed follows
the repository convention and was not screened using model performance or feature
values. The one original-training customer without a usable window is retained as
`DEV_TRAIN`, marked representation-unavailable, and excluded transparently from
Stage 16 representation fitting until a separate policy exists.

## Frozen role audit

| Role | Customers | Normal | Theft | Theft prevalence | Stage 16 representation available | Unavailable |
|---|---:|---:|---:|---:|---:|---:|
{rows}

All six pairwise role overlaps are zero. Their union is exactly all 42,367
authoritative customers. Every original-validation customer is `CONFIRMATION`,
every original-test customer is `FINAL_TEST`, and both development roles contain
only original-training customers.

The machine-readable manifest is `{summary['manifest']['path']}`, SHA-256
`{summary['manifest']['sha256']}`. It contains customer ID, historical split,
new research role, and Stage 16 representation availability; it contains no label
or feature columns. Confirmation/final class and availability counts above come
from the already tracked aggregate Stage 13 split audit. Row-level test labels were
not reopened.

## Frozen future-development policy

Routine Stage 17+ candidate selection, feature experiments, hyperparameter choices,
imbalance experiments, and architecture comparisons must fit on `DEV_TRAIN` and
select only on `DEV_VAL`. `CONFIRMATION` and `FINAL_TEST` must not be consulted
during routine iteration. Split IDs and seed are frozen; candidate splits must not
be regenerated.

`CONFIRMATION` may be accessed only after a candidate methodology is frozen—for
example after controlled supervised benchmarking, a major representation change,
or before treating a proposed contribution as sufficiently credible to continue.
A confirmation result must be logged as a gate and must not trigger small direct
tweaks. A failed gate returns research to `DEV_TRAIN`/`DEV_VAL`; a later gate
requires a materially revised, pre-specified hypothesis.

`FINAL_TEST` remains sealed until the final AMP methodology is frozen. The one-time
historical Stage 13 test evaluation must remain disclosed, so final test is not
described as historically untouched. No Stage 14+ decision used test, and future
development must not access it before final evaluation.

## Historical results and next experiment

Stages 13–16 remain valid historical exploratory evidence under their documented
old protocol; they are not rewritten as if this hierarchy already existed. The
Stage 16 direct-customer HGB remains the current historical best, but is not
automatically confirmed. Future benchmarking must reconstruct the frozen 48-feature
representation using `DEV_TRAIN`/`DEV_VAL` only.

Stage 17B is pre-defined but not executed:

**On the frozen 48-feature direct-customer representation, which conventional
supervised learner provides the strongest discrimination under the new
DEV_TRAIN/DEV_VAL protocol?**

The expected controlled families are logistic regression, HistGradientBoosting,
and XGBoost, LightGBM, or CatBoost only if available without environment disruption.
There will be no broad AutoML, neural network, GAN, or feature redesign. Primary
metrics are customer ROC-AUC and average precision; precision, recall, and F1 are
secondary. Exact Stage 17B details will be specified separately.

**NO NEW MODEL BENCHMARKING, CONFIRMATION EVALUATION, OR FINAL-TEST EVALUATION WAS
PERFORMED DURING STAGE 17A.**
"""
    temporary = OUTPUT_REPORT.with_name(OUTPUT_REPORT.name + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    temporary.replace(OUTPUT_REPORT)


def main() -> None:
    if any(path.exists() for path in (OUTPUT_MANIFEST, OUTPUT_SUMMARY, OUTPUT_REPORT)):
        raise FileExistsError("Refusing to overwrite an existing Stage 17A protocol artifact")
    historical, train_labels, usable_train_ids, historical_audit = load_inputs()
    manifest, split_procedure = create_roles(
        historical, train_labels, usable_train_ids
    )
    role_counts, verification = verify_and_summarize(
        manifest, train_labels, historical_audit
    )
    save_csv(manifest, OUTPUT_MANIFEST)
    summary = {
        "stage": "Stage 17A — development and confirmation protocol governance",
        "created_without_model_training_or_performance_evaluation": True,
        "historical_split": {
            "path": relative(HISTORICAL_MANIFEST),
            "sha256": sha256(HISTORICAL_MANIFEST),
            "overwritten": False,
        },
        "historical_aggregate_audit": {
            "path": relative(HISTORICAL_SPLIT_AUDIT),
            "sha256": sha256(HISTORICAL_SPLIT_AUDIT),
            "purpose": "Existing aggregate confirmation/final class and representation-availability counts; no row-level test label access",
        },
        "manifest": {
            "path": relative(OUTPUT_MANIFEST),
            "sha256": sha256(OUTPUT_MANIFEST),
            "columns": list(manifest.columns),
            "contains_labels": False,
            "contains_features": False,
        },
        "split_procedure": split_procedure,
        "role_counts": role_counts,
        "verification": verification,
        "historical_validation_reuse": {
            "statistically_pristine": False,
            "disclosure": "Original validation influenced Stage 13 model selection, Stage 14 representation selection, Stage 15 hybrid assessment, and Stage 16 formulation selection.",
            "future_role": "CONFIRMATION only at explicit frozen-methodology gates",
        },
        "future_development_policy": {
            "fit": "DEV_TRAIN",
            "routine_selection": "DEV_VAL",
            "confirmation_during_routine_iteration": "PROHIBITED",
            "final_test_during_development": "PROHIBITED",
        },
        "confirmation_gate_policy": {
            "access_condition": "candidate methodology frozen and gate explicitly recorded",
            "failed_gate_response": "return to DEV_TRAIN/DEV_VAL with a materially revised pre-specified hypothesis",
            "tiny_confirmation_driven_tweaks": "PROHIBITED",
        },
        "final_test_policy": {
            "status": "SEALED until final AMP methodology is frozen",
            "historical_disclosure": "One-time Stage 13 test evaluation exists; no Stage 14+ research decision used test.",
        },
        "stage16_status": "CURRENT HISTORICAL BEST under the old validation protocol; not automatically confirmed under the new protocol",
        "stage17b_question": "On the frozen 48-feature direct-customer representation, which conventional supervised learner provides the strongest discrimination under the new DEV_TRAIN/DEV_VAL protocol?",
        "prohibitions_observed": {
            "new_candidate_model_training": False,
            "validation_performance_benchmarking": False,
            "confirmation_evaluation": False,
            "final_test_evaluation": False,
            "feature_redesign": False,
        },
    }
    save_json(summary, OUTPUT_SUMMARY)
    write_report(summary)
    print(f"Created {relative(OUTPUT_MANIFEST)}")
    print(f"Manifest SHA-256: {summary['manifest']['sha256']}")
    for role in ROLE_ORDER:
        print(role, role_counts[role])
    print("No models trained and no confirmation/final-test evaluation performed.")


if __name__ == "__main__":
    main()
